#pragma once

// Hybrid CPU/GPU two-level AMR. Same algorithm as Amr2 (the CPU
// reference), but every grid lives in shared Metal buffers:
//   GPU — coarse step + all patches batched in one pooled dispatch
//   CPU — ghost fill, refluxing, restriction, tagging, regridding
// Unified memory means the CPU orchestrates in place: zero copies.

#include "amr/Amr2.hpp" // AmrConfig
#include "backend/metal/Euler2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/Limiter.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace mm {

class AmrGpu {
public:
    struct Patch {
        int bi, bj, ci0, cj0;
        int slot; // index into the pooled buffers
    };

    std::vector<Patch> patches;

    std::function<void(GridRef&, double)> fillPhysicalGhosts;
    std::function<void(GridRef&, double, unsigned)> fillPatchPhysical;

    AmrGpu(MetalContext& ctx, int nx, int ny, Real x0, Real y0, Real lx,
           Real ly, AmrConfig cfg)
        : ctx_(ctx), cfg_(cfg), x0_(x0), y0_(y0),
          coarse_(ctx, nx, ny, lx / nx, ly / ny),
          nbx_(nx / cfg.blockC), nby_(ny / cfg.blockC),
          blockOf_(std::size_t(nbx_) * nby_, -1) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
        nf_ = 2 * cfg.blockC;
        pTot_ = nf_ + 2 * NG;
        stride_ = pTot_ * pTot_;
        capacity_ = nbx_ * nby_;

        const std::size_t bytes =
            std::size_t(capacity_) * stride_ * sizeof(Cons);
        const auto mk = [&] {
            return ctx.device()->newBuffer(bytes,
                                           MTL::ResourceStorageModeShared);
        };
        qP_ = mk(); xLP_ = mk(); xRP_ = mk(); yBP_ = mk(); yTP_ = mk();
        FxP_ = mk(); FyP_ = mk();
        smaxP_ = ctx.device()->newBuffer(2 * sizeof(std::uint32_t),
                                         MTL::ResourceStorageModeShared);
        slotsBuf_ = ctx.device()->newBuffer(
            std::size_t(capacity_) * sizeof(std::uint32_t),
            MTL::ResourceStorageModeShared);
        for (int s = capacity_ - 1; s >= 0; --s) freeSlots_.push_back(s);

        lib_ = ctx.compileLibrary("euler2d.metal");
        predictorP_ = ctx.makePipeline(lib_, "predictor_pool");
        fluxXP_ = ctx.makePipeline(lib_, "flux_x_pool");
        fluxYP_ = ctx.makePipeline(lib_, "flux_y_pool");
        updateP_ = ctx.makePipeline(lib_, "update_pool");
        waveP_ = ctx.makePipeline(lib_, "wave_pool");
    }

    ~AmrGpu() {
        for (MTL::Buffer* b :
             {qP_, xLP_, xRP_, yBP_, yTP_, FxP_, FyP_, smaxP_, slotsBuf_})
            b->release();
        for (MTL::ComputePipelineState* p :
             {predictorP_, fluxXP_, fluxYP_, updateP_, waveP_})
            p->release();
        lib_->release();
    }

    AmrGpu(const AmrGpu&) = delete;
    AmrGpu& operator=(const AmrGpu&) = delete;

    GridRef coarseRef() { return coarse_.ref(x0_, y0_); }
    GridRef coarseRef() const {
        return const_cast<AmrGpu*>(this)->coarseRef();
    }

    GridRef patchRef(const Patch& p) const {
        return GridRef{nf_,
                       nf_,
                       x0_ + p.ci0 * dxc_(),
                       y0_ + p.cj0 * dyc_(),
                       dxc_() / 2,
                       dyc_() / 2,
                       static_cast<Cons*>(qP_->contents()) +
                           std::size_t(p.slot) * stride_};
    }

    bool covered(int bi, int bj) const {
        return blockOf_[std::size_t(bj) * nbx_ + bi] >= 0;
    }

    unsigned domainSides(const Patch& p) const {
        return (p.bi == 0 ? SideLeft : 0u) |
               (p.bi == nbx_ - 1 ? SideRight : 0u) |
               (p.bj == 0 ? SideBottom : 0u) |
               (p.bj == nby_ - 1 ? SideTop : 0u);
    }

    template <class IC>
    void init(IC ic) {
        GridRef c = coarseRef();
        for (int j = NG; j < NG + c.ny; ++j)
            for (int i = NG; i < NG + c.nx; ++i)
                c.at(i, j) = ic(c.xc(i), c.yc(j));
        fillPhysicalGhosts(c, 0);
        regrid();
        for (const Patch& p : patches) {
            GridRef g = patchRef(p);
            for (int j = NG; j < NG + g.ny; ++j)
                for (int i = NG; i < NG + g.nx; ++i)
                    g.at(i, j) = ic(g.xc(i), g.yc(j));
        }
        restrictFine_();
    }

    // CFL dt. step() leaves the post-update wave speeds behind (the
    // reduction rides at the end of its command buffer), so this is free
    // except on the very first call.
    Real maxStableDtAll(Real cfl) {
        if (!haveWave_) {
            coarse_.zeroWave();
            auto* smp = static_cast<std::uint32_t*>(smaxP_->contents());
            smp[0] = smp[1] = 0;
            MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
            encodeWaveAll_(cmd);
            cmd->commit();
            cmd->waitUntilCompleted();
            readWave_();
        }
        Real dt = cfl * std::min(dxc_() / sxC_, dyc_() / syC_);
        if (sxF_ > 0)
            dt = std::min(dt, cfl * std::min(dxc_() / 2 / sxF_,
                                             dyc_() / 2 / syF_));
        return dt;
    }

    void step(Real dt, double t) {
        GridRef c = coarseRef();
        fillPhysicalGhosts(c, t);
        for (const Patch& p : patches) {
            fillPatchGhosts_(p);
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides(p)) {
                    GridRef g = patchRef(p);
                    fillPatchPhysical(g, t, sides);
                }
        }

        // GPU: coarse + all patches + next-step wave reduction, one
        // command buffer, one sync.
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        coarse_.encodeStep(cmd, dt);
        if (!patches.empty()) {
            encodePool_(cmd, predictorP_, {qP_, xLP_, xRP_, yBP_, yTP_},
                        dt, pTot_ - 2, pTot_ - 2, 1, 1);
            encodePool_(cmd, fluxXP_, {xLP_, xRP_, FxP_}, dt, nf_ + 1, nf_,
                        NG - 1, NG);
            encodePool_(cmd, fluxYP_, {yBP_, yTP_, FyP_}, dt, nf_, nf_ + 1,
                        NG, NG - 1);
            encodePool_(cmd, updateP_, {qP_, FxP_, FyP_}, dt, nf_, nf_, NG,
                        NG);
        }
        coarse_.zeroWave();
        auto* smp = static_cast<std::uint32_t*>(smaxP_->contents());
        smp[0] = smp[1] = 0;
        encodeWaveAll_(cmd); // hazard tracking runs it after the updates
        cmd->commit();
        cmd->waitUntilCompleted();
        readWave_(); // post-step speeds; regrid below only prolongates
                     // coarse data, so they remain a valid dt bound

        // CPU: conservation fix-up and synchronization.
        if (cfg_.reflux)
            for (const Patch& p : patches) reflux_(p, dt);
        restrictFine_();

        if (++stepCount_ % cfg_.regridEvery == 0) {
            const bool hadPatches = !patches.empty();
            fillPhysicalGhosts(c, t + dt);
            regrid();
            // The cached wave speeds lack a fine-level bound if the pool
            // was empty when they were measured.
            if (!hadPatches && !patches.empty()) haveWave_ = false;
        }
    }

    double totalMass() const {
        double m = 0;
        const GridRef c = coarseRef();
        const double ac = double(c.dx) * c.dy;
        for (int j = 0; j < c.ny; ++j)
            for (int i = 0; i < c.nx; ++i)
                if (!covered(i / cfg_.blockC, j / cfg_.blockC))
                    m += double(c.at(NG + i, NG + j).rho) * ac;
        for (const Patch& p : patches) {
            const GridRef g = patchRef(p);
            const double af = double(g.dx) * g.dy;
            for (int j = NG; j < NG + g.ny; ++j)
                for (int i = NG; i < NG + g.nx; ++i)
                    m += double(g.at(i, j).rho) * af;
        }
        return m;
    }

    std::size_t cellCount() const {
        const GridRef c = coarseRef();
        std::size_t n = std::size_t(c.nx) * c.ny;
        n += patches.size() * std::size_t(nf_) * nf_;
        return n;
    }

    int blockCount() const { return nbx_ * nby_; }
    int fineCells() const { return nf_; }

    void regrid() {
        const GridRef c = coarseRef();
        const int nx = c.nx, ny = c.ny, bc = cfg_.blockC;

        std::vector<std::uint8_t> tag(std::size_t(nx) * ny, 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int ip = std::min(i + 1, nx - 1),
                          im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, ny - 1),
                          jm = std::max(j - 1, 0);
                const Real r0 = c.at(NG + i, NG + j).rho;
                const Real ex = std::fabs(c.at(NG + ip, NG + j).rho -
                                          c.at(NG + im, NG + j).rho);
                const Real ey = std::fabs(c.at(NG + i, NG + jp).rho -
                                          c.at(NG + i, NG + jm).rho);
                tag[std::size_t(j) * nx + i] =
                    std::max(ex, ey) / r0 > cfg_.tagThreshold;
            }

        std::vector<std::uint8_t> want(std::size_t(nbx_) * nby_, 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (!tag[std::size_t(j) * nx + i]) continue;
                // dilation by 2 folded into the block mapping
                for (int b = std::max(j - 2, 0);
                     b <= std::min(j + 2, ny - 1); ++b)
                    for (int a = std::max(i - 2, 0);
                         a <= std::min(i + 2, nx - 1); ++a)
                        want[std::size_t(b / bc) * nbx_ + a / bc] = 1;
            }

        std::vector<Patch> next;
        std::vector<int> nextOf(std::size_t(nbx_) * nby_, -1);
        // Free removed blocks first so their slots are reusable below.
        for (const Patch& p : patches)
            if (!want[std::size_t(p.bj) * nbx_ + p.bi])
                freeSlots_.push_back(p.slot);
        for (int bj = 0; bj < nby_; ++bj)
            for (int bi = 0; bi < nbx_; ++bi) {
                if (!want[std::size_t(bj) * nbx_ + bi]) continue;
                const int old = blockOf_[std::size_t(bj) * nbx_ + bi];
                if (old >= 0) {
                    next.push_back(patches[old]);
                } else {
                    next.push_back(makePatch_(bi, bj));
                }
                nextOf[std::size_t(bj) * nbx_ + bi] = int(next.size()) - 1;
            }
        patches = std::move(next);
        blockOf_ = std::move(nextOf);

        auto* slots = static_cast<std::uint32_t*>(slotsBuf_->contents());
        for (std::size_t k = 0; k < patches.size(); ++k)
            slots[k] = std::uint32_t(patches[k].slot);
    }

private:
    Real dxc_() const { return coarseRef().dx; }
    Real dyc_() const { return coarseRef().dy; }

    void encodeWaveAll_(MTL::CommandBuffer* cmd) const {
        coarse_.encodeWave(cmd);
        if (!patches.empty())
            encodePool_(cmd, waveP_, {qP_, smaxP_}, 0, nf_, nf_, NG, NG);
    }

    void readWave_() {
        const auto [sx, sy] = coarse_.waveSpeeds();
        sxC_ = sx;
        syC_ = sy;
        const auto* smp =
            static_cast<const std::uint32_t*>(smaxP_->contents());
        sxF_ = patches.empty() ? Real(0) : std::bit_cast<float>(smp[0]);
        syF_ = patches.empty() ? Real(0) : std::bit_cast<float>(smp[1]);
        haveWave_ = true;
    }

    void encodePool_(MTL::CommandBuffer* cmd,
                     MTL::ComputePipelineState* pso,
                     std::initializer_list<MTL::Buffer*> bufs, Real dt,
                     int w, int h, int, int) const {
        const Euler2DGpu::Params p{pTot_, pTot_,  nf_, nf_,
                                   dxc_() / 2, dyc_() / 2, dt, stride_};
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        int slot = 0;
        for (MTL::Buffer* b : bufs) enc->setBuffer(b, 0, slot++);
        enc->setBytes(&p, sizeof(p), slot++);
        enc->setBuffer(slotsBuf_, 0, slot);
        enc->dispatchThreads(MTL::Size(w, h, patches.size()),
                             MTL::Size(8, 8, 1));
        enc->endEncoding();
    }

    Patch makePatch_(int bi, int bj) {
        assert(!freeSlots_.empty());
        const int slot = freeSlots_.back();
        freeSlots_.pop_back();
        Patch p{bi, bj, bi * cfg_.blockC, bj * cfg_.blockC, slot};
        GridRef g = patchRef(p);
        for (int j = NG; j < NG + nf_; ++j)
            for (int i = NG; i < NG + nf_; ++i) {
                const int gfi = 2 * p.ci0 + (i - NG);
                const int gfj = 2 * p.cj0 + (j - NG);
                g.at(i, j) = prolong_(gfi / 2, gfj / 2, gfi & 1, gfj & 1);
            }
        return p;
    }

    Cons prolong_(int ci, int cj, int ox, int oy) const {
        const GridRef c = coarseRef();
        const int I = NG + ci, J = NG + cj;
        const Cons& q0 = c.at(I, J);
        const Cons dqx = limitedSlope(c.at(I - 1, J), q0, c.at(I + 1, J));
        const Cons dqy = limitedSlope(c.at(I, J - 1), q0, c.at(I, J + 1));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    void fillPatchGhosts_(const Patch& p) const {
        GridRef g = patchRef(p);
        const GridRef c = coarseRef();
        const int nxf = 2 * c.nx, nyf = 2 * c.ny;
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i) {
                if (i >= NG && i < NG + nf_ && j >= NG && j < NG + nf_)
                    continue;
                const int gfi = 2 * p.ci0 + (i - NG);
                const int gfj = 2 * p.cj0 + (j - NG);

                if (gfi >= 0 && gfi < nxf && gfj >= 0 && gfj < nyf) {
                    const int bi = gfi / (2 * cfg_.blockC);
                    const int bj = gfj / (2 * cfg_.blockC);
                    const int pi = blockOf_[std::size_t(bj) * nbx_ + bi];
                    if (pi >= 0) {
                        const GridRef s = patchRef(patches[pi]);
                        g.at(i, j) =
                            s.at(NG + gfi - 2 * patches[pi].ci0,
                                 NG + gfj - 2 * patches[pi].cj0);
                        continue;
                    }
                }
                const int ci = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                g.at(i, j) = prolong_(ci, cj, gfi - 2 * ci, gfj - 2 * cj);
            }
    }

    void reflux_(const Patch& p, Real dt) {
        GridRef c = coarseRef();
        const int bc = cfg_.blockC;
        const Real lx = dt / c.dx, ly = dt / c.dy;
        const int ctx = c.totx();
        const Cons* fxC = coarse_.fx();
        const Cons* fyC = coarse_.fy();
        const Cons* fxF = static_cast<const Cons*>(FxP_->contents()) +
                          std::size_t(p.slot) * stride_;
        const Cons* fyF = static_cast<const Cons*>(FyP_->contents()) +
                          std::size_t(p.slot) * stride_;
        const auto fid = [&](int i, int j) { return j * pTot_ + i; };
        const auto cid = [&](int i, int j) { return j * ctx + i; };

        if (p.bi > 0 && !covered(p.bi - 1, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fxF[fid(NG - 1, NG + 2 * r)] +
                                 fxF[fid(NG - 1, NG + 2 * r + 1)]);
                const Cons& fc = fxC[cid(NG + p.ci0 - 1, NG + p.cj0 + r)];
                c.at(NG + p.ci0 - 1, NG + p.cj0 + r) += lx * (fc - ff);
            }
        if (p.bi < nbx_ - 1 && !covered(p.bi + 1, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fxF[fid(NG + nf_ - 1, NG + 2 * r)] +
                                 fxF[fid(NG + nf_ - 1, NG + 2 * r + 1)]);
                const Cons& fc =
                    fxC[cid(NG + p.ci0 + bc - 1, NG + p.cj0 + r)];
                c.at(NG + p.ci0 + bc, NG + p.cj0 + r) += lx * (ff - fc);
            }
        if (p.bj > 0 && !covered(p.bi, p.bj - 1))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fyF[fid(NG + 2 * r, NG - 1)] +
                                 fyF[fid(NG + 2 * r + 1, NG - 1)]);
                const Cons& fc = fyC[cid(NG + p.ci0 + r, NG + p.cj0 - 1)];
                c.at(NG + p.ci0 + r, NG + p.cj0 - 1) += ly * (fc - ff);
            }
        if (p.bj < nby_ - 1 && !covered(p.bi, p.bj + 1))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fyF[fid(NG + 2 * r, NG + nf_ - 1)] +
                                 fyF[fid(NG + 2 * r + 1, NG + nf_ - 1)]);
                const Cons& fc =
                    fyC[cid(NG + p.ci0 + r, NG + p.cj0 + bc - 1)];
                c.at(NG + p.ci0 + r, NG + p.cj0 + bc) += ly * (ff - fc);
            }
    }

    void restrictFine_() {
        GridRef c = coarseRef();
        for (const Patch& p : patches) {
            const GridRef g = patchRef(p);
            for (int b = 0; b < cfg_.blockC; ++b)
                for (int a = 0; a < cfg_.blockC; ++a) {
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    const Cons sum = g.at(fi, fj) + g.at(fi + 1, fj) +
                                     g.at(fi, fj + 1) +
                                     g.at(fi + 1, fj + 1);
                    c.at(NG + p.ci0 + a, NG + p.cj0 + b) =
                        Real(0.25) * sum;
                }
        }
    }

    MetalContext& ctx_;
    AmrConfig cfg_;
    Real x0_, y0_;
    Euler2DGpu coarse_;
    int nbx_, nby_, nf_ = 0, pTot_ = 0, stride_ = 0, capacity_ = 0;
    std::vector<int> blockOf_;
    std::vector<int> freeSlots_;
    int stepCount_ = 0;
    bool haveWave_ = false;
    Real sxC_ = 0, syC_ = 0, sxF_ = 0, syF_ = 0;

    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *predictorP_ = nullptr, *fluxXP_ = nullptr,
                              *fluxYP_ = nullptr, *updateP_ = nullptr,
                              *waveP_ = nullptr;
    MTL::Buffer *qP_ = nullptr, *xLP_ = nullptr, *xRP_ = nullptr,
                *yBP_ = nullptr, *yTP_ = nullptr, *FxP_ = nullptr,
                *FyP_ = nullptr, *smaxP_ = nullptr, *slotsBuf_ = nullptr;
};

} // namespace mm

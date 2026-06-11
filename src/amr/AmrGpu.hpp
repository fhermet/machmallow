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
#include "core/Parallel.hpp"
#include "numerics/Limiter.hpp"

#include <cassert>
#include <chrono>
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

    // Cumulative wall time per step section (seconds).
    struct Timings {
        double ghost = 0, gpu = 0, reflux = 0, restrict_ = 0, regrid = 0;
        double total() const {
            return ghost + gpu + reflux + restrict_ + regrid;
        }
    };
    Timings timings;

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
        if (sxF_ > 0) {
            Real dtF = cfl * std::min(dxc_() / 2 / sxF_, dyc_() / 2 / syF_);
            if (cfg_.subcycle) dtF *= 2; // fine takes two half steps
            dt = std::min(dt, dtF);
        }
        return dt;
    }

    void step(Real dt, double t) {
        using Clk = std::chrono::steady_clock;
        auto mark = Clk::now();
        const auto lap = [&](double& acc) {
            const auto now = Clk::now();
            acc += std::chrono::duration<double>(now - mark).count();
            mark = now;
        };

        GridRef c = coarseRef();
        if (cfg_.subcycle && !patches.empty())
            stepSubcycled_(dt, t, lap);
        else
            stepSingleRate_(dt, t, lap);

        restrictFine_();
        lap(timings.restrict_);

        if (++stepCount_ % cfg_.regridEvery == 0) {
            const bool hadPatches = !patches.empty();
            fillPhysicalGhosts(c, t + dt);
            regrid();
            // The cached wave speeds lack a fine-level bound if the pool
            // was empty when they were measured.
            if (!hadPatches && !patches.empty()) haveWave_ = false;
            lap(timings.regrid);
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
        parallelFor(std::size_t(ny), [&](std::size_t row) {
            const int j = int(row);
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
        });

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

    void fillAllPatchGhosts_(double t, Real theta) {
        parallelFor(patches.size(), [&](std::size_t k) {
            const Patch& p = patches[k];
            fillPatchGhosts_(p, theta);
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides(p)) {
                    GridRef g = patchRef(p);
                    fillPatchPhysical(g, t, sides);
                }
        });
    }

    void encodePoolStep_(MTL::CommandBuffer* cmd, Real dt) const {
        encodePool_(cmd, predictorP_, {qP_, xLP_, xRP_, yBP_, yTP_}, dt,
                    pTot_ - 2, pTot_ - 2, 1, 1);
        encodePool_(cmd, fluxXP_, {xLP_, xRP_, FxP_}, dt, nf_ + 1, nf_,
                    NG - 1, NG);
        encodePool_(cmd, fluxYP_, {yBP_, yTP_, FyP_}, dt, nf_, nf_ + 1, NG,
                    NG - 1);
        encodePool_(cmd, updateP_, {qP_, FxP_, FyP_}, dt, nf_, nf_, NG, NG);
    }

    void zeroWaveAll_() {
        coarse_.zeroWave();
        auto* smp = static_cast<std::uint32_t*>(smaxP_->contents());
        smp[0] = smp[1] = 0;
    }

    template <class Lap>
    void stepSingleRate_(Real dt, double t, Lap&& lap) {
        GridRef c = coarseRef();
        fillPhysicalGhosts(c, t);
        fillAllPatchGhosts_(t, Real(-1));
        lap(timings.ghost);

        // GPU: coarse + all patches + next-step wave reduction, one
        // command buffer, one sync.
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        coarse_.encodeStep(cmd, dt);
        if (!patches.empty()) encodePoolStep_(cmd, dt);
        zeroWaveAll_();
        encodeWaveAll_(cmd); // hazard tracking runs it after the updates
        cmd->commit();
        cmd->waitUntilCompleted();
        readWave_(); // post-step speeds; regrid below only prolongates
                     // coarse data, so they remain a valid dt bound
        lap(timings.gpu);

        if (cfg_.reflux)
            for (const Patch& p : patches) {
                refluxCoarse_(p, dt);
                refluxFine_(p, dt);
            }
        lap(timings.reflux);
    }

    // Berger-Colella subcycling: coarse and fine substep 1 share the
    // first submission (independent buffers), substep 2 rides with the
    // wave reduction — still two syncs per coarse step, half the coarse
    // work of two single-rate steps.
    template <class Lap>
    void stepSubcycled_(Real dtC, double t, Lap&& lap) {
        const Real dtF = dtC / 2;
        GridRef c = coarseRef();

        coarseOld_.assign(c.q, c.q + std::size_t(c.totx()) * c.toty());
        fillPhysicalGhosts(c, t);
        fillAllPatchGhosts_(t, Real(-1)); // substep 1 ghosts at t^n
        lap(timings.ghost);

        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        coarse_.encodeStep(cmd, dtC);
        encodePoolStep_(cmd, dtF); // substep 1
        cmd->commit();
        cmd->waitUntilCompleted();
        lap(timings.gpu);

        if (cfg_.reflux)
            for (const Patch& p : patches) {
                refluxCoarse_(p, dtC);
                refluxFine_(p, dtF);
            }
        lap(timings.reflux);

        // Substep 2 ghosts: siblings are current, coarse prolongation
        // blends t^n and t^{n+1} at the half time.
        fillAllPatchGhosts_(t + dtF, Real(0.5));
        lap(timings.ghost);

        cmd = ctx_.queue()->commandBuffer();
        encodePoolStep_(cmd, dtF); // substep 2
        zeroWaveAll_();
        encodeWaveAll_(cmd);
        cmd->commit();
        cmd->waitUntilCompleted();
        readWave_();
        lap(timings.gpu);

        if (cfg_.reflux)
            for (const Patch& p : patches) refluxFine_(p, dtF);
        lap(timings.reflux);
    }

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

    // Coarse value at (I, J): current state, or the theta-blend between
    // the saved t^n copy and the current state (subcycled ghosts).
    Cons coarseAt_(const GridRef& c, int I, int J, Real theta) const {
        const Cons& cur = c.at(I, J);
        if (theta < 0) return cur;
        const Cons& old = coarseOld_[c.idx(I, J)];
        return (1 - theta) * old + theta * cur;
    }

    Cons prolong_(int ci, int cj, int ox, int oy,
                  Real theta = Real(-1)) const {
        const GridRef c = coarseRef();
        const int I = NG + ci, J = NG + cj;
        const Cons q0 = coarseAt_(c, I, J, theta);
        const Cons dqx = limitedSlope(coarseAt_(c, I - 1, J, theta), q0,
                                      coarseAt_(c, I + 1, J, theta));
        const Cons dqy = limitedSlope(coarseAt_(c, I, J - 1, theta), q0,
                                      coarseAt_(c, I, J + 1, theta));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    void fillPatchGhosts_(const Patch& p, Real theta = Real(-1)) const {
        GridRef g = patchRef(p);
        const GridRef c = coarseRef();
        const int nxf = 2 * c.nx, nyf = 2 * c.ny;

        const auto fillCell = [&](int i, int j) {
            const int gfi = 2 * p.ci0 + (i - NG);
            const int gfj = 2 * p.cj0 + (j - NG);

            if (gfi >= 0 && gfi < nxf && gfj >= 0 && gfj < nyf) {
                const int bi = gfi / (2 * cfg_.blockC);
                const int bj = gfj / (2 * cfg_.blockC);
                const int pi = blockOf_[std::size_t(bj) * nbx_ + bi];
                if (pi >= 0) {
                    const GridRef s = patchRef(patches[pi]);
                    g.at(i, j) = s.at(NG + gfi - 2 * patches[pi].ci0,
                                      NG + gfj - 2 * patches[pi].cj0);
                    return;
                }
            }
            const int ci = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
            const int cj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
            g.at(i, j) =
                prolong_(ci, cj, gfi - 2 * ci, gfj - 2 * cj, theta);
        };

        // Ghost bands only (bottom/top full width, then side strips).
        for (int j = 0; j < NG; ++j)
            for (int i = 0; i < g.totx(); ++i) fillCell(i, j);
        for (int j = NG + nf_; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i) fillCell(i, j);
        for (int j = NG; j < NG + nf_; ++j) {
            for (int i = 0; i < NG; ++i) fillCell(i, j);
            for (int i = NG + nf_; i < g.totx(); ++i) fillCell(i, j);
        }
    }

    // Refluxing split as in Amr2: refluxCoarse_ backs the coarse flux out
    // of uncovered neighbours, refluxFine_ applies one substep's fine flux
    // (read straight from the pool buffers).
    void refluxCoarse_(const Patch& p, Real dt) {
        GridRef c = coarseRef();
        const int bc = cfg_.blockC;
        const Real lx = dt / c.dx, ly = dt / c.dy;
        const int ctx = c.totx();
        const Cons* fxC = coarse_.fx();
        const Cons* fyC = coarse_.fy();
        const auto cid = [&](int i, int j) { return j * ctx + i; };

        if (p.bi > 0 && !covered(p.bi - 1, p.bj))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 - 1, NG + p.cj0 + r) +=
                    lx * fxC[cid(NG + p.ci0 - 1, NG + p.cj0 + r)];
        if (p.bi < nbx_ - 1 && !covered(p.bi + 1, p.bj))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 + bc, NG + p.cj0 + r) -=
                    lx * fxC[cid(NG + p.ci0 + bc - 1, NG + p.cj0 + r)];
        if (p.bj > 0 && !covered(p.bi, p.bj - 1))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 + r, NG + p.cj0 - 1) +=
                    ly * fyC[cid(NG + p.ci0 + r, NG + p.cj0 - 1)];
        if (p.bj < nby_ - 1 && !covered(p.bi, p.bj + 1))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 + r, NG + p.cj0 + bc) -=
                    ly * fyC[cid(NG + p.ci0 + r, NG + p.cj0 + bc - 1)];
    }

    void refluxFine_(const Patch& p, Real dt) {
        GridRef c = coarseRef();
        const int bc = cfg_.blockC;
        const Real lx = dt / c.dx, ly = dt / c.dy;
        const Cons* fxF = static_cast<const Cons*>(FxP_->contents()) +
                          std::size_t(p.slot) * stride_;
        const Cons* fyF = static_cast<const Cons*>(FyP_->contents()) +
                          std::size_t(p.slot) * stride_;
        const auto fid = [&](int i, int j) { return j * pTot_ + i; };

        if (p.bi > 0 && !covered(p.bi - 1, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fxF[fid(NG - 1, NG + 2 * r)] +
                                 fxF[fid(NG - 1, NG + 2 * r + 1)]);
                c.at(NG + p.ci0 - 1, NG + p.cj0 + r) -= lx * ff;
            }
        if (p.bi < nbx_ - 1 && !covered(p.bi + 1, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fxF[fid(NG + nf_ - 1, NG + 2 * r)] +
                                 fxF[fid(NG + nf_ - 1, NG + 2 * r + 1)]);
                c.at(NG + p.ci0 + bc, NG + p.cj0 + r) += lx * ff;
            }
        if (p.bj > 0 && !covered(p.bi, p.bj - 1))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fyF[fid(NG + 2 * r, NG - 1)] +
                                 fyF[fid(NG + 2 * r + 1, NG - 1)]);
                c.at(NG + p.ci0 + r, NG + p.cj0 - 1) -= ly * ff;
            }
        if (p.bj < nby_ - 1 && !covered(p.bi, p.bj + 1))
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (fyF[fid(NG + 2 * r, NG + nf_ - 1)] +
                                 fyF[fid(NG + 2 * r + 1, NG + nf_ - 1)]);
                c.at(NG + p.ci0 + r, NG + p.cj0 + bc) += ly * ff;
            }
    }

    // Patches cover disjoint coarse blocks: parallel-safe.
    void restrictFine_() {
        GridRef c = coarseRef();
        parallelFor(patches.size(), [&](std::size_t k) {
            const Patch& p = patches[k];
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
        });
    }

    MetalContext& ctx_;
    AmrConfig cfg_;
    Real x0_, y0_;
    Euler2DGpu coarse_;
    int nbx_, nby_, nf_ = 0, pTot_ = 0, stride_ = 0, capacity_ = 0;
    std::vector<int> blockOf_;
    std::vector<int> freeSlots_;
    std::vector<Cons> coarseOld_; // coarse t^n copy (subcycled ghosts)
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

#pragma once

// Hybrid CPU/GPU multi-level AMR: same recursive Berger-Colella scheme
// as AmrML (the CPU reference), with every patch of every level living
// in ONE shared slot pool (identical patch shape at all depths). The
// GPU steps the base grid and each level's patches (one batched dispatch
// per level per substep, via per-level slot lists); the CPU does ghost
// fill, refluxing, restriction, tagging and regridding in place.

#include "amr/Amr2.hpp" // AmrConfig
#include "backend/metal/Euler2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "core/Parallel.hpp"
#include "numerics/Limiter.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace mm {

class AmrGpuML {
public:
    struct Patch {
        int bi, bj, ci0, cj0;
        int slot;
        std::vector<Cons> old; // own state at parent-substep start
        std::vector<Euler2DGpu::PhiG> sold; // scalar (phi, Gamma) old
    };
    using PhiG = Euler2DGpu::PhiG;

    struct Level {
        int nbx = 0, nby = 0;
        std::vector<int> blockOf;
        std::vector<Patch> patches;
        MTL::Buffer* slots = nullptr;     // active slot list for dispatch
        MTL::Buffer* slotTable = nullptr; // block -> slot (~0u = none),
                                          // read by the live view
    };

    std::function<void(GridRef&, double)> fillPhysicalGhosts;
    std::function<void(GridRef&, double, unsigned)> fillPatchPhysical;

    AmrGpuML(MetalContext& ctx, int nx, int ny, Real x0, Real y0, Real lx,
             Real ly, AmrConfig cfg)
        : ctx_(ctx), cfg_(cfg), x0_(x0), y0_(y0),
          coarse_(ctx, nx, ny, lx / nx, ly / ny), nf_(2 * cfg.blockC) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
        coarse_.setViscosity(cfg.mu);
        coarse_.setGravity(cfg.gx, cfg.gy);
        pTot_ = nf_ + 2 * NG;
        stride_ = pTot_ * pTot_;

        lvls_.resize(cfg.maxLevels - 1);
        std::size_t totalBlocks = 0;
        for (int l = 1; l < cfg_.maxLevels; ++l) {
            Level& L = lvls_[l - 1];
            L.nbx = (nx << (l - 1)) / cfg.blockC;
            L.nby = (ny << (l - 1)) / cfg.blockC;
            L.blockOf.assign(std::size_t(L.nbx) * L.nby, -1);
            totalBlocks += std::size_t(L.nbx) * L.nby;
            L.slots = ctx.device()->newBuffer(
                std::size_t(L.nbx) * L.nby * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            L.slotTable = ctx.device()->newBuffer(
                std::size_t(L.nbx) * L.nby * sizeof(std::uint32_t),
                MTL::ResourceStorageModeShared);
            auto* tbl =
                static_cast<std::uint32_t*>(L.slotTable->contents());
            std::fill(tbl, tbl + std::size_t(L.nbx) * L.nby, ~0u);
        }
        capacity_ = int(std::min<std::size_t>(totalBlocks, 8192));

        const std::size_t bytes =
            std::size_t(capacity_) * stride_ * sizeof(Cons);
        const auto mk = [&] {
            return ctx.device()->newBuffer(bytes,
                                           MTL::ResourceStorageModeShared);
        };
        qP_ = mk(); xLP_ = mk(); xRP_ = mk(); yBP_ = mk(); yTP_ = mk();
        FxP_ = mk(); FyP_ = mk();
        smaxP_ = ctx.device()->newBuffer(3 * sizeof(std::uint32_t),
                                         MTL::ResourceStorageModeShared);
        for (int s = capacity_ - 1; s >= 0; --s) freeSlots_.push_back(s);

        lib_ = ctx.compileLibrary("euler2d.metal");
        predictorP_ = ctx.makePipeline(lib_, "predictor_pool");
        fluxXP_ = ctx.makePipeline(lib_, "flux_x_pool");
        fluxYP_ = ctx.makePipeline(lib_, "flux_y_pool");
        updateP_ = ctx.makePipeline(lib_, "update_pool");
        waveP_ = ctx.makePipeline(lib_, "wave_pool");

        if (cfg_.species) {
            gas_ = GasPair{cfg_.gamma1, cfg_.gamma2};
            coarse_.enableSpecies(gas_);
            sP_ = ctx.device()->newBuffer(
                std::size_t(capacity_) * stride_ * sizeof(PhiG),
                MTL::ResourceStorageModeShared);
            fXP_ = mk(); fYP_ = mk(); sFxP_ = mk(); sFyP_ = mk();
            predictorYP_ = ctx.makePipeline(lib_, "predictor_y_pool");
            fluxXYP_ = ctx.makePipeline(lib_, "flux_x_y_pool");
            fluxYYP_ = ctx.makePipeline(lib_, "flux_y_y_pool");
            updateYP_ = ctx.makePipeline(lib_, "update_y_pool");
            waveYP_ = ctx.makePipeline(lib_, "wave_y_pool");
        }
    }

    ~AmrGpuML() {
        for (Level& L : lvls_) {
            L.slots->release();
            L.slotTable->release();
        }
        for (MTL::Buffer* b :
             {qP_, xLP_, xRP_, yBP_, yTP_, FxP_, FyP_, smaxP_})
            b->release();
        for (MTL::Buffer* b : {sP_, fXP_, fYP_, sFxP_, sFyP_})
            if (b) b->release();
        for (MTL::ComputePipelineState* p :
             {predictorP_, fluxXP_, fluxYP_, updateP_, waveP_})
            p->release();
        for (MTL::ComputePipelineState* p :
             {predictorYP_, fluxXYP_, fluxYYP_, updateYP_, waveYP_})
            if (p) p->release();
        lib_->release();
    }
    AmrGpuML(const AmrGpuML&) = delete;
    AmrGpuML& operator=(const AmrGpuML&) = delete;

    int numLevels() const { return cfg_.maxLevels; }
    bool species() const { return cfg_.species; }
    int fineCells() const { return nf_; }
    const Level& level(int l) const { return lvls_[l - 1]; }

    GridRef coarseRef() const {
        return const_cast<AmrGpuML*>(this)->coarse_.ref(x0_, y0_);
    }
    GridRef patchRef(int l, const Patch& p) const {
        const Real dxp = coarseRef().dx / Real(1 << (l - 1));
        const Real dyp = coarseRef().dy / Real(1 << (l - 1));
        return GridRef{nf_,
                       nf_,
                       x0_ + p.ci0 * dxp,
                       y0_ + p.cj0 * dyp,
                       dxp / 2,
                       dyp / 2,
                       static_cast<Cons*>(qP_->contents()) +
                           std::size_t(p.slot) * stride_};
    }

    PhiG* baseS() const {
        return const_cast<AmrGpuML*>(this)->coarse_.sData();
    }
    PhiG* sOf(const Patch& p) const {
        return static_cast<PhiG*>(sP_->contents()) +
               std::size_t(p.slot) * stride_;
    }

    bool covered(int l, int bi, int bj) const {
        const Level& L = lvls_[l - 1];
        return L.blockOf[std::size_t(bj) * L.nbx + bi] >= 0;
    }

    template <class IC>
    void init(IC ic) {
        GridRef b = coarseRef();
        for (int j = NG; j < NG + b.ny; ++j)
            for (int i = NG; i < NG + b.nx; ++i)
                b.at(i, j) = ic(b.xc(i), b.yc(j));
        fillPhysicalGhosts(b, 0);
        for (int pass = 1; pass < cfg_.maxLevels; ++pass) {
            regrid();
            for (int l = 1; l < cfg_.maxLevels; ++l)
                for (Patch& p : lvls_[l - 1].patches) {
                    GridRef g = patchRef(l, p);
                    for (int j = NG; j < NG + nf_; ++j)
                        for (int i = NG; i < NG + nf_; ++i)
                            g.at(i, j) = ic(g.xc(i), g.yc(j));
                }
            for (int l = cfg_.maxLevels - 1; l >= 1; --l)
                restrictLevel_(l);
        }
    }

    // Two-gas init: icY(x, y) -> mass fraction of gas 2. phi and Gamma
    // are rebuilt from Y on every level after each regrid pass.
    template <class IC, class ICY>
    void init(IC ic, ICY icY) {
        GridRef b = coarseRef();
        PhiG* bs = baseS();
        const auto setS = [&](PhiG* arr, const GridRef& g, int i, int j) {
            const Real Y = icY(g.xc(i), g.yc(j));
            arr[g.idx(i, j)] = {float(g.at(i, j).rho * Y),
                                float(gas_.Gamma(Y))};
        };
        for (int j = NG; j < NG + b.ny; ++j)
            for (int i = NG; i < NG + b.nx; ++i)
                b.at(i, j) = ic(b.xc(i), b.yc(j));
        fillPhysicalGhosts(b, 0);
        for (int j = 0; j < b.toty(); ++j)
            for (int i = 0; i < b.totx(); ++i) setS(bs, b, i, j);
        for (int pass = 1; pass < cfg_.maxLevels; ++pass) {
            regrid();
            for (int l = 1; l < cfg_.maxLevels; ++l)
                for (Patch& p : lvls_[l - 1].patches) {
                    GridRef g = patchRef(l, p);
                    PhiG* ps = sOf(p);
                    for (int j = NG; j < NG + nf_; ++j)
                        for (int i = NG; i < NG + nf_; ++i) {
                            g.at(i, j) = ic(g.xc(i), g.yc(j));
                            setS(ps, g, i, j);
                        }
                }
            for (int l = cfg_.maxLevels - 1; l >= 1; --l)
                restrictLevel_(l);
        }
    }

    Real maxStableDtAll(Real cfl) {
        coarse_.zeroWave();
        auto* smp = static_cast<std::uint32_t*>(smaxP_->contents());
        smp[0] = smp[1] = 0;
        smp[2] = std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::max());

        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        coarse_.encodeWave(cmd);
        for (int l = 1; l < cfg_.maxLevels; ++l)
            if (!lvls_[l - 1].patches.empty()) {
                if (cfg_.species)
                    encodePool_(cmd, l, waveYP_, {qP_, sP_, smaxP_}, 0,
                                nf_, nf_);
                else
                    encodePool_(cmd, l, waveP_, {qP_, smaxP_}, 0, nf_,
                                nf_);
            }
        cmd->commit();
        cmd->waitUntilCompleted();

        const auto [sxC, syC] = coarse_.waveSpeeds();
        Real sx = sxC, sy = syC, rhoMin = coarse_.rhoMin();
        bool any = false;
        for (int l = 1; l < cfg_.maxLevels; ++l)
            any = any || !lvls_[l - 1].patches.empty();
        if (any) {
            sx = std::max(sx, std::bit_cast<float>(smp[0]));
            sy = std::max(sy, std::bit_cast<float>(smp[1]));
            rhoMin = std::min(rhoMin, std::bit_cast<float>(smp[2]));
        }

        const GridRef b = coarseRef();
        const int D = deepestActive_();
        // Subcycled: dt0 = cfl*dx0/smax satisfies every level (dx and dt
        // both halve per level). Non-subcycled: the finest spacing binds.
        Real dt = cfg_.subcycle
            ? cfl * std::min(b.dx / sx, b.dy / sy)
            : cfl * std::min(b.dx / Real(1 << D) / sx,
                             b.dy / Real(1 << D) / sy);
        if (cfg_.mu > 0) {
            // Viscous limit scales as dx^2: the deepest level binds even
            // subcycled (its dt only halves per level).
            Real dtV = Euler2DGpu::viscousDtLimit(
                cfl, b.dx / Real(1 << D), b.dy / Real(1 << D),
                cfg_.mu / rhoMin);
            if (cfg_.subcycle) dtV *= Real(1 << D);
            dt = std::min(dt, dtV);
        }
        return dt;
    }

    void step(Real dt, double t) {
        GridRef b = coarseRef();
        fillPhysicalGhosts(b, t);
        if (cfg_.species)
            scalarPhysicalSides_(b, baseS(),
                                 SideLeft | SideRight | SideBottom |
                                     SideTop);
        advanceTree_(0, dt, t);
    }

    double totalMass() const {
        double m = 0;
        const GridRef b = coarseRef();
        const double a0 = double(b.dx) * b.dy;
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i)
                if (cfg_.maxLevels < 2 ||
                    !covered(1, i / cfg_.blockC, j / cfg_.blockC))
                    m += double(b.at(NG + i, NG + j).rho) * a0;
        for (int l = 1; l < cfg_.maxLevels; ++l)
            for (const Patch& p : lvls_[l - 1].patches) {
                const GridRef g = patchRef(l, p);
                const double al = double(g.dx) * g.dy;
                for (int j = 0; j < nf_; ++j)
                    for (int i = 0; i < nf_; ++i) {
                        const int gi = 2 * p.ci0 + i;
                        const int gj = 2 * p.cj0 + j;
                        if (l < cfg_.maxLevels - 1 &&
                            covered(l + 1, gi / cfg_.blockC,
                                    gj / cfg_.blockC))
                            continue;
                        m += double(g.at(NG + i, NG + j).rho) * al;
                    }
            }
        return m;
    }

    double totalSpeciesMass() const {
        double m = 0;
        const GridRef b = coarseRef();
        const PhiG* bs = baseS();
        const double a0 = double(b.dx) * b.dy;
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i)
                if (cfg_.maxLevels < 2 ||
                    !covered(1, i / cfg_.blockC, j / cfg_.blockC))
                    m += double(bs[b.idx(NG + i, NG + j)].phi) * a0;
        for (int l = 1; l < cfg_.maxLevels; ++l)
            for (const Patch& p : lvls_[l - 1].patches) {
                const GridRef g = patchRef(l, p);
                const PhiG* ps = sOf(p);
                const double al = double(g.dx) * g.dy;
                for (int j = 0; j < nf_; ++j)
                    for (int i = 0; i < nf_; ++i) {
                        const int gi = 2 * p.ci0 + i;
                        const int gj = 2 * p.cj0 + j;
                        if (l < cfg_.maxLevels - 1 &&
                            covered(l + 1, gi / cfg_.blockC,
                                    gj / cfg_.blockC))
                            continue;
                        m += double(ps[g.idx(NG + i, NG + j)].phi) * al;
                    }
            }
        return m;
    }

    std::size_t cellCount() const {
        const GridRef b = coarseRef();
        std::size_t n = std::size_t(b.nx) * b.ny;
        for (const Level& L : lvls_)
            n += L.patches.size() * std::size_t(nf_) * nf_;
        return n;
    }
    std::size_t patchCount(int l) const {
        return lvls_[l - 1].patches.size();
    }

    MetalContext& context() const { return ctx_; }

    // ---- live-view access (zero-copy rendering of the hierarchy) ----
    MTL::Buffer* renderBaseQ() const { return coarse_.qBuffer(); }
    MTL::Buffer* renderPoolQ() const { return qP_; }
    MTL::Buffer* renderSlotTable(int l) const {
        return lvls_[l - 1].slotTable;
    }
    int renderPTot() const { return pTot_; }
    int renderStride() const { return stride_; }
    int renderBlockC() const { return cfg_.blockC; }

    void regrid() { regridFrom_(1); }

private:
    // ---- geometry --------------------------------------------------------
    int nxAt_(int l) const { return coarseRef().nx << l; }
    int nyAt_(int l) const { return coarseRef().ny << l; }
    int deepestActive_() const {
        int d = 0;
        for (int l = 1; l < cfg_.maxLevels; ++l)
            if (!lvls_[l - 1].patches.empty()) d = l;
        return d;
    }

    const Patch* ownerAt_(int l, int cg, int cgj) const {
        const Level& L = lvls_[l - 1];
        const int pi =
            L.blockOf[std::size_t(cgj / nf_) * L.nbx + cg / nf_];
        return pi >= 0 ? &L.patches[pi] : nullptr;
    }

    unsigned domainSides_(int l, const Patch& p) const {
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        unsigned s = 0;
        if (!cfg_.periodicX)
            s |= (p.ci0 == 0 ? SideLeft : 0u) |
                 (p.ci0 + cfg_.blockC == nxp ? SideRight : 0u);
        if (!cfg_.periodicY)
            s |= (p.cj0 == 0 ? SideBottom : 0u) |
                 (p.cj0 + cfg_.blockC == nyp ? SideTop : 0u);
        return s;
    }

    // ---- GPU dispatch ------------------------------------------------------
    void encodePool_(MTL::CommandBuffer* cmd, int l,
                     MTL::ComputePipelineState* pso,
                     std::initializer_list<MTL::Buffer*> bufs, Real dt,
                     int w, int h) const {
        const Level& L = lvls_[l - 1];
        const GridRef b = coarseRef();
        const float dxl = b.dx / Real(1 << l);
        const float dyl = b.dy / Real(1 << l);
        const float kT = cfg_.mu > 0
            ? cfg_.mu * GAMMA / ((GAMMA - 1) * PRANDTL)
            : 0;
        const Euler2DGpu::Params p{pTot_, pTot_, nf_, nf_, dxl, dyl,
                                   dt,    stride_, cfg_.mu, kT,
                                   cfg_.gx, cfg_.gy, gas_.gamma1,
                                   gas_.gamma2};
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        int slot = 0;
        for (MTL::Buffer* bb : bufs) enc->setBuffer(bb, 0, slot++);
        enc->setBytes(&p, sizeof(p), slot++);
        enc->setBuffer(L.slots, 0, slot);
        enc->dispatchThreads(MTL::Size(w, h, L.patches.size()),
                             MTL::Size(8, 8, 1));
        enc->endEncoding();
    }

    void stepLevel_(int l, Real dt) {
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        if (l == 0) {
            coarse_.encodeStep(cmd, dt);
        } else if (cfg_.species) {
            encodePool_(cmd, l, predictorYP_,
                        {qP_, sP_, xLP_, xRP_, yBP_, yTP_, fXP_, fYP_},
                        dt, pTot_ - 2, pTot_ - 2);
            encodePool_(cmd, l, fluxXYP_, {xLP_, xRP_, fXP_, FxP_, sFxP_},
                        dt, nf_ + 1, nf_);
            encodePool_(cmd, l, fluxYYP_, {yBP_, yTP_, fYP_, FyP_, sFyP_},
                        dt, nf_, nf_ + 1);
            encodePool_(cmd, l, updateYP_,
                        {qP_, sP_, FxP_, FyP_, sFxP_, sFyP_}, dt, nf_,
                        nf_);
        } else {
            encodePool_(cmd, l, predictorP_, {qP_, xLP_, xRP_, yBP_, yTP_},
                        dt, pTot_ - 2, pTot_ - 2);
            encodePool_(cmd, l, fluxXP_, {xLP_, xRP_, qP_, FxP_}, dt,
                        nf_ + 1, nf_);
            encodePool_(cmd, l, fluxYP_, {yBP_, yTP_, qP_, FyP_}, dt, nf_,
                        nf_ + 1);
            encodePool_(cmd, l, updateP_, {qP_, FxP_, FyP_}, dt, nf_, nf_);
        }
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    const Cons* fxOf_(const Patch& p) const {
        return static_cast<const Cons*>(FxP_->contents()) +
               std::size_t(p.slot) * stride_;
    }
    const Cons* fyOf_(const Patch& p) const {
        return static_cast<const Cons*>(FyP_->contents()) +
               std::size_t(p.slot) * stride_;
    }
    // packed (Fpx, Ssx, Fgx, 0) — only .rho (= Fp) is refluxed
    const Cons* sfxOf_(const Patch& p) const {
        return static_cast<const Cons*>(sFxP_->contents()) +
               std::size_t(p.slot) * stride_;
    }
    const Cons* sfyOf_(const Patch& p) const {
        return static_cast<const Cons*>(sFyP_->contents()) +
               std::size_t(p.slot) * stride_;
    }

    // ---- tagging (same criteria as AmrML) ----------------------------------
    template <class At>
    bool tagCell_(At&& at, int i, int j, int ip, int im, int jp,
                  int jm) const {
        const Real r0 = at(i, j).rho;
        const Real ex = std::fabs(at(ip, j).rho - at(im, j).rho);
        const Real ey = std::fabs(at(i, jp).rho - at(i, jm).rho);
        if (std::max(ex, ey) / r0 > cfg_.tagThreshold) return true;
        if (cfg_.tagVelocity <= 0) return false;
        const auto u = [&](int a, int b) {
            const Cons& q = at(a, b);
            return q.mx / std::max(q.rho, RHO_FLOOR);
        };
        const auto v = [&](int a, int b) {
            const Cons& q = at(a, b);
            return q.my / std::max(q.rho, RHO_FLOOR);
        };
        const Real du = std::max(std::fabs(u(ip, j) - u(im, j)),
                                 std::fabs(u(i, jp) - u(i, jm)));
        const Real dv = std::max(std::fabs(v(ip, j) - v(im, j)),
                                 std::fabs(v(i, jp) - v(i, jm)));
        const Real c0 = soundSpeed(toPrim(at(i, j)));
        return std::max(du, dv) / c0 > cfg_.tagVelocity;
    }

    void markDilated_(std::vector<std::uint8_t>& want, const Level& lv,
                      int nx, int ny, int ci, int cj) const {
        for (int b = cj - 2; b <= cj + 2; ++b)
            for (int a = ci - 2; a <= ci + 2; ++a) {
                int x = a, y = b;
                if (cfg_.periodicX) x = (x % nx + nx) % nx;
                if (cfg_.periodicY) y = (y % ny + ny) % ny;
                if (x < 0 || x >= nx || y < 0 || y >= ny) continue;
                want[std::size_t(y / cfg_.blockC) * lv.nbx +
                     x / cfg_.blockC] = 1;
            }
    }

    void tagBase_(std::vector<std::uint8_t>& want) const {
        const Level& lv = lvls_[0];
        const GridRef b = coarseRef();
        const auto at = [&](int a, int c) -> const Cons& {
            return b.at(NG + a, NG + c);
        };
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i) {
                const int ip = std::min(i + 1, b.nx - 1),
                          im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, b.ny - 1),
                          jm = std::max(j - 1, 0);
                if (tagCell_(at, i, j, ip, im, jp, jm))
                    markDilated_(want, lv, b.nx, b.ny, i, j);
            }
    }

    void tagLevel_(int l, std::vector<std::uint8_t>& want) const {
        const Level& kid = lvls_[l];
        const int nx = nxAt_(l), ny = nyAt_(l);
        for (const Patch& p : lvls_[l - 1].patches) {
            const GridRef g = patchRef(l, p);
            const auto at = [&](int a, int b) -> const Cons& {
                return g.at(NG + a, NG + b);
            };
            for (int j = 0; j < nf_; ++j)
                for (int i = 0; i < nf_; ++i)
                    if (tagCell_(at, i, j, i + 1, i - 1, j + 1, j - 1))
                        markDilated_(want, kid, nx, ny, 2 * p.ci0 + i,
                                     2 * p.cj0 + j);
        }
    }

    // ---- prolongation -------------------------------------------------------
    Cons blend_(const Cons& cur, const Cons* oldArr, std::size_t idx,
                Real theta) const {
        if (theta < 0 || theta >= 1 || oldArr == nullptr) return cur;
        if (theta <= 0) return oldArr[idx];
        return (1 - theta) * oldArr[idx] + theta * cur;
    }

    Cons parentVal_(int l, const Patch* home, int cg, int cgj,
                    Real theta) const {
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
        if (l == 1) {
            const GridRef b = coarseRef();
            const std::size_t id = b.idx(NG + cg, NG + cgj);
            return blend_(b.q[id],
                          baseOld_.empty() ? nullptr : baseOld_.data(),
                          id, theta);
        }
        const Patch* Q = (cg >= 0 && cg < nxp && cgj >= 0 && cgj < nyp)
                             ? ownerAt_(l - 1, cg, cgj)
                             : nullptr;
        if (Q == nullptr) Q = home;
        const GridRef qg = patchRef(l - 1, *Q);
        const std::size_t id =
            qg.idx(NG + cg - 2 * Q->ci0, NG + cgj - 2 * Q->cj0);
        return blend_(qg.q[id],
                      Q->old.empty() ? nullptr : Q->old.data(), id,
                      theta);
    }

    Cons prolong_(int l, const Patch* home, int cg, int cgj, int ox,
                  int oy, Real theta) const {
        const Cons q0 = parentVal_(l, home, cg, cgj, theta);
        const Cons dqx =
            limitedSlope(parentVal_(l, home, cg - 1, cgj, theta), q0,
                         parentVal_(l, home, cg + 1, cgj, theta));
        const Cons dqy =
            limitedSlope(parentVal_(l, home, cg, cgj - 1, theta), q0,
                         parentVal_(l, home, cg, cgj + 1, theta));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    PhiG sblend_(const PhiG& cur, const PhiG* oldArr, std::size_t idx,
                 Real theta) const {
        if (theta < 0 || theta >= 1 || oldArr == nullptr) return cur;
        if (theta <= 0) return oldArr[idx];
        const Real t = theta;
        return {float((1 - t) * oldArr[idx].phi + t * cur.phi),
                float((1 - t) * oldArr[idx].G + t * cur.G)};
    }

    PhiG sparentVal_(int l, const Patch* home, int cg, int cgj,
                     Real theta) const {
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
        if (l == 1) {
            const GridRef b = coarseRef();
            const std::size_t id = b.idx(NG + cg, NG + cgj);
            return sblend_(baseS()[id],
                           baseOldS_.empty() ? nullptr : baseOldS_.data(),
                           id, theta);
        }
        const Patch* Q = (cg >= 0 && cg < nxp && cgj >= 0 && cgj < nyp)
                             ? ownerAt_(l - 1, cg, cgj)
                             : nullptr;
        if (Q == nullptr) Q = home;
        const std::size_t id = patchRef(l - 1, *Q).idx(
            NG + cg - 2 * Q->ci0, NG + cgj - 2 * Q->cj0);
        return sblend_(sOf(*Q)[id],
                       Q->sold.empty() ? nullptr : Q->sold.data(), id,
                       theta);
    }

    PhiG sprolong_(int l, const Patch* home, int cg, int cgj, int ox,
                   int oy, Real theta) const {
        const PhiG q0 = sparentVal_(l, home, cg, cgj, theta);
        const PhiG qw = sparentVal_(l, home, cg - 1, cgj, theta);
        const PhiG qe = sparentVal_(l, home, cg + 1, cgj, theta);
        const PhiG qs = sparentVal_(l, home, cg, cgj - 1, theta);
        const PhiG qn = sparentVal_(l, home, cg, cgj + 1, theta);
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        const auto one = [&](Real w, Real c, Real e, Real so, Real n) {
            return c + sx * mcSlope(c - w, e - c) +
                   sy * mcSlope(c - so, n - c);
        };
        return {float(one(qw.phi, q0.phi, qe.phi, qs.phi, qn.phi)),
                float(one(qw.G, q0.G, qe.G, qs.G, qn.G))};
    }

    // Transmissive scalar ghosts on the requested domain sides only.
    void scalarPhysicalSides_(const GridRef& g, PhiG* a,
                              unsigned sides) const {
        for (int j = 0; j < g.toty(); ++j)
            for (int k = 0; k < NG; ++k) {
                if (sides & SideLeft)
                    a[g.idx(k, j)] = a[g.idx(NG, j)];
                if (sides & SideRight)
                    a[g.idx(NG + g.nx + k, j)] =
                        a[g.idx(NG + g.nx - 1, j)];
            }
        for (int i = 0; i < g.totx(); ++i)
            for (int k = 0; k < NG; ++k) {
                if (sides & SideBottom)
                    a[g.idx(i, k)] = a[g.idx(i, NG)];
                if (sides & SideTop)
                    a[g.idx(i, NG + g.ny + k)] =
                        a[g.idx(i, NG + g.ny - 1)];
            }
    }

    Patch makePatch_(int l, int bi, int bj) {
        assert(!freeSlots_.empty());
        const int slot = freeSlots_.back();
        freeSlots_.pop_back();
        Patch p{bi, bj, bi * cfg_.blockC, bj * cfg_.blockC, slot, {}, {}};
        const Patch* home =
            l >= 2 ? ownerAt_(l - 1, p.ci0, p.cj0) : nullptr;
        GridRef g = patchRef(l, p);
        PhiG* ps = cfg_.species ? sOf(p) : nullptr;
        for (int j = NG; j < NG + nf_; ++j)
            for (int i = NG; i < NG + nf_; ++i) {
                const int gfi = 2 * p.ci0 + (i - NG);
                const int gfj = 2 * p.cj0 + (j - NG);
                g.at(i, j) = prolong_(l, home, gfi / 2, gfj / 2, gfi & 1,
                                      gfj & 1, Real(-1));
                if (ps)
                    ps[g.idx(i, j)] =
                        sprolong_(l, home, gfi / 2, gfj / 2, gfi & 1,
                                  gfj & 1, Real(-1));
            }
        return p;
    }

    // ---- ghosts --------------------------------------------------------------
    void fillLevelGhosts_(int l, double t, Real theta) {
        Level& lv = lvls_[l - 1];
        const int nxl = nxAt_(l), nyl = nyAt_(l);
        parallelFor(lv.patches.size(), [&](std::size_t k) {
            Patch& p = lv.patches[k];
            GridRef g = patchRef(l, p);
            PhiG* ps = cfg_.species ? sOf(p) : nullptr;
            const Patch* home =
                l >= 2 ? ownerAt_(l - 1, p.ci0, p.cj0) : nullptr;
            const auto fillCell = [&](int i, int j) {
                int gfi = 2 * p.ci0 + (i - NG);
                int gfj = 2 * p.cj0 + (j - NG);
                if (cfg_.periodicX) gfi = (gfi % nxl + nxl) % nxl;
                if (cfg_.periodicY) gfj = (gfj % nyl + nyl) % nyl;
                if (gfi >= 0 && gfi < nxl && gfj >= 0 && gfj < nyl)
                    if (const Patch* sp = ownerAt_(l, gfi, gfj)) {
                        const GridRef sg = patchRef(l, *sp);
                        const std::size_t src =
                            sg.idx(NG + gfi - 2 * sp->ci0,
                                   NG + gfj - 2 * sp->cj0);
                        g.q[g.idx(i, j)] = sg.q[src];
                        if (ps) ps[g.idx(i, j)] = sOf(*sp)[src];
                        return;
                    }
                const int cg = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cgj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                g.at(i, j) = prolong_(l, home, cg, cgj, gfi - 2 * cg,
                                      gfj - 2 * cgj, theta);
                if (ps)
                    ps[g.idx(i, j)] = sprolong_(l, home, cg, cgj,
                                                gfi - 2 * cg,
                                                gfj - 2 * cgj, theta);
            };
            for (int j = 0; j < NG; ++j)
                for (int i = 0; i < g.totx(); ++i) fillCell(i, j);
            for (int j = NG + nf_; j < g.toty(); ++j)
                for (int i = 0; i < g.totx(); ++i) fillCell(i, j);
            for (int j = NG; j < NG + nf_; ++j) {
                for (int i = 0; i < NG; ++i) fillCell(i, j);
                for (int i = NG + nf_; i < g.totx(); ++i) fillCell(i, j);
            }
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides_(l, p))
                    fillPatchPhysical(g, t, sides);
            if (ps)
                if (const unsigned sides = domainSides_(l, p))
                    scalarPhysicalSides_(g, ps, sides);
        });
    }

    // ---- time advance ----------------------------------------------------------
    void saveOld_(int l) {
        if (l == 0) {
            const GridRef b = coarseRef();
            baseOld_.assign(b.q, b.q + std::size_t(b.totx()) * b.toty());
            if (cfg_.species) {
                const PhiG* bs = baseS();
                baseOldS_.assign(bs,
                                 bs + std::size_t(b.totx()) * b.toty());
            }
            return;
        }
        for (Patch& p : lvls_[l - 1].patches) {
            const GridRef g = patchRef(l, p);
            p.old.assign(g.q, g.q + std::size_t(stride_));
            if (cfg_.species) {
                const PhiG* ps = sOf(p);
                p.sold.assign(ps, ps + std::size_t(stride_));
            }
        }
    }

    void advanceTree_(int l, Real dt, double t) {
        const bool hasKids =
            (l + 1 < cfg_.maxLevels) && !lvls_[l].patches.empty();
        if (hasKids) saveOld_(l);
        stepLevel_(l, dt);
        if (hasKids) {
            const int lc = l + 1;
            if (cfg_.reflux) refluxBackOut_(lc, dt);
            const int n = cfg_.subcycle ? 2 : 1;
            const Real cdt = dt / n;
            for (int k = 0; k < n; ++k) {
                fillLevelGhosts_(lc, t + k * cdt, Real(k) / Real(n));
                advanceTree_(lc, cdt, t + k * cdt);
                if (cfg_.reflux) refluxFineApply_(lc, cdt);
            }
            restrictLevel_(lc);
        }
        if (l + 1 < cfg_.maxLevels &&
            ++stepCounts_[l] % cfg_.regridEvery == 0) {
            if (l == 0) {
                GridRef b = coarseRef();
                fillPhysicalGhosts(b, t + dt);
            }
            regridFrom_(l + 1);
        }
    }

    // ---- coarse-fine coupling ------------------------------------------------
    struct ParentCell {
        Cons* cells;
        const Cons *Fx, *Fy;
        int totx, li, lj;
        // two-gas fields (null when species disabled); the scalar flux
        // arrays are packed (Fp, Ss, Fg, 0) and only .rho is refluxed
        PhiG* sc = nullptr;
        const Cons *sFx = nullptr, *sFy = nullptr;
    };
    ParentCell parentCell_(int lp, int cg, int cgj) {
        if (lp == 0) {
            const GridRef b = coarseRef();
            ParentCell pc{b.q,      coarse_.fx(), coarse_.fy(),
                          b.totx(), NG + cg,      NG + cgj};
            if (cfg_.species) {
                pc.sc = baseS();
                pc.sFx = coarse_.sfx();
                pc.sFy = coarse_.sfy();
            }
            return pc;
        }
        const Patch* Q = ownerAt_(lp, cg, cgj);
        assert(Q != nullptr);
        ParentCell pc{patchRef(lp, *Q).q,
                      fxOf_(*Q),
                      fyOf_(*Q),
                      pTot_,
                      NG + cg - 2 * Q->ci0,
                      NG + cgj - 2 * Q->cj0};
        if (cfg_.species) {
            pc.sc = sOf(*Q);
            pc.sFx = sfxOf_(*Q);
            pc.sFy = sfyOf_(*Q);
        }
        return pc;
    }

    struct SideNb {
        bool ok;
        int blockBi, blockBj, cg, cgj;
    };
    SideNb sideNb_(int lc, const Patch& p, int dir) const {
        const Level& L = lvls_[lc - 1];
        const int bC = cfg_.blockC;
        const int nxp = nxAt_(lc - 1), nyp = nyAt_(lc - 1);
        int bi = p.bi, bj = p.bj, cg = 0, cgj = 0;
        switch (dir) {
        case 0:
            bi -= 1; cg = p.ci0 - 1; cgj = p.cj0;
            if (bi < 0) {
                if (!cfg_.periodicX) return {false, 0, 0, 0, 0};
                bi = L.nbx - 1; cg = nxp - 1;
            }
            break;
        case 1:
            bi += 1; cg = p.ci0 + bC; cgj = p.cj0;
            if (bi >= L.nbx) {
                if (!cfg_.periodicX) return {false, 0, 0, 0, 0};
                bi = 0; cg = 0;
            }
            break;
        case 2:
            bj -= 1; cg = p.ci0; cgj = p.cj0 - 1;
            if (bj < 0) {
                if (!cfg_.periodicY) return {false, 0, 0, 0, 0};
                bj = L.nby - 1; cgj = nyp - 1;
            }
            break;
        default:
            bj += 1; cg = p.ci0; cgj = p.cj0 + bC;
            if (bj >= L.nby) {
                if (!cfg_.periodicY) return {false, 0, 0, 0, 0};
                bj = 0; cgj = 0;
            }
            break;
        }
        return {true, bi, bj, cg, cgj};
    }

    void refluxBackOut_(int lc, Real dtParent) {
        const int bC = cfg_.blockC;
        for (const Patch& p : lvls_[lc - 1].patches)
            for (int dir = 0; dir < 4; ++dir) {
                const SideNb n = sideNb_(lc, p, dir);
                if (!n.ok || covered(lc, n.blockBi, n.blockBj)) continue;
                for (int r = 0; r < bC; ++r) {
                    const int cg = (dir < 2) ? n.cg : n.cg + r;
                    const int cgj = (dir < 2) ? n.cgj + r : n.cgj;
                    ParentCell pc = parentCell_(lc - 1, cg, cgj);
                    const GridRef b = coarseRef();
                    const Real dxp = b.dx / Real(1 << (lc - 1));
                    const Real dyp = b.dy / Real(1 << (lc - 1));
                    const Real lam = dtParent / (dir < 2 ? dxp : dyp);
                    Cons& cell = pc.cells[std::size_t(pc.lj) * pc.totx +
                                          pc.li];
                    const auto fid = [&](int a, int b2) {
                        return std::size_t(b2) * pc.totx + a;
                    };
                    float* ph = pc.sc
                        ? &pc.sc[fid(pc.li, pc.lj)].phi
                        : nullptr;
                    if (dir == 0) {
                        cell += lam * pc.Fx[fid(pc.li, pc.lj)];
                        if (ph)
                            *ph += lam * pc.sFx[fid(pc.li, pc.lj)].rho;
                    } else if (dir == 1) {
                        cell -= lam * pc.Fx[fid(pc.li - 1, pc.lj)];
                        if (ph)
                            *ph -= lam * pc.sFx[fid(pc.li - 1, pc.lj)].rho;
                    } else if (dir == 2) {
                        cell += lam * pc.Fy[fid(pc.li, pc.lj)];
                        if (ph)
                            *ph += lam * pc.sFy[fid(pc.li, pc.lj)].rho;
                    } else {
                        cell -= lam * pc.Fy[fid(pc.li, pc.lj - 1)];
                        if (ph)
                            *ph -= lam * pc.sFy[fid(pc.li, pc.lj - 1)].rho;
                    }
                }
            }
    }

    void refluxFineApply_(int lc, Real dtChild) {
        const int bC = cfg_.blockC;
        for (const Patch& p : lvls_[lc - 1].patches) {
            const Cons* fxF = fxOf_(p);
            const Cons* fyF = fyOf_(p);
            const Cons* sfxF = cfg_.species ? sfxOf_(p) : nullptr;
            const Cons* sfyF = cfg_.species ? sfyOf_(p) : nullptr;
            const auto pid = [&](int a, int b) {
                return std::size_t(b) * pTot_ + a;
            };
            for (int dir = 0; dir < 4; ++dir) {
                const SideNb n = sideNb_(lc, p, dir);
                if (!n.ok || covered(lc, n.blockBi, n.blockBj)) continue;
                for (int r = 0; r < bC; ++r) {
                    const int cg = (dir < 2) ? n.cg : n.cg + r;
                    const int cgj = (dir < 2) ? n.cgj + r : n.cgj;
                    ParentCell pc = parentCell_(lc - 1, cg, cgj);
                    const GridRef b = coarseRef();
                    const Real dxp = b.dx / Real(1 << (lc - 1));
                    const Real dyp = b.dy / Real(1 << (lc - 1));
                    const Real lam = dtChild / (dir < 2 ? dxp : dyp);
                    Cons ff;
                    Real fp = 0;
                    if (dir == 0) {
                        ff = Real(0.5) * (fxF[pid(NG - 1, NG + 2 * r)] +
                                          fxF[pid(NG - 1, NG + 2 * r + 1)]);
                        if (sfxF)
                            fp = Real(0.5) *
                                 (sfxF[pid(NG - 1, NG + 2 * r)].rho +
                                  sfxF[pid(NG - 1, NG + 2 * r + 1)].rho);
                    } else if (dir == 1) {
                        ff = Real(0.5) *
                             (fxF[pid(NG + nf_ - 1, NG + 2 * r)] +
                              fxF[pid(NG + nf_ - 1, NG + 2 * r + 1)]);
                        if (sfxF)
                            fp = Real(0.5) *
                                 (sfxF[pid(NG + nf_ - 1, NG + 2 * r)].rho +
                                  sfxF[pid(NG + nf_ - 1, NG + 2 * r + 1)]
                                      .rho);
                    } else if (dir == 2) {
                        ff = Real(0.5) * (fyF[pid(NG + 2 * r, NG - 1)] +
                                          fyF[pid(NG + 2 * r + 1, NG - 1)]);
                        if (sfyF)
                            fp = Real(0.5) *
                                 (sfyF[pid(NG + 2 * r, NG - 1)].rho +
                                  sfyF[pid(NG + 2 * r + 1, NG - 1)].rho);
                    } else {
                        ff = Real(0.5) *
                             (fyF[pid(NG + 2 * r, NG + nf_ - 1)] +
                              fyF[pid(NG + 2 * r + 1, NG + nf_ - 1)]);
                        if (sfyF)
                            fp = Real(0.5) *
                                 (sfyF[pid(NG + 2 * r, NG + nf_ - 1)].rho +
                                  sfyF[pid(NG + 2 * r + 1, NG + nf_ - 1)]
                                      .rho);
                    }
                    Cons& cell = pc.cells[std::size_t(pc.lj) * pc.totx +
                                          pc.li];
                    float* ph = pc.sc
                        ? &pc.sc[std::size_t(pc.lj) * pc.totx + pc.li].phi
                        : nullptr;
                    if (dir == 0 || dir == 2) {
                        cell -= lam * ff;
                        if (ph) *ph -= lam * fp;
                    } else {
                        cell += lam * ff;
                        if (ph) *ph += lam * fp;
                    }
                }
            }
        }
    }

    void restrictLevel_(int l) {
        const int bC = cfg_.blockC;
        for (const Patch& p : lvls_[l - 1].patches) {
            const GridRef g = patchRef(l, p);
            const PhiG* ps = cfg_.species ? sOf(p) : nullptr;
            for (int b = 0; b < bC; ++b)
                for (int a = 0; a < bC; ++a) {
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    const Cons sum = g.at(fi, fj) + g.at(fi + 1, fj) +
                                     g.at(fi, fj + 1) +
                                     g.at(fi + 1, fj + 1);
                    ParentCell pc =
                        parentCell_(l - 1, p.ci0 + a, p.cj0 + b);
                    const std::size_t pid =
                        std::size_t(pc.lj) * pc.totx + pc.li;
                    pc.cells[pid] = Real(0.25) * sum;
                    if (ps) {
                        const auto id = [&](int x, int y) {
                            return g.idx(x, y);
                        };
                        pc.sc[pid] = {
                            Real(0.25) * (ps[id(fi, fj)].phi +
                                          ps[id(fi + 1, fj)].phi +
                                          ps[id(fi, fj + 1)].phi +
                                          ps[id(fi + 1, fj + 1)].phi),
                            Real(0.25) * (ps[id(fi, fj)].G +
                                          ps[id(fi + 1, fj)].G +
                                          ps[id(fi, fj + 1)].G +
                                          ps[id(fi + 1, fj + 1)].G)};
                    }
                }
        }
    }

    // ---- regrid ----------------------------------------------------------------
    void regridFrom_(int lstart) {
        const int L = cfg_.maxLevels;
        if (lstart >= L) return;

        std::vector<std::vector<std::uint8_t>> want(L - 1);
        for (int l = lstart; l < L; ++l) {
            Level& lv = lvls_[l - 1];
            want[l - 1].assign(std::size_t(lv.nbx) * lv.nby, 0);
        }
        if (lstart == 1) tagBase_(want[0]);
        for (int l = std::max(lstart - 1, 1); l < L - 1; ++l)
            tagLevel_(l, want[l]);

        for (int l = L - 1; l >= lstart + 1; --l)
            forceParents_(l, want[l - 1], want[l - 2]);

        if (lstart >= 2) {
            Level& lv = lvls_[lstart - 1];
            for (int bj = 0; bj < lv.nby; ++bj)
                for (int bi = 0; bi < lv.nbx; ++bi) {
                    std::uint8_t& w =
                        want[lstart - 1][std::size_t(bj) * lv.nbx + bi];
                    if (w && !grownRegionCovered_(lstart, bi, bj)) w = 0;
                }
            for (int l = lstart + 1; l < L; ++l)
                clipToParentWant_(l, want[l - 1], want[l - 2]);
        }

        for (int l = lstart; l < L; ++l) {
            Level& lv = lvls_[l - 1];
            std::vector<Patch> next;
            std::vector<int> nextOf(lv.blockOf.size(), -1);
            // free dropped slots first so new patches can reuse them
            for (const Patch& p : lv.patches)
                if (!want[l - 1][std::size_t(p.bj) * lv.nbx + p.bi])
                    freeSlots_.push_back(p.slot);
            for (int bj = 0; bj < lv.nby; ++bj)
                for (int bi = 0; bi < lv.nbx; ++bi) {
                    const std::size_t b = std::size_t(bj) * lv.nbx + bi;
                    if (!want[l - 1][b]) continue;
                    const int old = lv.blockOf[b];
                    if (old >= 0)
                        next.push_back(std::move(lv.patches[old]));
                    else
                        next.push_back(makePatch_(l, bi, bj));
                    nextOf[b] = int(next.size()) - 1;
                }
            lv.patches = std::move(next);
            lv.blockOf = std::move(nextOf);
            auto* slots =
                static_cast<std::uint32_t*>(lv.slots->contents());
            auto* tbl =
                static_cast<std::uint32_t*>(lv.slotTable->contents());
            std::fill(tbl, tbl + lv.blockOf.size(), ~0u);
            for (std::size_t k = 0; k < lv.patches.size(); ++k) {
                slots[k] = std::uint32_t(lv.patches[k].slot);
                tbl[std::size_t(lv.patches[k].bj) * lv.nbx +
                    lv.patches[k].bi] = std::uint32_t(lv.patches[k].slot);
            }
        }
    }

    void forceParents_(int l, const std::vector<std::uint8_t>& kidWant,
                       std::vector<std::uint8_t>& parWant) const {
        const int bC = cfg_.blockC;
        const Level& kid = lvls_[l - 1];
        const Level& par = lvls_[l - 2];
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        for (int bj = 0; bj < kid.nby; ++bj)
            for (int bi = 0; bi < kid.nbx; ++bi) {
                if (!kidWant[std::size_t(bj) * kid.nbx + bi]) continue;
                for (int y = bj * bC - 1; y <= (bj + 1) * bC; ++y)
                    for (int x = bi * bC - 1; x <= (bi + 1) * bC; ++x) {
                        int cx = x, cy = y;
                        if (cfg_.periodicX) cx = (cx % nxp + nxp) % nxp;
                        if (cfg_.periodicY) cy = (cy % nyp + nyp) % nyp;
                        if (cx < 0 || cx >= nxp || cy < 0 || cy >= nyp)
                            continue;
                        parWant[std::size_t(cy / nf_) * par.nbx +
                                cx / nf_] = 1;
                    }
            }
    }

    bool grownRegionCovered_(int l, int bi, int bj) const {
        const int bC = cfg_.blockC;
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        for (int y = bj * bC - 1; y <= (bj + 1) * bC; ++y)
            for (int x = bi * bC - 1; x <= (bi + 1) * bC; ++x) {
                int cx = x, cy = y;
                if (cfg_.periodicX) cx = (cx % nxp + nxp) % nxp;
                if (cfg_.periodicY) cy = (cy % nyp + nyp) % nyp;
                if (cx < 0 || cx >= nxp || cy < 0 || cy >= nyp) continue;
                if (!covered(l - 1, cx / nf_, cy / nf_)) return false;
            }
        return true;
    }

    void clipToParentWant_(int l, std::vector<std::uint8_t>& kidWant,
                           const std::vector<std::uint8_t>& parWant) const {
        const int bC = cfg_.blockC;
        const Level& kid = lvls_[l - 1];
        const Level& par = lvls_[l - 2];
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        for (int bj = 0; bj < kid.nby; ++bj)
            for (int bi = 0; bi < kid.nbx; ++bi) {
                std::uint8_t& w = kidWant[std::size_t(bj) * kid.nbx + bi];
                if (!w) continue;
                for (int y = bj * bC - 1; w && y <= (bj + 1) * bC; ++y)
                    for (int x = bi * bC - 1; w && x <= (bi + 1) * bC;
                         ++x) {
                        int cx = x, cy = y;
                        if (cfg_.periodicX) cx = (cx % nxp + nxp) % nxp;
                        if (cfg_.periodicY) cy = (cy % nyp + nyp) % nyp;
                        if (cx < 0 || cx >= nxp || cy < 0 || cy >= nyp)
                            continue;
                        if (!parWant[std::size_t(cy / nf_) * par.nbx +
                                     cx / nf_])
                            w = 0;
                    }
            }
    }

    MetalContext& ctx_;
    AmrConfig cfg_;
    Real x0_, y0_;
    Euler2DGpu coarse_;
    int nf_, pTot_ = 0, stride_ = 0, capacity_ = 0;
    std::vector<Level> lvls_;
    std::vector<int> freeSlots_;
    std::vector<Cons> baseOld_;
    std::vector<PhiG> baseOldS_;
    GasPair gas_;
    std::vector<int> stepCounts_ = std::vector<int>(16, 0);

    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *predictorP_ = nullptr, *fluxXP_ = nullptr,
                              *fluxYP_ = nullptr, *updateP_ = nullptr,
                              *waveP_ = nullptr;
    MTL::Buffer *qP_ = nullptr, *xLP_ = nullptr, *xRP_ = nullptr,
                *yBP_ = nullptr, *yTP_ = nullptr, *FxP_ = nullptr,
                *FyP_ = nullptr, *smaxP_ = nullptr;
    MTL::Buffer *sP_ = nullptr, *fXP_ = nullptr, *fYP_ = nullptr,
                *sFxP_ = nullptr, *sFyP_ = nullptr;
    MTL::ComputePipelineState *predictorYP_ = nullptr,
                              *fluxXYP_ = nullptr, *fluxYYP_ = nullptr,
                              *updateYP_ = nullptr, *waveYP_ = nullptr;
};

} // namespace mm

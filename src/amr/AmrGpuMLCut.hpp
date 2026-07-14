#pragma once

// Hybrid CPU/GPU ARBITRARY-DEPTH AMR with embedded-boundary CUT CELLS — the
// GPU analogue of AmrML's cut-cell path (which AmrGpuCut is to Amr2). The base
// level and every refinement level live in shared Metal buffers: a
// CutCell2DGpu for the base, one pooled CutCell2DGpu per level. The GPU forms
// the divergence (aperture-weighted fluxes + gather-form flux redistribution);
// the CPU orchestrates the recursive Berger-Colella subcycling, cross-patch
// D^c ghost exchange, cut-aware reflux, kappa-restriction, EB-band tagging and
// slot management in place (unified memory, zero copies).
//
// As in AmrGpuCut, the gather-form FRD makes the CPU D-scatter unnecessary:
// exchanging each patch's D^c ghosts from its same-level siblings reproduces
// the cross-patch coupling exactly (a patch's interior D equals the monolithic
// single-grid divergence over the tiled region), so this matches AmrML's
// scatter path. Inviscid, single-gas (the cut path); no WENO / species / solid
// mask (cut cells replace the staircase).

#include "amr/Amr2.hpp" // AmrConfig, cut helpers, limitedSlope, floorState
#include "backend/metal/CutCell2DGpu.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace mm {

class AmrGpuMLCut {
public:
    struct Patch {
        int bi, bj;    // block coords (in level l-1 cells / blockC)
        int ci0, cj0;  // block origin in level-(l-1) cells
        int slot;      // index into the level's pool
        cutcell::Geometry geo;   // cut-cell moments (CPU mirror)
        std::vector<Cons> old;   // own state at parent-substep start (subcycle)
    };

    struct Level { // level l >= 1, stored at lvls_[l-1]
        int nbx = 0, nby = 0;
        std::vector<int> blockOf;
        std::vector<Patch> patches;
        std::vector<int> freeSlots;
        std::unique_ptr<CutCell2DGpu> pool;
    };

    std::function<void(GridRef&, double)> fillPhysicalGhosts;
    std::function<void(GridRef&, double, unsigned)> fillPatchPhysical;
    std::function<CellMoments(double, double, double, double)> momentFn;

    AmrGpuMLCut(MetalContext& ctx, int nx, int ny, Real x0, Real y0, Real lx,
                Real ly, AmrConfig cfg)
        : ctx_(ctx), cfg_(cfg), x0_(x0), y0_(y0), nx_(nx), ny_(ny),
          dxc_(lx / nx), dyc_(ly / ny),
          coarse_(ctx, nx, ny, lx / nx, ly / ny), nf_(2 * cfg.blockC) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
        assert(cfg.maxLevels >= 1);
        ptx_ = nf_ + 2 * NG;
        ctxs_ = nx_ + 2 * NG;
        lvls_.resize(cfg.maxLevels - 1);
        for (int l = 1; l < cfg.maxLevels; ++l) {
            Level& L = lvls_[l - 1];
            L.nbx = (nx << (l - 1)) / cfg.blockC;
            L.nby = (ny << (l - 1)) / cfg.blockC;
            L.blockOf.assign(std::size_t(L.nbx) * L.nby, -1);
            const int cap = L.nbx * L.nby;
            L.pool = std::make_unique<CutCell2DGpu>(ctx, nf_, nf_, dxAt_(l),
                                                    dyAt_(l));
            L.pool->enablePool(cap);
            for (int s = cap - 1; s >= 0; --s) L.freeSlots.push_back(s);
        }
    }

    AmrGpuMLCut(const AmrGpuMLCut&) = delete;
    AmrGpuMLCut& operator=(const AmrGpuMLCut&) = delete;

    int numLevels() const { return cfg_.maxLevels; }
    int fineCells() const { return nf_; }
    const Level& level(int l) const { return lvls_[l - 1]; }
    std::size_t patchCount(int l) const {
        return l >= 1 ? lvls_[l - 1].patches.size() : 1;
    }

    bool covered(int l, int bi, int bj) const {
        const Level& L = lvls_[l - 1];
        return L.blockOf[std::size_t(bj) * L.nbx + bi] >= 0;
    }

    GridRef coarseRef() const {
        return const_cast<CutCell2DGpu&>(coarse_).ref(x0_, y0_);
    }
    GridRef patchRef(int l, const Patch& p) const {
        return GridRef{nf_,
                       nf_,
                       x0_ + p.ci0 * dxAt_(l - 1),
                       y0_ + p.cj0 * dyAt_(l - 1),
                       dxAt_(l),
                       dyAt_(l),
                       lvls_[l - 1].pool->poolData(p.slot)};
    }

    template <class IC>
    void init(IC ic) {
        GridRef b = coarseRef();
        for (int j = NG; j < NG + ny_; ++j)
            for (int i = NG; i < NG + nx_; ++i)
                b.at(i, j) = ic(b.xc(i), b.yc(j));
        buildBaseGeo_();
        fillPhysicalGhosts(b, 0);
        for (int pass = 1; pass < cfg_.maxLevels; ++pass) {
            regrid();
            for (Level& L : lvls_)
                for (Patch& p : L.patches) {
                    GridRef g = patchRefFor_(L, p);
                    for (int j = NG; j < NG + nf_; ++j)
                        for (int i = NG; i < NG + nf_; ++i)
                            g.at(i, j) = ic(g.xc(i), g.yc(j));
                }
            for (int l = cfg_.maxLevels - 1; l >= 1; --l) restrictLevel_(l);
        }
    }

    Real maxStableDtAll(Real cfl) const {
        const auto waveDt = [&](const GridRef& g) {
            Real sx = 0, sy = 0;
            for (int j = NG; j < NG + g.ny; ++j)
                for (int i = NG; i < NG + g.nx; ++i) {
                    const Prim w = toPrim(g.at(i, j));
                    const Real c = soundSpeed(w);
                    sx = std::max(sx, std::fabs(w.u) + c);
                    sy = std::max(sy, std::fabs(w.v) + c);
                }
            return cfl * std::min(g.dx / sx, g.dy / sy);
        };
        Real dt = waveDt(coarseRef());
        for (std::size_t k = 0; k < lvls_.size(); ++k)
            for (const Patch& p : lvls_[k].patches) {
                Real dtl = waveDt(patchRef(int(k) + 1, p));
                if (cfg_.subcycle) dtl *= Real(1 << (k + 1));
                dt = std::min(dt, dtl);
            }
        return dt;
    }

    void step(Real dt, double t) {
        if (baseGeo_.cell.empty()) buildBaseGeo_();
        GridRef b = coarseRef();
        fillPhysicalGhosts(b, t);
        advanceTree_(0, dt, t, 0, 1);
    }

    double totalMass() const {
        double m = 0;
        const GridRef b = coarseRef();
        const double Vb = double(b.dx) * b.dy;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                if (cfg_.maxLevels >= 2 &&
                    covered(1, i / cfg_.blockC, j / cfg_.blockC))
                    continue;
                m += double(baseGeo_.at(NG + i, NG + j).vol) * Vb *
                     double(b.at(NG + i, NG + j).rho);
            }
        for (int l = 1; l < cfg_.maxLevels; ++l) {
            const double Vl = Vb / double(1 << (2 * l));
            for (const Patch& p : lvls_[l - 1].patches) {
                const GridRef g = patchRef(l, p);
                for (int j = 0; j < nf_; ++j)
                    for (int i = 0; i < nf_; ++i) {
                        const int gi = 2 * p.ci0 + i, gj = 2 * p.cj0 + j;
                        if (l < cfg_.maxLevels - 1 &&
                            covered(l + 1, gi / cfg_.blockC, gj / cfg_.blockC))
                            continue;
                        m += double(p.geo.at(NG + i, NG + j).vol) * Vl *
                             double(g.at(NG + i, NG + j).rho);
                    }
            }
        }
        return m;
    }

    void regrid() { regridFrom_(1); }

private:
    Real dxAt_(int l) const { return dxc_ / Real(1 << l); }
    Real dyAt_(int l) const { return dyc_ / Real(1 << l); }
    int nxAt_(int l) const { return nx_ << l; }
    int nyAt_(int l) const { return ny_ << l; }
    std::size_t pidx_(int i, int j) const { return std::size_t(j) * ptx_ + i; }
    std::size_t cidx_(int i, int j) const { return std::size_t(j) * ctxs_ + i; }

    // patchRef given the level object (used when the level index is implicit).
    GridRef patchRefFor_(Level& L, const Patch& p) const {
        // find l such that &lvls_[l-1] == &L
        for (std::size_t k = 0; k < lvls_.size(); ++k)
            if (&lvls_[k] == &L) return patchRef(int(k) + 1, p);
        return GridRef{};
    }

    const Patch* ownerAt_(int l, int cg, int cgj) const { // l>=1, own-level cell
        const Level& L = lvls_[l - 1];
        const int pi = L.blockOf[std::size_t(cgj / nf_) * L.nbx + cg / nf_];
        return pi >= 0 ? &L.patches[pi] : nullptr;
    }
    Patch* ownerAt_(int l, int cg, int cgj) {
        return const_cast<Patch*>(
            const_cast<const AmrGpuMLCut*>(this)->ownerAt_(l, cg, cgj));
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

    // ---- geometry --------------------------------------------------------
    cutcell::Geometry buildGeo_(int nx, int ny, Real ox, Real oy, Real lx,
                                Real ly) const {
        Grid tmp(nx, ny, ox, oy, lx, ly); // geometry-only
        return cutcell::build(tmp, [&](double a, double b, double c0,
                                       double d) {
            return momentFn(a, b, c0, d);
        });
    }
    void buildBaseGeo_() {
        baseGeo_ = buildGeo_(nx_, ny_, x0_, y0_, nx_ * dxc_, ny_ * dyc_);
        coarse_.setGeometry(baseGeo_, x0_, y0_);
        if (cfg_.cutCellO2 && !o2Ready_) {
            coarse_.enableO2();
            for (Level& L : lvls_) L.pool->enableO2Pool();
            if (cfg_.mu > 0) {
                const Real kT = heatConductivity(cfg_.mu);
                coarse_.setViscosity(cfg_.mu, kT);
                for (Level& L : lvls_) L.pool->setViscosity(cfg_.mu, kT);
            }
            o2Ready_ = true;
        }
    }
    const cutcell::Geometry* parentGeo_(int lp, int cg, int cgj) const {
        if (lp == 0) return &baseGeo_;
        return &ownerAt_(lp, cg, cgj)->geo;
    }

    // ---- composite cut-cell advance of one level (no reflux) -------------
    void advanceCutLevel_(int l, Real dt) {
        if (l == 0) {
            coarse_.step(dt); // base: single-grid GPU cut operator
            return;
        }
        Level& L = lvls_[l - 1];
        if (L.patches.empty()) return;
        std::vector<int> active;
        active.reserve(L.patches.size());
        for (const Patch& p : L.patches) active.push_back(p.slot);

        L.pool->dcPhasePool(active);            // interior D^c + fluxes
        for (const Patch& p : L.patches) fillPatchDcGhosts_(l, p);
        L.pool->hybridPhasePool(active);        // interior D (gather, no scatter)
        L.pool->updatePhasePool(dt, active);    // floored update
    }

    // Global own-level cell that patch cell (i,j) maps to + owning sibling.
    const Patch* siblingCell_(int l, const Patch& p, int i, int j, int& si,
                              int& sj) const {
        const int nxl = nxAt_(l), nyl = nyAt_(l);
        int gfi = 2 * p.ci0 + (i - NG), gfj = 2 * p.cj0 + (j - NG);
        if (cfg_.periodicX) gfi = (gfi % nxl + nxl) % nxl;
        if (cfg_.periodicY) gfj = (gfj % nyl + nyl) % nyl;
        if (gfi < 0 || gfi >= nxl || gfj < 0 || gfj >= nyl) return nullptr;
        const Patch* s = ownerAt_(l, gfi, gfj);
        if (!s) return nullptr;
        si = NG + gfi - 2 * s->ci0;
        sj = NG + gfj - 2 * s->cj0;
        return s;
    }

    // Fill patch p's D^c ghosts from same-level siblings; zero the rest
    // (coarse-fine / domain) to match the CPU (cutCellDc: zero ghosts).
    void fillPatchDcGhosts_(int l, const Patch& p) {
        Level& L = lvls_[l - 1];
        Cons* dcp = L.pool->poolDc(p.slot);
        for (int j = 0; j < ptx_; ++j)
            for (int i = 0; i < ptx_; ++i) {
                if (i >= NG && i < NG + nf_ && j >= NG && j < NG + nf_)
                    continue;
                int si, sj;
                const Patch* s = siblingCell_(l, p, i, j, si, sj);
                dcp[pidx_(i, j)] =
                    s ? L.pool->poolDc(s->slot)[pidx_(si, sj)]
                      : Cons{0, 0, 0, 0};
            }
    }

    // ---- 2nd-order composite (SSP-RK2) -----------------------------------
    bool cutO2On_() const { return cfg_.cutCellO2; }
    std::size_t coarseN_() const {
        return std::size_t(ctxs_) * (ny_ + 2 * NG);
    }
    std::size_t poolN_() const { return std::size_t(ptx_) * ptx_; }
    static void avgInPlace_(Cons* a, const Cons* b, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) a[i] = Real(0.5) * (a[i] + b[i]);
    }

    // Fill patch p's gradient ghosts from same-level siblings (2nd-order at
    // seams); no sibling -> zeroed (constant reconstruction there).
    void fillPatchGradGhosts_(int l, const Patch& p) {
        Level& L = lvls_[l - 1];
        Cons* gx = L.pool->poolGdx(p.slot);
        Cons* gy = L.pool->poolGdy(p.slot);
        for (int j = 0; j < ptx_; ++j)
            for (int i = 0; i < ptx_; ++i) {
                if (i >= NG && i < NG + nf_ && j >= NG && j < NG + nf_)
                    continue;
                int si, sj;
                const Patch* s = siblingCell_(l, p, i, j, si, sj);
                if (s) {
                    gx[pidx_(i, j)] = L.pool->poolGdx(s->slot)[pidx_(si, sj)];
                    gy[pidx_(i, j)] = L.pool->poolGdy(s->slot)[pidx_(si, sj)];
                } else {
                    gx[pidx_(i, j)] = Cons{0, 0, 0, 0};
                    gy[pidx_(i, j)] = Cons{0, 0, 0, 0};
                }
            }
    }

    // One composite 2nd-order divergence of level l's patches (no update).
    void fineDivergenceO2_(int l, const std::vector<int>& active) {
        Level& L = lvls_[l - 1];
        L.pool->gradPhasePool(active);
        for (const Patch& p : L.patches) fillPatchGradGhosts_(l, p);
        L.pool->dcO2PhasePool(active);
        for (const Patch& p : L.patches) fillPatchDcGhosts_(l, p);
        L.pool->hybridPhasePool(active);
    }

    // One SSP-RK2 advance of level l over dt; stage-2 ghosts refilled at
    // (tS2, thetaS2). Leaves the time-averaged extensive fluxes in the flux
    // registers so the existing reflux stays conservative.
    void advanceCutLevelO2_(int l, Real dt, double tS2, Real thetaS2) {
        if (l == 0) {
            coarse_.divO2();
            std::vector<Cons> F1x(coarse_.fxData(),
                                  coarse_.fxData() + coarseN_());
            std::vector<Cons> F1y(coarse_.fyData(),
                                  coarse_.fyData() + coarseN_());
            coarse_.rk2Stage1(dt);
            GridRef b = coarseRef();
            fillPhysicalGhosts(b, tS2);
            coarse_.divO2();
            coarse_.rk2Stage2(dt);
            avgInPlace_(const_cast<Cons*>(coarse_.fxData()), F1x.data(),
                        coarseN_());
            avgInPlace_(const_cast<Cons*>(coarse_.fyData()), F1y.data(),
                        coarseN_());
            return;
        }
        Level& L = lvls_[l - 1];
        if (L.patches.empty()) return;
        std::vector<int> active;
        active.reserve(L.patches.size());
        for (const Patch& p : L.patches) active.push_back(p.slot);
        const std::size_t np = L.patches.size();

        fineDivergenceO2_(l, active); // stage 1
        std::vector<std::vector<Cons>> F1x(np), F1y(np);
        for (std::size_t k = 0; k < np; ++k) {
            const int s = L.patches[k].slot;
            F1x[k].assign(L.pool->poolFx(s), L.pool->poolFx(s) + poolN_());
            F1y[k].assign(L.pool->poolFy(s), L.pool->poolFy(s) + poolN_());
        }
        L.pool->rk2Stage1Pool(dt, active);

        fillLevelGhosts_(l, tS2, thetaS2);
        fineDivergenceO2_(l, active); // stage 2
        L.pool->rk2Stage2Pool(dt, active);
        for (std::size_t k = 0; k < np; ++k) {
            const int s = L.patches[k].slot;
            avgInPlace_(const_cast<Cons*>(L.pool->poolFx(s)), F1x[k].data(),
                        poolN_());
            avgInPlace_(const_cast<Cons*>(L.pool->poolFy(s)), F1y[k].data(),
                        poolN_());
        }
    }

    // ---- ghost fill / prolongation ---------------------------------------
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
            return blend_(b.at(NG + cg, NG + cgj),
                          baseOld_.empty() ? nullptr : baseOld_.data(), id,
                          theta);
        }
        const Patch* Q = (cg >= 0 && cg < nxp && cgj >= 0 && cgj < nyp)
                             ? ownerAt_(l - 1, cg, cgj)
                             : nullptr;
        if (Q == nullptr) Q = home;
        const GridRef g = patchRef(l - 1, *Q);
        const std::size_t id = g.idx(NG + cg - 2 * Q->ci0,
                                     NG + cgj - 2 * Q->cj0);
        return blend_(g.at(NG + cg - 2 * Q->ci0, NG + cgj - 2 * Q->cj0),
                      Q->old.empty() ? nullptr : Q->old.data(), id, theta);
    }

    Cons prolong_(int l, const Patch* home, int cg, int cgj, int ox, int oy,
                  Real theta) const {
        const Cons q0 = parentVal_(l, home, cg, cgj, theta);
        const Cons dqx = limitedSlope(parentVal_(l, home, cg - 1, cgj, theta),
                                      q0,
                                      parentVal_(l, home, cg + 1, cgj, theta));
        const Cons dqy = limitedSlope(parentVal_(l, home, cg, cgj - 1, theta),
                                      q0,
                                      parentVal_(l, home, cg, cgj + 1, theta));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    void fillLevelGhosts_(int l, double t, Real theta) {
        Level& lv = lvls_[l - 1];
        const int nxl = nxAt_(l), nyl = nyAt_(l);
        for (Patch& p : lv.patches) {
            const Patch* home =
                l >= 2 ? ownerAt_(l - 1, p.ci0, p.cj0) : nullptr;
            GridRef g = patchRef(l, p);
            const auto fillCell = [&](int i, int j) {
                int gfi = 2 * p.ci0 + (i - NG);
                int gfj = 2 * p.cj0 + (j - NG);
                if (cfg_.periodicX) gfi = (gfi % nxl + nxl) % nxl;
                if (cfg_.periodicY) gfj = (gfj % nyl + nyl) % nyl;
                if (gfi >= 0 && gfi < nxl && gfj >= 0 && gfj < nyl)
                    if (const Patch* s = ownerAt_(l, gfi, gfj)) {
                        const GridRef sg = patchRef(l, *s);
                        g.at(i, j) = sg.at(NG + gfi - 2 * s->ci0,
                                           NG + gfj - 2 * s->cj0);
                        return;
                    }
                const int cg = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cgj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                g.at(i, j) = prolong_(l, home, cg, cgj, gfi - 2 * cg,
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
        }
    }

    // ---- time advance ----------------------------------------------------
    void saveOld_(int l) {
        if (l == 0) {
            const GridRef b = coarseRef();
            baseOld_.assign(b.q, b.q + std::size_t(b.totx()) * b.toty());
            return;
        }
        Level& L = lvls_[l - 1];
        for (Patch& p : L.patches) {
            const Cons* q = L.pool->poolData(p.slot);
            p.old.assign(q, q + std::size_t(ptx_) * ptx_);
        }
    }

    void advanceTree_(int l, Real dt, double t, Real thBase, Real thSpan) {
        const bool hasKids =
            (l + 1 < cfg_.maxLevels) && !lvls_[l].patches.empty();
        if (hasKids) saveOld_(l);
        if (cutO2On_())
            advanceCutLevelO2_(l, dt, t + dt, thBase + thSpan);
        else
            advanceCutLevel_(l, dt);
        if (hasKids) {
            const int lc = l + 1;
            if (cfg_.reflux) refluxBackOut_(lc, dt);
            const int n = cfg_.subcycle ? 2 : 1;
            const Real cdt = dt / n;
            for (int k = 0; k < n; ++k) {
                fillLevelGhosts_(lc, t + k * cdt, Real(k) / Real(n));
                advanceTree_(lc, cdt, t + k * cdt, Real(k) / Real(n),
                             Real(1) / Real(n));
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

    // ---- coarse-fine coupling --------------------------------------------
    struct ParentCell {
        GridRef g;
        const Cons *Fx, *Fy;
        int li, lj;
        const cutcell::Geometry* geo;
    };
    ParentCell parentCell_(int lp, int cg, int cgj) {
        if (lp == 0)
            return ParentCell{coarseRef(), coarse_.fxData(), coarse_.fyData(),
                              NG + cg, NG + cgj, &baseGeo_};
        Patch* Q = ownerAt_(lp, cg, cgj);
        assert(Q != nullptr);
        Level& L = lvls_[lp - 1];
        return ParentCell{patchRef(lp, *Q), L.pool->poolFx(Q->slot),
                          L.pool->poolFy(Q->slot), NG + cg - 2 * Q->ci0,
                          NG + cgj - 2 * Q->cj0, &Q->geo};
    }

    // Cut-aware reflux correction: pass the extensive increment through the
    // SAME hybrid flux redistribution as the advance (kappa D^c + (1-kappa)
    // D^nc, shedding the rest over the 3x3 fluid neighbourhood).
    void cutRefluxCorr_(ParentCell& pc, const cutcell::Geometry& geo,
                        const Cons& dflux, Real dt) {
        GridRef g = pc.g;
        const int i = pc.li, j = pc.lj;
        const double V = double(g.dx) * double(g.dy);
        const double kc = double(geo.at(i, j).vol);
        if (kc <= 1e-9) return;
        const Cons Dc = Real(1.0 / (kc * V)) * dflux;
        if (kc >= 1.0 - 1e-9) {
            g.at(i, j) = floorState(g.at(i, j) + dt * Dc);
            return;
        }
        double S = 0;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                const double kk = double(geo.at(i + di, j + dj).vol);
                if (kk > 1e-9) S += kk;
            }
        const Cons Dnc = Real(kc / S) * Dc;
        const Cons dD = Real(kc * (1 - kc) / S) * (Dc - Dnc);
        g.at(i, j) = floorState(g.at(i, j) +
                                dt * (Real(kc) * Dc + Real(1 - kc) * Dnc));
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
                if (double(geo.at(i + di, j + dj).vol) > 1e-9)
                    g.at(i + di, j + dj) =
                        floorState(g.at(i + di, j + dj) + dt * dD);
    }

    struct SideNb {
        bool ok;
        int blockBi, blockBj;
        int cg, cgj;
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
                    const cutcell::Geometry& geo = *pc.geo;
                    Cons d;
                    if (dir == 0)
                        d = pc.Fx[parentIdx_(pc, pc.li, pc.lj)];
                    else if (dir == 1)
                        d = Real(-1) * pc.Fx[parentIdx_(pc, pc.li - 1, pc.lj)];
                    else if (dir == 2)
                        d = pc.Fy[parentIdx_(pc, pc.li, pc.lj)];
                    else
                        d = Real(-1) * pc.Fy[parentIdx_(pc, pc.li, pc.lj - 1)];
                    cutRefluxCorr_(pc, geo, d, dtParent);
                }
            }
    }

    void refluxFineApply_(int lc, Real dtChild) {
        const int bC = cfg_.blockC;
        Level& L = lvls_[lc - 1];
        for (const Patch& p : L.patches)
            for (int dir = 0; dir < 4; ++dir) {
                const SideNb n = sideNb_(lc, p, dir);
                if (!n.ok || covered(lc, n.blockBi, n.blockBj)) continue;
                const Cons* Fx = L.pool->poolFx(p.slot);
                const Cons* Fy = L.pool->poolFy(p.slot);
                for (int r = 0; r < bC; ++r) {
                    const int cg = (dir < 2) ? n.cg : n.cg + r;
                    const int cgj = (dir < 2) ? n.cgj + r : n.cgj;
                    ParentCell pc = parentCell_(lc - 1, cg, cgj);
                    const cutcell::Geometry& geo = *pc.geo;
                    std::size_t a, b;
                    if (dir == 0) {
                        a = pidx_(NG - 1, NG + 2 * r);
                        b = pidx_(NG - 1, NG + 2 * r + 1);
                    } else if (dir == 1) {
                        a = pidx_(NG + nf_ - 1, NG + 2 * r);
                        b = pidx_(NG + nf_ - 1, NG + 2 * r + 1);
                    } else if (dir == 2) {
                        a = pidx_(NG + 2 * r, NG - 1);
                        b = pidx_(NG + 2 * r + 1, NG - 1);
                    } else {
                        a = pidx_(NG + 2 * r, NG + nf_ - 1);
                        b = pidx_(NG + 2 * r + 1, NG + nf_ - 1);
                    }
                    const Cons sf = (dir < 2) ? Fx[a] + Fx[b] : Fy[a] + Fy[b];
                    cutRefluxCorr_(pc, geo,
                                   (dir == 0 || dir == 2) ? Real(-1) * sf : sf,
                                   dtChild);
                }
            }
    }

    std::size_t parentIdx_(const ParentCell& pc, int i, int j) const {
        return std::size_t(j) * pc.g.totx() + i;
    }

    void restrictLevel_(int l) { // level l (>=1) onto l-1
        const int bC = cfg_.blockC;
        Level& L = lvls_[l - 1];
        for (const Patch& p : L.patches) {
            const GridRef g = patchRef(l, p);
            for (int b = 0; b < bC; ++b)
                for (int a = 0; a < bC; ++a) {
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    ParentCell pc = parentCell_(l - 1, p.ci0 + a, p.cj0 + b);
                    const cutcell::Geometry& geo = *pc.geo;
                    if (double(geo.at(pc.li, pc.lj).vol) <= 1e-9) continue;
                    double nr = 0, nmx = 0, nmy = 0, nE = 0, den = 0;
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const double kk =
                                double(p.geo.at(fi + di, fj + dj).vol);
                            if (kk <= 1e-9) continue;
                            const Cons& q = g.at(fi + di, fj + dj);
                            nr += kk * double(q.rho);
                            nmx += kk * double(q.mx);
                            nmy += kk * double(q.my);
                            nE += kk * double(q.E);
                            den += kk;
                        }
                    if (den > 1e-12)
                        pc.g.at(pc.li, pc.lj) =
                            Cons{Real(nr / den), Real(nmx / den),
                                 Real(nmy / den), Real(nE / den)};
                }
        }
    }

    // ---- tagging ---------------------------------------------------------
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
    bool cutBand_(const cutcell::Geometry& geo, int i, int j) const {
        if (geo.cell.empty()) return false;
        const Real k = geo.at(NG + i, NG + j).vol;
        return k > Real(1e-6) && k < Real(1) - Real(1e-6);
    }
    void markDilated_(std::vector<std::uint8_t>& want, const Level& lv, int nx,
                      int ny, int ci, int cj) const {
        for (int b = cj - 2; b <= cj + 2; ++b)
            for (int a = ci - 2; a <= ci + 2; ++a) {
                int x = a, y = b;
                if (cfg_.periodicX) x = (x % nx + nx) % nx;
                if (cfg_.periodicY) y = (y % ny + ny) % ny;
                if (x < 0 || x >= nx || y < 0 || y >= ny) continue;
                want[std::size_t(y / cfg_.blockC) * lv.nbx + x / cfg_.blockC] =
                    1;
            }
    }
    void tagBase_(std::vector<std::uint8_t>& want) const {
        const Level& lv = lvls_[0];
        const GridRef b = coarseRef();
        const auto at = [&](int a, int bb) -> const Cons& {
            return b.at(NG + a, NG + bb);
        };
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const int ip = std::min(i + 1, nx_ - 1),
                          im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, ny_ - 1),
                          jm = std::max(j - 1, 0);
                if (tagCell_(at, i, j, ip, im, jp, jm) ||
                    cutBand_(baseGeo_, i, j))
                    markDilated_(want, lv, nx_, ny_, i, j);
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
                    if (tagCell_(at, i, j, i + 1, i - 1, j + 1, j - 1) ||
                        cutBand_(p.geo, i, j))
                        markDilated_(want, kid, nx, ny, 2 * p.ci0 + i,
                                     2 * p.cj0 + j);
        }
    }

    // ---- nesting helpers -------------------------------------------------
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
                        parWant[std::size_t(cy / nf_) * par.nbx + cx / nf_] = 1;
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
                    for (int x = bi * bC - 1; w && x <= (bi + 1) * bC; ++x) {
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
    bool checkNesting() const {
        for (std::size_t k = 1; k < lvls_.size(); ++k)
            for (const Patch& p : lvls_[k].patches)
                for (int dj = -1; dj <= cfg_.blockC; ++dj)
                    for (int di = -1; di <= cfg_.blockC; ++di) {
                        int cg = p.ci0 + di, cgj = p.cj0 + dj;
                        const int nxp = nx_ << k, nyp = ny_ << k;
                        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
                        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
                        if (cg < 0 || cg >= nxp || cgj < 0 || cgj >= nyp)
                            continue;
                        if (!covered(int(k), cg / nf_, cgj / nf_)) return false;
                    }
        return true;
    }

    Patch makePatch_(int l, int bi, int bj) {
        Level& L = lvls_[l - 1];
        assert(!L.freeSlots.empty());
        const int slot = L.freeSlots.back();
        L.freeSlots.pop_back();
        const int bC = cfg_.blockC;
        const int ci0 = bi * bC, cj0 = bj * bC;
        Patch p{bi, bj, ci0, cj0, slot, {}, {}};
        const Patch* home = l >= 2 ? ownerAt_(l - 1, ci0, cj0) : nullptr;
        const GridRef g = patchRef(l, p);
        for (int j = NG; j < NG + nf_; ++j)
            for (int i = NG; i < NG + nf_; ++i) {
                const int gfi = 2 * ci0 + (i - NG);
                const int gfj = 2 * cj0 + (j - NG);
                g.at(i, j) = prolong_(l, home, gfi / 2, gfj / 2, gfi & 1,
                                      gfj & 1, Real(-1));
            }
        p.geo = buildGeo_(nf_, nf_, x0_ + ci0 * dxAt_(l - 1),
                          y0_ + cj0 * dyAt_(l - 1), nf_ * dxAt_(l),
                          nf_ * dyAt_(l));
        L.pool->setPoolGeometry(slot, p.geo, x0_ + ci0 * dxAt_(l - 1),
                                y0_ + cj0 * dyAt_(l - 1));
        return p;
    }

    void regridFrom_(int lstart) {
        const int L = cfg_.maxLevels;
        if (lstart >= L) return;
        if (baseGeo_.cell.empty()) buildBaseGeo_();

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
            for (const Patch& p : lv.patches)
                if (!want[l - 1][std::size_t(p.bj) * lv.nbx + p.bi])
                    lv.freeSlots.push_back(p.slot);
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
        }
        assert(checkNesting());
    }

    MetalContext& ctx_;
    AmrConfig cfg_;
    Real x0_, y0_;
    int nx_, ny_;
    Real dxc_, dyc_;
    CutCell2DGpu coarse_;
    int nf_, ptx_ = 0, ctxs_ = 0;
    std::vector<Level> lvls_;
    cutcell::Geometry baseGeo_;
    std::vector<Cons> baseOld_;
    std::vector<int> stepCounts_ = std::vector<int>(16, 0);
    bool o2Ready_ = false; // O2 pipelines/buffers enabled
};

} // namespace mm

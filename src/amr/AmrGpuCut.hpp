#pragma once

// Hybrid CPU/GPU two-level AMR with embedded-boundary CUT CELLS. The GPU
// analogue of Amr2's cut-cell path: every grid lives in shared Metal buffers
// (a CutCell2DGpu for the coarse level, a pooled CutCell2DGpu for the patches),
// and the CPU orchestrates the cross-patch coupling / reflux / restriction /
// tagging in place (unified memory, zero copies).
//
// Per step: fill ghosts, advance the coarse cut-cell grid (monolithic GPU
// step, extensive fluxes kept), advance the patches as a COMPOSITE, then
// reflux each coarse-fine interface, restrict fine onto covered coarse cells,
// regrid every cfg.regridEvery steps (EB-band tagging).
//
// The composite fine advance uses the phase-split pool: the GPU computes each
// patch's conservative divergence D^c (interior), the CPU fills each patch's
// D^c ghosts from same-level siblings, and the GPU forms the hybrid divergence
// + flux redistribution. Because the FRD kernel is written in GATHER form (a
// cell pulls the excess its cut-cell neighbours shed to it), the cross-patch
// coupling is captured by the D^c ghost exchange alone — no D scatter is
// needed, unlike the CPU scatter path (Amr2::scatterPatchDGhosts_). The two
// are mathematically identical: a patch's interior D equals the divergence the
// monolithic single-grid operator would produce over the tiled region.

#include "amr/Amr2.hpp" // AmrConfig, cut-cell helpers, limitedSlope, floorState
#include "backend/metal/CutCell2DGpu.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mm {

class AmrGpuCut {
public:
    struct Patch {
        int bi, bj;    // block coordinates
        int ci0, cj0;  // coarse-cell origin of the block
        int slot;      // index into the pooled buffers
        cutcell::Geometry geo; // cut-cell moments (CPU mirror)
    };

    std::vector<Patch> patches;

    std::function<void(GridRef&, double)> fillPhysicalGhosts;
    std::function<void(GridRef&, double, unsigned)> fillPatchPhysical;

    // Analytic EB moments for the box [x0,x1] x [y0,y1] (fluid fraction, face
    // apertures, EB face). Required (with cfg.cutCell) for the cut-cell path.
    std::function<CellMoments(double, double, double, double)> momentFn;

    AmrGpuCut(MetalContext& ctx, int nx, int ny, Real x0, Real y0, Real lx,
              Real ly, AmrConfig cfg)
        : ctx_(ctx), cfg_(cfg), x0_(x0), y0_(y0), nx_(nx), ny_(ny),
          dxc_(lx / nx), dyc_(ly / ny),
          coarse_(ctx, nx, ny, lx / nx, ly / ny),
          pool_(ctx, 2 * cfg.blockC, 2 * cfg.blockC, (lx / nx) / 2,
                (ly / ny) / 2),
          nbx_(nx / cfg.blockC), nby_(ny / cfg.blockC),
          blockOf_(std::size_t(nbx_) * nby_, -1) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
        nf_ = 2 * cfg.blockC;
        ptx_ = nf_ + 2 * NG;
        ctx_stride_ = nx_ + 2 * NG;
        capacity_ = nbx_ * nby_;
        pool_.enablePool(capacity_);
        for (int s = capacity_ - 1; s >= 0; --s) freeSlots_.push_back(s);
    }

    AmrGpuCut(const AmrGpuCut&) = delete;
    AmrGpuCut& operator=(const AmrGpuCut&) = delete;

    bool covered(int bi, int bj) const {
        return blockOf_[std::size_t(bj) * nbx_ + bi] >= 0;
    }

    GridRef coarseRef() const {
        return const_cast<CutCell2DGpu&>(coarse_).ref(x0_, y0_);
    }
    GridRef patchRef(const Patch& p) const {
        return GridRef{nf_,
                       nf_,
                       x0_ + p.ci0 * dxc_,
                       y0_ + p.cj0 * dyc_,
                       dxc_ / 2,
                       dyc_ / 2,
                       const_cast<CutCell2DGpu&>(pool_).poolData(p.slot)};
    }
    int fineCells() const { return nf_; }

    unsigned domainSides(const Patch& p) const {
        unsigned s = 0;
        if (!cfg_.periodicX)
            s |= (p.bi == 0 ? SideLeft : 0u) |
                 (p.bi == nbx_ - 1 ? SideRight : 0u);
        if (!cfg_.periodicY)
            s |= (p.bj == 0 ? SideBottom : 0u) |
                 (p.bj == nby_ - 1 ? SideTop : 0u);
        return s;
    }

    template <class IC>
    void init(IC ic) {
        GridRef c = coarseRef();
        for (int j = NG; j < NG + c.ny; ++j)
            for (int i = NG; i < NG + c.nx; ++i)
                c.at(i, j) = ic(c.xc(i), c.yc(j));
        buildCoarseGeo_();
        fillPhysicalGhosts(c, 0); // regrid prolongation reads coarse ghosts
        regrid();
        for (const Patch& p : patches) {
            GridRef g = patchRef(p);
            for (int j = NG; j < NG + g.ny; ++j)
                for (int i = NG; i < NG + g.nx; ++i)
                    g.at(i, j) = ic(g.xc(i), g.yc(j));
        }
        restrictFine_();
    }

    // CFL dt over the composite (coarse + fine patches), computed on the CPU
    // from the shared buffers. Fine cells are half-size; subcycling lets them
    // take two dt/2 substeps, so their bound is doubled.
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
        for (const Patch& p : patches) {
            Real dtF = waveDt(patchRef(p));
            if (cfg_.subcycle) dtF *= 2;
            dt = std::min(dt, dtF);
        }
        return dt;
    }

    void step(Real dt, double t) {
        if (cfg_.subcycle && !patches.empty())
            stepCutSubcycled_(dt, t);
        else
            stepCutSingleRate_(dt, t);
        restrictFine_();

        if (++stepCount_ % cfg_.regridEvery == 0) {
            GridRef c = coarseRef();
            fillPhysicalGhosts(c, t + dt);
            regrid();
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
        std::size_t n = std::size_t(nx_) * ny_;
        n += patches.size() * std::size_t(nf_) * nf_;
        return n;
    }

    int blockCount() const { return nbx_ * nby_; }
    int stepCount() const { return stepCount_; }

    // Rebuild the patch set at the given blocks (pinned patches / restart).
    // Patch data is prolongated then expected to be overwritten by the caller.
    void restoreBlocks(const std::vector<std::pair<int, int>>& blocks,
                       int stepCount) {
        if (coarseGeo_.cell.empty()) buildCoarseGeo_();
        for (const Patch& p : patches) freeSlots_.push_back(p.slot);
        patches.clear();
        std::fill(blockOf_.begin(), blockOf_.end(), -1);
        for (const auto& [bi, bj] : blocks) {
            patches.push_back(makePatch_(bi, bj));
            blockOf_[std::size_t(bj) * nbx_ + bi] = int(patches.size()) - 1;
        }
        stepCount_ = stepCount;
    }

    void regrid() {
        const int nx = nx_, ny = ny_, bc = cfg_.blockC;
        const GridRef c = coarseRef();
        if (coarseGeo_.cell.empty()) buildCoarseGeo_();

        // 1. Tag on relative density gradient, velocity jump, and the EB band.
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
                bool t = std::max(ex, ey) / r0 > cfg_.tagThreshold;
                if (!t && cfg_.tagVelocity > 0) {
                    const auto uOf = [&](int a, int b) {
                        const Cons& q = c.at(NG + a, NG + b);
                        return q.mx / std::max(q.rho, RHO_FLOOR);
                    };
                    const auto vOf = [&](int a, int b) {
                        const Cons& q = c.at(NG + a, NG + b);
                        return q.my / std::max(q.rho, RHO_FLOOR);
                    };
                    const Real du = std::max(
                        std::fabs(uOf(ip, j) - uOf(im, j)),
                        std::fabs(uOf(i, jp) - uOf(i, jm)));
                    const Real dv = std::max(
                        std::fabs(vOf(ip, j) - vOf(im, j)),
                        std::fabs(vOf(i, jp) - vOf(i, jm)));
                    const Real c0 =
                        soundSpeed(toPrim(c.at(NG + i, NG + j)));
                    t = std::max(du, dv) / c0 > cfg_.tagVelocity;
                }
                // EB band: tag cells the solid cuts (0 < kappa < 1); the
                // dilation below extends it into the surrounding fluid.
                if (!t) {
                    const Real k = coarseGeo_.at(NG + i, NG + j).vol;
                    t = k > Real(1e-6) && k < Real(1) - Real(1e-6);
                }
                tag[std::size_t(j) * nx + i] = t;
            }

        // 2. Dilate by 2 cells.
        std::vector<std::uint8_t> dil(tag.size(), 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (!tag[std::size_t(j) * nx + i]) continue;
                for (int b = std::max(j - 2, 0);
                     b <= std::min(j + 2, ny - 1); ++b)
                    for (int a = std::max(i - 2, 0);
                         a <= std::min(i + 2, nx - 1); ++a)
                        dil[std::size_t(b) * nx + a] = 1;
            }

        // 3. A block is refined if it holds any dilated tag.
        std::vector<std::uint8_t> want(std::size_t(nbx_) * nby_, 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (dil[std::size_t(j) * nx + i])
                    want[std::size_t(j / bc) * nbx_ + i / bc] = 1;

        // 4. Rebuild the patch list, freeing removed slots first.
        std::vector<Patch> next;
        std::vector<int> nextOf(std::size_t(nbx_) * nby_, -1);
        for (const Patch& p : patches)
            if (!want[std::size_t(p.bj) * nbx_ + p.bi])
                freeSlots_.push_back(p.slot);
        for (int bj = 0; bj < nby_; ++bj)
            for (int bi = 0; bi < nbx_; ++bi) {
                if (!want[std::size_t(bj) * nbx_ + bi]) continue;
                const int old = blockOf_[std::size_t(bj) * nbx_ + bi];
                if (old >= 0)
                    next.push_back(std::move(patches[old]));
                else
                    next.push_back(makePatch_(bi, bj));
                nextOf[std::size_t(bj) * nbx_ + bi] = int(next.size()) - 1;
            }
        patches = std::move(next);
        blockOf_ = std::move(nextOf);
    }

private:
    bool cutCellOn_() const { return cfg_.cutCell && bool(momentFn); }

    cutcell::Geometry buildGeo_(int nx, int ny, Real ox, Real oy, Real lx,
                                Real ly) const {
        Grid tmp(nx, ny, ox, oy, lx, ly); // geometry-only (q unused by build)
        return cutcell::build(tmp, [&](double a, double b, double c0,
                                       double d) {
            return momentFn(a, b, c0, d);
        });
    }

    void buildCoarseGeo_() {
        coarseGeo_ = buildGeo_(nx_, ny_, x0_, y0_, nx_ * dxc_, ny_ * dyc_);
        coarse_.setGeometry(coarseGeo_);
    }

    // ---- single-rate cut-cell step ---------------------------------------
    void stepCutSingleRate_(Real dt, double t) {
        GridRef c = coarseRef();
        fillPhysicalGhosts(c, t);
        fillAllPatchGhosts_(t, Real(-1));

        coarse_.step(dt); // base grid: single-grid GPU cut-cell operator
        advanceCutFinePatchesGpu_(dt);
        if (cfg_.reflux)
            for (const Patch& p : patches) {
                cutRefluxBackout_(p, dt);
                cutRefluxFineApply_(p, dt, pool_.poolFx(p.slot),
                                    pool_.poolFy(p.slot));
            }
    }

    // Berger-Colella subcycling: base advances dtC, each patch takes two
    // dtF = dtC/2 substeps with time-interpolated coarse ghosts. Reflux
    // backs the (single) coarse flux out once, applies each substep's fine
    // flux (2 x dtF = dtC).
    void stepCutSubcycled_(Real dtC, double t) {
        const Real dtF = dtC / 2;
        GridRef c = coarseRef();
        coarseOld_.assign(c.q, c.q + std::size_t(c.totx()) * c.toty());

        fillPhysicalGhosts(c, t);
        fillAllPatchGhosts_(t, Real(-1)); // substep-1 ghosts at t^n

        coarse_.step(dtC);
        if (cfg_.reflux)
            for (const Patch& p : patches) cutRefluxBackout_(p, dtC);

        advanceCutFinePatchesGpu_(dtF); // substep 1
        if (cfg_.reflux)
            for (const Patch& p : patches)
                cutRefluxFineApply_(p, dtF, pool_.poolFx(p.slot),
                                    pool_.poolFy(p.slot));

        // substep 2: siblings are current, coarse prolongation blends t^n/t^n+1
        fillAllPatchGhosts_(t + dtF, Real(0.5));
        advanceCutFinePatchesGpu_(dtF); // substep 2
        if (cfg_.reflux)
            for (const Patch& p : patches)
                cutRefluxFineApply_(p, dtF, pool_.poolFx(p.slot),
                                    pool_.poolFy(p.slot));
    }

    // Composite advance of the fine patches (no reflux): the GPU computes each
    // patch's D^c (interior) + extensive fluxes, the CPU fills each patch's D^c
    // ghosts from same-level siblings (gather-form FRD needs no D scatter), and
    // the GPU forms the hybrid divergence + FRD and updates.
    void advanceCutFinePatchesGpu_(Real dt) {
        if (patches.empty()) return;
        std::vector<int> active;
        active.reserve(patches.size());
        for (const Patch& p : patches) active.push_back(p.slot);

        pool_.dcPhasePool(active);
        for (std::size_t k = 0; k < patches.size(); ++k)
            fillPatchDcGhosts_(k);
        pool_.hybridPhasePool(active);
        pool_.updatePhasePool(dt, active);
    }

    // Global fine cell (gfi,gfj) that patch cell (i,j) maps to, periodic wrap.
    // Returns the owning patch index (-1 if none / coarse-fine / domain) and
    // the corresponding sibling cell (si,sj).
    int siblingCell_(const Patch& p, int i, int j, int& si, int& sj) const {
        const int nxf = 2 * nx_, nyf = 2 * ny_;
        int gfi = 2 * p.ci0 + (i - NG), gfj = 2 * p.cj0 + (j - NG);
        if (cfg_.periodicX) gfi = (gfi % nxf + nxf) % nxf;
        if (cfg_.periodicY) gfj = (gfj % nyf + nyf) % nyf;
        if (gfi < 0 || gfi >= nxf || gfj < 0 || gfj >= nyf) return -1;
        const int bi = gfi / (2 * cfg_.blockC), bj = gfj / (2 * cfg_.blockC);
        const int pi = blockOf_[std::size_t(bj) * nbx_ + bi];
        if (pi < 0) return -1;
        const Patch& s = patches[pi];
        si = NG + gfi - 2 * s.ci0;
        sj = NG + gfj - 2 * s.cj0;
        return pi;
    }

    std::size_t pidx_(int i, int j) const {
        return std::size_t(j) * ptx_ + i;
    }

    // Fill patch k's D^c ghosts from the sibling that owns each ghost cell
    // (same-level copy). Ghosts with no sibling (coarse-fine / domain) are
    // zeroed, matching the CPU (cutCellDc returns a zero-ghost divergence).
    void fillPatchDcGhosts_(std::size_t k) {
        const Patch& p = patches[k];
        Cons* dck = pool_.poolDc(p.slot);
        for (int j = 0; j < ptx_; ++j)
            for (int i = 0; i < ptx_; ++i) {
                if (i >= NG && i < NG + nf_ && j >= NG && j < NG + nf_)
                    continue;
                int si, sj;
                const int pi = siblingCell_(p, i, j, si, sj);
                dck[pidx_(i, j)] = pi >= 0
                    ? pool_.poolDc(patches[pi].slot)[pidx_(si, sj)]
                    : Cons{0, 0, 0, 0};
            }
    }

    void fillAllPatchGhosts_(double t, Real theta) {
        for (const Patch& p : patches) {
            fillPatchGhosts_(p, theta);
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides(p)) {
                    GridRef g = patchRef(p);
                    fillPatchPhysical(g, t, sides);
                }
        }
    }

    // Coarse value at (I,J): current state, or the theta-blend of the saved
    // t^n copy and the current state (subcycled ghosts).
    Cons coarseAt_(int I, int J, Real theta) const {
        const GridRef c = coarseRef();
        const Cons& cur = c.at(I, J);
        if (theta < 0) return cur;
        const Cons& old = coarseOld_[c.idx(I, J)];
        return (1 - theta) * old + theta * cur;
    }

    Cons prolong_(int ci, int cj, int ox, int oy,
                  Real theta = Real(-1)) const {
        const int I = NG + ci, J = NG + cj;
        const Cons q0 = coarseAt_(I, J, theta);
        const Cons dqx = limitedSlope(coarseAt_(I - 1, J, theta), q0,
                                      coarseAt_(I + 1, J, theta));
        const Cons dqy = limitedSlope(coarseAt_(I, J - 1, theta), q0,
                                      coarseAt_(I, J + 1, theta));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    void fillPatchGhosts_(const Patch& p, Real theta = Real(-1)) const {
        GridRef g = patchRef(p);
        const int nxf = 2 * nx_, nyf = 2 * ny_;
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i) {
                if (i >= NG && i < NG + nf_ && j >= NG && j < NG + nf_)
                    continue; // interior
                int gfi = 2 * p.ci0 + (i - NG);
                int gfj = 2 * p.cj0 + (j - NG);
                if (cfg_.periodicX) gfi = (gfi % nxf + nxf) % nxf;
                if (cfg_.periodicY) gfj = (gfj % nyf + nyf) % nyf;

                // Same-level copy when a sibling patch owns the cell.
                if (gfi >= 0 && gfi < nxf && gfj >= 0 && gfj < nyf) {
                    const int bi = gfi / (2 * cfg_.blockC);
                    const int bj = gfj / (2 * cfg_.blockC);
                    const int pi = blockOf_[std::size_t(bj) * nbx_ + bi];
                    if (pi >= 0) {
                        const GridRef s = patchRef(patches[pi]);
                        g.at(i, j) = s.at(NG + gfi - 2 * patches[pi].ci0,
                                          NG + gfj - 2 * patches[pi].cj0);
                        continue;
                    }
                }
                // Otherwise limited prolongation from the coarse level.
                const int ci = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                g.at(i, j) =
                    prolong_(ci, cj, gfi - 2 * ci, gfj - 2 * cj, theta);
            }
    }

    // ---- cut-aware reflux (mirror Amr2) ----------------------------------
    struct SideNb {
        bool ok;
        int block, cell;
    };
    SideNb leftNb_(const Patch& p) const {
        if (p.bi > 0) return {true, p.bi - 1, p.ci0 - 1};
        return cfg_.periodicX ? SideNb{true, nbx_ - 1, nx_ - 1}
                              : SideNb{false, 0, 0};
    }
    SideNb rightNb_(const Patch& p) const {
        const int bc = cfg_.blockC;
        if (p.bi < nbx_ - 1) return {true, p.bi + 1, p.ci0 + bc};
        return cfg_.periodicX ? SideNb{true, 0, 0} : SideNb{false, 0, 0};
    }
    SideNb bottomNb_(const Patch& p) const {
        if (p.bj > 0) return {true, p.bj - 1, p.cj0 - 1};
        return cfg_.periodicY ? SideNb{true, nby_ - 1, ny_ - 1}
                              : SideNb{false, 0, 0};
    }
    SideNb topNb_(const Patch& p) const {
        const int bc = cfg_.blockC;
        if (p.bj < nby_ - 1) return {true, p.bj + 1, p.cj0 + bc};
        return cfg_.periodicY ? SideNb{true, 0, 0} : SideNb{false, 0, 0};
    }

    std::size_t cidx_(int i, int j) const {
        return std::size_t(j) * ctx_stride_ + i;
    }

    // Apply dt/(kappa V_c)*flux (extensive) to the uncovered coarse cell just
    // outside the patch, through the positivity floor; reduces to the uniform
    // reflux when kappa = apertures = 1.
    Cons cutRefluxCorr_(int ci, int cj, const Cons& d, Real dt) {
        GridRef c = coarseRef();
        const double k = double(coarseGeo_.at(NG + ci, NG + cj).vol);
        if (k > 1e-9)
            return floorState(c.at(NG + ci, NG + cj) +
                              Real(dt / (k * dxc_ * dyc_)) * d);
        return c.at(NG + ci, NG + cj);
    }

    // Back the COARSE interface flux out of the uncovered neighbour (once).
    void cutRefluxBackout_(const Patch& p, Real dtC) {
        GridRef c = coarseRef();
        const int bc = cfg_.blockC;
        const Cons* fxC = coarse_.fxData();
        const Cons* fyC = coarse_.fyData();
        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r)
                c.at(NG + n.cell, NG + p.cj0 + r) = cutRefluxCorr_(
                    n.cell, p.cj0 + r,
                    fxC[cidx_(NG + n.cell, NG + p.cj0 + r)], dtC);
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r)
                c.at(NG + n.cell, NG + p.cj0 + r) = cutRefluxCorr_(
                    n.cell, p.cj0 + r,
                    Real(-1) * fxC[cidx_(NG + n.cell - 1, NG + p.cj0 + r)],
                    dtC);
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 + r, NG + n.cell) = cutRefluxCorr_(
                    p.ci0 + r, n.cell,
                    fyC[cidx_(NG + p.ci0 + r, NG + n.cell)], dtC);
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r)
                c.at(NG + p.ci0 + r, NG + n.cell) = cutRefluxCorr_(
                    p.ci0 + r, n.cell,
                    Real(-1) * fyC[cidx_(NG + p.ci0 + r, NG + n.cell - 1)],
                    dtC);
    }

    // Apply this substep's FINE flux (sum of the two fine faces) to the
    // uncovered neighbour (once per substep). Uses the patch's pooled
    // extensive fluxes.
    void cutRefluxFineApply_(const Patch& p, Real dtF, const Cons* Fx,
                             const Cons* Fy) {
        const int bc = cfg_.blockC, nf = nf_;
        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fx[pidx_(NG - 1, NG + 2 * r)] +
                                Fx[pidx_(NG - 1, NG + 2 * r + 1)];
                GridRef c = coarseRef();
                c.at(NG + n.cell, NG + p.cj0 + r) =
                    cutRefluxCorr_(n.cell, p.cj0 + r, Real(-1) * sf, dtF);
            }
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fx[pidx_(NG + nf - 1, NG + 2 * r)] +
                                Fx[pidx_(NG + nf - 1, NG + 2 * r + 1)];
                GridRef c = coarseRef();
                c.at(NG + n.cell, NG + p.cj0 + r) =
                    cutRefluxCorr_(n.cell, p.cj0 + r, sf, dtF);
            }
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fy[pidx_(NG + 2 * r, NG - 1)] +
                                Fy[pidx_(NG + 2 * r + 1, NG - 1)];
                GridRef c = coarseRef();
                c.at(NG + p.ci0 + r, NG + n.cell) =
                    cutRefluxCorr_(p.ci0 + r, n.cell, Real(-1) * sf, dtF);
            }
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fy[pidx_(NG + 2 * r, NG + nf - 1)] +
                                Fy[pidx_(NG + 2 * r + 1, NG + nf - 1)];
                GridRef c = coarseRef();
                c.at(NG + p.ci0 + r, NG + n.cell) =
                    cutRefluxCorr_(p.ci0 + r, n.cell, sf, dtF);
            }
    }

    // kappa-weighted restriction of each 2x2 fine block onto its covered
    // coarse cell (conservative: exact moments make coarse fluid volume equal
    // the sum of the four children's fluid volumes).
    void restrictFine_() {
        GridRef c = coarseRef();
        for (const Patch& p : patches) {
            const GridRef g = patchRef(p);
            for (int b = 0; b < cfg_.blockC; ++b)
                for (int a = 0; a < cfg_.blockC; ++a) {
                    const int ci = NG + p.ci0 + a, cj = NG + p.cj0 + b;
                    double nr = 0, nmx = 0, nmy = 0, nE = 0, den = 0;
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const int fi = NG + 2 * a + di, fj = NG + 2 * b + dj;
                            const double k = double(p.geo.at(fi, fj).vol);
                            if (k <= 1e-9) continue;
                            const Cons& q = g.at(fi, fj);
                            nr += k * double(q.rho);
                            nmx += k * double(q.mx);
                            nmy += k * double(q.my);
                            nE += k * double(q.E);
                            den += k;
                        }
                    if (den > 1e-12)
                        c.at(ci, cj) = Cons{Real(nr / den), Real(nmx / den),
                                            Real(nmy / den), Real(nE / den)};
                }
        }
    }

    Patch makePatch_(int bi, int bj) {
        assert(!freeSlots_.empty());
        const int slot = freeSlots_.back();
        freeSlots_.pop_back();
        const int bc = cfg_.blockC;
        Patch p{bi, bj, bi * bc, bj * bc, slot, {}};
        GridRef g = patchRef(p);
        for (int j = NG; j < NG + nf_; ++j)
            for (int i = NG; i < NG + nf_; ++i) {
                const int gfi = 2 * p.ci0 + (i - NG);
                const int gfj = 2 * p.cj0 + (j - NG);
                g.at(i, j) = prolong_(gfi / 2, gfj / 2, gfi & 1, gfj & 1);
            }
        p.geo = buildGeo_(nf_, nf_, x0_ + p.ci0 * dxc_, y0_ + p.cj0 * dyc_,
                          nf_ * (dxc_ / 2), nf_ * (dyc_ / 2));
        pool_.setPoolGeometry(slot, p.geo);
        return p;
    }

    MetalContext& ctx_;
    AmrConfig cfg_;
    Real x0_, y0_;
    int nx_, ny_;
    Real dxc_, dyc_;
    CutCell2DGpu coarse_;
    CutCell2DGpu pool_;
    int nbx_, nby_, nf_ = 0, ptx_ = 0, ctx_stride_ = 0, capacity_ = 0;
    std::vector<int> blockOf_;
    std::vector<int> freeSlots_;
    std::vector<Cons> coarseOld_; // coarse t^n copy (subcycled ghosts)
    cutcell::Geometry coarseGeo_; // CPU mirror of coarse EB moments
    int stepCount_ = 0;
};

} // namespace mm

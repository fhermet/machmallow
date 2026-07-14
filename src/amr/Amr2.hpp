#pragma once

// Two-level block-structured AMR (CPU). The coarse level is one uniform
// grid, tiled into fixed blocks of cfg.blockC x cfg.blockC coarse cells;
// a tagged block is refined into one patch at ratio 2. No subcycling:
// every level advances with the same dt.
//
// Per step: fill ghosts (physical / sibling-copy / limited prolongation),
// advance coarse keeping its fluxes, advance each patch and reflux its
// coarse-fine faces, restrict fine onto covered coarse cells, regrid
// every cfg.regridEvery steps.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "numerics/Limiter.hpp"
#include "physics/Reaction.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mm {

struct AmrConfig {
    int blockC = 8;             // block size in coarse cells
    int maxLevels = 2;          // total levels incl. base (AmrML only)
    Real tagThreshold = Real(0.03); // relative density gradient
    Real tagVelocity = 0;       // velocity jump / sound speed (0 = off)
    int regridEvery = 4;
    int maxPatches = 0;         // GPU slot-pool capacity (0 = auto from
                                // device memory; AmrGpuML only)
    bool reflux = true;         // off only to demonstrate the leak
    bool subcycle = false;      // coarse at dt, fine at 2 x dt/2
    Real mu = 0;                // dynamic viscosity (0 = inviscid Euler)
    Real gx = 0, gy = 0;        // gravity (split source)
    bool species = false; // two-gas model (AmrML/AmrGpuML)
    bool weno = false; // WENO5 + SSP-RK3 instead of MUSCL-Hancock
                       // (multi-level classes only, single-gas)
    bool react = false;   // single-step reaction (implies species,
                          // gamma1 = gamma2); Strang-split source
    Reaction reaction;    // rate (A, Ea), heat q, ignition Tign
    Real gamma1 = Real(1.4), gamma2 = Real(1.4);
    bool periodicX = false;     // periodic domain (patch ghosts, reflux
    bool periodicY = false;     // and side masks wrap accordingly)
    bool cutCell = false;       // embedded-boundary cut cells (CPU, single-
                                // rate, Amr2): aperture-weighted fluxes +
                                // flux redistribution instead of a staircase
                                // mask. Requires Amr2::momentFn.
    bool cutCellO2 = false;     // 2nd-order cut cells (LSQ gradients +
                                // reconstruction + SSP-RK2) instead of the
                                // 1st-order operator. Full cells reduce to
                                // 2nd-order MUSCL-HLLC; only cut cells get the
                                // aperture / EB / FRD treatment. Implies
                                // cutCell.
};

class Amr2 {
public:
    struct Patch {
        int bi, bj;   // block coordinates
        int ci0, cj0; // coarse-cell origin of the block
        Grid grid;    // fine grid: 2*blockC cells square + ghosts
        std::vector<std::uint8_t> solid; // immersed mask (empty = none)
        cutcell::Geometry geo; // cut-cell moments (empty unless cfg.cutCell)
    };

    Grid coarse;
    std::vector<Patch> patches;

    // Case-supplied physical BC fill for the coarse grid (time-dependent
    // BCs receive t).
    std::function<void(Grid&, double)> fillPhysicalGhosts;

    // Optional fine-level physical BC for patches touching the domain
    // boundary (side bitmask). When unset, those ghosts come from
    // prolongated coarse ghosts.
    std::function<void(Grid&, double, unsigned)> fillPatchPhysical;

    // Optional immersed-solid mask: 1 where (x, y) is solid. When set,
    // the coarse grid is masked off there (reflective fluid/solid faces).
    // Refinement near solids is not yet supported — the case runner
    // disables AMR when solids are present, so only the coarse grid runs.
    std::function<std::uint8_t(Real, Real)> solidAt;

    // Cut-cell embedded boundary: analytic moments for the cell box
    // [x0,x1] x [y0,y1] (fluid volume fraction, face apertures, EB face).
    // When set together with cfg.cutCell, the solver uses aperture-weighted
    // cut-cell fluxes + flux redistribution instead of the staircase mask.
    std::function<CellMoments(double, double, double, double)> momentFn;

    Amr2(int nx, int ny, Real x0, Real y0, Real lx, Real ly, AmrConfig cfg)
        : coarse(nx, ny, x0, y0, lx, ly), cfg_(cfg), nbx_(nx / cfg.blockC),
          nby_(ny / cfg.blockC), blockOf_(std::size_t(nbx_) * nby_, -1) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
    }

    bool covered(int bi, int bj) const {
        return blockOf_[std::size_t(bj) * nbx_ + bi] >= 0;
    }

    // View interface shared with AmrGpu (writers, composite tooling).
    GridRef coarseRef() const {
        return GridRef{coarse.nx, coarse.ny, coarse.x0,
                       coarse.y0, coarse.dx, coarse.dy,
                       const_cast<Cons*>(coarse.q.data())};
    }
    GridRef patchRef(const Patch& p) const {
        return GridRef{p.grid.nx, p.grid.ny, p.grid.x0,
                       p.grid.y0, p.grid.dx, p.grid.dy,
                       const_cast<Cons*>(p.grid.q.data())};
    }
    int fineCells() const { return 2 * cfg_.blockC; }

    template <class IC>
    void init(IC ic) {
        for (int j = NG; j < NG + coarse.ny; ++j)
            for (int i = NG; i < NG + coarse.nx; ++i)
                coarse.at(i, j) = ic(coarse.xc(i), coarse.yc(j));
        buildCoarseSolid_();
        fillPhysicalGhosts(coarse, 0); // regrid prolongation reads ghosts
        regrid();
        for (Patch& p : patches)
            for (int j = NG; j < NG + p.grid.ny; ++j)
                for (int i = NG; i < NG + p.grid.nx; ++i)
                    p.grid.at(i, j) = ic(p.grid.xc(i), p.grid.yc(j));
        restrictFine_();
    }

    Real maxStableDtAll(Real cfl) const {
        Real dt = maxStableDt(coarse, cfl, cfg_.mu);
        for (const Patch& p : patches) {
            Real dtF = maxStableDt(p.grid, cfl, cfg_.mu);
            if (cfg_.subcycle) dtF *= 2; // fine takes two half steps
            dt = std::min(dt, dtF);
        }
        return dt;
    }

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

    void step(Real dt, double t) {
        if (cutCellOn_() && coarseGeo_.cell.empty())
            coarseGeo_ = buildGeo_(coarse);
        if (cutO2On_() && cfg_.subcycle && !patches.empty())
            stepCutO2Subcycled_(dt, t); // 2nd-order subcycled (RK2)
        else if (cutO2On_())
            stepCutO2SingleRate_(dt, t); // 2nd-order CPU cut cells (RK2)
        else if (cutCellOn_() && cfg_.subcycle && !patches.empty())
            stepCutSubcycled_(dt, t); // subcycled CPU cut cells
        else if (cutCellOn_())
            stepCutSingleRate_(dt, t); // single-rate CPU cut cells
        else if (cfg_.subcycle && !patches.empty())
            stepSubcycled_(dt, t);
        else
            stepSingleRate_(dt, t);
        restrictFine_();

        if (++stepCount_ % cfg_.regridEvery == 0) {
            fillPhysicalGhosts(coarse, t + dt); // for edge-block prolongation
            regrid();
        }
    }

    // Composite mass (uncovered coarse + fine), accumulated in double.
    double totalMass() const {
        double m = 0;
        const double ac = double(coarse.dx) * coarse.dy;
        for (int j = 0; j < coarse.ny; ++j)
            for (int i = 0; i < coarse.nx; ++i)
                if (!covered(i / cfg_.blockC, j / cfg_.blockC))
                    m += double(coarse.at(NG + i, NG + j).rho) * ac;
        for (const Patch& p : patches) {
            const double af = double(p.grid.dx) * p.grid.dy;
            for (int j = NG; j < NG + p.grid.ny; ++j)
                for (int i = NG; i < NG + p.grid.nx; ++i)
                    m += double(p.grid.at(i, j).rho) * af;
        }
        return m;
    }

    std::size_t cellCount() const {
        std::size_t n = std::size_t(coarse.nx) * coarse.ny;
        for (const Patch& p : patches)
            n += std::size_t(p.grid.nx) * p.grid.ny;
        return n;
    }

    int blockCount() const { return nbx_ * nby_; }
    int stepCount() const { return stepCount_; }

    // Rebuild the patch set at the given blocks (checkpoint restart).
    // Patch data is prolongated then expected to be overwritten by the
    // caller; stepCount keeps the regrid cadence in phase.
    void restoreBlocks(const std::vector<std::pair<int, int>>& blocks,
                       int stepCount) {
        patches.clear();
        std::fill(blockOf_.begin(), blockOf_.end(), -1);
        for (const auto& [bi, bj] : blocks) {
            patches.push_back(makePatch_(bi, bj));
            blockOf_[std::size_t(bj) * nbx_ + bi] = int(patches.size()) - 1;
        }
        stepCount_ = stepCount;
    }

    void regrid() {
        const int nx = coarse.nx, ny = coarse.ny, bc = cfg_.blockC;
        if (cutCellOn_() && coarseGeo_.cell.empty())
            coarseGeo_ = buildGeo_(coarse);

        // 1. Tag on relative density gradient, optionally on velocity
        //    jumps normalized by the local sound speed (shear layers and
        //    vortices are invisible to the density criterion).
        std::vector<std::uint8_t> tag(std::size_t(nx) * ny, 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int ip = std::min(i + 1, nx - 1),
                          im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, ny - 1),
                          jm = std::max(j - 1, 0);
                const Real r0 = coarse.at(NG + i, NG + j).rho;
                const Real ex = std::fabs(coarse.at(NG + ip, NG + j).rho -
                                          coarse.at(NG + im, NG + j).rho);
                const Real ey = std::fabs(coarse.at(NG + i, NG + jp).rho -
                                          coarse.at(NG + i, NG + jm).rho);
                bool t = std::max(ex, ey) / r0 > cfg_.tagThreshold;
                if (!t && cfg_.tagVelocity > 0) {
                    const auto uOf = [&](int a, int b) {
                        const Cons& q = coarse.at(NG + a, NG + b);
                        return q.mx / std::max(q.rho, RHO_FLOOR);
                    };
                    const auto vOf = [&](int a, int b) {
                        const Cons& q = coarse.at(NG + a, NG + b);
                        return q.my / std::max(q.rho, RHO_FLOOR);
                    };
                    const Real du = std::max(
                        std::fabs(uOf(ip, j) - uOf(im, j)),
                        std::fabs(uOf(i, jp) - uOf(i, jm)));
                    const Real dv = std::max(
                        std::fabs(vOf(ip, j) - vOf(im, j)),
                        std::fabs(vOf(i, jp) - vOf(i, jm)));
                    const Real c0 =
                        soundSpeed(toPrim(coarse.at(NG + i, NG + j)));
                    t = std::max(du, dv) / c0 > cfg_.tagVelocity;
                }
                // Immersed boundary: tag fluid cells touching a solid so
                // the body surface (the staircase) gets refined — unless
                // refinement is disabled (tag_threshold sentinel).
                if (!t && !coarseSolidMask_.empty() &&
                    cfg_.tagThreshold < Real(1e29) &&
                    !solCoarse_(NG + i, NG + j) &&
                    (solCoarse_(NG + ip, NG + j) ||
                     solCoarse_(NG + im, NG + j) ||
                     solCoarse_(NG + i, NG + jp) ||
                     solCoarse_(NG + i, NG + jm)))
                    t = true;
                // Cut-cell embedded boundary: tag the EB band — cells the
                // solid cuts (0 < kappa < 1). The ±2 dilation below extends it
                // into the surrounding fluid, so the body surface is refined
                // without over-refining the solid interior (kappa = 0 cells).
                if (!t && cutCellOn_()) {
                    const Real k = coarseGeo_.at(NG + i, NG + j).vol;
                    t = k > Real(1e-6) && k < Real(1) - Real(1e-6);
                }
                tag[std::size_t(j) * nx + i] = t;
            }

        // 2. Dilate by 2 cells (buffer so features stay inside patches
        //    between regrids).
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

        // 4. Rebuild the patch list, keeping existing patches' data and
        //    prolongating newly refined blocks from the coarse level.
        std::vector<Patch> next;
        std::vector<int> nextOf(std::size_t(nbx_) * nby_, -1);
        for (int bj = 0; bj < nby_; ++bj)
            for (int bi = 0; bi < nbx_; ++bi) {
                if (!want[std::size_t(bj) * nbx_ + bi]) continue;
                const int old = blockOf_[std::size_t(bj) * nbx_ + bi];
                if (old >= 0) {
                    next.push_back(std::move(patches[old]));
                } else {
                    next.push_back(makePatch_(bi, bj));
                }
                nextOf[std::size_t(bj) * nbx_ + bi] = int(next.size()) - 1;
            }
        patches = std::move(next);
        blockOf_ = std::move(nextOf);
    }

private:
    // Sample the solid mask onto every coarse cell (ghosts included, so
    // step2D's neighbour lookups stay in range). No-op without solidAt.
    void buildCoarseSolid_() {
        if (!solidAt) { coarseSolidMask_.clear(); return; }
        coarseSolidMask_.assign(coarse.q.size(), 0);
        for (int j = 0; j < coarse.toty(); ++j)
            for (int i = 0; i < coarse.totx(); ++i)
                coarseSolidMask_[coarse.idx(i, j)] =
                    solidAt(coarse.xc(i), coarse.yc(j));
    }

    void fillAllPatchGhosts_(double t, Real theta) {
        for (Patch& p : patches) {
            fillPatchGhosts_(p, theta);
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides(p))
                    fillPatchPhysical(p.grid, t, sides);
        }
    }

    // Coarse solid mask pointer (nullptr when no [solid] regions).
    const std::uint8_t* coarseSolid_() const {
        return coarseSolidMask_.empty() ? nullptr : coarseSolidMask_.data();
    }
    bool solCoarse_(int i, int j) const {
        return !coarseSolidMask_.empty() &&
               coarseSolidMask_[coarse.idx(i, j)] != 0;
    }
    static const std::uint8_t* psolid_(const Patch& p) {
        return p.solid.empty() ? nullptr : p.solid.data();
    }
    // Sample the solid mask onto a patch (ghosts included). No-op if unset.
    void buildPatchSolid_(Patch& p) const {
        if (!solidAt) { p.solid.clear(); return; }
        p.solid.assign(p.grid.q.size(), 0);
        for (int j = 0; j < p.grid.toty(); ++j)
            for (int i = 0; i < p.grid.totx(); ++i)
                p.solid[p.grid.idx(i, j)] =
                    solidAt(p.grid.xc(i), p.grid.yc(j));
    }

    void stepSingleRate_(Real dt, double t) {
        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1));

        step2D(coarse, dt, scratchC_, cfg_.mu, cfg_.gx, cfg_.gy,
               coarseSolid_()); // + fluxes kept
        for (Patch& p : patches) {
            step2D(p.grid, dt, scratchF_, cfg_.mu, cfg_.gx, cfg_.gy,
                   psolid_(p));
            if (cfg_.reflux) {
                refluxCoarse_(p, dt);
                refluxFine_(p, dt, scratchF_);
            }
        }
    }

    // Berger-Colella subcycling: coarse advances dt, fine takes two dt/2
    // substeps with time-interpolated coarse ghosts; refluxing accumulates
    // dt*Fc against both substeps' dt/2*<Ff>.
    void stepSubcycled_(Real dtC, double t) {
        const Real dtF = dtC / 2;
        coarseOld_ = coarse.q; // t^n copy for the theta-blend prolongation

        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1)); // substep 1 ghosts at t^n

        step2D(coarse, dtC, scratchC_, cfg_.mu, cfg_.gx, cfg_.gy,
               coarseSolid_());
        for (Patch& p : patches) {
            if (cfg_.reflux) refluxCoarse_(p, dtC);
            step2D(p.grid, dtF, scratchF_, cfg_.mu, cfg_.gx, cfg_.gy,
                   psolid_(p));
            if (cfg_.reflux) refluxFine_(p, dtF, scratchF_);
        }

        // Substep 2: sibling copies are current; coarse prolongation uses
        // the half-time blend of t^n and t^{n+1}.
        fillAllPatchGhosts_(t + dtF, Real(0.5));
        for (Patch& p : patches) {
            step2D(p.grid, dtF, scratchF_, cfg_.mu, cfg_.gx, cfg_.gy,
                   psolid_(p));
            if (cfg_.reflux) refluxFine_(p, dtF, scratchF_);
        }
    }

    // ---- cut-cell embedded boundary (single-rate, CPU) -------------------

    bool cutCellOn_() const { return cfg_.cutCell && bool(momentFn); }

    cutcell::Geometry buildGeo_(const Grid& g) const {
        return cutcell::build(g, [&](double x0, double x1, double y0,
                                     double y1) {
            return momentFn(x0, x1, y0, y1);
        });
    }

    // One single-rate cut-cell step. Ghosts are filled from the t^n state, the
    // base grid advances with the single-grid cut-cell operator, and the fine
    // patches advance as a COMPOSITE so flux redistribution crosses patch
    // boundaries (a body spanning several patches would otherwise leak mass at
    // the sibling seams). Each coarse-fine interface is then refluxed (back the
    // coarse flux out, apply the fine flux) once.
    void stepCutSingleRate_(Real dt, double t) {
        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1)); // patch ghosts from coarse t^n

        // base grid: single grid, no siblings -> the single-grid operator.
        cutCellStepFluxed(coarse, coarseGeo_, dt, [](Grid&) {}, cutFxC_,
                          cutFyC_);
        advanceCutFinePatches_(dt);
        if (cfg_.reflux)
            for (std::size_t k = 0; k < patches.size(); ++k) {
                cutRefluxBackout_(patches[k], dt);
                cutRefluxFineApply_(patches[k], dt, pFx_[k], pFy_[k]);
            }
    }

    // Berger-Colella subcycling for cut cells: the base grid advances one dtC,
    // each patch takes two dtF = dtC/2 substeps with time-interpolated coarse
    // ghosts. Refluxing accumulates: back the (single) coarse flux out once,
    // apply each substep's fine flux (2 x dtF = dtC). The fine advance is the
    // same composite (cross-patch FRD) as single-rate, run per substep.
    void stepCutSubcycled_(Real dtC, double t) {
        const Real dtF = dtC / 2;
        coarseOld_ = coarse.q; // t^n copy for the theta-blend prolongation

        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1)); // substep-1 ghosts at t^n

        cutCellStepFluxed(coarse, coarseGeo_, dtC, [](Grid&) {}, cutFxC_,
                          cutFyC_);
        if (cfg_.reflux)
            for (Patch& p : patches) cutRefluxBackout_(p, dtC);

        advanceCutFinePatches_(dtF); // substep 1
        if (cfg_.reflux)
            for (std::size_t k = 0; k < patches.size(); ++k)
                cutRefluxFineApply_(patches[k], dtF, pFx_[k], pFy_[k]);

        // substep 2: siblings are current; coarse prolongation blends t^n/t^n+1.
        fillAllPatchGhosts_(t + dtF, Real(0.5));
        advanceCutFinePatches_(dtF); // substep 2
        if (cfg_.reflux)
            for (std::size_t k = 0; k < patches.size(); ++k)
                cutRefluxFineApply_(patches[k], dtF, pFx_[k], pFy_[k]);
    }

    // Composite advance of the fine patches (no reflux): compute each patch's
    // conservative divergence Dc, fill Dc ghosts from same-level siblings, form
    // the hybrid divergence + redistribution, SCATTER the redistribution that
    // lands in a patch's ghosts into the sibling that owns those cells, then
    // update. Records the per-patch extensive fluxes (pFx_/pFy_) for reflux.
    void advanceCutFinePatches_(Real dt) {
        if (patches.empty()) return;
        cutFineDivergence_(cfg_.cutCellO2);
        for (std::size_t k = 0; k < patches.size(); ++k)
            applyCutUpdate(patches[k].grid, patches[k].geo, pD_[k], dt);
    }

    // Composite divergence of every patch (no update): per-patch Dc (1st or 2nd
    // order), Dc ghosts from siblings, hybrid + FRD, redistribution scattered
    // into the owning siblings. Records extensive fluxes in pFx_/pFy_.
    void cutFineDivergence_(bool o2) {
        const std::size_t np = patches.size();
        pDc_.resize(np);
        pD_.resize(np);
        pFx_.resize(np);
        pFy_.resize(np);
        if (o2) {
            pGrad_.resize(np);
            for (std::size_t k = 0; k < np; ++k)
                pGrad_[k] = lsqGradients(patches[k].grid, patches[k].geo);
            for (std::size_t k = 0; k < np; ++k) fillPatchGradGhosts_(k);
            for (std::size_t k = 0; k < np; ++k)
                pDc_[k] = cutCellDcO2(patches[k].grid, patches[k].geo,
                                      pGrad_[k], pFx_[k], pFy_[k],
                                      domainSides(patches[k]), cfg_.mu);
        } else {
            for (std::size_t k = 0; k < np; ++k)
                pDc_[k] = cutCellDc(patches[k].grid, patches[k].geo, pFx_[k],
                                    pFy_[k]);
        }
        for (std::size_t k = 0; k < np; ++k) fillPatchDcGhosts_(k);
        for (std::size_t k = 0; k < np; ++k)
            pD_[k] = cutCellHybridD(patches[k].grid, patches[k].geo, pDc_[k]);
        for (std::size_t k = 0; k < np; ++k) scatterPatchDGhosts_(k);
    }

    bool cutO2On_() const { return cutCellOn_() && cfg_.cutCellO2; }

    // Base-grid 2nd-order divergence (single grid: ghost Dc stays 0, the FRD
    // sheds nothing across the domain boundary). Records base fluxes.
    std::vector<Cons> cutBaseDivergenceO2_() {
        const auto grad = lsqGradients(coarse, coarseGeo_);
        const std::vector<Cons> Dc =
            cutCellDcO2(coarse, coarseGeo_, grad, cutFxC_, cutFyC_,
                        SideLeft | SideRight | SideBottom | SideTop, cfg_.mu);
        return cutCellHybridD(coarse, coarseGeo_, Dc);
    }

    // SSP-RK2 combine of one grid: U <- floor(0.5 Uold + 0.5 (U + dt D)).
    void rk2Combine_(Grid& g, const cutcell::Geometry& geo,
                     const std::vector<Cons>& old, const std::vector<Cons>& D,
                     Real dt) {
        const Real tiny = Real(1e-9);
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                if (double(geo.at(i, j).vol) > double(tiny)) {
                    const std::size_t id = g.idx(i, j);
                    g.at(i, j) = floorState(Real(0.5) * old[id] +
                                            Real(0.5) * (g.at(i, j) +
                                                         dt * D[id]));
                }
    }

    // 2nd-order (SSP-RK2) single-rate cut-cell step. Both the base grid and the
    // fine composite advance with two stages; reflux uses the time-AVERAGED
    // extensive fluxes 0.5*(F1+F2), so conservation is preserved. Full cells
    // reduce to 2nd-order MUSCL-HLLC; only cut cells get aperture / EB / FRD.
    void stepCutO2SingleRate_(Real dt, double t) {
        const std::size_t np = patches.size();
        // ---- stage 1 (state = U^n) ----
        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1));
        const std::vector<Cons> DC1 = cutBaseDivergenceO2_();
        std::vector<Cons> Fc1x = cutFxC_, Fc1y = cutFyC_; // base stage-1 flux
        cutFineDivergence_(true);
        std::vector<std::vector<Cons>> pD1 = pD_, pF1x = pFx_, pF1y = pFy_;

        coarseO2Old_ = coarse.q;
        pOld_.resize(np);
        for (std::size_t k = 0; k < np; ++k) pOld_[k] = patches[k].grid.q;

        applyCutUpdate(coarse, coarseGeo_, DC1, dt); // base -> U1
        for (std::size_t k = 0; k < np; ++k)
            applyCutUpdate(patches[k].grid, patches[k].geo, pD1[k], dt);

        // ---- stage 2 (state = U1) ----
        fillPhysicalGhosts(coarse, t + dt);
        fillAllPatchGhosts_(t + dt, Real(-1));
        const std::vector<Cons> DC2 = cutBaseDivergenceO2_(); // records Fc2
        cutFineDivergence_(true);                             // records Ff2

        rk2Combine_(coarse, coarseGeo_, coarseO2Old_, DC2, dt);
        for (std::size_t k = 0; k < np; ++k)
            rk2Combine_(patches[k].grid, patches[k].geo, pOld_[k], pD_[k], dt);

        // ---- reflux with the time-averaged fluxes 0.5*(F1+F2) ----
        if (cfg_.reflux) {
            avgFlux_(cutFxC_, Fc1x);
            avgFlux_(cutFyC_, Fc1y);
            for (std::size_t k = 0; k < np; ++k) {
                avgFlux_(pFx_[k], pF1x[k]);
                avgFlux_(pFy_[k], pF1y[k]);
            }
            for (std::size_t k = 0; k < np; ++k) {
                cutRefluxBackout_(patches[k], dt);
                cutRefluxFineApply_(patches[k], dt, pFx_[k], pFy_[k]);
            }
        }
    }

    static void avgFlux_(std::vector<Cons>& a, const std::vector<Cons>& b) {
        for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = Real(0.5) * (a[i] + b[i]);
    }

    // Base-grid SSP-RK2 over dtC. Leaves the base at t^{n+1} and the TIME-
    // AVERAGED extensive base fluxes 0.5*(Fc1+Fc2) in cutFxC_/cutFyC_ (for the
    // single reflux back-out). Assumes base ghosts filled at t; refills at t+dtC.
    void cutBaseRK2_(Real dtC, double t) {
        const std::vector<Cons> DC1 = cutBaseDivergenceO2_(); // records Fc1
        std::vector<Cons> Fc1x = cutFxC_, Fc1y = cutFyC_;
        coarseO2Old_ = coarse.q;
        applyCutUpdate(coarse, coarseGeo_, DC1, dtC); // -> U1
        fillPhysicalGhosts(coarse, t + dtC);
        const std::vector<Cons> DC2 = cutBaseDivergenceO2_(); // records Fc2
        rk2Combine_(coarse, coarseGeo_, coarseO2Old_, DC2, dtC);
        avgFlux_(cutFxC_, Fc1x);
        avgFlux_(cutFyC_, Fc1y);
    }

    // One fine-composite SSP-RK2 substep over dtF, with time-interpolated coarse
    // ghosts (theta1 at the substep start, theta2 at its end). Leaves the
    // patches advanced and the TIME-AVERAGED fine fluxes in pFx_/pFy_.
    void cutFineRK2Substep_(Real dtF, double tS1, Real theta1, double tS2,
                            Real theta2) {
        const std::size_t np = patches.size();
        // stage 1
        fillAllPatchGhosts_(tS1, theta1);
        cutFineDivergence_(true);
        std::vector<std::vector<Cons>> pF1x = pFx_, pF1y = pFy_;
        pOld_.resize(np);
        for (std::size_t k = 0; k < np; ++k) pOld_[k] = patches[k].grid.q;
        for (std::size_t k = 0; k < np; ++k)
            applyCutUpdate(patches[k].grid, patches[k].geo, pD_[k], dtF);
        // stage 2
        fillAllPatchGhosts_(tS2, theta2);
        cutFineDivergence_(true);
        for (std::size_t k = 0; k < np; ++k)
            rk2Combine_(patches[k].grid, patches[k].geo, pOld_[k], pD_[k], dtF);
        for (std::size_t k = 0; k < np; ++k) {
            avgFlux_(pFx_[k], pF1x[k]);
            avgFlux_(pFy_[k], pF1y[k]);
        }
    }

    // 2nd-order (SSP-RK2) SUBCYCLED cut-cell step: the base advances one dtC
    // (RK2), each patch takes two dtF = dtC/2 substeps (RK2 each) with time-
    // interpolated coarse ghosts. Reflux backs the (single, averaged) coarse
    // flux out once, applies each substep's averaged fine flux.
    void stepCutO2Subcycled_(Real dtC, double t) {
        const Real dtF = dtC / 2;
        coarseOld_ = coarse.q; // t^n for the theta-blend prolongation
        fillPhysicalGhosts(coarse, t);
        cutBaseRK2_(dtC, t);   // base -> t^{n+1}, net flux in cutFxC_/cutFyC_
        if (cfg_.reflux)
            for (Patch& p : patches) cutRefluxBackout_(p, dtC);

        cutFineRK2Substep_(dtF, t, Real(0), t + dtF, Real(0.5)); // substep 1
        if (cfg_.reflux)
            for (std::size_t k = 0; k < patches.size(); ++k)
                cutRefluxFineApply_(patches[k], dtF, pFx_[k], pFy_[k]);

        cutFineRK2Substep_(dtF, t + dtF, Real(0.5), t + dtC, Real(1)); // sub 2
        if (cfg_.reflux)
            for (std::size_t k = 0; k < patches.size(); ++k)
                cutRefluxFineApply_(patches[k], dtF, pFx_[k], pFy_[k]);
    }

    // Global fine-cell (gfi,gfj) that a patch cell (i,j) maps to, with periodic
    // wrap. Returns the owning patch index (-1 if none / coarse-fine / domain).
    int siblingCell_(const Patch& p, int i, int j, int& si, int& sj) const {
        const int nxf = 2 * coarse.nx, nyf = 2 * coarse.ny;
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

    // Fill patch k's Dc ghosts from the sibling patch that owns each ghost cell
    // (same-level copy). Ghosts with no sibling (coarse-fine / domain) stay 0.
    void fillPatchDcGhosts_(std::size_t k) {
        const Patch& p = patches[k];
        const int nf = p.grid.nx;
        for (int j = 0; j < p.grid.toty(); ++j)
            for (int i = 0; i < p.grid.totx(); ++i) {
                if (i >= NG && i < NG + nf && j >= NG && j < NG + nf) continue;
                int si, sj;
                const int pi = siblingCell_(p, i, j, si, sj);
                if (pi >= 0)
                    pDc_[k][p.grid.idx(i, j)] = pDc_[pi][patches[pi].grid.idx(
                        si, sj)];
            }
    }

    // Fill patch k's gradient ghosts from the sibling that owns each ghost cell
    // (same-level copy), so 2nd-order reconstruction at sibling seams sees the
    // neighbour's slope. Ghosts with no sibling (coarse-fine / domain) stay 0 —
    // constant reconstruction there, which cutCellDcO2's physSides handles.
    void fillPatchGradGhosts_(std::size_t k) {
        const Patch& p = patches[k];
        const int nf = p.grid.nx;
        for (int j = 0; j < p.grid.toty(); ++j)
            for (int i = 0; i < p.grid.totx(); ++i) {
                if (i >= NG && i < NG + nf && j >= NG && j < NG + nf) continue;
                int si, sj;
                const int pi = siblingCell_(p, i, j, si, sj);
                if (pi >= 0)
                    pGrad_[k][p.grid.idx(i, j)] =
                        pGrad_[pi][patches[pi].grid.idx(si, sj)];
            }
    }

    // Scatter the redistribution that landed in patch k's ghost cells into the
    // sibling interior cell that owns them (reads only ghosts, writes only
    // interiors -> order-independent across patches).
    void scatterPatchDGhosts_(std::size_t k) {
        const Patch& p = patches[k];
        const int nf = p.grid.nx;
        for (int j = 0; j < p.grid.toty(); ++j)
            for (int i = 0; i < p.grid.totx(); ++i) {
                if (i >= NG && i < NG + nf && j >= NG && j < NG + nf) continue;
                int si, sj;
                const int pi = siblingCell_(p, i, j, si, sj);
                if (pi >= 0) {
                    const std::size_t d = patches[pi].grid.idx(si, sj);
                    pD_[pi][d] = pD_[pi][d] + pD_[k][p.grid.idx(i, j)];
                }
            }
    }

    // Cut-aware reflux: for each uncovered coarse cell just outside the patch,
    // replace the coarse interface flux by the sum of the two fine fluxes on
    // that face. Extensive (aperture-weighted) fluxes make this reduce to the
    // uniform (dt/dx)*F reflux when kappa = apertures = 1. correction is
    // applied as dt/(kappa*V_c)*dF through the positivity floor.
    // Reflux, split so subcycling can accumulate several fine substeps against
    // one coarse step. Both parts apply dt/(kappa V_c)*flux (extensive) to the
    // uncovered coarse cell just outside the patch, through the positivity
    // floor, and reduce to the uniform reflux when kappa = apertures = 1.
    Cons cutRefluxCorr_(int ci, int cj, const Cons& d, Real dt) {
        const double k = double(coarseGeo_.at(NG + ci, NG + cj).vol);
        if (k > 1e-9)
            return floorState(coarse.at(NG + ci, NG + cj) +
                              Real(dt / (k * coarse.dx * coarse.dy)) * d);
        return coarse.at(NG + ci, NG + cj);
    }

    // Back the COARSE interface flux out of the uncovered neighbour (once, at
    // dtC).
    void cutRefluxBackout_(const Patch& p, Real dtC) {
        const int bc = cfg_.blockC;
        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + n.cell, NG + p.cj0 + r) = cutRefluxCorr_(
                    n.cell, p.cj0 + r,
                    cutFxC_[coarse.idx(NG + n.cell, NG + p.cj0 + r)], dtC);
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + n.cell, NG + p.cj0 + r) = cutRefluxCorr_(
                    n.cell, p.cj0 + r,
                    Real(-1) *
                        cutFxC_[coarse.idx(NG + n.cell - 1, NG + p.cj0 + r)],
                    dtC);
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 + r, NG + n.cell) = cutRefluxCorr_(
                    p.ci0 + r, n.cell,
                    cutFyC_[coarse.idx(NG + p.ci0 + r, NG + n.cell)], dtC);
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 + r, NG + n.cell) = cutRefluxCorr_(
                    p.ci0 + r, n.cell,
                    Real(-1) *
                        cutFyC_[coarse.idx(NG + p.ci0 + r, NG + n.cell - 1)],
                    dtC);
    }

    // Apply this substep's FINE flux (sum of the two fine faces) to the
    // uncovered neighbour (at dtF; called once per substep). Uses the patch's
    // recorded extensive fluxes (pFx_/pFy_).
    void cutRefluxFineApply_(const Patch& p, Real dtF, const std::vector<Cons>& Fx,
                             const std::vector<Cons>& Fy) {
        const int bc = cfg_.blockC, nf = 2 * bc;
        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fx[p.grid.idx(NG - 1, NG + 2 * r)] +
                                Fx[p.grid.idx(NG - 1, NG + 2 * r + 1)];
                coarse.at(NG + n.cell, NG + p.cj0 + r) =
                    cutRefluxCorr_(n.cell, p.cj0 + r, Real(-1) * sf, dtF);
            }
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fx[p.grid.idx(NG + nf - 1, NG + 2 * r)] +
                                Fx[p.grid.idx(NG + nf - 1, NG + 2 * r + 1)];
                coarse.at(NG + n.cell, NG + p.cj0 + r) =
                    cutRefluxCorr_(n.cell, p.cj0 + r, sf, dtF);
            }
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fy[p.grid.idx(NG + 2 * r, NG - 1)] +
                                Fy[p.grid.idx(NG + 2 * r + 1, NG - 1)];
                coarse.at(NG + p.ci0 + r, NG + n.cell) =
                    cutRefluxCorr_(p.ci0 + r, n.cell, Real(-1) * sf, dtF);
            }
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                const Cons sf = Fy[p.grid.idx(NG + 2 * r, NG + nf - 1)] +
                                Fy[p.grid.idx(NG + 2 * r + 1, NG + nf - 1)];
                coarse.at(NG + p.ci0 + r, NG + n.cell) =
                    cutRefluxCorr_(p.ci0 + r, n.cell, sf, dtF);
            }
    }

    // Volume-weighted (kappa) restriction of each 2x2 fine block onto its
    // covered coarse cell. Exact-moment geometry makes this conservative: the
    // coarse fluid volume equals the sum of the four children's fluid volumes.
    void restrictCut_() {
        for (const Patch& p : patches)
            for (int b = 0; b < cfg_.blockC; ++b)
                for (int a = 0; a < cfg_.blockC; ++a) {
                    const int ci = NG + p.ci0 + a, cj = NG + p.cj0 + b;
                    double nr = 0, nmx = 0, nmy = 0, nE = 0, den = 0;
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const int fi = NG + 2 * a + di, fj = NG + 2 * b + dj;
                            const double k = double(p.geo.at(fi, fj).vol);
                            if (k <= 1e-9) continue;
                            const Cons& q = p.grid.at(fi, fj);
                            nr += k * double(q.rho);
                            nmx += k * double(q.mx);
                            nmy += k * double(q.my);
                            nE += k * double(q.E);
                            den += k;
                        }
                    if (den > 1e-12)
                        coarse.at(ci, cj) =
                            Cons{Real(nr / den), Real(nmx / den),
                                 Real(nmy / den), Real(nE / den)};
                }
    }

    Patch makePatch_(int bi, int bj) {
        const int bc = cfg_.blockC, nf = 2 * bc;
        const int ci0 = bi * bc, cj0 = bj * bc;
        Patch p{bi, bj, ci0, cj0,
                Grid(nf, nf, coarse.x0 + ci0 * coarse.dx,
                     coarse.y0 + cj0 * coarse.dy, bc * coarse.dx,
                     bc * coarse.dy)};
        for (int j = NG; j < NG + nf; ++j)
            for (int i = NG; i < NG + nf; ++i) {
                const int gfi = 2 * ci0 + (i - NG);
                const int gfj = 2 * cj0 + (j - NG);
                p.grid.at(i, j) =
                    prolong_(gfi / 2, gfj / 2, gfi & 1, gfj & 1);
            }
        buildPatchSolid_(p);
        if (cutCellOn_()) p.geo = buildGeo_(p.grid);
        return p;
    }

    // Coarse value at (I, J): current state, or the theta-blend between
    // the saved t^n copy and the current state (subcycled ghosts).
    Cons coarseAt_(int I, int J, Real theta) const {
        const Cons& cur = coarse.at(I, J);
        if (theta < 0) return cur;
        const Cons& old = coarseOld_[coarse.idx(I, J)];
        return (1 - theta) * old + theta * cur;
    }

    // Conservative limited-linear interpolation of fine cell (offset
    // ox, oy in {0,1} within coarse cell ci, cj; ci/cj may index ghosts).
    Cons prolong_(int ci, int cj, int ox, int oy,
                  Real theta = Real(-1)) const {
        const int I = NG + ci, J = NG + cj;
        const Cons q0 = coarseAt_(I, J, theta);
        // Near a solid, drop to piecewise-constant in the affected
        // direction: a frozen solid neighbour must not feed the slope.
        const bool sX = solCoarse_(I - 1, J) || solCoarse_(I + 1, J);
        const bool sY = solCoarse_(I, J - 1) || solCoarse_(I, J + 1);
        const Cons dqx = sX ? Cons{0, 0, 0, 0}
                            : limitedSlope(coarseAt_(I - 1, J, theta), q0,
                                           coarseAt_(I + 1, J, theta));
        const Cons dqy = sY ? Cons{0, 0, 0, 0}
                            : limitedSlope(coarseAt_(I, J - 1, theta), q0,
                                           coarseAt_(I, J + 1, theta));
        const Real sx = ox ? Real(0.25) : Real(-0.25);
        const Real sy = oy ? Real(0.25) : Real(-0.25);
        return q0 + sx * dqx + sy * dqy;
    }

    void fillPatchGhosts_(Patch& p, Real theta = Real(-1)) {
        const int nf = p.grid.nx;
        const int nxf = 2 * coarse.nx, nyf = 2 * coarse.ny;
        for (int j = 0; j < p.grid.toty(); ++j)
            for (int i = 0; i < p.grid.totx(); ++i) {
                if (i >= NG && i < NG + nf && j >= NG && j < NG + nf)
                    continue; // interior
                int gfi = 2 * p.ci0 + (i - NG);
                int gfj = 2 * p.cj0 + (j - NG);
                // Periodic wrap: the cell beyond the seam lives on the
                // other side (sibling copy or interior prolongation).
                if (cfg_.periodicX) gfi = (gfi % nxf + nxf) % nxf;
                if (cfg_.periodicY) gfj = (gfj % nyf + nyf) % nyf;

                // Same-level copy when a sibling patch owns the cell.
                if (gfi >= 0 && gfi < nxf && gfj >= 0 && gfj < nyf) {
                    const int bi = gfi / (2 * cfg_.blockC);
                    const int bj = gfj / (2 * cfg_.blockC);
                    const int pi = blockOf_[std::size_t(bj) * nbx_ + bi];
                    if (pi >= 0) {
                        const Patch& s = patches[pi];
                        p.grid.at(i, j) = s.grid.at(NG + gfi - 2 * s.ci0,
                                                    NG + gfj - 2 * s.cj0);
                        continue;
                    }
                }
                // Otherwise limited prolongation from the coarse level
                // (coarse ghosts carry the physical BCs).
                const int ci = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                p.grid.at(i, j) =
                    prolong_(ci, cj, gfi - 2 * ci, gfj - 2 * cj, theta);
            }
    }

    // Refluxing, split so subcycling can accumulate several fine substeps
    // against one coarse step: refluxCoarse_ backs the coarse flux out of
    // the uncovered neighbour, refluxFine_ applies a substep's
    // area-averaged fine flux. Single-rate calls both once.
    // Neighbour-side geometry for refluxing, with periodic wrap.
    // ok=false means a physical (non-periodic) domain boundary: no
    // coarse-fine face there. `cell` is the corrected uncovered coarse
    // cell index in the side's direction; the coarse face adjacent to it
    // is its right/top face for low sides and left/bottom face for high
    // sides (handled at the call sites).
    struct SideNb {
        bool ok;
        int block, cell;
    };
    SideNb leftNb_(const Patch& p) const {
        if (p.bi > 0) return {true, p.bi - 1, p.ci0 - 1};
        return cfg_.periodicX ? SideNb{true, nbx_ - 1, coarse.nx - 1}
                              : SideNb{false, 0, 0};
    }
    SideNb rightNb_(const Patch& p) const {
        const int bc = cfg_.blockC;
        if (p.bi < nbx_ - 1) return {true, p.bi + 1, p.ci0 + bc};
        return cfg_.periodicX ? SideNb{true, 0, 0} : SideNb{false, 0, 0};
    }
    SideNb bottomNb_(const Patch& p) const {
        if (p.bj > 0) return {true, p.bj - 1, p.cj0 - 1};
        return cfg_.periodicY ? SideNb{true, nby_ - 1, coarse.ny - 1}
                              : SideNb{false, 0, 0};
    }
    SideNb topNb_(const Patch& p) const {
        const int bc = cfg_.blockC;
        if (p.bj < nby_ - 1) return {true, p.bj + 1, p.cj0 + bc};
        return cfg_.periodicY ? SideNb{true, 0, 0} : SideNb{false, 0, 0};
    }

    void refluxCoarse_(const Patch& p, Real dt) {
        const int bc = cfg_.blockC;
        const Real lx = dt / coarse.dx, ly = dt / coarse.dy;

        // A solid uncovered neighbour is frozen — never reflux into it.
        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + n.cell, NG + p.cj0 + r)) continue;
                coarse.at(NG + n.cell, NG + p.cj0 + r) +=
                    lx * scratchC_.Fx[coarse.idx(NG + n.cell,
                                                 NG + p.cj0 + r)];
            }
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + n.cell, NG + p.cj0 + r)) continue;
                coarse.at(NG + n.cell, NG + p.cj0 + r) -=
                    lx * scratchC_.Fx[coarse.idx(NG + n.cell - 1,
                                                 NG + p.cj0 + r)];
            }
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + p.ci0 + r, NG + n.cell)) continue;
                coarse.at(NG + p.ci0 + r, NG + n.cell) +=
                    ly * scratchC_.Fy[coarse.idx(NG + p.ci0 + r,
                                                 NG + n.cell)];
            }
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + p.ci0 + r, NG + n.cell)) continue;
                coarse.at(NG + p.ci0 + r, NG + n.cell) -=
                    ly * scratchC_.Fy[coarse.idx(NG + p.ci0 + r,
                                                 NG + n.cell - 1)];
            }
    }

    void refluxFine_(const Patch& p, Real dt, const Scratch2D& sf) {
        const int bc = cfg_.blockC, nf = 2 * bc;
        const Real lx = dt / coarse.dx, ly = dt / coarse.dy;

        if (const SideNb n = leftNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + n.cell, NG + p.cj0 + r)) continue;
                const Cons ff =
                    Real(0.5) * (sf.Fx[p.grid.idx(NG - 1, NG + 2 * r)] +
                                 sf.Fx[p.grid.idx(NG - 1, NG + 2 * r + 1)]);
                coarse.at(NG + n.cell, NG + p.cj0 + r) -= lx * ff;
            }
        if (const SideNb n = rightNb_(p); n.ok && !covered(n.block, p.bj))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + n.cell, NG + p.cj0 + r)) continue;
                const Cons ff =
                    Real(0.5) *
                    (sf.Fx[p.grid.idx(NG + nf - 1, NG + 2 * r)] +
                     sf.Fx[p.grid.idx(NG + nf - 1, NG + 2 * r + 1)]);
                coarse.at(NG + n.cell, NG + p.cj0 + r) += lx * ff;
            }
        if (const SideNb n = bottomNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + p.ci0 + r, NG + n.cell)) continue;
                const Cons ff =
                    Real(0.5) * (sf.Fy[p.grid.idx(NG + 2 * r, NG - 1)] +
                                 sf.Fy[p.grid.idx(NG + 2 * r + 1, NG - 1)]);
                coarse.at(NG + p.ci0 + r, NG + n.cell) -= ly * ff;
            }
        if (const SideNb n = topNb_(p); n.ok && !covered(p.bi, n.block))
            for (int r = 0; r < bc; ++r) {
                if (solCoarse_(NG + p.ci0 + r, NG + n.cell)) continue;
                const Cons ff =
                    Real(0.5) *
                    (sf.Fy[p.grid.idx(NG + 2 * r, NG + nf - 1)] +
                     sf.Fy[p.grid.idx(NG + 2 * r + 1, NG + nf - 1)]);
                coarse.at(NG + p.ci0 + r, NG + n.cell) += ly * ff;
            }
    }

    // Average each 2x2 fine block onto its covered coarse cell. With an
    // immersed mask: solid coarse cells stay frozen, and only the FLUID
    // fine children contribute (solid children carry frozen data). Without
    // a mask this is the plain 0.25*sum.
    void restrictFine_() {
        if (cutCellOn_()) { restrictCut_(); return; }
        for (const Patch& p : patches) {
            const std::uint8_t* ps = psolid_(p);
            for (int b = 0; b < cfg_.blockC; ++b)
                for (int a = 0; a < cfg_.blockC; ++a) {
                    const int ci = NG + p.ci0 + a, cj = NG + p.cj0 + b;
                    if (solCoarse_(ci, cj)) continue; // solid: frozen
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    if (!ps) {
                        coarse.at(ci, cj) =
                            Real(0.25) * (p.grid.at(fi, fj) +
                                          p.grid.at(fi + 1, fj) +
                                          p.grid.at(fi, fj + 1) +
                                          p.grid.at(fi + 1, fj + 1));
                        continue;
                    }
                    Cons sum{0, 0, 0, 0};
                    int n = 0;
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            if (ps[p.grid.idx(fi + di, fj + dj)]) continue;
                            sum += p.grid.at(fi + di, fj + dj);
                            ++n;
                        }
                    if (n > 0) coarse.at(ci, cj) = (Real(1) / n) * sum;
                }
        }
    }

    AmrConfig cfg_;
    int nbx_, nby_;
    std::vector<int> blockOf_; // block -> patch index, -1 if unrefined
    Scratch2D scratchC_, scratchF_;
    std::vector<Cons> coarseOld_; // coarse t^n copy (subcycled ghosts)
    std::vector<std::uint8_t> coarseSolidMask_; // 1 = solid (empty = none)
    cutcell::Geometry coarseGeo_; // cut-cell moments (empty unless cfg.cutCell)
    std::vector<Cons> cutFxC_, cutFyC_;          // base extensive fluxes
    std::vector<std::vector<Cons>> pDc_, pD_;    // per-patch divergences
    std::vector<std::vector<Cons>> pFx_, pFy_;   // per-patch extensive fluxes
    std::vector<std::vector<PrimGrad>> pGrad_;   // per-patch LSQ gradients (O2)
    std::vector<Cons> coarseO2Old_;              // base U^n (RK2 combine)
    std::vector<std::vector<Cons>> pOld_;        // per-patch U^n (RK2 combine)
    int stepCount_ = 0;
};

} // namespace mm

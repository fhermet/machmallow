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
#include "numerics/Limiter.hpp"
#include "solver/Muscl2D.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mm {

struct AmrConfig {
    int blockC = 8;             // block size in coarse cells
    Real tagThreshold = Real(0.03); // relative density gradient
    int regridEvery = 4;
    bool reflux = true;         // off only to demonstrate the leak
    bool subcycle = false;      // coarse at dt, fine at 2 x dt/2
};

class Amr2 {
public:
    struct Patch {
        int bi, bj;   // block coordinates
        int ci0, cj0; // coarse-cell origin of the block
        Grid grid;    // fine grid: 2*blockC cells square + ghosts
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
        fillPhysicalGhosts(coarse, 0); // regrid prolongation reads ghosts
        regrid();
        for (Patch& p : patches)
            for (int j = NG; j < NG + p.grid.ny; ++j)
                for (int i = NG; i < NG + p.grid.nx; ++i)
                    p.grid.at(i, j) = ic(p.grid.xc(i), p.grid.yc(j));
        restrictFine_();
    }

    Real maxStableDtAll(Real cfl) const {
        Real dt = maxStableDt(coarse, cfl);
        for (const Patch& p : patches) {
            Real dtF = maxStableDt(p.grid, cfl);
            if (cfg_.subcycle) dtF *= 2; // fine takes two half steps
            dt = std::min(dt, dtF);
        }
        return dt;
    }

    unsigned domainSides(const Patch& p) const {
        return (p.bi == 0 ? SideLeft : 0u) |
               (p.bi == nbx_ - 1 ? SideRight : 0u) |
               (p.bj == 0 ? SideBottom : 0u) |
               (p.bj == nby_ - 1 ? SideTop : 0u);
    }

    void step(Real dt, double t) {
        if (cfg_.subcycle && !patches.empty())
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

    void regrid() {
        const int nx = coarse.nx, ny = coarse.ny, bc = cfg_.blockC;

        // 1. Tag on relative density gradient (clamped central diffs).
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
                tag[std::size_t(j) * nx + i] =
                    std::max(ex, ey) / r0 > cfg_.tagThreshold;
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
    void fillAllPatchGhosts_(double t, Real theta) {
        for (Patch& p : patches) {
            fillPatchGhosts_(p, theta);
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides(p))
                    fillPatchPhysical(p.grid, t, sides);
        }
    }

    void stepSingleRate_(Real dt, double t) {
        fillPhysicalGhosts(coarse, t);
        fillAllPatchGhosts_(t, Real(-1));

        step2D(coarse, dt, scratchC_); // keeps coarse fluxes for refluxing
        for (Patch& p : patches) {
            step2D(p.grid, dt, scratchF_);
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

        step2D(coarse, dtC, scratchC_);
        for (Patch& p : patches) {
            if (cfg_.reflux) refluxCoarse_(p, dtC);
            step2D(p.grid, dtF, scratchF_);
            if (cfg_.reflux) refluxFine_(p, dtF, scratchF_);
        }

        // Substep 2: sibling copies are current; coarse prolongation uses
        // the half-time blend of t^n and t^{n+1}.
        fillAllPatchGhosts_(t + dtF, Real(0.5));
        for (Patch& p : patches) {
            step2D(p.grid, dtF, scratchF_);
            if (cfg_.reflux) refluxFine_(p, dtF, scratchF_);
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
        const Cons dqx = limitedSlope(coarseAt_(I - 1, J, theta), q0,
                                      coarseAt_(I + 1, J, theta));
        const Cons dqy = limitedSlope(coarseAt_(I, J - 1, theta), q0,
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
                const int gfi = 2 * p.ci0 + (i - NG);
                const int gfj = 2 * p.cj0 + (j - NG);

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
    void refluxCoarse_(const Patch& p, Real dt) {
        const int bc = cfg_.blockC;
        const Real lx = dt / coarse.dx, ly = dt / coarse.dy;

        if (p.bi > 0 && !covered(p.bi - 1, p.bj)) // left side
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 - 1, NG + p.cj0 + r) +=
                    lx * scratchC_.Fx[coarse.idx(NG + p.ci0 - 1,
                                                 NG + p.cj0 + r)];
        if (p.bi < nbx_ - 1 && !covered(p.bi + 1, p.bj)) // right side
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 + bc, NG + p.cj0 + r) -=
                    lx * scratchC_.Fx[coarse.idx(NG + p.ci0 + bc - 1,
                                                 NG + p.cj0 + r)];
        if (p.bj > 0 && !covered(p.bi, p.bj - 1)) // bottom side
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 + r, NG + p.cj0 - 1) +=
                    ly * scratchC_.Fy[coarse.idx(NG + p.ci0 + r,
                                                 NG + p.cj0 - 1)];
        if (p.bj < nby_ - 1 && !covered(p.bi, p.bj + 1)) // top side
            for (int r = 0; r < bc; ++r)
                coarse.at(NG + p.ci0 + r, NG + p.cj0 + bc) -=
                    ly * scratchC_.Fy[coarse.idx(NG + p.ci0 + r,
                                                 NG + p.cj0 + bc - 1)];
    }

    void refluxFine_(const Patch& p, Real dt, const Scratch2D& sf) {
        const int bc = cfg_.blockC, nf = 2 * bc;
        const Real lx = dt / coarse.dx, ly = dt / coarse.dy;

        if (p.bi > 0 && !covered(p.bi - 1, p.bj)) // left side
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (sf.Fx[p.grid.idx(NG - 1, NG + 2 * r)] +
                                 sf.Fx[p.grid.idx(NG - 1, NG + 2 * r + 1)]);
                coarse.at(NG + p.ci0 - 1, NG + p.cj0 + r) += Real(-1) * lx * ff;
            }
        if (p.bi < nbx_ - 1 && !covered(p.bi + 1, p.bj)) // right side
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) *
                    (sf.Fx[p.grid.idx(NG + nf - 1, NG + 2 * r)] +
                     sf.Fx[p.grid.idx(NG + nf - 1, NG + 2 * r + 1)]);
                coarse.at(NG + p.ci0 + bc, NG + p.cj0 + r) += lx * ff;
            }
        if (p.bj > 0 && !covered(p.bi, p.bj - 1)) // bottom side
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) * (sf.Fy[p.grid.idx(NG + 2 * r, NG - 1)] +
                                 sf.Fy[p.grid.idx(NG + 2 * r + 1, NG - 1)]);
                coarse.at(NG + p.ci0 + r, NG + p.cj0 - 1) += Real(-1) * ly * ff;
            }
        if (p.bj < nby_ - 1 && !covered(p.bi, p.bj + 1)) // top side
            for (int r = 0; r < bc; ++r) {
                const Cons ff =
                    Real(0.5) *
                    (sf.Fy[p.grid.idx(NG + 2 * r, NG + nf - 1)] +
                     sf.Fy[p.grid.idx(NG + 2 * r + 1, NG + nf - 1)]);
                coarse.at(NG + p.ci0 + r, NG + p.cj0 + bc) += ly * ff;
            }
    }

    // Average each 2x2 fine block onto its covered coarse cell.
    void restrictFine_() {
        for (const Patch& p : patches)
            for (int b = 0; b < cfg_.blockC; ++b)
                for (int a = 0; a < cfg_.blockC; ++a) {
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    const Cons sum = p.grid.at(fi, fj) +
                                     p.grid.at(fi + 1, fj) +
                                     p.grid.at(fi, fj + 1) +
                                     p.grid.at(fi + 1, fj + 1);
                    coarse.at(NG + p.ci0 + a, NG + p.cj0 + b) =
                        Real(0.25) * sum;
                }
    }

    AmrConfig cfg_;
    int nbx_, nby_;
    std::vector<int> blockOf_; // block -> patch index, -1 if unrefined
    Scratch2D scratchC_, scratchF_;
    std::vector<Cons> coarseOld_; // coarse t^n copy (subcycled ghosts)
    int stepCount_ = 0;
};

} // namespace mm

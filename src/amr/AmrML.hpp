#pragma once

// Multi-level block-structured AMR (CPU reference), arbitrary depth.
// Level 0 is one uniform grid; a patch at level l >= 1 refines one block
// of blockC level-(l-1) cells into 2*blockC cells (ratio 2), so every
// patch has the same shape regardless of depth.
//
// Time integration is the recursive Berger-Colella scheme:
//   advanceTree(l, dt): save level-l state, step level l, back the
//   level-l fluxes out of the uncovered cells at every (l, l+1)
//   interface, then advance level l+1 twice with dt/2 (ghosts
//   theta-blended in time from level l), applying its fine fluxes after
//   each substep, and finally restrict l+1 onto l.
// With cfg.subcycle = false the child advances once with the full dt.
//
// Proper nesting is enforced at regrid time: a wanted level-(l+1) block
// forces the level-l blocks covering its region grown by one level-l
// cell, so a fine patch never touches a level skip.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/Limiter.hpp"
#include "solver/Muscl2D.hpp"
#include "solver/Muscl2DSpecies.hpp"

#include "amr/Amr2.hpp" // AmrConfig

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mm {

class AmrML {
public:
    struct Patch {
        int bi, bj;   // block coordinates (in level l-1 cells / blockC)
        int ci0, cj0; // block origin in level-(l-1) cells
        Grid grid;    // own-level data: 2*blockC square + ghosts
        std::vector<Cons> Fx, Fy; // fluxes of this patch's last step
        std::vector<Cons> old;    // own state at parent-substep start
        // two-gas fields (allocated only when cfg.species)
        std::vector<Real> phi, Gmf, oldPhi, oldGm, Fpx, Fpy;
    };

    struct Level { // level l >= 1; stored at lvls_[l-1]
        int nbx = 0, nby = 0;  // block grid dims
        std::vector<int> blockOf;
        std::vector<Patch> patches;
    };

    Grid base;

    std::function<void(Grid&, double)> fillPhysicalGhosts;
    std::function<void(Grid&, double, unsigned)> fillPatchPhysical;

    AmrML(int nx, int ny, Real x0, Real y0, Real lx, Real ly,
          AmrConfig cfg)
        : base(nx, ny, x0, y0, lx, ly), cfg_(cfg), nf_(2 * cfg.blockC) {
        assert(nx % cfg.blockC == 0 && ny % cfg.blockC == 0);
        assert(cfg.maxLevels >= 1);
        gas_ = GasPair{cfg.gamma1, cfg.gamma2};
        if (cfg_.species) {
            basePhi_.assign(base.q.size(), 0);
            baseGm_.assign(base.q.size(), gas_.Gamma(0));
        }
        lvls_.resize(cfg.maxLevels - 1);
        for (int l = 1; l < cfg_.maxLevels; ++l) {
            Level& L = lvls_[l - 1];
            L.nbx = (nx << (l - 1)) / cfg.blockC;
            L.nby = (ny << (l - 1)) / cfg.blockC;
            L.blockOf.assign(std::size_t(L.nbx) * L.nby, -1);
        }
    }

    int numLevels() const { return cfg_.maxLevels; }
    bool species() const { return cfg_.species; }
    std::vector<Real>& basePhi() { return basePhi_; }
    std::vector<Real>& baseGm() { return baseGm_; }
    const GasPair& gas() const { return gas_; }
    int fineCells() const { return nf_; }
    const Level& level(int l) const { return lvls_[l - 1]; }

    bool covered(int l, int bi, int bj) const { // block of level l (>=1)
        const Level& L = lvls_[l - 1];
        return L.blockOf[std::size_t(bj) * L.nbx + bi] >= 0;
    }

    template <class IC>
    void init(IC ic) {
        init(ic, [](Real, Real) { return Real(0); });
    }

    // Two-gas init: icY gives the mass fraction Y of gas 2.
    template <class IC, class ICY>
    void init(IC ic, ICY icY) {
        for (int j = NG; j < NG + base.ny; ++j)
            for (int i = NG; i < NG + base.nx; ++i) {
                base.at(i, j) = ic(base.xc(i), base.yc(j));
                if (cfg_.species) {
                    const std::size_t id = base.idx(i, j);
                    const Real Y = icY(base.xc(i), base.yc(j));
                    basePhi_[id] = base.q[id].rho * Y;
                    baseGm_[id] = gas_.Gamma(Y);
                }
            }
        fillPhysicalGhosts(base, 0);
        scalarPhysical_(base, basePhi_, baseGm_);
        // Repeated regrids let each new level tag from real (IC) data.
        for (int pass = 1; pass < cfg_.maxLevels; ++pass) {
            regrid();
            for (Level& L : lvls_)
                for (Patch& p : L.patches)
                    for (int j = NG; j < NG + nf_; ++j)
                        for (int i = NG; i < NG + nf_; ++i) {
                            p.grid.at(i, j) =
                                ic(p.grid.xc(i), p.grid.yc(j));
                            if (cfg_.species) {
                                const std::size_t id = p.grid.idx(i, j);
                                const Real Y =
                                    icY(p.grid.xc(i), p.grid.yc(j));
                                p.phi[id] = p.grid.q[id].rho * Y;
                                p.Gmf[id] = gas_.Gamma(Y);
                            }
                        }
            for (int l = cfg_.maxLevels - 1; l >= 1; --l)
                restrictLevel_(l);
        }
    }

    Real maxStableDtAll(Real cfl) const {
        Real dt = cfg_.species ? maxStableDtY(base, baseGm_, cfl)
                               : maxStableDt(base, cfl, cfg_.mu);
        for (std::size_t k = 0; k < lvls_.size(); ++k)
            for (const Patch& p : lvls_[k].patches) {
                Real dtl = cfg_.species
                    ? maxStableDtY(p.grid, p.Gmf, cfl)
                    : maxStableDt(p.grid, cfl, cfg_.mu);
                if (cfg_.subcycle) dtl *= Real(1 << (k + 1));
                dt = std::min(dt, dtl);
            }
        return dt;
    }

    void step(Real dt, double t) {
        fillPhysicalGhosts(base, t);
        if (cfg_.species) scalarPhysical_(base, basePhi_, baseGm_);
        advanceTree_(0, dt, t);
    }

    double totalMass() const {
        double m = 0;
        const double a0 = double(base.dx) * base.dy;
        for (int j = 0; j < base.ny; ++j)
            for (int i = 0; i < base.nx; ++i)
                if (cfg_.maxLevels < 2 ||
                    !covered(1, i / cfg_.blockC, j / cfg_.blockC))
                    m += double(base.at(NG + i, NG + j).rho) * a0;
        for (std::size_t k = 0; k < lvls_.size(); ++k) {
            const int l = int(k) + 1;
            for (const Patch& p : lvls_[k].patches) {
                const double al = double(p.grid.dx) * p.grid.dy;
                for (int j = 0; j < nf_; ++j)
                    for (int i = 0; i < nf_; ++i) {
                        // own-level cell index of this patch cell
                        const int gi = 2 * p.ci0 + i;
                        const int gj = 2 * p.cj0 + j;
                        if (l < cfg_.maxLevels - 1 &&
                            covered(l + 1, gi / cfg_.blockC,
                                    gj / cfg_.blockC))
                            continue;
                        m += double(p.grid.at(NG + i, NG + j).rho) * al;
                    }
            }
        }
        return m;
    }

    std::size_t cellCount() const {
        std::size_t n = std::size_t(base.nx) * base.ny;
        for (const Level& L : lvls_)
            n += L.patches.size() * std::size_t(nf_) * nf_;
        return n;
    }

    std::size_t patchCount(int l) const {
        return lvls_[l - 1].patches.size();
    }

    // Nesting invariant: every patch's parent region (grown by one
    // parent cell) must be representable at the parent level.
    bool checkNesting() const {
        for (std::size_t k = 1; k < lvls_.size(); ++k)
            for (const Patch& p : lvls_[k].patches) {
                // parent-level cells [ci0-1, ci0+blockC] must be covered
                for (int dj = -1; dj <= cfg_.blockC; dj += 1)
                    for (int di = -1; di <= cfg_.blockC; di += 1) {
                        int cg = p.ci0 + di, cgj = p.cj0 + dj;
                        const int nxp = base.nx << k, nyp = base.ny << k;
                        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
                        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
                        if (cg < 0 || cg >= nxp || cgj < 0 || cgj >= nyp)
                            continue; // physical boundary
                        if (!covered(int(k), cg / nf_, cgj / nf_))
                            return false;
                    }
            }
        return true;
    }

    void regrid() { regridFrom_(1); }

    // ---- composite views (writers) ------------------------------------
    GridRef coarseRef() const {
        return GridRef{base.nx, base.ny, base.x0,
                       base.y0, base.dx, base.dy,
                       const_cast<Cons*>(base.q.data())};
    }
    GridRef patchRef(const Patch& p) const {
        return GridRef{p.grid.nx, p.grid.ny, p.grid.x0,
                       p.grid.y0, p.grid.dx, p.grid.dy,
                       const_cast<Cons*>(p.grid.q.data())};
    }
    GridRef patchRef(int /*level*/, const Patch& p) const {
        return patchRef(p); // grids are self-describing on the CPU side
    }

private:
    // ---- geometry helpers ----------------------------------------------
    int nxAt_(int l) const { return base.nx << l; }
    int nyAt_(int l) const { return base.ny << l; }

    const Patch* ownerAt_(int l, int cg, int cgj) const { // l >= 1 cells
        const Level& L = lvls_[l - 1];
        const int pi =
            L.blockOf[std::size_t(cgj / nf_) * L.nbx + cg / nf_];
        return pi >= 0 ? &L.patches[pi] : nullptr;
    }
    Patch* ownerAt_(int l, int cg, int cgj) {
        return const_cast<Patch*>(
            const_cast<const AmrML*>(this)->ownerAt_(l, cg, cgj));
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

    // ---- tagging --------------------------------------------------------
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
        // mark level-(l+1) blocks holding cells [ci±2, cj±2] (wrapped)
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
        const auto at = [&](int a, int b) -> const Cons& {
            return base.at(NG + a, NG + b);
        };
        for (int j = 0; j < base.ny; ++j)
            for (int i = 0; i < base.nx; ++i) {
                const int ip = std::min(i + 1, base.nx - 1),
                          im = std::max(i - 1, 0);
                const int jp = std::min(j + 1, base.ny - 1),
                          jm = std::max(j - 1, 0);
                if (tagCell_(at, i, j, ip, im, jp, jm))
                    markDilated_(want, lv, base.nx, base.ny, i, j);
            }
    }

    void tagLevel_(int l, std::vector<std::uint8_t>& want) const {
        // tags from level-l patch interiors, in own-level global cells
        const Level& kid = lvls_[l]; // level l+1 bookkeeping
        const int nx = nxAt_(l), ny = nyAt_(l);
        for (const Patch& p : lvls_[l - 1].patches) {
            const auto at = [&](int a, int b) -> const Cons& {
                return p.grid.at(NG + a, NG + b);
            };
            for (int j = 0; j < nf_; ++j)
                for (int i = 0; i < nf_; ++i) {
                    // the +-1 stencil reads the ghost ring at patch
                    // edges — clamping there would blind the tagging to
                    // gradients straddling a patch seam
                    if (tagCell_(at, i, j, i + 1, i - 1, j + 1, j - 1))
                        markDilated_(want, kid, nx, ny, 2 * p.ci0 + i,
                                     2 * p.cj0 + j);
                }
        }
    }

    // ---- patch construction / prolongation -------------------------------
    // theta < 0: current state (regrid-time prolongation); theta = 0:
    // substep-start copy; in between: time blend.
    Cons blend_(const Cons& cur, const Cons* oldArr, std::size_t idx,
                Real theta) const {
        if (theta < 0 || theta >= 1 || oldArr == nullptr) return cur;
        if (theta <= 0) return oldArr[idx];
        return (1 - theta) * oldArr[idx] + theta * cur;
    }

    // Value of level-(l-1) cell (cg, cgj) for prolongation, theta-blended
    // between the saved substep-start state and the current one. `home`
    // is the patch whose block contains the requesting child (fallback
    // storage whose ghosts always cover the stencil).
    Cons parentVal_(int l, const Patch* home, int cg, int cgj,
                    Real theta) const {
        // Periodic wrap at the PARENT level too: stencil neighbours of a
        // wrapped coordinate cross the seam again (cg = 0 -> cg-1 = -1),
        // and the home fallback would index wildly out of range there.
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
        if (l == 1) {
            const std::size_t id = base.idx(NG + cg, NG + cgj);
            return blend_(base.q[id],
                          baseOld_.empty() ? nullptr : baseOld_.data(),
                          id, theta);
        }
        const Patch* Q = (cg >= 0 && cg < nxp && cgj >= 0 && cgj < nyp)
                             ? ownerAt_(l - 1, cg, cgj)
                             : nullptr;
        if (Q == nullptr) Q = home;
        const std::size_t id =
            Q->grid.idx(NG + cg - 2 * Q->ci0, NG + cgj - 2 * Q->cj0);
        return blend_(Q->grid.q[id],
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

    Patch makePatch_(int l, int bi, int bj) {
        const int bC = cfg_.blockC;
        const int ci0 = bi * bC, cj0 = bj * bC;
        const Real dxp = base.dx / Real(1 << (l - 1));
        const Real dyp = base.dy / Real(1 << (l - 1));
        Patch p{bi,
                bj,
                ci0,
                cj0,
                Grid(nf_, nf_, base.x0 + ci0 * dxp, base.y0 + cj0 * dyp,
                     bC * dxp, bC * dyp),
                {},
                {},
                {}};
        const Patch* home = l >= 2 ? ownerAt_(l - 1, ci0, cj0) : nullptr;
        if (cfg_.species) {
            p.phi.assign(p.grid.q.size(), 0);
            p.Gmf.assign(p.grid.q.size(), gas_.Gamma(0));
        }
        for (int j = NG; j < NG + nf_; ++j)
            for (int i = NG; i < NG + nf_; ++i) {
                const int gfi = 2 * ci0 + (i - NG);
                const int gfj = 2 * cj0 + (j - NG);
                p.grid.at(i, j) = prolong_(l, home, gfi / 2, gfj / 2,
                                           gfi & 1, gfj & 1, Real(-1));
                if (cfg_.species) {
                    const std::size_t id = p.grid.idx(i, j);
                    p.phi[id] = scalarProlong_(l, home, true, gfi / 2,
                                               gfj / 2, gfi & 1, gfj & 1,
                                               Real(-1));
                    p.Gmf[id] = scalarProlong_(l, home, false, gfi / 2,
                                               gfj / 2, gfi & 1, gfj & 1,
                                               Real(-1));
                }
            }
        return p;
    }

    // Scalar (phi / Gamma) analogues of parentVal_/prolong_.
    Real scalarParentVal_(int l, const Patch* home, bool isPhi, int cg,
                          int cgj, Real theta) const {
        const int nxp = nxAt_(l - 1), nyp = nyAt_(l - 1);
        if (cfg_.periodicX) cg = (cg % nxp + nxp) % nxp;
        if (cfg_.periodicY) cgj = (cgj % nyp + nyp) % nyp;
        const auto pick = [&](const std::vector<Real>& cur,
                              const std::vector<Real>& old,
                              std::size_t id) {
            if (theta < 0 || theta >= 1 || old.empty()) return cur[id];
            if (theta <= 0) return old[id];
            return (1 - theta) * old[id] + theta * cur[id];
        };
        if (l == 1) {
            const std::size_t id = base.idx(NG + cg, NG + cgj);
            return pick(isPhi ? basePhi_ : baseGm_,
                        isPhi ? baseOldPhi_ : baseOldGm_, id);
        }
        const Patch* Q = (cg >= 0 && cg < nxp && cgj >= 0 && cgj < nyp)
                             ? ownerAt_(l - 1, cg, cgj)
                             : nullptr;
        if (Q == nullptr) Q = home;
        const std::size_t id =
            Q->grid.idx(NG + cg - 2 * Q->ci0, NG + cgj - 2 * Q->cj0);
        return pick(isPhi ? Q->phi : Q->Gmf,
                    isPhi ? Q->oldPhi : Q->oldGm, id);
    }

    Real scalarProlong_(int l, const Patch* home, bool isPhi, int cg,
                        int cgj, int ox, int oy, Real theta) const {
        const Real q0 = scalarParentVal_(l, home, isPhi, cg, cgj, theta);
        const Real dx = mcSlope(
            q0 - scalarParentVal_(l, home, isPhi, cg - 1, cgj, theta),
            scalarParentVal_(l, home, isPhi, cg + 1, cgj, theta) - q0);
        const Real dy = mcSlope(
            q0 - scalarParentVal_(l, home, isPhi, cg, cgj - 1, theta),
            scalarParentVal_(l, home, isPhi, cg, cgj + 1, theta) - q0);
        return q0 + (ox ? Real(0.25) : Real(-0.25)) * dx +
               (oy ? Real(0.25) : Real(-0.25)) * dy;
    }

    // Transmissive physical ghosts for the scalar fields, restricted to
    // the requested domain sides (interior sides keep their sibling /
    // prolongation values).
    static void scalarPhysicalSides_(const Grid& g, std::vector<Real>& a,
                                     std::vector<Real>& b,
                                     unsigned sides) {
        for (auto* f : {&a, &b}) {
            if (f->empty()) continue;
            for (int j = 0; j < g.toty(); ++j)
                for (int k = 0; k < NG; ++k) {
                    if (sides & SideLeft)
                        (*f)[g.idx(k, j)] = (*f)[g.idx(NG, j)];
                    if (sides & SideRight)
                        (*f)[g.idx(NG + g.nx + k, j)] =
                            (*f)[g.idx(NG + g.nx - 1, j)];
                }
            for (int i = 0; i < g.totx(); ++i)
                for (int k = 0; k < NG; ++k) {
                    if (sides & SideBottom)
                        (*f)[g.idx(i, k)] = (*f)[g.idx(i, NG)];
                    if (sides & SideTop)
                        (*f)[g.idx(i, NG + g.ny + k)] =
                            (*f)[g.idx(i, NG + g.ny - 1)];
                }
        }
    }
    static void scalarPhysical_(const Grid& g, std::vector<Real>& a,
                                std::vector<Real>& b) {
        scalarPhysicalSides_(g, a, b,
                             SideLeft | SideRight | SideBottom | SideTop);
    }

    // ---- ghosts -----------------------------------------------------------
    void fillLevelGhosts_(int l, double t, Real theta) {
        Level& lv = lvls_[l - 1];
        const int nxl = nxAt_(l), nyl = nyAt_(l);
        for (Patch& p : lv.patches) {
            const Patch* home =
                l >= 2 ? ownerAt_(l - 1, p.ci0, p.cj0) : nullptr;
            const auto fillCell = [&](int i, int j) {
                int gfi = 2 * p.ci0 + (i - NG);
                int gfj = 2 * p.cj0 + (j - NG);
                if (cfg_.periodicX) gfi = (gfi % nxl + nxl) % nxl;
                if (cfg_.periodicY) gfj = (gfj % nyl + nyl) % nyl;
                if (gfi >= 0 && gfi < nxl && gfj >= 0 && gfj < nyl)
                    if (const Patch* s = ownerAt_(l, gfi, gfj)) {
                        const std::size_t src = s->grid.idx(
                            NG + gfi - 2 * s->ci0, NG + gfj - 2 * s->cj0);
                        p.grid.q[p.grid.idx(i, j)] = s->grid.q[src];
                        if (cfg_.species) {
                            p.phi[p.grid.idx(i, j)] = s->phi[src];
                            p.Gmf[p.grid.idx(i, j)] = s->Gmf[src];
                        }
                        return;
                    }
                const int cg = (gfi >= 0) ? gfi / 2 : (gfi - 1) / 2;
                const int cgj = (gfj >= 0) ? gfj / 2 : (gfj - 1) / 2;
                p.grid.at(i, j) = prolong_(l, home, cg, cgj,
                                           gfi - 2 * cg, gfj - 2 * cgj,
                                           theta);
                if (cfg_.species) {
                    const std::size_t id = p.grid.idx(i, j);
                    p.phi[id] =
                        scalarProlong_(l, home, true, cg, cgj,
                                       gfi - 2 * cg, gfj - 2 * cgj, theta);
                    p.Gmf[id] =
                        scalarProlong_(l, home, false, cg, cgj,
                                       gfi - 2 * cg, gfj - 2 * cgj, theta);
                }
            };
            for (int j = 0; j < NG; ++j)
                for (int i = 0; i < p.grid.totx(); ++i) fillCell(i, j);
            for (int j = NG + nf_; j < p.grid.toty(); ++j)
                for (int i = 0; i < p.grid.totx(); ++i) fillCell(i, j);
            for (int j = NG; j < NG + nf_; ++j) {
                for (int i = 0; i < NG; ++i) fillCell(i, j);
                for (int i = NG + nf_; i < p.grid.totx(); ++i)
                    fillCell(i, j);
            }
            if (fillPatchPhysical)
                if (const unsigned sides = domainSides_(l, p))
                    fillPatchPhysical(p.grid, t, sides);
            if (cfg_.species)
                if (const unsigned sides = domainSides_(l, p))
                    scalarPhysicalSides_(p.grid, p.phi, p.Gmf, sides);
        }
    }

    // ---- time advance -----------------------------------------------------
    void saveOld_(int l) {
        if (l == 0) {
            baseOld_ = base.q;
            if (cfg_.species) {
                baseOldPhi_ = basePhi_;
                baseOldGm_ = baseGm_;
            }
            return;
        }
        for (Patch& p : lvls_[l - 1].patches) {
            p.old = p.grid.q;
            if (cfg_.species) {
                p.oldPhi = p.phi;
                p.oldGm = p.Gmf;
            }
        }
    }

    void stepLevel_(int l, Real dt) {
        if (l == 0) {
            if (cfg_.species)
                step2DY(base, basePhi_, baseGm_, dt, scratchYB_, gas_);
            else
                step2D(base, dt, scratchB_, cfg_.mu, cfg_.gx, cfg_.gy);
            return;
        }
        for (Patch& p : lvls_[l - 1].patches) {
            if (cfg_.species) {
                step2DY(p.grid, p.phi, p.Gmf, dt, scratchY_, gas_);
                p.Fx = scratchY_.Fx;
                p.Fy = scratchY_.Fy;
                p.Fpx = scratchY_.Fpx;
                p.Fpy = scratchY_.Fpy;
            } else {
                step2D(p.grid, dt, scratch_, cfg_.mu, cfg_.gx, cfg_.gy);
                p.Fx = scratch_.Fx;
                p.Fy = scratch_.Fy;
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
        // Per-level regrid cadence: children of level l are remeshed
        // every regridEvery steps OF LEVEL l, so the dilation buffer is
        // scale-invariant (features advance the same number of own-level
        // cells per own-level regrid period at every depth).
        if (l + 1 < cfg_.maxLevels &&
            ++stepCounts_[l] % cfg_.regridEvery == 0) {
            if (l == 0) fillPhysicalGhosts(base, t + dt);
            regridFrom_(l + 1);
        }
    }

    // ---- coarse-fine coupling ----------------------------------------------
    struct ParentCell { // an (l-1)-level cell with its flux arrays
        Grid* g;
        const std::vector<Cons>*Fx, *Fy;
        int li, lj; // local indices (ghost offset included)
        // two-gas fields (null when species disabled)
        std::vector<Real>*phi = nullptr, *Gm = nullptr;
        const std::vector<Real>*Fpx = nullptr, *Fpy = nullptr;
    };
    ParentCell parentCell_(int lp, int cg, int cgj) {
        if (lp == 0) {
            ParentCell pc{&base,
                          cfg_.species ? &scratchYB_.Fx : &scratchB_.Fx,
                          cfg_.species ? &scratchYB_.Fy : &scratchB_.Fy,
                          NG + cg,
                          NG + cgj};
            if (cfg_.species) {
                pc.phi = &basePhi_;
                pc.Gm = &baseGm_;
                pc.Fpx = &scratchYB_.Fpx;
                pc.Fpy = &scratchYB_.Fpy;
            }
            return pc;
        }
        Patch* Q = ownerAt_(lp, cg, cgj);
        assert(Q != nullptr); // guaranteed by nesting
        ParentCell pc{&Q->grid, &Q->Fx, &Q->Fy, NG + cg - 2 * Q->ci0,
                      NG + cgj - 2 * Q->cj0};
        if (cfg_.species) {
            pc.phi = &Q->phi;
            pc.Gm = &Q->Gmf;
            pc.Fpx = &Q->Fpx;
            pc.Fpy = &Q->Fpy;
        }
        return pc;
    }

    struct SideNb {
        bool ok;
        int blockBi, blockBj; // neighbour block at child level
        int cg, cgj;          // corrected (l-1) cell start (wrapped)
    };
    // dir: 0 left, 1 right, 2 bottom, 3 top
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

    // Back the parent-level flux out of every uncovered neighbour cell
    // along the coarse-fine interfaces of level lc.
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
                    const Real lam =
                        dtParent / (dir < 2 ? pc.g->dx : pc.g->dy);
                    const std::size_t pid =
                        std::size_t(pc.lj) * pc.g->totx() + pc.li;
                    if (dir == 0) { // its right face
                        pc.g->at(pc.li, pc.lj) +=
                            lam * (*pc.Fx)[pc.g->idx(pc.li, pc.lj)];
                        if (pc.phi)
                            (*pc.phi)[pid] +=
                                lam *
                                (*pc.Fpx)[pc.g->idx(pc.li, pc.lj)];
                    } else if (dir == 1) { // its left face
                        pc.g->at(pc.li, pc.lj) -=
                            lam * (*pc.Fx)[pc.g->idx(pc.li - 1, pc.lj)];
                        if (pc.phi)
                            (*pc.phi)[pid] -=
                                lam *
                                (*pc.Fpx)[pc.g->idx(pc.li - 1, pc.lj)];
                    } else if (dir == 2) { // its top face
                        pc.g->at(pc.li, pc.lj) +=
                            lam * (*pc.Fy)[pc.g->idx(pc.li, pc.lj)];
                        if (pc.phi)
                            (*pc.phi)[pid] +=
                                lam *
                                (*pc.Fpy)[pc.g->idx(pc.li, pc.lj)];
                    } else { // its bottom face
                        pc.g->at(pc.li, pc.lj) -=
                            lam * (*pc.Fy)[pc.g->idx(pc.li, pc.lj - 1)];
                        if (pc.phi)
                            (*pc.phi)[pid] -=
                                lam *
                                (*pc.Fpy)[pc.g->idx(pc.li, pc.lj - 1)];
                    }
                }
            }
    }

    // Apply one substep's area-averaged fine fluxes to the same cells.
    void refluxFineApply_(int lc, Real dtChild) {
        const int bC = cfg_.blockC;
        for (const Patch& p : lvls_[lc - 1].patches)
            for (int dir = 0; dir < 4; ++dir) {
                const SideNb n = sideNb_(lc, p, dir);
                if (!n.ok || covered(lc, n.blockBi, n.blockBj)) continue;
                for (int r = 0; r < bC; ++r) {
                    const int cg = (dir < 2) ? n.cg : n.cg + r;
                    const int cgj = (dir < 2) ? n.cgj + r : n.cgj;
                    ParentCell pc = parentCell_(lc - 1, cg, cgj);
                    const Real lam =
                        dtChild / (dir < 2 ? pc.g->dx : pc.g->dy);
                    Cons ff;
                    Real fp = 0;
                    const auto take = [&](const std::vector<Cons>& F,
                                          const std::vector<Real>& Fp,
                                          std::size_t a, std::size_t b) {
                        ff = Real(0.5) * (F[a] + F[b]);
                        if (cfg_.species)
                            fp = Real(0.5) * (Fp[a] + Fp[b]);
                    };
                    if (dir == 0)
                        take(p.Fx, p.Fpx,
                             p.grid.idx(NG - 1, NG + 2 * r),
                             p.grid.idx(NG - 1, NG + 2 * r + 1));
                    else if (dir == 1)
                        take(p.Fx, p.Fpx,
                             p.grid.idx(NG + nf_ - 1, NG + 2 * r),
                             p.grid.idx(NG + nf_ - 1, NG + 2 * r + 1));
                    else if (dir == 2)
                        take(p.Fy, p.Fpy,
                             p.grid.idx(NG + 2 * r, NG - 1),
                             p.grid.idx(NG + 2 * r + 1, NG - 1));
                    else
                        take(p.Fy, p.Fpy,
                             p.grid.idx(NG + 2 * r, NG + nf_ - 1),
                             p.grid.idx(NG + 2 * r + 1, NG + nf_ - 1));
                    const std::size_t pid =
                        std::size_t(pc.lj) * pc.g->totx() + pc.li;
                    if (dir == 0 || dir == 2) {
                        pc.g->at(pc.li, pc.lj) -= lam * ff;
                        if (pc.phi) (*pc.phi)[pid] -= lam * fp;
                    } else {
                        pc.g->at(pc.li, pc.lj) += lam * ff;
                        if (pc.phi) (*pc.phi)[pid] += lam * fp;
                    }
                }
            }
    }

    void restrictLevel_(int l) { // level l (>=1) onto l-1
        const int bC = cfg_.blockC;
        for (const Patch& p : lvls_[l - 1].patches)
            for (int b = 0; b < bC; ++b)
                for (int a = 0; a < bC; ++a) {
                    const int fi = NG + 2 * a, fj = NG + 2 * b;
                    const Cons sum = p.grid.at(fi, fj) +
                                     p.grid.at(fi + 1, fj) +
                                     p.grid.at(fi, fj + 1) +
                                     p.grid.at(fi + 1, fj + 1);
                    ParentCell pc =
                        parentCell_(l - 1, p.ci0 + a, p.cj0 + b);
                    pc.g->at(pc.li, pc.lj) = Real(0.25) * sum;
                    if (cfg_.species) {
                        const auto avg = [&](const std::vector<Real>& f) {
                            return Real(0.25) *
                                   (f[p.grid.idx(fi, fj)] +
                                    f[p.grid.idx(fi + 1, fj)] +
                                    f[p.grid.idx(fi, fj + 1)] +
                                    f[p.grid.idx(fi + 1, fj + 1)]);
                        };
                        const std::size_t pid =
                            std::size_t(pc.lj) * pc.g->totx() + pc.li;
                        (*pc.phi)[pid] = avg(p.phi);
                        (*pc.Gm)[pid] = avg(p.Gmf);
                    }
                }
    }

    // Rebuild levels lstart..L-1 from tags on current data. Nesting is
    // enforced two ways: wanted blocks force their parent blocks within
    // the rebuilt range (bottom-up), and want[lstart] is clipped to nest
    // inside the *existing* level lstart-1 coverage (whose own dilation
    // buffer guarantees features have not reached its edge).
    void regridFrom_(int lstart) {
        const int L = cfg_.maxLevels;
        if (lstart >= L) return;
        const int bC = cfg_.blockC;

        std::vector<std::vector<std::uint8_t>> want(L - 1);
        for (int l = lstart; l < L; ++l) {
            Level& lv = lvls_[l - 1];
            want[l - 1].assign(std::size_t(lv.nbx) * lv.nby, 0);
        }
        if (lstart == 1) tagBase_(want[0]);
        for (int l = std::max(lstart - 1, 1); l < L - 1; ++l)
            tagLevel_(l, want[l]);

        // Bottom-up nesting force within the rebuilt range.
        for (int l = L - 1; l >= lstart + 1; --l)
            forceParents_(l, want[l - 1], want[l - 2]);

        // Clip want[lstart] against the existing lstart-1 coverage.
        if (lstart >= 2) {
            Level& lv = lvls_[lstart - 1];
            for (int bj = 0; bj < lv.nby; ++bj)
                for (int bi = 0; bi < lv.nbx; ++bi) {
                    std::uint8_t& w =
                        want[lstart - 1][std::size_t(bj) * lv.nbx + bi];
                    if (w && !grownRegionCovered_(lstart, bi, bj)) w = 0;
                }
            // Drop deeper wants whose parent want vanished (top-down).
            for (int l = lstart + 1; l < L; ++l)
                clipToParentWant_(l, want[l - 1], want[l - 2]);
        }

        for (int l = lstart; l < L; ++l) {
            Level& lv = lvls_[l - 1];
            std::vector<Patch> next;
            std::vector<int> nextOf(lv.blockOf.size(), -1);
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

    // For every wanted block of level l, force the level-(l-1) blocks
    // covering its region grown by one level-(l-1) cell.
    void forceParents_(int l, const std::vector<std::uint8_t>& kidWant,
                       std::vector<std::uint8_t>& parWant) const {
        const int bC = cfg_.blockC;
        const Level& kid = lvls_[l - 1];
        const Level& par = lvls_[l - 2];
        const int nxp = base.nx << (l - 1);
        const int nyp = base.ny << (l - 1);
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

    // Is block (bi, bj) of level l, grown by one level-(l-1) cell,
    // covered by the existing level-(l-1) patches?
    bool grownRegionCovered_(int l, int bi, int bj) const {
        const int bC = cfg_.blockC;
        const int nxp = base.nx << (l - 1), nyp = base.ny << (l - 1);
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

    // Drop level-l wants whose grown region is not inside the level-(l-1)
    // wants (used after clipping shallower levels).
    void clipToParentWant_(int l, std::vector<std::uint8_t>& kidWant,
                           const std::vector<std::uint8_t>& parWant) const {
        const int bC = cfg_.blockC;
        const Level& kid = lvls_[l - 1];
        const Level& par = lvls_[l - 2];
        const int nxp = base.nx << (l - 1), nyp = base.ny << (l - 1);
        for (int bj = 0; bj < kid.nby; ++bj)
            for (int bi = 0; bi < kid.nbx; ++bi) {
                std::uint8_t& w =
                    kidWant[std::size_t(bj) * kid.nbx + bi];
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

    AmrConfig cfg_;
    int nf_;
    std::vector<Level> lvls_;
    Scratch2D scratchB_, scratch_;
    ScratchY scratchYB_, scratchY_; // two-gas variants
    std::vector<Cons> baseOld_;
    std::vector<Real> basePhi_, baseGm_, baseOldPhi_, baseOldGm_;
    GasPair gas_;
    std::vector<int> stepCounts_ = std::vector<int>(16, 0);
};

} // namespace mm

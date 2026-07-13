// Cut-cell / embedded-boundary finite-volume solver (phase 2: inviscid,
// 1st-order Godunov). Kept separate from the Cartesian step2D.
//
// Per cut/regular cell the conservative update is
//   U_c += dt/(Lambda_c V) * [ -sum_faces alpha_f F_f A_f - F_EB A_EB ]
// with HLLC face fluxes weighted by the face aperture alpha, and an
// embedded-boundary slip-wall flux F_EB = p*(0, n_out) (pressure only) using
// the exact wall-pressure Riemann closure. Because the EB face area & normal
// come from the divergence closure (see CutCell.hpp), a uniform state at rest
// is preserved to round-off (the face and EB pressure terms cancel exactly).
//
// Phase 2b will add flux redistribution so the time step is limited by the
// FULL cells, not by the arbitrarily small cut cells.
#pragma once

#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "numerics/Hllc.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"   // hllcFluxY, maxStableDt

#include <algorithm>
#include <cmath>
#include <vector>

namespace mm {

// slip-wall flux through an EB face whose OUTWARD-from-fluid normal is
// (nox,noy) = -n_EB (n_EB points solid->fluid).
inline Cons ebWallFlux(const Prim& w, Real nox, Real noy) {
    const Real un = w.u * nox + w.v * noy;      // fluid velocity into the wall
    const Real ps = wallPressure(w, un);        // exact slip-wall pressure
    return Cons{0, ps * nox, ps * noy, 0};
}

// One 1st-order Godunov cut-cell step with hybrid FLUX REDISTRIBUTION, so the
// time step is limited by the FULL cells, not by the small cut cells.
// Covered cells (Lambda = 0) are frozen; ghosts must be filled by the caller.
//
// Redistribution (Colella / AMReX-EB): with the conservative divergence
// D^c = dU/(kappa V), a smooth non-conservative value D^nc = kappa-weighted
// average over the 3x3 fluid neighbourhood, the update uses the hybrid
// D^hyb = kappa D^c + (1-kappa) D^nc and the removed mass
// kappa(1-kappa)(D^c - D^nc) V is spread to the neighbourhood (weight kappa),
// which keeps the scheme conservative while bounding small-cell updates.
inline void stepCutCell(Grid& g, const cutcell::Geometry& geo, Real dt) {
    const Real dx = g.dx, dy = g.dy, V = dx * dy;
    const Real tiny = Real(1e-9);
    std::vector<Cons> dU(g.q.size(), Cons{0, 0, 0, 0});

    // x-faces: between (i,j) and (i+1,j)
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const Real a = geo.at(i, j).apXhi;  // == geo.at(i+1,j).apXlo
            if (a <= tiny) continue;
            const Cons F = hllcFluxX(toPrim(g.at(i, j)), toPrim(g.at(i + 1, j)));
            const Cons f = (a * dy) * F;
            dU[g.idx(i, j)] = dU[g.idx(i, j)] - f;
            dU[g.idx(i + 1, j)] = dU[g.idx(i + 1, j)] + f;
        }
    // y-faces: between (i,j) and (i,j+1)
    for (int i = NG; i < NG + g.nx; ++i)
        for (int j = NG - 1; j < NG + g.ny; ++j) {
            const Real a = geo.at(i, j).apYhi;  // == geo.at(i,j+1).apYlo
            if (a <= tiny) continue;
            const Cons F = hllcFluxY(toPrim(g.at(i, j)), toPrim(g.at(i, j + 1)));
            const Cons f = (a * dx) * F;
            dU[g.idx(i, j)] = dU[g.idx(i, j)] - f;
            dU[g.idx(i, j + 1)] = dU[g.idx(i, j + 1)] + f;
        }
    // embedded-boundary (wall) flux on cut cells
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const CellMoments& m = geo.at(i, j);
            if (m.vol <= tiny || m.vol >= Real(1) - tiny || m.eb.area <= tiny)
                continue;
            const Prim w = toPrim(g.at(i, j));
            const Cons Feb = ebWallFlux(w, -m.eb.nx, -m.eb.ny);
            dU[g.idx(i, j)] = dU[g.idx(i, j)] - m.eb.area * Feb;
        }
    // conservative divergence D^c = dU/(kappa V) on fluid cells
    std::vector<Cons> Dc(g.q.size(), Cons{0, 0, 0, 0});
    const auto kap = [&](int i, int j) { return double(geo.at(i, j).vol); };
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const double k = kap(i, j);
            if (k > double(tiny))
                Dc[g.idx(i, j)] = Real(1.0 / (k * V)) * dU[g.idx(i, j)];
        }

    // hybrid divergence + mass redistribution over the 3x3 fluid neighbourhood
    std::vector<Cons> D(g.q.size(), Cons{0, 0, 0, 0});
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const double kc = kap(i, j);
            if (kc <= double(tiny)) continue;   // covered
            double sumk = 0;
            Cons wsum{0, 0, 0, 0};
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const double kk = kap(i + di, j + dj);
                    if (kk > double(tiny)) {
                        sumk += kk;
                        wsum = wsum + Real(kk) * Dc[g.idx(i + di, j + dj)];
                    }
                }
            const Cons Dnc = Real(1.0 / sumk) * wsum;
            D[g.idx(i, j)] = D[g.idx(i, j)] +
                             (Real(kc) * Dc[g.idx(i, j)] + Real(1 - kc) * Dnc);
            if (kc < 1.0 - double(tiny)) {      // spread the removed mass
                const Cons dD = Real(kc * (1 - kc) / sumk) *
                                (Dc[g.idx(i, j)] - Dnc);
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di)
                        if (kap(i + di, j + dj) > double(tiny))
                            D[g.idx(i + di, j + dj)] =
                                D[g.idx(i + di, j + dj)] + dD;
            }
        }

    // conservative update on fluid cells, with a positivity floor (density /
    // pressure clamp) for robustness in strong shocks — inactive in smooth /
    // at-rest flow, so it does not perturb the conservation gate.
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            if (kap(i, j) <= double(tiny)) continue;
            Cons q = g.at(i, j) + dt * D[g.idx(i, j)];
            Prim w = toPrim(q);
            const Real SPEED_CAP = Real(50);   // >> any physical speed here
            const bool bad = !(w.rho >= RHO_FLOOR) ||          // catches NaN
                             !std::isfinite(double(w.p)) ||
                             !std::isfinite(double(w.u)) ||
                             !std::isfinite(double(w.v)) ||
                             std::hypot(w.u, w.v) > SPEED_CAP;
            if (bad) {
                // near-vacuum / non-finite / runaway momentum: the state is
                // untrustworthy (a huge u = mx/rho would blow up or collapse
                // the time step). Reset to a quiescent floored state — only
                // ever hits arbitrarily thin, tangent slivers (see note: a
                // proper fix is state redistribution over a fuller
                // neighbourhood).
                w.rho = RHO_FLOOR; w.u = 0; w.v = 0; w.p = P_FLOOR;
                q = toCons(w);
            } else if (w.p < P_FLOOR) {
                w.p = P_FLOOR;
                q = toCons(w);
            }
            g.at(i, j) = q;
        }
}

} // namespace mm

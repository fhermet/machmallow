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

#include <vector>

namespace mm {

// slip-wall flux through an EB face whose OUTWARD-from-fluid normal is
// (nox,noy) = -n_EB (n_EB points solid->fluid).
inline Cons ebWallFlux(const Prim& w, Real nox, Real noy) {
    const Real un = w.u * nox + w.v * noy;      // fluid velocity into the wall
    const Real ps = wallPressure(w, un);        // exact slip-wall pressure
    return Cons{0, ps * nox, ps * noy, 0};
}

// One 1st-order Godunov cut-cell step (no redistribution yet — phase 2a).
// Covered cells (Lambda = 0) are frozen; ghosts must be filled by the caller.
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
    // conservative update on fluid cells
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Real vol = geo.at(i, j).vol;
            if (vol <= tiny) continue;          // covered
            g.at(i, j) = g.at(i, j) + (dt / (vol * V)) * dU[g.idx(i, j)];
        }
}

} // namespace mm

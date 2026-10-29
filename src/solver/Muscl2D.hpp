#pragma once

// Unsplit 2D MUSCL-Hancock: limited slopes in x and y, half-dt predictor
// using both flux gradients, HLLC at every face, conservative update.
// One flux set per face per step — what AMR refluxing will need.

#include "core/Grid.hpp"
#include "numerics/Hllc.hpp"
#include "numerics/Limiter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mm {

// Physical flux in the y direction.
inline Cons fluxY(const Prim& w) {
    const Cons q = toCons(w);
    return {q.my, q.mx * w.v, q.my * w.v + w.p, (q.E + w.p) * w.v};
}

// HLLC in y by rotating (u <-> v, mx <-> my) into the x solver.
inline Cons hllcFluxY(const Prim& L, const Prim& R) {
    const Prim Lr{L.rho, L.v, L.u, L.p};
    const Prim Rr{R.rho, R.v, R.u, R.p};
    const Cons f = hllcFluxX(Lr, Rr);
    return {f.rho, f.my, f.mx, f.E};
}

inline Real heatConductivity(Real mu) {
    return mu * GAMMA / ((GAMMA - 1) * PRANDTL);
}

inline Real maxStableDt(const Grid& g, Real cfl, Real mu = 0) {
    Real sx = 0, sy = 0, nuMax = 0;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Prim w = toPrim(g.at(i, j));
            const Real c = soundSpeed(w);
            sx = std::max(sx, std::fabs(w.u) + c);
            sy = std::max(sy, std::fabs(w.v) + c);
            if (mu > 0) nuMax = std::max(nuMax, mu / w.rho);
        }
    Real dt = cfl * std::min(g.dx / sx, g.dy / sy);
    if (mu > 0) {
        // Explicit diffusion limit; gamma/Pr covers heat conduction.
        const Real nuEff =
            nuMax * std::max(Real(4.0 / 3.0), GAMMA / PRANDTL);
        const Real dtV = cfl * Real(0.5) /
                         (nuEff * (1 / (g.dx * g.dx) + 1 / (g.dy * g.dy)));
        dt = std::min(dt, dtV);
    }
    return dt;
}

// Scratch buffers reused across steps.
struct Scratch2D {
    std::vector<Cons> xL, xR, yB, yT, Fx, Fy;
    std::vector<Cons> Vx, Vy; // viscous face fluxes at t^n (mu > 0)
    void resize(std::size_t n) {
        xL.resize(n); xR.resize(n); yB.resize(n); yT.resize(n);
        Fx.resize(n); Fy.resize(n);
    }
};

// Central-difference viscous fluxes (Stokes stress + Fourier heat flux),
// subtracted from the convective face fluxes. Gradients use the current
// cell values; compact normal stencil, 4-point transverse average — fits
// within the ghost ring. Operates on the flux arrays directly so both
// the MUSCL scratch and the WENO scratch can reuse it (the parabolic
// term is 2nd-order central in both: only the convective part needs the
// high-order reconstruction).
// `solid` (optional): immersed mask. A solid neighbour becomes a NO-SLIP
// ghost — the mirror of the face's fluid cell with BOTH velocity components
// flipped (so the wall velocity is zero) and rho/p kept (adiabatic, zero
// normal heat flux). Solid/solid faces carry no viscous flux. The slip
// pressure wall is handled by the convective flux; this shear makes it
// no-slip. With solid = nullptr the path is byte-identical to before.
// `ext` widens the face range by that many cells around the interior
// (ext = 1: the faces of the 1-cell ghost ring, which the Hancock
// predictor needs; the stencil then reaches 2 ghost layers, corners
// included — every ghost fill covers them).
inline void addViscousFluxes(const Grid& g, std::vector<Cons>& Fx,
                             std::vector<Cons>& Fy, Real mu,
                             const std::uint8_t* solid = nullptr,
                             int ext = 0) {
    const Real kT = heatConductivity(mu);
    const Real c23 = Real(2.0 / 3.0), c43 = Real(4.0 / 3.0);
    const auto sol = [&](int ii, int jj) {
        return solid != nullptr && solid[g.idx(ii, jj)] != 0;
    };
    const auto NS = [](Prim w) { w.u = -w.u; w.v = -w.v; return w; };

    for (int j = NG - ext; j < NG + g.ny + ext; ++j) // x faces (i+1/2, j)
        for (int i = NG - 1 - ext; i < NG + g.nx + ext; ++i) {
            const bool sl = sol(i, j), sr = sol(i + 1, j);
            if (sl && sr) continue; // no viscous flux across a solid wall
            const Prim f00 = toPrim(g.at(i, j));
            const Prim f10 = toPrim(g.at(i + 1, j));
            const Prim refF = sl ? f10 : f00; // fluid cell at this face
            const Prim w00 = sl ? NS(f10) : f00;
            const Prim w10 = sr ? NS(f00) : f10;
            const Prim w0p = sol(i, j + 1) ? NS(refF) : toPrim(g.at(i, j + 1));
            const Prim w0m = sol(i, j - 1) ? NS(refF) : toPrim(g.at(i, j - 1));
            const Prim w1p =
                sol(i + 1, j + 1) ? NS(refF) : toPrim(g.at(i + 1, j + 1));
            const Prim w1m =
                sol(i + 1, j - 1) ? NS(refF) : toPrim(g.at(i + 1, j - 1));
            const Real ux = (w10.u - w00.u) / g.dx;
            const Real vx = (w10.v - w00.v) / g.dx;
            const Real Tx =
                (w10.p / w10.rho - w00.p / w00.rho) / g.dx;
            const Real uy =
                ((w0p.u + w1p.u) - (w0m.u + w1m.u)) / (4 * g.dy);
            const Real vy =
                ((w0p.v + w1p.v) - (w0m.v + w1m.v)) / (4 * g.dy);
            const Real txx = mu * (c43 * ux - c23 * vy);
            const Real txy = mu * (uy + vx);
            const Real ub = Real(0.5) * (w00.u + w10.u);
            const Real vb = Real(0.5) * (w00.v + w10.v);
            Cons& F = Fx[g.idx(i, j)];
            F.mx -= txx;
            F.my -= txy;
            F.E -= ub * txx + vb * txy + kT * Tx;
        }

    for (int j = NG - 1 - ext; j < NG + g.ny + ext; ++j) // y faces (i, j+1/2)
        for (int i = NG - ext; i < NG + g.nx + ext; ++i) {
            const bool sb = sol(i, j), st = sol(i, j + 1);
            if (sb && st) continue;
            const Prim f00 = toPrim(g.at(i, j));
            const Prim f01 = toPrim(g.at(i, j + 1));
            const Prim refF = sb ? f01 : f00;
            const Prim w00 = sb ? NS(f01) : f00;
            const Prim w01 = st ? NS(f00) : f01;
            const Prim wp0 = sol(i + 1, j) ? NS(refF) : toPrim(g.at(i + 1, j));
            const Prim wm0 = sol(i - 1, j) ? NS(refF) : toPrim(g.at(i - 1, j));
            const Prim wp1 =
                sol(i + 1, j + 1) ? NS(refF) : toPrim(g.at(i + 1, j + 1));
            const Prim wm1 =
                sol(i - 1, j + 1) ? NS(refF) : toPrim(g.at(i - 1, j + 1));
            const Real uy = (w01.u - w00.u) / g.dy;
            const Real vy = (w01.v - w00.v) / g.dy;
            const Real Ty =
                (w01.p / w01.rho - w00.p / w00.rho) / g.dy;
            const Real ux =
                ((wp0.u + wp1.u) - (wm0.u + wm1.u)) / (4 * g.dx);
            const Real vx =
                ((wp0.v + wp1.v) - (wm0.v + wm1.v)) / (4 * g.dx);
            const Real txy = mu * (uy + vx);
            const Real tyy = mu * (c43 * vy - c23 * ux);
            const Real ub = Real(0.5) * (w00.u + w01.u);
            const Real vb = Real(0.5) * (w00.v + w01.v);
            Cons& F = Fy[g.idx(i, j)];
            F.mx -= txy;
            F.my -= tyy;
            F.E -= ub * txy + vb * tyy + kT * Ty;
        }
}

// Advance one step of size dt. Ghost cells must be filled by the caller.
// Sources (gravity, and the optional per-cell `src`) enter the Hancock
// predictor as a half-dt term, so the face states of an equilibrium
// (flux divergence = source) do not move: without it the steady state
// carries an O(dt) error (MMS, hydrostatic drift). They are then applied
// after the conservative update — gravity with the energy work term taken
// at the momentum midpoint; cell-local, so AMR refluxing is unaffected.
// `src` (optional, size g.q.size()) must also be set on the 1-cell ghost
// ring the predictor reads. The viscous flux divergence (mu > 0, from t^n
// cell values) enters the predictor the same way; the same face fluxes are
// then added to the corrector, so the viscous operator is evaluated once.
//
// `solid` (optionnel, taille g.q.size(), 0 = fluide) immerge un corps
// solide : les cellules solides ne sont pas mises à jour, et chaque face
// fluide/solide reçoit un flux de PAROI réfléchissante (état solide =
// miroir de la vitesse normale de l'état fluide) — conservatif (pas de
// masse à travers la paroi), exact sur les faces alignées (escalier sur
// les formes obliques).
inline void step2D(Grid& g, Real dt, Scratch2D& s, Real mu = 0,
                   Real gx = 0, Real gy = 0,
                   const std::uint8_t* solid = nullptr,
                   const Cons* src = nullptr) {
    const int tx = g.totx(), ty = g.toty();
    const bool grav = gx != 0 || gy != 0;
    s.resize(g.q.size());

    const Real hx = Real(0.5) * dt / g.dx;
    const Real hy = Real(0.5) * dt / g.dy;

    const auto sol = [&](int ii, int jj) {
        return solid != nullptr && solid[g.idx(ii, jj)] != 0;
    };
    const auto mirX = [](Cons c) { c.mx = -c.mx; return c; };
    const auto mirY = [](Cons c) { c.my = -c.my; return c; };

    // Viscous face fluxes at t^n, interior + ghost-ring faces.
    if (mu > 0) {
        s.Vx.assign(g.q.size(), Cons{0, 0, 0, 0});
        s.Vy.assign(g.q.size(), Cons{0, 0, 0, 0});
        addViscousFluxes(g, s.Vx, s.Vy, mu, solid, 1);
    }
    const Real idx = Real(1) / g.dx, idy = Real(1) / g.dy;
    const auto inRing = [&](int ii, int jj) {
        return ii >= NG - 1 && ii <= NG + g.nx && jj >= NG - 1 &&
               jj <= NG + g.ny;
    };

    // Predictor: time-advanced face states (needs a 1-cell ring of ghosts).
    for (int j = 1; j < ty - 1; ++j) {
        for (int i = 1; i < tx - 1; ++i) {
            const std::size_t id = g.idx(i, j);
            if (sol(i, j)) continue; // cellule solide : pas reconstruite
            const Cons& q0 = g.q[id];
            // voisins solides -> état miroir (paroi) pour la pente
            const Cons qxl = sol(i - 1, j) ? mirX(q0) : g.at(i - 1, j);
            const Cons qxr = sol(i + 1, j) ? mirX(q0) : g.at(i + 1, j);
            const Cons qyb = sol(i, j - 1) ? mirY(q0) : g.at(i, j - 1);
            const Cons qyt = sol(i, j + 1) ? mirY(q0) : g.at(i, j + 1);
            const Cons dqx = limitedSlope(qxl, q0, qxr);
            const Cons dqy = limitedSlope(qyb, q0, qyt);
            const Cons xl = q0 - Real(0.5) * dqx;
            const Cons xr = q0 + Real(0.5) * dqx;
            const Cons yb = q0 - Real(0.5) * dqy;
            const Cons yt = q0 + Real(0.5) * dqy;
            Cons adv =
                hx * (fluxX(toPrim(xl)) - fluxX(toPrim(xr))) +
                hy * (fluxY(toPrim(yb)) - fluxY(toPrim(yt)));
            if (grav) { // half-dt gravity: keeps equilibria well balanced
                const Real rho = std::max(q0.rho, RHO_FLOOR);
                adv += Real(0.5) * dt *
                       Cons{0, rho * gx, rho * gy, gx * q0.mx + gy * q0.my};
            }
            if (src != nullptr) adv += Real(0.5) * dt * src[id];
            if (mu > 0 && inRing(i, j)) { // half-dt viscous divergence
                const Cons dv =
                    idx * (s.Vx[g.idx(i - 1, j)] - s.Vx[id]) +
                    idy * (s.Vy[g.idx(i, j - 1)] - s.Vy[id]);
                adv += Real(0.5) * dt * dv;
            }
            s.xL[id] = xl + adv;
            s.xR[id] = xr + adv;
            s.yB[id] = yb + adv;
            s.yT[id] = yt + adv;

            // Positivity: near vacuum the conserved-variable
            // reconstruction can produce faces with rho <= 0 or negative
            // internal energy (huge velocities after flooring) — fall
            // back to first order for this cell.
            const auto bad = [](const Cons& c) {
                if (c.rho <= RHO_FLOOR) return true;
                const Real ke =
                    Real(0.5) * (c.mx * c.mx + c.my * c.my) / c.rho;
                return c.E - ke <= 0;
            };
            if (bad(s.xL[id]) || bad(s.xR[id]) || bad(s.yB[id]) ||
                bad(s.yT[id]))
                s.xL[id] = s.xR[id] = s.yB[id] = s.yT[id] = q0;
        }
    }

    // HLLC fluxes. Fx[id] is the flux through face (i+1/2, j), Fy through
    // (i, j+1/2).
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const bool sl = sol(i, j), sr = sol(i + 1, j);
            if (sl && sr) { s.Fx[id] = Cons{0, 0, 0, 0}; continue; }
            const Cons L = s.xR[id], R = s.xL[g.idx(i + 1, j)];
            // Paroi : flux de pression exact (robuste en supersonique ;
            // un = vitesse du fluide vers la paroi).
            if (sr) { const Prim w = toPrim(L);
                      s.Fx[id] = wallFluxX(w, w.u); continue; }
            if (sl) { const Prim w = toPrim(R);
                      s.Fx[id] = wallFluxX(w, -w.u); continue; }
            s.Fx[id] = hllcFluxX(toPrim(L), toPrim(R));
        }
    for (int j = NG - 1; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const bool sb = sol(i, j), st = sol(i, j + 1);
            if (sb && st) { s.Fy[id] = Cons{0, 0, 0, 0}; continue; }
            const Cons B = s.yT[id], T = s.yB[g.idx(i, j + 1)];
            if (st) { const Prim w = toPrim(B);
                      s.Fy[id] = wallFluxY(w, w.v); continue; }
            if (sb) { const Prim w = toPrim(T);
                      s.Fy[id] = wallFluxY(w, -w.v); continue; }
            s.Fy[id] = hllcFluxY(toPrim(B), toPrim(T));
        }

    if (mu > 0) { // same t^n viscous fluxes as the predictor
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG - 1; i < NG + g.nx; ++i)
                s.Fx[g.idx(i, j)] += s.Vx[g.idx(i, j)];
        for (int j = NG - 1; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                s.Fy[g.idx(i, j)] += s.Vy[g.idx(i, j)];
    }

    // Conservative update.
    const Real lx = dt / g.dx, ly = dt / g.dy;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            if (sol(i, j)) continue; // cellule solide : figée
            g.q[id] += lx * (s.Fx[g.idx(i - 1, j)] - s.Fx[id]) +
                       ly * (s.Fy[g.idx(i, j - 1)] - s.Fy[id]);
        }

    if (src != nullptr)
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i) {
                const std::size_t id = g.idx(i, j);
                if (!sol(i, j)) g.q[id] += dt * src[id];
            }

    if (grav) {
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i) {
                if (sol(i, j)) continue;
                Cons& q = g.q[g.idx(i, j)];
                const Real rho = std::max(q.rho, RHO_FLOOR);
                const Real mx0 = q.mx, my0 = q.my;
                q.mx += dt * rho * gx;
                q.my += dt * rho * gy;
                q.E += Real(0.5) * dt *
                       (gx * (mx0 + q.mx) + gy * (my0 + q.my));
            }
    }
}

} // namespace mm

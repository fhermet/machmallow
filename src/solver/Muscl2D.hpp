#pragma once

// Unsplit 2D MUSCL-Hancock: limited slopes in x and y, half-dt predictor
// using both flux gradients, HLLC at every face, conservative update.
// One flux set per face per step — what AMR refluxing will need.

#include "core/Grid.hpp"
#include "numerics/Hllc.hpp"
#include "numerics/Limiter.hpp"

#include <algorithm>
#include <cmath>
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
    void resize(std::size_t n) {
        xL.resize(n); xR.resize(n); yB.resize(n); yT.resize(n);
        Fx.resize(n); Fy.resize(n);
    }
};

// Central-difference viscous fluxes (Stokes stress + Fourier heat flux),
// subtracted from the convective fluxes already stored in the scratch.
// Gradients use the t^n cell values; compact normal stencil, 4-point
// transverse average — fits within the 2 ghost layers.
inline void addViscousFluxes(const Grid& g, Scratch2D& s, Real mu) {
    const Real kT = heatConductivity(mu);
    const Real c23 = Real(2.0 / 3.0), c43 = Real(4.0 / 3.0);

    for (int j = NG; j < NG + g.ny; ++j) // x faces (i+1/2, j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const Prim w00 = toPrim(g.at(i, j));
            const Prim w10 = toPrim(g.at(i + 1, j));
            const Prim w0p = toPrim(g.at(i, j + 1));
            const Prim w0m = toPrim(g.at(i, j - 1));
            const Prim w1p = toPrim(g.at(i + 1, j + 1));
            const Prim w1m = toPrim(g.at(i + 1, j - 1));
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
            Cons& F = s.Fx[g.idx(i, j)];
            F.mx -= txx;
            F.my -= txy;
            F.E -= ub * txx + vb * txy + kT * Tx;
        }

    for (int j = NG - 1; j < NG + g.ny; ++j) // y faces (i, j+1/2)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Prim w00 = toPrim(g.at(i, j));
            const Prim w01 = toPrim(g.at(i, j + 1));
            const Prim wp0 = toPrim(g.at(i + 1, j));
            const Prim wm0 = toPrim(g.at(i - 1, j));
            const Prim wp1 = toPrim(g.at(i + 1, j + 1));
            const Prim wm1 = toPrim(g.at(i - 1, j + 1));
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
            Cons& F = s.Fy[g.idx(i, j)];
            F.mx -= txy;
            F.my -= tyy;
            F.E -= ub * txy + vb * tyy + kT * Ty;
        }
}

// Advance one step of size dt. Ghost cells must be filled by the caller.
inline void step2D(Grid& g, Real dt, Scratch2D& s, Real mu = 0) {
    const int tx = g.totx(), ty = g.toty();
    s.resize(g.q.size());

    const Real hx = Real(0.5) * dt / g.dx;
    const Real hy = Real(0.5) * dt / g.dy;

    // Predictor: time-advanced face states (needs a 1-cell ring of ghosts).
    for (int j = 1; j < ty - 1; ++j) {
        for (int i = 1; i < tx - 1; ++i) {
            const std::size_t id = g.idx(i, j);
            const Cons& q0 = g.q[id];
            const Cons dqx =
                limitedSlope(g.at(i - 1, j), q0, g.at(i + 1, j));
            const Cons dqy =
                limitedSlope(g.at(i, j - 1), q0, g.at(i, j + 1));
            const Cons xl = q0 - Real(0.5) * dqx;
            const Cons xr = q0 + Real(0.5) * dqx;
            const Cons yb = q0 - Real(0.5) * dqy;
            const Cons yt = q0 + Real(0.5) * dqy;
            const Cons adv =
                hx * (fluxX(toPrim(xl)) - fluxX(toPrim(xr))) +
                hy * (fluxY(toPrim(yb)) - fluxY(toPrim(yt)));
            s.xL[id] = xl + adv;
            s.xR[id] = xr + adv;
            s.yB[id] = yb + adv;
            s.yT[id] = yt + adv;
        }
    }

    // HLLC fluxes. Fx[id] is the flux through face (i+1/2, j), Fy through
    // (i, j+1/2).
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            s.Fx[id] = hllcFluxX(toPrim(s.xR[id]), toPrim(s.xL[g.idx(i + 1, j)]));
        }
    for (int j = NG - 1; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            s.Fy[id] = hllcFluxY(toPrim(s.yT[id]), toPrim(s.yB[g.idx(i, j + 1)]));
        }

    if (mu > 0) addViscousFluxes(g, s, mu);

    // Conservative update.
    const Real lx = dt / g.dx, ly = dt / g.dy;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            g.q[id] += lx * (s.Fx[g.idx(i - 1, j)] - s.Fx[id]) +
                       ly * (s.Fy[g.idx(i, j - 1)] - s.Fy[id]);
        }
}

} // namespace mm

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

inline Real maxStableDt(const Grid& g, Real cfl) {
    Real sx = 0, sy = 0;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Prim w = toPrim(g.at(i, j));
            const Real c = soundSpeed(w);
            sx = std::max(sx, std::fabs(w.u) + c);
            sy = std::max(sy, std::fabs(w.v) + c);
        }
    return cfl * std::min(g.dx / sx, g.dy / sy);
}

// Scratch buffers reused across steps.
struct Scratch2D {
    std::vector<Cons> xL, xR, yB, yT, Fx, Fy;
    void resize(std::size_t n) {
        xL.resize(n); xR.resize(n); yB.resize(n); yT.resize(n);
        Fx.resize(n); Fy.resize(n);
    }
};

// Advance one step of size dt. Ghost cells must be filled by the caller.
inline void step2D(Grid& g, Real dt, Scratch2D& s) {
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

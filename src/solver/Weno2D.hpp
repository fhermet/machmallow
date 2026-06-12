#pragma once

// Finite-difference WENO5 (Jiang & Shu) with local Lax-Friedrichs flux
// splitting, advanced by SSP-RK3 — the v1.3 high-order scheme, kept
// separate from the validated MUSCL-Hancock paths.
//
// Dimension-by-dimension: at each x-face i+1/2 the split fluxes
// f± = (F(q) ± alpha*q)/2 are reconstructed component-wise from their
// 5-point upwind stencils (alpha = max |u|+c over the stencil, per
// face), and the divergence of the face fluxes is the RHS. Conservative
// by construction, and the face fluxes are retained so the AMR
// refluxing machinery can consume them like the MUSCL ones.
//
// SSP-RK3 (Shu-Osher):
//   u1 = u + dt L(u)
//   u2 = 3/4 u + 1/4 (u1 + dt L(u1))
//   u  = 1/3 u + 2/3 (u2 + dt L(u2))
// The caller provides the ghost fill, re-applied before every stage.

#include "core/Grid.hpp"
#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mm {

struct ScratchW {
    std::vector<Cons> Fx, Fy; // face fluxes of the last RHS evaluation
    std::vector<Cons> u0, u1; // RK stage storage
    void resize(std::size_t n) {
        Fx.resize(n);
        Fy.resize(n);
        u0.resize(n);
        u1.resize(n);
    }
};

namespace wenodetail {

// WENO5 reconstruction of the face value from the 5-point stencil
// {v0..v4} ordered so that v2 is the upwind cell adjacent to the face.
inline Real weno5(Real v0, Real v1, Real v2, Real v3, Real v4) {
    constexpr Real EPS = Real(1e-6);
    const Real q0 = (2 * v0 - 7 * v1 + 11 * v2) / 6;
    const Real q1 = (-v1 + 5 * v2 + 2 * v3) / 6;
    const Real q2 = (2 * v2 + 5 * v3 - v4) / 6;
    const Real b0 = Real(13.0 / 12.0) * (v0 - 2 * v1 + v2) *
                        (v0 - 2 * v1 + v2) +
                    Real(0.25) * (v0 - 4 * v1 + 3 * v2) *
                        (v0 - 4 * v1 + 3 * v2);
    const Real b1 = Real(13.0 / 12.0) * (v1 - 2 * v2 + v3) *
                        (v1 - 2 * v2 + v3) +
                    Real(0.25) * (v1 - v3) * (v1 - v3);
    const Real b2 = Real(13.0 / 12.0) * (v2 - 2 * v3 + v4) *
                        (v2 - 2 * v3 + v4) +
                    Real(0.25) * (3 * v2 - 4 * v3 + v4) *
                        (3 * v2 - 4 * v3 + v4);
    const Real w0 = Real(0.1) / ((EPS + b0) * (EPS + b0));
    const Real w1 = Real(0.6) / ((EPS + b1) * (EPS + b1));
    const Real w2 = Real(0.3) / ((EPS + b2) * (EPS + b2));
    return (w0 * q0 + w1 * q1 + w2 * q2) / (w0 + w1 + w2);
}

} // namespace wenodetail

// Face fluxes of the current state into s.Fx/s.Fy (x-faces stored at
// the left cell's index, like the MUSCL scratch). Ghosts must be valid.
inline void wenoFluxes(const Grid& g, ScratchW& s) {
    s.resize(g.q.size());
    using wenodetail::weno5;

    // direction-agnostic kernel: str is the cell stride along the
    // direction, flux() the directional physical flux, vel() the
    // directional velocity
    const auto faces = [&](int str, auto&& flux, auto&& vel,
                           std::vector<Cons>& F, int i0, int i1, int j0,
                           int j1) {
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i) {
                const std::size_t id = g.idx(i, j);
                // stencil cells id-2str .. id+3str around the face
                std::size_t c[6];
                for (int k = 0; k < 6; ++k)
                    c[k] = id + std::size_t(k - 2) * str;
                Real alpha = 0;
                Cons f[6];
                for (int k = 0; k < 6; ++k) {
                    const Prim w = toPrim(g.q[c[k]]);
                    alpha = std::max(alpha,
                                     std::fabs(vel(w)) + soundSpeed(w));
                    f[k] = flux(w);
                }
                Cons out{};
                Real* po = &out.rho;
                for (int m = 0; m < NVARS; ++m) {
                    Real fp[6], fm[6];
                    for (int k = 0; k < 6; ++k) {
                        const Real fv = (&f[k].rho)[m];
                        const Real qv = (&g.q[c[k]].rho)[m];
                        fp[k] = Real(0.5) * (fv + alpha * qv);
                        fm[k] = Real(0.5) * (fv - alpha * qv);
                    }
                    po[m] = weno5(fp[0], fp[1], fp[2], fp[3], fp[4]) +
                            weno5(fm[5], fm[4], fm[3], fm[2], fm[1]);
                }
                F[id] = out;
            }
    };

    const auto fx = [](const Prim& w) { return fluxX(w); };
    const auto fy = [](const Prim& w) { return fluxY(w); };
    const auto ux = [](const Prim& w) { return w.u; };
    const auto uy = [](const Prim& w) { return w.v; };
    // x-faces between i and i+1 for i in [NG-1, NG+nx)
    faces(1, fx, ux, s.Fx, NG - 1, NG + g.nx, NG, NG + g.ny);
    // y-faces between j and j+1
    faces(g.totx(), fy, uy, s.Fy, NG, NG + g.nx, NG - 1, NG + g.ny);
}

// One SSP-RK3 step; fill(g) refreshes the ghosts and is called before
// every stage's flux evaluation.
template <class Fill>
inline void stepWeno2D(Grid& g, Real dt, ScratchW& s, Fill&& fill) {
    s.resize(g.q.size());
    const Real lx = dt / g.dx, ly = dt / g.dy;
    const auto applyRhs = [&](Real a, Real b) {
        // q = a*u0 + b*(q + dt*L(q)) on the interior
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i) {
                const std::size_t id = g.idx(i, j);
                const Cons adv =
                    g.q[id] +
                    lx * (s.Fx[g.idx(i - 1, j)] - s.Fx[id]) +
                    ly * (s.Fy[g.idx(i, j - 1)] - s.Fy[id]);
                g.q[id] = a * s.u0[id] + b * adv;
            }
    };
    s.u0 = g.q;
    fill(g);
    wenoFluxes(g, s);
    applyRhs(0, 1);
    fill(g);
    wenoFluxes(g, s);
    applyRhs(Real(0.75), Real(0.25));
    fill(g);
    wenoFluxes(g, s);
    applyRhs(Real(1.0 / 3.0), Real(2.0 / 3.0));
}

} // namespace mm

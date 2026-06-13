#pragma once

// WENO5 (Jiang & Shu) face reconstruction + HLLC, advanced by SSP-RK3
// — the v1.3 high-order scheme, kept separate from the validated
// MUSCL-Hancock paths.
//
// Dimension-by-dimension: at each face the left/right PRIMITIVE states
// are reconstructed component-wise from their 5-point upwind stencils
// and fed to the HLLC solver. A first cut used local Lax-Friedrichs
// flux splitting instead (textbook FD-WENO5): its dissipation scales
// with |u|+c on EVERY wave, so contacts and shear layers — the waves
// HLLC resolves almost exactly — came out visibly more diffused than
// MUSCL-HLLC on Kelvin-Helmholtz despite the formal order. WENO5
// states + HLLC keeps the high-order smooth-flow accuracy AND the
// contact sharpness. Conservative by construction; the face fluxes are
// retained so the AMR refluxing machinery can consume them like the
// MUSCL ones.
//
// SSP-RK3 (Shu-Osher):
//   u1 = u + dt L(u)
//   u2 = 3/4 u + 1/4 (u1 + dt L(u1))
//   u  = 1/3 u + 2/3 (u2 + dt L(u2))
// The caller provides the ghost fill, re-applied before every stage.

#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp" // hllcFluxX/Y, addViscousFluxes

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
// With mu > 0 the central-difference viscous flux is added to each face
// (2nd-order parabolic term; the convective part stays WENO5+HLLC).
inline void wenoFluxes(const Grid& g, ScratchW& s, Real mu = 0) {
    s.resize(g.q.size());
    using wenodetail::weno5;

    // direction-agnostic kernel: str is the cell stride along the
    // direction; the face sits between cells id and id+str
    const auto faces = [&](int str, bool xDir, std::vector<Cons>& F,
                           int i0, int i1, int j0, int j1) {
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i) {
                const std::size_t id = g.idx(i, j);
                Prim w[6];
                for (int k = 0; k < 6; ++k)
                    w[k] = toPrim(g.q[id + std::size_t(k - 2) * str]);
                Prim L{}, R{};
                Real* pl = &L.rho;
                Real* pr = &R.rho;
                for (int m = 0; m < NVARS; ++m) {
                    const auto v = [&](int k) { return (&w[k].rho)[m]; };
                    pl[m] = weno5(v(0), v(1), v(2), v(3), v(4));
                    pr[m] = weno5(v(5), v(4), v(3), v(2), v(1));
                }
                L.rho = std::max(L.rho, RHO_FLOOR);
                L.p = std::max(L.p, P_FLOOR);
                R.rho = std::max(R.rho, RHO_FLOOR);
                R.p = std::max(R.p, P_FLOOR);
                F[id] = xDir ? hllcFluxX(L, R) : hllcFluxY(L, R);
            }
    };

    // x-faces between i and i+1 for i in [NG-1, NG+nx)
    faces(1, true, s.Fx, NG - 1, NG + g.nx, NG, NG + g.ny);
    // y-faces between j and j+1
    faces(g.totx(), false, s.Fy, NG, NG + g.nx, NG - 1, NG + g.ny);

    if (mu > 0) addViscousFluxes(g, s.Fx, s.Fy, mu);
}

// One SSP-RK3 step; fill(g) refreshes the ghosts and is called before
// every stage's flux evaluation. mu > 0 enables the viscous flux.
template <class Fill>
inline void stepWeno2D(Grid& g, Real dt, ScratchW& s, Fill&& fill,
                       Real mu = 0) {
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
    wenoFluxes(g, s, mu);
    applyRhs(0, 1);
    fill(g);
    wenoFluxes(g, s, mu);
    applyRhs(Real(0.75), Real(0.25));
    fill(g);
    wenoFluxes(g, s, mu);
    applyRhs(Real(1.0 / 3.0), Real(2.0 / 3.0));
}

} // namespace mm

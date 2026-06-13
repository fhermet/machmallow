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
#include "physics/TwoGas.hpp" // GasPair, hllcFluxXG/YG (two-gas WENO)
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

// ---- two-gas WENO5 ---------------------------------------------------------
//
// Same WENO5 face reconstruction, but each face state carries its own
// gamma: the mass fraction Y and the advected Gamma are reconstructed
// alongside (rho, u, v, p), the per-side-gamma HLLC closes the flux and
// returns the contact speed, the species mass flux is the HLLC mass flux
// upwinded by Y, and Gamma rides quasi-conservatively on the contact
// speed (Shyue / Johnsen-Colonius), exactly as the MUSCL two-gas step.
// Method-of-lines: no half-dt face-Gamma advance is needed — the face p
// and face Gamma come from the same reconstruction at the same time
// level, so they are synchronous by construction (that half-dt term only
// existed to fix the MUSCL Hancock predictor mismatch).
struct ScratchWY {
    std::vector<Cons> Fx, Fy;       // conserved face flux
    std::vector<Real> Fpx, Fpy;     // species mass flux (phi = rho*Y)
    std::vector<Real> Fgx, Fgy;     // gamma flux S* * Gamma_upwind
    std::vector<Real> Ssx, Ssy;     // contact speeds
    std::vector<Cons> u0;           // RK start: conserved
    std::vector<Real> phi0, Gm0;    // RK start: scalars
    void resize(std::size_t n) {
        Fx.resize(n); Fy.resize(n);
        Fpx.resize(n); Fpy.resize(n);
        Fgx.resize(n); Fgy.resize(n);
        Ssx.resize(n); Ssy.resize(n);
        u0.resize(n); phi0.resize(n); Gm0.resize(n);
    }
};

inline void wenoFluxesY(const Grid& g, const std::vector<Real>& phi,
                        const std::vector<Real>& Gm, ScratchWY& s,
                        const GasPair& gas) {
    s.resize(g.q.size());
    using wenodetail::weno5;
    const Real Gmin = std::min(gas.Gamma(0), gas.Gamma(1));
    const Real Gmax = std::max(gas.Gamma(0), gas.Gamma(1));
    const auto Yof = [&](std::size_t id) {
        return std::clamp(phi[id] / std::max(g.q[id].rho, RHO_FLOOR),
                          Real(0), Real(1));
    };

    const auto faces = [&](int str, bool xDir, std::vector<Cons>& F,
                           std::vector<Real>& Fp, std::vector<Real>& Fg,
                           std::vector<Real>& Ss, int i0, int i1, int j0,
                           int j1) {
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i) {
                const std::size_t id = g.idx(i, j);
                Prim w[6];
                Real Yc[6], Gc[6];
                for (int k = 0; k < 6; ++k) {
                    const std::size_t c = id + std::size_t(k - 2) * str;
                    w[k] = toPrimG(g.q[c], Gm[c]);
                    Yc[k] = Yof(c);
                    Gc[k] = Gm[c];
                }
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
                const Real YL = std::clamp(
                    weno5(Yc[0], Yc[1], Yc[2], Yc[3], Yc[4]), Real(0),
                    Real(1));
                const Real YR = std::clamp(
                    weno5(Yc[5], Yc[4], Yc[3], Yc[2], Yc[1]), Real(0),
                    Real(1));
                const Real GL = std::clamp(
                    weno5(Gc[0], Gc[1], Gc[2], Gc[3], Gc[4]), Gmin, Gmax);
                const Real GR = std::clamp(
                    weno5(Gc[5], Gc[4], Gc[3], Gc[2], Gc[1]), Gmin, Gmax);
                Real ss = 0;
                const Cons f =
                    xDir ? hllcFluxXG(L, 1 + 1 / GL, R, 1 + 1 / GR, &ss)
                         : hllcFluxYG(L, 1 + 1 / GL, R, 1 + 1 / GR, &ss);
                F[id] = f;
                Fp[id] = f.rho * (f.rho > 0 ? YL : YR);
                Ss[id] = ss;
                Fg[id] = ss * (ss > 0 ? GL : GR);
            }
    };

    faces(1, true, s.Fx, s.Fpx, s.Fgx, s.Ssx, NG - 1, NG + g.nx, NG,
          NG + g.ny);
    faces(g.totx(), false, s.Fy, s.Fpy, s.Fgy, s.Ssy, NG, NG + g.nx,
          NG - 1, NG + g.ny);
}

// SSP-RK3 two-gas step on a uniform grid; fill(g) refreshes ALL ghosts
// (conserved + phi + Gamma) before each stage.
template <class Fill>
inline void stepWeno2DY(Grid& g, std::vector<Real>& phi,
                        std::vector<Real>& Gm, Real dt, ScratchWY& s,
                        const GasPair& gas, Fill&& fill) {
    s.resize(g.q.size());
    const Real lx = dt / g.dx, ly = dt / g.dy;
    const Real Gmin = std::min(gas.Gamma(0), gas.Gamma(1));
    const Real Gmax = std::max(gas.Gamma(0), gas.Gamma(1));
    const auto applyRhs = [&](Real a, Real b) {
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i) {
                const std::size_t id = g.idx(i, j);
                const std::size_t iw = g.idx(i - 1, j);
                const std::size_t js = g.idx(i, j - 1);
                const Cons adv = g.q[id] + lx * (s.Fx[iw] - s.Fx[id]) +
                                 ly * (s.Fy[js] - s.Fy[id]);
                g.q[id] = a * s.u0[id] + b * adv;
                const Real phiAdv = phi[id] +
                                    lx * (s.Fpx[iw] - s.Fpx[id]) +
                                    ly * (s.Fpy[js] - s.Fpy[id]);
                phi[id] = a * s.phi0[id] + b * phiAdv;
                const Real GmAdv =
                    Gm[id] -
                    lx * (s.Fgx[id] - s.Fgx[iw] -
                          Gm[id] * (s.Ssx[id] - s.Ssx[iw])) -
                    ly * (s.Fgy[id] - s.Fgy[js] -
                          Gm[id] * (s.Ssy[id] - s.Ssy[js]));
                Gm[id] =
                    std::clamp(a * s.Gm0[id] + b * GmAdv, Gmin, Gmax);
            }
    };
    s.u0 = g.q;
    s.phi0 = phi;
    s.Gm0 = Gm;
    fill(g);
    wenoFluxesY(g, phi, Gm, s, gas);
    applyRhs(0, 1);
    fill(g);
    wenoFluxesY(g, phi, Gm, s, gas);
    applyRhs(Real(0.75), Real(0.25));
    fill(g);
    wenoFluxesY(g, phi, Gm, s, gas);
    applyRhs(Real(1.0 / 3.0), Real(2.0 / 3.0));
}

} // namespace mm

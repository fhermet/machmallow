#pragma once

// Two-gas MUSCL-Hancock step: the standard unsplit scheme of Muscl2D.hpp
// extended with a conservative species field phi = rho*Y (same storage
// layout as the grid, ghosts included) and a per-cell mixture EOS
// Gamma(Y). The species face flux is the HLLC mass flux times the
// upwinded face Y, so uniform-Y regions keep Y exactly constant and the
// species mass telescopes like every other conserved quantity.
// Kept separate from the validated single-gas step on purpose: this is
// the uniform-grid core of milestone v1.2 (AMR/GPU follow).

#include "core/Grid.hpp"
#include "numerics/Limiter.hpp"
#include "physics/TwoGas.hpp"
#include "solver/Muscl2D.hpp" // addViscousFluxes (shared viscous flux)

#include <algorithm>
#include <cmath>
#include <vector>

namespace mm {

struct ScratchY {
    std::vector<Cons> xL, xR, yB, yT, Fx, Fy;
    std::vector<Real> yxL, yxR, yyB, yyT, Fpx, Fpy; // face Y and phi flux
    std::vector<Real> gxL, gxR, gyB, gyT;           // face Gamma
    std::vector<Real> Ssx, Ssy, Fgx, Fgy; // contact speeds, Gamma flux
    void resize(std::size_t n) {
        xL.resize(n); xR.resize(n); yB.resize(n); yT.resize(n);
        Fx.resize(n); Fy.resize(n);
        yxL.resize(n); yxR.resize(n); yyB.resize(n); yyT.resize(n);
        Fpx.resize(n); Fpy.resize(n);
        gxL.resize(n); gxR.resize(n); gyB.resize(n); gyT.resize(n);
        Ssx.resize(n); Ssy.resize(n); Fgx.resize(n); Fgy.resize(n);
    }
};

inline Real maxStableDtY(const Grid& g, const std::vector<Real>& Gm,
                         Real cfl, Real mu = 0) {
    Real sx = 0, sy = 0, rhoMin = Real(1e30);
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const Prim w = toPrimG(g.q[id], Gm[id]);
            const Real c = soundSpeedG(w, 1 + 1 / Gm[id]);
            sx = std::max(sx, std::fabs(w.u) + c);
            sy = std::max(sy, std::fabs(w.v) + c);
            if (mu > 0) rhoMin = std::min(rhoMin, w.rho);
        }
    Real dt = cfl * std::min(g.dx / sx, g.dy / sy);
    if (mu > 0) {
        const Real nuEff = (mu / rhoMin) *
                           std::max(Real(4.0 / 3.0), GAMMA / PRANDTL);
        dt = std::min(dt, cfl * Real(0.5) /
                              (nuEff * (1 / (g.dx * g.dx) +
                                        1 / (g.dy * g.dy))));
    }
    return dt;
}

// Pressure closes on the ADVECTED Gamma field (quasi-conservative
// transport by the HLLC contact speed) — not on Gamma(Y) — so the
// energy and Gamma mixing weights match (Shyue / Johnsen & Colonius).
// Reconstruction is PRIMITIVE (rho, u, v, p) + Gamma: face energies are
// built from face p and face Gamma, so a uniform-(p, u) interface has
// exactly uniform face states and no pressure oscillation can be
// generated. phi = rho*Y remains the conserved species mass.
// mu > 0 adds the central-difference Stokes + Fourier viscous flux to
// the convective face fluxes (shared single-gas operator) — needed for
// deflagrations, where the flame propagates by heat conduction.
inline void step2DY(Grid& g, std::vector<Real>& phi,
                    std::vector<Real>& Gm, Real dt, ScratchY& s,
                    const GasPair& gas, Real mu = 0) {
    const int tx = g.totx(), ty = g.toty();
    s.resize(g.q.size());

    const Real hx = Real(0.5) * dt / g.dx;
    const Real hy = Real(0.5) * dt / g.dy;

    const auto Yof = [&](std::size_t id) {
        return std::clamp(phi[id] / std::max(g.q[id].rho, RHO_FLOOR),
                          Real(0), Real(1));
    };
    const Real Gmin = std::min(gas.Gamma(0), gas.Gamma(1));
    const Real Gmax = std::max(gas.Gamma(0), gas.Gamma(1));

    // Predictor: limited slopes on conserved vars AND on Y; the face
    // Gamma comes from the face Y (reconstructed, not time-advanced —
    // uniform-Y faces are exact either way).
    for (int j = 1; j < ty - 1; ++j)
        for (int i = 1; i < tx - 1; ++i) {
            const std::size_t id = g.idx(i, j);
            const Cons& q0 = g.q[id];
            const Real Y0 = Yof(id);
            const Real dYx = mcSlope(Y0 - Yof(g.idx(i - 1, j)),
                                     Yof(g.idx(i + 1, j)) - Y0);
            const Real dYy = mcSlope(Y0 - Yof(g.idx(i, j - 1)),
                                     Yof(g.idx(i, j + 1)) - Y0);
            s.yxL[id] = std::clamp(Y0 - Real(0.5) * dYx, Real(0), Real(1));
            s.yxR[id] = std::clamp(Y0 + Real(0.5) * dYx, Real(0), Real(1));
            s.yyB[id] = std::clamp(Y0 - Real(0.5) * dYy, Real(0), Real(1));
            s.yyT[id] = std::clamp(Y0 + Real(0.5) * dYy, Real(0), Real(1));

            const Real G0 = Gm[id];
            const Real dGx = mcSlope(G0 - Gm[g.idx(i - 1, j)],
                                     Gm[g.idx(i + 1, j)] - G0);
            const Real dGy = mcSlope(G0 - Gm[g.idx(i, j - 1)],
                                     Gm[g.idx(i, j + 1)] - G0);
            // half-dt advection of the face Gamma/Y (cell velocity):
            // keeps them synchronous with the half-dt-advanced energy —
            // a t^n face Gamma against a t^(n+1/2) face E is exactly
            // the residual interface wiggle
            const Real u0c = q0.mx / std::max(q0.rho, RHO_FLOOR);
            const Real v0c = q0.my / std::max(q0.rho, RHO_FLOOR);
            const Real gAdv = hx * u0c * dGx + hy * v0c * dGy;
            s.gxL[id] = std::clamp(G0 - Real(0.5) * dGx - gAdv, Gmin, Gmax);
            s.gxR[id] = std::clamp(G0 + Real(0.5) * dGx - gAdv, Gmin, Gmax);
            s.gyB[id] = std::clamp(G0 - Real(0.5) * dGy - gAdv, Gmin, Gmax);
            s.gyT[id] = std::clamp(G0 + Real(0.5) * dGy - gAdv, Gmin, Gmax);

            // primitive slopes; face states rebuilt from face prims
            // and face Gamma (positivity by construction: rho, p
            // clamped at the floors)
            const Prim w0 = toPrimG(q0, G0);
            const Prim wm = toPrimG(g.at(i - 1, j), Gm[g.idx(i - 1, j)]);
            const Prim wp = toPrimG(g.at(i + 1, j), Gm[g.idx(i + 1, j)]);
            const Prim wb = toPrimG(g.at(i, j - 1), Gm[g.idx(i, j - 1)]);
            const Prim wt = toPrimG(g.at(i, j + 1), Gm[g.idx(i, j + 1)]);
            const auto slope = [&](Real m, Real c, Real pl) {
                return mcSlope(c - m, pl - c);
            };
            const auto face = [&](const Prim& w, Real sgn, bool xDir,
                                  Real dr, Real du, Real dv, Real dp) {
                Prim f{std::max(w.rho + sgn * Real(0.5) * dr, RHO_FLOOR),
                       w.u + sgn * Real(0.5) * du,
                       w.v + sgn * Real(0.5) * dv,
                       std::max(w.p + sgn * Real(0.5) * dp, P_FLOOR)};
                (void)xDir;
                return f;
            };
            const Real drx = slope(wm.rho, w0.rho, wp.rho);
            const Real dux = slope(wm.u, w0.u, wp.u);
            const Real dvx = slope(wm.v, w0.v, wp.v);
            const Real dpx = slope(wm.p, w0.p, wp.p);
            const Real dry = slope(wb.rho, w0.rho, wt.rho);
            const Real duy = slope(wb.u, w0.u, wt.u);
            const Real dvy = slope(wb.v, w0.v, wt.v);
            const Real dpy = slope(wb.p, w0.p, wt.p);

            const Prim pxl = face(w0, -1, true, drx, dux, dvx, dpx);
            const Prim pxr = face(w0, +1, true, drx, dux, dvx, dpx);
            const Prim pyb = face(w0, -1, false, dry, duy, dvy, dpy);
            const Prim pyt = face(w0, +1, false, dry, duy, dvy, dpy);

            const Cons xl = toConsG(pxl, s.gxL[id]);
            const Cons xr = toConsG(pxr, s.gxR[id]);
            const Cons yb = toConsG(pyb, s.gyB[id]);
            const Cons yt = toConsG(pyt, s.gyT[id]);
            const Cons adv =
                hx * (fluxXG(pxl, s.gxL[id]) - fluxXG(pxr, s.gxR[id])) +
                hy * (fluxYG(pyb, s.gyB[id]) - fluxYG(pyt, s.gyT[id]));
            s.xL[id] = xl + adv;
            s.xR[id] = xr + adv;
            s.yB[id] = yb + adv;
            s.yT[id] = yt + adv;

            // the half-dt advance can still produce a non-physical face
            // in extreme cells: same first-order fallback
            const auto bad = [](const Cons& c) {
                if (c.rho <= RHO_FLOOR) return true;
                const Real ke =
                    Real(0.5) * (c.mx * c.mx + c.my * c.my) / c.rho;
                return c.E - ke <= 0;
            };
            if (bad(s.xL[id]) || bad(s.xR[id]) || bad(s.yB[id]) ||
                bad(s.yT[id])) {
                s.xL[id] = s.xR[id] = s.yB[id] = s.yT[id] = q0;
                s.yxL[id] = s.yxR[id] = s.yyB[id] = s.yyT[id] = Y0;
                s.gxL[id] = s.gxR[id] = s.gyB[id] = s.gyT[id] = G0;
            }
        }

    // HLLC fluxes with per-side gamma; species flux = mass flux times
    // the upwinded face Y.
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const std::size_t idp = g.idx(i + 1, j);
            const Real GL = s.gxR[id], GR = s.gxL[idp];
            Real ss = 0;
            const Cons F = hllcFluxXG(toPrimG(s.xR[id], GL), 1 + 1 / GL,
                                      toPrimG(s.xL[idp], GR), 1 + 1 / GR,
                                      &ss);
            s.Fx[id] = F;
            s.Fpx[id] = F.rho * (F.rho > 0 ? s.yxR[id] : s.yxL[idp]);
            s.Ssx[id] = ss;
            s.Fgx[id] = ss * (ss > 0 ? GL : GR);
        }
    for (int j = NG - 1; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const std::size_t idp = g.idx(i, j + 1);
            const Real GB = s.gyT[id], GT = s.gyB[idp];
            Real ss = 0;
            const Cons F = hllcFluxYG(toPrimG(s.yT[id], GB), 1 + 1 / GB,
                                      toPrimG(s.yB[idp], GT), 1 + 1 / GT,
                                      &ss);
            s.Fy[id] = F;
            s.Fpy[id] = F.rho * (F.rho > 0 ? s.yyT[id] : s.yyB[idp]);
            s.Ssy[id] = ss;
            s.Fgy[id] = ss * (ss > 0 ? GB : GT);
        }

    // Viscous flux (Stokes + Fourier) on the conserved faces — drives
    // the deflagration's heat conduction; species/Gamma fluxes unchanged.
    if (mu > 0) addViscousFluxes(g, s.Fx, s.Fy, mu);

    // Conservative update (Cons and species together).
    const Real lx = dt / g.dx, ly = dt / g.dy;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            g.q[id] += lx * (s.Fx[g.idx(i - 1, j)] - s.Fx[id]) +
                       ly * (s.Fy[g.idx(i, j - 1)] - s.Fy[id]);
            phi[id] += lx * (s.Fpx[g.idx(i - 1, j)] - s.Fpx[id]) +
                       ly * (s.Fpy[g.idx(i, j - 1)] - s.Fpy[id]);
            // quasi-conservative Gamma transport by the contact speed:
            // dG/dt + u dG/dx = 0 in flux-difference form
            const std::size_t iw = g.idx(i - 1, j), js = g.idx(i, j - 1);
            Gm[id] = std::clamp(
                Gm[id] -
                    lx * (s.Fgx[id] - s.Fgx[iw] -
                          Gm[id] * (s.Ssx[id] - s.Ssx[iw])) -
                    ly * (s.Fgy[id] - s.Fgy[js] -
                          Gm[id] * (s.Ssy[id] - s.Ssy[js])),
                Gmin, Gmax);
        }
}

} // namespace mm

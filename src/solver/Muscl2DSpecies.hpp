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
                         Real cfl) {
    Real sx = 0, sy = 0;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            const Prim w = toPrimG(g.q[id], Gm[id]);
            const Real c = soundSpeedG(w, 1 + 1 / Gm[id]);
            sx = std::max(sx, std::fabs(w.u) + c);
            sy = std::max(sy, std::fabs(w.v) + c);
        }
    return cfl * std::min(g.dx / sx, g.dy / sy);
}

// Pressure closes on the ADVECTED Gamma field (quasi-conservative
// transport by the HLLC contact speed) — not on Gamma(Y) — so the
// energy and Gamma mixing weights match and material interfaces stay
// free of pressure oscillations (Shyue / Johnsen & Colonius). phi = rho*Y
// remains the conserved species mass.
inline void step2DY(Grid& g, std::vector<Real>& phi,
                    std::vector<Real>& Gm, Real dt, ScratchY& s,
                    const GasPair& gas) {
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
            const Cons dqx = limitedSlope(g.at(i - 1, j), q0, g.at(i + 1, j));
            const Cons dqy = limitedSlope(g.at(i, j - 1), q0, g.at(i, j + 1));
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
            s.gxL[id] = std::clamp(G0 - Real(0.5) * dGx, Gmin, Gmax);
            s.gxR[id] = std::clamp(G0 + Real(0.5) * dGx, Gmin, Gmax);
            s.gyB[id] = std::clamp(G0 - Real(0.5) * dGy, Gmin, Gmax);
            s.gyT[id] = std::clamp(G0 + Real(0.5) * dGy, Gmin, Gmax);

            const Cons xl = q0 - Real(0.5) * dqx;
            const Cons xr = q0 + Real(0.5) * dqx;
            const Cons yb = q0 - Real(0.5) * dqy;
            const Cons yt = q0 + Real(0.5) * dqy;
            const Cons adv =
                hx * (fluxXG(toPrimG(xl, s.gxL[id]), s.gxL[id]) -
                      fluxXG(toPrimG(xr, s.gxR[id]), s.gxR[id])) +
                hy * (fluxYG(toPrimG(yb, s.gyB[id]), s.gyB[id]) -
                      fluxYG(toPrimG(yt, s.gyT[id]), s.gyT[id]));
            s.xL[id] = xl + adv;
            s.xR[id] = xr + adv;
            s.yB[id] = yb + adv;
            s.yT[id] = yt + adv;

            // positivity fallback, as in the single-gas scheme
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

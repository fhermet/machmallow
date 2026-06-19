#pragma once

// HLLC approximate Riemann solver (Toro, ch. 10), x-direction.
// Wave speed estimates from the PVRS pressure estimate (Toro §10.5.2).

#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

namespace detail {

// One-sided HLLC star-region flux: F_K + S_K (U*_K - U_K).
inline Cons hllcSideFlux(const Prim& w, Real S, Real Sstar) {
    const Cons q = toCons(w);
    const Cons f = fluxX(w);
    const Real coef = w.rho * (S - w.u) / (S - Sstar);
    const Cons qStar = {
        coef,
        coef * Sstar,
        coef * w.v,
        coef * (q.E / w.rho +
                (Sstar - w.u) * (Sstar + w.p / (w.rho * (S - w.u)))),
    };
    return f + S * (qStar - q);
}

} // namespace detail

inline Cons hllcFluxX(const Prim& L, const Prim& R) {
    const Real cL = soundSpeed(L);
    const Real cR = soundSpeed(R);

    // PVRS pressure estimate -> q factors -> wave speeds.
    const Real rhoBar = Real(0.5) * (L.rho + R.rho);
    const Real cBar = Real(0.5) * (cL + cR);
    const Real pPvrs =
        Real(0.5) * (L.p + R.p) - Real(0.5) * (R.u - L.u) * rhoBar * cBar;
    const Real pStar = std::max(Real(0), pPvrs);

    const auto qK = [](Real pK, Real pS) {
        if (pS <= pK) return Real(1);
        return std::sqrt(1 + (GAMMA + 1) / (2 * GAMMA) * (pS / pK - 1));
    };
    const Real SL = L.u - cL * qK(L.p, pStar);
    const Real SR = R.u + cR * qK(R.p, pStar);

    const Real Sstar =
        (R.p - L.p + L.rho * L.u * (SL - L.u) - R.rho * R.u * (SR - R.u)) /
        (L.rho * (SL - L.u) - R.rho * (SR - R.u));

    if (SL >= 0) return fluxX(L);
    if (Sstar >= 0) return detail::hllcSideFlux(L, SL, Sstar);
    if (SR >= 0) return detail::hllcSideFlux(R, SR, Sstar);
    return fluxX(R);
}

// Pressure on a reflective (slip) wall facing fluid state W, whose velocity
// component INTO the wall is `un` (> 0 compresses). Solves Toro's f_W(p*) =
// un exactly (Newton): a shock branch for un > 0, a rarefaction branch for
// un < 0. This is the correct wall closure for BOTH subsonic and SUPERSONIC
// normal inflow — the mirror-state HLLC leaks at supersonic walls because
// its PVRS wave-speed estimate (SL = uL - cL*q) stays positive and upwinds
// the full incoming flux. The wall flux carries no mass/energy: only the
// normal momentum sees p* (slip: tangential convective flux vanishes since
// the normal velocity is zero).
inline Real wallPressure(const Prim& W, Real un) {
    const Real c = soundSpeed(W);
    const Real A = Real(2) / ((GAMMA + 1) * W.rho);
    const Real B = (GAMMA - 1) / (GAMMA + 1) * W.p;
    const Real m = (GAMMA - 1) / (2 * GAMMA);
    const auto f = [&](Real p) {
        if (p > W.p) return (p - W.p) * std::sqrt(A / (p + B));   // shock
        return Real(2) * c / (GAMMA - 1) *
               (std::pow(p / W.p, m) - 1);                       // raréfaction
    };
    const auto df = [&](Real p) {
        if (p > W.p) {
            const Real s = std::sqrt(A / (p + B));
            return s * (1 - Real(0.5) * (p - W.p) / (p + B));
        }
        return Real(1) / (W.rho * c) * std::pow(p / W.p, -(GAMMA + 1) /
                                                             (2 * GAMMA));
    };
    Real p = std::max(P_FLOOR, W.p + un * W.rho * c); // acoustic guess
    for (int it = 0; it < 30; ++it) {
        const Real step = (f(p) - un) / df(p);
        p = std::max(P_FLOOR, p - step);
        if (std::fabs(step) < Real(1e-6) * (p + P_FLOOR)) break;
    }
    return p;
}
// Slip-wall face fluxes (wall normal +x / +y). `un` = fluid velocity into
// the wall. Layout: Cons{rho, mx, my, E}.
inline Cons wallFluxX(const Prim& W, Real un) {
    return Cons{0, wallPressure(W, un), 0, 0};
}
inline Cons wallFluxY(const Prim& W, Real un) {
    return Cons{0, 0, wallPressure(W, un), 0};
}

} // namespace mm

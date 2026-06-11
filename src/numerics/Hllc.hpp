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

} // namespace mm

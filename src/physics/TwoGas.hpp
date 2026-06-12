#pragma once

// Two-gas (binary mixture) model: a mass fraction Y of gas 2 is carried
// as the conservative field phi = rho*Y, and the EOS closes with the
// mixture Gamma = 1/(gamma-1) mass-weighted between the two gases:
//   Gamma(Y) = (1-Y)/(gamma1-1) + Y/(gamma2-1),  p = (E - ke)/Gamma.
// The species flux uses the HLLC mass flux upwinded by its sign, so Y
// stays exactly constant wherever it is uniform (Abgrall's consistency
// requirement) and species mass is conserved to the flux telescoping.

#include "core/Types.hpp"
#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

struct GasPair {
    Real gamma1 = Real(1.4);
    Real gamma2 = Real(1.4);

    Real Gamma(Real Y) const { // 1/(gamma_mix - 1)
        return (1 - Y) / (gamma1 - 1) + Y / (gamma2 - 1);
    }
    Real gammaOf(Real Y) const { return 1 + 1 / Gamma(Y); }
};

// EOS helpers with an explicit Gamma (per cell / per face).
inline Prim toPrimG(const Cons& q, Real G) {
    const Real rho = std::max(q.rho, RHO_FLOOR);
    const Real u = q.mx / rho, v = q.my / rho;
    const Real ke = Real(0.5) * rho * (u * u + v * v);
    const Real p = std::max((q.E - ke) / G, P_FLOOR);
    return {rho, u, v, p};
}
inline Cons toConsG(const Prim& w, Real G) {
    const Real ke = Real(0.5) * w.rho * (w.u * w.u + w.v * w.v);
    return {w.rho, w.rho * w.u, w.rho * w.v, w.p * G + ke};
}
inline Real soundSpeedG(const Prim& w, Real gamma) {
    return std::sqrt(gamma * w.p / w.rho);
}
inline Cons fluxXG(const Prim& w, Real G) {
    const Cons q = toConsG(w, G);
    return {q.mx, q.mx * w.u + w.p, q.my * w.u, (q.E + w.p) * w.u};
}
inline Cons fluxYG(const Prim& w, Real G) {
    const Cons q = toConsG(w, G);
    return {q.my, q.mx * w.v, q.my * w.v + w.p, (q.E + w.p) * w.v};
}

// HLLC with per-side gamma (the star-state algebra is gamma-free; only
// the sound speeds and the q-factor use each side's gamma).
inline Cons hllcFluxXG(const Prim& L, Real gL, const Prim& R, Real gR,
                       Real* ssOut = nullptr) {
    const Real cL = soundSpeedG(L, gL);
    const Real cR = soundSpeedG(R, gR);
    const Real GL = 1 / (gL - 1), GR = 1 / (gR - 1);

    const Real rhoBar = Real(0.5) * (L.rho + R.rho);
    const Real cBar = Real(0.5) * (cL + cR);
    const Real pPvrs =
        Real(0.5) * (L.p + R.p) - Real(0.5) * (R.u - L.u) * rhoBar * cBar;
    const Real pStar = std::max(Real(0), pPvrs);

    const auto qK = [](Real g, Real pK, Real pS) {
        if (pS <= pK) return Real(1);
        return std::sqrt(1 + (g + 1) / (2 * g) * (pS / pK - 1));
    };
    const Real SL = L.u - cL * qK(gL, L.p, pStar);
    const Real SR = R.u + cR * qK(gR, R.p, pStar);
    const Real Ss =
        (R.p - L.p + L.rho * L.u * (SL - L.u) - R.rho * R.u * (SR - R.u)) /
        (L.rho * (SL - L.u) - R.rho * (SR - R.u));
    if (ssOut != nullptr) *ssOut = Ss;

    const auto side = [&](const Prim& w, Real G, Real S) {
        const Cons q = toConsG(w, G);
        const Cons f = {q.mx, q.mx * w.u + w.p, q.my * w.u,
                        (q.E + w.p) * w.u};
        const Real coef = w.rho * (S - w.u) / (S - Ss);
        const Cons qs = {coef, coef * Ss, coef * w.v,
                         coef * (q.E / w.rho +
                                 (Ss - w.u) *
                                     (Ss + w.p / (w.rho * (S - w.u))))};
        return f + S * (qs - q);
    };
    if (SL >= 0) return fluxXG(L, GL);
    if (Ss >= 0) return side(L, GL, SL);
    if (SR >= 0) return side(R, GR, SR);
    return fluxXG(R, GR);
}

inline Cons hllcFluxYG(const Prim& L, Real gL, const Prim& R, Real gR,
                       Real* ssOut = nullptr) {
    const Prim Lr{L.rho, L.v, L.u, L.p};
    const Prim Rr{R.rho, R.v, R.u, R.p};
    const Cons f = hllcFluxXG(Lr, gL, Rr, gR, ssOut);
    return {f.rho, f.my, f.mx, f.E};
}

} // namespace mm

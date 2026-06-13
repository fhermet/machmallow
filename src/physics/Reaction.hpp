#pragma once

// Single-step irreversible reaction (v1.5 reacting flow): a progress
// variable lambda in [0, 1] (0 = unburnt, 1 = burnt) obeys
//   dlambda/dt = A (1 - lambda) exp(-Ea / T),   T = p / rho  (R = 1)
// and each unit of progress releases q (heat per unit mass) into the
// internal energy. Operator-split from the flow: the reaction step holds
// rho and velocity fixed and advances (e, lambda), so it is a pointwise
// ODE — distinct from any global-implicit solve (a project non-goal).
//
// Arrhenius is exponentially stiff in T, so react() ADAPTIVELY
// sub-cycles an RK4 within the outer dt (bounding the per-substep
// progress); this is the explicit "chemical sub-cycling" path.

#include "core/Types.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

struct Reaction {
    Real A = 0;   // pre-exponential rate factor
    Real Ea = 0;  // activation "energy" (R = 1, so Ea has units of T)
    Real q = 0;   // heat release per unit mass over the full burn
    Real Tign = 0; // optional ignition cutoff: no reaction below this T

    // reaction rate dlambda/dt at progress lambda and temperature T
    Real rate(Real lam, Real T) const {
        if (lam >= 1 || T <= Tign) return 0;
        return A * (1 - lam) * std::exp(-Ea / std::max(T, Real(1e-30)));
    }
};

// Advance the chemistry over dt at fixed density and velocity, updating
// the internal energy per mass eInt and the progress lam. T = (g-1) eInt
// (R = 1: e = T/(g-1), T = p/rho).
//
// Since de/dt = q dlambda/dt, internal energy is slaved to progress:
// e(lambda) = e0 + q (lambda - lambda0) EXACTLY. So we integrate the
// single stiff ODE for lambda (adaptive RK4 sub-cycling, T taken from
// the slaved e so the thermal runaway is captured) and set the energy
// algebraically at the end — energy-conserving by construction, and
// free of the clamp inconsistency that splitting (e, lambda) would have.
inline void react(Real& eInt, Real& lam, Real dt, const Reaction& r,
                  Real gamma = GAMMA) {
    const Real gm1 = gamma - 1;
    const Real e0 = eInt, lam0 = lam;
    const auto w = [&](Real l) {
        return r.rate(l, gm1 * (e0 + r.q * (l - lam0)));
    };
    Real t = 0;
    int guard = 0;
    while (t < dt && ++guard < 100000) {
        const Real rate = w(lam);
        Real h = dt - t;
        // bound the progress per sub-step (<= 0.05) for stiff ignition
        if (rate > 0) h = std::min(h, Real(0.05) / rate);
        const Real k1 = w(lam);
        const Real k2 = w(lam + Real(0.5) * h * k1);
        const Real k3 = w(lam + Real(0.5) * h * k2);
        const Real k4 = w(lam + h * k3);
        lam = std::clamp(lam + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4),
                         Real(0), Real(1));
        t += h;
    }
    eInt = e0 + r.q * (lam - lam0); // heat release, energy-conserving
}

} // namespace mm

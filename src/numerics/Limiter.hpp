#pragma once

#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

// Monotonized-central (MC) limited slope from neighbour differences.
inline Real mcSlope(Real dm, Real dp) {
    if (dm * dp <= 0) return 0;
    const Real c = Real(0.5) * (dm + dp);
    const Real lim = 2 * std::min(std::fabs(dm), std::fabs(dp));
    return std::copysign(std::min(std::fabs(c), lim), c);
}

// Component-wise MC slope on conserved variables.
inline Cons limitedSlope(const Cons& qm, const Cons& q0, const Cons& qp) {
    return {
        mcSlope(q0.rho - qm.rho, qp.rho - q0.rho),
        mcSlope(q0.mx - qm.mx, qp.mx - q0.mx),
        mcSlope(q0.my - qm.my, qp.my - q0.my),
        mcSlope(q0.E - qm.E, qp.E - q0.E),
    };
}

} // namespace mm

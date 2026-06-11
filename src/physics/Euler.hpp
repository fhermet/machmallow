#pragma once

#include "core/Types.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

struct Prim {
    Real rho, u, v, p;
};

struct Cons {
    Real rho, mx, my, E;

    Cons& operator+=(const Cons& o) {
        rho += o.rho; mx += o.mx; my += o.my; E += o.E;
        return *this;
    }
    Cons& operator-=(const Cons& o) {
        rho -= o.rho; mx -= o.mx; my -= o.my; E -= o.E;
        return *this;
    }
};

inline Cons operator+(Cons a, const Cons& b) { return a += b; }
inline Cons operator-(const Cons& a, const Cons& b) {
    return {a.rho - b.rho, a.mx - b.mx, a.my - b.my, a.E - b.E};
}
inline Cons operator*(Real s, const Cons& a) {
    return {s * a.rho, s * a.mx, s * a.my, s * a.E};
}

inline Cons toCons(const Prim& w) {
    const Real ke = Real(0.5) * w.rho * (w.u * w.u + w.v * w.v);
    return {w.rho, w.rho * w.u, w.rho * w.v, w.p / (GAMMA - 1) + ke};
}

inline Prim toPrim(const Cons& q) {
    const Real rho = std::max(q.rho, RHO_FLOOR);
    const Real u = q.mx / rho;
    const Real v = q.my / rho;
    const Real ke = Real(0.5) * rho * (u * u + v * v);
    const Real p = std::max((GAMMA - 1) * (q.E - ke), P_FLOOR);
    return {rho, u, v, p};
}

inline Real soundSpeed(const Prim& w) {
    return std::sqrt(GAMMA * w.p / w.rho);
}

// Physical flux in the x direction.
inline Cons fluxX(const Prim& w) {
    const Cons q = toCons(w);
    return {q.mx, q.mx * w.u + w.p, q.my * w.u, (q.E + w.p) * w.u};
}

} // namespace mm

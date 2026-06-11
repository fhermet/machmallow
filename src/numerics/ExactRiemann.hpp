#pragma once

// Exact Riemann solver for the 1D Euler equations (Toro, ch. 4).
// Runs in double precision: this is the validation reference, independent
// of the float32 solver path.

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mm::exact {

inline constexpr double G = 1.4; // gamma

struct State {
    double rho, u, p;
};

inline double soundSpeed(const State& s) { return std::sqrt(G * s.p / s.rho); }

namespace detail {

// Toro's pressure function f_K(p) and its derivative.
inline void pressureFunction(double p, const State& K, double& f, double& df) {
    const double cK = soundSpeed(K);
    if (p > K.p) { // shock
        const double A = 2.0 / ((G + 1) * K.rho);
        const double B = (G - 1) / (G + 1) * K.p;
        const double sq = std::sqrt(A / (p + B));
        f = (p - K.p) * sq;
        df = sq * (1.0 - 0.5 * (p - K.p) / (p + B));
    } else { // rarefaction
        const double pr = p / K.p;
        f = 2.0 * cK / (G - 1) * (std::pow(pr, (G - 1) / (2 * G)) - 1.0);
        df = std::pow(pr, -(G + 1) / (2 * G)) / (K.rho * cK);
    }
}

inline double starPressure(const State& L, const State& R) {
    // Two-rarefaction initial guess: positive and usually close.
    const double cL = soundSpeed(L), cR = soundSpeed(R);
    const double z = (G - 1) / (2 * G);
    double p = std::pow(
        (cL + cR - 0.5 * (G - 1) * (R.u - L.u)) /
            (cL / std::pow(L.p, z) + cR / std::pow(R.p, z)),
        1.0 / z);
    p = std::max(p, 1e-14);

    for (int it = 0; it < 100; ++it) {
        double fL, dfL, fR, dfR;
        pressureFunction(p, L, fL, dfL);
        pressureFunction(p, R, fR, dfR);
        const double dp = (fL + fR + (R.u - L.u)) / (dfL + dfR);
        p -= dp;
        if (p <= 0) p = 1e-14;
        if (std::fabs(dp) < 1e-12 * p) return p;
    }
    throw std::runtime_error("exact Riemann solver did not converge");
}

} // namespace detail

// Solution of the Riemann problem (L, R) sampled at speed s = x/t.
inline State sample(const State& L, const State& R, double s) {
    const double pS = detail::starPressure(L, R);
    double fL, dfL, fR, dfR;
    detail::pressureFunction(pS, L, fL, dfL);
    detail::pressureFunction(pS, R, fR, dfR);
    const double uS = 0.5 * (L.u + R.u) + 0.5 * (fR - fL);

    const double gm = (G - 1) / (G + 1);

    if (s <= uS) { // left of contact
        const double cL = soundSpeed(L);
        if (pS > L.p) { // left shock
            const double SL =
                L.u - cL * std::sqrt((G + 1) / (2 * G) * pS / L.p +
                                     (G - 1) / (2 * G));
            if (s <= SL) return L;
            const double rho =
                L.rho * (pS / L.p + gm) / (gm * pS / L.p + 1.0);
            return {rho, uS, pS};
        }
        // left rarefaction
        const double cS = cL * std::pow(pS / L.p, (G - 1) / (2 * G));
        if (s <= L.u - cL) return L;
        if (s >= uS - cS) {
            return {L.rho * std::pow(pS / L.p, 1.0 / G), uS, pS};
        }
        // inside the fan
        const double f = 2.0 / (G + 1) + gm / cL * (L.u - s);
        return {L.rho * std::pow(f, 2.0 / (G - 1)),
                2.0 / (G + 1) * (cL + 0.5 * (G - 1) * L.u + s),
                L.p * std::pow(f, 2.0 * G / (G - 1))};
    }

    // right of contact (mirror)
    const double cR = soundSpeed(R);
    if (pS > R.p) { // right shock
        const double SR =
            R.u + cR * std::sqrt((G + 1) / (2 * G) * pS / R.p +
                                 (G - 1) / (2 * G));
        if (s >= SR) return R;
        const double rho = R.rho * (pS / R.p + gm) / (gm * pS / R.p + 1.0);
        return {rho, uS, pS};
    }
    // right rarefaction
    const double cS = cR * std::pow(pS / R.p, (G - 1) / (2 * G));
    if (s >= R.u + cR) return R;
    if (s <= uS + cS) {
        return {R.rho * std::pow(pS / R.p, 1.0 / G), uS, pS};
    }
    const double f = 2.0 / (G + 1) - gm / cR * (R.u - s);
    return {R.rho * std::pow(f, 2.0 / (G - 1)),
            2.0 / (G + 1) * (-cR + 0.5 * (G - 1) * R.u + s),
            R.p * std::pow(f, 2.0 * G / (G - 1))};
}

} // namespace mm::exact

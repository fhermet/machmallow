#pragma once

// Exact Riemann solver for the 1D Euler equations (Toro, ch. 4).
// Runs in double precision: this is the validation reference, independent
// of the float32 solver path.

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mm::exact {

inline constexpr double G = 1.4; // default gamma

struct State {
    double rho, u, p;
    double g = G; // per-side gamma (two-gas Riemann problems)
};

inline double soundSpeed(const State& s) {
    return std::sqrt(s.g * s.p / s.rho);
}

namespace detail {

// Toro's pressure function f_K(p) and its derivative (gamma of side K).
inline void pressureFunction(double p, const State& K, double& f, double& df) {
    const double cK = soundSpeed(K);
    const double g = K.g;
    if (p > K.p) { // shock
        const double A = 2.0 / ((g + 1) * K.rho);
        const double B = (g - 1) / (g + 1) * K.p;
        const double sq = std::sqrt(A / (p + B));
        f = (p - K.p) * sq;
        df = sq * (1.0 - 0.5 * (p - K.p) / (p + B));
    } else { // rarefaction
        const double pr = p / K.p;
        f = 2.0 * cK / (g - 1) * (std::pow(pr, (g - 1) / (2 * g)) - 1.0);
        df = std::pow(pr, -(g + 1) / (2 * g)) / (K.rho * cK);
    }
}

inline double starPressure(const State& L, const State& R) {
    // Two-rarefaction initial guess (mean gamma): positive, close enough.
    const double cL = soundSpeed(L), cR = soundSpeed(R);
    const double gm2 = 0.5 * (L.g + R.g);
    const double z = (gm2 - 1) / (2 * gm2);
    double p = std::pow(
        (cL + cR - 0.5 * (gm2 - 1) * (R.u - L.u)) /
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
// Each side keeps its own gamma across its wave (the contact separates
// the gases).
inline State sample(const State& L, const State& R, double s) {
    const double pS = detail::starPressure(L, R);
    double fL, dfL, fR, dfR;
    detail::pressureFunction(pS, L, fL, dfL);
    detail::pressureFunction(pS, R, fR, dfR);
    const double uS = 0.5 * (L.u + R.u) + 0.5 * (fR - fL);

    if (s <= uS) { // left of contact: gas L, gamma L.g
        const double g = L.g, gm = (g - 1) / (g + 1);
        const double cL = soundSpeed(L);
        if (pS > L.p) { // left shock
            const double SL =
                L.u - cL * std::sqrt((g + 1) / (2 * g) * pS / L.p +
                                     (g - 1) / (2 * g));
            if (s <= SL) return L;
            const double rho =
                L.rho * (pS / L.p + gm) / (gm * pS / L.p + 1.0);
            return {rho, uS, pS, g};
        }
        // left rarefaction
        const double cS = cL * std::pow(pS / L.p, (g - 1) / (2 * g));
        if (s <= L.u - cL) return L;
        if (s >= uS - cS) {
            return {L.rho * std::pow(pS / L.p, 1.0 / g), uS, pS, g};
        }
        // inside the fan
        const double f = 2.0 / (g + 1) + gm / cL * (L.u - s);
        return {L.rho * std::pow(f, 2.0 / (g - 1)),
                2.0 / (g + 1) * (cL + 0.5 * (g - 1) * L.u + s),
                L.p * std::pow(f, 2.0 * g / (g - 1)), g};
    }

    // right of contact: gas R, gamma R.g
    const double g = R.g, gm = (g - 1) / (g + 1);
    const double cR = soundSpeed(R);
    if (pS > R.p) { // right shock
        const double SR =
            R.u + cR * std::sqrt((g + 1) / (2 * g) * pS / R.p +
                                 (g - 1) / (2 * g));
        if (s >= SR) return R;
        const double rho = R.rho * (pS / R.p + gm) / (gm * pS / R.p + 1.0);
        return {rho, uS, pS, g};
    }
    // right rarefaction
    const double cS = cR * std::pow(pS / R.p, (g - 1) / (2 * g));
    if (s >= R.u + cR) return R;
    if (s <= uS + cS) {
        return {R.rho * std::pow(pS / R.p, 1.0 / g), uS, pS, g};
    }
    const double f = 2.0 / (g + 1) - gm / cR * (R.u - s);
    return {R.rho * std::pow(f, 2.0 / (g - 1)),
            2.0 / (g + 1) * (-cR + 0.5 * (g - 1) * R.u + s),
            R.p * std::pow(f, 2.0 * g / (g - 1)), g};
}

} // namespace mm::exact

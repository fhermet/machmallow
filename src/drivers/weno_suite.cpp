// WENO5 + SSP-RK3 validation (uniform grid, CPU):
//   1. smooth entropy wave (density bump advected at u = 1): L1
//      convergence order at fixed CFL — the global order is capped at 3
//      by RK3, which is exactly what the gate checks; the spatial-only
//      order (fixed tiny dt) is printed for reference
//   2. Sod tube vs the exact Riemann solution, against MUSCL-Hancock on
//      the same grid: WENO5-LLF smears the contact a bit more than
//      HLLC but must stay in the same error class, with no spurious
//      over/undershoot beyond the exact extrema
//   3. isentropic vortex transport: L1 order at fixed CFL and the
//      head-to-head dissipation ratio vs MUSCL at the same resolution

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2D.hpp"
#include "solver/Weno2D.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);

void fillPeriodic(Grid& g) {
    fillPeriodicX(g);
    fillPeriodicY(g);
}

double entropyError(int N, double dtFixed) {
    Grid g(N, 4, 0, 0, 1, Real(4.0) / N);
    const auto rhoOf = [](double x) {
        return 1.0 + 0.3 * std::sin(2 * M_PI * x);
    };
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) = toCons({Real(rhoOf(g.xc(i))), 1, 0, 1});
    ScratchW s;
    double t = 0;
    const double tEnd = 1; // one period
    while (t < tEnd * (1 - 1e-12)) {
        Real dt = dtFixed > 0 ? Real(dtFixed)
                              : maxStableDt(g, CFL, 0);
        dt = std::min(dt, Real(tEnd - t));
        stepWeno2D(g, dt, s, fillPeriodic);
        t += dt;
    }
    double e = 0;
    for (int i = NG; i < NG + g.nx; ++i)
        e += std::fabs(double(g.at(i, NG + 1).rho) -
                       rhoOf(double(g.xc(i)) - tEnd));
    return e / g.nx;
}

bool gate1_entropyOrder() {
    const double e16 = entropyError(16, 0);
    const double e32 = entropyError(32, 0);
    const double e64 = entropyError(64, 0);
    const double o1 = std::log2(e16 / e32);
    const double o2 = std::log2(e32 / e64);
    // spatial-only reference (fixed dt well under the temporal error)
    const double s8 = entropyError(8, 2e-4);
    const double s16 = entropyError(16, 2e-4);
    const double sOrd = std::log2(s8 / s16);
    std::printf("gate 1 — entropy wave: L1 %.3e -> %.3e -> %.3e, order "
                "%.2f / %.2f (gate >= 2.7, RK3-capped); spatial-only "
                "8->16: %.2f\n",
                e16, e32, e64, o1, o2, sOrd);
    return o2 >= 2.7;
}

bool gate2_sod() {
    const exact::State L{1, 0, 1}, R{0.125, 0, 0.1};
    const double tEnd = 0.2;
    const int N = 400;

    const auto runSod = [&](bool weno) {
        Grid g(N, 4, 0, 0, 1, Real(4.0) / N);
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i)
                g.at(i, j) = toCons(g.xc(i) < Real(0.5)
                                        ? Prim{1, 0, 0, 1}
                                        : Prim{Real(0.125), 0, 0,
                                               Real(0.1)});
        const auto fill = [](Grid& gg) {
            fillTransmissiveLeft(gg);
            fillTransmissiveRight(gg);
            fillPeriodicY(gg);
        };
        ScratchW sw;
        Scratch2D sm;
        double t = 0;
        while (t < tEnd * (1 - 1e-12)) {
            const Real dt =
                std::min(maxStableDt(g, CFL, 0), Real(tEnd - t));
            if (weno) {
                stepWeno2D(g, dt, sw, fill);
            } else {
                fill(g);
                step2D(g, dt, sm);
            }
            t += dt;
        }
        double e = 0, rmin = 1e30, rmax = -1e30;
        for (int i = NG; i < NG + g.nx; ++i) {
            const double r = double(g.at(i, NG + 1).rho);
            e += std::fabs(
                r - exact::sample(L, R, (double(g.xc(i)) - 0.5) / tEnd)
                        .rho);
            rmin = std::min(rmin, r);
            rmax = std::max(rmax, r);
        }
        return std::tuple{e / N, rmin, rmax};
    };
    const auto [eW, rminW, rmaxW] = runSod(true);
    const auto [eM, rminM, rmaxM] = runSod(false);
    const bool bounded = rmaxW < 1.0 + 1e-3 && rminW > 0.125 - 1e-3;
    std::printf("gate 2 — Sod 400: L1 weno %.4e vs muscl %.4e (gate < "
                "1.6x), rho in [%.4f, %.4f] (gate: no over/undershoot "
                "beyond 1e-3)\n",
                eW, eM, rminW, rmaxW);
    return eW < 1.6 * eM && bounded;
}

// isentropic vortex (beta = 5) advected diagonally, exact = shifted IC
Cons vortexState(Real x, Real y, Real cx, Real cy) {
    constexpr Real beta = 5;
    const Real dx = x - cx, dy = y - cy;
    const Real r2 = dx * dx + dy * dy;
    const Real e = std::exp(Real(0.5) * (1 - r2));
    const Real du = beta / Real(2 * M_PI) * e;
    const Real T = 1 - (GAMMA - 1) * beta * beta /
                           (Real(8) * GAMMA * Real(M_PI * M_PI)) *
                           std::exp(1 - r2);
    const Real rho = std::pow(T, Real(1) / (GAMMA - 1));
    return toCons({rho, 1 - du * dy, 1 + du * dx,
                   rho * T});
}

double vortexError(int N, bool weno) {
    Grid g(N, N, 0, 0, 10, 10);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) = vortexState(g.xc(i), g.yc(j), 5, 5);
    ScratchW sw;
    Scratch2D sm;
    const double tEnd = 2;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt =
            std::min(maxStableDt(g, CFL, 0), Real(tEnd - t));
        if (weno) {
            stepWeno2D(g, dt, sw, fillPeriodic);
        } else {
            fillPeriodic(g);
            step2D(g, dt, sm);
        }
        t += dt;
    }
    double e = 0;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            e += std::fabs(
                double(g.at(i, j).rho) -
                double(vortexState(g.xc(i), g.yc(j), 7, 7).rho));
    return e / (double(N) * N);
}

bool gate3_vortex() {
    const double w32 = vortexError(32, true);
    const double w64 = vortexError(64, true);
    const double m64 = vortexError(64, false);
    const double ord = std::log2(w32 / w64);
    std::printf("gate 3 — vortex t=2: order %.2f (gate >= 2.7), L1 "
                "weno %.3e vs muscl %.3e at 64^2 (gate: weno < muscl/3)"
                "\n",
                ord, w64, m64);
    return ord >= 2.7 && w64 < m64 / 3;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_entropyOrder() && ok;
    ok = gate2_sod() && ok;
    ok = gate3_vortex() && ok;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

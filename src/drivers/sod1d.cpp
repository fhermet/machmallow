// Phase 1 driver: 1D Sod shock tube with MUSCL-Hancock + HLLC, validated
// against the exact Riemann solution. Runs a grid-convergence study and
// fails (non-zero exit) if the L1 error or convergence order regresses.

#include "core/Types.hpp"
#include "numerics/ExactRiemann.hpp"
#include "numerics/Hllc.hpp"
#include "numerics/Limiter.hpp"
#include "physics/Euler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace mm;

constexpr int NG = 2; // ghost cells per side
constexpr Real CFL = Real(0.8);

constexpr exact::State SOD_L = {1.0, 0.0, 1.0};
constexpr exact::State SOD_R = {0.125, 0.0, 0.1};
constexpr double SOD_X0 = 0.5;
constexpr double SOD_TEND = 0.2;

struct Solution {
    Real dx;
    std::vector<Cons> U; // size n + 2*NG
};

void fillGhostTransmissive(std::vector<Cons>& U, int n) {
    for (int g = 0; g < NG; ++g) {
        U[g] = U[NG];
        U[NG + n + g] = U[NG + n - 1];
    }
}

// One MUSCL-Hancock step; returns dt actually taken (capped at dtMax).
Real step(std::vector<Cons>& U, int n, Real dx, Real dtMax) {
    const int tot = n + 2 * NG;

    // CFL time step from max wave speed.
    Real smax = 0;
    for (int i = NG; i < NG + n; ++i) {
        const Prim w = toPrim(U[i]);
        smax = std::max(smax, std::fabs(w.u) + soundSpeed(w));
    }
    const Real dt = std::min(CFL * dx / smax, dtMax);

    // Limited slopes, then half-dt predictor on face values.
    std::vector<Cons> qL(tot), qR(tot); // time-advanced face states
    for (int i = 1; i < tot - 1; ++i) {
        const Cons dq = limitedSlope(U[i - 1], U[i], U[i + 1]);
        Cons l = U[i] - Real(0.5) * dq;
        Cons r = U[i] + Real(0.5) * dq;
        const Cons adv =
            (Real(0.5) * dt / dx) * (fluxX(toPrim(l)) - fluxX(toPrim(r)));
        qL[i] = l + adv;
        qR[i] = r + adv;
    }

    // HLLC fluxes at interfaces and conservative update.
    std::vector<Cons> F(tot);
    for (int i = NG - 1; i < NG + n; ++i) { // interface i+1/2
        F[i] = hllcFluxX(toPrim(qR[i]), toPrim(qL[i + 1]));
    }
    const Real lam = dt / dx;
    for (int i = NG; i < NG + n; ++i) {
        U[i] += lam * (F[i - 1] - F[i]);
    }
    return dt;
}

Solution runSod(int n) {
    Solution sol;
    sol.dx = Real(1.0 / n);
    sol.U.resize(n + 2 * NG);
    for (int i = 0; i < n; ++i) {
        const double xc = (i + 0.5) * sol.dx;
        const exact::State& s = (xc < SOD_X0) ? SOD_L : SOD_R;
        sol.U[NG + i] =
            toCons({Real(s.rho), Real(s.u), Real(0), Real(s.p)});
    }

    double t = 0;
    while (t < SOD_TEND) {
        fillGhostTransmissive(sol.U, n);
        t += step(sol.U, n, sol.dx, Real(SOD_TEND - t));
    }
    return sol;
}

double l1DensityError(const Solution& sol, int n, bool writeCsv) {
    double err = 0;
    FILE* f = nullptr;
    if (writeCsv) {
        std::filesystem::create_directories("out");
        const std::string path = "out/sod_" + std::to_string(n) + ".csv";
        f = std::fopen(path.c_str(), "w");
        std::fprintf(f, "x,rho,u,p,rho_exact\n");
    }
    for (int i = 0; i < n; ++i) {
        const double xc = (i + 0.5) * sol.dx;
        const exact::State ex =
            exact::sample(SOD_L, SOD_R, (xc - SOD_X0) / SOD_TEND);
        const Prim w = toPrim(sol.U[NG + i]);
        err += std::fabs(double(w.rho) - ex.rho) * sol.dx;
        if (f != nullptr) {
            std::fprintf(f, "%.8g,%.8g,%.8g,%.8g,%.8g\n", xc, double(w.rho),
                         double(w.u), double(w.p), ex.rho);
        }
    }
    if (f != nullptr) std::fclose(f);
    return err;
}

} // namespace

int main() {
    std::printf("Sod shock tube, MUSCL-Hancock + HLLC, t = %.2f\n", SOD_TEND);
    std::printf("%8s %12s %8s\n", "N", "L1(rho)", "order");

    const int grids[] = {100, 200, 400, 800, 1600};
    std::vector<double> errs;
    for (int n : grids) {
        const Solution sol = runSod(n);
        errs.push_back(l1DensityError(sol, n, /*writeCsv=*/true));
        const std::size_t k = errs.size();
        if (k > 1) {
            const double order = std::log2(errs[k - 2] / errs[k - 1]);
            std::printf("%8d %12.4e %8.2f\n", n, errs.back(), order);
        } else {
            std::printf("%8d %12.4e %8s\n", n, errs.back(), "-");
        }
    }

    // Regression gates: absolute error on the coarse grid and mean order.
    const double meanOrder =
        std::log2(errs.front() / errs.back()) / double(errs.size() - 1);
    std::printf("mean order: %.2f\n", meanOrder);
    if (errs.front() > 1.0e-2 || meanOrder < 0.7) {
        std::fprintf(stderr, "FAIL: convergence regression\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

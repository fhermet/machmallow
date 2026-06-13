// Convergence study: observed L1 order of the two schemes
// (MUSCL-Hancock + HLLC, and WENO5 + HLLC + SSP-RK3) on
//   1. a smooth grid-aligned entropy wave (exact = advected IC) — the
//      design spatial order shows here (WENO ~5, capped by RK3 at 3 at
//      fixed CFL once the spatial error drops below the temporal one;
//      MUSCL ~2, limiter-clipped at the extremum);
//   2. the 2D isentropic vortex (exact = advected IC) — the realized
//      multidimensional order (the dimension-by-dimension midpoint face
//      flux caps both schemes near 2; WENO keeps a smaller constant);
//   3. the Sod shock tube vs the exact Riemann solution — a problem
//      WITH discontinuities, where the L1 error is dominated by the
//      O(h) smearing of the shock and contact, so BOTH schemes drop to
//      first order regardless of their smooth-flow accuracy (the
//      classic result: high order buys a smaller constant, not a
//      steeper slope, once a discontinuity is present).
//
// Writes out/convergence.csv (problem, scheme, N, h, L1) for the
// plotting script and prints a per-problem table with the order
// measured between consecutive grids and the least-squares slope.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2D.hpp"
#include "solver/Weno2D.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);

void fillPeriodic(Grid& g) {
    fillPeriodicX(g);
    fillPeriodicY(g);
}
void fillSodBC(Grid& g) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillPeriodicY(g);
}

// Drive one scheme on a prepared grid to tEnd at fixed CFL.
template <class Fill>
void advance(Grid& g, double tEnd, bool weno, Fill&& fill) {
    ScratchW sw;
    Scratch2D sm;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt = std::min(maxStableDt(g, CFL, 0), Real(tEnd - t));
        if (weno) {
            stepWeno2D(g, dt, sw, fill);
        } else {
            fill(g);
            step2D(g, dt, sm);
        }
        t += dt;
    }
}

// ---- problem 1: smooth grid-aligned entropy wave --------------------------
// rho = 1 + 0.3 sin(2 pi x), u = 1, v = 0, p = 1. The entropy wave is
// advected exactly; exact(t) = IC shifted by u*t. One period on [0,1].
double entropyL1(int N, bool weno) {
    Grid g(N, 4, 0, 0, 1, Real(4.0) / N);
    const auto rhoOf = [](double x) {
        return 1.0 + 0.3 * std::sin(2 * M_PI * x);
    };
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) = toCons({Real(rhoOf(g.xc(i))), 1, 0, 1});
    advance(g, 1.0, weno, fillPeriodic);
    double e = 0;
    for (int i = NG; i < NG + g.nx; ++i)
        e += std::fabs(double(g.at(i, NG + 1).rho) -
                       rhoOf(double(g.xc(i)) - 1.0));
    return e / g.nx;
}

// ---- problem 2: 2D isentropic vortex --------------------------------------
Cons vortexState(Real x, Real y, Real cx, Real cy) {
    constexpr Real beta = 5;
    const Real dx = x - cx, dy = y - cy;
    const Real r2 = dx * dx + dy * dy;
    const Real du = beta / Real(2 * M_PI) * std::exp(Real(0.5) * (1 - r2));
    const Real T = 1 - (GAMMA - 1) * beta * beta /
                           (Real(8) * GAMMA * Real(M_PI * M_PI)) *
                           std::exp(1 - r2);
    const Real rho = std::pow(T, Real(1) / (GAMMA - 1));
    return toCons({rho, 1 - du * dy, 1 + du * dx, rho * T});
}
double vortexL1(int N, bool weno) {
    Grid g(N, N, 0, 0, 10, 10);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) = vortexState(g.xc(i), g.yc(j), 5, 5);
    advance(g, 2.0, weno, fillPeriodic); // travels (2,2): center 5->7
    double e = 0;
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            e += std::fabs(double(g.at(i, j).rho) -
                           double(vortexState(g.xc(i), g.yc(j), 7, 7).rho));
    return e / (double(N) * N);
}

// ---- problem 3: Sod shock tube vs exact Riemann ---------------------------
double sodL1(int N, bool weno) {
    const exact::State L{1, 0, 1}, R{0.125, 0, 0.1};
    const double tEnd = 0.2;
    Grid g(N, 4, 0, 0, 1, Real(4.0) / N);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) = toCons(g.xc(i) < Real(0.5)
                                    ? Prim{1, 0, 0, 1}
                                    : Prim{Real(0.125), 0, 0, Real(0.1)});
    advance(g, tEnd, weno, fillSodBC);
    double e = 0;
    for (int i = NG; i < NG + g.nx; ++i)
        e += std::fabs(
            double(g.at(i, NG + 1).rho) -
            exact::sample(L, R, (double(g.xc(i)) - 0.5) / tEnd).rho);
    return e / g.nx;
}

struct Row {
    std::string problem, scheme;
    int N;
    double h, L1;
};

struct Orders {
    double muscl = 0, weno = 0;
}; // pre-floor LSQ orders

Orders study(const char* name, double (*fn)(int, bool),
             const std::vector<int>& Ns, double domain,
             std::vector<Row>& out) {
    Orders ord{};
    std::printf("\n=== %s ===\n", name);
    for (bool weno : {false, true}) {
        const char* s = weno ? "weno5" : "muscl";
        std::printf("  %-6s  %8s  %12s  %8s\n", s, "N", "L1(rho)",
                    "order");
        double prevL1 = 0, prevH = 0;
        // LSQ fit only over the points still in the discretization
        // regime: once the error stops dropping by a clear factor it is
        // float32-roundoff-limited (more steps -> more accumulated
        // roundoff, the order can even go negative), so those points
        // would corrupt the measured slope.
        std::vector<double> logh, logL1;
        for (int N : Ns) {
            const double h = domain / N;
            const double L1 = fn(N, weno);
            const bool floored = prevL1 > 0 && L1 > Real(0.7) * prevL1;
            double ord = 0;
            if (prevL1 > 0)
                ord = std::log(prevL1 / L1) / std::log(prevH / h);
            std::printf("  %-6s  %8d  %12.4e  ", "", N, L1);
            if (prevL1 <= 0) std::printf("      —");
            else if (floored) std::printf(" (fp32)");
            else std::printf("%8.2f", ord);
            std::printf("\n");
            out.push_back({name, s, N, h, L1});
            if (!floored) { logh.push_back(std::log(h));
                            logL1.push_back(std::log(L1)); }
            prevL1 = L1;
            prevH = h;
        }
        const std::size_t n = logh.size();
        if (n >= 2) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (std::size_t k = 0; k < n; ++k) {
                sx += logh[k]; sy += logL1[k];
                sxx += logh[k] * logh[k]; sxy += logh[k] * logL1[k];
            }
            const double slope =
                (n * sxy - sx * sy) / (n * sxx - sx * sx);
            std::printf("  %-6s  LSQ order (pre-floor, %zu pts): "
                        "%.2f\n", s, n, slope);
            (weno ? ord.weno : ord.muscl) = slope;
        }
    }
    return ord;
}

} // namespace

int main() {
    std::vector<Row> rows;

    const Orders ent =
        study("entropy_wave", entropyL1, {8, 16, 32, 64, 128}, 1.0, rows);
    study("isentropic_vortex", vortexL1, {32, 64, 128, 256}, 10.0, rows);
    const Orders sod =
        study("sod", sodL1, {100, 200, 400, 800, 1600}, 1.0, rows);

    std::filesystem::create_directories("out");
    FILE* f = std::fopen("out/convergence.csv", "w");
    std::fprintf(f, "problem,scheme,N,h,L1\n");
    for (const Row& r : rows)
        std::fprintf(f, "%s,%s,%d,%.9e,%.9e\n", r.problem.c_str(),
                     r.scheme.c_str(), r.N, r.h, r.L1);
    std::fclose(f);
    std::printf("\nwrote out/convergence.csv (%zu rows)\n", rows.size());
    std::printf("plot: python3 tools/plot_convergence.py\n");

    // Robust sanity checks (well clear of the fp32 floor): MUSCL is
    // formally 2nd order on the smooth wave; the high-order WENO
    // reconstruction shows >3 before the floor; both collapse to ~1st
    // order on the discontinuous Sod tube.
    bool ok = ent.muscl > 1.6 && ent.muscl < 2.4 && ent.weno > 3.0 &&
              sod.muscl > 0.75 && sod.muscl < 1.15 && sod.weno > 0.75 &&
              sod.weno < 1.15;
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

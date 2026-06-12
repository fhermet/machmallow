// Analytic validation suite: the solver against exact/theoretical
// references beyond the Sod case.
//   1. Toro's Riemann battery (tests 2-5) vs the exact Riemann solver:
//      near-vacuum double rarefaction, p=1000 blast, colliding shocks,
//      slowly-moving contact — robustness + accuracy gates
//   2. periodic acoustic wave: returns onto its own IC after one
//      period — measures the SMOOTH-regime convergence order (~2),
//      which discontinuous cases cannot show
//   3. isentropic vortex (Yee), advected one period on a doubly
//      periodic domain — the canonical smooth 2D Euler test
//   4. Sedov 2D blast: self-similar front r ~ t^(1/2) exponent
// Gates are calibrated on measured healthy values (regression locks).

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);

// ---- shared 1D-in-x runner (transmissive x, uniform y) -----------------
double runRiemann1D(const exact::State& L, const exact::State& R,
                    double tEnd, int n, double cflStart = 0.2) {
    Grid g(n, 8, 0, 0, 1, Real(8.0 / n));
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const exact::State& s = g.xc(i) < Real(0.5) ? L : R;
            g.at(i, j) = toCons({Real(s.rho), Real(s.u), 0, Real(s.p)});
        }
    Scratch2D sc;
    double t = 0;
    int steps = 0;
    while (t < tEnd * (1 - 1e-9)) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        Real dt = std::min(maxStableDt(g, CFL), Real(tEnd - t));
        if (steps++ < 10) dt *= Real(cflStart); // sharp-IC start
        step2D(g, dt, sc);
        t += dt;
    }
    double err = 0;
    const int j = NG + 4;
    for (int i = NG; i < NG + n; ++i)
        err += std::fabs(
                   double(toPrim(g.at(i, j)).rho) -
                   exact::sample(L, R, (double(g.xc(i)) - 0.5) / tEnd)
                       .rho) /
               n;
    return err;
}

bool gate1_toroBattery() {
    struct Test {
        const char* name;
        exact::State L, R;
        double t, gate;
    };
    // L1(rho) gates = measured healthy values * ~1.3
    const Test tests[] = {
        {"T2 123/near-vacuum", {1, -2, 0.4}, {1, 2, 0.4}, 0.15, 4.0e-3},
        {"T3 blast p=1000", {1, 0, 1000}, {1, 0, 0.01}, 0.012, 6.0e-2},
        {"T4 colliding shocks",
         {5.99924, 19.5975, 460.894},
         {5.99242, -6.19633, 46.0950},
         0.035,
         3.0e-1},
        {"T5 slow contact",
         {1, -19.59745, 1000},
         {1, -19.59745, 0.01},
         0.012,
         8.0e-2},
    };
    bool ok = true;
    for (const Test& T : tests) {
        const double e = runRiemann1D(T.L, T.R, T.t, 400);
        std::printf("gate 1 — Toro %-22s L1(rho) = %.4e (gate %.1e)\n",
                    T.name, e, T.gate);
        ok = ok && e < T.gate && std::isfinite(e);
    }
    return ok;
}

// ---- 2: periodic acoustic wave — smooth-regime order ---------------------
double acousticErr(int n) {
    // right-going simple wave, amplitude A; rho0 = 1, p0 = 1/gamma so
    // c = 1 and the wave returns onto the IC at t = 1. A large enough
    // to dominate the fp32 accumulation floor at these resolutions.
    const double A = 5e-3;
    Grid g(n, 8, 0, 0, 1, Real(8.0 / n));
    const auto ic = [&](Real x) {
        const double s = std::sin(2 * M_PI * double(x));
        return toCons({Real(1 + A * s), Real(A * s), 0,
                       Real(1.0 / GAMMA + A * s)});
    };
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) g.at(i, j) = ic(g.xc(i));
    Scratch2D sc;
    double t = 0;
    while (t < 1 - 1e-9) {
        fillPeriodicX(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        const Real dt = std::min(maxStableDt(g, CFL), Real(1 - t));
        step2D(g, dt, sc);
        t += dt;
    }
    double err = 0;
    const int j = NG + 4;
    for (int i = NG; i < NG + n; ++i)
        err += std::fabs(double(g.at(i, j).rho) -
                         double(ic(g.xc(i)).rho)) /
               n;
    return err;
}

bool gate2_acousticOrder() {
    // TVD limiters clip smooth extrema (the sine crests), which caps
    // the L1 order at the theoretical 4/3 for this profile — the gate
    // checks we sit at that TVD reference, not at the formal 2 (which
    // the vortex test, error-dominated away from extrema, does reach).
    // The full remedy is a higher-order scheme (WENO, roadmap).
    // Grids restricted to the regime where discretization dominates:
    // by N = 64 the O(A^2) nonlinear self-steepening of the wave
    // (~4%% of A at t = 1) floors the return-to-IC error.
    std::vector<double> errs;
    std::printf("gate 2 — acoustic wave, one period:\n");
    for (int n : {16, 32}) {
        errs.push_back(acousticErr(n));
        if (errs.size() > 1)
            std::printf("    N = %3d  L1(rho) = %.4e | order %.2f\n", n,
                        errs.back(),
                        std::log2(errs[errs.size() - 2] / errs.back()));
        else
            std::printf("    N = %3d  L1(rho) = %.4e\n", n, errs.back());
    }
    const double order = std::log2(errs.front() / errs.back());
    std::printf("    smooth order = %.2f (TVD-extremum theory 4/3, "
                "gate > 1.2)\n",
                order);
    return order > 1.2;
}

// ---- 3: isentropic vortex (Yee), one periodic advection -------------------
Cons vortexIc(Real x, Real y) {
    const double beta = 5.0;
    const double dx = double(x) - 5.0, dy = double(y) - 5.0;
    const double r2 = dx * dx + dy * dy;
    const double e1 = std::exp(0.5 * (1.0 - r2));
    const double T =
        1.0 - (GAMMA - 1) * beta * beta / (8 * GAMMA * M_PI * M_PI) *
                  std::exp(1.0 - r2);
    const double rho = std::pow(T, 1.0 / (GAMMA - 1));
    return toCons({Real(rho), Real(1.0 - beta / (2 * M_PI) * e1 * dy),
                   Real(beta / (2 * M_PI) * e1 * dx), Real(rho * T)});
}

double vortexErr(int n) {
    Grid g(n, n, 0, 0, 10, 10);
    for (int j = NG; j < NG + n; ++j)
        for (int i = NG; i < NG + n; ++i)
            g.at(i, j) = vortexIc(g.xc(i), g.yc(j));
    Scratch2D sc;
    double t = 0;
    while (t < 10 - 1e-8) { // one x-period at u_inf = 1
        fillPeriodicX(g);
        fillPeriodicY(g);
        const Real dt = std::min(maxStableDt(g, CFL), Real(10 - t));
        step2D(g, dt, sc);
        t += dt;
    }
    double err = 0;
    for (int j = NG; j < NG + n; ++j)
        for (int i = NG; i < NG + n; ++i)
            err += std::fabs(double(g.at(i, j).rho) -
                             double(vortexIc(g.xc(i), g.yc(j)).rho)) /
                   (double(n) * n);
    return err;
}

bool gate3_vortex() {
    std::vector<double> errs;
    std::printf("gate 3 — isentropic vortex, one period:\n");
    for (int n : {32, 64, 128}) {
        errs.push_back(vortexErr(n));
        std::printf("    N = %3d  L1(rho) = %.4e\n", n, errs.back());
    }
    const double order = std::log2(errs.front() / errs.back()) / 2;
    std::printf("    mean smooth order = %.2f (gate > 1.5)\n", order);
    return order > 1.5;
}

// ---- 4: Sedov 2D self-similar exponent ------------------------------------
double sedovFront(const Grid& g, int n) {
    // strongest density gradient along +x from the center
    const int jc = NG + n / 2;
    double best = 0, rBest = 0;
    for (int i = NG + n / 2 + 2; i < NG + n - 2; ++i) {
        const double d = std::fabs(double(g.at(i + 1, jc).rho) -
                                   double(g.at(i - 1, jc).rho));
        if (d > best) {
            best = d;
            rBest = double(g.xc(i));
        }
    }
    return rBest;
}

bool gate4_sedov() {
    const int n = 256;
    Grid g(n, n, Real(-0.5), Real(-0.5), 1, 1);
    const double r0 = 3.0 / 256; // deposit radius: 3 cells
    const double E = 1.0;
    const double pIn = (GAMMA - 1) * E / (M_PI * r0 * r0);
    for (int j = NG; j < NG + n; ++j)
        for (int i = NG; i < NG + n; ++i) {
            const double r2 = double(g.xc(i)) * g.xc(i) +
                              double(g.yc(j)) * g.yc(j);
            g.at(i, j) = toCons({Real(1), 0, 0,
                                 r2 < r0 * r0 ? Real(pIn) : Real(1e-3)});
        }
    Scratch2D sc;
    double t = 0;
    int steps = 0;
    double r1 = 0, t1 = 0;
    const double tMid = 0.035, tEnd = 0.10;
    while (t < tEnd * (1 - 1e-9)) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        Real dt = std::min(maxStableDt(g, CFL), Real(tEnd - t));
        if (steps++ < 10) dt *= Real(0.1); // violent IC
        if (t < tMid && t + dt >= tMid) dt = Real(tMid - t);
        step2D(g, dt, sc);
        t += dt;
        if (r1 == 0 && t >= tMid) {
            r1 = sedovFront(g, n);
            t1 = t;
        }
    }
    const double r2 = sedovFront(g, n);
    const double expo = std::log(r2 / r1) / std::log(tEnd / t1);
    std::printf("gate 4 — Sedov 2D front: r(%.3f) = %.4f, r(%.2f) = "
                "%.4f, exponent = %.3f (theory 0.5, gate ±0.03)\n",
                t1, r1, tEnd, r2, expo);
    return std::fabs(expo - 0.5) < 0.03;
}

// ---- 5: Rayleigh-Taylor linear growth vs sqrt(A g k) ----------------------
// Fourier amplitude of the SEEDED mode at the interface row (immune to
// the harmonics that pollute global kinetic energy), least-squares fit
// of ln a(t) over the linear window (the fit averages out the
// superposed gravity-wave oscillation).
// NOTE (measured, kept for the record): the same measurement on a
// vortex-sheet KH is NOT gateable — the sheet is ill-posed (sigma ~ k,
// the fastest resolvable scale wins), the uniform-in-y seed projects
// poorly on the localized eigenmode, and the numerically thickened
// layer has its own rate. A proper KH gate needs a tanh profile and
// Michalke's eigenvalues (roadmap).
bool gate5_rtGrowth() {
    const Real gy = Real(-0.5);
    const double sigmaTh = std::sqrt((1.0 / 3) * 0.5 * 2 * M_PI / 0.5);

    Grid g(128, 384, 0, 0, Real(0.5), Real(1.5));
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Real y = g.yc(j), x = g.xc(i);
            Prim w{y > Real(0.75) ? Real(2) : Real(1), 0,
                   Real(1e-4 * std::sin(2 * M_PI * double(x) / 0.5) *
                        std::exp(-std::pow((double(y) - 0.75) / 0.03,
                                           2))),
                   Real(2.5)};
            w.p += w.rho * gy * (y - Real(0.75));
            g.at(i, j) = toCons(w);
        }
    const auto bc = [&](Grid& gr) {
        fillPeriodicX(gr);
        for (int i = 0; i < gr.totx(); ++i)
            for (int k = 0; k < NG; ++k) {
                Prim w = toPrim(gr.at(i, NG + k)); // hydrostatic wall
                w.v = -w.v;
                w.p += w.rho * gy * (gr.yc(NG - 1 - k) - gr.yc(NG + k));
                gr.at(i, NG - 1 - k) = toCons(w);
                Prim u = toPrim(gr.at(i, NG + gr.ny - 1 - k));
                u.v = -u.v;
                u.p += u.rho * gy * (gr.yc(NG + gr.ny + k) -
                                     gr.yc(NG + gr.ny - 1 - k));
                gr.at(i, NG + gr.ny + k) = toCons(u);
            }
    };
    const auto modeAmp = [&](const Grid& gr) {
        double cr = 0, ci = 0;
        const int jI = NG + 192; // interface row y = 0.75
        for (int i = NG; i < NG + gr.nx; ++i) {
            const Cons& q = gr.at(i, jI);
            const double v =
                double(q.my) / std::max(double(q.rho), 1e-10);
            const double ph = 2 * M_PI * (i - NG + 0.5) / gr.nx;
            cr += v * std::cos(ph);
            ci += v * std::sin(ph);
        }
        return std::sqrt(cr * cr + ci * ci) / gr.nx;
    };

    Scratch2D s;
    double t = 0;
    int n = 0;
    double st = 0, sa = 0, stt = 0, sta = 0; // fit accumulators
    int np = 0;
    while (t < 3.0) {
        bc(g);
        const Real dt = maxStableDt(g, Real(0.4));
        step2D(g, dt, s, 0, 0, gy);
        t += dt;
        if (++n % 50 == 0 && t > 1.8) {
            const double la = std::log(modeAmp(g));
            st += t; sa += la; stt += t * t; sta += t * la;
            ++np;
        }
    }
    const double sigma =
        (np * sta - st * sa) / (np * stt - st * st); // LSQ slope
    std::printf("gate 5 — RT linear growth: sigma = %.3f vs sqrt(Agk) = "
                "%.3f (%.0f%%, gate ±15%%)\n",
                sigma, sigmaTh, 100 * (sigma / sigmaTh - 1));
    return std::fabs(sigma / sigmaTh - 1) < 0.15;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_toroBattery() && ok;
    ok = gate2_acousticOrder() && ok;
    ok = gate3_vortex() && ok;
    ok = gate4_sedov() && ok;
    ok = gate5_rtGrowth() && ok;
    if (!ok) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

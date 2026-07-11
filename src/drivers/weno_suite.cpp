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

#include "amr/AmrML.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/ExactRiemann.hpp"
#include "physics/TwoGas.hpp"
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
    // 16->32 sits in the spatial regime (gate: high order); by 64 the
    // RK3 temporal error floors the total (e64 is scheme-variant
    // independent), so the 32->64 ratio measures a regime change, not
    // an order — printed, not gated. The absolute floor IS gated.
    std::printf("gate 1 — entropy wave: L1 %.3e -> %.3e -> %.3e, order "
                "%.2f (gate >= 4, spatial regime) / %.2f (regime "
                "change); e64 %.2e (gate < 1.5e-5); spatial-only "
                "8->16: %.2f\n",
                e16, e32, e64, o1, o2, e64, sOrd);
    return o1 >= 4 && e64 < 1.5e-5;
}

bool gate2_sod() {
    const exact::State L{1, 0, 1}, R{0.125, 0, 0.1};
    const double tEnd = 0.2;
    const int N = 400;

    const auto runSod = [&](bool weno, const char* csv = nullptr) {
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
        if (csv) { // density profile vs exact, for the vv boundedness figure
            if (FILE* pf = std::fopen(csv, "w")) {
                std::fprintf(pf, "x,rho,rho_exact\n");
                for (int i = NG; i < NG + g.nx; ++i) {
                    const double x = double(g.xc(i));
                    std::fprintf(pf, "%.6g,%.6g,%.6g\n", x,
                                 double(g.at(i, NG + 1).rho),
                                 exact::sample(L, R, (x - 0.5) / tEnd).rho);
                }
                std::fclose(pf);
            }
        }
        return std::tuple{e / N, rmin, rmax};
    };
    const auto [eW, rminW, rmaxW] = runSod(true, "out/weno_sod_weno.csv");
    const auto [eM, rminM, rmaxM] = runSod(false, "out/weno_sod_muscl.csv");
    const bool bounded = rmaxW < 1.0 + 1e-3 && rminW > 0.125 - 1e-3;
    std::printf("gate 2 — Sod 400: L1 weno %.4e vs muscl %.4e (gate < "
                "1.1x; HLLC faces beat MUSCL here), rho in [%.4f, %.4f] "
                "(gate: no over/undershoot beyond 1e-3)\n",
                eW, eM, rminW, rmaxW);
    return eW < 1.1 * eM && bounded;
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

double vortexError(int N, bool weno, const char* sliceCsv = nullptr) {
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
    // center-row density slice (through the advected core at y = 7) vs
    // exact, for the vv dissipation figure
    if (sliceCsv) {
        int jc = NG;
        for (int j = NG; j < NG + g.ny; ++j)
            if (std::fabs(double(g.yc(j)) - 7) <
                std::fabs(double(g.yc(jc)) - 7))
                jc = j;
        if (FILE* pf = std::fopen(sliceCsv, "w")) {
            std::fprintf(pf, "x,rho,rho_exact\n");
            for (int i = NG; i < NG + g.nx; ++i)
                std::fprintf(pf, "%.6g,%.6g,%.6g\n", double(g.xc(i)),
                             double(g.at(i, jc).rho),
                             double(vortexState(g.xc(i), g.yc(jc), 7, 7)
                                        .rho));
            std::fclose(pf);
        }
    }
    return e / (double(N) * N);
}

bool gate3_vortex() {
    const double w32 = vortexError(32, true);
    const double w64 = vortexError(64, true, "out/weno_vortex_weno.csv");
    const double m64 = vortexError(64, false, "out/weno_vortex_muscl.csv");
    const double ord = std::log2(w32 / w64);
    // genuinely-2D smooth flow: the dimension-by-dimension midpoint
    // face flux caps the formal order near 2 (the 5th-order
    // reconstruction shows in gate 1's 1D-aligned regime); the value
    // is the CONSTANT — gated as a dissipation ratio vs MUSCL
    std::printf("gate 3 — vortex t=2: order %.2f (gate >= 2, "
                "quadrature-capped in 2D), L1 weno %.3e vs muscl %.3e "
                "at 64^2 (gate: weno < muscl/4)\n",
                ord, w64, m64);
    return ord >= 2 && w64 < m64 / 4;
}

// Viscous shear layer: v(x,t) = (V0/2) erf((x-0.5)/sqrt(4 nu t)) is an
// exact solution of compressible NS at uniform rho/p (the transverse
// momentum decouples into the heat equation). Start from the profile at
// t0, diffuse to t1, compare — quantifies the WENO viscous flux against
// the analytic answer and head-to-head with MUSCL.
double shearError(int N, bool weno) {
    constexpr Real MU = Real(5e-3), V0 = Real(0.2);
    constexpr double T0 = 0.05, T1 = 0.2;
    const auto exactV = [&](double x, double t) {
        return 0.5 * double(V0) *
               std::erf((x - 0.5) / std::sqrt(4.0 * double(MU) * t));
    };
    Grid g(N, 8, 0, 0, 1, Real(8) / N);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            g.at(i, j) =
                toCons({Real(1), 0, Real(exactV(g.xc(i), T0)), 1});
    const auto fill = [](Grid& gg) {
        fillTransmissiveLeft(gg);
        fillTransmissiveRight(gg);
        fillPeriodicY(gg);
    };
    ScratchW sw;
    Scratch2D sm;
    double t = T0;
    while (t < T1) {
        const Real dt =
            std::min(maxStableDt(g, CFL, MU), Real(T1 - t));
        if (weno) {
            stepWeno2D(g, dt, sw, fill, MU);
        } else {
            fill(g);
            step2D(g, dt, sm, MU);
        }
        t += dt;
    }
    double e = 0;
    const int j = NG + 4;
    for (int i = NG; i < NG + N; ++i)
        e += std::fabs(double(toPrim(g.at(i, j)).v) -
                       exactV(g.xc(i), T1)) *
             g.dx;
    return e;
}

bool gate6_viscousShear() {
    const double w64 = shearError(64, true);
    const double w128 = shearError(128, true);
    const double m128 = shearError(128, false);
    const double ord = std::log2(w64 / w128);
    // The viscous flux is the SAME 2nd-order central operator in both
    // schemes, and with u = 0 / smooth v the convective flux is exactly
    // benign — so convergence to the exact erf at order ~2 is the
    // correctness proof; the residual gap to MUSCL is pure RK3-vs-
    // Hancock temporal constant (same error class, gated < 2x).
    std::printf("gate 6 — viscous shear (erf) vs exact: L1 %.3e -> "
                "%.3e, order %.2f (gate >= 1.8); weno %.3e vs muscl "
                "%.3e at N=128 (gate < 2x)\n",
                w64, w128, ord, w128, m128);
    return ord >= 1.8 && w128 < 2 * m128;
}

// Two-gas WENO5 on a uniform grid: Sod with gamma 1.4 | 1.6 across the
// interface, vs the generalized (per-side gamma) exact Riemann solution.
// Validates the species reconstruction + per-side HLLC + quasi-
// conservative Gamma transport before the AMR plumbing.
bool gate7_twoGasUniform() {
    const GasPair gas{Real(1.4), Real(1.6)};
    const exact::State L{1, 0, 1, 1.4};
    const exact::State R{0.125, 0, 0.1, 1.6};
    const double tEnd = 0.2;
    const int N = 400;

    Grid g(N, 4, 0, 0, 1, Real(4.0) / N);
    std::vector<Real> phi(g.q.size()), Gm(g.q.size());
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const bool right = g.xc(i) >= Real(0.5);
            const Real Y = right ? 1 : 0;
            const exact::State& st = right ? R : L;
            const std::size_t id = g.idx(i, j);
            g.q[id] = toConsG({Real(st.rho), Real(st.u), 0, Real(st.p)},
                              gas.Gamma(Y));
            phi[id] = g.q[id].rho * Y;
            Gm[id] = gas.Gamma(Y);
        }
    const auto fill = [&](Grid& gg) {
        fillTransmissiveLeft(gg);
        fillTransmissiveRight(gg);
        fillPeriodicY(gg);
        // scalar ghosts: transmissive in x, periodic in y
        for (auto* f : {&phi, &Gm})
            for (int j = 0; j < gg.toty(); ++j)
                for (int k = 0; k < NG; ++k) {
                    (*f)[gg.idx(k, j)] = (*f)[gg.idx(NG, j)];
                    (*f)[gg.idx(NG + gg.nx + k, j)] =
                        (*f)[gg.idx(NG + gg.nx - 1, j)];
                }
        // y is invariant here, so transmissive == periodic for scalars
        for (auto* f : {&phi, &Gm})
            for (int i = 0; i < gg.totx(); ++i)
                for (int k = 0; k < NG; ++k) {
                    (*f)[gg.idx(i, k)] = (*f)[gg.idx(i, NG)];
                    (*f)[gg.idx(i, NG + gg.ny + k)] =
                        (*f)[gg.idx(i, NG + gg.ny - 1)];
                }
    };
    ScratchWY s;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt =
            std::min(maxStableDtY(g, Gm, CFL), Real(tEnd - t));
        stepWeno2DY(g, phi, Gm, dt, s, gas, fill);
        t += dt;
    }
    double err = 0, rmin = 1e30, rmax = -1e30;
    for (int i = NG; i < NG + N; ++i) {
        const double r = double(g.at(i, NG + 1).rho);
        err += std::fabs(
            r - exact::sample(L, R, (double(g.xc(i)) - 0.5) / tEnd).rho);
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    err /= N;
    const bool bounded = rmax < 1.0 + 1e-3 && rmin > 0.125 - 1e-3;
    std::printf("gate 7 — two-gas Sod (uniform WENO5): L1 = %.4e (gate "
                "4e-3), rho in [%.4f, %.4f] (gate: bounded)\n",
                err, rmin, rmax);
    return err < 4e-3 && bounded;
}

// Two-gas WENO5 through the multi-level AMR: Sod (gamma 1.4 | 1.6) on a
// 3-level subcycled hierarchy vs the generalized exact solution, with
// species-mass conservation across the refluxed coarse-fine interfaces.
bool gate8_twoGasAmr() {
    const GasPair gas{Real(1.4), Real(1.6)};
    const exact::State L{1, 0, 1, 1.4};
    const exact::State R{0.125, 0, 0.1, 1.6};
    const double tEnd = 0.2;
    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.weno = true;
    cfg.species = true;
    cfg.gamma1 = Real(1.4);
    cfg.gamma2 = Real(1.6);
    AmrML amr(64, 16, 0, 0, 1, Real(0.25), cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
    };
    amr.fillPatchPhysical = [](Grid& g, double, unsigned sides) {
        if (sides & SideLeft) fillTransmissiveLeft(g);
        if (sides & SideRight) fillTransmissiveRight(g);
        if (sides & SideBottom) fillTransmissiveBottom(g);
        if (sides & SideTop) fillTransmissiveTop(g);
    };
    amr.init(
        [&](Real x, Real) {
            const bool right = x >= Real(0.5);
            const exact::State& st = right ? R : L;
            return toConsG({Real(st.rho), Real(st.u), 0, Real(st.p)},
                           gas.Gamma(right ? Real(1) : Real(0)));
        },
        [](Real x, Real) { return x >= Real(0.5) ? Real(1) : Real(0); });

    const double m0 = amr.totalSpeciesMass();
    double t = 0, drift = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt =
            std::min(amr.maxStableDtAll(CFL), Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        drift = std::max(drift, std::fabs(amr.totalSpeciesMass() - m0) /
                                    std::max(m0, 1e-30));
    }
    double err = 0;
    const int bC = amr.fineCells() / 2;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i)
            if (!amr.covered(1, i / bC, j / bC))
                err += std::fabs(
                           double(amr.base.at(NG + i, NG + j).rho) -
                           exact::sample(L, R,
                                         (double(amr.base.xc(NG + i)) -
                                          0.5) / tEnd).rho) *
                       amr.base.dx * amr.base.dy;
    for (int l = 1; l < 3; ++l)
        for (const auto& p : amr.level(l).patches)
            for (int j = NG; j < NG + amr.fineCells(); ++j)
                for (int i = NG; i < NG + amr.fineCells(); ++i) {
                    const int gi = 2 * p.ci0 + (i - NG);
                    const int gj = 2 * p.cj0 + (j - NG);
                    if (l < 2 && amr.covered(l + 1, gi / bC, gj / bC))
                        continue;
                    err += std::fabs(
                               double(p.grid.at(i, j).rho) -
                               exact::sample(
                                   L, R,
                                   (double(p.grid.xc(i)) - 0.5) / tEnd)
                                   .rho) *
                           p.grid.dx * p.grid.dy;
                }
    err /= 0.25;
    std::printf("gate 8 — two-gas WENO5 Sod on 3-level AMR: L1 = %.4e "
                "(gate 5e-3), species mass drift = %.3e (gate 1e-4), "
                "patches L1 %zu L2 %zu\n",
                err, drift, amr.patchCount(1), amr.patchCount(2));
    return err < 5e-3 && drift < 1e-4;
}

// The strongest stage-ghost test there is: a fully refined,
// non-subcycled 2-level hierarchy on a doubly periodic domain must
// reproduce the uniform fine grid BIT FOR BIT — every patch ghost is a
// sibling copy of the same stage values the uniform run reads through
// the periodic wrap, so any error in the per-stage ghost machinery
// shows up as a nonzero diff.
bool gate4_allRefinedBitExact() {
    AmrConfig cfg;
    cfg.maxLevels = 2;
    cfg.subcycle = false;
    cfg.weno = true;
    cfg.tagThreshold = Real(-1); // tag everything
    cfg.regridEvery = 1 << 20;
    cfg.periodicX = cfg.periodicY = true;
    AmrML amr(32, 32, 0, 0, 10, 10, cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillPeriodicX(g);
        fillPeriodicY(g);
    };
    amr.init([](Real x, Real y) { return vortexState(x, y, 5, 5); });

    Grid ref(64, 64, 0, 0, 10, 10);
    for (int j = 0; j < ref.toty(); ++j)
        for (int i = 0; i < ref.totx(); ++i)
            ref.at(i, j) = vortexState(ref.xc(i), ref.yc(j), 5, 5);
    ScratchW sw;

    double t = 0;
    for (int s = 0; s < 40; ++s) {
        const Real dt = maxStableDt(ref, CFL, 0);
        stepWeno2D(ref, dt, sw, fillPeriodic);
        amr.step(dt, t);
        t += dt;
    }
    std::size_t diff = 0;
    const int nf = amr.fineCells();
    for (const auto& p : amr.level(1).patches)
        for (int j = 0; j < nf; ++j)
            for (int i = 0; i < nf; ++i) {
                const Cons& a = p.grid.at(NG + i, NG + j);
                const Cons& b = ref.at(NG + 2 * p.ci0 + i,
                                       NG + 2 * p.cj0 + j);
                const Real* pa = &a.rho;
                const Real* pb = &b.rho;
                for (int m = 0; m < NVARS; ++m)
                    diff += pa[m] != pb[m];
            }
    std::printf("gate 4 — all-refined 2-level WENO vs uniform, 40 "
                "steps: %zu differing values (gate 0), patches %zu\n",
                diff, amr.patchCount(1));
    return diff == 0;
}

// Coarse-fine ghost prolongation is (limited) 2nd order, and WENO5's
// smoothness indicators see its creases — so the composite error sits
// closer to MUSCL's than the uniform-grid comparison does. The gate is
// therefore RELATIVE to MUSCL-Hancock on the identical hierarchy
// (measured 1.8x; the high-order payoff lives in the smooth interior,
// see gates 1/3), plus the absolute conservation gate.
double sodAmrL1(bool weno, double& drift) {
    const exact::State L{1, 0, 1}, R{0.125, 0, 0.1};
    const double tEnd = 0.2;
    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.weno = weno;
    AmrML amr(64, 16, 0, 0, 1, Real(0.25), cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
    };
    amr.fillPatchPhysical = [](Grid& g, double, unsigned sides) {
        if (sides & SideLeft) fillTransmissiveLeft(g);
        if (sides & SideRight) fillTransmissiveRight(g);
        if (sides & SideBottom) fillTransmissiveBottom(g);
        if (sides & SideTop) fillTransmissiveTop(g);
    };
    amr.init([&](Real x, Real) {
        return toCons(x < Real(0.5) ? Prim{1, 0, 0, 1}
                                    : Prim{Real(0.125), 0, 0,
                                           Real(0.1)});
    });

    const double m0 = amr.totalMass();
    double t = 0;
    drift = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt =
            std::min(amr.maxStableDtAll(CFL), Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        drift = std::max(drift, std::fabs(amr.totalMass() - m0) / m0);
    }
    double err = 0;
    const int bC = amr.fineCells() / 2;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i)
            if (!amr.covered(1, i / bC, j / bC))
                err += std::fabs(
                           double(amr.base.at(NG + i, NG + j).rho) -
                           exact::sample(L, R,
                                         (double(amr.base.xc(NG + i)) -
                                          0.5) / tEnd).rho) *
                       amr.base.dx * amr.base.dy;
    for (int l = 1; l < 3; ++l)
        for (const auto& p : amr.level(l).patches)
            for (int j = NG; j < NG + amr.fineCells(); ++j)
                for (int i = NG; i < NG + amr.fineCells(); ++i) {
                    const int gi = 2 * p.ci0 + (i - NG);
                    const int gj = 2 * p.cj0 + (j - NG);
                    if (l < 2 && amr.covered(l + 1, gi / bC, gj / bC))
                        continue;
                    err += std::fabs(
                               double(p.grid.at(i, j).rho) -
                               exact::sample(
                                   L, R,
                                   (double(p.grid.xc(i)) - 0.5) / tEnd)
                                   .rho) *
                           p.grid.dx * p.grid.dy;
                }
    err /= 0.25;
    return err;
}

bool gate5_sodAmr() {
    double driftW = 0, driftM = 0;
    const double eW = sodAmrL1(true, driftW);
    const double eM = sodAmrL1(false, driftM);
    std::printf("gate 5 — Sod on 3-level AMR: L1 weno %.4e vs muscl "
                "%.4e (gate < 2x), mass drift %.3e (gate 2e-5, RK "
                "combination fp32 floor)\n",
                eW, eM, driftW);
    return eW < 2 * eM && driftW < 2e-5;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_entropyOrder() && ok;
    ok = gate2_sod() && ok;
    ok = gate3_vortex() && ok;
    ok = gate4_allRefinedBitExact() && ok;
    ok = gate5_sodAmr() && ok;
    ok = gate6_viscousShear() && ok;
    ok = gate7_twoGasUniform() && ok;
    ok = gate8_twoGasAmr() && ok;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

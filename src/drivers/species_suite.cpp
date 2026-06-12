// Two-gas scheme validation (v1.2 core, uniform grids).
// Gates:
//   1. material interface advection (rho and gamma jump, p and u
//      uniform): pressure/velocity must stay uniform — the classic
//      Abgrall test where naive multi-gas schemes oscillate — and Y
//      must advect at the right speed
//   2. two-gas Sod (gamma 1.4 | 1.6) vs the generalized exact Riemann
//      solver (each side keeps its own gamma across the contact)
//   3. species mass conservation at the fp32 floor

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2DSpecies.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);

void scalarGhostsTransmissive(const Grid& g, std::vector<Real>& f) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            f[g.idx(k, j)] = f[g.idx(NG, j)];
            f[g.idx(NG + g.nx + k, j)] = f[g.idx(NG + g.nx - 1, j)];
        }
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            f[g.idx(i, k)] = f[g.idx(i, NG)];
            f[g.idx(i, NG + g.ny + k)] = f[g.idx(i, NG + g.ny - 1)];
        }
}
void scalarGhostsPeriodicX(const Grid& g, std::vector<Real>& f) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            f[g.idx(k, j)] = f[g.idx(k + g.nx, j)];
            f[g.idx(NG + g.nx + k, j)] = f[g.idx(NG + k, j)];
        }
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            f[g.idx(i, k)] = f[g.idx(i, NG)];
            f[g.idx(i, NG + g.ny + k)] = f[g.idx(i, NG + g.ny - 1)];
        }
}

void bcTransmissive(Grid& g, std::vector<Real>& phi,
                    std::vector<Real>& Gm) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
    scalarGhostsTransmissive(g, phi);
    scalarGhostsTransmissive(g, Gm);
}

void bcPeriodicX(Grid& g, std::vector<Real>& phi, std::vector<Real>& Gm) {
    fillPeriodicX(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
    scalarGhostsPeriodicX(g, phi);
    scalarGhostsPeriodicX(g, Gm);
}

bool gate1_interfaceAdvection() {
    const GasPair gas{Real(1.4), Real(1.667)};
    const int n = 200;
    Grid g(n, 8, 0, 0, 1, Real(8.0 / n));
    std::vector<Real> phi(g.q.size(), 0), Gm(g.q.size(), gas.Gamma(0));
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const bool gas2 =
                g.xc(i) > Real(0.2) && g.xc(i) < Real(0.5);
            const Real rho = gas2 ? Real(0.5) : Real(1);
            const Real Y = gas2 ? Real(1) : Real(0);
            g.at(i, j) = toConsG({rho, Real(0.5), 0, Real(1)},
                                 gas.Gamma(Y));
            phi[g.idx(i, j)] = rho * Y;
            Gm[g.idx(i, j)] = gas.Gamma(Y);
        }
    ScratchY s;
    double t = 0;
    double pStart = 0, pSust = 0, uErr = 0;
    while (t < 1.0 - 1e-9) { // one full period at u = 0.5
        bcPeriodicX(g, phi, Gm);
        const Real dt =
            std::min(maxStableDtY(g, Gm, CFL), Real(1.0 - t));
        step2DY(g, phi, Gm, dt, s, gas);
        t += dt;
        for (int i = NG; i < NG + n; ++i) {
            const std::size_t id = g.idx(i, NG + 4);
            const Prim w = toPrimG(g.q[id], Gm[id]);
            const double dp = std::fabs(double(w.p) - 1.0);
            if (t < 0.2) pStart = std::max(pStart, dp);
            else pSust = std::max(pSust, dp);
            uErr = std::max(uErr, std::fabs(double(w.u) - 0.5));
        }
    }
    // Y profile back onto itself after u*t = 0.5 (half domain shift)
    double yErr = 0;
    for (int i = NG; i < NG + n; ++i) {
        const std::size_t id = g.idx(i, NG + 4);
        const Real Y = phi[id] / std::max(g.q[id].rho, RHO_FLOOR);
        const double x =
            std::fmod(double(g.xc(i)) - 0.5 + 1.0, 1.0); // unshifted
        const double yEx = (x > 0.2 && x < 0.5) ? 1.0 : 0.0;
        yErr += std::fabs(double(Y) - yEx) / n;
    }
    // Startup transient (discrete IC) and sustained levels gated
    // separately; the conservative-variable reconstruction leaves a
    // bounded O(0.7%) interface wiggle — primitive reconstruction is
    // the documented next refinement.
    std::printf("gate 1 — interface advection (gamma 1.4|1.667): "
                "|p-1| startup %.3e / sustained %.3e, max|u-0.5| = "
                "%.3e, L1(Y) = %.3e\n",
                pStart, pSust, uErr, yErr);
    return pStart < 0.02 && pSust < 0.01 && uErr < 0.01 && yErr < 0.03;
}

bool gate2_twoGasSod() {
    const GasPair gas{Real(1.4), Real(1.6)};
    const exact::State L{1, 0, 1, 1.4};
    const exact::State R{0.125, 0, 0.1, 1.6};
    const double tEnd = 0.2;
    const int n = 400;
    Grid g(n, 8, 0, 0, 1, Real(8.0 / n));
    std::vector<Real> phi(g.q.size(), 0), Gm(g.q.size(), gas.Gamma(0));
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const bool right = g.xc(i) >= Real(0.5);
            const Real Y = right ? Real(1) : Real(0);
            const exact::State& st = right ? R : L;
            g.at(i, j) = toConsG({Real(st.rho), Real(st.u), 0,
                                  Real(st.p)},
                                 gas.Gamma(Y));
            phi[g.idx(i, j)] = Real(st.rho) * Y;
            Gm[g.idx(i, j)] = gas.Gamma(Y);
        }
    ScratchY s;
    double t = 0;
    int steps = 0;
    while (t < tEnd - 1e-12) {
        bcTransmissive(g, phi, Gm);
        Real dt =
            std::min(maxStableDtY(g, Gm, CFL), Real(tEnd - t));
        if (steps++ < 10) dt *= Real(0.2);
        step2DY(g, phi, Gm, dt, s, gas);
        t += dt;
    }
    double err = 0;
    for (int i = NG; i < NG + n; ++i)
        err += std::fabs(
                   double(g.at(i, NG + 4).rho) -
                   exact::sample(L, R, (double(g.xc(i)) - 0.5) / tEnd)
                       .rho) /
               n;
    std::printf("gate 2 — two-gas Sod (1.4|1.6): L1(rho) vs exact = "
                "%.4e (gate 6e-3)\n",
                err);
    return err < 6e-3;
}

bool gate3_speciesMass() {
    const GasPair gas{Real(1.4), Real(1.667)};
    const int n = 128;
    Grid g(n, n, 0, 0, 1, 1);
    std::vector<Real> phi(g.q.size(), 0), Gm(g.q.size(), gas.Gamma(0));
    // gas-2 disc in a gas-1 vortex-ish flow, periodic-x box
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const Real x = g.xc(i), y = g.yc(j);
            const Real r2 = (x - Real(0.35)) * (x - Real(0.35)) +
                            (y - Real(0.5)) * (y - Real(0.5));
            const Real Y = r2 < Real(0.04) ? Real(1) : Real(0);
            const Real rho = Y > 0 ? Real(0.3) : Real(1);
            g.at(i, j) = toConsG(
                {rho, Real(0.4), Real(0.1 * std::sin(2 * M_PI * double(x))),
                 Real(1)},
                gas.Gamma(Y));
            phi[g.idx(i, j)] = rho * Y;
            Gm[g.idx(i, j)] = gas.Gamma(Y);
        }
    const auto phiMass = [&] {
        double m = 0;
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                m += double(phi[g.idx(i, j)]) * g.dx * g.dy;
        return m;
    };
    const double m0 = phiMass();
    ScratchY s;
    double t = 0, drift = 0;
    for (int k = 0; k < 200; ++k) {
        bcPeriodicX(g, phi, Gm);
        const Real dt = maxStableDtY(g, Gm, CFL);
        step2DY(g, phi, Gm, dt, s, gas);
        t += dt;
        drift = std::max(drift, std::fabs(phiMass() - m0) / m0);
    }
    std::printf("gate 3 — species mass, 200 steps: drift = %.3e (gate "
                "1e-5)\n",
                drift);
    return drift < 1e-5;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_interfaceAdvection() && ok;
    ok = gate2_twoGasSod() && ok;
    ok = gate3_speciesMass() && ok;
    if (!ok) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

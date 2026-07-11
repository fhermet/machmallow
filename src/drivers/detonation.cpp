// v1.5: Strang-split reacting Euler (1D) -> Chapman-Jouguet detonation.
//
// The reaction couples to the flow by Strang splitting:
//   R(dt/2) . A(dt) . R(dt/2)
// where A is the hyperbolic step (reuses the validated two-gas MUSCL
// step2DY with gamma1 = gamma2, so the progress variable lambda rides as
// phi = rho*lambda with a constant Gamma) and R is the per-cell reaction
// source react() (constant volume: holds rho, momentum fixed; raises E
// by the heat release, advances lambda).
//
// A self-sustaining detonation initiated by a hot driver relaxes to the
// Chapman-Jouguet state: its leading shock propagates at D_CJ, which
// depends only on (q, gamma, upstream state) — not on the reaction rate.
// The exact D_CJ is solved here from the Rankine-Hugoniot relations with
// heat release plus the CJ sonic condition (no remembered formula):
//   given D, the burnt CJ state has
//     w1 = gamma/(gamma+1) * (p0/(rho0 D) + D)      [mass + momentum + CJ]
//   and energy balance closes as F(D) = 0 with
//     F = c0^2/(g-1) + D^2/2 + q - w1^2 (g+1)/(2(g-1)).
// (Strong-q limit reproduces D_CJ -> sqrt(2(g^2-1) q), the known result.)

#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Reaction.hpp"
#include "physics/TwoGas.hpp"
#include "solver/Muscl2DSpecies.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);
constexpr Real GAM = GAMMA; // 1.4
constexpr Real RHO0 = 1, P0 = 1;

double cjSpeed(double q, double gamma, double rho0, double p0) {
    const double c0sq = gamma * p0 / rho0;
    const auto F = [&](double D) {
        const double w1 =
            gamma / (gamma + 1) * (p0 / (rho0 * D) + D);
        return c0sq / (gamma - 1) + 0.5 * D * D + q -
               w1 * w1 * (gamma + 1) / (2 * (gamma - 1));
    };
    // F(c0) = q > 0, F -> -inf for large D: bisect.
    double lo = std::sqrt(c0sq), hi = 10 * std::sqrt(2 * q + c0sq);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (F(mid) > 0) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Reaction half-step (constant volume) over the interior cells: extract
// internal energy and progress from (q, phi), advance, write back.
void reactGrid(Grid& g, std::vector<Real>& phi, Real dt,
               const Reaction& r) {
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const std::size_t id = g.idx(i, j);
            Cons& c = g.q[id];
            const Real rho = std::max(c.rho, RHO_FLOOR);
            const Real ke = Real(0.5) * (c.mx * c.mx + c.my * c.my) / rho;
            Real eInt = (c.E - ke) / rho;
            Real lam = std::clamp(phi[id] / rho, Real(0), Real(1));
            react(eInt, lam, dt, r, GAM);
            c.E = rho * eInt + ke;
            phi[id] = rho * lam;
        }
}

void fillBC(Grid& g, std::vector<Real>& phi, std::vector<Real>& gm) {
    fillReflectiveLeft(g); // closed tube: Taylor rarefaction -> CJ
    fillTransmissiveRight(g);
    fillPeriodicY(g);
    for (auto* f : {&phi, &gm})
        for (int j = 0; j < g.toty(); ++j)
            for (int k = 0; k < NG; ++k) {
                (*f)[g.idx(k, j)] = (*f)[g.idx(NG, j)];
                (*f)[g.idx(NG + g.nx + k, j)] =
                    (*f)[g.idx(NG + g.nx - 1, j)];
            }
    for (auto* f : {&phi, &gm})
        for (int i = 0; i < g.totx(); ++i)
            for (int k = 0; k < NG; ++k) {
                (*f)[g.idx(i, k)] = (*f)[g.idx(i, NG)];
                (*f)[g.idx(i, NG + g.ny + k)] = (*f)[g.idx(i, NG + k)];
            }
}

// Least-squares leading-shock speed from (t, x_front) over a window.
double fitSpeed(const std::vector<double>& ts,
                const std::vector<double>& xs, double xlo, double xhi,
                int& n) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    n = 0;
    for (std::size_t k = 0; k < ts.size(); ++k)
        if (xs[k] > xlo && xs[k] < xhi) {
            sx += ts[k]; sy += xs[k];
            sxx += ts[k] * ts[k]; sxy += ts[k] * xs[k];
            ++n;
        }
    return n >= 2 ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : 0;
}

// The same closed-tube CJ detonation on a 2-level AMR hierarchy, the
// refinement tracking the shock + reaction zone (tagged on the density
// gradient). Validates the Strang reaction through the recursive AMR
// step; returns the measured leading-shock speed.
double runAmr(const Reaction& r, const GasPair& gas, double q, double L,
              double Dcj) {
    AmrConfig cfg;
    cfg.maxLevels = 3;        // base 1/500 -> finest 1/2000 (= uniform)
    cfg.subcycle = false;     // all levels at the finest dt
    cfg.react = true;         // implies species (phi = rho*lambda)
    cfg.gamma1 = cfg.gamma2 = GAM;
    cfg.reaction = r;
    cfg.tagThreshold = Real(0.10); // the shock's density jump tags it
    cfg.regridEvery = 2;           // track the fast-moving front
    const int nb = 512, ny = 8;
    AmrML amr(nb, ny, 0, 0, L, Real(ny) * (L / nb), cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);
        fillTransmissiveRight(g);
        fillPeriodicY(g);
    };
    amr.fillPatchPhysical = [](Grid& g, double, unsigned s) {
        if (s & SideLeft) fillReflectiveLeft(g);
        if (s & SideRight) fillTransmissiveRight(g);
    };
    amr.init(
        [&](Real x, Real) {
            const bool drive = x < Real(0.2);
            return toConsG({RHO0, 0, 0, drive ? Real(30) : P0},
                           gas.Gamma(0));
        },
        [](Real x, Real) { return x < Real(0.2) ? Real(1) : Real(0); });

    const int jr = NG + ny / 2;
    const auto frontX = [&]() {
        const GridRef b = amr.coarseRef();
        for (int i = NG + b.nx - 1; i >= NG; --i)
            if (toPrim(b.at(i, jr)).p > Real(1.5) * P0)
                return double(b.xc(i));
        return 0.0;
    };
    double t = 0;
    std::vector<double> ts, xs;
    const double tEnd = 1.5;
    while (t < tEnd) {
        const Real dt = std::min(amr.maxStableDtAll(CFL), Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        ts.push_back(t);
        xs.push_back(frontX());
    }
    int n = 0;
    return fitSpeed(ts, xs, 0.6 * L, 0.85 * L, n);
}

// Same detonation on the hybrid GPU AMR (AmrGpuML): the reaction source
// runs as a Metal kernel per level. Returns the measured CJ speed.
double runAmrGpu(MetalContext& ctx, const Reaction& r, const GasPair& gas,
                 double L) {
    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = false;
    cfg.react = true;
    cfg.gamma1 = cfg.gamma2 = GAM;
    cfg.reaction = r;
    cfg.tagThreshold = Real(0.10);
    cfg.regridEvery = 2;
    const int nb = 512, ny = 8;
    AmrGpuML amr(ctx, nb, ny, 0, 0, L, Real(ny) * (L / nb), cfg);
    amr.fillPhysicalGhosts = [](GridRef& g, double) {
        fillReflectiveLeft(g);
        fillTransmissiveRight(g);
        fillPeriodicY(g);
    };
    amr.fillPatchPhysical = [](GridRef& g, double, unsigned s) {
        if (s & SideLeft) fillReflectiveLeft(g);
        if (s & SideRight) fillTransmissiveRight(g);
    };
    amr.init(
        [&](Real x, Real) {
            const bool drive = x < Real(0.2);
            return toConsG({RHO0, 0, 0, drive ? Real(30) : P0},
                           gas.Gamma(0));
        },
        [](Real x, Real) { return x < Real(0.2) ? Real(1) : Real(0); });

    const int jr = NG + ny / 2;
    const auto frontX = [&]() {
        const GridRef b = amr.coarseRef();
        for (int i = NG + b.nx - 1; i >= NG; --i)
            if (toPrim(b.at(i, jr)).p > Real(1.5) * P0)
                return double(b.xc(i));
        return 0.0;
    };
    double t = 0;
    std::vector<double> ts, xs;
    while (t < 1.5) {
        const Real dt = std::min(amr.maxStableDtAll(CFL), Real(1.5 - t));
        amr.step(dt, t);
        t += dt;
        ts.push_back(t);
        xs.push_back(frontX());
    }
    int n = 0;
    return fitSpeed(ts, xs, 0.6 * L, 0.85 * L, n);
}

} // namespace

int main() {
    const double q = 10.0;
    const double Dcj = cjSpeed(q, GAM, RHO0, P0);
    const double c0 = std::sqrt(double(GAM) * P0 / RHO0);

    // The CJ speed is rate-independent; pick A/Ea/Tign for a resolved,
    // self-sustaining reaction zone that ignites only behind the shock.
    const Reaction r{/*A*/ Real(3000), /*Ea*/ Real(8), /*q*/ Real(q),
                     /*Tign*/ Real(2.5)};
    const GasPair gas{GAM, GAM}; // single gas: Gamma constant

    const int nx = 2000, ny = 4;
    const double L = 8.0;
    Grid g(nx, ny, 0, 0, L, Real(ny) * (L / nx));
    std::vector<Real> phi(g.q.size()), gm(g.q.size(), Real(1 / (GAM - 1)));

    // unburnt reactive gas, with a hot high-pressure driver on the left
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const bool drive = g.xc(i) < 0.2;
            const Prim w{RHO0, 0, 0, drive ? Real(30) : P0};
            const std::size_t id = g.idx(i, j);
            g.q[id] = toConsG(w, gas.Gamma(0));
            phi[id] = drive ? g.q[id].rho : 0; // lambda = 1 in the driver
        }

    ScratchY sc;
    const auto frontX = [&]() {
        const int jr = NG + ny / 2;
        for (int i = NG + nx - 1; i >= NG; --i)
            if (toPrimG(g.at(i, jr), gm[g.idx(i, jr)]).p > Real(1.5) * P0)
                return double(g.xc(i));
        return 0.0;
    };

    double t = 0;
    std::vector<double> ts, xs;
    const double tEnd = 1.5; // front travels ~0.85 L at D_CJ
    while (t < tEnd) {
        const Real dt = std::min(maxStableDtY(g, gm, CFL), Real(tEnd - t));
        reactGrid(g, phi, dt / 2, r);
        fillBC(g, phi, gm);
        step2DY(g, phi, gm, dt, sc, gas);
        reactGrid(g, phi, dt / 2, r);
        t += dt;
        ts.push_back(t);
        xs.push_back(frontX());
    }

    // dump the front trajectory x(t) for the V&V figure (vv/): the local
    // speed dx/dt relaxes from the overdriven ignition toward D_CJ.
    if (FILE* ff = std::fopen("out/detonation_front.csv", "w")) {
        std::fprintf(ff, "t,x\n");
        for (std::size_t k = 0; k < ts.size(); ++k)
            std::fprintf(ff, "%.6g,%.6g\n", ts[k], xs[k]);
        std::fclose(ff);
    }

    int n = 0;
    const double Dmeas = fitSpeed(ts, xs, 0.6 * L, 0.85 * L, n);
    // relaxation diagnostic: the overdriven ignition decays to CJ
    for (double w0 : {0.2, 0.4, 0.6}) {
        int np;
        const double D = fitSpeed(ts, xs, w0 * L, (w0 + 0.2) * L, np);
        std::printf("  window [%.1f,%.1f]L: D=%.4f (%.1f%%)\n", w0,
                    w0 + 0.2, D, 100 * (D / Dcj - 1));
    }

    // ---- same detonation through the multi-level AMR (refines the
    // reaction zone) -> validate the Strang reaction in AmrML ----
    const double Damr = runAmr(r, gas, q, L, Dcj);
    MetalContext ctx;
    const double Dgpu = runAmrGpu(ctx, r, gas, L);

    std::printf("CJ detonation: q=%.1f, gamma=%.1f, c0=%.3f\n", q,
                double(GAM), c0);
    std::printf("  D_CJ exact      = %.4f  (%.2f c0; strong-limit "
                "sqrt(2(g^2-1)q) = %.4f)\n",
                Dcj, Dcj / c0, std::sqrt(2 * (GAM * GAM - 1) * q));
    std::printf("  D uniform       = %.4f  (%.1f%%, gate 3%%)\n", Dmeas,
                100 * (Dmeas / Dcj - 1));
    std::printf("  D on AMR (CPU)  = %.4f  (%.1f%%, gate 5%%)\n", Damr,
                100 * (Damr / Dcj - 1));
    std::printf("  D on AMR (GPU)  = %.4f  (%.1f%%, gate 5%%)\n", Dgpu,
                100 * (Dgpu / Dcj - 1));

    const bool ok = std::fabs(Dmeas / Dcj - 1) < 0.03 &&
                    std::fabs(Damr / Dcj - 1) < 0.05 &&
                    std::fabs(Dgpu / Dcj - 1) < 0.05;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Phase 2 validation of the inviscid cut-cell solver (1st order + flux
// redistribution):
//   1. well-balancedness — a uniform state at rest around a cylinder is
//      preserved (face and EB pressures cancel through the closure);
//   2. conservation — in a closed reflective box with an immersed cylinder,
//      total mass and energy stay at the float32 floor even though the time
//      step is set by the FULL cells (this is what flux redistribution buys);
//   3. physics — a Mach 2 flow over the cylinder builds a bow shock; the
//      stagnation pressure matches the Rayleigh pitot value (5.64 p_inf).

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

cutcell::Geometry cylinder(const Grid& g, double cx, double cy, double r) {
    return cutcell::build(g, [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    });
}

// volume-weighted (kappa) totals over fluid cells
void totals(const Grid& g, const cutcell::Geometry& geo, double& mass,
           double& energy) {
    mass = energy = 0;
    const double V = double(g.dx) * double(g.dy);
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const double k = double(geo.at(i, j).vol);
            if (k <= 1e-9) continue;
            mass += k * V * double(g.at(i, j).rho);
            energy += k * V * double(g.at(i, j).E);
        }
}

} // namespace

int main() {
    bool ok = true;

    // ---- gate 1: well-balancedness (at rest around a cylinder) ------------
    {
        const int N = 128;
        Grid g(N, N, 0, 0, 1, 1);
        const auto geo = cylinder(g, 0.5, 0.5, 0.2);
        const Cons u0 = toCons({1, 0, 0, 1});
        for (auto& q : g.q) q = u0;
        const auto fill = [](Grid& gg) {
            fillTransmissiveLeft(gg); fillTransmissiveRight(gg);
            fillTransmissiveBottom(gg); fillTransmissiveTop(gg);
        };
        for (int s = 0; s < 100; ++s)
            stepCutCell(g, geo, maxStableDt(g, Real(0.4)), fill);
        double dev = 0;
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const Cons& q = g.at(i, j);
                dev = std::max({dev, std::fabs(double(q.rho) - 1),
                                std::fabs(double(q.mx)), std::fabs(double(q.my)),
                                std::fabs(double(q.E) - double(u0.E))});
            }
        std::printf("gate 1 — at-rest cylinder, 100 steps: max deviation "
                    "%.3e (gate 1e-5)\n", dev);
        ok = ok && dev < 1e-5;
    }

    // ---- gate 2: conservation in a closed box (FRD, dt from full cells) ---
    {
        const int N = 128;
        Grid g(N, N, 0, 0, 1, 1);
        const auto geo = cylinder(g, 0.5, 0.5, 0.2);
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                const double x = double(g.xc(i)), y = double(g.yc(j));
                const double rr = (x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3);
                // smooth mild pressure bump — sub-sonic sloshing, no shock, so
                // the positivity floor never engages and conservation is exact
                g.at(i, j) = toCons({1, 0, 0, Real(1.0 + 0.5 * std::exp(-rr / 0.01))});
            }
        double m0, e0;
        totals(g, geo, m0, e0);
        const auto fill = [](Grid& gg) {
            fillReflectiveLeft(gg); fillReflectiveRight(gg);
            fillReflectiveBottom(gg); fillReflectiveTop(gg);
        };
        double drift = 0, edrift = 0;
        for (int s = 0; s < 300; ++s) {
            stepCutCell(g, geo, maxStableDt(g, Real(0.4)), fill);
            double m, e;
            totals(g, geo, m, e);
            drift = std::max(drift, std::fabs(m - m0) / m0);
            edrift = std::max(edrift, std::fabs(e - e0) / e0);
        }
        std::printf("gate 2 — closed box + cylinder, 300 steps: mass drift "
                    "%.3e, energy drift %.3e (gate 1e-5)\n", drift, edrift);
        ok = ok && drift < 1e-5 && edrift < 1e-5;
    }

    // ---- gate 3: Mach 2 bow shock, stagnation pressure vs Rayleigh pitot --
    {
        const int nx = 180, ny = 120;
        const double Lx = 1.5, Ly = 1.0;
        Grid g(nx, ny, 0, 0, Real(Lx), Real(Ly));
        const auto geo = cylinder(g, 0.5, 0.5, 0.15);
        const Prim fs{Real(1.4), Real(2), 0, Real(1)};   // rho 1.4, M=2 (c=1)
        const Cons ufs = toCons(fs);
        for (auto& q : g.q) q = ufs;
        const auto fill = [&](Grid& gg) {          // supersonic inflow (left)
            for (int j = 0; j < gg.toty(); ++j)
                for (int k = 0; k < NG; ++k) gg.at(k, j) = ufs;
            fillTransmissiveRight(gg);
            fillTransmissiveBottom(gg); fillTransmissiveTop(gg);
        };
        const double tEnd = 3.0;
        double t = 0;
        int steps = 0;
        while (t < tEnd) {
            const Real dt = std::min(maxStableDt(g, Real(0.4)), Real(tEnd - t));
            stepCutCell(g, geo, dt, fill);
            t += double(dt);
            ++steps;
        }
        double pmax = 0, rmax = 0;
        bool finite = true;
        for (int j = NG; j < NG + ny; ++j)
            for (int i = NG; i < NG + nx; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const Prim w = toPrim(g.at(i, j));
                pmax = std::max(pmax, double(w.p));
                rmax = std::max(rmax, double(w.rho));
                if (!std::isfinite(double(w.p)) || !std::isfinite(double(w.rho)))
                    finite = false;
            }
        const double pitot = 5.640;    // p0/p_inf behind a normal shock, M=2
        std::printf("gate 3 — Mach 2 cylinder, %d steps: stagnation p = %.3f "
                    "vs pitot %.3f (%+.1f%%), rho_max %.2f, finite %d\n",
                    steps, pmax, pitot, 100 * (pmax / pitot - 1), rmax,
                    int(finite));
        ok = ok && finite && std::fabs(pmax / pitot - 1) < 0.15;
    }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

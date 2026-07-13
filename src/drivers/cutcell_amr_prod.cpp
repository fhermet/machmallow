// Cut-cells in the PRODUCTION AMR (Amr2), first increment: 2 levels,
// single-rate, CPU, one analytic solid. This exercises the same physics the
// standalone `cutcell_amr` proved (aperture-weighted fluxes, FRD,
// volume-weighted restriction, cut-aware reflux) but through the real Amr2
// class — per-patch geometry, the flux register, restriction and reflux all
// wired into Amr2::step.
//
//   gate 1 — composite mass is conserved over the run WITH reflux (< 1e-6);
//   gate 2 — reflux matters: turning it off makes the drift >=100x worse.
// The refined region is pinned around the body (regrid disabled) so the
// coarse-fine interfaces sit in clean fluid, isolating the reflux path.

#include "amr/Amr2.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace mm;

namespace {

// Worst relative composite-mass drift over a fixed-patch 2-level cut-cell run
// driven by Amr2, with the flux register (reflux) on or off.
double amr2MassDrift(bool reflux) {
    const int NC = 48, bc = 16;                // coarse cells, block size
    const double cx = 0.5, cy = 0.5, r = 0.1;  // immersed cylinder
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };

    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.reflux = reflux;
    cfg.subcycle = false;
    cfg.regridEvery = 1 << 30;                 // pinned patches: never retag
    cfg.mu = 0;

    Amr2 amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle;
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };

    // IC: smooth off-centre pressure bump, at rest (sub-sonic sloshing).
    const auto ic = [&](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    for (int j = 0; j < amr.coarse.toty(); ++j)
        for (int i = 0; i < amr.coarse.totx(); ++i)
            amr.coarse.at(i, j) = ic(double(amr.coarse.xc(i)),
                                     double(amr.coarse.yc(j)));

    // Pin a single refined patch (the centre block [1/3,2/3]^2) containing
    // the whole body plus a fluid halo, so the coarse-fine interfaces stay in
    // clean fluid and every cut cell is interior to the one patch (matches the
    // standalone cutcell_amr topology; cross-patch FRD is a later increment).
    std::vector<std::pair<int, int>> blocks{{1, 1}};
    amr.restoreBlocks(blocks, 0);
    for (Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                p.grid.at(i, j) = ic(double(p.grid.xc(i)),
                                     double(p.grid.yc(j)));

    // Composite mass = uncovered coarse (kappa-weighted) + fine (kappa) using
    // exact analytic geometry (patch geometry lives in p.geo already).
    const auto Gc = cutcell::build(amr.coarse, circle);
    const double Vc = double(amr.coarse.dx) * double(amr.coarse.dy);
    const auto mass = [&]() {
        double m = 0;
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (amr.covered(i / bc, j / bc)) continue;
                m += double(Gc.at(NG + i, NG + j).vol) * Vc *
                     double(amr.coarse.at(NG + i, NG + j).rho);
            }
        for (const Amr2::Patch& p : amr.patches) {
            const double Vf = double(p.grid.dx) * double(p.grid.dy);
            for (int j = NG; j < NG + p.grid.ny; ++j)
                for (int i = NG; i < NG + p.grid.nx; ++i)
                    m += double(p.geo.at(i, j).vol) * Vf *
                         double(p.grid.at(i, j).rho);
        }
        return m;
    };

    const double m0 = mass();
    double drift = 0, t = 0;
    for (int s = 0; s < 200; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4));
        amr.step(dt, t);
        t += double(dt);
        drift = std::max(drift, std::fabs(mass() - m0) / m0);
    }
    return drift;
}

} // namespace

int main() {
    bool ok = true;

    const double driftOn = amr2MassDrift(true);
    const double driftOff = amr2MassDrift(false);
    std::printf("gate 1 — Amr2 cut-cell composite mass drift: %.3e (gate "
                "1e-6)\n", driftOn);
    ok = ok && driftOn < 1e-6;

    std::printf("gate 2 — reflux vs no reflux: %.3e off / %.3e on (%.0fx "
                "worse)\n", driftOff, driftOn, driftOff / driftOn);
    ok = ok && driftOff > 100 * driftOn;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

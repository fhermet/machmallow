// Cut-cells in the PRODUCTION AMR (Amr2), first increment: 2 levels,
// single-rate, CPU, one analytic solid. This exercises the same physics the
// standalone `cutcell_amr` proved (aperture-weighted fluxes, FRD,
// volume-weighted restriction, cut-aware reflux) but through the real Amr2
// class — per-patch geometry, the flux register, restriction and reflux all
// wired into Amr2::step.
//
//   gate 1 — composite mass is conserved over the run WITH reflux (< 1e-6);
//   gate 2 — reflux matters: turning it off makes the drift >=100x worse.
// The body SPANS a 4x4 block of refined patches (regrid disabled), so cut
// cells sit on internal sibling seams: this exercises both the cut-aware
// reflux at the coarse-fine boundary AND the cross-patch flux redistribution
// (the fine sibling seams). Without cross-patch FRD the seams leak (~1e-5).

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
double amr2MassDrift(bool reflux, bool subcycle = false) {
    const int NC = 48, bc = 8;                 // coarse cells, block size
    const double cx = 0.5, cy = 0.5, r = 0.2;  // immersed cylinder
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };

    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.reflux = reflux;
    cfg.subcycle = subcycle;
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

    // Pin a 4x4 block of refined patches around the body so the body SPANS
    // several patches: cut cells sit on internal sibling seams, exercising the
    // cross-patch flux redistribution. Coarse-fine interfaces (blocks 0 and 5)
    // stay in clean fluid.
    std::vector<std::pair<int, int>> blocks;
    for (int bj = 1; bj <= 4; ++bj)
        for (int bi = 1; bi <= 4; ++bi) blocks.emplace_back(bi, bj);
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

// Composite-mass drift over a run with REGRID ENABLED, refinement chosen by
// the tagger's EB-band criterion (0<kappa<1 + dilation), not pinned. The flow
// tag is disabled (tagThreshold sentinel) so the refined set is driven purely
// by the embedded boundary and stays stable for the static body — this isolates
// the EB-band tagging + conservative regrid path. (Flow-driven dynamic regrid
// rides the general Amr2 float32 conservation floor, ~few x 1e-6, unrelated to
// cut cells: a tiny/absent body drifts ~2e-6 the same way.) Reports whether the
// block over the body centre ended up refined, and the patch count.
double amr2RegridDrift(bool& bodyRefined, int& nPatches) {
    const int NC = 48, bc = 8;
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };

    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.reflux = true;
    cfg.subcycle = false;
    cfg.regridEvery = 4;                        // regrid during the run
    cfg.tagThreshold = Real(1e30);            // EB band only (isolate regrid)
    cfg.mu = 0;

    Amr2 amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle;
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    const auto ic = [&](Real x, Real y) {
        const double rr = (double(x) - 0.3) * (double(x) - 0.3) +
                          (double(y) - 0.5) * (double(y) - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    amr.init(ic);                              // auto-tag + prolong + restrict

    bodyRefined = amr.covered(int(cx * NC) / bc, int(cy * NC) / bc);
    nPatches = int(amr.patches.size());

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

    bool bodyRefined = false;
    int nPatches = 0;
    const double driftRegrid = amr2RegridDrift(bodyRefined, nPatches);
    std::printf("gate 3 — EB-band regrid: body block refined %s, "
                "%d patches, composite drift %.3e (gate 1e-6)\n",
                bodyRefined ? "yes" : "NO", nPatches, driftRegrid);
    ok = ok && bodyRefined && driftRegrid < 1e-6;

    const double driftSub = amr2MassDrift(true, /*subcycle=*/true);
    std::printf("gate 4 — subcycled (coarse dt, fine 2x dt/2): composite drift "
                "%.3e (gate 1e-6)\n", driftSub);
    ok = ok && driftSub < 1e-6;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

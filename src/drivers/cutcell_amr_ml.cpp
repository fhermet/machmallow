// Cut-cells in the ARBITRARY-DEPTH production AMR (AmrML). The 2-level Amr2
// integration (per-patch geometry, cross-patch flux redistribution, cut-aware
// reflux, kappa-restriction, EB-band tagging) is ported to AmrML's recursive
// Berger-Colella machinery, so it works at any depth and with subcycling
// through the existing refluxBackOut_/refluxFineApply_ split.
//
//   gate 1 — 2 levels, single-rate   : composite mass conserved (< 1e-6);
//   gate 2 — 2 levels, subcycled     : conserved (dt/2 substeps);
//   gate 3 — 3 levels, single-rate   : conserved across the hierarchy;
//   gate 4 — 3 levels, subcycled     : conserved (recursive subcycling).
// The body is auto-refined to the finest level by the EB-band tagger and is
// nested within it (its cut cells sit strictly inside the finest patches).
//
// KNOWN LIMITATION (reported, not gated): if the body is large enough that its
// cut cells ABUT a coarse-fine boundary (the finest level cannot fully contain
// the band), the hybrid redistribution that crosses that boundary is lost to
// the covering-level restriction (~2e-5). Reaching the fp32 floor there needs
// AMReX-style cross-level re-redistribution — a later increment.

#include "amr/AmrML.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

// Worst relative composite-mass drift over an AmrML cut-cell run with the given
// depth and subcycling. Sets deepestRefined = body reached the finest level.
double mlMassDrift(int maxLevels, bool subcycle, bool& deepestRefined,
                   double r, int nSteps = 120) {
    const int NC = 32, bc = 8;
    const double cx = 0.5, cy = 0.5;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };

    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = maxLevels;
    cfg.cutCell = true;
    cfg.reflux = true;
    cfg.subcycle = subcycle;
    cfg.regridEvery = 1 << 30;      // EB is static: tag once at init, then hold
    cfg.tagThreshold = Real(1e30);  // EB-band only (no flow tag) -> stable set
    cfg.mu = 0;

    AmrML amr(NC, NC, 0, 0, 1, 1, cfg);
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
    amr.init(ic);

    // deepest level has a patch over the body centre?
    deepestRefined = amr.patchCount(maxLevels - 1) > 0;

    const int nf = amr.fineCells();
    const auto Gb = cutcell::build(amr.base, circle);
    const double Vb = double(amr.base.dx) * double(amr.base.dy);
    const auto mass = [&]() {
        double m = 0;
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (maxLevels >= 2 && amr.covered(1, i / bc, j / bc)) continue;
                m += double(Gb.at(NG + i, NG + j).vol) * Vb *
                     double(amr.base.at(NG + i, NG + j).rho);
            }
        for (int l = 1; l < maxLevels; ++l) {
            const double Vl = Vb / double(1 << (2 * l));
            for (const AmrML::Patch& p : amr.level(l).patches)
                for (int j = 0; j < nf; ++j)
                    for (int i = 0; i < nf; ++i) {
                        const int gi = 2 * p.ci0 + i, gj = 2 * p.cj0 + j;
                        if (l < maxLevels - 1 &&
                            amr.covered(l + 1, gi / bc, gj / bc))
                            continue;
                        m += double(p.geo.at(NG + i, NG + j).vol) * Vl *
                             double(p.grid.at(NG + i, NG + j).rho);
                    }
        }
        return m;
    };

    const double m0 = mass();
    double drift = 0, t = 0;
    for (int s = 0; s < nSteps; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4));
        amr.step(dt, t);
        t += double(dt);
        drift = std::max(drift, std::fabs(mass() - m0) / m0);
    }
    return drift;
}

} // namespace

int main() {
    bool ok = true, ref = false;

    const double d1 = mlMassDrift(2, false, ref, 0.15);
    std::printf("gate 1 — 2 levels, single-rate : L1 refined %s, drift %.3e "
                "(gate 1e-6)\n", ref ? "yes" : "NO", d1);
    ok = ok && ref && d1 < 1e-6;

    const double d2 = mlMassDrift(2, true, ref, 0.15);
    std::printf("gate 2 — 2 levels, subcycled   : L1 refined %s, drift %.3e "
                "(gate 1e-6)\n", ref ? "yes" : "NO", d2);
    ok = ok && ref && d2 < 1e-6;

    const double d3 = mlMassDrift(3, false, ref, 0.05);
    std::printf("gate 3 — 3 levels, single-rate : L2 refined %s, drift %.3e "
                "(gate 1e-6)\n", ref ? "yes" : "NO", d3);
    ok = ok && ref && d3 < 1e-6;

    const double d4 = mlMassDrift(3, true, ref, 0.05);
    std::printf("gate 4 — 3 levels, subcycled   : L2 refined %s, drift %.3e "
                "(gate 1e-6)\n", ref ? "yes" : "NO", d4);
    ok = ok && ref && d4 < 1e-6;

    // Known limitation (reported, not gated): a body straddling a coarse-fine
    // seam loses the cross-boundary redistribution to restriction.
    const double dSeam = mlMassDrift(3, false, ref, 0.15);
    std::printf("info   — 3 levels, body straddling a coarse-fine seam: drift "
                "%.3e (cross-level re-redistribution: future work)\n", dSeam);

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

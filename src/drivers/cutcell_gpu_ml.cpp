// Hybrid CPU/GPU ARBITRARY-DEPTH cut-cell AMR (AmrGpuMLCut) validated in
// LOCK-STEP against the CPU cut-cell oracle (AmrML cut path). Both classes
// auto-tag the EB band at init (geometry-determined, static topology), then
// advance from the SAME state with the SAME dt through the SAME recursive
// Berger-Colella machinery (per-level composite FRD, cut-aware reflux,
// kappa-restriction). The GPU forms the divergence on Metal (gather-form FRD);
// the composite fields must match the CPU scatter path to fp32 lock-step.
//
// Body large enough to straddle the coarse-fine seams at every level (the
// demanding cross-level case), reflective box.
//   gate 1 — 2 levels, single-rate : lock-step (< 1e-2) + GPU mass (< 1e-6)
//   gate 2 — 2 levels, subcycled   : lock-step + GPU mass
//   gate 3 — 3 levels, single-rate : lock-step + GPU mass
//   gate 4 — 3 levels, subcycled   : lock-step + GPU mass

#include "amr/AmrGpuMLCut.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

const int NC = 48, BC = 8;
const double CXc = 0.5, CYc = 0.5, R = 0.20;

auto circle() {
    return [](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(CXc, CYc, R, x0, x1, y0, y1);
    };
}
auto initCond() {
    return [](Real x, Real y) {
        const double rr = (double(x) - 0.3) * (double(x) - 0.3) +
                          (double(y) - 0.5) * (double(y) - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
}
AmrConfig cutConfig(int maxLevels, bool subcycle) {
    AmrConfig cfg;
    cfg.blockC = BC;
    cfg.maxLevels = maxLevels;
    cfg.cutCell = true;
    cfg.reflux = true;
    cfg.subcycle = subcycle;
    cfg.regridEvery = 1 << 30;     // static EB: tag once at init, then hold
    cfg.tagThreshold = Real(1e30); // EB-band only -> stable, matching topology
    cfg.mu = 0;
    return cfg;
}

// Lock-step + GPU mass drift over an N-level cut-cell run. worst = worst
// relative rho diff GPU vs AmrML; massDrift = GPU composite conservation.
void run(int maxLevels, bool subcycle, double& worst, double& massDrift,
         bool& deepestRefined) {
    const auto ic = initCond();
    const AmrConfig cfg = cutConfig(maxLevels, subcycle);

    // CPU oracle.
    AmrML amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle();
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    amr.init(ic);

    // GPU composite.
    MetalContext ctx;
    AmrGpuMLCut gpu(ctx, NC, NC, 0, 0, 1, 1, cfg);
    gpu.momentFn = circle();
    gpu.fillPhysicalGhosts = [](GridRef& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    gpu.init(ic);

    deepestRefined = amr.patchCount(maxLevels - 1) > 0 &&
                     gpu.patchCount(maxLevels - 1) ==
                         amr.patchCount(maxLevels - 1);

    const double m0 = gpu.totalMass();
    worst = 0;
    massDrift = 0;
    double t = 0;
    for (int s = 0; s < 120; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4)); // shared dt
        amr.step(dt, t);
        gpu.step(dt, t);
        t += double(dt);

        // Compare uncovered base + every patch (matching topology / order).
        const GridRef gb = gpu.coarseRef();
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (maxLevels >= 2 && amr.covered(1, i / BC, j / BC)) continue;
                const double a = double(amr.base.at(NG + i, NG + j).rho);
                const double b = double(gb.at(NG + i, NG + j).rho);
                worst = std::max(worst,
                                 std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
            }
        for (int l = 1; l < maxLevels; ++l) {
            const auto& cp = amr.level(l).patches;
            const auto& gp = gpu.level(l).patches;
            if (cp.size() != gp.size()) { worst = 1e9; continue; }
            for (std::size_t k = 0; k < cp.size(); ++k) {
                const GridRef g = gpu.patchRef(l, gp[k]);
                const Grid& pg = cp[k].grid;
                for (int j = NG; j < NG + pg.ny; ++j)
                    for (int i = NG; i < NG + pg.nx; ++i) {
                        const double a = double(pg.at(i, j).rho);
                        const double b = double(g.at(i, j).rho);
                        worst = std::max(
                            worst,
                            std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
                    }
            }
        }
        massDrift = std::max(massDrift, std::fabs(gpu.totalMass() - m0) / m0);
    }
}

} // namespace

int main() {
    bool ok = true;
    struct Case { int lv; bool sub; const char* name; };
    const Case cases[] = {
        {2, false, "2 levels, single-rate"},
        {2, true,  "2 levels, subcycled  "},
        {3, false, "3 levels, single-rate"},
        {3, true,  "3 levels, subcycled  "},
    };
    int gate = 1;
    for (const Case& c : cases) {
        double worst = 0, mass = 0;
        bool ref = false;
        run(c.lv, c.sub, worst, mass, ref);
        std::printf("gate %d — %s : refined %s, lock-step rho %.3e (gate 1e-2), "
                    "GPU mass %.3e (gate 1e-6)\n",
                    gate++, c.name, ref ? "yes" : "NO", worst, mass);
        ok = ok && ref && worst < 1e-2 && mass < 1e-6;
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

// Hybrid CPU/GPU two-level cut-cell AMR (AmrGpuCut) validated in LOCK-STEP
// against the CPU cut-cell AMR oracle (Amr2, cut path). Both classes advance
// from the SAME initial condition with the SAME dt, through the SAME algorithm
// (aperture-weighted fluxes + flux redistribution, cross-patch FRD coupling,
// cut-aware reflux, kappa-weighted restriction). The GPU forms the divergence
// on Metal (gather-form FRD), the CPU exchanges D^c ghosts across patch seams;
// the composite fields must match the CPU scatter path to fp32 lock-step
// tolerance. Immersed cylinder spanning a 4x4 block of patches (sibling seams),
// reflective box.
//
//   gate 1 — single-rate GPU vs CPU lock-step over 200 steps (< 1e-2);
//   gate 2 — GPU composite mass conserved with reflux (< 1e-6);
//   gate 3 — subcycled GPU vs CPU lock-step (< 1e-2).

#include "amr/Amr2.hpp"
#include "amr/AmrGpuCut.hpp"
#include "backend/metal/MetalContext.hpp"
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

const int NC = 48, BC = 8;
const double CX = 0.5, CY = 0.5, R = 0.2;

auto circle() {
    return [](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(CX, CY, R, x0, x1, y0, y1);
    };
}
auto initCond() {
    return [](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
}
std::vector<std::pair<int, int>> bodyBlocks() {
    std::vector<std::pair<int, int>> blocks;
    for (int bj = 1; bj <= 4; ++bj)
        for (int bi = 1; bi <= 4; ++bi) blocks.emplace_back(bi, bj);
    return blocks;
}

AmrConfig cutConfig(bool subcycle, bool o2 = false) {
    AmrConfig cfg;
    cfg.blockC = BC;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.cutCellO2 = o2;
    cfg.reflux = true;
    cfg.subcycle = subcycle;
    cfg.regridEvery = 1 << 30; // pinned patches: never retag
    cfg.mu = 0;
    return cfg;
}

// Worst relative rho difference between the GPU composite (AmrGpuCut) and the
// CPU oracle (Amr2 cut) over a fixed-patch run, driven with a shared dt.
double lockStepDrift(bool subcycle, bool o2 = false) {
    const auto ic = initCond();
    const AmrConfig cfg = cutConfig(subcycle, o2);

    // CPU oracle.
    Amr2 amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle();
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    for (int j = 0; j < amr.coarse.toty(); ++j)
        for (int i = 0; i < amr.coarse.totx(); ++i)
            amr.coarse.at(i, j) = ic(double(amr.coarse.xc(i)),
                                     double(amr.coarse.yc(j)));
    amr.restoreBlocks(bodyBlocks(), 0);
    for (Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                p.grid.at(i, j) = ic(double(p.grid.xc(i)),
                                     double(p.grid.yc(j)));

    // GPU composite.
    MetalContext ctx;
    AmrGpuCut gpu(ctx, NC, NC, 0, 0, 1, 1, cfg);
    gpu.momentFn = circle();
    gpu.fillPhysicalGhosts = [](GridRef& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    {
        GridRef c = gpu.coarseRef();
        for (int j = 0; j < c.toty(); ++j)
            for (int i = 0; i < c.totx(); ++i)
                c.at(i, j) = ic(double(c.xc(i)), double(c.yc(j)));
    }
    gpu.restoreBlocks(bodyBlocks(), 0);
    for (const AmrGpuCut::Patch& p : gpu.patches) {
        GridRef g = gpu.patchRef(p);
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                g.at(i, j) = ic(double(g.xc(i)), double(g.yc(j)));
    }

    double worst = 0, t = 0;
    for (int s = 0; s < 200; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4)); // shared dt
        amr.step(dt, t);
        gpu.step(dt, t);
        t += double(dt);

        // Compare coarse (uncovered) + every patch cell.
        const GridRef gc = gpu.coarseRef();
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (amr.covered(i / BC, j / BC)) continue;
                const double a = double(amr.coarse.at(NG + i, NG + j).rho);
                const double b = double(gc.at(NG + i, NG + j).rho);
                worst = std::max(worst,
                                 std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
            }
        for (std::size_t k = 0; k < amr.patches.size(); ++k) {
            const GridRef g = gpu.patchRef(gpu.patches[k]);
            const Grid& pg = amr.patches[k].grid;
            for (int j = NG; j < NG + pg.ny; ++j)
                for (int i = NG; i < NG + pg.nx; ++i) {
                    const double a = double(pg.at(i, j).rho);
                    const double b = double(g.at(i, j).rho);
                    worst = std::max(
                        worst, std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
                }
        }
    }
    return worst;
}

// GPU composite-mass drift over the fixed-patch run (conservation floor).
double gpuMassDrift(bool subcycle) {
    const auto ic = initCond();
    const auto circ = circle();
    const AmrConfig cfg = cutConfig(subcycle);

    MetalContext ctx;
    AmrGpuCut gpu(ctx, NC, NC, 0, 0, 1, 1, cfg);
    gpu.momentFn = circ;
    gpu.fillPhysicalGhosts = [](GridRef& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    {
        GridRef c = gpu.coarseRef();
        for (int j = 0; j < c.toty(); ++j)
            for (int i = 0; i < c.totx(); ++i)
                c.at(i, j) = ic(double(c.xc(i)), double(c.yc(j)));
    }
    gpu.restoreBlocks(bodyBlocks(), 0);
    for (const AmrGpuCut::Patch& p : gpu.patches) {
        GridRef g = gpu.patchRef(p);
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                g.at(i, j) = ic(double(g.xc(i)), double(g.yc(j)));
    }

    Grid tmp(NC, NC, 0, 0, 1, 1);
    const auto Gc = cutcell::build(tmp, circ);
    const double Vc = double(tmp.dx) * double(tmp.dy);
    const auto mass = [&]() {
        double m = 0;
        const GridRef c = gpu.coarseRef();
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (gpu.covered(i / BC, j / BC)) continue;
                m += double(Gc.at(NG + i, NG + j).vol) * Vc *
                     double(c.at(NG + i, NG + j).rho);
            }
        for (const AmrGpuCut::Patch& p : gpu.patches) {
            const GridRef g = gpu.patchRef(p);
            const double Vf = double(g.dx) * double(g.dy);
            for (int j = NG; j < NG + g.ny; ++j)
                for (int i = NG; i < NG + g.nx; ++i)
                    m += double(p.geo.at(i, j).vol) * Vf *
                         double(g.at(i, j).rho);
        }
        return m;
    };

    const double m0 = mass();
    double drift = 0, t = 0;
    for (int s = 0; s < 200; ++s) {
        const Real dt = gpu.maxStableDtAll(Real(0.4)); // respects fine level
        gpu.step(dt, t);
        t += double(dt);
        drift = std::max(drift, std::fabs(mass() - m0) / m0);
    }
    return drift;
}

} // namespace

int main() {
    bool ok = true;

    const double d1 = lockStepDrift(/*subcycle=*/false);
    std::printf("gate 1 — single-rate GPU vs CPU cut-cell AMR lock-step: worst "
                "relative rho diff %.3e (gate 1e-2)\n", d1);
    ok = ok && d1 < 1e-2;

    const double d2 = gpuMassDrift(/*subcycle=*/false);
    std::printf("gate 2 — GPU composite mass drift with reflux: %.3e "
                "(gate 1e-6)\n", d2);
    ok = ok && d2 < 1e-6;

    const double d3 = lockStepDrift(/*subcycle=*/true);
    std::printf("gate 3 — subcycled GPU vs CPU cut-cell AMR lock-step: worst "
                "relative rho diff %.3e (gate 1e-2)\n", d3);
    ok = ok && d3 < 1e-2;

    const double d4 = lockStepDrift(/*subcycle=*/false, /*o2=*/true);
    std::printf("gate 4 — 2nd-order single-rate GPU vs CPU (Amr2 O2) lock-step: "
                "worst relative rho diff %.3e (gate 1e-2)\n", d4);
    ok = ok && d4 < 1e-2;

    const double d5 = lockStepDrift(/*subcycle=*/true, /*o2=*/true);
    std::printf("gate 5 — 2nd-order subcycled GPU vs CPU (Amr2 O2) lock-step: "
                "worst relative rho diff %.3e (gate 1e-2)\n", d5);
    ok = ok && d5 < 1e-2;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

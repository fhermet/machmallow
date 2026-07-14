// GPU cut-cell solver validated in LOCK-STEP against the CPU cut-cell oracle.
// The CPU cutCellStepFluxed (the AMR building block: 1st-order forward Euler,
// aperture-weighted fluxes + flux redistribution + positivity floor) and the
// Metal CutCell2DGpu advance from the SAME initial condition with the SAME dt;
// the fields must match to fp32 lock-step tolerance (the GPU FRD is a gather,
// the CPU a scatter, so sums reassociate). Immersed cylinder, reflective box.

#include "backend/metal/CutCell2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

// 2nd-order GPU cut-cell (LSQ gradients + reconstructed face/EB fluxes, SSP-RK2)
// vs the CPU oracle stepCutCell, in lock-step over 100 steps. Same immersed
// cylinder / reflective box / pressure-bump IC as gate 1.
double secondOrderDrift() {
    const int NC = 96;
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };
    const auto ic = [&](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    const auto fill = [](auto& g) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };

    Grid gc(NC, NC, 0, 0, 1, 1);
    const auto geo = cutcell::build(gc, circle);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i)
            gc.at(i, j) = ic(double(gc.xc(i)), double(gc.yc(j)));

    MetalContext ctx;
    CutCell2DGpu gpu(ctx, NC, NC, gc.dx, gc.dy);
    gpu.setGeometry(geo); // x0=y0=0: EB-centroid offset exact
    gpu.enableO2();
    GridRef gg = gpu.ref(0, 0);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i) gg.at(i, j) = gc.at(i, j);

    double worst = 0;
    for (int s = 0; s < 100; ++s) {
        const Real dt = maxStableDt(gc, Real(0.4));
        stepCutCell(gc, geo, dt, fill, /*limited=*/true, /*mu=*/0);
        fill(gg);
        gpu.divO2();
        gpu.rk2Stage1(dt);
        fill(gg);
        gpu.divO2();
        gpu.rk2Stage2(dt);
        for (int j = NG; j < NG + NC; ++j)
            for (int i = NG; i < NG + NC; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const double a = double(gc.at(i, j).rho);
                const double b = double(gg.at(i, j).rho);
                worst = std::max(worst,
                                 std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
            }
    }
    return worst;
}

} // namespace

int main() {
    const int NC = 96;
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };
    const auto ic = [&](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    const auto fill = [](auto& g) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };

    Grid gc(NC, NC, 0, 0, 1, 1);
    const auto geo = cutcell::build(gc, circle);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i)
            gc.at(i, j) = ic(double(gc.xc(i)), double(gc.yc(j)));

    MetalContext ctx;
    CutCell2DGpu gpu(ctx, NC, NC, gc.dx, gc.dy);
    gpu.setGeometry(geo);
    GridRef gg = gpu.ref(0, 0);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i)
            gg.at(i, j) = gc.at(i, j);

    // Pooled layout: the SAME problem in 3 slots, advanced in one dispatch.
    // Each slot must match the plain single-grid run (proves pool indexing +
    // no cross-slot bleed) — the layout AmrGpu will batch patches in.
    const int NS = 3;
    gpu.enablePool(NS);
    auto slotRef = [&](int sl) {
        return GridRef{NC, NC, 0, 0, gc.dx, gc.dy, gpu.poolData(sl)};
    };
    for (int sl = 0; sl < NS; ++sl) {
        gpu.setPoolGeometry(sl, geo);
        GridRef gp = slotRef(sl);
        for (int j = 0; j < gc.toty(); ++j)
            for (int i = 0; i < gc.totx(); ++i) gp.at(i, j) = gc.at(i, j);
    }

    // Phase-split single-grid instance (dc -> hybrid -> update as separate
    // dispatches, the path the composite AMR interleaves CPU passes into).
    // Must match the monolithic step() bit-exact.
    CutCell2DGpu gpu3(ctx, NC, NC, gc.dx, gc.dy);
    gpu3.setGeometry(geo);
    GridRef g3 = gpu3.ref(0, 0);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i) g3.at(i, j) = gc.at(i, j);

    std::vector<Cons> Fx, Fy;
    double worst = 0, worstPool = 0, worstPhase = 0;
    const int steps = 100;
    for (int s = 0; s < steps; ++s) {
        const Real dt = maxStableDt(gc, Real(0.4)); // same dt everywhere
        cutCellStepFluxed(gc, geo, dt, fill, Fx, Fy);
        fill(gg);
        gpu.step(dt);
        fill(g3);
        gpu3.dcPhase();
        gpu3.hybridPhase();
        gpu3.updatePhase(dt);
        for (int sl = 0; sl < NS; ++sl) { GridRef gp = slotRef(sl); fill(gp); }
        gpu.stepPool(dt, {0, 1, 2});
        for (int j = NG; j < NG + NC; ++j)
            for (int i = NG; i < NG + NC; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const Cons a = gc.at(i, j), b = gg.at(i, j);
                const double sc = std::max(std::fabs(double(a.rho)), 1e-3);
                worst = std::max(worst, std::fabs(double(a.rho - b.rho)) / sc);
                for (int sl = 0; sl < NS; ++sl) {
                    const Cons c = slotRef(sl).at(i, j);
                    worstPool = std::max(
                        worstPool, std::fabs(double(b.rho - c.rho)) / sc);
                }
                worstPhase = std::max(
                    worstPhase, std::fabs(double(b.rho - g3.at(i, j).rho)) / sc);
            }
    }
    std::printf("gate 1 — GPU vs CPU cut-cell lock-step over %d steps: worst "
                "relative rho diff %.3e (gate 1e-2)\n", steps, worst);
    std::printf("gate 2 — pooled (3 slots) vs plain GPU: worst relative rho "
                "diff %.3e (gate 1e-6, bit-exact expected)\n", worstPool);
    std::printf("gate 3 — phase-split vs monolithic step: worst relative rho "
                "diff %.3e (gate 1e-6, bit-exact expected)\n", worstPhase);
    const double worstO2 = secondOrderDrift();
    std::printf("gate 4 — 2nd-order GPU (LSQ + RK2) vs CPU stepCutCell lock-step "
                "over 100 steps: worst relative rho diff %.3e (gate 1e-2)\n",
                worstO2);

    const bool ok = worst < 1e-2 && worstPool < 1e-6 && worstPhase < 1e-6 &&
                    worstO2 < 1e-2;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

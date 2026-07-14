// Viscous (Navier-Stokes) cut cells in the AMR, on the GPU. The 2nd-order cut
// path now carries the Stokes/Fourier face fluxes + no-slip EB traction through
// the composite (they ride in the recorded extensive fluxes, so the averaged-
// flux reflux stays conservative). Validated in LOCK-STEP: the GPU classes
// (AmrGpuCut / AmrGpuMLCut, mu>0) against the CPU oracles (Amr2 / AmrML O2,
// mu>0), driven with a shared dt. Immersed cylinder, reflective box, a smooth
// pressure bump relaxing viscously.
//
//   gate 1 — 2-level GPU vs CPU viscous lock-step (< 1e-2) + GPU mass (< 1e-6);
//   gate 2 — 3-level GPU vs CPU viscous lock-step (< 1e-2) + GPU mass (< 1e-6).

#include "amr/Amr2.hpp"
#include "amr/AmrGpuCut.hpp"
#include "amr/AmrGpuMLCut.hpp"
#include "amr/AmrML.hpp"
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
const double CX = 0.5, CY = 0.5, R = 0.20, MU = 0.05;

auto circle() {
    return [](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(CX, CY, R, x0, x1, y0, y1);
    };
}
auto initCond() {
    // gentle off-body bump (not centred on the cut cells) so 2nd-order
    // reconstruction does not overshoot into the positivity floor there.
    return [](Real x, Real y) {
        const double rr = (double(x) - 0.3) * (double(x) - 0.3) +
                          (double(y) - 0.5) * (double(y) - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.2 * std::exp(-rr / 0.02))});
    };
}
void reflectRef(GridRef& g, double) {
    fillReflectiveLeft(g);  fillReflectiveRight(g);
    fillReflectiveBottom(g); fillReflectiveTop(g);
}
void reflectGrid(Grid& g, double) {
    fillReflectiveLeft(g);  fillReflectiveRight(g);
    fillReflectiveBottom(g); fillReflectiveTop(g);
}
AmrConfig cfgO2v(int maxLevels, bool eb) {
    AmrConfig cfg;
    cfg.blockC = BC;
    cfg.maxLevels = maxLevels;
    cfg.cutCell = true;
    cfg.cutCellO2 = true;
    cfg.reflux = true;
    cfg.mu = Real(MU);
    cfg.regridEvery = 1 << 30;
    if (eb) cfg.tagThreshold = Real(1e30); // EB-band only (stable topology)
    return cfg;
}

// ---- gate 1: 2-level AmrGpuCut vs Amr2, viscous -------------------------
void run2(double& worst, double& mass) {
    const auto ic = initCond();
    const AmrConfig cfg = cfgO2v(2, false);
    std::vector<std::pair<int, int>> blocks;
    for (int bj = 1; bj <= 4; ++bj)
        for (int bi = 1; bi <= 4; ++bi) blocks.emplace_back(bi, bj);

    Amr2 amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle();
    amr.fillPhysicalGhosts = reflectGrid;
    for (int j = 0; j < amr.coarse.toty(); ++j)
        for (int i = 0; i < amr.coarse.totx(); ++i)
            amr.coarse.at(i, j) = ic(amr.coarse.xc(i), amr.coarse.yc(j));
    amr.restoreBlocks(blocks, 0);
    for (Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                p.grid.at(i, j) = ic(p.grid.xc(i), p.grid.yc(j));

    MetalContext ctx;
    AmrGpuCut gpu(ctx, NC, NC, 0, 0, 1, 1, cfg);
    gpu.momentFn = circle();
    gpu.fillPhysicalGhosts = reflectRef;
    {
        GridRef c = gpu.coarseRef();
        for (int j = 0; j < c.toty(); ++j)
            for (int i = 0; i < c.totx(); ++i)
                c.at(i, j) = ic(c.xc(i), c.yc(j));
    }
    gpu.restoreBlocks(blocks, 0);
    for (const AmrGpuCut::Patch& p : gpu.patches) {
        GridRef g = gpu.patchRef(p);
        for (int j = NG; j < NG + g.ny; ++j)
            for (int i = NG; i < NG + g.nx; ++i)
                g.at(i, j) = ic(g.xc(i), g.yc(j));
    }

    const double m0 = gpu.totalMass();
    worst = 0; mass = 0;
    double t = 0;
    for (int s = 0; s < 150; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4));
        amr.step(dt, t);
        gpu.step(dt, t);
        t += double(dt);
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
        mass = std::max(mass, std::fabs(gpu.totalMass() - m0) / m0);
    }
}

// ---- gate 2: 3-level AmrGpuMLCut vs AmrML, viscous ---------------------
void run3(double& worst, double& mass) {
    const auto ic = initCond();
    const AmrConfig cfg = cfgO2v(3, true);

    AmrML amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle();
    amr.fillPhysicalGhosts = reflectGrid;
    amr.init(ic);

    MetalContext ctx;
    AmrGpuMLCut gpu(ctx, NC, NC, 0, 0, 1, 1, cfg);
    gpu.momentFn = circle();
    gpu.fillPhysicalGhosts = reflectRef;
    gpu.init(ic);

    const double m0 = gpu.totalMass();
    worst = 0; mass = 0;
    double t = 0;
    for (int s = 0; s < 120; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4));
        amr.step(dt, t);
        gpu.step(dt, t);
        t += double(dt);
        const GridRef gb = gpu.coarseRef();
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (amr.covered(1, i / BC, j / BC)) continue;
                const double a = double(amr.base.at(NG + i, NG + j).rho);
                const double b = double(gb.at(NG + i, NG + j).rho);
                worst = std::max(worst,
                                 std::fabs(a - b) / std::max(std::fabs(a), 1e-3));
            }
        for (int l = 1; l < 3; ++l) {
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
        mass = std::max(mass, std::fabs(gpu.totalMass() - m0) / m0);
    }
}

} // namespace

int main() {
    bool ok = true;

    double w2 = 0, m2 = 0;
    run2(w2, m2);
    std::printf("gate 1 — 2-level viscous GPU vs CPU lock-step: rho %.3e "
                "(gate 1e-2), GPU mass %.3e (gate 1e-6)\n", w2, m2);
    ok = ok && w2 < 1e-2 && m2 < 1e-6;

    double w3 = 0, m3 = 0;
    run3(w3, m3);
    std::printf("gate 2 — 3-level viscous GPU vs CPU lock-step: rho %.3e "
                "(gate 1e-2), GPU mass %.3e (gate 1e-6)\n", w3, m3);
    ok = ok && w3 < 1e-2 && m3 < 1e-6;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

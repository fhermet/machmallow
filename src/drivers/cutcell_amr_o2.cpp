// 2nd-order cut cells in the production AMR (Amr2). The cut path now runs the
// 2nd-order operator (least-squares gradients + reconstruction to face centres
// / EB centroid, SSP-RK2): full cells reduce to 2nd-order MUSCL-HLLC, only cut
// cells get the aperture / EB / flux-redistribution treatment. RK2 reflux uses
// the time-averaged extensive fluxes 0.5*(F1+F2) so conservation is preserved.
//
//   gate 1 — composite mass conserved with O2 + reflux (body spanning a 4x4
//            block of patches, coarse-fine seams around it): drift < 1e-6;
//   gate 2 — order of accuracy: an isentropic vortex on an all-refined 2-level
//            hierarchy (cross-patch composite exercised at EVERY sibling seam)
//            converges at ~order 2 with cutCellO2, ~order 1 without.

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

// ---- gate 1: conservation with O2 + reflux ------------------------------
double o2MassDrift() {
    const int NC = 48, bc = 8;
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };
    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.cutCellO2 = true;
    cfg.reflux = true;
    cfg.regridEvery = 1 << 30;
    cfg.mu = 0;

    Amr2 amr(NC, NC, 0, 0, 1, 1, cfg);
    amr.momentFn = circle;
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillReflectiveLeft(g);  fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    const auto ic = [&](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    for (int j = 0; j < amr.coarse.toty(); ++j)
        for (int i = 0; i < amr.coarse.totx(); ++i)
            amr.coarse.at(i, j) = ic(double(amr.coarse.xc(i)),
                                     double(amr.coarse.yc(j)));
    std::vector<std::pair<int, int>> blocks;
    for (int bj = 1; bj <= 4; ++bj)
        for (int bi = 1; bi <= 4; ++bi) blocks.emplace_back(bi, bj);
    amr.restoreBlocks(blocks, 0);
    for (Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                p.grid.at(i, j) = ic(double(p.grid.xc(i)),
                                     double(p.grid.yc(j)));

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

// ---- gate 2: order of accuracy on the isentropic vortex -----------------
Cons vortexState(Real x, Real y, Real cx, Real cy) {
    constexpr Real beta = 5;
    const Real dx = x - cx, dy = y - cy;
    const Real r2 = dx * dx + dy * dy;
    const Real du = beta / Real(2 * M_PI) * std::exp(Real(0.5) * (1 - r2));
    const Real T = 1 - (GAMMA - 1) * beta * beta /
                           (Real(8) * GAMMA * Real(M_PI * M_PI)) *
                           std::exp(1 - r2);
    const Real rho = std::pow(T, Real(1) / (GAMMA - 1));
    return toCons({rho, 1 - du * dy, 1 + du * dx, rho * T});
}

// L1(rho) of the vortex advected to (7,7) over t=2 on an ALL-REFINED 2-level
// hierarchy (every base block is a patch): the composite is entirely fine and
// the cross-patch FRD / ghost exchange is exercised at every sibling seam.
double vortexL1(int N, bool o2) {
    const int bc = 8;
    // moments far outside the domain -> every cell full (kappa=1): the cut
    // operator reduces to plain MUSCL-HLLC, isolating the reconstruction order.
    const auto full = [](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(-100, -100, 0.1, x0, x1, y0, y1);
    };
    AmrConfig cfg;
    cfg.blockC = bc;
    cfg.maxLevels = 2;
    cfg.cutCell = true;
    cfg.cutCellO2 = o2;
    cfg.reflux = true;
    cfg.regridEvery = 1 << 30;
    cfg.periodicX = true;
    cfg.periodicY = true;

    Amr2 amr(N, N, 0, 0, 10, 10, cfg);
    amr.momentFn = full;
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillPeriodicX(g); fillPeriodicY(g);
    };
    for (int j = 0; j < amr.coarse.toty(); ++j)
        for (int i = 0; i < amr.coarse.totx(); ++i)
            amr.coarse.at(i, j) = vortexState(amr.coarse.xc(i),
                                              amr.coarse.yc(j), 5, 5);
    std::vector<std::pair<int, int>> blocks;
    for (int bj = 0; bj < N / bc; ++bj)
        for (int bi = 0; bi < N / bc; ++bi) blocks.emplace_back(bi, bj);
    amr.restoreBlocks(blocks, 0);
    for (Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                p.grid.at(i, j) =
                    vortexState(p.grid.xc(i), p.grid.yc(j), 5, 5);

    double t = 0;
    while (t < 2.0) {
        Real dt = amr.maxStableDtAll(Real(0.4));
        if (t + double(dt) > 2.0) dt = Real(2.0 - t);
        amr.step(dt, t);
        t += double(dt);
    }
    double e = 0;
    std::size_t n = 0;
    for (const Amr2::Patch& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i) {
                e += std::fabs(double(p.grid.at(i, j).rho) -
                               double(vortexState(p.grid.xc(i), p.grid.yc(j),
                                                  7, 7).rho));
                ++n;
            }
    return e / double(n);
}

double order(int N1, int N2, bool o2) {
    const double e1 = vortexL1(N1, o2), e2 = vortexL1(N2, o2);
    return std::log(e1 / e2) / std::log(double(N2) / double(N1));
}

} // namespace

int main() {
    bool ok = true;

    const double drift = o2MassDrift();
    std::printf("gate 1 — O2 cut-cell composite mass drift (RK2 + averaged-flux "
                "reflux): %.3e (gate 1e-6)\n", drift);
    ok = ok && drift < 1e-6;

    const double ord2 = order(32, 64, true);
    const double ord1 = order(32, 64, false);
    std::printf("gate 2 — isentropic vortex order (all-refined 2-level, every "
                "seam): O2 %.2f vs O1 %.2f (gate O2 > 1.7)\n", ord2, ord1);
    ok = ok && ord2 > 1.7;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

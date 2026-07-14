// GPU viscous cut-cell (Stokes/Fourier face fluxes + no-slip embedded-boundary
// traction, 2nd-order LSQ + RK2) validated two ways on planar Couette flow: a
// stationary no-slip wall carried by the immersed boundary at y=yw (solid
// below), a wall moving at U along the top edge, periodic in x. The steady
// solution is the exact linear profile u(y) = U (y-yw)/(1-yw).
//
//   gate 1 — GPU vs CPU stepCutCell(mu) in lock-step (worst rel u diff < 1e-2);
//   gate 2 — GPU steady profile within 5% of the exact linear Couette solution.

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

using namespace mm;

namespace {

const double YW = 0.2, U = 0.1, MU = 0.1;

// no-slip top wall moving at U, periodic x, solid below yw (bottom transmissive).
template <class G>
void fillCouette(G& g) {
    fillPeriodicX(g);
    fillTransmissiveBottom(g);
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Prim w = toPrim(g.at(i, NG + g.ny - 1 - k));
            w.u = Real(2 * U) - w.u;
            w.v = -w.v;
            g.at(i, NG + g.ny + k) = toCons(w);
        }
}

} // namespace

int main() {
    const int N = 64;
    const auto halfplane = [&](double x0, double x1, double y0, double y1) {
        return cutcell::halfplaneMoments(0, 1, YW, x0, x1, y0, y1);
    };

    Grid gc(N, N, 0, 0, 1, 1);
    const auto geo = cutcell::build(gc, halfplane);
    for (auto& q : gc.q) q = toCons({1, 0, 0, 1}); // rest

    MetalContext ctx;
    CutCell2DGpu gpu(ctx, N, N, gc.dx, gc.dy);
    gpu.setGeometry(geo);
    gpu.enableO2();
    gpu.setViscosity(Real(MU), heatConductivity(Real(MU)));
    GridRef gg = gpu.ref(0, 0);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i) gg.at(i, j) = gc.at(i, j);

    double worst = 0, t = 0;
    const double tEnd = 12.0; // long enough to reach steady state
    while (t < tEnd) {
        const Real dt = std::min(maxStableDt(gc, Real(0.4), Real(MU)),
                                 Real(tEnd - t));
        // CPU oracle (2nd-order viscous cut step).
        stepCutCell(gc, geo, dt, fillCouette<Grid>, /*limited=*/true, Real(MU));
        // GPU: SSP-RK2 with viscous divergence, ghosts refilled between stages.
        fillCouette(gg);
        gpu.divO2();
        gpu.rk2Stage1(dt);
        fillCouette(gg);
        gpu.divO2();
        gpu.rk2Stage2(dt);
        t += double(dt);
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const double a = double(toPrim(gc.at(i, j)).u);
                const double b = double(toPrim(gg.at(i, j)).u);
                worst = std::max(worst, std::fabs(a - b) / U);
            }
    }

    // Physical error of the GPU steady profile vs the exact linear solution.
    double emax = 0;
    const int col = NG + N / 2;
    for (int j = NG; j < NG + N; ++j) {
        if (geo.at(col, j).vol <= Real(1e-9)) continue;
        const double y = double(gc.yc(j));
        const double ue = U * (y - YW) / (1 - YW);
        emax = std::max(emax,
                        std::fabs(double(toPrim(gg.at(col, j)).u) - ue) / U);
    }

    std::printf("gate 1 — GPU viscous cut vs CPU stepCutCell(mu) lock-step: "
                "worst relative u diff %.3e (gate 1e-2)\n", worst);
    std::printf("gate 2 — GPU steady Couette vs exact linear profile: "
                "max|u-exact|/U %.3e (gate 5e-2)\n", emax);
    const bool ok = worst < 1e-2 && emax < 5e-2;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

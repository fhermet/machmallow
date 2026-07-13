// Phase 3 validation: 2nd-order cut-cell reconstruction.
//   1. linearity preservation — the least-squares gradients recover an exact
//      linear field on the irregular cut-cell stencils (necessary for 2nd
//      order);
//   2. order of accuracy — a smooth entropy wave advected along a 45° wall
//      (tangent slip boundary, so the wave is an exact solution) converges at
//      ~2nd order in L1, versus ~1st order for the plain Godunov scheme.

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

// 45° wall: solid = {x - y > 0.25}; fluid = {x - y <= 0.25}. As a half-plane
// a*x + b*y < c (solid) this is a=-1, b=1, c=-0.25.
cutcell::Geometry wedge45(const Grid& g) {
    return cutcell::build(g, [&](double x0, double x1, double y0, double y1) {
        return cutcell::halfplaneMoments(-1, 1, -0.25, x0, x1, y0, y1);
    });
}

} // namespace

int main() {
    bool ok = true;

    // ---- gate 1: linearity preservation of the LSQ gradients --------------
    {
        const int N = 96;
        Grid g(N, N, 0, 0, 1, 1);
        const auto geo = wedge45(g);
        // exact linear primitive field
        const double a_rho = 0.7, b_rho = -0.4, a_u = 0.3, b_u = 0.2;
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i) {
                const double x = double(g.xc(i)), y = double(g.yc(j));
                g.at(i, j) = toCons({Real(1 + a_rho * x + b_rho * y),
                                     Real(0.2 + a_u * x + b_u * y),
                                     Real(0.1 - 0.15 * x + 0.25 * y), Real(1)});
            }
        const auto grad = lsqGradients(g, geo);
        double emax = 0;
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                if (geo.at(i, j).vol <= Real(1e-9)) continue;
                const PrimGrad& gr = grad[g.idx(i, j)];
                emax = std::max({emax,
                                 std::fabs(double(gr.dx.rho) - a_rho),
                                 std::fabs(double(gr.dy.rho) - b_rho),
                                 std::fabs(double(gr.dx.u) - a_u),
                                 std::fabs(double(gr.dy.u) - b_u)});
            }
        std::printf("gate 1 — linearity preservation (LSQ gradient error on "
                    "cut cells): %.3e (gate 1e-4)\n", emax);
        ok = ok && emax < 1e-4;
    }

    // ---- gate 2: order of accuracy — entropy blob sliding along the wall --
    // a COMPACT 2D Gaussian entropy blob centred on the 45° wall, advected by
    // a uniform flow (1,1)/sqrt2 tangent to it. The blob stays in the interior
    // (domain boundaries only ever see uniform freestream, so their 1st-order
    // constant reconstruction is exact) and straddles the wall (so the cut-cell
    // EB reconstruction is exercised). Exact = the rigidly advected blob.
    // Run unlimited: this verifies the reconstruction's DESIGN order (the BJ
    // limiter would clip the smooth peak to ~1st order, the usual TVD 4/3 cap).
    {
        const double A = 0.2, sig = 0.06, s2 = std::sqrt(2.0);
        const double xc0 = 0.5, yc0 = 0.25;      // on the wall x - y = 0.25
        const auto exact = [&](double x, double y, double t) {
            const double cx = xc0 + t / s2, cy = yc0 + t / s2;
            const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            return Prim{Real(1 + A * std::exp(-r2 / (2 * sig * sig))),
                        Real(1 / s2), Real(1 / s2), Real(1)};
        };
        const double tEnd = 0.15;
        std::printf("gate 2 — entropy wave along a 45° wall, L1(rho) vs "
                    "exact:\n");
        double prevErr = 0, order = 0;
        for (int N : {32, 64, 128}) {
            Grid g(N, N, 0, 0, 1, 1);
            const auto geo = wedge45(g);
            for (int j = 0; j < g.toty(); ++j)
                for (int i = 0; i < g.totx(); ++i)
                    g.at(i, j) = toCons(exact(g.xc(i), g.yc(j), 0));
            double tNow = 0;
            const auto fill = [&](Grid& gg) {   // exact Dirichlet in ghosts
                for (int j = 0; j < gg.toty(); ++j)
                    for (int i = 0; i < gg.totx(); ++i)
                        if (i < NG || i >= NG + gg.nx || j < NG ||
                            j >= NG + gg.ny)
                            gg.at(i, j) = toCons(exact(gg.xc(i), gg.yc(j), tNow));
            };
            double t = 0;
            while (t < tEnd) {
                tNow = t;
                const Real dt =
                    std::min(maxStableDt(g, Real(0.4)), Real(tEnd - t));
                stepCutCell(g, geo, dt, fill, /*limited=*/false);
                t += double(dt);
            }
            double l1 = 0, area = 0;
            for (int j = NG; j < NG + N; ++j)
                for (int i = NG; i < NG + N; ++i) {
                    const double k = double(geo.at(i, j).vol);
                    if (k <= 1e-9) continue;
                    const double re =
                        double(exact(g.xc(i), g.yc(j), tEnd).rho);
                    l1 += k * g.dx * g.dy *
                          std::fabs(double(toPrim(g.at(i, j)).rho) - re);
                    area += k * g.dx * g.dy;
                }
            l1 /= area;
            if (prevErr > 0) order = std::log2(prevErr / l1);
            std::printf("    N=%3d  L1 = %.4e%s\n", N, l1,
                        prevErr > 0 ? (" order " + std::to_string(order)).c_str()
                                    : "");
            prevErr = l1;
        }
        std::printf("  final order = %.2f (gate > 1.7)\n", order);
        ok = ok && order > 1.7;
    }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

// Phase 4 validation: viscous no-slip on the embedded boundary.
// Planar Couette flow: a stationary no-slip wall carried by the immersed
// boundary at y = yw (solid below), a no-slip wall moving at U along the top
// (Cartesian) edge, periodic in x. The steady solution is the exact linear
// profile u(y) = U (y - yw)/(1 - yw); we start from rest, march the viscous
// cut-cell solver to steady state and compare.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

int main() {
    const double yw = 0.2, U = 0.1, mu = 0.1;   // low Mach, nu = mu/rho = 0.1
    bool ok = true;
    std::printf("cut-cell viscous — Couette, immersed no-slip wall at y=%.2f, "
                "top wall U=%.2f, mu=%.2f\n", yw, U, mu);
    std::printf("%6s %14s %10s\n", "N", "max|u-exact|/U", "order");
    double prev = 0, order = 0;
    for (int N : {48, 96}) {
        Grid g(N, N, 0, 0, 1, 1);
        // solid = { y < yw }  ->  half-plane a*x+b*y<c with a=0,b=1,c=yw
        const auto geo = cutcell::build(
            g, [&](double x0, double x1, double y0, double y1) {
                return cutcell::halfplaneMoments(0, 1, yw, x0, x1, y0, y1);
            });
        for (auto& q : g.q) q = toCons({1, 0, 0, 1});   // start from rest
        const auto fill = [&](Grid& gg) {
            fillPeriodicX(gg);
            // bottom edge sits inside the solid — value irrelevant
            fillTransmissiveBottom(gg);
            // top: no-slip wall moving at U (u_face = U, no penetration)
            for (int i = 0; i < gg.totx(); ++i)
                for (int k = 0; k < NG; ++k) {
                    Prim w = toPrim(gg.at(i, NG + gg.ny - 1 - k));
                    w.u = Real(2 * U) - w.u;
                    w.v = -w.v;
                    gg.at(i, NG + gg.ny + k) = toCons(w);
                }
        };
        const double tEnd = 12.0;                 // long enough to reach steady
        double t = 0;
        while (t < tEnd) {
            const Real dt = std::min(maxStableDt(g, Real(0.4), Real(mu)),
                                     Real(tEnd - t));
            stepCutCell(g, geo, dt, fill, /*limited=*/true, Real(mu));
            t += double(dt);
        }
        // compare the mid-column profile to the exact linear Couette solution
        double emax = 0;
        const int ic = NG + N / 2;
        for (int j = NG; j < NG + N; ++j) {
            if (geo.at(ic, j).vol <= Real(1e-9)) continue;
            const double y = double(g.yc(j));
            const double ue = U * (y - yw) / (1 - yw);
            emax = std::max(emax,
                            std::fabs(double(toPrim(g.at(ic, j)).u) - ue) / U);
        }
        if (prev > 0) order = std::log2(prev / emax);
        std::printf("%6d %14.4e %10s\n", N, emax,
                    prev > 0 ? std::to_string(order).c_str() : "-");
        prev = emax;
        ok = ok && emax < 0.05;                   // within 5% of exact
    }
    std::printf("  convergence order ~ %.2f\n", order);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

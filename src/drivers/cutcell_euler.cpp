// Phase 2a validation of the cut-cell inviscid solver: WELL-BALANCEDNESS.
// A uniform state at rest around an immersed cylinder must be preserved to
// round-off — the aperture-weighted face pressures and the embedded-boundary
// wall pressure cancel exactly through the divergence closure. This pins the
// EB flux signs and the aperture/closure assembly before we add a real flow
// and flux redistribution (phase 2b).

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
    const int N = 128;
    Grid g(N, N, 0, 0, 1, 1);
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto geo = cutcell::build(
        g, [&](double x0, double x1, double y0, double y1) {
            return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
        });
    std::printf("cut-cell inviscid — at-rest cylinder r=%.2f on %dx%d "
                "(cut cells %zu)\n",
                r, N, N, geo.nCut);

    const Prim w0{1, 0, 0, 1};
    const Cons u0 = toCons(w0);
    for (auto& q : g.q) q = u0;

    for (int s = 0; s < 100; ++s) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        const Real dt = maxStableDt(g, Real(0.4));
        stepCutCell(g, geo, dt);
    }

    double dev = 0;
    for (int j = NG; j < NG + N; ++j)
        for (int i = NG; i < NG + N; ++i) {
            if (geo.at(i, j).vol <= Real(1e-9)) continue;   // covered
            const Cons& q = g.at(i, j);
            dev = std::max({dev, std::fabs(double(q.rho) - double(u0.rho)),
                            std::fabs(double(q.mx)), std::fabs(double(q.my)),
                            std::fabs(double(q.E) - double(u0.E))});
        }
    std::printf("  max deviation from rest after 100 steps: %.3e (gate 1e-5)\n",
                dev);
    const bool ok = dev < 1e-5;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

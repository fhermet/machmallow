// Immersed solid body in the Cartesian grid — validation of the wall
// treatment (solid mask + reflective wall flux in step2D).
//
// Case: a planar shock of Mach Ms propagates in +x into a gas at rest and
// reflects off an IMMERSED WALL (solid block, face aligned with the grid).
// The wall pressure after reflection has an EXACT value (1D shock
// reflection):
//     p_r/p_i = ((3γ-1)ξ - (γ-1)) / ((γ-1)ξ + (γ+1)),   ξ = p_i/p_0
// (strong limit (3γ-1)/(γ-1) = 8 for γ=1.4). Since the face is aligned,
// there is no staircase error: this is an exact check of the immersed
// wall. We also check non-penetration (u ≈ 0 at the wall).
//
// We test TWO regimes:
//   Ms=2 : post-shock gas SUBSONIC toward the wall (M1≈0.96)
//   Ms=3 : post-shock gas SUPERSONIC toward the wall (M1≈1.36)
// The supersonic case specifically locks down the exact wall flux: the
// mirror wall + HLLC LEAKS in supersonic (its PVRS estimate keeps SL > 0
// and upwinds all the incoming flux) — hence `wallPressure`.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

// Reflection of a shock Ms off an immersed wall; returns the relative
// wall-pressure error vs the exact 1D value (and |u|/u_i).
struct Result { double pErr, uRel, p, pExact, M1; };

Result runReflection(double Ms, double tEnd) {
    const double G = double(GAMMA);
    const double rho0 = 1.0, p0 = 1.0;
    const double c0 = std::sqrt(G * p0 / rho0);

    const double xi = 1.0 + 2.0 * G / (G + 1.0) * (Ms * Ms - 1.0);
    const double p1 = xi * p0;
    const double rho1 = rho0 * (G + 1.0) * Ms * Ms /
                        ((G - 1.0) * Ms * Ms + 2.0);
    const double u1 = 2.0 / (G + 1.0) * (Ms - 1.0 / Ms) * c0;
    const double c1 = std::sqrt(G * p1 / rho1);
    const double p2 = p1 * ((3 * G - 1) * xi - (G - 1)) /
                      ((G - 1) * xi + (G + 1));

    const int nx = 400, ny = 4;
    const double L = 1.0, xWall = 0.7, xShock0 = 0.2;
    Grid g(nx, ny, 0, 0, L, Real(ny) * (L / nx));

    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            if (double(g.xc(i)) >= xWall) solid[g.idx(i, j)] = 1;

    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const bool behind = double(g.xc(i)) < xShock0;
            const Prim w = behind ? Prim{Real(rho1), Real(u1), 0, Real(p1)}
                                  : Prim{Real(rho0), 0, 0, Real(p0)};
            g.q[g.idx(i, j)] = toCons(w);
        }

    Scratch2D s;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt = std::min(maxStableDt(g, Real(0.4), 0),
                                 Real(tEnd - t));
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillPeriodicY(g);
        step2D(g, dt, s, 0, 0, 0, solid.data());
        t += dt;
    }

    int iw = NG;
    while (iw < NG + nx && double(g.xc(iw)) < xWall) ++iw;
    const int jm = NG + ny / 2;
    const Prim w = toPrim(g.at(iw - 1, jm));
    return {std::fabs(double(w.p) - p2) / p2, std::fabs(double(w.u)) / u1,
            double(w.p), p2, u1 / c1};
}

} // namespace

int main() {
    bool ok = true;
    // (Ms, tEnd, gate) — Ms=3 reaches the wall sooner, we measure sooner.
    const struct { double Ms, tEnd, tol; } cases[] = {
        {2.0, 0.32, 0.04}, {3.0, 0.22, 0.05}};
    for (const auto& c : cases) {
        const Result r = runReflection(c.Ms, c.tEnd);
        const bool pass = r.pErr < c.tol && r.uRel < 0.05;
        ok = ok && pass;
        std::printf("Ms=%.1f (post-shock M1=%.2f, %s toward the wall) : "
                    "p=%.3f vs %.3f exact  err %.2f%% (gate %.0f%%), "
                    "|u|/u_i=%.3f  %s\n",
                    c.Ms, r.M1, r.M1 < 1 ? "subsonic" : "SUPERSONIC",
                    r.p, r.pExact, 100 * r.pErr, 100 * c.tol, r.uRel,
                    pass ? "PASS" : "FAIL");
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

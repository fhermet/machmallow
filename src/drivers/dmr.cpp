// Double Mach reflection (Woodward & Colella 1984): Mach 10 shock hitting
// a reflecting wall at 60 degrees. Domain [0,4]x[0,1], wall starts at
// x = 1/6 on the bottom boundary. Qualitative AMR showcase + CPU
// performance baseline.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "io/VtiWriter.hpp"
#include "solver/Muscl2D.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr double TEND = 0.2;
constexpr double XWALL = 1.0 / 6.0;

// Pre-shock (quiescent) and post-shock (Mach 10, 60 deg) states.
const Cons PRE = toCons({Real(1.4), 0, 0, Real(1.0)});
const Cons POST = toCons({Real(8.0), Real(8.25 * std::sqrt(3.0) / 2.0),
                          Real(-8.25 * 0.5), Real(116.5)});

bool behindShock(double x, double y, double t) {
    // Shock line: passes through (XWALL, 0) at 60 deg, speed 10 (normal).
    return x < XWALL + (y + 20.0 * t) / std::sqrt(3.0);
}

void fillDmrGhosts(Grid& g, double t) {
    // Left: post-shock inflow.
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi) g.at(gi, j) = POST;
    fillTransmissiveRight(g);

    // Bottom: post-shock inflow ahead of the wall start, reflecting wall
    // after it.
    fillReflectiveBottom(g);
    for (int i = 0; i < g.totx(); ++i)
        if (g.xc(i) < XWALL)
            for (int gj = 0; gj < NG; ++gj) g.at(i, gj) = POST;

    // Top: exact shock position (time dependent).
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj) {
            const int j = NG + g.ny + gj;
            g.at(i, j) = behindShock(g.xc(i), g.yc(j), t) ? POST : PRE;
        }
}

} // namespace

int main(int argc, char** argv) {
    const int ny = (argc > 1) ? std::atoi(argv[1]) : 120;
    const int nx = 4 * ny;
    std::printf("Double Mach reflection, %dx%d (1/%d), t = %.2f\n", nx, ny,
                ny, TEND);

    Grid g(nx, ny, 0, 0, 4, 1);
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i)
            g.at(i, j) = behindShock(g.xc(i), g.yc(j), 0) ? POST : PRE;

    std::filesystem::create_directories("out");
    Scratch2D scratch;
    double t = 0, nextFrame = 0.05;
    int frame = 0, steps = 0;
    writeVti("out/dmr_0000.vti", g);

    const auto t0 = std::chrono::steady_clock::now();
    while (t < TEND) {
        fillDmrGhosts(g, t);
        Real dt = std::min(maxStableDt(g, CFL), Real(TEND - t));
        if (steps < 10) dt *= Real(0.3); // gentle start on the sharp IC
        step2D(g, dt, scratch);
        t += dt;
        ++steps;
        if (t >= nextFrame - 1e-12 || t >= TEND) {
            char name[64];
            std::snprintf(name, sizeof(name), "out/dmr_%04d.vti", ++frame);
            writeVti(name, g);
            nextFrame += 0.05;
        }
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

    const double mcells = double(nx) * ny * steps / wall / 1e6;
    std::printf("%d steps in %.1f s -> %.1f Mcell-steps/s\n", steps, wall,
                mcells);

    // Sanity: density must stay positive and bounded (max ~22 expected).
    Real rhoMin = 1e30f, rhoMax = 0;
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const Real r = toPrim(g.at(i, j)).rho;
            rhoMin = std::min(rhoMin, r);
            rhoMax = std::max(rhoMax, r);
        }
    std::printf("rho in [%.3f, %.3f]\n", double(rhoMin), double(rhoMax));
    if (rhoMin <= 0 || rhoMax > 30 || !std::isfinite(double(rhoMax))) {
        std::fprintf(stderr, "FAIL: unphysical density\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS — frames in out/dmr_*.vti\n");
    return EXIT_SUCCESS;
}

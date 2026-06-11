// Double Mach reflection on the CPU path. Qualitative validation +
// single-thread performance baseline.

#include "cases/Dmr.hpp"
#include "core/Grid.hpp"
#include "io/VtiWriter.hpp"
#include "solver/Muscl2D.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {
using namespace mm;
constexpr Real CFL = Real(0.4);
} // namespace

int main(int argc, char** argv) {
    const int ny = (argc > 1) ? std::atoi(argv[1]) : 120;
    const int nx = 4 * ny;
    std::printf("Double Mach reflection (CPU), %dx%d (1/%d), t = %.2f\n", nx,
                ny, ny, dmr::TEND);

    Grid g(nx, ny, 0, 0, 4, 1);
    dmr::init(g);

    std::filesystem::create_directories("out");
    Scratch2D scratch;
    double t = 0, nextFrame = 0.05;
    int frame = 0, steps = 0;
    writeVti("out/dmr_0000.vti", g);

    const auto t0 = std::chrono::steady_clock::now();
    while (t < dmr::TEND) {
        dmr::fillGhosts(g, t);
        Real dt = std::min(maxStableDt(g, CFL), Real(dmr::TEND - t));
        if (steps < 10) dt *= Real(0.3); // gentle start on the sharp IC
        step2D(g, dt, scratch);
        t += dt;
        ++steps;
        if (t >= nextFrame - 1e-12 || t >= dmr::TEND) {
            char name[64];
            std::snprintf(name, sizeof(name), "out/dmr_%04d.vti", ++frame);
            writeVti(name, g);
            nextFrame += 0.05;
        }
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

    std::printf("%d steps in %.1f s -> %.1f Mcell-steps/s\n", steps, wall,
                double(nx) * ny * steps / wall / 1e6);

    // Sanity: density must stay positive and bounded (max ~22 expected).
    Real rhoMin = Real(1e30), rhoMax = 0;
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

// Phase 3 driver: GPU vs CPU on the double Mach reflection.
// 1) Correctness: both paths advance the same state with the same dt
//    sequence; fields must agree to float32 tolerance.
// 2) Performance: full DMR runs on each path, Mcell-steps/s + speedup.

#include "backend/metal/Euler2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
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
using Clock = std::chrono::steady_clock;

constexpr Real CFL = Real(0.4);

Real startupDt(Real dt, int step) {
    return step < 10 ? dt * Real(0.3) : dt;
}

// Lock-step comparison over `nsteps`, same dt on both paths.
bool checkCorrectness(MetalContext& ctx, int ny, int nsteps) {
    const int nx = 4 * ny;
    Grid cpu(nx, ny, 0, 0, 4, 1);
    dmr::init(cpu);

    Euler2DGpu gpu(ctx, nx, ny, cpu.dx, cpu.dy);
    GridRef gref = gpu.ref(0, 0);
    dmr::init(gref);

    Scratch2D scratch;
    double t = 0, maxRel = 0;
    for (int s = 0; s < nsteps; ++s) {
        dmr::fillGhosts(cpu, t);
        dmr::fillGhosts(gref, t);
        const Real dt = startupDt(maxStableDt(cpu, CFL), s);
        step2D(cpu, dt, scratch);
        gpu.step(dt);
        t += dt;
    }
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const Cons& a = cpu.at(i, j);
            const Cons& b = gref.at(i, j);
            const Real* pa = &a.rho;
            const Real* pb = &b.rho;
            for (int k = 0; k < NVARS; ++k) {
                const double rel = std::fabs(double(pa[k]) - pb[k]) /
                                   (std::fabs(double(pa[k])) + 1e-3);
                maxRel = std::max(maxRel, rel);
            }
        }
    std::printf("correctness (%dx%d, %d steps): max rel diff = %.3e\n", nx,
                ny, nsteps, maxRel);
    return maxRel < 1e-2;
}

struct BenchResult {
    double wall;
    int steps;
};

BenchResult benchCpu(int ny) {
    const int nx = 4 * ny;
    Grid g(nx, ny, 0, 0, 4, 1);
    dmr::init(g);
    Scratch2D scratch;
    double t = 0;
    int steps = 0;
    const auto t0 = Clock::now();
    while (t < dmr::TEND) {
        dmr::fillGhosts(g, t);
        const Real dt = startupDt(
            std::min(maxStableDt(g, CFL), Real(dmr::TEND - t)), steps);
        step2D(g, dt, scratch);
        t += dt;
        ++steps;
    }
    return {std::chrono::duration<double>(Clock::now() - t0).count(), steps};
}

BenchResult benchGpu(MetalContext& ctx, int ny, bool writeOutput) {
    const int nx = 4 * ny;
    Euler2DGpu gpu(ctx, nx, ny, Real(4.0 / nx), Real(1.0 / ny));
    GridRef g = gpu.ref(0, 0);
    dmr::init(g);
    double t = 0;
    int steps = 0;
    const auto t0 = Clock::now();
    while (t < dmr::TEND) {
        dmr::fillGhosts(g, t);
        const Real dt = startupDt(
            std::min(gpu.maxStableDt(CFL), Real(dmr::TEND - t)), steps);
        gpu.step(dt);
        t += dt;
        ++steps;
    }
    const double wall =
        std::chrono::duration<double>(Clock::now() - t0).count();
    if (writeOutput) {
        std::filesystem::create_directories("out");
        writeVti("out/dmr_gpu_" + std::to_string(ny) + ".vti", g);
    }
    return {wall, steps};
}

} // namespace

int main(int argc, char** argv) {
    const int ny = (argc > 1) ? std::atoi(argv[1]) : 240;
    const bool gpuOnly =
        (argc > 2) && std::string_view(argv[2]) == "gpu";

    MetalContext ctx;
    std::printf("GPU: %s\n", ctx.device()->name()->utf8String());

    if (!checkCorrectness(ctx, 60, 30)) {
        std::fprintf(stderr, "FAIL: GPU diverges from CPU\n");
        return EXIT_FAILURE;
    }

    const int nx = 4 * ny;
    std::printf("\nbenchmark %dx%d (1/%d), t = %.2f\n", nx, ny, ny,
                dmr::TEND);

    const BenchResult g = benchGpu(ctx, ny, true);
    const double gRate = double(nx) * ny * g.steps / g.wall / 1e6;
    std::printf("GPU: %d steps in %.2f s -> %.1f Mcell-steps/s\n", g.steps,
                g.wall, gRate);

    if (!gpuOnly) {
        const BenchResult c = benchCpu(ny);
        const double cRate = double(nx) * ny * c.steps / c.wall / 1e6;
        std::printf("CPU: %d steps in %.2f s -> %.1f Mcell-steps/s\n",
                    c.steps, c.wall, cRate);
        std::printf("speedup: %.1fx\n", gRate / cRate);
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

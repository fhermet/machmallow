// Phase 5: hybrid CPU/GPU AMR on the double Mach reflection.
// 1) Correctness: AmrGpu advances in lock-step with the validated CPU
//    Amr2; restricted coarse fields must agree to float32 tolerance.
// 2) Performance: full DMR at fine resolution 1/256, hybrid vs CPU AMR,
//    with vtkOverlappingAMR (.vthb) snapshots for ParaView.

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/Dmr.hpp"
#include "io/VthbWriter.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using namespace mm;
using Clock = std::chrono::steady_clock;

constexpr Real CFL = Real(0.4);

int g_blockC = 8; // overridable from argv for block-size tuning

AmrConfig dmrConfig() {
    AmrConfig cfg;
    cfg.blockC = g_blockC;
    cfg.tagThreshold = Real(0.12);
    cfg.regridEvery = 4;
    return cfg;
}

Cons dmrIc(Real x, Real y) {
    return dmr::behindShock(x, y, 0) ? dmr::POST : dmr::PRE;
}

template <class AMR>
void setDmrCallbacks(AMR& amr) {
    amr.fillPhysicalGhosts = [](auto& g, double t) {
        dmr::fillGhosts(g, t);
    };
    amr.fillPatchPhysical = [](auto& g, double t, unsigned sides) {
        dmr::fillGhostsSides(g, t, sides);
    };
}

Real startupDt(Real dt, int step) {
    return step < 10 ? dt * Real(0.3) : dt;
}

bool checkCorrectness(MetalContext& ctx, int nsteps, bool subcycle) {
    AmrConfig cfg = dmrConfig();
    cfg.subcycle = subcycle;
    Amr2 cpu(128, 32, 0, 0, 4, 1, cfg);
    AmrGpu gpu(ctx, 128, 32, 0, 0, 4, 1, cfg);
    setDmrCallbacks(cpu);
    setDmrCallbacks(gpu);
    cpu.init(dmrIc);
    gpu.init(dmrIc);

    double t = 0;
    for (int s = 0; s < nsteps; ++s) {
        const Real dt = startupDt(cpu.maxStableDtAll(CFL), s);
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
    }

    // Compare the restricted coarse fields (composite view of both
    // hierarchies).
    const GridRef g = gpu.coarseRef();
    double maxRel = 0;
    for (int j = NG; j < NG + cpu.coarse.ny; ++j)
        for (int i = NG; i < NG + cpu.coarse.nx; ++i) {
            const Real* pa = &cpu.coarse.at(i, j).rho;
            const Real* pb = &g.at(i, j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    std::printf("correctness (%d lock-steps%s): max rel diff = %.3e, "
                "patches CPU %zu | GPU %zu\n",
                nsteps, subcycle ? ", subcycled" : "", maxRel,
                cpu.patches.size(), gpu.patches.size());
    return maxRel < 1e-2 && cpu.patches.size() == gpu.patches.size();
}

struct BenchResult {
    double wall = 0;
    int steps = 0;
    std::size_t cellSteps = 0;
    std::size_t maxPatches = 0;
};

template <class AMR>
BenchResult runDmr(AMR& amr, bool writeFrames) {
    amr.init(dmrIc);
    std::filesystem::create_directories("out");

    BenchResult r;
    double t = 0, nextFrame = 0.05;
    int frame = 0;
    const auto t0 = Clock::now();
    while (t < dmr::TEND) {
        const Real dt = startupDt(
            std::min(amr.maxStableDtAll(CFL), Real(dmr::TEND - t)),
            r.steps);
        amr.step(dt, t);
        t += dt;
        ++r.steps;
        r.cellSteps += amr.cellCount();
        r.maxPatches = std::max(r.maxPatches, amr.patches.size());
        if (writeFrames && (t >= nextFrame - 1e-12 || t >= dmr::TEND)) {
            char name[64];
            std::snprintf(name, sizeof(name), "out/dmr_amr_%04d", ++frame);
            writeVthb(name, amr);
            nextFrame += 0.05;
        }
    }
    r.wall = std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const int nyc = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int nxc = 4 * nyc;
    const bool gpuOnly =
        (argc > 2) && std::string_view(argv[2]) == "gpu";
    if (argc > 3) g_blockC = std::atoi(argv[3]);

    MetalContext ctx;
    std::printf("GPU: %s\n", ctx.device()->name()->utf8String());

    if (!checkCorrectness(ctx, 30, false) ||
        !checkCorrectness(ctx, 30, true)) {
        std::fprintf(stderr, "FAIL: hybrid AMR diverges from CPU AMR\n");
        return EXIT_FAILURE;
    }

    std::printf("\nDMR on 2-level AMR: coarse %dx%d (fine 1/%d), "
                "t = %.2f\n",
                nxc, nyc, 2 * nyc, dmr::TEND);

    AmrGpu gpu(ctx, nxc, nyc, 0, 0, 4, 1, dmrConfig());
    setDmrCallbacks(gpu);
    const BenchResult g = runDmr(gpu, true);
    std::printf("hybrid: %d steps in %.2f s -> %.1f Mcell-steps/s, "
                "max %zu/%d patches\n",
                g.steps, g.wall, g.cellSteps / g.wall / 1e6, g.maxPatches,
                gpu.blockCount());
    const auto& tm = gpu.timings;
    std::printf("breakdown: ghost %.0f%% | gpu %.0f%% | reflux %.0f%% | "
                "restrict %.0f%% | regrid %.0f%% (%.2f s timed)\n",
                100 * tm.ghost / tm.total(), 100 * tm.gpu / tm.total(),
                100 * tm.reflux / tm.total(),
                100 * tm.restrict_ / tm.total(),
                100 * tm.regrid / tm.total(), tm.total());

    // Equivalent-resolution uniform work for context.
    const double uniWork =
        double(2 * nxc) * (2 * nyc) * g.steps / 1e6;
    std::printf("work vs uniform 1/%d: %.0f vs %.0f Mcell-steps (%.0f%%)\n",
                2 * nyc, g.cellSteps / 1e6, uniWork,
                100.0 * g.cellSteps / 1e6 / uniWork);

    // Subcycled hybrid: half the coarse steps for the same physics.
    AmrConfig subCfg = dmrConfig();
    subCfg.subcycle = true;
    AmrGpu gpuSub(ctx, nxc, nyc, 0, 0, 4, 1, subCfg);
    setDmrCallbacks(gpuSub);
    const BenchResult s = runDmr(gpuSub, false);
    std::printf("hybrid subcycled: %d coarse steps in %.2f s (%.2fx vs "
                "single-rate)\n",
                s.steps, s.wall, g.wall / s.wall);

    if (!gpuOnly) {
        Amr2 cpu(nxc, nyc, 0, 0, 4, 1, dmrConfig());
        setDmrCallbacks(cpu);
        const BenchResult c = runDmr(cpu, false);
        std::printf("CPU AMR: %d steps in %.2f s -> %.1f Mcell-steps/s\n",
                    c.steps, c.wall, c.cellSteps / c.wall / 1e6);
        std::printf("hybrid speedup: %.1fx\n", c.wall / g.wall);
    }

    // Sanity on the final composite (via restricted coarse).
    const GridRef c = gpu.coarseRef();
    Real rhoMin = Real(1e30), rhoMax = 0;
    for (int j = NG; j < NG + c.ny; ++j)
        for (int i = NG; i < NG + c.nx; ++i) {
            const Real r = toPrim(c.at(i, j)).rho;
            rhoMin = std::min(rhoMin, r);
            rhoMax = std::max(rhoMax, r);
        }
    std::printf("rho in [%.3f, %.3f]\n", double(rhoMin), double(rhoMax));
    if (rhoMin <= 0 || rhoMax > 30 || !std::isfinite(double(rhoMax))) {
        std::fprintf(stderr, "FAIL: unphysical density\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS — AMR frames in out/dmr_amr_*.vthb\n");
    return EXIT_SUCCESS;
}

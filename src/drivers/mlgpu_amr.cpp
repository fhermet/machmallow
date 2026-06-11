// Hybrid multi-level AMR validation + 3-level DMR showcase.
// Gates:
//   1. AmrGpuML advances in lock-step with the CPU AmrML on a 3-level
//      subcycled DMR (same dt), composite base fields agree to fp32
//      tolerance with identical per-level patch counts
//   2. doubly periodic 3-level KH on the GPU: closed-domain mass drift
//      at the fp32 floor
// Then: 3-level DMR (base 1/64 -> finest 1/256) with timing, frames out.

#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/Dmr.hpp"
#include "core/Boundary.hpp"
#include "io/VtiWriter.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

using namespace mm;
using Clock = std::chrono::steady_clock;

constexpr Real CFL = Real(0.4);

AmrConfig dmrCfg() {
    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.tagThreshold = Real(0.12);
    return cfg;
}

Cons dmrIc(Real x, Real y) {
    return dmr::behindShock(x, y, 0) ? dmr::POST : dmr::PRE;
}

template <class AMR>
void wireDmr(AMR& amr) {
    amr.fillPhysicalGhosts = [](auto& g, double t) {
        dmr::fillGhosts(g, t);
    };
    amr.fillPatchPhysical = [](auto& g, double t, unsigned sides) {
        dmr::fillGhostsSides(g, t, sides);
    };
}

Real startupDt(Real dt, int s) { return s < 10 ? dt * Real(0.3) : dt; }

bool gate1_lockstep(MetalContext& ctx) {
    AmrML cpu(64, 16, 0, 0, 4, 1, dmrCfg());
    AmrGpuML gpu(ctx, 64, 16, 0, 0, 4, 1, dmrCfg());
    wireDmr(cpu);
    wireDmr(gpu);
    cpu.init(dmrIc);
    gpu.init(dmrIc);

    double t = 0;
    for (int s = 0; s < 30; ++s) {
        const Real dt = startupDt(cpu.maxStableDtAll(CFL), s);
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
    }
    const GridRef g = gpu.coarseRef();
    double maxRel = 0;
    for (int j = NG; j < NG + 16; ++j)
        for (int i = NG; i < NG + 64; ++i) {
            const Real* pa = &cpu.base.at(i, j).rho;
            const Real* pb = &g.at(i, j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    std::printf("gate 1 — 3-level DMR lock-step, 30 steps: max rel diff "
                "%.3e, patches L1 %zu|%zu L2 %zu|%zu\n",
                maxRel, cpu.patchCount(1), gpu.patchCount(1),
                cpu.patchCount(2), gpu.patchCount(2));
    return maxRel < 1e-2 && cpu.patchCount(1) == gpu.patchCount(1) &&
           cpu.patchCount(2) == gpu.patchCount(2);
}

bool gate2_periodic(MetalContext& ctx) {
    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.tagVelocity = Real(0.05);
    cfg.periodicX = cfg.periodicY = true;
    AmrGpuML amr(ctx, 64, 64, 0, 0, 1, 1, cfg);
    amr.fillPhysicalGhosts = [](GridRef& g, double) {
        fillPeriodicX(g);
        fillPeriodicY(g);
    };
    amr.init([](Real x, Real y) {
        const bool band = std::fabs(y - Real(0.5)) < Real(0.25);
        return toCons({band ? Real(2) : Real(1),
                       band ? Real(0.25) : Real(-0.25),
                       Real(0.01 * std::sin(4 * M_PI * double(x))),
                       Real(2.5)});
    });

    const double m0 = amr.totalMass();
    double t = 0, drift = 0;
    for (int s = 0; s < 150; ++s) {
        const Real dt = amr.maxStableDtAll(CFL);
        amr.step(dt, t);
        t += dt;
        drift = std::max(drift, std::fabs(amr.totalMass() - m0) / m0);
    }
    std::printf("gate 2 — 3-level periodic KH (GPU), 150 steps: mass "
                "drift %.3e (gate 1e-6), patches L1 %zu L2 %zu\n",
                drift, amr.patchCount(1), amr.patchCount(2));
    return drift < 1e-6;
}

} // namespace

int main(int argc, char** argv) {
    const int ny = (argc > 1) ? std::atoi(argv[1]) : 64;
    MetalContext ctx;
    std::printf("GPU: %s\n", ctx.device()->name()->utf8String());

    if (!gate1_lockstep(ctx) || !gate2_periodic(ctx)) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }

    // Showcase: 3-level DMR, base 4ny x ny -> finest 1/(4 ny).
    AmrGpuML amr(ctx, 4 * ny, ny, 0, 0, 4, 1, dmrCfg());
    wireDmr(amr);
    amr.init(dmrIc);

    std::filesystem::create_directories("out");
    double t = 0;
    int steps = 0;
    std::size_t cellSteps = 0;
    const auto t0 = Clock::now();
    while (t < dmr::TEND) {
        const Real dt = startupDt(
            std::min(amr.maxStableDtAll(CFL), Real(dmr::TEND - t)),
            steps);
        amr.step(dt, t);
        t += dt;
        ++steps;
        cellSteps += amr.cellCount();
    }
    const double wall =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("3-level DMR base 1/%d -> finest 1/%d: %d base steps in "
                "%.2f s, patches L1 %zu L2 %zu\n",
                ny, 4 * ny, steps, wall, amr.patchCount(1),
                amr.patchCount(2));

    const GridRef c = amr.coarseRef();
    Real rhoMin = Real(1e30), rhoMax = 0;
    for (int j = NG; j < NG + c.ny; ++j)
        for (int i = NG; i < NG + c.nx; ++i) {
            const Real r = toPrim(c.at(i, j)).rho;
            rhoMin = std::min(rhoMin, r);
            rhoMax = std::max(rhoMax, r);
        }
    std::printf("rho in [%.3f, %.3f]\n", double(rhoMin), double(rhoMax));
    writeVti("out/dmr_ml.vti", c);
    if (rhoMin <= 0 || rhoMax > 30 || !std::isfinite(double(rhoMax))) {
        std::fprintf(stderr, "FAIL: unphysical density\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

// Validation of velocity tagging + periodic AMR on Kelvin-Helmholtz.
// Gates:
//   1. velocity tagging works — a *uniform-density* shear layer (invisible
//      to the density criterion) must still get refined
//   2. periodic conservation — doubly periodic domain is closed, so the
//      composite mass must sit at the fp32 floor (exercises the wrapped
//      ghost fill and the wrapped refluxing at the seams)
//   3. CPU/GPU lock-step parity with periodic + velocity tagging on

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr Real U0 = Real(0.5);

// Uniform density: only the velocity criterion can see this flow.
Cons khIc(Real x, Real y) {
    const bool band = std::fabs(y - Real(0.5)) < Real(0.25);
    return toCons({Real(1), band ? U0 / 2 : -U0 / 2,
                   Real(0.01 * std::sin(4 * M_PI * double(x))),
                   Real(2.5)});
}

AmrConfig khConfig() {
    AmrConfig cfg;
    cfg.blockC = 8;
    cfg.tagThreshold = Real(0.08);
    cfg.tagVelocity = Real(0.05);
    cfg.regridEvery = 4;
    cfg.periodicX = cfg.periodicY = true;
    return cfg;
}

template <class AMR>
void wire(AMR& amr) {
    amr.fillPhysicalGhosts = [](auto& g, double) {
        fillPeriodicX(g);
        fillPeriodicY(g);
    };
}

} // namespace

int main() {
    constexpr int N = 64;
    MetalContext ctx;

    // 1 + 2: velocity tagging and closed-domain conservation (CPU ref).
    {
        Amr2 amr(N, N, 0, 0, 1, 1, khConfig());
        wire(amr);
        amr.init(khIc);
        if (amr.patches.empty()) {
            std::fprintf(stderr,
                         "FAIL: velocity tagging found no patches\n");
            return EXIT_FAILURE;
        }
        AmrConfig noVel = khConfig();
        noVel.tagVelocity = 0;
        Amr2 ref(N, N, 0, 0, 1, 1, noVel);
        wire(ref);
        ref.init(khIc);
        std::printf("tagging: %zu patches with velocity criterion, %zu "
                    "with density only\n",
                    amr.patches.size(), ref.patches.size());
        if (!ref.patches.empty()) {
            std::fprintf(stderr, "FAIL: density tagging should be blind "
                                 "to a uniform-density shear\n");
            return EXIT_FAILURE;
        }

        const double m0 = amr.totalMass();
        double t = 0, drift = 0;
        for (int s = 0; s < 200; ++s) {
            const Real dt = amr.maxStableDtAll(CFL);
            amr.step(dt, t);
            t += dt;
            drift = std::max(drift,
                             std::fabs(amr.totalMass() - m0) / m0);
        }
        std::printf("closed-domain mass drift (200 steps, %zu patches): "
                    "%.3e (gate 1e-6)\n",
                    amr.patches.size(), drift);
        if (drift > 1e-6) {
            std::fprintf(stderr, "FAIL: periodic seam leaks mass\n");
            return EXIT_FAILURE;
        }
    }

    // 3: hybrid lock-step parity.
    {
        Amr2 cpu(N, N, 0, 0, 1, 1, khConfig());
        AmrGpu gpu(ctx, N, N, 0, 0, 1, 1, khConfig());
        wire(cpu);
        wire(gpu);
        cpu.init(khIc);
        gpu.init(khIc);
        double t = 0;
        for (int s = 0; s < 30; ++s) {
            const Real dt = cpu.maxStableDtAll(CFL);
            cpu.step(dt, t);
            gpu.step(dt, t);
            t += dt;
        }
        const GridRef g = gpu.coarseRef();
        double maxRel = 0;
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                const Real* pa = &cpu.coarse.at(i, j).rho;
                const Real* pb = &g.at(i, j).rho;
                for (int k = 0; k < NVARS; ++k)
                    maxRel = std::max(
                        maxRel, std::fabs(double(pa[k]) - pb[k]) /
                                    (std::fabs(double(pa[k])) + 1e-3));
            }
        std::printf("lock-step CPU/GPU (30 steps): max rel diff = %.3e, "
                    "patches %zu | %zu\n",
                    maxRel, cpu.patches.size(), gpu.patches.size());
        if (maxRel > 1e-2 || cpu.patches.size() != gpu.patches.size()) {
            std::fprintf(stderr, "FAIL: hybrid diverges on periodic KH\n");
            return EXIT_FAILURE;
        }
    }

    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

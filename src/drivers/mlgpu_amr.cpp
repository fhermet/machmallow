// Hybrid multi-level AMR validation + 3-level DMR showcase.
// Gates:
//   1. AmrGpuML advances in lock-step with the CPU AmrML on a 3-level
//      subcycled DMR (same dt), composite base fields agree to fp32
//      tolerance with identical per-level patch counts
//   2. doubly periodic 3-level KH on the GPU: closed-domain mass drift
//      at the fp32 floor
//   3. two-gas Sod on 3-level subcycled AMR, full run in lock-step with
//      the CPU AmrML: composite L1 vs the generalized exact solution,
//      species mass drift, and CPU/GPU agreement
//   4. WENO5+RK3 Sod on the same hierarchy, full run in lock-step with
//      the CPU AmrML WENO path: per-stage ghosts, RK-weighted
//      refluxing and conservation on the GPU
// Then: 3-level DMR (base 1/64 -> finest 1/256) with timing, frames out.

#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/Dmr.hpp"
#include "core/Boundary.hpp"
#include "io/VtiWriter.hpp"
#include "numerics/ExactRiemann.hpp"
#include "physics/TwoGas.hpp"

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

bool gate3_speciesGpu(MetalContext& ctx) {
    const GasPair gas{Real(1.4), Real(1.6)};
    const exact::State L{1, 0, 1, 1.4};
    const exact::State R{0.125, 0, 0.1, 1.6};
    const double tEnd = 0.2;

    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.species = true;
    cfg.gamma1 = Real(1.4);
    cfg.gamma2 = Real(1.6);

    AmrML cpu(64, 16, 0, 0, 1, Real(0.25), cfg);
    AmrGpuML gpu(ctx, 64, 16, 0, 0, 1, Real(0.25), cfg);
    const auto wire = [](auto& amr) {
        amr.fillPhysicalGhosts = [](auto& g, double) {
            fillTransmissiveLeft(g);
            fillTransmissiveRight(g);
            fillTransmissiveBottom(g);
            fillTransmissiveTop(g);
        };
        amr.fillPatchPhysical = [](auto& g, double, unsigned sides) {
            if (sides & SideLeft) fillTransmissiveLeft(g);
            if (sides & SideRight) fillTransmissiveRight(g);
            if (sides & SideBottom) fillTransmissiveBottom(g);
            if (sides & SideTop) fillTransmissiveTop(g);
        };
    };
    wire(cpu);
    wire(gpu);
    const auto ic = [&](Real x, Real) {
        const bool right = x >= Real(0.5);
        const exact::State& st = right ? R : L;
        return toConsG({Real(st.rho), Real(st.u), 0, Real(st.p)},
                       gas.Gamma(right ? Real(1) : Real(0)));
    };
    const auto icY = [](Real x, Real) {
        return x >= Real(0.5) ? Real(1) : Real(0);
    };
    cpu.init(ic, icY);
    gpu.init(ic, icY);

    const double m0 = gpu.totalSpeciesMass();
    double t = 0, drift = 0;
    while (t < tEnd - 1e-12) {
        const Real dt =
            std::min(cpu.maxStableDtAll(CFL), Real(tEnd - t));
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
        drift = std::max(drift,
                         std::fabs(gpu.totalSpeciesMass() - m0) /
                             std::max(m0, 1e-30));
    }

    // CPU/GPU agreement on the base (uncovered cells)
    const GridRef g = gpu.coarseRef();
    const int bC = gpu.fineCells() / 2;
    double maxRel = 0;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i) {
            if (gpu.covered(1, i / bC, j / bC)) continue;
            const Real* pa = &cpu.base.at(NG + i, NG + j).rho;
            const Real* pb = &g.at(NG + i, NG + j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }

    // composite L1(rho) vs the generalized exact solution
    double err = 0;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i)
            if (!gpu.covered(1, i / bC, j / bC))
                err += std::fabs(
                           double(g.at(NG + i, NG + j).rho) -
                           exact::sample(L, R,
                                         (double(g.xc(NG + i)) - 0.5) /
                                             tEnd)
                               .rho) *
                       g.dx * g.dy;
    for (int l = 1; l < 3; ++l)
        for (const auto& p : gpu.level(l).patches) {
            const GridRef pg = gpu.patchRef(l, p);
            for (int j = NG; j < NG + gpu.fineCells(); ++j)
                for (int i = NG; i < NG + gpu.fineCells(); ++i) {
                    const int gi = 2 * p.ci0 + (i - NG);
                    const int gj = 2 * p.cj0 + (j - NG);
                    if (l < 2 && gpu.covered(l + 1, gi / bC, gj / bC))
                        continue;
                    err += std::fabs(
                               double(pg.at(i, j).rho) -
                               exact::sample(L, R,
                                             (double(pg.xc(i)) - 0.5) /
                                                 tEnd)
                                   .rho) *
                           pg.dx * pg.dy;
                }
        }
    err /= 0.25;
    std::printf("gate 3 — two-gas Sod on 3-level AMR (GPU): L1 = %.4e "
                "(gate 5e-3), species mass drift = %.3e (gate 1e-5), "
                "CPU/GPU max rel diff = %.3e (gate 1e-2), patches "
                "L1 %zu|%zu L2 %zu|%zu\n",
                err, drift, maxRel, cpu.patchCount(1), gpu.patchCount(1),
                cpu.patchCount(2), gpu.patchCount(2));
    return err < 5e-3 && drift < 1e-5 && maxRel < 1e-2 &&
           cpu.patchCount(1) == gpu.patchCount(1) &&
           cpu.patchCount(2) == gpu.patchCount(2);
}

bool gate4_wenoGpu(MetalContext& ctx) {
    const exact::State L{1, 0, 1}, R{0.125, 0, 0.1};
    const double tEnd = 0.2;

    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.weno = true;

    AmrML cpu(64, 16, 0, 0, 1, Real(0.25), cfg);
    AmrGpuML gpu(ctx, 64, 16, 0, 0, 1, Real(0.25), cfg);
    const auto wire = [](auto& amr) {
        amr.fillPhysicalGhosts = [](auto& g, double) {
            fillTransmissiveLeft(g);
            fillTransmissiveRight(g);
            fillTransmissiveBottom(g);
            fillTransmissiveTop(g);
        };
        amr.fillPatchPhysical = [](auto& g, double, unsigned sides) {
            if (sides & SideLeft) fillTransmissiveLeft(g);
            if (sides & SideRight) fillTransmissiveRight(g);
            if (sides & SideBottom) fillTransmissiveBottom(g);
            if (sides & SideTop) fillTransmissiveTop(g);
        };
    };
    wire(cpu);
    wire(gpu);
    const auto ic = [&](Real x, Real) {
        return toCons(x < Real(0.5) ? Prim{1, 0, 0, 1}
                                    : Prim{Real(0.125), 0, 0,
                                           Real(0.1)});
    };
    cpu.init(ic);
    gpu.init(ic);

    const double m0 = gpu.totalMass();
    double t = 0, drift = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt =
            std::min(cpu.maxStableDtAll(CFL), Real(tEnd - t));
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
        drift = std::max(drift,
                         std::fabs(gpu.totalMass() - m0) / m0);
    }

    const GridRef g = gpu.coarseRef();
    const int bC = gpu.fineCells() / 2;
    double maxRel = 0, err = 0;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i) {
            if (gpu.covered(1, i / bC, j / bC)) continue;
            const Real* pa = &cpu.base.at(NG + i, NG + j).rho;
            const Real* pb = &g.at(NG + i, NG + j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
            err += std::fabs(
                       double(pb[0]) -
                       exact::sample(L, R,
                                     (double(g.xc(NG + i)) - 0.5) /
                                         tEnd).rho) *
                   g.dx * g.dy;
        }
    for (int l = 1; l < 3; ++l)
        for (const auto& p : gpu.level(l).patches) {
            const GridRef pg = gpu.patchRef(l, p);
            for (int j = NG; j < NG + gpu.fineCells(); ++j)
                for (int i = NG; i < NG + gpu.fineCells(); ++i) {
                    const int gi = 2 * p.ci0 + (i - NG);
                    const int gj = 2 * p.cj0 + (j - NG);
                    if (l < 2 && gpu.covered(l + 1, gi / bC, gj / bC))
                        continue;
                    err += std::fabs(
                               double(pg.at(i, j).rho) -
                               exact::sample(L, R,
                                             (double(pg.xc(i)) - 0.5) /
                                                 tEnd).rho) *
                           pg.dx * pg.dy;
                }
        }
    err /= 0.25;
    std::printf("gate 4 — WENO5 Sod on 3-level AMR (GPU): L1 = %.4e "
                "(gate 6e-3), mass drift %.3e (gate 2e-5, RK fp32 "
                "floor), CPU/GPU max rel diff %.3e (gate 1e-2), patches "
                "L1 %zu|%zu L2 %zu|%zu\n",
                err, drift, maxRel, cpu.patchCount(1), gpu.patchCount(1),
                cpu.patchCount(2), gpu.patchCount(2));
    return err < 6e-3 && drift < 2e-5 && maxRel < 1e-2 &&
           cpu.patchCount(1) == gpu.patchCount(1) &&
           cpu.patchCount(2) == gpu.patchCount(2);
}

// Viscous WENO5 on the GPU: a diffusing shear layer on a 2-level
// hierarchy, GPU (AmrGpuML) in lock-step with the CPU AmrML WENO path.
// Exercises the Metal WENO viscous flux kernel + the pool path.
bool gate5_wenoViscousGpu(MetalContext& ctx) {
    constexpr Real MU = Real(5e-3), V0 = Real(0.2);
    const auto exactV = [&](double x, double t) {
        return 0.5 * double(V0) *
               std::erf((x - 0.5) / std::sqrt(4.0 * double(MU) * t));
    };
    AmrConfig cfg;
    cfg.maxLevels = 2;
    cfg.subcycle = true;
    cfg.weno = true;
    cfg.mu = MU;
    // rho is uniform, so tag on the velocity jump to refine the layer
    // and exercise the patch (pool) viscous WENO kernel
    cfg.tagThreshold = Real(1e30);
    cfg.tagVelocity = Real(0.02);
    cfg.regridEvery = 2;
    AmrML cpu(64, 16, 0, 0, 1, Real(0.25), cfg);
    AmrGpuML gpu(ctx, 64, 16, 0, 0, 1, Real(0.25), cfg);
    const auto wire = [](auto& amr) {
        amr.fillPhysicalGhosts = [](auto& g, double) {
            fillTransmissiveLeft(g);
            fillTransmissiveRight(g);
            fillPeriodicY(g);
        };
        amr.fillPatchPhysical = [](auto& g, double, unsigned sides) {
            if (sides & SideLeft) fillTransmissiveLeft(g);
            if (sides & SideRight) fillTransmissiveRight(g);
        };
    };
    wire(cpu);
    wire(gpu);
    const auto ic = [&](Real x, Real) {
        return toCons({Real(1), 0, Real(exactV(x, 0.05)), 1});
    };
    cpu.init(ic);
    gpu.init(ic);

    double t = 0.05;
    while (t < 0.2 * (1 - 1e-9)) {
        const Real dt =
            std::min(cpu.maxStableDtAll(CFL), Real(0.2 - t));
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
    }
    const GridRef g = gpu.coarseRef();
    const int bC = gpu.fineCells() / 2;
    double maxRel = 0, err = 0;
    int n = 0;
    const int jrow = NG + 8;
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 64; ++i) {
            if (gpu.covered(1, i / bC, j / bC)) continue;
            const Real* pa = &cpu.base.at(NG + i, NG + j).rho;
            const Real* pb = &g.at(NG + i, NG + j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
            if (NG + j == jrow) {
                err += std::fabs(double(toPrim(g.at(NG + i, NG + j)).v) -
                                 exactV(double(g.xc(NG + i)), 0.2));
                ++n;
            }
        }
    if (n) err /= n;
    std::printf("gate 5 — viscous WENO5 shear on 2-level AMR (GPU): "
                "mean |v-exact| %.3e (gate 5e-3), CPU/GPU max rel diff "
                "%.3e (gate 1e-2), patches %zu|%zu\n",
                err, maxRel, cpu.patchCount(1), gpu.patchCount(1));
    return err < 5e-3 && maxRel < 1e-2 &&
           cpu.patchCount(1) == gpu.patchCount(1);
}

} // namespace

int main(int argc, char** argv) {
    const int ny = (argc > 1) ? std::atoi(argv[1]) : 64;
    MetalContext ctx;
    std::printf("GPU: %s\n", ctx.device()->name()->utf8String());

    if (!gate1_lockstep(ctx) || !gate2_periodic(ctx) ||
        !gate3_speciesGpu(ctx) || !gate4_wenoGpu(ctx) ||
        !gate5_wenoViscousGpu(ctx)) {
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

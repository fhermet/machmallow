// GPU/hybrid vs pure-CPU throughput study (étude 1: speedup vs taille).
//
// Runs the SAME case at increasing resolution on both backends and
// reports throughput (Mcell-steps/s), wall time and speedup. Two cases:
//   - uniform (1 level, no AMR): isolates the raw per-cell compute
//     speedup — the GPU's best case, no CPU orchestration in the loop;
//   - DMR (3-level hybrid AMR): the realized speedup, where the CPU
//     still does ghost fill / refluxing / restriction / tagging /
//     regrid every step (Amdahl: that serial fraction caps the gain).
//
// IMPORTANT framing: the pure-CPU class (AmrML) is SINGLE-THREADED
// (stepLevel_ loops patches serially on one core), so this measures the
// GPU against ONE M4 core, not the full CPU. It is the honest number
// for what the code does today; a fairer "GPU vs full CPU" would need a
// threaded CPU step (noted in the ROADMAP).
//
// Apple-Silicon timing is noisy (~30% on small cases, GPU frequency
// governor), so each point is the BEST of several trials, with warm-up
// steps discarded (shader compile + GPU spin-up). Writes
// out/benchmark.csv for tools/plot_benchmark.py.

#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/Dmr.hpp"
#include "core/Boundary.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mm;
using Clock = std::chrono::steady_clock;

namespace {

constexpr Real CFL = Real(0.4);
constexpr int WARMUP = 3, TIMED = 30, TRIALS = 3;

// Time TIMED steps after WARMUP, summing the cells advanced per step.
template <class AMR>
double timeBlock(AMR& amr, std::size_t& cellSteps) {
    double t = 0;
    for (int s = 0; s < WARMUP; ++s) {
        const Real dt = amr.maxStableDtAll(CFL);
        amr.step(dt, t);
        t += dt;
    }
    const auto t0 = Clock::now();
    std::size_t cs = 0;
    for (int s = 0; s < TIMED; ++s) {
        const Real dt = amr.maxStableDtAll(CFL);
        amr.step(dt, t);
        t += dt;
        cs += amr.cellCount();
    }
    cellSteps = cs;
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Row {
    std::string scase, backend;
    int N;
    std::size_t cells;
    double mcellps, wall;
};

void record(std::vector<Row>& out, const char* scase, const char* be,
            int N, std::size_t cells, std::size_t cellSteps,
            double bestWall) {
    const double mcellps = double(cellSteps) / bestWall / 1e6;
    out.push_back({scase, be, N, cells, mcellps, bestWall});
    std::printf("  %-8s %-7s N=%-5d cells=%-9zu %8.1f Mcell/s  "
                "%7.3f s\n",
                scase, be, N, cells, mcellps, bestWall);
}

// ---- uniform single-level box (periodic), smooth IC -----------------------
AmrConfig uniformCfg() {
    AmrConfig c;
    c.maxLevels = 1;
    c.periodicX = c.periodicY = true;
    return c;
}
Cons smoothIc(Real x, Real y) {
    const Real rho =
        Real(1 + 0.2 * std::sin(2 * M_PI * double(x)) *
                     std::cos(2 * M_PI * double(y)));
    return toCons({rho, Real(0.5), Real(-0.3), Real(1)});
}
template <class AMR>
void wireUniform(AMR& amr) {
    amr.fillPhysicalGhosts = [](auto& g, double) {
        fillPeriodicX(g);
        fillPeriodicY(g);
    };
    amr.fillPatchPhysical = [](auto&, double, unsigned) {};
}

// ---- 3-level DMR (the real AMR workload) ----------------------------------
AmrConfig dmrCfg() {
    AmrConfig c;
    c.maxLevels = 3;
    c.subcycle = true;
    c.tagThreshold = Real(0.12);
    return c;
}
Cons dmrIc(Real x, Real y) {
    return dmr::behindShock(x, y, 0) ? dmr::POST : dmr::PRE;
}
template <class AMR>
void wireDmr(AMR& amr) {
    amr.fillPhysicalGhosts = [](auto& g, double t) {
        dmr::fillGhosts(g, t);
    };
    amr.fillPatchPhysical = [](auto& g, double t, unsigned s) {
        dmr::fillGhostsSides(g, t, s);
    };
}

} // namespace

int main() {
    MetalContext ctx;
    std::printf("GPU: %s | CPU path is single-threaded (one core)\n",
                ctx.device()->name()->utf8String());
    std::printf("warmup %d, timed %d steps, best of %d trials\n\n",
                WARMUP, TIMED, TRIALS);

    std::vector<Row> rows;

    std::printf("== uniform box (1 level, no AMR orchestration) ==\n");
    for (int N : {64, 128, 256, 512, 1024}) {
        std::size_t cpuCells = 0, gpuCells = 0, cs = 0;
        double cpuW = 1e30, gpuW = 1e30;
        for (int r = 0; r < TRIALS; ++r) {
            AmrML cpu(N, N, 0, 0, 1, 1, uniformCfg());
            wireUniform(cpu);
            cpu.init(smoothIc);
            cpuW = std::min(cpuW, timeBlock(cpu, cs));
            cpuCells = cpu.cellCount();
        }
        record(rows, "uniform", "cpu", N, cpuCells, cs, cpuW);
        for (int r = 0; r < TRIALS; ++r) {
            AmrGpuML gpu(ctx, N, N, 0, 0, 1, 1, uniformCfg());
            wireUniform(gpu);
            gpu.init(smoothIc);
            gpuW = std::min(gpuW, timeBlock(gpu, cs));
            gpuCells = gpu.cellCount();
        }
        record(rows, "uniform", "hybrid", N, gpuCells, cs, gpuW);
    }

    std::printf("\n== DMR (3-level hybrid AMR) ==\n");
    for (int ny : {16, 32, 64, 128}) {
        const int nx = 4 * ny;
        std::size_t cpuCells = 0, gpuCells = 0, cs = 0;
        double cpuW = 1e30, gpuW = 1e30;
        for (int r = 0; r < TRIALS; ++r) {
            AmrML cpu(nx, ny, 0, 0, 4, 1, dmrCfg());
            wireDmr(cpu);
            cpu.init(dmrIc);
            cpuW = std::min(cpuW, timeBlock(cpu, cs));
            cpuCells = cpu.cellCount();
        }
        record(rows, "dmr", "cpu", ny, cpuCells, cs, cpuW);
        for (int r = 0; r < TRIALS; ++r) {
            AmrGpuML gpu(ctx, nx, ny, 0, 0, 4, 1, dmrCfg());
            wireDmr(gpu);
            gpu.init(dmrIc);
            gpuW = std::min(gpuW, timeBlock(gpu, cs));
            gpuCells = gpu.cellCount();
        }
        record(rows, "dmr", "hybrid", ny, gpuCells, cs, gpuW);
    }

    std::filesystem::create_directories("out");
    FILE* f = std::fopen("out/benchmark.csv", "w");
    std::fprintf(f, "case,backend,N,cells,mcell_per_s,wall_s\n");
    for (const Row& r : rows)
        std::fprintf(f, "%s,%s,%d,%zu,%.6e,%.6e\n", r.scase.c_str(),
                     r.backend.c_str(), r.N, r.cells, r.mcellps, r.wall);
    std::fclose(f);
    std::printf("\nwrote out/benchmark.csv (%zu rows)\n", rows.size());
    std::printf("plot: python3 tools/plot_benchmark.py\n");
    return 0;
}

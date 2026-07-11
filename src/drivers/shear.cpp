// Phase 7b validation: viscous diffusion of a shear layer.
// With rho ~ 1 and a small transverse velocity step, the y-momentum
// decouples and obeys the heat equation exactly:
//   v(x, t) = (V0/2) * erf((x - x0) / sqrt(4 nu t)),  nu = mu / rho.
// Start from the exact profile at t0, advance to t1, compare. Quantifies
// the viscous fluxes (2nd-order convergence) and checks GPU parity.

#include "backend/metal/Euler2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr Real MU = Real(5e-3); // rho = 1 -> nu = MU
constexpr Real V0 = Real(0.2);
// Thin initial layer (~2 cells at N=64) so the discretization error sits
// well above the fp32 floor on every grid of the study.
constexpr double T0 = 0.05, T1 = 0.2;
constexpr int NY = 8;

double exactV(double x, double t) {
    return 0.5 * V0 * std::erf((x - 0.5) / std::sqrt(4.0 * double(MU) * t));
}

void bc(Grid& g) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
}

void initShear(Grid& g) {
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            g.at(i, j) = toCons(
                {Real(1), Real(0), Real(exactV(g.xc(i), T0)), Real(1)});
}

double runCpu(int nx) {
    Grid g(nx, NY, 0, 0, 1, Real(NY) / nx);
    initShear(g);
    Scratch2D s;
    double t = T0;
    while (t < T1) {
        bc(g);
        const Real dt =
            std::min(maxStableDt(g, CFL, MU), Real(T1 - t));
        step2D(g, dt, s, MU);
        t += dt;
    }
    double err = 0;
    const int j = NG + NY / 2;
    for (int i = NG; i < NG + nx; ++i)
        err += std::fabs(double(toPrim(g.at(i, j)).v) -
                         exactV(g.xc(i), T1)) *
               g.dx;
    if (nx == 256) { // transverse velocity profile vs exact erf (vv figure)
        if (FILE* pf = std::fopen("out/shear_profile.csv", "w")) {
            std::fprintf(pf, "x,v,v_exact\n");
            for (int i = NG; i < NG + nx; ++i)
                std::fprintf(pf, "%.6g,%.6g,%.6g\n", double(g.xc(i)),
                             double(toPrim(g.at(i, j)).v),
                             exactV(g.xc(i), T1));
            std::fclose(pf);
        }
    }
    return err;
}

// GPU parity: same dt sequence on both paths for `nsteps`.
bool checkGpuParity(MetalContext& ctx, int nx, int nsteps) {
    Grid cpu(nx, NY, 0, 0, 1, Real(NY) / nx);
    initShear(cpu);

    Euler2DGpu gpu(ctx, nx, NY, cpu.dx, cpu.dy);
    gpu.setViscosity(MU);
    GridRef gref = gpu.ref(0, 0);
    for (int j = NG; j < NG + NY; ++j)
        for (int i = NG; i < NG + nx; ++i) gref.at(i, j) = cpu.at(i, j);

    Scratch2D s;
    for (int k = 0; k < nsteps; ++k) {
        bc(cpu);
        for (int j = 0; j < cpu.toty(); ++j) // mirror ghosts to GPU view
            for (int i = 0; i < cpu.totx(); ++i)
                gref.at(i, j) = cpu.at(i, j);
        const Real dt = maxStableDt(cpu, CFL, MU);
        step2D(cpu, dt, s, MU);
        gpu.step(dt);
    }
    double maxRel = 0;
    for (int j = NG; j < NG + NY; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const Real* pa = &cpu.at(i, j).rho;
            const Real* pb = &gref.at(i, j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    std::printf("GPU parity (%d steps): max rel diff = %.3e\n", nsteps,
                maxRel);
    return maxRel < 1e-4;
}

} // namespace

int main() {
    std::printf("Viscous shear layer, mu = %.0e, t %.2f -> %.2f\n",
                double(MU), T0, T1);
    std::printf("%8s %12s %8s\n", "N", "L1(v)", "order");

    const int grids[] = {64, 128, 256};
    std::vector<double> errs;
    for (int n : grids) {
        errs.push_back(runCpu(n));
        const std::size_t k = errs.size();
        if (k > 1)
            std::printf("%8d %12.4e %8.2f\n", n, errs.back(),
                        std::log2(errs[k - 2] / errs[k - 1]));
        else
            std::printf("%8d %12.4e %8s\n", n, errs.back(), "-");
    }
    const double meanOrder =
        std::log2(errs.front() / errs.back()) / double(errs.size() - 1);
    std::printf("mean order: %.2f\n", meanOrder);

    MetalContext ctx;
    const bool parity = checkGpuParity(ctx, 128, 50);

    if (meanOrder < 1.5 || !parity) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

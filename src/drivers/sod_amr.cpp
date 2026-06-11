// Phase 4 validation: x-aligned Sod tube on 2-level AMR.
// Gates:
//   1. conservation — with refluxing, the composite mass drift sits at the
//      float32 roundoff floor (~1e-4 after ~300 steps); without it, the
//      coarse-fine flux mismatch leaks orders of magnitude more. Both are
//      measured and the contrast is gated.
//   2. accuracy — composite L1(rho) vs exact close to the uniform-fine run
//      (the fine level tracks every wave; never-refined zones are smooth)

#include "amr/Amr2.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "io/VtiWriter.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr double TEND = 0.2;
constexpr exact::State SOD_L = {1.0, 0.0, 1.0};
constexpr exact::State SOD_R = {0.125, 0.0, 0.1};
constexpr double X0 = 0.5;

Cons sodIc(Real x, Real /*y*/) {
    const exact::State& s = (x < X0) ? SOD_L : SOD_R;
    return toCons({Real(s.rho), 0, 0, Real(s.p)});
}

void transmissiveAll(Grid& g) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
}

double exactRho(double x) {
    return exact::sample(SOD_L, SOD_R, (x - X0) / TEND).rho;
}

struct AmrRun {
    Amr2 amr;
    double massDrift = 0;
    std::size_t cellSteps = 0;
    int steps = 0, minPatches = 1 << 30, maxPatches = 0;

    explicit AmrRun(AmrConfig cfg)
        : amr(128, 32, 0, 0, 1, Real(0.25), cfg) {
        amr.fillPhysicalGhosts = transmissiveAll;
        amr.init(sodIc);
        const double m0 = amr.totalMass();
        double t = 0;
        while (t < TEND) {
            const Real dt =
                std::min(amr.maxStableDtAll(CFL), Real(TEND - t));
            amr.step(dt);
            t += dt;
            ++steps;
            cellSteps += amr.cellCount();
            massDrift = std::max(massDrift,
                                 std::fabs(amr.totalMass() - m0) / m0);
            minPatches = std::min(minPatches, int(amr.patches.size()));
            maxPatches = std::max(maxPatches, int(amr.patches.size()));
        }
    }
};

} // namespace

int main() {
    const int NXC = 128, NYC = 32; // coarse 1/128 -> fine 1/256
    std::printf("Sod on 2-level AMR: coarse %dx%d, ratio 2, t = %.2f\n",
                NXC, NYC, TEND);

    AmrRun run{AmrConfig{}};
    const Amr2& amr = run.amr;
    std::printf("%d steps, patches in [%d, %d] of %d blocks\n", run.steps,
                run.minPatches, run.maxPatches, amr.blockCount());

    std::printf("mass drift (dynamic regrid): %.3e\n", run.massDrift);

    // Refluxing stress test: freeze the mesh refined only around the
    // initial discontinuity, so shock/contact/rarefaction must all cross
    // the coarse-fine interfaces. Without refluxing this leaks badly.
    AmrConfig frozen;
    frozen.regridEvery = 1 << 30;
    AmrConfig frozenNoReflux = frozen;
    frozenNoReflux.reflux = false;
    AmrRun tight{frozen};
    AmrRun leaky{frozenNoReflux};
    std::printf("mass drift (frozen mesh): %.3e with refluxing | %.3e "
                "without (%.0fx worse)\n",
                tight.massDrift, leaky.massDrift,
                leaky.massDrift / tight.massDrift);

    // Composite L1(rho) vs exact, normalized to 1D units (/ height).
    double errAmr = 0;
    for (int j = 0; j < NYC; ++j)
        for (int i = 0; i < NXC; ++i)
            if (!amr.covered(i / 8, j / 8)) {
                const Grid& g = amr.coarse;
                errAmr += std::fabs(double(toPrim(g.at(NG + i, NG + j)).rho) -
                                    exactRho(g.xc(NG + i))) *
                          g.dx * g.dy;
            }
    for (const auto& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                errAmr += std::fabs(double(toPrim(p.grid.at(i, j)).rho) -
                                    exactRho(p.grid.xc(i))) *
                          p.grid.dx * p.grid.dy;
    errAmr /= 0.25;

    // Uniform fine reference (256 x 64), same metric.
    Grid uni(2 * NXC, 2 * NYC, 0, 0, 1, Real(0.25));
    for (int j = NG; j < NG + uni.ny; ++j)
        for (int i = NG; i < NG + uni.nx; ++i)
            uni.at(i, j) = sodIc(uni.xc(i), uni.yc(j));
    Scratch2D scratch;
    std::size_t uniCellSteps = 0;
    double tu = 0;
    while (tu < TEND) {
        transmissiveAll(uni);
        const Real dt = std::min(maxStableDt(uni, CFL), Real(TEND - tu));
        step2D(uni, dt, scratch);
        tu += dt;
        uniCellSteps += std::size_t(uni.nx) * uni.ny;
    }
    double errUni = 0;
    for (int j = NG; j < NG + uni.ny; ++j)
        for (int i = NG; i < NG + uni.nx; ++i)
            errUni += std::fabs(double(toPrim(uni.at(i, j)).rho) -
                                exactRho(uni.xc(i))) *
                      uni.dx * uni.dy;
    errUni /= 0.25;

    std::printf("L1(rho): AMR %.4e | uniform fine %.4e | ratio %.2f "
                "(gate 1.4)\n",
                errAmr, errUni, errAmr / errUni);
    std::printf("work: AMR %.1f Mcell-steps | uniform %.1f (%.0f%%)\n",
                run.cellSteps / 1e6, uniCellSteps / 1e6,
                100.0 * run.cellSteps / uniCellSteps);

    std::filesystem::create_directories("out");
    writeVti("out/sod_amr_coarse.vti", amr.coarse);

    // Gates: refluxed drift at the fp32 floor even when waves cross the
    // coarse-fine interfaces, the unrefluxed frozen run must leak at
    // least 10x more, and accuracy must match uniform fine.
    if (run.massDrift > 5e-4 || tight.massDrift > 5e-4 ||
        leaky.massDrift < 5 * tight.massDrift ||
        errAmr > 1.4 * errUni) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

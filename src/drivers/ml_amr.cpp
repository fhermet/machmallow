// Multi-level AMR validation (CPU reference class AmrML).
// Gates:
//   1. with maxLevels = 2, AmrML reproduces the battle-tested Amr2 on a
//      subcycled Sod run (same dt sequence) to fp32 reordering tolerance
//   2. 3-level Sod (base 1/64 -> finest 1/256): composite L1 vs the
//      exact solution stays close to a uniform 1/256 run
//   3. doubly periodic 3-level KH: closed-domain mass drift at the fp32
//      floor (multi-level refluxing + periodic wrap, all pairs)
//   4. proper nesting invariant holds after every regrid

#include "amr/Amr2.hpp"
#include "amr/AmrML.hpp"
#include "core/Boundary.hpp"
#include "numerics/ExactRiemann.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr exact::State SOD_L = {1.0, 0.0, 1.0};
constexpr exact::State SOD_R = {0.125, 0.0, 0.1};
constexpr double SOD_T = 0.2;

Cons sodIc(Real x, Real) {
    const exact::State& s = (x < Real(0.5)) ? SOD_L : SOD_R;
    return toCons({Real(s.rho), 0, 0, Real(s.p)});
}

void transmissiveAll(Grid& g, double) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
}
void transmissiveSides(Grid& g, double, unsigned sides) {
    if (sides & SideLeft) fillTransmissiveLeft(g);
    if (sides & SideRight) fillTransmissiveRight(g);
    if (sides & SideBottom) fillTransmissiveBottom(g);
    if (sides & SideTop) fillTransmissiveTop(g);
}

template <class AMR>
void wireSod(AMR& amr) {
    amr.fillPhysicalGhosts = transmissiveAll;
    amr.fillPatchPhysical = transmissiveSides;
}

double exactRho(double x) {
    return exact::sample(SOD_L, SOD_R, (x - 0.5) / SOD_T).rho;
}

// Composite L1(rho) of an AmrML Sod run, normalized to 1D units.
double l1Sod(const AmrML& amr) {
    double err = 0;
    const Grid& b = amr.base;
    const int bC = amr.fineCells() / 2;
    for (int j = 0; j < b.ny; ++j)
        for (int i = 0; i < b.nx; ++i)
            if (amr.numLevels() < 2 || !amr.covered(1, i / bC, j / bC))
                err += std::fabs(double(toPrim(b.at(NG + i, NG + j)).rho) -
                                 exactRho(b.xc(NG + i))) *
                       b.dx * b.dy;
    for (int l = 1; l < amr.numLevels(); ++l)
        for (const auto& p : amr.level(l).patches)
            for (int j = 0; j < amr.fineCells(); ++j)
                for (int i = 0; i < amr.fineCells(); ++i) {
                    const int gi = 2 * p.ci0 + i, gj = 2 * p.cj0 + j;
                    if (l < amr.numLevels() - 1 &&
                        amr.covered(l + 1, gi / bC, gj / bC))
                        continue;
                    err += std::fabs(
                               double(toPrim(p.grid.at(NG + i, NG + j))
                                          .rho) -
                               exactRho(p.grid.xc(NG + i))) *
                           p.grid.dx * p.grid.dy;
                }
    return err / 0.25;
}

bool gate1_twoLevelEquivalence() {
    AmrConfig cfg;
    cfg.subcycle = true;
    cfg.maxLevels = 2;
    Amr2 ref(128, 32, 0, 0, 1, Real(0.25), cfg);
    AmrML ml(128, 32, 0, 0, 1, Real(0.25), cfg);
    wireSod(ref);
    wireSod(ml);
    ref.init(sodIc);
    ml.init(sodIc);

    double t = 0;
    for (int s = 0; s < 100; ++s) {
        const Real dt = ref.maxStableDtAll(CFL);
        ref.step(dt, t);
        ml.step(dt, t);
        t += dt;
    }
    double maxRel = 0;
    for (int j = NG; j < NG + 32; ++j)
        for (int i = NG; i < NG + 128; ++i) {
            const Real* pa = &ref.coarse.at(i, j).rho;
            const Real* pb = &ml.base.at(i, j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    std::printf("gate 1 — AmrML(2) vs Amr2, 100 subcycled steps: max rel "
                "diff %.3e, patches %zu | %zu\n",
                maxRel, ref.patches.size(), ml.patchCount(1));
    return maxRel < 1e-3 && ref.patches.size() == ml.patchCount(1);
}

bool gate2_threeLevelSod() {
    AmrConfig cfg;
    cfg.subcycle = true;
    cfg.maxLevels = 3;
    AmrML amr(64, 16, 0, 0, 1, Real(0.25), cfg);
    wireSod(amr);
    amr.init(sodIc);

    const double m0 = amr.totalMass();
    double t = 0, drift = 0;
    bool nested = true;
    int steps = 0;
    while (t < SOD_T) {
        const Real dt =
            std::min(amr.maxStableDtAll(CFL), Real(SOD_T - t));
        amr.step(dt, t);
        t += dt;
        ++steps;
        drift = std::max(drift, std::fabs(amr.totalMass() - m0) / m0);
        nested = nested && amr.checkNesting();
    }
    const double errMl = l1Sod(amr);

    // Uniform 1/256 reference (256 x 64), same metric.
    Grid uni(256, 64, 0, 0, 1, Real(0.25));
    for (int j = NG; j < NG + uni.ny; ++j)
        for (int i = NG; i < NG + uni.nx; ++i)
            uni.at(i, j) = sodIc(uni.xc(i), uni.yc(j));
    Scratch2D s;
    double tu = 0;
    while (tu < SOD_T) {
        transmissiveAll(uni, tu);
        const Real dt = std::min(maxStableDt(uni, CFL), Real(SOD_T - tu));
        step2D(uni, dt, s);
        tu += dt;
    }
    double errUni = 0;
    for (int j = NG; j < NG + uni.ny; ++j)
        for (int i = NG; i < NG + uni.nx; ++i)
            errUni += std::fabs(double(toPrim(uni.at(i, j)).rho) -
                                exactRho(uni.xc(i))) *
                      uni.dx * uni.dy;
    errUni /= 0.25;

    std::printf("gate 2 — 3-level Sod (1/64 -> 1/256), %d steps: L1 %.4e "
                "| uniform 1/256 %.4e | ratio %.2f, drift %.2e, patches "
                "L1 %zu L2 %zu, nesting %s\n",
                steps, errMl, errUni, errMl / errUni, drift,
                amr.patchCount(1), amr.patchCount(2),
                nested ? "ok" : "VIOLATED");
    // Ratio gate is looser than the 2-level 1.4: with two transitions
    // the contact carries some memory of its transiently coarser past
    // and the smooth fan sees derefinement churn — ~1.3 even with
    // near-full fine coverage (Richardson tagging is the roadmap fix).
    return errMl < 1.6 * errUni && drift < 1e-5 && nested;
}

bool gate3_periodicConservation() {
    AmrConfig cfg;
    cfg.subcycle = true;
    cfg.maxLevels = 3;
    cfg.tagVelocity = Real(0.05);
    cfg.periodicX = cfg.periodicY = true;
    AmrML amr(64, 64, 0, 0, 1, 1, cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
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
    bool nested = true;
    for (int s = 0; s < 150; ++s) {
        const Real dt = amr.maxStableDtAll(CFL);
        amr.step(dt, t);
        t += dt;
        drift = std::max(drift, std::fabs(amr.totalMass() - m0) / m0);
        nested = nested && amr.checkNesting();
    }
    std::printf("gate 3 — 3-level periodic KH, 150 steps: mass drift "
                "%.3e (gate 1e-6), patches L1 %zu L2 %zu, nesting %s\n",
                drift, amr.patchCount(1), amr.patchCount(2),
                nested ? "ok" : "VIOLATED");
    return drift < 1e-6 && nested;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_twoLevelEquivalence() && ok;
    ok = gate2_threeLevelSod() && ok;
    ok = gate3_periodicConservation() && ok;
    if (!ok) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

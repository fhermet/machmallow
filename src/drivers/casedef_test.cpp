// Validation of the declarative case system (CaseDef) against the
// historical C++ presets and exact solutions.
// Gates:
//   1. declarative Sod (parsed from cases/sod.ini) run on AMR: L1 vs the
//      exact Riemann solution matches the preset-era value
//   2. DMR ghost fill: CaseDef's analytic/segmented/reflective sides
//      reproduce dmr::fillGhosts cell for cell — any differing ghost must
//      sit on the moving front line (fp tie-break zone), nowhere else
//   3. declarative KH initial condition matches the analytic formula

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/CaseDef.hpp"
#include "cases/Dmr.hpp"
#include "core/Boundary.hpp"
#include "core/Config.hpp"
#include "numerics/ExactRiemann.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace mm;

bool gate1_sodL1() {
    const Config cfg = Config::load("cases/sod.ini");
    const CaseDef cd = CaseDef::parse(cfg);

    AmrConfig acfg;
    acfg.tagThreshold = Real(0.03);
    Amr2 amr(128, 32, cd.x0, cd.y0, cd.lx, cd.ly, acfg);
    amr.fillPhysicalGhosts = [&](Grid& g, double t) {
        cd.fillGhosts(g, t);
    };
    amr.fillPatchPhysical = [&](Grid& g, double t, unsigned s) {
        cd.fillGhostSides(g, t, s);
    };
    amr.init([&](Real x, Real y) { return cd.state(x, y, 0); });

    double t = 0;
    while (t < 0.2) {
        const Real dt =
            std::min(amr.maxStableDtAll(Real(0.4)), Real(0.2 - t));
        amr.step(dt, t);
        t += dt;
    }
    constexpr exact::State L = {1.0, 0.0, 1.0}, R = {0.125, 0.0, 0.1};
    double err = 0;
    for (int j = 0; j < 32; ++j)
        for (int i = 0; i < 128; ++i)
            if (!amr.covered(i / 8, j / 8)) {
                const Grid& g = amr.coarse;
                err += std::fabs(
                           double(toPrim(g.at(NG + i, NG + j)).rho) -
                           exact::sample(L, R, (g.xc(NG + i) - 0.5) / 0.2)
                               .rho) *
                       g.dx * g.dy;
            }
    for (const auto& p : amr.patches)
        for (int j = NG; j < NG + p.grid.ny; ++j)
            for (int i = NG; i < NG + p.grid.nx; ++i)
                err += std::fabs(
                           double(toPrim(p.grid.at(i, j)).rho) -
                           exact::sample(L, R,
                                         (p.grid.xc(i) - 0.5) / 0.2)
                               .rho) *
                       p.grid.dx * p.grid.dy;
    err /= 0.25;
    std::printf("gate 1 — declarative Sod on AMR: L1 = %.4e (gate "
                "2.4e-3)\n",
                err);
    return err < 2.4e-3;
}

bool gate2_dmrGhosts() {
    const Config cfg = Config::load("cases/dmr.ini");
    const CaseDef cd = CaseDef::parse(cfg);

    const int nx = 128, ny = 32;
    Grid a(nx, ny, 0, 0, 4, 1), b(nx, ny, 0, 0, 4, 1);
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i)
            a.at(i, j) = b.at(i, j) =
                dmr::behindShock(a.xc(i), a.yc(j), 0) ? dmr::POST
                                                      : dmr::PRE;
    const double t = 0.07;
    dmr::fillGhosts(a, t);
    cd.fillGhosts(b, t);

    int diff = 0, offFront = 0;
    for (int j = 0; j < a.toty(); ++j)
        for (int i = 0; i < a.totx(); ++i) {
            if (j >= NG && j < NG + ny && i >= NG && i < NG + nx)
                continue; // ghosts only
            const Real* pa = &a.at(i, j).rho;
            const Real* pb = &b.at(i, j).rho;
            bool same = true;
            for (int k = 0; k < NVARS; ++k)
                same = same && pa[k] == pb[k];
            if (same) continue;
            ++diff;
            // distance to the moving front line x = xw + (y+20t)/sqrt(3)
            const double xs =
                1.0 / 6.0 + (double(a.yc(j)) + 20.0 * t) / std::sqrt(3.0);
            if (std::fabs(double(a.xc(i)) - xs) > 2 * double(a.dx))
                ++offFront;
        }
    std::printf("gate 2 — DMR ghosts CaseDef vs preset: %d differing "
                "cells, %d away from the front (gate 0)\n",
                diff, offFront);
    return offFront == 0;
}

bool gate3_khIc() {
    const Config cfg = Config::load("cases/kh.ini");
    const CaseDef cd = CaseDef::parse(cfg);

    double maxd = 0;
    for (int j = 0; j < 64; ++j)
        for (int i = 0; i < 64; ++i) {
            const Real x = Real((i + 0.5) / 64), y = Real((j + 0.5) / 64);
            const bool band = std::fabs(y - Real(0.5)) < Real(0.25);
            const Cons want =
                toCons({band ? Real(2) : Real(1),
                        band ? Real(0.25) : Real(-0.25),
                        Real(0.01 * std::sin(4 * M_PI * double(x))),
                        Real(2.5)});
            const Cons got = cd.state(x, y, 0);
            const Real* pw = &want.rho;
            const Real* pg = &got.rho;
            for (int k = 0; k < NVARS; ++k)
                maxd = std::max(maxd,
                                double(std::fabs(pw[k] - pg[k])));
        }
    std::printf("gate 3 — declarative KH IC vs analytic: max |diff| = "
                "%.3e (gate 1e-6)\n",
                maxd);
    return maxd < 1e-6;
}

bool gate4_rankineHugoniot() {
    // bubble.ini derives its post-shock state (Mach 1.5 into rho = 1.4,
    // p = 1 at rest); reference values computed by hand.
    const Config cfg = Config::load("cases/bubble.ini");
    const CaseDef cd = CaseDef::parse(cfg);
    Prim post{};
    Real speed = 0;
    for (const auto& s : cd.listStates())
        if (s.shockSpeed != 0) {
            post = s.w;
            speed = s.shockSpeed;
        }
    const double e =
        std::max({std::fabs(double(post.rho) - 2.6068966),
                  std::fabs(double(post.u) - 0.6944444),
                  std::fabs(double(post.p) - 2.4583333),
                  std::fabs(double(speed) - 1.5)});
    std::printf("gate 4 — Rankine-Hugoniot state (Ms=1.5): max |diff| = "
                "%.3e (gate 1e-5)\n",
                e);
    return e < 1e-5;
}

bool gate5_freeFall() {
    // Uniform gas under gravity, no gradients: only the split source
    // acts, so v(t) = g*t and E(t) = E0 + rho*v^2/2 to fp32 precision.
    AmrConfig cfg;
    cfg.tagThreshold = Real(1e30);
    cfg.gy = Real(-0.1);
    Amr2 amr(32, 32, 0, 0, 1, 1, cfg);
    amr.fillPhysicalGhosts = [](Grid& g, double) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
    };
    amr.init([](Real, Real) { return toCons({Real(1), 0, 0, Real(1)}); });

    const double E0 = double(amr.coarse.at(NG, NG).E);
    double t = 0;
    for (int s = 0; s < 50; ++s) {
        const Real dt = amr.maxStableDtAll(Real(0.4));
        amr.step(dt, t);
        t += dt;
    }
    const Cons q = amr.coarse.at(NG + 16, NG + 16);
    const double vWant = -0.1 * t;
    const double eWant = E0 + 0.5 * vWant * vWant; // rho = 1
    const double err = std::max(std::fabs(double(q.my) - vWant),
                                std::fabs(double(q.E) - eWant));
    std::printf("gate 5 — free fall under gravity, 50 steps: max |diff| "
                "= %.3e (gate 1e-5)\n",
                err);
    return err < 1e-5;
}

bool gate6_gravityGpuParity() {
    // RT-like stratified setup, CPU vs GPU lock-step with gravity on.
    MetalContext ctx;
    AmrConfig cfg;
    cfg.tagThreshold = Real(0.04);
    cfg.gy = Real(-0.1);
    cfg.periodicX = true;
    const auto ic = [](Real x, Real y) {
        Prim w{y > Real(0.75) ? Real(2) : Real(1), 0,
               Real(0.01 * std::sin(2 * M_PI * double(x) / 0.5) *
                    std::exp(-std::pow((double(y) - 0.75) / 0.05, 2))),
               Real(2.5)};
        w.p += w.rho * Real(-0.1) * (y - Real(0.75));
        return toCons(w);
    };
    const auto bc = [](auto& g, double) {
        fillPeriodicX(g);
        fillReflectiveBottom(g);
        fillReflectiveTop(g);
    };
    Amr2 cpu(32, 96, 0, 0, Real(0.5), Real(1.5), cfg);
    AmrGpu gpu(ctx, 32, 96, 0, 0, Real(0.5), Real(1.5), cfg);
    cpu.fillPhysicalGhosts = bc;
    gpu.fillPhysicalGhosts = bc;
    cpu.init(ic);
    gpu.init(ic);

    double t = 0;
    for (int s = 0; s < 30; ++s) {
        const Real dt = cpu.maxStableDtAll(Real(0.4));
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
    }
    const GridRef g = gpu.coarseRef();
    double maxRel = 0;
    for (int j = NG; j < NG + 96; ++j)
        for (int i = NG; i < NG + 32; ++i) {
            const Real* pa = &cpu.coarse.at(i, j).rho;
            const Real* pb = &g.at(i, j).rho;
            for (int k = 0; k < NVARS; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    // Near hydrostatic balance my ~ 0 everywhere, so the relative
    // metric divides fp32-level absolute differences (~3e-7) by the
    // 1e-3 floor — 1e-3 here is the roundoff regime, not a bug.
    std::printf("gate 6 — gravity CPU/GPU lock-step, 30 steps: max rel "
                "diff = %.3e (gate 1e-3), patches %zu | %zu\n",
                maxRel, cpu.patches.size(), gpu.patches.size());
    return maxRel < 1e-3 && cpu.patches.size() == gpu.patches.size();
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_sodL1() && ok;
    ok = gate2_dmrGhosts() && ok;
    ok = gate3_khIc() && ok;
    ok = gate4_rankineHugoniot() && ok;
    ok = gate5_freeFall() && ok;
    ok = gate6_gravityGpuParity() && ok;
    if (!ok) {
        std::fprintf(stderr, "FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

// Gate for the DECLARATIVE solid mask: we load cases/shock_wall.ini (which
// places an immersed wall via a [solid] section), run it through the same
// path as the runner (CaseDef -> Amr2, mask threaded into the MUSCL step
// of the base grid), and check that the wall pressure after shock
// reflection reaches the exact 1D value. This exercises the whole chain:
// parsing [solid] -> solidAt() -> mask-aware step2D.

#include "amr/Amr2.hpp"
#include "cases/CaseDef.hpp"
#include "core/Config.hpp"
#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace mm;

int main(int argc, char** argv) {
    const std::string path =
        argc > 1 ? argv[1] : std::string("cases/shock_wall.ini");

    const Config cfg = Config::load(path);
    const CaseDef cd = CaseDef::parse(cfg);
    if (!cd.hasSolids()) {
        std::printf("FAIL: %s declares no [solid] region\n",
                    path.c_str());
        return 1;
    }

    // AMR disabled (the runner does the same when a solid is present):
    // base grid only, mask threaded into the MUSCL step.
    AmrConfig acfg;
    acfg.blockC = cfg.getInt("amr.block", 8);
    acfg.maxLevels = 2;
    acfg.tagThreshold = Real(1e30); // no refinement
    acfg.periodicX = cd.periodicX;
    acfg.periodicY = cd.periodicY;

    Amr2 amr(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly, acfg);
    amr.fillPhysicalGhosts = [&cd](Grid& g, double t) {
        cd.fillGhosts(g, t);
    };
    amr.fillPatchPhysical = [&cd](Grid& g, double t, unsigned s) {
        cd.fillGhostSides(g, t, s);
    };
    amr.solidAt = [&cd](Real x, Real y) { return cd.solidAt(x, y); };
    amr.init([&](Real x, Real y) { return cd.state(x, y, 0); });

    const Real cfl = Real(cfg.getReal("cfl", 0.4));
    const double tEnd = cfg.getReal("t_end", 0.32);
    double t = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt = std::min(amr.maxStableDtAll(cfl), Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
    }

    // exact wall pressure (1D reflection, Ms = 2, ideal gas gamma)
    const double G = double(GAMMA), Ms = 2.0;
    const double xi = 1.0 + 2.0 * G / (G + 1.0) * (Ms * Ms - 1.0);
    const double p1 = xi;                              // p0 = 1
    const double p2 = p1 * ((3 * G - 1) * xi - (G - 1)) /
                      ((G - 1) * xi + (G + 1));

    // fluid cell adjacent to the solid wall (face at x = 0.7)
    const GridRef c = amr.coarseRef();
    int iw = NG;
    while (iw < NG + c.nx &&
           !cd.solidAt(c.x0 + Real(iw - NG + 0.5) * c.dx, c.y0 + c.dy))
        ++iw;
    const int jm = NG + c.ny / 2;
    const Prim w = toPrim(c.at(iw - 1, jm));
    const double pErr = std::fabs(double(w.p) - p2) / p2;

    std::printf("Declarative immersed wall (%s)\n", path.c_str());
    std::printf("  wall : p=%.3f vs exact p_r=%.3f  (err %.2f%%, gate 5%%)"
                "\n", double(w.p), p2, 100 * pErr);
    std::printf("  non-penetration : |u|=%.3e  (gate 1e-2)\n",
                std::fabs(double(w.u)));

    const bool ok = pErr < 0.05 && std::fabs(double(w.u)) < 1e-2;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

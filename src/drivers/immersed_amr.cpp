// Corps immergé + AMR : on rejoue la réflexion de choc de cases/shock_wall.ini
// (paroi solide alignée) avec le RAFFINEMENT ACTIVÉ (Amr2, 2 niveaux), et on
// vérifie que la chaîne AMR masque-aware (pas de patch, restriction, reflux,
// prolongation, tagging de bord) préserve le résultat :
//   1. la pression de paroi reste la valeur 1D exacte (15.0 p0, Ms=2) ;
//   2. elle coïncide avec le run grille-de-base seule (cohérence AMR) ;
//   3. le raffinement s'est bien produit (patches > 0, dont le bord du corps).

#include "amr/Amr2.hpp"
#include "amr/AmrML.hpp"
#include "cases/CaseDef.hpp"
#include "core/Config.hpp"
#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using namespace mm;

namespace {

double runShockWall(const CaseDef& cd, double tEnd, bool refine,
                    bool subcycle, std::size_t& maxPatches) {
    AmrConfig a;
    a.blockC = 8;
    a.maxLevels = 2;
    a.periodicX = cd.periodicX;
    a.periodicY = cd.periodicY;
    a.regridEvery = 2;
    a.subcycle = subcycle;
    a.tagThreshold = refine ? Real(0.05) : Real(1e30); // 1e30 = base seule
    Amr2 amr(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly, a);
    amr.fillPhysicalGhosts = [&cd](Grid& g, double t) { cd.fillGhosts(g, t); };
    amr.fillPatchPhysical = [&cd](Grid& g, double t, unsigned s) {
        cd.fillGhostSides(g, t, s);
    };
    amr.solidAt = [&cd](Real x, Real y) { return cd.solidAt(x, y); };
    amr.init([&](Real x, Real y) { return cd.state(x, y, 0); });

    double t = 0;
    maxPatches = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt = std::min(amr.maxStableDtAll(Real(0.4)),
                                 Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        maxPatches = std::max(maxPatches, amr.patches.size());
    }

    // pression dans la cellule fluide grossière adjacente à la paroi
    const GridRef c = amr.coarseRef();
    int iw = NG;
    while (iw < NG + c.nx &&
           !cd.solidAt(c.x0 + Real(iw - NG + 0.5) * c.dx, c.y0 + c.dy))
        ++iw;
    return double(toPrim(c.at(iw - 1, NG + c.ny / 2)).p);
}

// Multi-level (AmrML, profondeur `levels`) — même cas, classe N-niveaux.
double runShockWallML(const CaseDef& cd, double tEnd, int levels,
                      bool subcycle, std::size_t& maxPatches) {
    AmrConfig a;
    a.blockC = 8;
    a.maxLevels = levels;
    a.periodicX = cd.periodicX;
    a.periodicY = cd.periodicY;
    a.regridEvery = 2;
    a.subcycle = subcycle;
    a.tagThreshold = Real(0.05);
    AmrML amr(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly, a);
    amr.fillPhysicalGhosts = [&cd](Grid& g, double t) { cd.fillGhosts(g, t); };
    amr.fillPatchPhysical = [&cd](Grid& g, double t, unsigned s) {
        cd.fillGhostSides(g, t, s);
    };
    amr.solidAt = [&cd](Real x, Real y) { return cd.solidAt(x, y); };
    amr.init([&](Real x, Real y) { return cd.state(x, y, 0); });

    double t = 0;
    maxPatches = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt = std::min(amr.maxStableDtAll(Real(0.4)),
                                 Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        std::size_t np = 0;
        for (int l = 1; l < levels; ++l) np += amr.level(l).patches.size();
        maxPatches = std::max(maxPatches, np);
    }
    const Grid& b = amr.base;
    int iw = NG;
    while (iw < NG + b.nx &&
           !cd.solidAt(b.xc(iw), b.yc(NG)))
        ++iw;
    return double(toPrim(b.at(iw - 1, NG + b.ny / 2)).p);
}

} // namespace

int main(int argc, char** argv) {
    const std::string path =
        argc > 1 ? argv[1] : std::string("cases/shock_wall.ini");
    const Config cfg = Config::load(path);
    const CaseDef cd = CaseDef::parse(cfg);
    const double tEnd = cfg.getReal("t_end", 0.32);

    const double G = double(GAMMA), Ms = 2.0;
    const double xi = 1.0 + 2.0 * G / (G + 1.0) * (Ms * Ms - 1.0);
    const double pExact = xi * ((3 * G - 1) * xi - (G - 1)) /
                          ((G - 1) * xi + (G + 1));

    std::size_t npBase = 0, npAmr = 0, npSub = 0, npML = 0;
    const double pBase = runShockWall(cd, tEnd, false, false, npBase);
    const double pAmr = runShockWall(cd, tEnd, true, false, npAmr);
    const double pSub = runShockWall(cd, tEnd, true, true, npSub);
    const double pML = runShockWallML(cd, tEnd, 3, true, npML); // 3 niveaux

    const double eAmr = std::fabs(pAmr - pExact) / pExact;
    const double eSub = std::fabs(pSub - pExact) / pExact;
    const double eML = std::fabs(pML - pExact) / pExact;
    const double eBase = std::fabs(pAmr - pBase) / pBase;

    std::printf("Paroi immergée + AMR (%s)\n", path.c_str());
    std::printf("  base seule        : p=%.3f (%zu patches)\n", pBase, npBase);
    std::printf("  Amr2 2 niv.       : p=%.3f (%zu patches) err %.2f%%\n",
                pAmr, npAmr, 100 * eAmr);
    std::printf("  Amr2 2 niv. subcyc: p=%.3f (%zu patches) err %.2f%%\n",
                pSub, npSub, 100 * eSub);
    std::printf("  AmrML 3 niv. subcyc: p=%.3f (%zu patches) err %.2f%%\n",
                pML, npML, 100 * eML);
    std::printf("  exact %.3f | AMR vs base %.2f%% (gate 2%%) | "
                "exact gate 5%%\n", pExact, 100 * eBase);

    const bool ok = eAmr < 0.05 && eSub < 0.05 && eML < 0.05 &&
                    eBase < 0.02 && npAmr > 0 && npSub > 0 && npML > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

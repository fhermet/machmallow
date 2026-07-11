// Validation chiffrée d'un corps immergé : choc oblique sur un DIÈDRE.
// Un écoulement supersonique Mach M est dévié de θ par une rampe solide
// (posée par le masque immergé) ; un choc oblique attaché en part, dont
// l'angle β obéit à la relation EXACTE θ-β-M (Anderson, gaz parfait,
// non visqueux) :
//     tan θ = 2 cot β (M² sin²β − 1) / (M²(γ + cos 2β) + 2).
// On mesure β sur le champ (position du choc à plusieurs hauteurs) et on
// le compare à la racine faible exacte. C'est la démo cylindre/marche
// transformée en vérification quantitative de la paroi immergée oblique.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Forces.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr double DEG = M_PI / 180.0;

// Déflexion d'un choc oblique d'angle β (rad) à Mach M (racine exacte).
double deflection(double beta, double M) {
    const double g = double(GAMMA);
    const double s = std::sin(beta);
    return std::atan(2.0 / std::tan(beta) * (M * M * s * s - 1.0) /
                     (M * M * (g + std::cos(2 * beta)) + 2.0));
}

// Racine FAIBLE β de θ-β-M (le plus petit β > angle de Mach donnant θ).
double obliqueBetaWeak(double thetaRad, double M) {
    const double mu = std::asin(1.0 / M); // angle de Mach (déflexion 0)
    double lo = mu, hi = mu;
    // monte jusqu'à dépasser θ (on reste sur la branche faible montante)
    for (double b = mu + 1e-4; b < M_PI / 2; b += 1e-4) {
        if (deflection(b, M) >= thetaRad) { hi = b; lo = b - 1e-4; break; }
    }
    for (int it = 0; it < 60; ++it) { // bisection
        const double mid = 0.5 * (lo + hi);
        if (deflection(mid, M) < thetaRad) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Symmetric body (cylinder in axial flow) → lift must vanish exactly.
// Validates the y-face bookkeeping / sign of wallForce.
WallForce cylinderForce() {
    const int nx = 160, ny = 128;
    const double Lx = 1.25, cx = 0.45, cy = 0.5, rad = 0.15;
    Grid g(nx, ny, 0, 0, Lx, Real(ny) * (Lx / nx));
    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const double dx = double(g.xc(i)) - cx, dy = double(g.yc(j)) - cy;
            solid[g.idx(i, j)] = dx * dx + dy * dy < rad * rad ? 1 : 0;
        }
    const Cons fs = toCons(Prim{Real(1.4), Real(2), 0, Real(1)}); // Mach 2
    for (std::size_t k = 0; k < g.q.size(); ++k) g.q[k] = fs;
    Scratch2D s;
    double t = 0;
    const double tEnd = 0.9;
    while (t < tEnd * (1 - 1e-12)) {
        Real dt = std::min(maxStableDt(g, Real(0.4), 0), Real(tEnd - t));
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < NG; ++i) g.at(i, j) = fs;
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        step2D(g, dt, s, 0, 0, 0, solid.data());
        t += dt;
    }
    return wallForce(g, solid.data());
}

} // namespace

int main(int argc, char** argv) {
    const double M = 2.5, thetaDeg = 15.0;
    const double theta = thetaDeg * DEG, tanT = std::tan(theta);
    const double rho0 = 1.4, p0 = 1.0;       // c = sqrt(1.4*1/1.4) = 1
    const double u0 = M;                       // -> u = M

    const double Lx = 1.6, Ly = 0.9, x0 = 0.4; // pointe du dièdre
    // optional resolution knob (`immersed_wedge <nx>`, default 400 -> CI/gate
    // unchanged) to show the staircase bias in beta shrink with refinement.
    const int nx = (argc > 1) ? std::atoi(argv[1]) : 400;
    const int ny = int(std::lround(nx * Ly / Lx)); // square cells
    Grid g(nx, ny, 0, 0, Lx, Real(ny) * (Lx / nx));

    // masque : solide sous la rampe y < tanθ (x - x0), pour x > x0
    const auto isSolid = [&](Real x, Real y) {
        return double(y) < tanT * (double(x) - x0) ? 1 : 0;
    };
    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            solid[g.idx(i, j)] = isSolid(g.xc(i), g.yc(j));

    const Cons fs = toCons(Prim{Real(rho0), Real(u0), 0, Real(p0)});
    for (std::size_t k = 0; k < g.q.size(); ++k) g.q[k] = fs;

    // BC : entrée Mach M à gauche, transmissif droite/haut, plancher
    // réfléchissant en bas (sous la rampe : couvert par le masque).
    const auto fillGhosts = [&](Grid& gg) {
        for (int j = 0; j < gg.toty(); ++j)
            for (int i = 0; i < NG; ++i) gg.at(i, j) = fs; // inflow
        fillTransmissiveRight(gg);
        fillReflectiveBottom(gg);
        fillTransmissiveTop(gg);
    };

    Scratch2D s;
    double t = 0;
    const double tEnd = 1.0;
    while (t < tEnd * (1 - 1e-12)) {
        Real dt = std::min(maxStableDt(g, Real(0.4), 0), Real(tEnd - t));
        fillGhosts(g);
        step2D(g, dt, s, 0, 0, 0, solid.data());
        t += dt;
    }

    // mesure de β : à plusieurs hauteurs on localise le choc (saut de
    // densité max), puis on AJUSTE x_choc(y) par moindres carrés. La pente
    // dx/dy = 1/tan β donne β directement — indépendant de l'origine (la
    // pointe en escalier est légèrement émoussée, donc on ne suppose pas
    // que le choc passe par (x0, 0)).
    std::vector<double> ys, xs;
    for (int k = 0; k < 9; ++k) {
        const double y = 0.15 + 0.04 * k; // 0.15 .. 0.47
        const int j = NG + int(y / double(g.dy));
        double best = 0;
        int iShock = -1;
        for (int i = NG + int(x0 / double(g.dx)); i < NG + nx - 1; ++i) {
            const double d = double(g.at(i + 1, j).rho) -
                             double(g.at(i, j).rho);
            if (d > best) { best = d; iShock = i; }
        }
        if (iShock >= 0) { ys.push_back(y); xs.push_back(double(g.xc(iShock))); }
    }
    // moindres carrés x = a + m y
    const int n = int(ys.size());
    double sy = 0, sx = 0, syy = 0, sxy = 0;
    for (int k = 0; k < n; ++k) {
        sy += ys[k]; sx += xs[k]; syy += ys[k] * ys[k];
        sxy += xs[k] * ys[k];
    }
    const double m = (n * sxy - sx * sy) / (n * syy - sy * sy);
    const double betaMeas = std::atan(1.0 / m) / DEG;
    const double betaR = obliqueBetaWeak(theta, M);
    const double betaExact = betaR / DEG;
    const double errBeta = std::fabs(betaMeas - betaExact);

    // Pression de paroi exacte derrière le choc oblique (Rankine-Hugoniot
    // sur la composante normale M1n = M sin β) — la rampe est à p2.
    const double M1n = M * std::sin(betaR);
    const double p2 = p0 * (1.0 + 2.0 * double(GAMMA) / (double(GAMMA) + 1.0) *
                                      (M1n * M1n - 1.0));

    // Pression de paroi moyenne (intégrande de la traînée) sur une fenêtre
    // propre de la rampe (loin de la pointe et de la réflexion en haut).
    double psum = 0;
    int np = 0;
    const auto solAt = [&](int i, int j) { return solid[g.idx(i, j)] != 0; };
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const double x = double(g.xc(i));
            if (solAt(i, j) || x < x0 + 0.3 || x > x0 + 0.9) continue;
            if (solAt(i, j - 1) || solAt(i + 1, j)) { // au contact de la rampe
                psum += double(toPrim(g.at(i, j)).p);
                ++np;
            }
        }
    const double pWall = np ? psum / np : 0;
    const double errP = std::fabs(pWall - p2) / p2;

    const WallForce F = wallForce(g, solid.data());

    std::printf("Choc oblique sur dièdre immergé (M=%.1f, theta=%.0f deg)\n",
                M, thetaDeg);
    std::printf("  beta : mesuré %.2f deg vs %.2f exact (theta-beta-M) | "
                "écart %.2f deg (gate 2)\n", betaMeas, betaExact, errBeta);
    std::printf("  pression de paroi : %.3f vs %.3f exact (choc oblique) | "
                "écart %.1f%% (gate 6%%)\n", pWall, p2, 100 * errP);
    std::printf("  force de paroi (∫p) : traînée Fx=%.4f, portance Fy=%.4f\n",
                F.fx, F.fy);

    // Cylindre symétrique : portance nulle (vérifie le signe / bilan ∫p).
    const WallForce C = cylinderForce();
    const double liftRatio = std::fabs(C.fy) / std::fabs(C.fx);
    std::printf("  symétrie (cylindre Mach 2) : Fx=%.3f Fy=%.3f -> "
                "|Fy/Fx|=%.3f (gate 0.03)\n", C.fx, C.fy, liftRatio);

    const bool ok = errBeta < 2.0 && errP < 0.06 && liftRatio < 0.03 &&
                    C.fx > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

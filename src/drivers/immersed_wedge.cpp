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

} // namespace

int main() {
    const double M = 2.5, thetaDeg = 15.0;
    const double theta = thetaDeg * DEG, tanT = std::tan(theta);
    const double rho0 = 1.4, p0 = 1.0;       // c = sqrt(1.4*1/1.4) = 1
    const double u0 = M;                       // -> u = M

    const int nx = 400, ny = 225;
    const double Lx = 1.6, Ly = 0.9, x0 = 0.4; // pointe du dièdre
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
    const double betaExact = obliqueBetaWeak(theta, M) / DEG;
    const double err = std::fabs(betaMeas - betaExact);

    std::printf("Choc oblique sur dièdre immergé (M=%.1f, theta=%.0f deg)\n",
                M, thetaDeg);
    std::printf("  beta mesuré = %.2f deg | exact (theta-beta-M) = %.2f deg"
                " | écart %.2f deg (gate 2 deg)\n",
                betaMeas, betaExact, err);

    const bool ok = err < 2.0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

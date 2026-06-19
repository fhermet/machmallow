// Corps solide immergé dans la grille cartésienne — validation du
// traitement de paroi (masque solide + flux de paroi réfléchissante dans
// step2D).
//
// Cas : un choc plan de Mach Ms se propage en +x dans un gaz au repos et
// se réfléchit sur une PAROI IMMERGÉE (bloc solide, face alignée à la
// grille). La pression de paroi après réflexion a une valeur EXACTE
// (réflexion 1D de choc) :
//     p_r/p_i = ((3γ-1)ξ - (γ-1)) / ((γ-1)ξ + (γ+1)),   ξ = p_i/p_0
// (limite forte (3γ-1)/(γ-1) = 8 pour γ=1.4). La face étant alignée, il
// n'y a pas d'erreur d'escalier : c'est une vérification exacte de la
// paroi immergée. On vérifie aussi la non-pénétration (u ≈ 0 à la paroi).
//
// On teste DEUX régimes :
//   Ms=2 : gaz post-choc SUBSONIQUE vers la paroi (M1≈0.96)
//   Ms=3 : gaz post-choc SUPERSONIQUE vers la paroi (M1≈1.36)
// Le cas supersonique verrouille spécifiquement le flux de paroi exact :
// la paroi miroir + HLLC FUIT en supersonique (son estimation PVRS garde
// SL > 0 et décentre tout le flux entrant) — d'où `wallPressure`.

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

// Réflexion d'un choc Ms sur une paroi immergée ; renvoie l'erreur
// relative de pression de paroi vs la valeur 1D exacte (et |u|/u_i).
struct Result { double pErr, uRel, p, pExact, M1; };

Result runReflection(double Ms, double tEnd) {
    const double G = double(GAMMA);
    const double rho0 = 1.0, p0 = 1.0;
    const double c0 = std::sqrt(G * p0 / rho0);

    const double xi = 1.0 + 2.0 * G / (G + 1.0) * (Ms * Ms - 1.0);
    const double p1 = xi * p0;
    const double rho1 = rho0 * (G + 1.0) * Ms * Ms /
                        ((G - 1.0) * Ms * Ms + 2.0);
    const double u1 = 2.0 / (G + 1.0) * (Ms - 1.0 / Ms) * c0;
    const double c1 = std::sqrt(G * p1 / rho1);
    const double p2 = p1 * ((3 * G - 1) * xi - (G - 1)) /
                      ((G - 1) * xi + (G + 1));

    const int nx = 400, ny = 4;
    const double L = 1.0, xWall = 0.7, xShock0 = 0.2;
    Grid g(nx, ny, 0, 0, L, Real(ny) * (L / nx));

    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            if (double(g.xc(i)) >= xWall) solid[g.idx(i, j)] = 1;

    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const bool behind = double(g.xc(i)) < xShock0;
            const Prim w = behind ? Prim{Real(rho1), Real(u1), 0, Real(p1)}
                                  : Prim{Real(rho0), 0, 0, Real(p0)};
            g.q[g.idx(i, j)] = toCons(w);
        }

    Scratch2D s;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt = std::min(maxStableDt(g, Real(0.4), 0),
                                 Real(tEnd - t));
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillPeriodicY(g);
        step2D(g, dt, s, 0, 0, 0, solid.data());
        t += dt;
    }

    int iw = NG;
    while (iw < NG + nx && double(g.xc(iw)) < xWall) ++iw;
    const int jm = NG + ny / 2;
    const Prim w = toPrim(g.at(iw - 1, jm));
    return {std::fabs(double(w.p) - p2) / p2, std::fabs(double(w.u)) / u1,
            double(w.p), p2, u1 / c1};
}

} // namespace

int main() {
    bool ok = true;
    // (Ms, tEnd, gate) — Ms=3 atteint la paroi plus tôt, on mesure plus tôt.
    const struct { double Ms, tEnd, tol; } cases[] = {
        {2.0, 0.32, 0.04}, {3.0, 0.22, 0.05}};
    for (const auto& c : cases) {
        const Result r = runReflection(c.Ms, c.tEnd);
        const bool pass = r.pErr < c.tol && r.uRel < 0.05;
        ok = ok && pass;
        std::printf("Ms=%.1f (post-choc M1=%.2f, %s vers la paroi) : "
                    "p=%.3f vs %.3f exact  err %.2f%% (gate %.0f%%), "
                    "|u|/u_i=%.3f  %s\n",
                    c.Ms, r.M1, r.M1 < 1 ? "subsonique" : "SUPERSONIQUE",
                    r.p, r.pExact, 100 * r.pErr, 100 * c.tol, r.uRel,
                    pass ? "PASS" : "FAIL");
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

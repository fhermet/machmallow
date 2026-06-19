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

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mm;

int main() {
    const double G = double(GAMMA);
    const double Ms = 2.0;                 // Mach du choc incident
    const double rho0 = 1.0, p0 = 1.0;     // gaz au repos en amont
    const double c0 = std::sqrt(G * p0 / rho0);

    // état post-choc (Rankine-Hugoniot, choc se déplaçant en +x)
    const double xi = 1.0 + 2.0 * G / (G + 1.0) * (Ms * Ms - 1.0); // p1/p0
    const double p1 = xi * p0;
    const double rho1 = rho0 * (G + 1.0) * Ms * Ms /
                        ((G - 1.0) * Ms * Ms + 2.0);
    const double u1 = 2.0 / (G + 1.0) * (Ms - 1.0 / Ms) * c0;
    // pression de paroi exacte après réflexion
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
    const double tEnd = 0.32; // choc atteint la paroi (~0.21) puis réflexion
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt = std::min(maxStableDt(g, Real(0.4), 0),
                                 Real(tEnd - t));
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillPeriodicY(g);
        step2D(g, dt, s, 0, 0, 0, solid.data());
        t += dt;
    }

    // cellule fluide adjacente à la paroi
    int iw = NG;
    while (iw < NG + nx && double(g.xc(iw)) < xWall) ++iw;
    const int jm = NG + ny / 2;
    const Prim w = toPrim(g.at(iw - 1, jm));
    const double pErr = std::fabs(double(w.p) - p2) / p2;
    const double uRel = std::fabs(double(w.u)) / u1;

    std::printf("Réflexion de choc sur paroi immergée (Ms=%.1f, gamma=%.1f)\n",
                Ms, G);
    std::printf("  post-choc : p_i=%.3f, u_i=%.3f\n", p1, u1);
    std::printf("  paroi : p=%.3f vs exact p_r=%.3f  (err %.2f%%, gate 4%%)\n",
                double(w.p), p2, 100 * pErr);
    std::printf("  non-pénétration : |u|/u_i = %.3f  (gate 5%%)\n", uRel);

    const bool ok = pErr < 0.04 && uRel < 0.05;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

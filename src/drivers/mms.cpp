// MMS — verification par solutions manufacturees (Method of Manufactured
// Solutions). L'etude `convergence` mesure deja l'ordre EULER lisse ; ici
// on verifie l'opérateur NAVIER-STOKES COMPLET (Euler + flux visqueux).
//
// Principe : on choisit une solution lisse periodique STATIONNAIRE
// (rho, u, v, p sinusoidaux), on calcule analytiquement le terme source
//   S = div(F_Euler) - div(F_visqueux)
// qui en fait une solution exacte des equations forcees A TOUT INSTANT.
// On part de cette solution, on avance jusqu'a T = 0.5 avec le source
// passe au schema (step2D / stepWeno2D : il entre dans le predicteur de
// Hancock / dans chaque etage RK), et on mesure l'erreur L1 a T : c'est
// l'erreur de troncature accumulee -> O(h^p). Pas besoin d'etat
// stationnaire (et en inviscide l'etat stationnaire force n'est pas
// attracteur : le calcul long diverge).
//
// Le source est calcule EN DOUBLE par differences finies d'ordre 4
// (h = 1e-3) sur les champs analytiques : son erreur (~1e-12) est tres
// inferieure a celle du schema teste, il est donc "exact".
//
// Gates : ordre observe (paire la plus fine) >= 1.8 (conception 2) en visqueux
// (MUSCL, WENO5, MUSCL + gravite) ET en inviscide MUSCL ; et erreur
// independante de la CFL en inviscide MUSCL (|e(0.4)/e(0.2) - 1| < 5 %) :
// un source ajoute APRES le pas (hors predicteur) laisse une erreur O(dt)
// -> ordre 1 et erreur proportionnelle a la CFL. En visqueux le flux
// visqueux n'est pas encore dans le predicteur : la dependance a la CFL
// y est affichee (informatif), pas encore une porte.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"
#include "solver/Weno2D.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr double PI2 = 2.0 * M_PI;
constexpr double G = 1.4;        // GAMMA en double
constexpr double PR = 0.72;      // PRANDTL en double
constexpr double HFD = 1e-3;     // pas des differences finies du source

using Vec4 = std::array<double, 4>; // {rho, rho u, rho v, E} ou un flux

struct Pm { double rho, u, v, p; };

// Solution manufacturee : lisse, periodique sur [0,1]^2, stationnaire.
// Ecoulement moyen non nul (l'advection est exercee), rho et p > 0.
Pm mfg(double x, double y) {
    return {1.00 + 0.20 * std::sin(PI2 * (x + y)),
            0.30 + 0.10 * std::sin(PI2 * x) * std::cos(PI2 * y),
            0.20 + 0.10 * std::cos(PI2 * x) * std::sin(PI2 * y),
            1.00 + 0.15 * std::cos(PI2 * (x + y))};
}

Vec4 toConsD(const Pm& w) {
    const double E = w.p / (G - 1) + 0.5 * w.rho * (w.u * w.u + w.v * w.v);
    return {w.rho, w.rho * w.u, w.rho * w.v, E};
}

Vec4 eulerFx(const Pm& w) {
    const double E = w.p / (G - 1) + 0.5 * w.rho * (w.u * w.u + w.v * w.v);
    return {w.rho * w.u, w.rho * w.u * w.u + w.p, w.rho * w.u * w.v,
            (E + w.p) * w.u};
}
Vec4 eulerFy(const Pm& w) {
    const double E = w.p / (G - 1) + 0.5 * w.rho * (w.u * w.u + w.v * w.v);
    return {w.rho * w.v, w.rho * w.u * w.v, w.rho * w.v * w.v + w.p,
            (E + w.p) * w.v};
}

// derivee centrale d'ordre 4 d'une fonction scalaire t -> f(t) en t=0
template <class F>
double d4(F f) {
    return (f(-2 * HFD) - 8 * f(-HFD) + 8 * f(HFD) - f(2 * HFD)) / (12 * HFD);
}

// flux TOTAL (Euler - visqueux) en x, puis en y, au point (x, y)
Vec4 totalFx(double x, double y, double mu) {
    Pm w = mfg(x, y);
    Vec4 F = eulerFx(w);
    if (mu > 0) {
        const double kT = mu * G / ((G - 1) * PR);
        const double ux = d4([&](double s) { return mfg(x + s, y).u; });
        const double vx = d4([&](double s) { return mfg(x + s, y).v; });
        const double uy = d4([&](double s) { return mfg(x, y + s).u; });
        const double vy = d4([&](double s) { return mfg(x, y + s).v; });
        const double Tx =
            d4([&](double s) { Pm q = mfg(x + s, y); return q.p / q.rho; });
        const double txx = mu * (4.0 / 3 * ux - 2.0 / 3 * vy);
        const double txy = mu * (uy + vx);
        F[1] -= txx;
        F[2] -= txy;
        F[3] -= w.u * txx + w.v * txy + kT * Tx;
    }
    return F;
}
Vec4 totalFy(double x, double y, double mu) {
    Pm w = mfg(x, y);
    Vec4 F = eulerFy(w);
    if (mu > 0) {
        const double kT = mu * G / ((G - 1) * PR);
        const double ux = d4([&](double s) { return mfg(x + s, y).u; });
        const double vx = d4([&](double s) { return mfg(x + s, y).v; });
        const double uy = d4([&](double s) { return mfg(x, y + s).u; });
        const double vy = d4([&](double s) { return mfg(x, y + s).v; });
        const double Ty =
            d4([&](double s) { Pm q = mfg(x, y + s); return q.p / q.rho; });
        const double txy = mu * (uy + vx);
        const double tyy = mu * (4.0 / 3 * vy - 2.0 / 3 * ux);
        F[1] -= txy;
        F[2] -= tyy;
        F[3] -= w.u * txy + w.v * tyy + kT * Ty;
    }
    return F;
}

// terme source manufacture : S = d/dx F_x + d/dy F_y (par composante)
Cons source(double x, double y, double mu, double gx, double gy) {
    Vec4 S{};
    for (int c = 0; c < 4; ++c) {
        const double dFx = d4([&](double s) { return totalFx(x + s, y, mu)[c]; });
        const double dFy = d4([&](double s) { return totalFy(x, y + s, mu)[c]; });
        S[c] = dFx + dFy;
    }
    if (gx != 0 || gy != 0) { // le solveur applique la gravite : on la retire
        const Pm w = mfg(x, y);
        S[1] -= w.rho * gx;
        S[2] -= w.rho * gy;
        S[3] -= w.rho * (w.u * gx + w.v * gy);
    }
    return {Real(S[0]), Real(S[1]), Real(S[2]), Real(S[3])};
}

// Erreur L1 (densite) a t = tEnd, partant de la solution manufacturee,
// le source S passe au schema. Le source couvre aussi les fantomes : le
// predicteur de Hancock lit l'anneau de fantomes.
double solve(int N, double mu, double tEnd, bool weno, double gx, double gy,
             Real cfl) {
    const auto periodic = [](Grid& gg) {
        fillPeriodicX(gg);
        fillPeriodicY(gg);
    };
    Grid g(N, N, 0, 0, 1, 1);
    std::vector<Cons> S(g.q.size());
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const std::size_t id = g.idx(i, j);
            const Pm w = mfg(g.xc(i), g.yc(j));
            g.q[id] = toCons({Real(w.rho), Real(w.u), Real(w.v), Real(w.p)});
            S[id] = source(g.xc(i), g.yc(j), mu, gx, gy);
        }
    Scratch2D s;
    ScratchW sw;
    double t = 0;
    while (t < tEnd * (1 - 1e-12)) {
        const Real dt =
            std::min(maxStableDt(g, cfl, Real(mu)), Real(tEnd - t));
        if (weno) {
            stepWeno2D(g, dt, sw, periodic, Real(mu), S.data());
        } else {
            periodic(g);
            step2D(g, dt, s, Real(mu), Real(gx), Real(gy), nullptr,
                   S.data());
        }
        t += dt;
    }
    double e = 0;
    for (int j = NG; j < NG + N; ++j)
        for (int i = NG; i < NG + N; ++i)
            e += std::fabs(double(g.at(i, j).rho) - mfg(g.xc(i), g.yc(j)).rho);
    return e / (double(N) * N);
}

constexpr double T_END = 0.5;

// Ordre observe = celui de la paire de maillages la plus fine (regime
// asymptotique ; N=16 est pre-asymptotique : en double, MUSCL visqueux
// donne 1.61 -> 1.81 -> 1.92). La pente LSQ est affichee pour info.
// WENO5 s'arrete a N=64 : a N=128 son increment par pas dt*(L+S) passe
// sous l'ulp float32 et l'etat stagne sur la condition initiale (mesure :
// L1 6.9e-7 en float contre 1.64e-5 en double).
double orderStudy(double mu, bool weno, double gx = 0, double gy = 0) {
    const std::vector<int> Ns =
        weno ? std::vector<int>{16, 32, 64} : std::vector<int>{16, 32, 64, 128};
    std::vector<double> errs;
    std::printf("  %s, mu = %.3g%s :\n", weno ? "WENO5" : "MUSCL", mu,
                (gx != 0 || gy != 0) ? ", gravite" : "");
    char lab[40];
    std::snprintf(lab, sizeof lab, "%s %s%s", mu > 0 ? "viscous" : "inviscid",
                  weno ? "WENO5" : "MUSCL",
                  (gx != 0 || gy != 0) ? "+gravity" : "");
    for (int N : Ns) {
        const double e = solve(N, mu, T_END, weno, gx, gy, Real(0.4));
        if (FILE* mf = std::fopen("out/mms.csv", "a")) { // vv dump
            std::fprintf(mf, "%s,%d,%.6g\n", lab, N, e);
            std::fclose(mf);
        }
        const double ord =
            errs.empty() ? 0.0 : std::log2(errs.back() / e);
        errs.push_back(e);
        if (ord == 0.0)
            std::printf("    N=%-4d  L1=%.3e\n", N, e);
        else
            std::printf("    N=%-4d  L1=%.3e  ordre=%.2f\n", N, e, ord);
    }
    // pente des moindres carres log(L1) vs log(N)
    const int n = int(errs.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < n; ++k) {
        const double lx = std::log(double(Ns[k])), ly = std::log(errs[k]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double order = std::log2(errs[n - 2] / errs[n - 1]);
    std::printf("    -> ordre (paire la plus fine) = %.2f  [LSQ %.2f]\n",
                order, -slope);
    return order;
}

// Sensibilite a la CFL a maillage fixe : e(0.4)/e(0.2) - 1. Une erreur
// purement spatiale (+ temporelle O(dt^2), negligeable ici) donne ~0 ;
// une erreur de splitting O(dt) donne ~ +100 %.
double cflSensitivity(double mu, bool weno, int N = 64) {
    const double e4 = solve(N, mu, T_END, weno, 0, 0, Real(0.4));
    const double e2 = solve(N, mu, T_END, weno, 0, 0, Real(0.2));
    const double r = e4 / e2 - 1;
    std::printf("  %s, mu = %.3g, N = %d : L1(cfl 0.4) = %.3e, "
                "L1(cfl 0.2) = %.3e -> ecart %+.1f %%\n",
                weno ? "WENO5" : "MUSCL", mu, N, e4, e2, 100 * r);
    return r;
}

} // namespace

int main() {
    std::printf("MMS — verification Navier-Stokes (MUSCL-Hancock + HLLC)\n");
    std::printf("solution manufacturee lisse periodique stationnaire,\n"
                "source = div(F_Euler) - div(F_visqueux) [FD ordre 4]\n\n");
    if (FILE* mf = std::fopen("out/mms.csv", "w")) { // vv dump (truncate)
        std::fprintf(mf, "label,N,L1\n");
        std::fclose(mf);
    }
    // Visqueux : aucune autre porte ne couvre l'ordre de l'operateur
    // Navier-Stokes. Le flux visqueux est central 2e ordre, commun aux
    // deux schemas -> les deux doivent donner ~2.
    std::printf("== Operateur visqueux (gate ~2) ==\n");
    const double vMuscl = orderStudy(0.01, false);
    const double vWeno = orderStudy(0.01, true);
    // Gravite : la densite n'a pas de source de gravite -> son ordre 2
    // confirme que le couplage gravite (quantite de mouvement / energie ->
    // vitesse / pression -> densite par les flux) est CONSISTANT.
    std::printf("== Operateur visqueux + source de gravite (gate ~2) ==\n");
    const double vGrav = orderStudy(0.01, false, 1.0, -0.5);
    // Inviscide : ordre 2 en MUSCL des que le source est dans le
    // predicteur (sinon erreur O(dt) -> ordre 1). WENO5 informatif (le
    // transitoire de cette solution n'est pas son regime de conception ;
    // l'ordre WENO est verifie par `convergence`).
    std::printf("== Euler inviscide (MUSCL gate ~2, WENO5 informatif) ==\n");
    const double eMuscl = orderStudy(0.0, false);
    const double eWeno = orderStudy(0.0, true);

    std::printf("== Sensibilite a la CFL (splitting du source) ==\n");
    const double cInv = cflSensitivity(0.0, false);
    const double cVisc = cflSensitivity(0.01, false);

    const bool okOrder = vMuscl > 1.8 && vWeno > 1.8 && vGrav > 1.8 &&
                         eMuscl > 1.8;
    const bool okCfl = std::fabs(cInv) < 0.05;
    const bool ok = okOrder && okCfl;
    std::printf("\n%s — operateur Navier-Stokes ordre 2 : visqueux MUSCL "
                "%.2f, WENO5 %.2f ; visqueux+gravite %.2f (gate 1.8)\n",
                ok ? "PASS" : "FAIL", vMuscl, vWeno, vGrav);
    std::printf("  inviscide MUSCL %.2f (gate 1.8) ; sensibilite CFL "
                "inviscide %+.1f %% (gate 5 %%)\n", eMuscl, 100 * cInv);
    std::printf("  [info] inviscide WENO5 %.2f ; sensibilite CFL visqueux "
                "%+.1f %% (flux visqueux hors predicteur)\n",
                eWeno, 100 * cVisc);
    return ok ? 0 : 1;
}

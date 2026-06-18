// MMS — verification par solutions manufacturees (Method of Manufactured
// Solutions). L'etude `convergence` mesure deja l'ordre EULER lisse ; ici
// on verifie l'opérateur NAVIER-STOKES COMPLET (Euler + flux visqueux).
//
// Principe : on choisit une solution lisse periodique STATIONNAIRE
// (rho, u, v, p sinusoidaux), on calcule analytiquement le terme source
//   S = div(F_Euler) - div(F_visqueux)
// qui rend cette solution stationnaire des equations, on l'injecte dans
// le pas mono-grille step2D(mu), et on laisse relaxer vers l'etat
// stationnaire discret. L'erreur L1 a l'etat stationnaire est la
// troncature du schema -> elle converge a l'ordre du schema.
//
// Le source est calcule EN DOUBLE par differences finies d'ordre 4
// (h = 1e-3) sur les champs analytiques : son erreur (~1e-12) est tres
// inferieure a celle du schema teste, il est donc "exact".
//
// Gate : ordre observe >= 1.8 (MUSCL-Hancock + flux visqueux central,
// ordre de conception 2), Euler seul (mu=0) ET visqueux (mu>0).

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

constexpr Real CFL = Real(0.4);
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
Cons source(double x, double y, double mu) {
    Vec4 S{};
    for (int c = 0; c < 4; ++c) {
        const double dFx = d4([&](double s) { return totalFx(x + s, y, mu)[c]; });
        const double dFy = d4([&](double s) { return totalFy(x, y + s, mu)[c]; });
        S[c] = dFx + dFy;
    }
    return {Real(S[0]), Real(S[1]), Real(S[2]), Real(S[3])};
}

// Etude d'ordre par ERREUR DE SOLUTION a l'etat stationnaire : on part
// de la solution manufacturee, on injecte le source S et on laisse
// relaxer. L'etat stationnaire discret s'ecarte de la solution exacte de
// la troncature du schema -> erreur L1 = O(h^p). (Mesure robuste en
// float : pas de soustraction divisee par un petit dt.)
double orderStudy(double mu, double tEnd, bool weno) {
    const int Ns[] = {16, 32, 64, 128};
    std::vector<double> errs;
    std::printf("  %s, mu = %.3g :\n", weno ? "WENO5" : "MUSCL", mu);
    const auto periodic = [](Grid& gg) {
        fillPeriodicX(gg);
        fillPeriodicY(gg);
    };
    for (int N : Ns) {
        Grid g(N, N, 0, 0, 1, 1);
        std::vector<Cons> S(g.q.size());
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                const std::size_t id = g.idx(i, j);
                g.q[id] = toCons({Real(mfg(g.xc(i), g.yc(j)).rho),
                                  Real(mfg(g.xc(i), g.yc(j)).u),
                                  Real(mfg(g.xc(i), g.yc(j)).v),
                                  Real(mfg(g.xc(i), g.yc(j)).p)});
                S[id] = source(g.xc(i), g.yc(j), mu);
            }
        Scratch2D s;
        ScratchW sw;
        double t = 0;
        while (t < tEnd * (1 - 1e-12)) {
            const Real dt = std::min(maxStableDt(g, CFL, Real(mu)),
                                     Real(tEnd - t));
            if (weno) {
                stepWeno2D(g, dt, sw, periodic, Real(mu));
            } else {
                periodic(g);
                step2D(g, dt, s, Real(mu));
            }
            for (int j = NG; j < NG + N; ++j) // injecter le source
                for (int i = NG; i < NG + N; ++i) {
                    const std::size_t id = g.idx(i, j);
                    g.q[id] = g.q[id] + dt * S[id];
                }
            t += dt;
        }
        double e = 0; // erreur L1 (densite) vs solution manufacturee
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i)
                e += std::fabs(double(g.at(i, j).rho) -
                               mfg(g.xc(i), g.yc(j)).rho);
        e /= double(N) * N;
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
    const double order = -slope;
    std::printf("    -> ordre moyen (LSQ) = %.2f\n", order);
    return order;
}

} // namespace

int main() {
    std::printf("MMS — verification Navier-Stokes (MUSCL-Hancock + HLLC)\n");
    std::printf("solution manufacturee lisse periodique stationnaire,\n"
                "source = div(F_Euler) - div(F_visqueux) [FD ordre 4]\n\n");
    // Visqueux : c'est ce que la MMS verifie (aucune autre porte ne couvre
    // l'ordre de l'operateur Navier-Stokes). Le flux visqueux est central
    // 2e ordre, commun aux deux schemas -> les deux doivent donner ~2.
    std::printf("== Operateur visqueux (gate ~2) ==\n");
    const double vMuscl = orderStudy(0.01, 0.5, false);
    const double vWeno = orderStudy(0.01, 0.5, true);
    // Inviscide stationnaire : informatif. Sans viscosite physique,
    // l'erreur de l'etat stationnaire est fixee par la viscosite NUMERIQUE
    // du schema (dissipation des flux de face), 1er ordre pour les deux
    // schemas sur cette solution -> ~1. Ce n'est PAS l'ordre transitoire de
    // conception (MUSCL ~2, WENO5 ~4-5), lui verifie par `convergence`
    // (advection d'une solution exacte : onde d'entropie, vortex).
    std::printf("== Inviscide stationnaire (informatif, != ordre transitoire) ==\n");
    const double eMuscl = orderStudy(0.0, 0.5, false);
    const double eWeno = orderStudy(0.0, 0.5, true);

    const bool ok = vMuscl > 1.8 && vWeno > 1.8;
    std::printf("\n%s — operateur visqueux Navier-Stokes : MUSCL %.2f, "
                "WENO5 %.2f (attendu ~2 ; gate 1.8)\n",
                ok ? "PASS" : "FAIL", vMuscl, vWeno);
    std::printf("  [info] inviscide stationnaire (visco. numerique) : "
                "MUSCL %.2f, WENO5 %.2f ; ordre de conception via "
                "`convergence`\n", eMuscl, eWeno);
    return ok ? 0 : 1;
}

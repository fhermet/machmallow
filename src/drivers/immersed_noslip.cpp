// No-slip visqueux sur paroi IMMERGÉE : couche limite de Blasius sur une
// plaque plane posée par le masque solide (et non par une BC de domaine).
// Un écoulement bas-Mach rencontre une plaque immergée (quelques rangées
// solides à partir du bord d'attaque) ; la paroi no-slip + les flux
// visqueux masque-aware doivent reproduire le profil de Blasius
// u/Ue = f'(eta), eta = y sqrt(Ue / (nu (x - x0))). Pendant du gate
// `blasius` (BC de domaine) mais avec la plaque immergée, sur CPU.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);
constexpr Real U0 = Real(0.3);   // Mach ~0.25
constexpr Real RHO0 = Real(1), P0 = Real(1);
constexpr Real MU = Real(1.2e-4);
constexpr double X0 = 0.2, XM = 0.8; // bord d'attaque / station de mesure

// f'(eta) de Blasius par RK4 (f''' + f f''/2 = 0, f''(0)=0.33206).
struct Blasius {
    std::vector<double> fp;
    double deta = 0.002;
    Blasius() {
        double f = 0, g = 0, h = 0.3320573362;
        for (double eta = 0; eta <= 10.0; eta += deta) {
            fp.push_back(g);
            auto d = [&](double F, double G, double H) {
                return std::array<double, 3>{G, H, -0.5 * F * H};
            };
            auto k1 = d(f, g, h);
            auto k2 = d(f + .5 * deta * k1[0], g + .5 * deta * k1[1],
                        h + .5 * deta * k1[2]);
            auto k3 = d(f + .5 * deta * k2[0], g + .5 * deta * k2[1],
                        h + .5 * deta * k2[2]);
            auto k4 = d(f + deta * k3[0], g + deta * k3[1], h + deta * k3[2]);
            f += deta / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
            g += deta / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
            h += deta / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
        }
    }
    double fpAt(double eta) const {
        if (eta <= 0) return 0;
        const double s = eta / deta;
        const std::size_t k = std::size_t(s);
        if (k + 1 >= fp.size()) return 1.0;
        return fp[k] + (s - k) * (fp[k + 1] - fp[k]);
    }
};

} // namespace

int main() {
    const int nx = 200, ny = 80;
    const double Lx = 1.0;
    Grid g(nx, ny, 0, 0, Lx, Real(ny) * (Lx / nx)); // cellules carrées
    const double dy = double(g.dy);
    const int plateRows = 2;            // épaisseur de la plaque immergée
    const double ys = plateRows * dy;   // surface de la plaque

    // plaque immergée : solide pour x >= X0 et y < ys
    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i)
            if (double(g.xc(i)) >= X0 && double(g.yc(j)) < ys)
                solid[g.idx(i, j)] = 1;

    const Cons fs = toCons(Prim{RHO0, U0, 0, P0});
    for (std::size_t k = 0; k < g.q.size(); ++k) g.q[k] = fs;

    // BC : entrée à gauche, transmissif droite, free-stream épinglé en haut,
    // plancher glissant en bas (sous la plaque : couvert par le masque).
    const auto fillBC = [&](Grid& gg) {
        for (int j = 0; j < gg.toty(); ++j)
            for (int k = 0; k < NG; ++k) gg.at(k, j) = fs;
        fillTransmissiveRight(gg);
        for (int i = 0; i < gg.totx(); ++i)
            for (int k = 0; k < NG; ++k) gg.at(i, NG + gg.ny + k) = fs;
        fillReflectiveBottom(gg);
    };

    Scratch2D s;
    const int im = NG + int(XM / Lx * nx);
    std::vector<double> prev(ny, 0);
    int steps = 0;
    const int MAXSTEPS = 12000;
    for (; steps < MAXSTEPS; ++steps) {
        fillBC(g);
        const Real dt = maxStableDt(g, CFL, MU);
        step2D(g, dt, s, MU, 0, 0, solid.data());
        if (steps % 400 == 0 && steps > 0) {
            double dmax = 0;
            for (int j = 0; j < ny; ++j) {
                const double u = double(toPrim(g.at(im, NG + j)).u);
                dmax = std::max(dmax, std::fabs(u - prev[j]));
                prev[j] = u;
            }
            if (dmax / double(U0) < 3e-4) { ++steps; break; }
        }
    }

    // profil au-dessus de la plaque vs Blasius (eta mesuré depuis la surface)
    const Blasius bl;
    const double xp = XM - X0, nu = double(MU) / double(RHO0);
    double Ue = 0;
    for (int j = plateRows; j < ny; ++j)
        Ue = std::max(Ue, double(toPrim(g.at(im, NG + j)).u));
    const double scale = std::sqrt(Ue / (nu * xp));
    double e2 = 0;
    int n = 0, uWallRow = -1;
    for (int j = plateRows; j < ny; ++j) {
        const double eta = (double(g.yc(NG + j)) - ys) * scale;
        const double u = double(toPrim(g.at(im, NG + j)).u) / Ue;
        if (eta <= 6.0) { const double d = u - bl.fpAt(eta); e2 += d * d; ++n; }
        if (uWallRow < 0) uWallRow = j; // première rangée fluide
    }
    const double rms = std::sqrt(e2 / n);

    // glissement résiduel à la paroi (doit être ~0 : no-slip)
    const double uWall = double(toPrim(g.at(im, NG + plateRows)).u);
    // frottement pariétal vs Cf = 0.664/sqrt(Re_x)
    const double dudy = uWall / (0.5 * dy);
    const double Cf = nu * double(RHO0) * dudy / (0.5 * double(RHO0) * Ue * Ue);
    const double Rex = Ue * xp / nu;
    const double CfExact = 0.664 / std::sqrt(Rex);

    std::printf("Blasius sur plaque IMMERGÉE (Re_x=%.0f, %d pas)\n", Rex,
                steps);
    std::printf("  profil RMS(u/Ue - f') = %.3e (gate 3e-2)\n", rms);
    std::printf("  glissement paroi u/Ue = %.3f (gate 0.12)\n", uWall / Ue);
    std::printf("  Cf = %.3e vs %.3e (Blasius) | écart %.0f%% (gate 25%%)\n",
                Cf, CfExact, 100 * std::fabs(Cf - CfExact) / CfExact);

    const bool ok = rms < 3e-2 && uWall / Ue < 0.12 &&
                    std::fabs(Cf - CfExact) / CfExact < 0.25;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// V&V: surface pressure coefficient Cp(theta) on a Mach-2 cylinder, cut cells
// vs the staircase mask. Two things are checked quantitatively:
//   gate 1 — the CUT-CELL stagnation Cp matches the exact Rayleigh pitot value
//            (normal-shock stagnation pressure) to a few percent;
//   gate 2 — the cut-cell surface Cp(theta) is SMOOTH while the staircase one
//            oscillates cell-to-cell: total variation TV(cut) << TV(staircase).
// The cut-cell surface pressure is the exact slip-wall pressure the scheme
// imposes at the EB centroid (wallPressure of the reconstructed state); the
// staircase "surface" is the fluid cells face-adjacent to the masked body.
// Also dumps out/cutcell_cp_{cut,staircase}.csv (theta_deg, Cp) + the modified
// Newtonian reference for the V&V figure.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr double CXc = 0.8, CYc = 0.8, RAD = 0.18;
constexpr double RHOINF = 1.4, UINF = 2.0, PINF = 1.0; // Mach 2 (c=1)
const double QINF = 0.5 * RHOINF * UINF * UINF;         // dynamic pressure
const double MACH = UINF / std::sqrt(GAMMA * PINF / RHOINF);

double cpOf(double p) { return (p - PINF) / QINF; }

// Rayleigh pitot: stagnation pressure behind a normal shock, /p_inf.
double pitotOverPinf(double M) {
    const double g = GAMMA;
    const double a = std::pow((g + 1) * (g + 1) * M * M /
                                  (4 * g * M * M - 2 * (g - 1)),
                              g / (g - 1));
    const double b = (1 - g + 2 * g * M * M) / (g + 1);
    return a * b;
}

template <class G>
void fillCase(G& g) { // left inflow (Mach 2 freestream), else transmissive
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
    const Cons fs = toCons({Real(RHOINF), Real(UINF), 0, Real(PINF)});
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) g.at(k, j) = fs;
}

// Cell-to-cell ROUGHNESS of Cp over the windward face: RMS of the residual
// from a smooth trend (5-point moving average, sorted by angle). This isolates
// the spurious oscillation from the physical rise to stagnation — a smooth
// distribution has ~zero roughness whatever its amplitude.
double windwardRoughness(std::vector<std::pair<double, double>> tp) {
    std::vector<std::pair<double, double>> w;
    for (auto& [t, cp] : tp)
        if (std::cos(t) < 0.2) w.push_back({t, cp}); // upstream ~2/3 of surface
    std::sort(w.begin(), w.end());
    const int n = int(w.size());
    if (n < 5) return 0;
    double s2 = 0;
    int cnt = 0;
    for (int k = 2; k < n - 2; ++k) {
        double avg = 0;
        for (int d = -2; d <= 2; ++d) avg += w[k + d].second;
        avg /= 5.0;
        const double r = w[k].second - avg;
        s2 += r * r;
        ++cnt;
    }
    return std::sqrt(s2 / std::max(cnt, 1));
}

void writeCsv(const char* path, const std::vector<std::pair<double, double>>& tp) {
    if (FILE* f = std::fopen(path, "w")) {
        std::fprintf(f, "theta_deg,cp\n");
        auto s = tp;
        std::sort(s.begin(), s.end());
        for (auto& [t, cp] : s)
            std::fprintf(f, "%.4f,%.6f\n", t * 180.0 / M_PI, cp);
        std::fclose(f);
    }
}

// ---- cut-cell surface Cp: exact slip-wall pressure at the EB centroid -----
std::vector<std::pair<double, double>> cutSurface(int nx, int ny, double tEnd,
                                                  double& stagCp) {
    Grid g(nx, ny, 0, 0, 2.4, 1.6);
    const auto geo = cutcell::build(
        g, [&](double x0, double x1, double y0, double y1) {
            return cutcell::circleMoments(CXc, CYc, RAD, x0, x1, y0, y1);
        });
    for (auto& q : g.q) q = toCons({Real(RHOINF), Real(UINF), 0, Real(PINF)});
    double t = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt = std::min(maxStableDt(g, Real(0.4)), Real(tEnd - t));
        stepCutCell(g, geo, dt, fillCase<Grid>, /*limited=*/true, /*mu=*/0);
        t += double(dt);
    }
    const auto grad = lsqGradients(g, geo);
    std::vector<std::pair<double, double>> tp;
    stagCp = -1e30;
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const CellMoments& m = geo.at(i, j);
            if (m.vol <= Real(1e-6) || m.vol >= Real(1) - Real(1e-6) ||
                m.eb.area <= Real(1e-9))
                continue;
            const Real ox = m.eb.cx - g.xc(i), oy = m.eb.cy - g.yc(j);
            const Prim w = reconstruct(toPrim(g.at(i, j)), grad[g.idx(i, j)],
                                       ox, oy);
            const Real nox = -m.eb.nx, noy = -m.eb.ny; // outward from fluid
            const Real un = w.u * nox + w.v * noy;
            const double pw = double(wallPressure(w, un));
            const double th = std::atan2(double(m.eb.cy) - CYc,
                                         double(m.eb.cx) - CXc);
            tp.push_back({th, cpOf(pw)});
            stagCp = std::max(stagCp, cpOf(pw)); // stagnation = max surface Cp
        }
    return tp;
}

// ---- staircase surface Cp: fluid cells face-adjacent to the masked body ---
std::vector<std::pair<double, double>> staircaseSurface(int nx, int ny,
                                                        double tEnd) {
    Grid g(nx, ny, 0, 0, 2.4, 1.6);
    std::vector<std::uint8_t> solid(g.q.size(), 0);
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) {
            const double dx = double(g.xc(i)) - CXc, dy = double(g.yc(j)) - CYc;
            solid[g.idx(i, j)] = (dx * dx + dy * dy < RAD * RAD) ? 1 : 0;
        }
    for (auto& q : g.q) q = toCons({Real(RHOINF), Real(UINF), 0, Real(PINF)});
    Scratch2D sc;
    double t = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt = std::min(maxStableDt(g, Real(0.4)), Real(tEnd - t));
        fillCase(g);
        step2D(g, dt, sc, /*mu=*/0, 0, 0, solid.data());
        t += double(dt);
    }
    std::vector<std::pair<double, double>> tp;
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            if (solid[g.idx(i, j)]) continue;
            if (!(solid[g.idx(i + 1, j)] || solid[g.idx(i - 1, j)] ||
                  solid[g.idx(i, j + 1)] || solid[g.idx(i, j - 1)]))
                continue; // not a surface-adjacent fluid cell
            const double th = std::atan2(double(g.yc(j)) - CYc,
                                         double(g.xc(i)) - CXc);
            tp.push_back({th, cpOf(double(toPrim(g.at(i, j)).p))});
        }
    return tp;
}

} // namespace

int main() {
    const int nx = 180, ny = 120;
    const double tEnd = 1.2; // ~1 flow-through: bow shock established

    double stagCp = 0;
    const auto cut = cutSurface(nx, ny, tEnd, stagCp);
    const auto stair = staircaseSurface(nx, ny, tEnd);
    writeCsv("out/cutcell_cp_cut.csv", cut);
    writeCsv("out/cutcell_cp_staircase.csv", stair);

    const double cpPitot = (pitotOverPinf(MACH) - 1.0) / QINF; // p0'-pinf over q
    const double err = std::fabs(stagCp - cpPitot) / cpPitot;
    const double rCut = windwardRoughness(cut), rStair = windwardRoughness(stair);

    std::printf("Mach %.3f cylinder | %dx%d | cut cells %zu | staircase surf "
                "cells %zu\n", MACH, nx, ny, cut.size(), stair.size());
    std::printf("gate 1 — cut stagnation Cp %.4f vs Rayleigh pitot %.4f: "
                "err %.2f%% (gate 3%%)\n", stagCp, cpPitot, 100 * err);
    std::printf("gate 2 — windward Cp roughness (RMS residual): cut %.4f "
                "vs staircase %.4f (%.1fx smoother, gate 3x)\n",
                rCut, rStair, rStair / rCut);
    std::printf("csv: out/cutcell_cp_{cut,staircase}.csv\n");

    const bool ok = err < 0.03 && rStair > 3.0 * rCut;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

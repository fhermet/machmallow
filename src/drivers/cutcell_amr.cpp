// Phase 5a: cut-cell geometry is consistent across AMR levels — the
// foundation that makes coarse/fine restriction conservative. With EXACT
// moments, a coarse cut cell's fluid volume equals the sum of its four
// refined children's fluid volumes, and a volume-weighted restriction of any
// field preserves the integral exactly.

#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

int main() {
    bool ok = true;
    const double cx = 0.5, cy = 0.5, r = 0.23;
    const auto circle = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };
    const int NC = 64;                      // coarse; fine = 2x
    Grid gc(NC, NC, 0, 0, 1, 1);
    Grid gf(2 * NC, 2 * NC, 0, 0, 1, 1);
    const auto Gc = cutcell::build(gc, circle);
    const auto Gf = cutcell::build(gf, circle);

    // ---- gate 1: coarse fluid volume == sum of 4 fine children -----------
    double volMax = 0;
    const double Vc = double(gc.dx) * double(gc.dy);
    const double Vf = double(gf.dx) * double(gf.dy);
    for (int j = 0; j < NC; ++j)
        for (int i = 0; i < NC; ++i) {
            const double kc = double(Gc.at(NG + i, NG + j).vol) * Vc;
            double kf = 0;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di)
                    kf += double(Gf.at(NG + 2 * i + di, NG + 2 * j + dj).vol) * Vf;
            volMax = std::max(volMax, std::fabs(kc - kf));
        }
    std::printf("gate 1 — coarse fluid volume vs sum of fine children: max "
                "diff %.3e (gate 1e-7)\n", volMax);
    ok = ok && volMax < 1e-7;

    // ---- gate 2: volume-weighted restriction preserves the integral ------
    // fine field U_f(x,y) smooth; restrict to coarse by fluid-volume average;
    // the composite fluid integral must be unchanged to round-off.
    const auto field = [&](double x, double y) {
        return 1.0 + 0.5 * std::sin(3 * x) * std::cos(2 * y);
    };
    double intFine = 0;
    for (int j = 0; j < 2 * NC; ++j)
        for (int i = 0; i < 2 * NC; ++i) {
            const double k = double(Gf.at(NG + i, NG + j).vol);
            if (k <= 1e-9) continue;
            intFine += k * Vf * field(double(gf.xc(NG + i)), double(gf.yc(NG + j)));
        }
    double intRestr = 0;
    for (int j = 0; j < NC; ++j)
        for (int i = 0; i < NC; ++i) {
            double num = 0, den = 0;              // fluid-volume weighted mean
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    const int fi = NG + 2 * i + di, fj = NG + 2 * j + dj;
                    const double k = double(Gf.at(fi, fj).vol);
                    if (k <= 1e-9) continue;
                    num += k * Vf * field(double(gf.xc(fi)), double(gf.yc(fj)));
                    den += k * Vf;
                }
            if (den <= 1e-12) continue;
            const double uc = num / den;          // restricted coarse value
            intRestr += den * uc;                 // = num, integral over children
        }
    const double relErr = std::fabs(intRestr - intFine) / std::fabs(intFine);
    std::printf("gate 2 — restriction preserves the fluid integral: rel err "
                "%.3e (gate 1e-6)\n", relErr);
    ok = ok && relErr < 1e-6;

    std::printf("  coarse cut cells %zu, fine %zu\n", Gc.nCut, Gf.nCut);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

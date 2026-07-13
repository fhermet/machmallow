// Phase 5a: cut-cell geometry is consistent across AMR levels — the
// foundation that makes coarse/fine restriction conservative. With EXACT
// moments, a coarse cut cell's fluid volume equals the sum of its four
// refined children's fluid volumes, and a volume-weighted restriction of any
// field preserves the integral exactly.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

// 2-level cut-cell run in a closed reflective box with an immersed cylinder,
// refined by one patch that fully contains the body. Returns the worst
// relative composite-mass drift over the run (reflux on/off).
double twoLevelMassDrift(bool reflux) {
    const int NC = 48;                       // coarse
    const int p0 = 12, p1 = 36;              // patch = coarse cells [p0,p1)^2
    const int NF = 2 * (p1 - p0);            // fine cells per side
    const double cx = 0.5, cy = 0.5, r = 0.2;
    const auto circ = [&](double x0, double x1, double y0, double y1) {
        return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
    };
    Grid gc(NC, NC, 0, 0, 1, 1);
    const double px0 = double(p0) / NC, py0 = double(p0) / NC;
    const double pL = double(p1 - p0) / NC;
    Grid gf(NF, NF, Real(px0), Real(py0), Real(pL), Real(pL));
    const auto geoC = cutcell::build(gc, circ);
    const auto geoF = cutcell::build(gf, circ);

    // IC: smooth pressure bump off-centre (sub-sonic sloshing), at rest.
    const auto ic = [&](double x, double y) {
        const double rr = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
        return toCons({1, 0, 0, Real(1.0 + 0.4 * std::exp(-rr / 0.01))});
    };
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i)
            gc.at(i, j) = ic(double(gc.xc(i)), double(gc.yc(j)));
    for (int j = 0; j < gf.toty(); ++j)
        for (int i = 0; i < gf.totx(); ++i)
            gf.at(i, j) = ic(double(gf.xc(i)), double(gf.yc(j)));

    const double Vc = double(gc.dx) * double(gc.dy);
    const double Vf = double(gf.dx) * double(gf.dy);
    const auto mass = [&]() {                 // composite: coarse-outside + fine
        double m = 0;
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < NC; ++i) {
                if (i >= p0 && i < p1 && j >= p0 && j < p1) continue; // covered
                m += double(geoC.at(NG + i, NG + j).vol) * Vc *
                     double(gc.at(NG + i, NG + j).rho);
            }
        for (int j = 0; j < NF; ++j)
            for (int i = 0; i < NF; ++i)
                m += double(geoF.at(NG + i, NG + j).vol) * Vf *
                     double(gf.at(NG + i, NG + j).rho);
        return m;
    };
    const double m0 = mass();

    const auto fillC = [](Grid& g) {
        fillReflectiveLeft(g); fillReflectiveRight(g);
        fillReflectiveBottom(g); fillReflectiveTop(g);
    };
    // fine ghosts: piecewise-constant prolongation from the coarse cell that
    // contains each fine ghost (the patch is interior, so a coarse cell always
    // exists there).
    const auto fillF = [&](Grid& g) {
        for (int j = 0; j < g.toty(); ++j)
            for (int i = 0; i < g.totx(); ++i) {
                if (i >= NG && i < NG + NF && j >= NG && j < NG + NF) continue;
                const int ci = p0 + (i - NG) / 2, cj = p0 + (j - NG) / 2;
                g.at(i, j) = gc.at(NG + ci, NG + cj);
            }
    };

    std::vector<Cons> FxC, FyC, FxF, FyF;
    double drift = 0;
    for (int s = 0; s < 200; ++s) {
        const Real dt = std::min(maxStableDt(gc, Real(0.4)),
                                 maxStableDt(gf, Real(0.4)));
        cutCellStepFluxed(gc, geoC, dt, fillC, FxC, FyC);
        cutCellStepFluxed(gf, geoF, dt, fillF, FxF, FyF);
        // restrict fine -> coarse (volume-weighted, conservative)
        for (int cj = p0; cj < p1; ++cj)
            for (int ci = p0; ci < p1; ++ci) {
                double num_r = 0, num_mx = 0, num_my = 0, num_E = 0, den = 0;
                for (int dj = 0; dj < 2; ++dj)
                    for (int di = 0; di < 2; ++di) {
                        const int fi = NG + 2 * (ci - p0) + di;
                        const int fj = NG + 2 * (cj - p0) + dj;
                        const double k = double(geoF.at(fi, fj).vol);
                        if (k <= 1e-9) continue;
                        const Cons& q = gf.at(fi, fj);
                        num_r += k * Vf * double(q.rho);
                        num_mx += k * Vf * double(q.mx);
                        num_my += k * Vf * double(q.my);
                        num_E += k * Vf * double(q.E);
                        den += k * Vf;
                    }
                if (den > 1e-12)
                    gc.at(NG + ci, NG + cj) =
                        Cons{Real(num_r / den), Real(num_mx / den),
                             Real(num_my / den), Real(num_E / den)};
            }
        // reflux: correct the coarse cells just OUTSIDE the patch so their
        // interface flux equals the sum of the fine fluxes there.
        if (reflux) {
            const auto corr = [&](int ci, int cj, const Cons& d) {
                const double k = double(geoC.at(NG + ci, NG + cj).vol);
                if (k > 1e-9)
                    gc.at(NG + ci, NG + cj) =
                        floorState(gc.at(NG + ci, NG + cj) +
                                   Real(dt / (k * Vc)) * d);
            };
            for (int cj = p0; cj < p1; ++cj) {   // left & right interfaces
                const int jf = 2 * (cj - p0);
                Cons sfL = FxF[gf.idx(NG - 1, NG + jf)] +
                           FxF[gf.idx(NG - 1, NG + jf + 1)];
                corr(p0 - 1, cj, FxC[gc.idx(NG + p0 - 1, NG + cj)] - sfL);
                Cons sfR = FxF[gf.idx(NG + NF - 1, NG + jf)] +
                           FxF[gf.idx(NG + NF - 1, NG + jf + 1)];
                corr(p1, cj, sfR - FxC[gc.idx(NG + p1 - 1, NG + cj)]);
            }
            for (int ci = p0; ci < p1; ++ci) {   // bottom & top interfaces
                const int if_ = 2 * (ci - p0);
                Cons sfB = FyF[gf.idx(NG + if_, NG - 1)] +
                           FyF[gf.idx(NG + if_ + 1, NG - 1)];
                corr(ci, p0 - 1, FyC[gc.idx(NG + ci, NG + p0 - 1)] - sfB);
                Cons sfT = FyF[gf.idx(NG + if_, NG + NF - 1)] +
                           FyF[gf.idx(NG + if_ + 1, NG + NF - 1)];
                corr(ci, p1, sfT - FyC[gc.idx(NG + ci, NG + p1 - 1)]);
            }
        }
        drift = std::max(drift, std::fabs(mass() - m0) / m0);
    }
    return drift;
}

} // namespace

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

    // ---- gate 3: 2-level composite conservation (aperture-aware reflux) ---
    const double driftOff = twoLevelMassDrift(false);
    const double driftOn = twoLevelMassDrift(true);
    std::printf("gate 3 — 2-level mass drift: %.3e with reflux | %.3e without "
                "(%.0fx worse)\n", driftOn, driftOff, driftOff / driftOn);
    ok = ok && driftOn < 1e-6 && driftOff > 100 * driftOn;

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

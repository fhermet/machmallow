// Phase 1 validation of the cut-cell geometry (moments only, no solver yet).
// Gates:
//   1. exact volume fraction — sum of fluid area equals the exact fluid area
//      of the domain (circle and half-plane) to fp32 roundoff, at EVERY
//      resolution (the whole point of analytic moments);
//   2. closure normal — the EB normal derived from the apertures points the
//      right way (radially outward for the circle, along (a,b) for the wedge);
//   3. perimeter — the summed EB length converges to the exact interface
//      length (2*pi*r / the chord across the box) as the grid refines.

#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

struct Stat {
    double area = 0, perim = 0, minAlign = 1e30;
    std::size_t nCut = 0;
};

template <class F>
Stat scan(int N, F momentFn, double cx, double cy) {
    Grid g(N, N, 0, 0, 1, 1);
    const auto G = cutcell::build(g, momentFn);
    Stat s;
    for (int j = NG; j < NG + N; ++j)
        for (int i = NG; i < NG + N; ++i) {
            const auto& m = G.at(i, j);
            s.area += double(m.vol) * g.dx * g.dy;
            s.perim += double(m.eb.area);
            if (double(m.vol) > 1e-6 && double(m.vol) < 1 - 1e-6) {
                ++s.nCut;
                // EB normal should point away from the reference direction
                // (radial for the circle: centre -> cell). Measure alignment.
                double rx = double(g.xc(i)) - cx, ry = double(g.yc(j)) - cy;
                const double rn = std::hypot(rx, ry);
                if (rn > 1e-9) {
                    rx /= rn; ry /= rn;
                    const double align = double(m.eb.nx) * rx +
                                         double(m.eb.ny) * ry;
                    s.minAlign = std::min(s.minAlign, align);
                }
            }
        }
    return s;
}

} // namespace

int main() {
    bool ok = true;
    const double PI = M_PI;

    // ---- circle (cylinder) : solid = disk, fluid = outside ----------------
    const double cx = 0.5, cy = 0.5, r = 0.25;
    const double exactFluid = 1.0 - PI * r * r;
    const double exactPerim = 2 * PI * r;
    std::printf("cut-cell geometry — circle r=%.3f in unit square\n", r);
    std::printf("  exact fluid area %.6f, circumference %.6f\n", exactFluid,
                exactPerim);
    std::printf("%6s %12s %12s %12s %8s\n", "N", "area err", "perim",
                "perim err", "nCut");
    double alignWorst = 1e30;
    for (int N : {32, 64, 128, 256}) {
        const Stat s = scan(
            N,
            [&](double x0, double x1, double y0, double y1) {
                return cutcell::circleMoments(cx, cy, r, x0, x1, y0, y1);
            },
            cx, cy);
        const double aerr = std::fabs(s.area - exactFluid);
        std::printf("%6d %12.3e %12.6f %12.3e %8zu\n", N, aerr, s.perim,
                    std::fabs(s.perim - exactPerim), s.nCut);
        ok = ok && aerr < 1e-5;               // exact at every resolution
        alignWorst = std::min(alignWorst, s.minAlign);
        if (N == 128)
            ok = ok && std::fabs(s.perim - exactPerim) / exactPerim < 1e-2;
    }
    std::printf("  worst EB-normal radial alignment: %.5f (gate > 0.99)\n",
                alignWorst);
    ok = ok && alignWorst > 0.99;

    // ---- half-plane (wedge) : solid = x + y < 0.8 -------------------------
    // fluid = x + y >= 0.8 -> a triangle cut off; exact area 1 - 0.32 = 0.68,
    // interface = the segment (0.8,0)-(0,0.8) of length 0.8*sqrt(2).
    const double a = 1, b = 1, c = 0.8;
    const double hpFluid = 1.0 - 0.5 * 0.8 * 0.8;
    const double hpPerim = 0.8 * std::sqrt(2.0);
    const double nrm = std::sqrt(2.0);
    std::printf("half-plane — solid x+y<%.2f (exact fluid %.4f, "
                "interface %.4f)\n",
                c, hpFluid, hpPerim);
    for (int N : {64, 256}) {
        Grid g(N, N, 0, 0, 1, 1);
        const auto G = cutcell::build(
            g, [&](double x0, double x1, double y0, double y1) {
                return cutcell::halfplaneMoments(a, b, c, x0, x1, y0, y1);
            });
        double area = 0, perim = 0, worstN = 1e30;
        for (int j = NG; j < NG + N; ++j)
            for (int i = NG; i < NG + N; ++i) {
                const auto& m = G.at(i, j);
                area += double(m.vol) * g.dx * g.dy;
                perim += double(m.eb.area);
                if (double(m.eb.area) > 1e-9) {
                    const double dotn = (double(m.eb.nx) * a +
                                         double(m.eb.ny) * b) / nrm;
                    worstN = std::min(worstN, dotn);
                }
            }
        std::printf("  N=%d: area %.6f (err %.2e), interface %.4f (err "
                    "%.2e), min normal.(a,b) %.5f\n",
                    N, area, std::fabs(area - hpFluid), perim,
                    std::fabs(perim - hpPerim), worstN);
        ok = ok && std::fabs(area - hpFluid) < 1e-5 && worstN > 0.999;
        if (N == 256)
            ok = ok && std::fabs(perim - hpPerim) / hpPerim < 1e-2;
    }

    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

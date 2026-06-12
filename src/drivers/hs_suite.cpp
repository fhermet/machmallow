// Quantitative Haas & Sturtevant (1987) gate: a Mach 1.22 shock in air
// hits a cylinder of helium contaminated with 28% air by mass
// (rho/rho_air = 0.18224, gamma = 1.645). The experiment reports the
// early-time velocities of the characteristic interface points, and
// Quirk & Karni (1996) is the canonical numerical comparison:
//
//                 experiment   Quirk & Karni
//   upstream edge    170 m/s        178 m/s
//   downstream edge  145 m/s        146 m/s
//   air jet          230 m/s        227 m/s
//
// Our ambient air (rho 1.4, p 1, gamma 1.4) has c = 1, and the
// experiment's ambient sound speed is 343 m/s, so expected velocities
// in code units are simply (V m/s) / 343. The interface points are
// tracked on the tube axis as the Y = 0.5 crossings of the base
// composite, and each velocity is a least-squares slope over the
// matching time window. Gates at +-10% of the experimental values
// (covering both the measurement uncertainty H&S report and our
// contaminated-helium model), asymmetric +15% for the jet whose
// measured window is the least sharp.

#include "amr/AmrGpuML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);

// least-squares slope of x(t) over [t0, t1]
double slope(const std::vector<double>& ts, const std::vector<double>& xs,
             double t0, double t1) {
    double st = 0, sx = 0, stt = 0, stx = 0;
    int n = 0;
    for (std::size_t k = 0; k < ts.size(); ++k) {
        if (ts[k] < t0 || ts[k] > t1) continue;
        st += ts[k]; sx += xs[k];
        stt += ts[k] * ts[k]; stx += ts[k] * xs[k];
        ++n;
    }
    if (n < 2) return 0;
    return (n * stx - st * sx) / (n * stt - st * st);
}

} // namespace

int main(int argc, char** argv) {
    const bool trace = argc > 1 && std::string(argv[1]) == "--trace";
    MetalContext ctx;
    std::printf("GPU: %s\n", ctx.device()->name()->utf8String());

    const GasPair gas{Real(1.4), Real(1.645)};
    // Mach 1.22 post-shock state in the ambient air (rho 1.4, p 1)
    const Real Ms = Real(1.22), M2 = Ms * Ms, g0 = Real(1.4);
    const Prim amb{Real(1.4), 0, 0, 1};
    const Prim post{amb.rho * (g0 + 1) * M2 / ((g0 - 1) * M2 + 2),
                    2 / (g0 + 1) * (Ms - 1 / Ms), 0,
                    amb.p * (1 + 2 * g0 / (g0 + 1) * (M2 - 1))};
    const Prim he{Real(0.2551), 0, 0, 1}; // contaminated helium
    const Real cx = Real(0.6), cy = Real(0.5), r = Real(0.25);

    AmrConfig cfg;
    cfg.maxLevels = 3;
    cfg.subcycle = true;
    cfg.tagThreshold = Real(0.05);
    cfg.tagVelocity = Real(0.05);
    cfg.species = true;
    cfg.gamma1 = gas.gamma1;
    cfg.gamma2 = gas.gamma2;

    const int NX = 128, NY = 64; // [0,2]x[0,1], finest 1/256
    AmrGpuML amr(ctx, NX, NY, 0, 0, 2, 1, cfg);
    amr.fillPhysicalGhosts = [&](GridRef& g, double) {
        // left: constant post-shock inflow (the front starts inside);
        // right open; top/bottom: tube walls
        const Cons qin = toConsG(post, gas.Gamma(0));
        for (int j = 0; j < g.toty(); ++j)
            for (int k = 0; k < NG; ++k)
                g.at(k, j) = qin;
        fillTransmissiveRight(g);
        fillReflectiveBottom(g);
        fillReflectiveTop(g);
    };
    amr.fillPatchPhysical = [&](GridRef& g, double, unsigned sides) {
        const Cons qin = toConsG(post, gas.Gamma(0));
        if (sides & SideLeft)
            for (int j = 0; j < g.toty(); ++j)
                for (int k = 0; k < NG; ++k)
                    g.at(k, j) = qin;
        if (sides & SideRight) fillTransmissiveRight(g);
        if (sides & SideBottom) fillReflectiveBottom(g);
        if (sides & SideTop) fillReflectiveTop(g);
    };
    const auto inBubble = [&](Real x, Real y) {
        const Real dx = x - cx, dy = y - cy;
        return dx * dx + dy * dy < r * r;
    };
    amr.init(
        [&](Real x, Real y) {
            const Prim& w =
                x < Real(0.3) ? post : (inBubble(x, y) ? he : amb);
            const Real Y = inBubble(x, y) && x >= Real(0.3) ? 1 : 0;
            return toConsG(w, gas.Gamma(Y));
        },
        [&](Real x, Real y) {
            return inBubble(x, y) && x >= Real(0.3) ? Real(1) : Real(0);
        });

    // track the axis Y=0.5 crossings of the base composite
    std::vector<double> ts, xu, xd;
    const auto measure = [&](double t) {
        const GridRef b = amr.coarseRef();
        const int j = NG + NY / 2;
        double up = -1, dn = -1;
        const auto Yat = [&](int i) {
            const auto s = amr.baseS()[b.idx(i, j)];
            return double(s.phi) /
                   std::max(double(b.at(i, j).rho), 1e-12);
        };
        for (int i = NG; i < NG + NX - 1; ++i) {
            const double a = Yat(i), c = Yat(i + 1);
            if (a < 0.5 && c >= 0.5 && up < 0)
                up = double(b.xc(i)) + (0.5 - a) / (c - a) * b.dx;
            if (a >= 0.5 && c < 0.5)
                dn = double(b.xc(i)) + (0.5 - a) / (c - a) * b.dx;
        }
        if (up > 0 && dn > 0) { ts.push_back(t); xu.push_back(up);
                                xd.push_back(dn); }
    };

    const double tEnd = 1.2;
    double t = 0;
    int steps = 0;
    while (t < tEnd * (1 - 1e-9)) {
        const Real dt =
            std::min(amr.maxStableDtAll(CFL), Real(tEnd - t));
        amr.step(dt, t);
        t += dt;
        if (++steps % 2 == 0) measure(t);
    }

    if (trace)
        for (std::size_t k = 0; k < ts.size(); ++k)
            std::printf("T %.4f %.5f %.5f\n", ts[k], xu[k], xd[k]);

    // Windows match the phases the experiment measures: V_ui right
    // after shock passage (the upstream point then ACCELERATES toward
    // the jet, so a late window would mix the two regimes), V_di once
    // the transmitted shock has cleanly separated from the interface,
    // V_jet at the jet's formation peak. The sliding-slope trace
    // (--trace) shows plateaus on each window.
    const double c343 = 343; // experimental ambient sound speed, m/s
    const double vUi = slope(ts, xu, 0.05, 0.25);
    const double vDi = slope(ts, xd, 0.55, 0.90);
    const double vJet = slope(ts, xu, 0.75, 1.10);
    const double eUi = 170 / c343, eDi = 145 / c343, eJet = 230 / c343;

    std::printf("Haas & Sturtevant, Mach 1.22 air -> helium cylinder "
                "(code units; exp / measured):\n");
    std::printf("  upstream edge  V_ui : %.4f / %.4f  (%+.1f%%, gate "
                "+-10%%)\n", eUi, vUi, 100 * (vUi / eUi - 1));
    std::printf("  downstream edge V_di: %.4f / %.4f  (%+.1f%%, gate "
                "+-10%%)\n", eDi, vDi, 100 * (vDi / eDi - 1));
    std::printf("  air jet        V_jet: %.4f / %.4f  (%+.1f%%, gate "
                "-10/+15%%)\n", eJet, vJet, 100 * (vJet / eJet - 1));
    std::printf("  patches L1 %zu L2 %zu, %d steps\n",
                amr.patchCount(1), amr.patchCount(2), steps);

    const bool ok = std::fabs(vUi / eUi - 1) < 0.10 &&
                    std::fabs(vDi / eDi - 1) < 0.10 &&
                    vJet / eJet - 1 > -0.10 && vJet / eJet - 1 < 0.15;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

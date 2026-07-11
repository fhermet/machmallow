// No-slip wall validation: the Blasius flat-plate boundary layer.
//
// A uniform low-Mach stream meets a flat plate whose leading edge sits
// at x0: the bottom boundary is a slip (reflective) wall upstream of x0
// and a NO-SLIP wall on the plate. The viscous layer grows downstream;
// at a measurement station the steady velocity profile must collapse
// onto the Blasius similarity solution u/U = f'(eta),
// eta = y sqrt(U / (nu (x - x0))), where f''' + f f''/2 = 0.
//
// Validates: the no-slip wall BC + the viscous fluxes produce the right
// wall shear and the right self-similar profile. Gates the RMS profile
// error, the boundary-layer thickness delta99, and the skin friction
// Cf against the exact Blasius values.
//
// Run to steady state with the explicit solver (no local time stepping
// yet), at Mach ~0.25 so compressibility corrections stay well under
// the gate tolerance.

#include "backend/metal/Euler2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);
constexpr Real U0 = Real(0.3);  // free-stream speed (Mach ~0.25)
constexpr Real RHO0 = Real(1), P0 = Real(1);
constexpr Real MU = Real(8e-5); // nu = mu/rho (higher Re_x)
constexpr double X0 = 0.15;     // leading edge
constexpr double XM = 0.85;     // measurement station

// Blasius f'(eta) table by RK4 from the wall with the known shear
// f''(0) = 0.332057; u/U = f'.
struct Blasius {
    std::vector<double> fp; // f'(eta) at eta = k*deta
    double deta = 0.002;
    Blasius() {
        double f = 0, g = 0, h = 0.3320573362; // f, f'=g, f''=h
        for (double eta = 0; eta <= 10.0; eta += deta) {
            fp.push_back(g);
            // RK4 on (f, g, h) with f' = g, g' = h, h' = -f h / 2
            auto deriv = [&](double F, double G, double H) {
                return std::array<double, 3>{G, H, -0.5 * F * H};
            };
            auto k1 = deriv(f, g, h);
            auto k2 = deriv(f + 0.5 * deta * k1[0], g + 0.5 * deta * k1[1],
                            h + 0.5 * deta * k1[2]);
            auto k3 = deriv(f + 0.5 * deta * k2[0], g + 0.5 * deta * k2[1],
                            h + 0.5 * deta * k2[2]);
            auto k4 = deriv(f + deta * k3[0], g + deta * k3[1],
                            h + deta * k3[2]);
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
        return fp[k] + (s - k) * (fp[k + 1] - fp[k]); // u/U
    }
};

// Slip upstream of the leading edge, no-slip on the plate; the other
// sides are uniform inflow (left) and zero-gradient (right, top).
void fillBC(GridRef& g) {
    const Cons inflow = toCons({RHO0, U0, 0, P0});
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) g.at(k, j) = inflow; // left inflow
    fillTransmissiveRight(g);
    // far-field top: pin the free stream (zero pressure gradient — a
    // transmissive top lets the layer's displacement accelerate the
    // flow, which thins the BL away from Blasius)
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k)
            g.at(i, NG + g.ny + k) = inflow;
    // bottom, per column: reflective (slip) ahead of x0, no-slip after
    for (int i = 0; i < g.totx(); ++i) {
        const bool plate = g.xc(i) >= Real(X0);
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + k);
            c.my = -c.my;
            if (plate) c.mx = -c.mx; // no-slip: kill tangential too
            g.at(i, NG - 1 - k) = c;
        }
    }
}

} // namespace

int main() {
    MetalContext ctx;
    const int nx = 320, ny = 256;
    const double Lx = 1.25, Ly = 1.0;
    const Real dx = Real(Lx / nx), dy = Real(Ly / ny);
    Euler2DGpu gpu(ctx, nx, ny, dx, dy);
    gpu.setViscosity(MU);
    GridRef g = gpu.ref(0, 0);

    const Cons init = toCons({RHO0, U0, 0, P0});
    for (int j = 0; j < g.toty(); ++j)
        for (int i = 0; i < g.totx(); ++i) g.at(i, j) = init;

    // March to steady state; stop when the station profile settles.
    const int im = NG + int(XM / Lx * nx);
    std::vector<double> prev(ny, 0);
    int steps = 0;
    const int MAXSTEPS = 30000;
    for (; steps < MAXSTEPS; ++steps) {
        fillBC(g);
        const Real dt = gpu.maxStableDt(CFL);
        gpu.step(dt);
        if (steps % 500 == 0 && steps > 0) {
            double dmax = 0;
            for (int j = 0; j < ny; ++j) {
                const double u = toPrim(g.at(im, NG + j)).u;
                dmax = std::max(dmax, std::fabs(u - prev[j]));
                prev[j] = u;
            }
            if (dmax / U0 < 2e-4) { steps++; break; }
        }
    }

    // Compare the station profile to Blasius. Blasius similarity is
    // defined with the EDGE velocity Ue (the local free stream), not the
    // inflow U0 — use the column top as Ue so any mild acceleration over
    // the plate doesn't bias the comparison.
    const Blasius bl;
    const double xp = XM - X0; // distance from the leading edge
    const double nu = double(MU) / double(RHO0);
    double Ue = 0;
    for (int j = 0; j < ny; ++j)
        Ue = std::max(Ue, double(toPrim(g.at(im, NG + j)).u));
    const double scale = std::sqrt(Ue / (nu * xp)); // eta = y * scale
    double e2 = 0;
    int n = 0;
    double d99 = -1;
    for (int j = 0; j < ny; ++j) {
        const double y = double(g.yc(NG + j));
        const double eta = y * scale;
        const double u = double(toPrim(g.at(im, NG + j)).u) / Ue;
        if (eta <= 6.0) { // within / just above the boundary layer
            const double d = u - bl.fpAt(eta);
            e2 += d * d;
            ++n;
        }
        if (d99 < 0 && u >= 0.99) d99 = y; // first y reaching 0.99 Ue
    }
    const double rms = std::sqrt(e2 / n);
    // dump the station profile (u/Ue vs eta, and Blasius f') for the V&V
    // figure (vv/). Guarded: skip silently if out/ is unavailable.
    if (FILE* pf = std::fopen("out/blasius_profile.csv", "w")) {
        std::fprintf(pf, "eta,u_computed,blasius\n");
        for (int j = 0; j < ny; ++j) {
            const double y = double(g.yc(NG + j));
            const double eta = y * scale;
            if (eta > 6.0) break;
            std::fprintf(pf, "%.6g,%.6g,%.6g\n", eta,
                         double(toPrim(g.at(im, NG + j)).u) / Ue,
                         bl.fpAt(eta));
        }
        std::fclose(pf);
    }
    if (std::getenv("PROFILE"))
        for (int j = 0; j < ny; j += 4) {
            const double y = double(g.yc(NG + j));
            const double eta = y * scale;
            if (eta > 6) break;
            std::fprintf(stderr, "  eta %.2f  u/Ue %.3f  Blasius %.3f\n",
                         eta, double(toPrim(g.at(im, NG + j)).u) / Ue,
                         bl.fpAt(eta));
        }

    // delta99: Blasius eta99 = 4.91 -> delta = 4.91 / scale.
    const double d99Exact = 4.91 / scale;
    // skin friction from the wall shear, vs Cf = 0.664/sqrt(Re_x).
    const double dudyWall =
        (double(toPrim(g.at(im, NG)).u)) / double(dy * 0.5);
    const double tauW = nu * RHO0 * dudyWall;
    const double Cf = tauW / (0.5 * RHO0 * Ue * Ue);
    const double Rex = Ue * xp / nu;
    const double CfExact = 0.664 / std::sqrt(Rex);
    std::printf("  Ue/U0 = %.3f (free-stream drift at the station)\n",
                Ue / double(U0));

    std::printf("Blasius flat plate: Re_x = %.0f, steady after %d "
                "steps\n", Rex, steps);
    std::printf("  profile RMS(u/U - f') over eta<6 : %.4e (gate "
                "3e-2)\n", rms);
    std::printf("  delta99: %.4f vs Blasius %.4f  (%.1f%%)\n", d99,
                d99Exact, 100 * (d99 / d99Exact - 1));
    std::printf("  Cf: %.4e vs 0.664/sqrt(Re_x) %.4e  (%.1f%%)\n", Cf,
                CfExact, 100 * (Cf / CfExact - 1));

    // multi-station sweep along the plate for the Cf(Re_x) & delta99(Re_x)
    // figures (vv/): same wall-shear / edge-velocity measurement at a range
    // of x stations, each compared to the Blasius law at its own Re_x.
    if (FILE* cf = std::fopen("out/blasius_cf.csv", "w")) {
        std::fprintf(cf, "x,Rex,Cf,Cf_exact,d99,d99_exact\n");
        for (double xs = 0.30; xs <= 1.10 + 1e-9; xs += 0.05) {
            const int ii = NG + int(xs / Lx * nx);
            const double xps = double(g.xc(ii)) - X0;
            if (ii < NG || ii >= NG + nx || xps <= 0) continue;
            double Ues = 0;
            for (int j = 0; j < ny; ++j)
                Ues = std::max(Ues, double(toPrim(g.at(ii, NG + j)).u));
            const double sc = std::sqrt(Ues / (nu * xps));
            double d99s = -1;
            for (int j = 0; j < ny; ++j)
                if (double(toPrim(g.at(ii, NG + j)).u) / Ues >= 0.99) {
                    d99s = double(g.yc(NG + j)); break;
                }
            const double dudy =
                double(toPrim(g.at(ii, NG)).u) / double(dy * 0.5);
            const double Cfs = (nu * RHO0 * dudy) / (0.5 * RHO0 * Ues * Ues);
            const double Rexs = Ues * xps / nu;
            std::fprintf(cf, "%.5g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                         double(g.xc(ii)), Rexs, Cfs,
                         0.664 / std::sqrt(Rexs), d99s, 4.91 / sc);
        }
        std::fclose(cf);
    }

    // ---- second pass in WENO5, for the scheme-comparison figures (vv/) ----
    // Same viscous case run with scheme = weno5; on a smooth steady boundary
    // layer the profiles nearly coincide with MUSCL (the viscous operator is
    // shared, no discontinuities to sharpen) — a useful consistency check.
    {
        Euler2DGpu gw(ctx, nx, ny, dx, dy);
        gw.enableWeno();
        gw.setViscosity(MU);
        GridRef gg = gw.ref(0, 0);
        for (int j = 0; j < gg.toty(); ++j)
            for (int i = 0; i < gg.totx(); ++i) gg.at(i, j) = init;
        std::vector<double> pv(ny, 0);
        for (int s = 0; s < MAXSTEPS; ++s) {
            fillBC(gg);
            gw.step(gw.maxStableDt(CFL));
            if (s % 500 == 0 && s > 0) {
                double dm = 0;
                for (int j = 0; j < ny; ++j) {
                    const double u = toPrim(gg.at(im, NG + j)).u;
                    dm = std::max(dm, std::fabs(u - pv[j])); pv[j] = u;
                }
                if (dm / U0 < 2e-4) break;
            }
        }
        double Ue2 = 0;
        for (int j = 0; j < ny; ++j)
            Ue2 = std::max(Ue2, double(toPrim(gg.at(im, NG + j)).u));
        const double sc2 = std::sqrt(Ue2 / (nu * xp));
        if (FILE* pf = std::fopen("out/blasius_profile_weno.csv", "w")) {
            std::fprintf(pf, "eta,u_computed,blasius\n");
            for (int j = 0; j < ny; ++j) {
                const double eta = double(gg.yc(NG + j)) * sc2;
                if (eta > 6.0) break;
                std::fprintf(pf, "%.6g,%.6g,%.6g\n", eta,
                             double(toPrim(gg.at(im, NG + j)).u) / Ue2,
                             bl.fpAt(eta));
            }
            std::fclose(pf);
        }
        if (FILE* cf = std::fopen("out/blasius_cf_weno.csv", "w")) {
            std::fprintf(cf, "x,Rex,Cf,Cf_exact,d99,d99_exact\n");
            for (double xs = 0.30; xs <= 1.10 + 1e-9; xs += 0.05) {
                const int ii = NG + int(xs / Lx * nx);
                const double xps = double(gg.xc(ii)) - X0;
                if (ii < NG || ii >= NG + nx || xps <= 0) continue;
                double Ues = 0;
                for (int j = 0; j < ny; ++j)
                    Ues = std::max(Ues, double(toPrim(gg.at(ii, NG + j)).u));
                const double scs = std::sqrt(Ues / (nu * xps));
                double d99s = -1;
                for (int j = 0; j < ny; ++j)
                    if (double(toPrim(gg.at(ii, NG + j)).u) / Ues >= 0.99) {
                        d99s = double(gg.yc(NG + j)); break;
                    }
                const double dudy =
                    double(toPrim(gg.at(ii, NG)).u) / double(dy * 0.5);
                const double Cfs =
                    (nu * RHO0 * dudy) / (0.5 * RHO0 * Ues * Ues);
                const double Rexs = Ues * xps / nu;
                std::fprintf(cf, "%.5g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                             double(gg.xc(ii)), Rexs, Cfs,
                             0.664 / std::sqrt(Rexs), d99s, 4.91 / scs);
            }
            std::fclose(cf);
        }
    }

    const bool ok = rms < 3e-2 && std::fabs(d99 / d99Exact - 1) < 0.15 &&
                    std::fabs(Cf / CfExact - 1) < 0.20;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

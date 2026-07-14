// V&V: subsonic viscous flow past a cylinder at Re=40 (steady regime), on the
// GPU cut-cell solver (2nd-order LSQ + RK2, no-slip embedded boundary). Two
// classic quantitative targets vs the literature:
//   - recirculation bubble length Lw/D behind the cylinder (Lw/D ≈ 2.2 at
//     Re=40; Coutanceau & Bouard 1977, Tritton 1959, many DNS);
//   - drag coefficient Cd (≈ 1.5–1.6 at Re=40).
// Low Mach (M=0.3) keeps it near-incompressible; single grid so the wake
// centreline is sampled directly. Dumps out/cutcell_cylinder_centerline.csv
// (x/D, u/U) for the V&V figure.

#include "backend/metal/CutCell2DGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "physics/Euler.hpp"
#include "solver/CutCell2D.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mm;

namespace {

constexpr double LX = 2.4, LY = 1.6;          // domain
constexpr double CXc = 0.6, CYc = 0.8, RAD = 0.05; // cylinder (D = 0.1)
constexpr double D = 2 * RAD;
constexpr double RHOINF = 1.0, PINF = 1.0;    // gamma=1.4 -> c = 1.1832
constexpr double RE = 40.0;

const double CINF = std::sqrt(GAMMA * PINF / RHOINF);
const double UINF = 0.3 * CINF;               // Mach 0.3
const double MU = RHOINF * UINF * D / RE;     // Re = rho U D / mu

template <class G>
void fillFlow(G& g) { // inflow left, transmissive elsewhere
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
    const Cons fs = toCons({Real(RHOINF), Real(UINF), 0, Real(PINF)});
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) g.at(k, j) = fs;
}

} // namespace

int main() {
    const int nx = 384, ny = 256;             // dx = 0.00625 -> 16 cells/D
    const double tEnd = 26.0;                  // ~4 flow-throughs to steady

    Grid gc(nx, ny, 0, 0, LX, LY);
    const auto geo = cutcell::build(
        gc, [&](double x0, double x1, double y0, double y1) {
            return cutcell::circleMoments(CXc, CYc, RAD, x0, x1, y0, y1);
        });
    for (auto& q : gc.q) q = toCons({Real(RHOINF), Real(UINF), 0, Real(PINF)});

    MetalContext ctx;
    CutCell2DGpu gpu(ctx, nx, ny, gc.dx, gc.dy);
    gpu.setGeometry(geo);
    gpu.enableO2();
    gpu.setViscosity(Real(MU), heatConductivity(Real(MU)));
    GridRef g = gpu.ref(0, 0);
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i) g.at(i, j) = gc.at(i, j);

    std::printf("Re %.0f | Mach 0.30 | %dx%d (%.0f cells/D) | mu %.3e | "
                "t_end %.0f\n", RE, nx, ny, D / double(gc.dx), MU, tEnd);
    double t = 0;
    int steps = 0;
    Real dtBase = maxStableDt(gc, Real(0.4), Real(MU)); // uniform init estimate
    while (t < tEnd * (1 - 1e-9)) {
        // The wave speeds are bounded (~U + c) and near-constant at steady low
        // Mach, so the CFL dt is only re-measured every 100 steps (avoids a
        // full-grid CPU copy every step); clamped so t lands exactly on t_end.
        if (steps % 100 == 0) {
            for (int j = 0; j < gc.toty(); ++j)
                for (int i = 0; i < gc.totx(); ++i) gc.at(i, j) = g.at(i, j);
            dtBase = maxStableDt(gc, Real(0.4), Real(MU));
        }
        const Real dt = std::min(dtBase, Real(tEnd - t));
        fillFlow(g); gpu.divO2(); gpu.rk2Stage1(dt);
        fillFlow(g); gpu.divO2(); gpu.rk2Stage2(dt);
        t += double(dt);
        ++steps;
    }

    // sync the final GPU state to the CPU grid for extraction (all on gc)
    for (int j = 0; j < gc.toty(); ++j)
        for (int i = 0; i < gc.totx(); ++i) gc.at(i, j) = g.at(i, j);

    // ---- recirculation bubble length: centreline u(x) behind the cylinder ---
    const int jc = NG + int((CYc - 0) / gc.dy);        // row nearest y = CY
    std::vector<std::pair<double, double>> line;         // (x/D, u/U)
    double lw = -1;
    double uprev = 1;
    for (int i = NG; i < NG + nx; ++i) {
        const double x = double(gc.xc(i));
        if (x < CXc + RAD) continue;                    // downstream of the body
        if (geo.at(i, jc).vol <= Real(1e-9)) continue;  // skip solid
        const double u = double(toPrim(gc.at(i, jc)).u);
        line.push_back({(x - CXc) / D, u / UINF});
        if (lw < 0 && uprev < 0 && u >= 0)              // reattachment (u: -→+)
            lw = (x - (CXc + RAD)) / D;                 // from the rear stagnation
        uprev = u;
    }

    // ---- drag coefficient: momentum flux through the EB over the surface ----
    const auto grad = lsqGradients(gc, geo);
    double fx = 0;
    const double kT = 0;
    (void)kT;
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const CellMoments& m = geo.at(i, j);
            if (m.vol <= Real(1e-6) || m.vol >= Real(1) - Real(1e-6) ||
                m.eb.area <= Real(1e-9))
                continue;
            const Real ox = m.eb.cx - gc.xc(i), oy = m.eb.cy - gc.yc(j);
            const Prim w = reconstruct(toPrim(gc.at(i, j)), grad[gc.idx(i, j)],
                                       ox, oy);
            const Real nox = -m.eb.nx, noy = -m.eb.ny;
            const Real un = w.u * nox + w.v * noy;
            const double pw = double(wallPressure(w, un));
            double febx = pw * nox;                     // pressure part
            const Real dn = std::max(std::fabs(ox * m.eb.nx + oy * m.eb.ny),
                                     Real(0.25) * gc.dx);
            const Real un2 = w.u * m.eb.nx + w.v * m.eb.ny;
            febx += MU * (w.u - un2 * m.eb.nx) / dn;    // viscous traction (x)
            fx += double(m.eb.area) * febx;             // momentum into the wall
        }
    const double cd = fx / (0.5 * RHOINF * UINF * UINF * D);

    if (FILE* f = std::fopen("out/cutcell_cylinder_centerline.csv", "w")) {
        std::fprintf(f, "x_over_D,u_over_U\n");
        for (auto& [xd, uu] : line) std::fprintf(f, "%.5f,%.6f\n", xd, uu);
        std::fclose(f);
    }

    // References at Re=40: Lw/D ≈ 2.2 (Coutanceau-Bouard 2.13, Tritton),
    // Cd ≈ 1.5–1.6.
    const double lwRef = 2.2, cdRef = 1.55;
    const double lwErr = std::fabs(lw - lwRef) / lwRef;
    const double cdErr = std::fabs(cd - cdRef) / cdRef;
    std::printf("%d steps | recirculation bubble Lw/D = %.3f (ref ~%.2f, "
                "err %.0f%%)\n", steps, lw, lwRef, 100 * lwErr);
    std::printf("drag coefficient Cd = %.3f (ref ~%.2f, err %.0f%%)\n",
                cd, cdRef, 100 * cdErr);
    std::printf("csv: out/cutcell_cylinder_centerline.csv\n");

    const bool ok = lw > 0 && lwErr < 0.20 && cdErr < 0.25;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

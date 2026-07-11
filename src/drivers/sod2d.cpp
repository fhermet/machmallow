// Phase 2 validation: Sod shock tube run diagonally across a 2D grid.
// The solution depends only on xi = (x + y - 1)/sqrt(2), so comparing the
// central region (untouched by boundary effects) against the exact 1D
// Riemann solution tests the 2D scheme and its isotropy at once.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"
#include "io/VtiWriter.hpp"
#include "numerics/ExactRiemann.hpp"
#include "solver/Muscl2D.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace {

using namespace mm;

constexpr Real CFL = Real(0.4);
constexpr double TEND = 0.15;
constexpr exact::State SOD_L = {1.0, 0.0, 1.0};
constexpr exact::State SOD_R = {0.125, 0.0, 0.1};

double runDiagonalSod(int n, bool writeOutput) {
    Grid g(n, n, 0, 0, 1, 1);
    for (int j = NG; j < NG + n; ++j)
        for (int i = NG; i < NG + n; ++i) {
            const bool left = g.xc(i) + g.yc(j) < 1.0;
            const exact::State& st = left ? SOD_L : SOD_R;
            g.at(i, j) = toCons({Real(st.rho), 0, 0, Real(st.p)});
        }

    Scratch2D scratch;
    double t = 0;
    while (t < TEND) {
        fillTransmissiveLeft(g);
        fillTransmissiveRight(g);
        fillTransmissiveBottom(g);
        fillTransmissiveTop(g);
        const Real dt =
            std::min(maxStableDt(g, CFL), Real(TEND - t));
        step2D(g, dt, scratch);
        t += dt;
    }

    // L1(rho) over the central square, where boundaries cannot have
    // influenced the solution yet (fastest signal ~1.75 * 0.15 ~ 0.26).
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    double err = 0;
    for (int j = NG; j < NG + n; ++j)
        for (int i = NG; i < NG + n; ++i) {
            const double x = g.xc(i), y = g.yc(j);
            if (x < 0.3 || x > 0.7 || y < 0.3 || y > 0.7) continue;
            const double xi = (x + y - 1.0) * inv_sqrt2;
            const exact::State ex = exact::sample(SOD_L, SOD_R, xi / TEND);
            err += std::fabs(double(toPrim(g.at(i, j)).rho) - ex.rho) *
                   g.dx * g.dy;
        }

    if (writeOutput) {
        std::filesystem::create_directories("out");
        writeVti("out/sod2d_" + std::to_string(n) + ".vti", g);
        // V&V dump: the 2D field (subsampled) + the exact 1D Riemann curve
        // vs the similarity variable xi -> shows the 2D solution collapses
        // onto the 1D solution (isotropy) and matches the exact Riemann.
        if (FILE* ff = std::fopen("out/sod2d_field.csv", "w")) {
            std::fprintf(ff, "x,y,rho\n");
            for (int j = NG; j < NG + n; j += 2)
                for (int i = NG; i < NG + n; i += 2)
                    std::fprintf(ff, "%.5g,%.5g,%.6g\n", double(g.xc(i)),
                                 double(g.yc(j)),
                                 double(toPrim(g.at(i, j)).rho));
            std::fclose(ff);
        }
        if (FILE* ef = std::fopen("out/sod2d_exact.csv", "w")) {
            std::fprintf(ef, "xi,rho_exact\n");
            for (int k = 0; k <= 400; ++k) {
                const double xi = -0.35 + 0.70 * k / 400.0;
                std::fprintf(ef, "%.6g,%.6g\n", xi,
                             exact::sample(SOD_L, SOD_R, xi / TEND).rho);
            }
            std::fclose(ef);
        }
    }
    return err;
}

} // namespace

int main() {
    std::printf("Diagonal 2D Sod, MUSCL-Hancock + HLLC, t = %.2f\n", TEND);
    std::printf("%8s %12s %8s\n", "N", "L1(rho)", "order");

    const int grids[] = {64, 128, 256};
    std::vector<double> errs;
    for (int n : grids) {
        errs.push_back(runDiagonalSod(n, n == 256));
        const std::size_t k = errs.size();
        if (k > 1) {
            std::printf("%8d %12.4e %8.2f\n", n, errs.back(),
                        std::log2(errs[k - 2] / errs[k - 1]));
        } else {
            std::printf("%8d %12.4e %8s\n", n, errs.back(), "-");
        }
    }

    const double meanOrder =
        std::log2(errs.front() / errs.back()) / double(errs.size() - 1);
    std::printf("mean order: %.2f\n", meanOrder);
    if (meanOrder < 0.7) {
        std::fprintf(stderr, "FAIL: convergence regression\n");
        return EXIT_FAILURE;
    }
    std::printf("PASS\n");
    return EXIT_SUCCESS;
}

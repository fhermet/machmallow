// Generic case runner: `./build/run cases/dmr.ini`
// The case file picks the physics preset (sod | dmr | shear), the
// backend (cpu = Amr2, hybrid = AmrGpu), resolution, viscosity, AMR
// parameters and output cadence. Validation drivers (sod*/dmr*/shear)
// stay hardcoded — this is the front door for exploratory runs.

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/Dmr.hpp"
#include "core/Boundary.hpp"
#include "core/Config.hpp"
#include "io/VthbWriter.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

namespace {

using namespace mm;
using Clock = std::chrono::steady_clock;

struct CaseSetup {
    Real x0 = 0, y0 = 0, lx = 1, ly = 1;
    int aspect = 4; // nx = aspect * ny
    double tEnd = 0.2;
    bool periodicX = false, periodicY = false;
    std::function<Cons(Real, Real)> ic;
};

template <class G>
void transmissiveSides(G& g, unsigned sides) {
    if (sides & SideLeft) fillTransmissiveLeft(g);
    if (sides & SideRight) fillTransmissiveRight(g);
    if (sides & SideBottom) fillTransmissiveBottom(g);
    if (sides & SideTop) fillTransmissiveTop(g);
}

constexpr unsigned ALL_SIDES =
    SideLeft | SideRight | SideBottom | SideTop;

// Geometry/time of a preset (needed before the AMR object exists).
CaseSetup caseGeometry(const std::string& name) {
    CaseSetup s;
    if (name == "sod") {
        s.lx = 1; s.ly = Real(0.25); s.aspect = 4; s.tEnd = 0.2;
    } else if (name == "dmr") {
        s.lx = 4; s.ly = 1; s.aspect = 4; s.tEnd = dmr::TEND;
    } else if (name == "shear") {
        s.lx = 1; s.ly = Real(0.25); s.aspect = 4; s.tEnd = 0.15;
    } else if (name == "kh") {
        s.lx = 1; s.ly = 1; s.aspect = 1; s.tEnd = 2.0;
        s.periodicX = s.periodicY = true;
    } else {
        throw std::runtime_error("unknown case: " + name +
                                 " (expected sod | dmr | shear | kh)");
    }
    return s;
}

// Wire the preset's IC and BC callbacks into the AMR object.
template <class AMR>
void wireCase(const std::string& name, const Config& cfg, AMR& amr,
              CaseSetup& s) {
    if (name == "sod") {
        s.ic = [](Real x, Real) {
            const bool l = x < Real(0.5);
            return toCons({l ? Real(1) : Real(0.125), 0, 0,
                           l ? Real(1) : Real(0.1)});
        };
        amr.fillPhysicalGhosts = [](auto& g, double) {
            transmissiveSides(g, ALL_SIDES);
        };
        amr.fillPatchPhysical = [](auto& g, double, unsigned sides) {
            transmissiveSides(g, sides);
        };
    } else if (name == "dmr") {
        s.ic = [](Real x, Real y) {
            return dmr::behindShock(x, y, 0) ? dmr::POST : dmr::PRE;
        };
        amr.fillPhysicalGhosts = [](auto& g, double t) {
            dmr::fillGhosts(g, t);
        };
        amr.fillPatchPhysical = [](auto& g, double t, unsigned sides) {
            dmr::fillGhostsSides(g, t, sides);
        };
    } else if (name == "shear") {
        const Real v0 = Real(cfg.getReal("case.v0", 0.2));
        const Real w0 = Real(cfg.getReal("case.width", 0.05));
        s.ic = [v0, w0](Real x, Real) {
            const Real v =
                Real(0.5) * v0 *
                Real(std::erf(double(x - Real(0.5)) / double(w0)));
            return toCons({Real(1), 0, v, Real(1)});
        };
        amr.fillPhysicalGhosts = [](auto& g, double) {
            transmissiveSides(g, ALL_SIDES);
        };
        amr.fillPatchPhysical = [](auto& g, double, unsigned sides) {
            transmissiveSides(g, sides);
        };
    } else if (name == "kh") {
        // Doubly periodic Kelvin-Helmholtz: dense band in counterflow,
        // sinusoidal v perturbation seeds the billows.
        const Real u0 = Real(cfg.getReal("case.u0", 0.5));
        const Real delta = Real(cfg.getReal("case.perturb", 0.01));
        s.ic = [u0, delta](Real x, Real y) {
            const bool band = std::fabs(y - Real(0.5)) < Real(0.25);
            return toCons({band ? Real(2) : Real(1),
                           band ? u0 / 2 : -u0 / 2,
                           delta * Real(std::sin(4 * M_PI * double(x))),
                           Real(2.5)});
        };
        amr.fillPhysicalGhosts = [](auto& g, double) {
            fillPeriodicX(g);
            fillPeriodicY(g);
        };
        // Fully periodic: no physical patch sides, callback stays unset.
    }
}

template <class AMR>
int runCase(AMR& amr, const CaseSetup& s, const Config& cfg) {
    const Real cfl = Real(cfg.getReal("cfl", 0.4));
    const double tEnd = cfg.getReal("t_end", s.tEnd);
    const int frames = cfg.getInt("output.frames", 4);
    const std::string prefix =
        cfg.getString("output.prefix", "out/run");
    std::filesystem::create_directories(
        std::filesystem::path(prefix).parent_path());

    amr.init(s.ic);
    const double m0 = amr.totalMass();

    double t = 0, nextFrame = frames > 0 ? tEnd / frames : 1e30;
    int steps = 0, frame = 0;
    std::size_t cellSteps = 0, maxPatches = 0;
    const auto t0 = Clock::now();
    while (t < tEnd) {
        Real dt = std::min(amr.maxStableDtAll(cfl), Real(tEnd - t));
        if (steps < 10) dt *= Real(0.3); // gentle start
        amr.step(dt, t);
        t += dt;
        ++steps;
        cellSteps += amr.cellCount();
        maxPatches = std::max(maxPatches, amr.patches.size());
        if (frames > 0 && (t >= nextFrame - 1e-12 || t >= tEnd)) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s_%04d", prefix.c_str(),
                          ++frame);
            writeVthb(name, amr);
            nextFrame += tEnd / frames;
        }
    }
    const double wall =
        std::chrono::duration<double>(Clock::now() - t0).count();

    // Final sanity on the restricted coarse field.
    const GridRef c = amr.coarseRef();
    Real rhoMin = Real(1e30), rhoMax = 0;
    for (int j = NG; j < NG + c.ny; ++j)
        for (int i = NG; i < NG + c.nx; ++i) {
            const Real r = toPrim(c.at(i, j)).rho;
            rhoMin = std::min(rhoMin, r);
            rhoMax = std::max(rhoMax, r);
        }

    std::printf("%d steps in %.2f s -> %.1f Mcell-steps/s, max %zu "
                "patches\n",
                steps, wall, cellSteps / wall / 1e6, maxPatches);
    std::printf("rho in [%.4f, %.4f] | mass drift %.2e%s\n",
                double(rhoMin), double(rhoMax),
                std::fabs(amr.totalMass() - m0) / m0,
                s.periodicX && s.periodicY ? " (closed domain)" : "");
    if (frame > 0)
        std::printf("output: %s_0001..%04d (.vthb, ParaView)\n",
                    prefix.c_str(), frame);
    if (rhoMin <= 0 || !std::isfinite(double(rhoMax))) {
        std::fprintf(stderr, "FAIL: unphysical density\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

AmrConfig amrConfigFrom(const Config& cfg) {
    AmrConfig a;
    a.blockC = cfg.getInt("amr.block", 8);
    a.tagThreshold = Real(cfg.getReal("amr.tag_threshold", 0.08));
    a.tagVelocity = Real(cfg.getReal("amr.tag_velocity", 0));
    a.regridEvery = cfg.getInt("amr.regrid_every", 4);
    a.subcycle = cfg.getBool("amr.subcycle", false);
    a.mu = Real(cfg.getReal("mu", 0));
    if (!cfg.getBool("amr.enabled", true))
        a.tagThreshold = Real(1e30); // never tag -> coarse grid only
    return a;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <case.ini>\n  cases: cases/*.ini\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    try {
        const Config cfg = Config::load(argv[1]);
        const std::string caseName = cfg.requireString("case");
        const std::string backend = cfg.getString("backend", "hybrid");
        AmrConfig acfg = amrConfigFrom(cfg);

        const int ny = cfg.getInt("grid.ny", 64);
        if (ny % acfg.blockC != 0)
            throw std::runtime_error(
                "grid.ny must be a multiple of amr.block");

        CaseSetup s = caseGeometry(caseName);
        acfg.periodicX = s.periodicX;
        acfg.periodicY = s.periodicY;
        const int nx = cfg.getInt("grid.nx", s.aspect * ny);
        if (nx % acfg.blockC != 0)
            throw std::runtime_error(
                "grid.nx must be a multiple of amr.block");

        std::printf("case %s | backend %s | grid %dx%d%s%s | mu %g\n",
                    caseName.c_str(), backend.c_str(), nx, ny,
                    cfg.getBool("amr.enabled", true) ? " | AMR" : "",
                    acfg.subcycle ? "+subcycle" : "", double(acfg.mu));

        if (backend == "cpu") {
            Amr2 amr(nx, ny, s.x0, s.y0, s.lx, s.ly, acfg);
            wireCase(caseName, cfg, amr, s);
            return runCase(amr, s, cfg);
        }
        if (backend == "hybrid") {
            MetalContext ctx;
            AmrGpu amr(ctx, nx, ny, s.x0, s.y0, s.lx, s.ly, acfg);
            wireCase(caseName, cfg, amr, s);
            return runCase(amr, s, cfg);
        }
        throw std::runtime_error("unknown backend: " + backend +
                                 " (expected cpu | hybrid)");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

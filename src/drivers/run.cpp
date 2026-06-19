// Generic case runner — the solver is fully driven by the case file:
//   ./build/run cases/dmr.ini            run the case
//   ./build/run --check cases/dmr.ini    parse + dump effective config
//   ./build/run --list                   grammar cheat-sheet
// Domain, states, regions (incl. moving shock fronts), perturbations and
// per-side BCs are all declared in the INI (see src/cases/CaseDef.hpp).
// No per-case C++.

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "cases/CaseDef.hpp"
#include "core/Config.hpp"
#include "io/Checkpoint.hpp"
#include "io/Diagnostics.hpp"
#include "io/VthbWriter.hpp"
#include "render/LiveView.hpp"

#include <chrono>
#include <memory>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using namespace mm;
using Clock = std::chrono::steady_clock;

// Bridge over the 2-level and multi-level interfaces.
template <class AMR>
std::size_t patchTotal(const AMR& amr) {
    if constexpr (requires { amr.patches; }) {
        return amr.patches.size();
    } else {
        std::size_t n = 0;
        for (int l = 1; l < amr.numLevels(); ++l) n += amr.patchCount(l);
        return n;
    }
}
template <class AMR>
void writeFrame(const std::string& base, const AMR& amr) {
    if constexpr (requires { amr.patches; })
        writeVthb(base, amr);
    else
        writeVthbML(base, amr);
}

template <class AMR>
int runCase(AMR& amr, const CaseDef& cd, const Config& cfg) {
    const Real cfl = Real(cfg.getReal("cfl", 0.4));
    const double tEnd = cfg.getReal("t_end", 0.2);
    const int frames = cfg.getInt("output.frames", 4);
    const int every = cfg.getInt("output.every", 0); // >0: 1 frame / K pas
    const int maxSteps = cfg.getInt("output.max_steps", 0); // 0 = all
    const std::string prefix =
        cfg.getString("output.prefix", "out/run");
    std::filesystem::create_directories(
        std::filesystem::path(prefix).parent_path());

    const std::string restart = cfg.getString("restart", "");
    double t = 0;
    const auto ic = [&](Real x, Real y) { return cd.state(x, y, 0); };
    if (restart.empty()) {
        if (cd.species()) {
            if constexpr (requires {
                              amr.init(ic, [](Real, Real) {
                                  return Real(0);
                              });
                          })
                amr.init(ic, [&](Real x, Real y) {
                    return cd.massFraction(x, y, 0);
                });
            else
                throw std::runtime_error(
                    "internal: this AMR class has no two-gas support");
        } else {
            amr.init(ic);
        }
    } else if (cd.species()) {
        throw std::runtime_error(
            "restart with [species] is not supported yet");
    } else if constexpr (requires { amr.patches; }) {
        amr.init(ic);
        t = loadCheckpoint(restart, amr);
        std::printf("restarted from %s at t = %.6f\n", restart.c_str(),
                    t);
    } else {
        throw std::runtime_error(
            "restart requires amr.levels = 2 (multi-level checkpoint is "
            "on the roadmap)");
    }
    const double m0 = amr.totalMass();

    // Run log: integral diagnostics + solver state, every N base steps.
    const int diagEvery = cfg.getInt("diagnostics.every", 0);
    DiagLog log;
    if (diagEvery > 0)
        log.open(cfg.getString("diagnostics.file", prefix + "_log.csv"));

    double nextFrame = frames > 0 ? tEnd / frames : 1e30;
    while (frames > 0 && nextFrame <= t) nextFrame += tEnd / frames;
    int steps = 0, frame = 0;
    std::size_t cellSteps = 0, maxPatches = 0;
    const auto t0 = Clock::now();
    // Live view (hybrid multi-level class only; headless-safe).
    [[maybe_unused]] const bool wantLive = cfg.getBool("render.live", false);
    const int renderEvery = cfg.getInt("render.every", 2);
    std::unique_ptr<LiveView> view;
    if constexpr (requires { amr.renderPoolQ(); }) {
        if (wantLive) {
            // Explicit [render] rho_min/rho_max, else (0/0) the live
            // view auto-scales the color range to the running density
            // extremes — robust when the IC density is uniform (e.g. a
            // detonation ignited by pressure), where a fixed IC-derived
            // range would be degenerate.
            const Real lo = Real(cfg.getReal("render.rho_min", 0));
            const Real hi = Real(cfg.getReal("render.rho_max", 0));
            view = std::make_unique<LiveView>(
                /*ctx*/ amr.context(), amr,
                cfg.getInt("render.scale", 4), "machmallow — espace: pause, q: quitter",
                lo, hi, cfg.getBool("render.grid", true));
            if (!view->ok()) {
                std::printf("note: pas de serveur de fenetres — vue "
                            "live desactivee\n");
                view.reset();
            }
        }
    } else if (wantLive) {
        std::printf("note: render.live demande backend = hybrid\n");
    }

    std::vector<Cons> prevBase; // base snapshot for the residuals
    const auto logRow = [&](double dt) {
        const double wall =
            std::chrono::duration<double>(Clock::now() - t0).count();
        double sm = 0;
        if constexpr (requires { amr.totalSpeciesMass(); })
            if (cd.species()) sm = amr.totalSpeciesMass();
        log.row(steps, t, dt, computeResiduals(amr, prevBase, dt),
                amr.cellCount(), patchTotal(amr),
                computeDiagnostics(amr), sm, wall,
                wall > 0 ? cellSteps / wall / 1e6 : 0);
    };
    if (log.active()) logRow(0); // initial state
    while (t < tEnd * (1 - 1e-9) && (maxSteps == 0 || steps < maxSteps)) {
        Real dt = std::min(amr.maxStableDtAll(cfl), Real(tEnd - t));
        if (steps < 10) dt *= Real(0.3); // gentle start
        if (log.active()) snapshotBase(amr, prevBase);
        amr.step(dt, t);
        t += dt;
        ++steps;
        cellSteps += amr.cellCount();
        maxPatches = std::max(maxPatches, patchTotal(amr));
        if (log.active() &&
            (steps % diagEvery == 0 || t >= tEnd * (1 - 1e-9) ||
             (maxSteps > 0 && steps >= maxSteps)))
            logRow(dt);
        if (view && steps % renderEvery == 0) {
            int st = view->frame();
            while (st == 2) { // paused: keep the window alive
                usleep(30000);
                st = view->frame();
            }
            if (st == 0) {
                std::printf("vue live fermee — arret a t = %.4f\n", t);
                break;
            }
        }
        // Frame output: `output.every = K` (1 image tous les K pas de base,
        // pour une cadence par iteration) a priorite ; sinon `output.frames`
        // (N images espacees dans le temps). Le pas final est toujours ecrit.
        const bool atEnd = t >= tEnd * (1 - 1e-9) ||
                           (maxSteps > 0 && steps >= maxSteps);
        const bool wantFrame =
            every > 0 ? (steps % every == 0 || atEnd)
                      : (frames > 0 && (t >= nextFrame - 1e-12 || atEnd));
        if (wantFrame) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s_%04d", prefix.c_str(),
                          ++frame);
            writeFrame(name, amr);
            if (every <= 0) nextFrame += tEnd / frames;
        }
    }
    const double wall =
        std::chrono::duration<double>(Clock::now() - t0).count();

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
                cd.periodicX && cd.periodicY ? " (closed domain)" : "");
    if (frame > 0)
        std::printf("output: %s_0001..%04d (.vthb, ParaView)\n",
                    prefix.c_str(), frame);

    const std::string ckPath = cfg.getString("output.checkpoint", "");
    if (!ckPath.empty()) {
        if constexpr (requires { amr.patches; }) {
            saveCheckpoint(ckPath, amr, t);
            std::printf("checkpoint: %s (restart = %s to resume)\n",
                        ckPath.c_str(), ckPath.c_str());
        } else {
            std::printf("warning: checkpoint requires amr.levels = 2 — "
                        "skipped\n");
        }
    }
    if (rhoMin <= 0 || !std::isfinite(double(rhoMax))) {
        std::fprintf(stderr, "FAIL: unphysical density\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

AmrConfig amrConfigFrom(const Config& cfg) {
    AmrConfig a;
    a.blockC = cfg.getInt("amr.block", 8);
    a.maxLevels = cfg.getInt("amr.levels", 2);
    a.tagThreshold = Real(cfg.getReal("amr.tag_threshold", 0.08));
    a.tagVelocity = Real(cfg.getReal("amr.tag_velocity", 0));
    a.regridEvery = cfg.getInt("amr.regrid_every", 4);
    a.maxPatches = cfg.getInt("amr.max_patches", 0); // 0 = auto (hybrid)
    a.subcycle = cfg.getBool("amr.subcycle", false);
    a.mu = Real(cfg.getReal("mu", 0));
    if (!cfg.getBool("amr.enabled", true))
        a.tagThreshold = Real(1e30); // never tag -> coarse grid only
    return a;
}

void warnUnusedKeys(const Config& cfg) {
    for (const std::string& k : cfg.unusedKeys())
        std::fprintf(stderr,
                     "warning: config key '%s' was never used (typo?)\n",
                     k.c_str());
}

// Preflight: named states (RH-derived included), numerical sanity
// warnings and a rough cost estimate — printed by --check and before
// every run.
void preflight(const CaseDef& cd, const AmrConfig& acfg, double tEnd,
               double cfl) {
    for (const auto& s : cd.listStates()) {
        std::printf("  state %-10s rho %-9.5g u %-9.5g v %-9.5g p %-9.5g",
                    s.name.c_str(), double(s.w.rho), double(s.w.u),
                    double(s.w.v), double(s.w.p));
        if (s.shockSpeed != 0)
            std::printf("  (front RH, vitesse %.5g)",
                        double(s.shockSpeed));
        std::printf("\n");
    }

    const double dx = double(cd.lx) / cd.nx, dy = double(cd.ly) / cd.ny;
    if (std::fabs(dx / dy - 1) > 1e-6) {
        const int nxSq = int(std::lround(double(cd.lx) / dy));
        std::fprintf(stderr,
                     "warning: non-square cells (dx/dy = %.3g) — "
                     "accuracy is anisotropic; suggested grid.nx = %d\n",
                     dx / dy, nxSq);
    }

    const int depth = acfg.maxLevels - 1;
    const double dxFine = dx / (1 << depth);
    const double smax = double(cd.maxSignalSpeed());
    const double dt0 = acfg.subcycle ? cfl * dx / smax
                                     : cfl * dxFine / smax;
    const double steps = tEnd / dt0;
    // crude cost: assume ~25% of every level refined; level l holds
    // cells*4^l cells and (subcycled) takes 2^l substeps per base step
    const double cells = double(cd.nx) * cd.ny;
    double work = cells * steps;
    for (int l = 1; l <= depth; ++l)
        work += 0.25 * cells * std::pow(4.0, l) * steps *
                (acfg.subcycle ? double(1 << l) : 1.0);
    std::printf("  finest 1/%g | ~%.0f pas de base | ~%.1f Gcell-steps "
                "(si ~25%% raffiné) | ~%.0f s à 150 Mcell/s\n",
                1.0 / dxFine, steps, work / 1e9, work / 150e6);
}

int usage() {
    std::fprintf(stderr,
                 "usage: run <case.ini>            lancer le cas\n"
                 "       run --check <case.ini>    config effective + "
                 "pré-vol, sans calculer\n"
                 "       run --preview <case.ini>  écrire l'IC en .vti "
                 "(vérif géométrie)\n"
                 "       run --list                grammaire des cas\n");
    return EXIT_FAILURE;
}

// Evaluate the IC on the base grid and write it for ParaView — geometry
// check in one second, no solver run.
int preview(const CaseDef& cd, const Config& cfg) {
    const std::string prefix = cfg.getString("output.prefix", "out/run");
    std::filesystem::create_directories(
        std::filesystem::path(prefix).parent_path());
    Grid g(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly);
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            g.at(i, j) = cd.state(g.xc(i), g.yc(j), 0);
    const std::string path = prefix + "_ic.vti";
    writeVti(path, g);
    std::printf("IC écrite dans %s\n", path.c_str());
    return EXIT_SUCCESS;
}

int list() {
    std::printf(
        "Case file grammar (all sections INI, see cases/*.ini):\n"
        "  [domain]   x0 x1 y0 y1\n"
        "  [grid]     nx ny (multiples of amr.block)\n"
        "  [state.X]  rho u v p        named primitive state\n"
        "  [ic]       default = X\n"
        "             region.N  = halfplane a b c [speed s] : X\n"
        "                         (a*x + b*y < c, front moves at normal\n"
        "                          speed s) | band x|y lo hi : X\n"
        "                         | rect x0 x1 y0 y1 : X\n"
        "                         | circle cx cy r : X\n"
        "             perturb.N = u|v|rho|p sin periods amp\n"
        "                         | u|v|rho|p erf x0 width amp\n"
        "                         | u|v|rho|p sing per amp yc sigma\n"
        "                         | p hydro yref (hydrostatique)\n"
        "  [solid]    region.N  = rect|circle|halfplane|band|sinex ...\n"
        "             (corps immergé : meme grammaire geometrique que\n"
        "              [ic], sans etat ; paroi reflechissante. backend\n"
        "              cpu + muscl mono-gaz ; grille de base seule)\n"
        "  [bc]       x|y = periodic\n"
        "             left|right|bottom|top = transmissive | reflective\n"
        "                 | analytic | inflow X\n"
        "                 [if x|y < val else <spec>]\n"
        "             ('analytic' evaluates the time-dependent region\n"
        "              stack at the ghosts: exact moving-shock BCs)\n"
        "  top level  t_end cfl mu backend=cpu|hybrid restart=<ck>\n"
        "  [physics]  gravity = gx gy (source splittée ; murs\n"
        "             reflective hydrostatiques)\n"
        "  [amr]      enabled levels block tag_threshold tag_velocity\n"
        "             regrid_every subcycle max_patches (cap du pool\n"
        "             GPU ; 0 = auto selon la memoire du device)\n"
        "  [output]   frames prefix checkpoint max_steps\n"
        "             every (1 frame tous les K pas ; prioritaire sur\n"
        "             frames, ex. every = 1 = chaque iteration)\n"
        "  [render]   live scale every grid rho_min rho_max (vue\n"
        "             temps reel Metal, backend hybrid ; espace = pause,\n"
        "             q = quitter)\n"
        "  [diagnostics]  every (pas de base, 0 = off) file (csv :\n"
        "             step t dt cells patches extrema masse energies\n"
        "             enstrophie perf)\n");
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string a1 = argv[1];
    if (a1 == "--list") return list();
    const bool checkOnly = a1 == "--check";
    const bool previewOnly = a1 == "--preview";
    if ((checkOnly || previewOnly) && argc < 3) return usage();
    const std::string path =
        (checkOnly || previewOnly) ? argv[2] : a1;

    try {
        const Config cfg = Config::load(path);
        const CaseDef cd = CaseDef::parse(cfg);
        const std::string backend = cfg.getString("backend", "hybrid");
        AmrConfig acfg = amrConfigFrom(cfg);
        acfg.periodicX = cd.periodicX;
        acfg.periodicY = cd.periodicY;
        acfg.gx = cd.gx;
        acfg.gy = cd.gy;
        if (cd.species()) {
            acfg.species = true;
            acfg.gamma1 = cd.gases().gamma1;
            acfg.gamma2 = cd.gases().gamma2;
        }
        if (cd.reacts()) { // single-step reaction (Strang-split source)
            acfg.react = true;
            acfg.reaction = cd.reaction();
        }
        const std::string scheme = cfg.getString("scheme", "muscl");
        if (scheme == "weno5") {
            acfg.weno = true;
        } else if (scheme != "muscl") {
            throw std::runtime_error("unknown scheme: " + scheme +
                                     " (expected muscl | weno5)");
        }

        if (cd.nx % acfg.blockC != 0 || cd.ny % acfg.blockC != 0)
            throw std::runtime_error(
                "grid.nx/ny must be multiples of amr.block");

        // Immersed solids ([solid] regions): the mask is threaded through
        // the solid-aware AMR — base/patch steps, restriction, refluxing,
        // prolongation, boundary tagging — at every level. CPU: Amr2
        // (2 levels) and AmrML (profondeur arbitraire). GPU: AmrGpu
        // (2 levels, lock-step). Bi-gas/WENO and GPU multi-level are not
        // ported yet — guard explicitly. The body boundary auto-refines.
        if (cd.hasSolids()) {
            if (acfg.species || acfg.weno)
                throw std::runtime_error(
                    "solides immergés : scheme = muscl mono-gaz requis "
                    "(bi-gaz / WENO à venir)");
        }

        std::printf("case %s | backend %s | scheme %s | grid %dx%d | domain "
                    "[%g,%g]x[%g,%g]%s%s | levels %d | mu %g\n",
                    path.c_str(), backend.c_str(), scheme.c_str(),
                    cd.nx, cd.ny,
                    double(cd.x0), double(cd.x0 + cd.lx), double(cd.y0),
                    double(cd.y0 + cd.ly),
                    cd.periodicX ? " | periodicX" : "",
                    cd.periodicY ? " | periodicY" : "", acfg.maxLevels,
                    double(acfg.mu));

        if (previewOnly) return preview(cd, cfg);

        preflight(cd, acfg, cfg.getReal("t_end", 0.2),
                  cfg.getReal("cfl", 0.4));
        if (checkOnly) {
            std::printf("t_end %g | cfl %g | frames %d | tag %g/%g | "
                        "regrid %d | subcycle %d\n",
                        cfg.getReal("t_end", 0.2), cfg.getReal("cfl", 0.4),
                        cfg.getInt("output.frames", 4),
                        double(acfg.tagThreshold),
                        double(acfg.tagVelocity), acfg.regridEvery,
                        int(acfg.subcycle));
            cfg.getString("output.prefix", "");
            cfg.getString("output.checkpoint", "");
            cfg.getInt("output.max_steps", 0);
            cfg.getInt("output.every", 0);
            cfg.getString("restart", "");
            cfg.getInt("diagnostics.every", 0);
            cfg.getString("diagnostics.file", "");
            warnUnusedKeys(cfg);
            std::printf("config OK\n");
            return EXIT_SUCCESS;
        }

        int rc;
        if (backend == "cpu") {
            // the 2-level classes have no species fields: two-gas
            // cases run on the multi-level classes at any depth
            if (acfg.maxLevels == 2 && !acfg.species && !acfg.weno) {
                Amr2 amr(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly, acfg);
                amr.fillPhysicalGhosts = [&cd](Grid& g, double t) {
                    cd.fillGhosts(g, t);
                };
                amr.fillPatchPhysical = [&cd](Grid& g, double t,
                                              unsigned s) {
                    cd.fillGhostSides(g, t, s);
                };
                if (cd.hasSolids())
                    amr.solidAt = [&cd](Real x, Real y) {
                        return cd.solidAt(x, y);
                    };
                rc = runCase(amr, cd, cfg);
            } else {
                AmrML amr(cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly, acfg);
                amr.fillPhysicalGhosts = [&cd](Grid& g, double t) {
                    cd.fillGhosts(g, t);
                };
                amr.fillPatchPhysical = [&cd](Grid& g, double t,
                                              unsigned s) {
                    cd.fillGhostSides(g, t, s);
                };
                if (cd.hasSolids())
                    amr.solidAt = [&cd](Real x, Real y) {
                        return cd.solidAt(x, y);
                    };
                rc = runCase(amr, cd, cfg);
            }
        } else if (backend == "hybrid") {
            MetalContext ctx;
            if (acfg.maxLevels == 2 && !acfg.species && !acfg.weno) {
                AmrGpu amr(ctx, cd.nx, cd.ny, cd.x0, cd.y0, cd.lx, cd.ly,
                           acfg);
                amr.fillPhysicalGhosts = [&cd](GridRef& g, double t) {
                    cd.fillGhosts(g, t);
                };
                amr.fillPatchPhysical = [&cd](GridRef& g, double t,
                                              unsigned s) {
                    cd.fillGhostSides(g, t, s);
                };
                if (cd.hasSolids())
                    amr.solidAt = [&cd](Real x, Real y) {
                        return cd.solidAt(x, y);
                    };
                rc = runCase(amr, cd, cfg);
            } else {
                AmrGpuML amr(ctx, cd.nx, cd.ny, cd.x0, cd.y0, cd.lx,
                             cd.ly, acfg);
                amr.fillPhysicalGhosts = [&cd](GridRef& g, double t) {
                    cd.fillGhosts(g, t);
                };
                amr.fillPatchPhysical = [&cd](GridRef& g, double t,
                                              unsigned s) {
                    cd.fillGhostSides(g, t, s);
                };
                if (cd.hasSolids())
                    amr.solidAt = [&cd](Real x, Real y) {
                        return cd.solidAt(x, y);
                    };
                rc = runCase(amr, cd, cfg);
            }
        } else {
            throw std::runtime_error("unknown backend: " + backend +
                                     " (expected cpu | hybrid)");
        }
        warnUnusedKeys(cfg);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

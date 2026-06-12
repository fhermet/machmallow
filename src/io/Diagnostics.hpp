#pragma once

// Composite integral diagnostics over an AMR hierarchy (covered cells
// excluded, finest data wins): mass, kinetic and total energy,
// enstrophy, density/pressure extrema. Works on both the 2-level
// classes (Amr2/AmrGpu) and the multi-level ones (AmrML/AmrGpuML).

#include "core/Grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mm {

struct DiagRow {
    double mass = 0, kinetic = 0, energy = 0, enstrophy = 0;
    double rhoMin = std::numeric_limits<double>::max(), rhoMax = 0;
    double pMin = std::numeric_limits<double>::max(), pMax = 0;
};

namespace diagdetail {

// Accumulate one grid's interior; covered(i, j) masks cells refined by
// a finer level (local interior indices, NG excluded).
template <class Covered>
inline void addGrid(DiagRow& d, const GridRef& g, Covered&& covered) {
    const double dA = double(g.dx) * g.dy;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            if (covered(i, j)) continue;
            const Prim w = toPrim(g.at(NG + i, NG + j));
            d.mass += double(w.rho) * dA;
            d.kinetic += 0.5 * double(w.rho) *
                         (double(w.u) * w.u + double(w.v) * w.v) * dA;
            d.energy += double(g.at(NG + i, NG + j).E) * dA;
            d.rhoMin = std::min(d.rhoMin, double(w.rho));
            d.rhoMax = std::max(d.rhoMax, double(w.rho));
            d.pMin = std::min(d.pMin, double(w.p));
            d.pMax = std::max(d.pMax, double(w.p));

            // vorticity via central differences clamped to the interior
            // (a diagnostic, not a scheme: edge bias is acceptable)
            const auto vel = [&](int a, int b, bool getU) {
                const Cons& q = g.at(NG + a, NG + b);
                const Real rho = std::max(q.rho, RHO_FLOOR);
                return double(getU ? q.mx / rho : q.my / rho);
            };
            const int ip = std::min(i + 1, g.nx - 1),
                      im = std::max(i - 1, 0);
            const int jp = std::min(j + 1, g.ny - 1),
                      jm = std::max(j - 1, 0);
            const double dvdx = (vel(ip, j, false) - vel(im, j, false)) /
                                ((ip - im) * double(g.dx));
            const double dudy = (vel(i, jp, true) - vel(i, jm, true)) /
                                ((jp - jm) * double(g.dy));
            const double om = dvdx - dudy;
            d.enstrophy += 0.5 * om * om * dA;
        }
}

} // namespace diagdetail

template <class AMR>
DiagRow computeDiagnostics(const AMR& amr) {
    DiagRow d;
    const int bC = amr.fineCells() / 2;
    if constexpr (requires { amr.patches; }) {
        // 2-level classes
        diagdetail::addGrid(d, amr.coarseRef(), [&](int i, int j) {
            return amr.covered(i / bC, j / bC);
        });
        for (const auto& p : amr.patches)
            diagdetail::addGrid(d, amr.patchRef(p),
                                [](int, int) { return false; });
    } else {
        // multi-level classes
        diagdetail::addGrid(d, amr.coarseRef(), [&](int i, int j) {
            return amr.numLevels() > 1 &&
                   amr.covered(1, i / bC, j / bC);
        });
        for (int l = 1; l < amr.numLevels(); ++l)
            for (const auto& p : amr.level(l).patches) {
                const int ci0 = p.ci0, cj0 = p.cj0;
                diagdetail::addGrid(
                    d, amr.patchRef(l, p), [&](int i, int j) {
                        if (l >= amr.numLevels() - 1) return false;
                        const int gi = 2 * ci0 + i, gj = 2 * cj0 + j;
                        return amr.covered(l + 1, gi / bC, gj / bC);
                    });
            }
    }
    return d;
}

// Per-equation residuals, industrial-CFD style: RMS of the discrete
// time derivative (U^{n+1} - U^n)/dt over the base grid (which carries
// the restricted composite, so this is regrid-stable). For a case that
// approaches a steady state they drop by orders of magnitude; for an
// unsteady flow they plateau at the physical activity level.
struct Residuals {
    double rho = 0, mx = 0, my = 0, E = 0;
};

template <class AMR>
void snapshotBase(const AMR& amr, std::vector<Cons>& out) {
    const GridRef b = amr.coarseRef();
    out.assign(b.q, b.q + std::size_t(b.totx()) * b.toty());
}

template <class AMR>
Residuals computeResiduals(const AMR& amr, const std::vector<Cons>& prev,
                           double dt) {
    Residuals r;
    if (dt <= 0 || prev.empty()) return r;
    const GridRef b = amr.coarseRef();
    for (int j = NG; j < NG + b.ny; ++j)
        for (int i = NG; i < NG + b.nx; ++i) {
            const std::size_t id = b.idx(i, j);
            const Cons& q = b.q[id];
            const Cons& p = prev[id];
            r.rho += double(q.rho - p.rho) * (q.rho - p.rho);
            r.mx += double(q.mx - p.mx) * (q.mx - p.mx);
            r.my += double(q.my - p.my) * (q.my - p.my);
            r.E += double(q.E - p.E) * (q.E - p.E);
        }
    const double n = double(b.nx) * b.ny * dt * dt;
    r.rho = std::sqrt(r.rho / n);
    r.mx = std::sqrt(r.mx / n);
    r.my = std::sqrt(r.my / n);
    r.E = std::sqrt(r.E / n);
    return r;
}

// CSV time-series writer for the per-run log file.
class DiagLog {
public:
    DiagLog() = default;
    void open(const std::string& path) {
        f_ = std::fopen(path.c_str(), "w");
        if (f_ == nullptr)
            throw std::runtime_error("cannot write log: " + path);
        std::fprintf(f_,
                     "step,t,dt,res_mass,res_momx,res_momy,res_energy,"
                     "cells,patches,rho_min,rho_max,p_min,p_max,mass,"
                     "kinetic_energy,total_energy,enstrophy,wall_s,"
                     "mcells_per_s\n");
    }
    bool active() const { return f_ != nullptr; }

    void row(int step, double t, double dt, const Residuals& r,
             std::size_t cells, std::size_t patches, const DiagRow& d,
             double wall, double mcells) {
        std::fprintf(f_,
                     "%d,%.9g,%.6g,%.6g,%.6g,%.6g,%.6g,%zu,%zu,%.6g,"
                     "%.6g,%.6g,%.6g,%.9g,%.9g,%.9g,%.9g,%.3f,%.1f\n",
                     step, t, dt, r.rho, r.mx, r.my, r.E, cells, patches,
                     d.rhoMin, d.rhoMax, d.pMin, d.pMax, d.mass,
                     d.kinetic, d.energy, d.enstrophy, wall, mcells);
        std::fflush(f_); // a crash should not lose the series
    }

    ~DiagLog() {
        if (f_ != nullptr) std::fclose(f_);
    }
    DiagLog(const DiagLog&) = delete;
    DiagLog& operator=(const DiagLog&) = delete;

private:
    FILE* f_ = nullptr;
};

} // namespace mm

// Cut-cell / embedded-boundary finite-volume solver.
//   phase 2: inviscid 1st-order Godunov + hybrid flux redistribution.
//   phase 3: 2nd order — least-squares primitive gradients (Barth-Jespersen
//            limited) reconstructed to the face centres and the embedded-
//            boundary centroid, advanced with SSP-RK2.
// step2D (the Cartesian solver) is left untouched.
#pragma once

#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"
#include "numerics/Hllc.hpp"
#include "physics/Euler.hpp"
#include "solver/Muscl2D.hpp"   // hllcFluxY, maxStableDt

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace mm {

// slip-wall flux through an EB face whose OUTWARD-from-fluid normal is
// (nox,noy) = -n_EB (n_EB points solid->fluid).
inline Cons ebWallFlux(const Prim& w, Real nox, Real noy) {
    const Real un = w.u * nox + w.v * noy;      // fluid velocity into the wall
    const Real ps = wallPressure(w, un);        // exact slip-wall pressure
    return Cons{0, ps * nox, ps * noy, 0};
}

// primitive gradient of a cell: dWdx / dWdy stored as Prim components.
struct PrimGrad {
    Prim dx{0, 0, 0, 0}, dy{0, 0, 0, 0};
};

// linear reconstruction of W at an offset (ox,oy) from the cell centre.
inline Prim reconstruct(const Prim& w, const PrimGrad& g, Real ox, Real oy) {
    return Prim{w.rho + g.dx.rho * ox + g.dy.rho * oy,
                w.u + g.dx.u * ox + g.dy.u * oy,
                w.v + g.dx.v * ox + g.dy.v * oy,
                w.p + g.dx.p * ox + g.dy.p * oy};
}

// least-squares gradients (fluid 3x3 neighbourhood), Barth-Jespersen limited.
// `limited=false` returns the raw LSQ gradients (for smooth order verification;
// the limiter clips smooth extrema down to ~1st order, the usual TVD 4/3 cap).
inline std::vector<PrimGrad> lsqGradients(const Grid& g,
                                          const cutcell::Geometry& geo,
                                          bool limited = true) {
    const Real tiny = Real(1e-9);
    std::vector<PrimGrad> grad(g.q.size());
    const auto kap = [&](int i, int j) { return double(geo.at(i, j).vol); };
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            if (kap(i, j) <= double(tiny)) continue;
            const Prim wc = toPrim(g.at(i, j));
            double Sxx = 0, Sxy = 0, Syy = 0;
            double bx[4] = {0, 0, 0, 0}, by[4] = {0, 0, 0, 0};
            Prim wlo = wc, whi = wc;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    if ((di == 0 && dj == 0) || kap(i + di, j + dj) <= double(tiny))
                        continue;
                    const double dxk = double(g.xc(i + di)) - double(g.xc(i));
                    const double dyk = double(g.yc(j + dj)) - double(g.yc(j));
                    const double w = 1.0 / (dxk * dxk + dyk * dyk);
                    Sxx += w * dxk * dxk;
                    Sxy += w * dxk * dyk;
                    Syy += w * dyk * dyk;
                    const Prim wk = toPrim(g.at(i + di, j + dj));
                    const double dW[4] = {double(wk.rho) - double(wc.rho),
                                          double(wk.u) - double(wc.u),
                                          double(wk.v) - double(wc.v),
                                          double(wk.p) - double(wc.p)};
                    for (int c = 0; c < 4; ++c) {
                        bx[c] += w * dxk * dW[c];
                        by[c] += w * dyk * dW[c];
                    }
                    const Real* pk = &wk.rho;
                    Real* plo = &wlo.rho;
                    Real* phi = &whi.rho;
                    for (int c = 0; c < 4; ++c) {
                        plo[c] = std::min(plo[c], pk[c]);
                        phi[c] = std::max(phi[c], pk[c]);
                    }
                }
            const double det = Sxx * Syy - Sxy * Sxy;
            PrimGrad gr;
            if (det > 1e-30) {
                Real* gdx = &gr.dx.rho;
                Real* gdy = &gr.dy.rho;
                const Real* wcp = &wc.rho;
                const Real* wlop = &wlo.rho;
                const Real* whip = &whi.rho;
                for (int c = 0; c < 4; ++c) {
                    double gx = (Syy * bx[c] - Sxy * by[c]) / det;
                    double gy = (Sxx * by[c] - Sxy * bx[c]) / det;
                    // Barth-Jespersen limiter over the 3x3 neighbour offsets
                    double phi = 1.0;
                    for (int dj = -1; dj <= 1 && limited; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            if ((di == 0 && dj == 0) ||
                                kap(i + di, j + dj) <= double(tiny))
                                continue;
                            const double ox =
                                double(g.xc(i + di)) - double(g.xc(i));
                            const double oy =
                                double(g.yc(j + dj)) - double(g.yc(j));
                            const double d = gx * ox + gy * oy;
                            double lim = 1.0;
                            if (d > 1e-30)
                                lim = std::min(1.0, (double(whip[c]) -
                                                     double(wcp[c])) / d);
                            else if (d < -1e-30)
                                lim = std::min(1.0, (double(wlop[c]) -
                                                     double(wcp[c])) / d);
                            phi = std::min(phi, lim);
                        }
                    gdx[c] = Real(phi * gx);
                    gdy[c] = Real(phi * gy);
                }
            }
            grad[g.idx(i, j)] = gr;
        }
    return grad;
}

// FRD'd conservative divergence D (dU/dt) for the whole grid; ghosts must be
// filled by the caller. Uses 2nd-order reconstruction of the face / EB states.
inline std::vector<Cons> cutCellDiv(const Grid& g,
                                    const cutcell::Geometry& geo,
                                    bool limited = true) {
    const Real dx = g.dx, dy = g.dy, V = dx * dy;
    const Real tiny = Real(1e-9);
    const Real hx = Real(0.5) * dx, hy = Real(0.5) * dy;
    const auto grad = lsqGradients(g, geo, limited);
    const auto kap = [&](int i, int j) { return double(geo.at(i, j).vol); };
    std::vector<Cons> dU(g.q.size(), Cons{0, 0, 0, 0});

    // x-faces (boundary faces use constant reconstruction so the ghost flux
    // stays symmetric — otherwise a nonzero-gradient interior cell against a
    // zero-gradient ghost would leak mass at the domain wall)
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG - 1; i < NG + g.nx; ++i) {
            const Real a = geo.at(i, j).apXhi;
            if (a <= tiny) continue;
            const Real ox = (i < NG || i + 1 >= NG + g.nx) ? Real(0) : hx;
            const Prim wL = reconstruct(toPrim(g.at(i, j)),
                                        grad[g.idx(i, j)], ox, 0);
            const Prim wR = reconstruct(toPrim(g.at(i + 1, j)),
                                        grad[g.idx(i + 1, j)], -ox, 0);
            const Cons f = (a * dy) * hllcFluxX(wL, wR);
            dU[g.idx(i, j)] = dU[g.idx(i, j)] - f;
            dU[g.idx(i + 1, j)] = dU[g.idx(i + 1, j)] + f;
        }
    // y-faces
    for (int i = NG; i < NG + g.nx; ++i)
        for (int j = NG - 1; j < NG + g.ny; ++j) {
            const Real a = geo.at(i, j).apYhi;
            if (a <= tiny) continue;
            const Real oy = (j < NG || j + 1 >= NG + g.ny) ? Real(0) : hy;
            const Prim wB = reconstruct(toPrim(g.at(i, j)),
                                        grad[g.idx(i, j)], 0, oy);
            const Prim wT = reconstruct(toPrim(g.at(i, j + 1)),
                                        grad[g.idx(i, j + 1)], 0, -oy);
            const Cons f = (a * dx) * hllcFluxY(wB, wT);
            dU[g.idx(i, j)] = dU[g.idx(i, j)] - f;
            dU[g.idx(i, j + 1)] = dU[g.idx(i, j + 1)] + f;
        }
    // embedded-boundary (wall) flux, reconstructed to the EB centroid
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const CellMoments& m = geo.at(i, j);
            if (m.vol <= tiny || m.vol >= Real(1) - tiny || m.eb.area <= tiny)
                continue;
            const Real ox = m.eb.cx - g.xc(i), oy = m.eb.cy - g.yc(j);
            const Prim w = reconstruct(toPrim(g.at(i, j)), grad[g.idx(i, j)],
                                       ox, oy);
            dU[g.idx(i, j)] =
                dU[g.idx(i, j)] - m.eb.area * ebWallFlux(w, -m.eb.nx, -m.eb.ny);
        }

    // conservative divergence D^c = dU/(kappa V)
    std::vector<Cons> Dc(g.q.size(), Cons{0, 0, 0, 0});
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const double k = kap(i, j);
            if (k > double(tiny))
                Dc[g.idx(i, j)] = Real(1.0 / (k * V)) * dU[g.idx(i, j)];
        }
    // hybrid divergence + mass redistribution over the 3x3 fluid neighbourhood
    std::vector<Cons> D(g.q.size(), Cons{0, 0, 0, 0});
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            const double kc = kap(i, j);
            if (kc <= double(tiny)) continue;
            double sumk = 0;
            Cons wsum{0, 0, 0, 0};
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const double kk = kap(i + di, j + dj);
                    if (kk > double(tiny)) {
                        sumk += kk;
                        wsum = wsum + Real(kk) * Dc[g.idx(i + di, j + dj)];
                    }
                }
            const Cons Dnc = Real(1.0 / sumk) * wsum;
            D[g.idx(i, j)] = D[g.idx(i, j)] +
                             (Real(kc) * Dc[g.idx(i, j)] + Real(1 - kc) * Dnc);
            if (kc < 1.0 - double(tiny)) {
                const Cons dD =
                    Real(kc * (1 - kc) / sumk) * (Dc[g.idx(i, j)] - Dnc);
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di)
                        if (kap(i + di, j + dj) > double(tiny))
                            D[g.idx(i + di, j + dj)] =
                                D[g.idx(i + di, j + dj)] + dD;
            }
        }
    return D;
}

// positivity-preserving state (resets near-vacuum / runaway slivers).
inline Cons floorState(const Cons& q) {
    Prim w = toPrim(q);
    const Real SPEED_CAP = Real(50);
    const bool bad = !(w.rho >= RHO_FLOOR) || !std::isfinite(double(w.p)) ||
                     !std::isfinite(double(w.u)) ||
                     !std::isfinite(double(w.v)) ||
                     std::hypot(w.u, w.v) > SPEED_CAP;
    if (bad) { w.rho = RHO_FLOOR; w.u = 0; w.v = 0; w.p = P_FLOOR; return toCons(w); }
    if (w.p < P_FLOOR) { w.p = P_FLOOR; return toCons(w); }
    return q;
}

// One SSP-RK2 cut-cell step. `fill(Grid&)` applies the boundary conditions;
// it is called before each stage (covered cells stay frozen).
template <class Fill>
inline void stepCutCell(Grid& g, const cutcell::Geometry& geo, Real dt,
                        Fill fill, bool limited = true) {
    const Real tiny = Real(1e-9);
    const auto kap = [&](int i, int j) { return double(geo.at(i, j).vol); };

    fill(g);
    const std::vector<Cons> D1 = cutCellDiv(g, geo, limited);
    Grid g1 = g;                                   // stage-1 state (U1)
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            if (kap(i, j) > double(tiny))
                g1.at(i, j) = floorState(g.at(i, j) + dt * D1[g.idx(i, j)]);

    fill(g1);
    const std::vector<Cons> D2 = cutCellDiv(g1, geo, limited);
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            if (kap(i, j) > double(tiny)) {
                const Cons un = g.at(i, j);
                const Cons u1 = g1.at(i, j);
                g.at(i, j) = floorState(Real(0.5) * un +
                                        Real(0.5) * (u1 + dt * D2[g.idx(i, j)]));
            }
}

} // namespace mm

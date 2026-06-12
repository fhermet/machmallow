#pragma once

#include "core/Types.hpp"
#include "physics/Euler.hpp"

#include <vector>

namespace mm {

// Ghost layers on each side (2 needed: limited slopes + Riemann stencil).
// 3 ghost layers: WENO5 needs a 5-cell stencil (r = 3); the MUSCL
// paths only read 2 but every ghost fill covers the full ring.
inline constexpr int NG = 3;

// Non-owning view over externally allocated cells (e.g. a shared Metal
// buffer). Same interface as Grid so case setup/BC code is shared.
struct GridRef {
    int nx = 0, ny = 0;
    Real x0 = 0, y0 = 0;
    Real dx = 0, dy = 0;
    Cons* q = nullptr;

    int totx() const { return nx + 2 * NG; }
    int toty() const { return ny + 2 * NG; }
    std::size_t idx(int i, int j) const {
        return std::size_t(j) * totx() + i;
    }
    Cons& at(int i, int j) const { return q[idx(i, j)]; }
    Real xc(int i) const { return x0 + (Real(i - NG) + Real(0.5)) * dx; }
    Real yc(int j) const { return y0 + (Real(j - NG) + Real(0.5)) * dy; }
};

// Uniform 2D grid of conserved states, row-major, ghosts included.
struct Grid {
    int nx = 0, ny = 0; // interior cells
    Real x0 = 0, y0 = 0;
    Real dx = 0, dy = 0;
    std::vector<Cons> q;

    Grid(int nx_, int ny_, Real x0_, Real y0_, Real lx, Real ly)
        : nx(nx_), ny(ny_), x0(x0_), y0(y0_), dx(lx / nx_), dy(ly / ny_),
          q(std::size_t(nx_ + 2 * NG) * (ny_ + 2 * NG)) {}

    int totx() const { return nx + 2 * NG; }
    int toty() const { return ny + 2 * NG; }
    std::size_t idx(int i, int j) const {
        return std::size_t(j) * totx() + i;
    }
    Cons& at(int i, int j) { return q[idx(i, j)]; }
    const Cons& at(int i, int j) const { return q[idx(i, j)]; }

    // Cell centers; i, j include ghosts (interior starts at NG).
    Real xc(int i) const { return x0 + (Real(i - NG) + Real(0.5)) * dx; }
    Real yc(int j) const { return y0 + (Real(j - NG) + Real(0.5)) * dy; }
};

} // namespace mm

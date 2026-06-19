#pragma once

// Aerodynamic force on an immersed body by surface-pressure integration.
// The fluid exerts a force on the solid equal to the sum, over every
// fluid↔solid face, of p · (face length) · (normal pointing into the body).
// On the Cartesian staircase the normals are axis-aligned, so each wetted
// x-face contributes to fx (drag, +x) and each y-face to fy (normal/lift,
// +y). We use the adjacent fluid cell's pressure — the conventional CFD
// surface pressure (what a wall pressure tap measures). Viscous skin
// friction is not included (pressure/form drag only).

#include "core/Grid.hpp"
#include "physics/Euler.hpp"

#include <cstdint>

namespace mm {

struct WallForce {
    double fx = 0; // drag  (force on the body in +x)
    double fy = 0; // normal / lift (force on the body in +y)
};

template <class G>
inline WallForce wallForce(const G& g, const std::uint8_t* solid) {
    WallForce f;
    const auto sol = [&](int i, int j) { return solid[g.idx(i, j)] != 0; };
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i) {
            if (sol(i, j)) continue; // only fluid cells push on the body
            const double p = double(toPrim(g.at(i, j)).p);
            if (sol(i + 1, j)) f.fx += p * double(g.dy); // wall on +x
            if (sol(i - 1, j)) f.fx -= p * double(g.dy); // wall on -x
            if (sol(i, j + 1)) f.fy += p * double(g.dx); // wall on +y
            if (sol(i, j - 1)) f.fy -= p * double(g.dx); // wall on -y
        }
    return f;
}

} // namespace mm

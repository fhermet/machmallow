#pragma once

// Double Mach reflection case definition (Woodward & Colella 1984),
// shared by the CPU and GPU drivers. Domain [0,4]x[0,1], reflecting wall
// starting at x = 1/6 on the bottom, Mach 10 shock at 60 degrees.

#include "core/Boundary.hpp"
#include "core/Grid.hpp"

#include <cmath>

namespace mm::dmr {

inline constexpr double XWALL = 1.0 / 6.0;
inline constexpr double TEND = 0.2;

// Pre-shock (quiescent) and post-shock (Mach 10, 60 deg) states.
inline const Cons PRE = toCons({Real(1.4), 0, 0, Real(1.0)});
inline const Cons POST =
    toCons({Real(8.0), Real(8.25 * std::sqrt(3.0) / 2.0), Real(-8.25 * 0.5),
            Real(116.5)});

inline bool behindShock(double x, double y, double t) {
    // Shock line through (XWALL, 0) at 60 deg, normal speed 10.
    return x < XWALL + (y + 20.0 * t) / std::sqrt(3.0);
}

template <class G>
void init(G& g) {
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            g.at(i, j) = behindShock(g.xc(i), g.yc(j), 0) ? POST : PRE;
}

template <class G>
void fillGhosts(G& g, double t) {
    // Left: post-shock inflow.
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi) g.at(gi, j) = POST;
    fillTransmissiveRight(g);

    // Bottom: post-shock inflow ahead of the wall start, reflecting wall
    // after it.
    fillReflectiveBottom(g);
    for (int i = 0; i < g.totx(); ++i)
        if (g.xc(i) < XWALL)
            for (int gj = 0; gj < NG; ++gj) g.at(i, gj) = POST;

    // Top: exact shock position (time dependent).
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj) {
            const int j = NG + g.ny + gj;
            g.at(i, j) = behindShock(g.xc(i), g.yc(j), t) ? POST : PRE;
        }
}

} // namespace mm::dmr

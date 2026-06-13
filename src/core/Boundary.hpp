#pragma once

#include "core/Grid.hpp"

namespace mm {

// Domain-side bitmask, used to tell patch-level BC callbacks which of a
// patch's sides touch the physical boundary.
enum : unsigned {
    SideLeft = 1u,
    SideRight = 2u,
    SideBottom = 4u,
    SideTop = 8u,
};

// Zero-gradient (transmissive) ghost fill, one side at a time.
template <class G>
inline void fillTransmissiveLeft(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi) g.at(gi, j) = g.at(NG, j);
}
template <class G>
inline void fillTransmissiveRight(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi)
            g.at(NG + g.nx + gi, j) = g.at(NG + g.nx - 1, j);
}
template <class G>
inline void fillTransmissiveBottom(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj) g.at(i, gj) = g.at(i, NG);
}
template <class G>
inline void fillTransmissiveTop(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj)
            g.at(i, NG + g.ny + gj) = g.at(i, NG + g.ny - 1);
}

// Periodic wrap: ghost columns/rows copy the opposite interior.
template <class G>
inline void fillPeriodicX(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            g.at(k, j) = g.at(k + g.nx, j);
            g.at(NG + g.nx + k, j) = g.at(NG + k, j);
        }
}
template <class G>
inline void fillPeriodicY(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            g.at(i, k) = g.at(i, k + g.ny);
            g.at(i, NG + g.ny + k) = g.at(i, NG + k);
        }
}

// Reflecting wall: mirror state, flip normal momentum.
template <class G>
inline void fillReflectiveLeft(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(NG + k, j);
            c.mx = -c.mx;
            g.at(NG - 1 - k, j) = c;
        }
}
template <class G>
inline void fillReflectiveRight(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(NG + g.nx - 1 - k, j);
            c.mx = -c.mx;
            g.at(NG + g.nx + k, j) = c;
        }
}
template <class G>
inline void fillReflectiveBottom(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + k);
            c.my = -c.my;
            g.at(i, NG - 1 - k) = c;
        }
}
template <class G>
inline void fillReflectiveTop(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + g.ny - 1 - k);
            c.my = -c.my;
            g.at(i, NG + g.ny + k) = c;
        }
}

// No-slip wall: mirror the state but flip BOTH momentum components, so
// the reconstructed wall velocity is zero (the viscous shear at the
// wall comes from the resulting one-sided gradient). Density and energy
// are mirrored evenly -> zero normal temperature gradient -> adiabatic
// wall. Only meaningful in viscous (mu > 0) runs.
template <class G>
inline void fillNoSlipLeft(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(NG + k, j);
            c.mx = -c.mx; c.my = -c.my;
            g.at(NG - 1 - k, j) = c;
        }
}
template <class G>
inline void fillNoSlipRight(G& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(NG + g.nx - 1 - k, j);
            c.mx = -c.mx; c.my = -c.my;
            g.at(NG + g.nx + k, j) = c;
        }
}
template <class G>
inline void fillNoSlipBottom(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + k);
            c.mx = -c.mx; c.my = -c.my;
            g.at(i, NG - 1 - k) = c;
        }
}
template <class G>
inline void fillNoSlipTop(G& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + g.ny - 1 - k);
            c.mx = -c.mx; c.my = -c.my;
            g.at(i, NG + g.ny + k) = c;
        }
}

} // namespace mm

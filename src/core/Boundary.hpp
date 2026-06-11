#pragma once

#include "core/Grid.hpp"

namespace mm {

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

// Reflecting wall: mirror state, flip normal momentum.
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

} // namespace mm

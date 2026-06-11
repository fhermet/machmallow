#pragma once

#include "core/Grid.hpp"

namespace mm {

// Zero-gradient (transmissive) ghost fill, one side at a time.
inline void fillTransmissiveLeft(Grid& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi) g.at(gi, j) = g.at(NG, j);
}
inline void fillTransmissiveRight(Grid& g) {
    for (int j = 0; j < g.toty(); ++j)
        for (int gi = 0; gi < NG; ++gi)
            g.at(NG + g.nx + gi, j) = g.at(NG + g.nx - 1, j);
}
inline void fillTransmissiveBottom(Grid& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj) g.at(i, gj) = g.at(i, NG);
}
inline void fillTransmissiveTop(Grid& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int gj = 0; gj < NG; ++gj)
            g.at(i, NG + g.ny + gj) = g.at(i, NG + g.ny - 1);
}

// Reflecting wall: mirror state, flip normal momentum.
inline void fillReflectiveBottom(Grid& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + k);
            c.my = -c.my;
            g.at(i, NG - 1 - k) = c;
        }
}
inline void fillReflectiveTop(Grid& g) {
    for (int i = 0; i < g.totx(); ++i)
        for (int k = 0; k < NG; ++k) {
            Cons c = g.at(i, NG + g.ny - 1 - k);
            c.my = -c.my;
            g.at(i, NG + g.ny + k) = c;
        }
}

} // namespace mm

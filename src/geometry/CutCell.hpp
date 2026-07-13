// Cut-cell / embedded-boundary geometry (phase 1: moments only).
//
// For each Cartesian cell we compute the "EB moments" a cut-cell finite-volume
// scheme needs:
//   - vol   : fluid volume fraction Lambda in [0,1]  (1 = full fluid, 0 = solid)
//   - apXlo/apXhi/apYlo/apYhi : face apertures (fraction of each cell face open
//             to fluid) in [0,1]
//   - eb    : the embedded-boundary face (the solid/fluid interface segment),
//             its length (area in 2D), outward-into-fluid normal and centroid.
//
// The boundary is analytic (phase 1: circle and half-plane -> cylinder and
// wedge). Volume fraction and apertures are computed in closed form; the EB
// face area & normal are DERIVED from the divergence (Gauss) closure
//     A_EB n_EB = sum_faces alpha_f A_f n_f ,
// which makes the resulting finite-volume operator conservative by
// construction. All geometry math is done in double (fp32 solver reads Real).
#pragma once

#include "core/Grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mm {

struct EBFace {
    Real area = 0;          // length of the interface segment (2D "area")
    Real nx = 0, ny = 0;    // unit normal, pointing solid -> fluid
    Real cx = 0, cy = 0;    // centroid (placeholder in phase 1; used at 2nd order)
};

struct CellMoments {
    Real vol = 1;                            // fluid volume fraction Lambda
    Real apXlo = 1, apXhi = 1;               // face apertures (fluid fraction)
    Real apYlo = 1, apYhi = 1;
    EBFace eb;                               // valid when 0 < vol < 1
};

namespace cutcell {

// ---- exact circle (solid = inside the disk) -----------------------------

// antiderivative of sqrt(r^2 - x^2)
inline double aint(double x, double r) {
    x = std::clamp(x, -r, r);
    return 0.5 * (x * std::sqrt(std::max(r * r - x * x, 0.0)) +
                  r * r * std::asin(x / r));
}

// area of the disk (radius r, centred at origin) intersected with the box
// [X0,X1] x [Y0,Y1] (box coords already relative to the disk centre).
inline double diskBoxArea(double r, double X0, double X1, double Y0,
                          double Y1) {
    const double xa = std::max(X0, -r), xb = std::min(X1, r);
    if (xb <= xa) return 0.0;
    // breakpoints wherever s = sqrt(r^2-x^2) crosses one of Y0, Y1, -Y0, -Y1,
    // i.e. wherever min(Y1,s) or max(Y0,-s) switches branch (or the overlap
    // appears/vanishes). Capturing all of them makes the per-piece midpoint
    // branch classification exact.
    std::vector<double> bp{xa, xb};
    for (double v : {std::fabs(Y0), std::fabs(Y1)})
        if (v > 0 && v < r) {
            const double xr = std::sqrt(std::max(r * r - v * v, 0.0));
            if (xr > xa && xr < xb) bp.push_back(xr);
            if (-xr > xa && -xr < xb) bp.push_back(-xr);
        }
    std::sort(bp.begin(), bp.end());
    double area = 0;
    for (std::size_t k = 0; k + 1 < bp.size(); ++k) {
        const double xl = bp[k], xr = bp[k + 1];
        if (xr <= xl) continue;
        const double xm = 0.5 * (xl + xr);
        const double s = std::sqrt(std::max(r * r - xm * xm, 0.0));
        const double up = std::min(Y1, s), lo = std::max(Y0, -s);
        if (up <= lo) continue;                 // chord clipped away here
        const bool upConst = (Y1 < s);          // upper = Y1 (const) vs s
        const bool loConst = (Y0 > -s);         // lower = Y0 (const) vs -s
        const double arc = aint(xr, r) - aint(xl, r);
        const double upInt = upConst ? Y1 * (xr - xl) : arc;
        const double loInt = loConst ? Y0 * (xr - xl) : -arc;
        area += upInt - loInt;
    }
    return area;
}

// fluid fraction of a cell edge: the edge spans t in [t0,t1] at fixed distance
// from the centre such that the disk covers |t - centre| < sqrt(disc).
inline double edgeFluidCircle(double disc, double t0, double t1,
                              double centre) {
    if (disc <= 0) return 1.0;                  // edge outside the disk
    const double h = std::sqrt(disc);
    const double cov = std::max(
        0.0, std::min(t1, centre + h) - std::max(t0, centre - h));
    return 1.0 - cov / (t1 - t0);
}

inline CellMoments circleMoments(double cx, double cy, double r, double x0,
                                 double x1, double y0, double y1) {
    CellMoments m;
    const double cellA = (x1 - x0) * (y1 - y0);
    const double diskA = diskBoxArea(r, x0 - cx, x1 - cx, y0 - cy, y1 - cy);
    m.vol = Real((cellA - diskA) / cellA);
    m.apXlo = Real(edgeFluidCircle(r * r - (x0 - cx) * (x0 - cx), y0, y1, cy));
    m.apXhi = Real(edgeFluidCircle(r * r - (x1 - cx) * (x1 - cx), y0, y1, cy));
    m.apYlo = Real(edgeFluidCircle(r * r - (y0 - cy) * (y0 - cy), x0, x1, cx));
    m.apYhi = Real(edgeFluidCircle(r * r - (y1 - cy) * (y1 - cy), x0, x1, cx));
    // EB centroid = midpoint of the arc's endpoints on the cell edges
    double sx = 0, sy = 0;
    int n = 0;
    const auto addH = [&](double Y, double xa, double xb) {
        const double d = r * r - (Y - cy) * (Y - cy);
        if (d > 0) {
            const double h = std::sqrt(d);
            for (double xx : {cx - h, cx + h})
                if (xx >= xa && xx <= xb) { sx += xx; sy += Y; ++n; }
        }
    };
    const auto addV = [&](double X, double ya, double yb) {
        const double d = r * r - (X - cx) * (X - cx);
        if (d > 0) {
            const double h = std::sqrt(d);
            for (double yy : {cy - h, cy + h})
                if (yy >= ya && yy <= yb) { sx += X; sy += yy; ++n; }
        }
    };
    addH(y0, x0, x1); addH(y1, x0, x1); addV(x0, y0, y1); addV(x1, y0, y1);
    m.eb.cx = Real(n ? sx / n : 0.5 * (x0 + x1));
    m.eb.cy = Real(n ? sy / n : 0.5 * (y0 + y1));
    return m;
}

// ---- exact half-plane (solid = a*x + b*y < c) ---------------------------

inline CellMoments halfplaneMoments(double a, double b, double c, double x0,
                                    double x1, double y0, double y1) {
    CellMoments m;
    const auto g = [&](double x, double y) { return a * x + b * y - c; };
    // Sutherland-Hodgman: keep the g >= 0 (fluid) part of the box polygon.
    const std::array<std::array<double, 2>, 4> box{
        {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
    std::vector<std::array<double, 2>> out;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto P = box[i], Q = box[(i + 1) % 4];
        const double gp = g(P[0], P[1]), gq = g(Q[0], Q[1]);
        if (gp >= 0) out.push_back(P);
        if ((gp >= 0) != (gq >= 0)) {
            const double t = gp / (gp - gq);
            out.push_back({P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])});
        }
    }
    double area = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto P = out[i], Q = out[(i + 1) % out.size()];
        area += P[0] * Q[1] - Q[0] * P[1];
    }
    area = 0.5 * std::fabs(area);
    m.vol = Real(area / ((x1 - x0) * (y1 - y0)));
    // fluid fraction of an edge from value gA at one end to gB at the other
    const auto frac = [](double gA, double gB) {
        if (gA >= 0 && gB >= 0) return 1.0;
        if (gA < 0 && gB < 0) return 0.0;
        const double t = gA / (gA - gB);
        return gA >= 0 ? t : 1.0 - t;
    };
    m.apXlo = Real(frac(g(x0, y0), g(x0, y1)));
    m.apXhi = Real(frac(g(x1, y0), g(x1, y1)));
    m.apYlo = Real(frac(g(x0, y0), g(x1, y0)));
    m.apYhi = Real(frac(g(x0, y1), g(x1, y1)));
    // EB centroid = midpoint of the cut segment's endpoints on the cell edges
    double sx = 0, sy = 0;
    int n = 0;
    const std::array<std::array<double, 2>, 4> box2{
        {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
    for (int i = 0; i < 4; ++i) {
        const auto P = box2[i], Q = box2[(i + 1) % 4];
        const double gp = g(P[0], P[1]), gq = g(Q[0], Q[1]);
        if ((gp >= 0) != (gq >= 0)) {
            const double t = gp / (gp - gq);
            sx += P[0] + t * (Q[0] - P[0]);
            sy += P[1] + t * (Q[1] - P[1]);
            ++n;
        }
    }
    m.eb.cx = Real(n ? sx / n : 0.5 * (x0 + x1));
    m.eb.cy = Real(n ? sy / n : 0.5 * (y0 + y1));
    return m;
}

// ---- EB face from the divergence closure --------------------------------

// A_EB n_EB = sum_faces alpha_f A_f n_f  (n_EB points solid -> fluid).
inline void closeEB(CellMoments& m, double dx, double dy) {
    const double sx = (double(m.apXhi) - double(m.apXlo)) * dy;
    const double sy = (double(m.apYhi) - double(m.apYlo)) * dx;
    const double A = std::hypot(sx, sy);
    m.eb.area = Real(A);
    if (A > 1e-30) {
        m.eb.nx = Real(sx / A);
        m.eb.ny = Real(sy / A);
    }
}

// ---- grid build ---------------------------------------------------------

struct Geometry {
    int totx = 0, toty = 0;
    Real dx = 0, dy = 0;
    std::vector<CellMoments> cell;
    std::size_t nFull = 0, nCovered = 0, nCut = 0;
    const CellMoments& at(int i, int j) const {
        return cell[std::size_t(j) * totx + i];
    }
};

// momentFn(x0,x1,y0,y1) -> CellMoments (apertures + vol); EB filled by closure.
template <class F>
inline Geometry build(const Grid& g, F momentFn) {
    Geometry G;
    G.totx = g.totx();
    G.toty = g.toty();
    G.dx = g.dx;
    G.dy = g.dy;
    G.cell.resize(std::size_t(G.totx) * G.toty);
    const double hx = 0.5 * double(g.dx), hy = 0.5 * double(g.dy);
    for (int j = 0; j < G.toty; ++j)
        for (int i = 0; i < G.totx; ++i) {
            const double xc = double(g.xc(i)), yc = double(g.yc(j));
            CellMoments m =
                momentFn(xc - hx, xc + hx, yc - hy, yc + hy);
            closeEB(m, g.dx, g.dy);
            if (m.vol > Real(1) - Real(1e-6))   // 1e-6 > fp32 epsilon
                ++G.nFull;
            else if (m.vol < Real(1e-6))
                ++G.nCovered;
            else
                ++G.nCut;
            G.cell[std::size_t(j) * G.totx + i] = m;
        }
    return G;
}

} // namespace cutcell
} // namespace mm

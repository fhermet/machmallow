#pragma once

// Minimal VTK ImageData (.vti) writer, cell data, appended raw binary.
// Opens directly in ParaView.

#include "core/Grid.hpp"
#include "physics/TwoGas.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace mm {

// Optional scalar provider for two-gas grids: sc(idxWithGhosts) ->
// (phi, Gamma). When present the pressure closes on the local Gamma
// and a Y (mass fraction) array is appended.
template <class G, class SC>
inline void writeVtiSc(const std::string& path, const G& g, SC sc,
                       bool species) {
    const std::size_t n = std::size_t(g.nx) * g.ny;
    std::vector<float> rho(n), u(n), v(n), p(n), Y;
    if (species) Y.resize(n);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t id = g.idx(NG + i, NG + j);
            const std::size_t k = std::size_t(j) * g.nx + i;
            const Cons& q = g.q[id];
            if (species) {
                const auto [phi, Gm] = sc(id);
                const Prim w = toPrimG(q, Gm);
                rho[k] = w.rho; u[k] = w.u; v[k] = w.v; p[k] = w.p;
                Y[k] = std::clamp(phi / std::max(w.rho, RHO_FLOOR),
                                  Real(0), Real(1));
            } else {
                const Prim w = toPrim(q);
                rho[k] = w.rho; u[k] = w.u; v[k] = w.v; p[k] = w.p;
            }
        }

    const char* names[] = {"rho", "u", "v", "p", "Y"};
    const std::vector<float>* fields[] = {&rho, &u, &v, &p, &Y};
    const int nf = species ? 5 : 4;
    const std::uint64_t block = n * sizeof(float);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) throw std::runtime_error("cannot write " + path);

    std::fprintf(f,
                 "<VTKFile type=\"ImageData\" version=\"1.0\" "
                 "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
                 "<ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
                 "Origin=\"%g %g 0\" Spacing=\"%g %g 1\">\n"
                 "<Piece Extent=\"0 %d 0 %d 0 0\">\n<CellData>\n",
                 g.nx, g.ny, double(g.x0), double(g.y0), double(g.dx),
                 double(g.dy), g.nx, g.ny);
    for (int a = 0; a < nf; ++a) {
        std::fprintf(f,
                     "<DataArray type=\"Float32\" Name=\"%s\" "
                     "format=\"appended\" offset=\"%llu\"/>\n",
                     names[a],
                     (unsigned long long)(a * (sizeof(block) + block)));
    }
    std::fprintf(f, "</CellData>\n</Piece>\n</ImageData>\n"
                    "<AppendedData encoding=\"raw\">_");
    for (int a = 0; a < nf; ++a) {
        std::fwrite(&block, sizeof(block), 1, f);
        std::fwrite(fields[a]->data(), sizeof(float), n, f);
    }
    std::fprintf(f, "\n</AppendedData>\n</VTKFile>\n");
    std::fclose(f);
}

template <class G>
inline void writeVti(const std::string& path, const G& g) {
    writeVtiSc(path, g, [](std::size_t) {
        return std::pair<Real, Real>{0, 1 / (GAMMA - 1)};
    }, false);
}

} // namespace mm

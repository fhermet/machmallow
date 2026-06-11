#pragma once

// Minimal VTK ImageData (.vti) writer, cell data, appended raw binary.
// Opens directly in ParaView.

#include "core/Grid.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mm {

inline void writeVti(const std::string& path, const Grid& g) {
    const std::size_t n = std::size_t(g.nx) * g.ny;
    std::vector<float> rho(n), u(n), v(n), p(n);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const Prim w = toPrim(g.at(NG + i, NG + j));
            const std::size_t k = std::size_t(j) * g.nx + i;
            rho[k] = w.rho; u[k] = w.u; v[k] = w.v; p[k] = w.p;
        }

    const char* names[] = {"rho", "u", "v", "p"};
    const std::vector<float>* fields[] = {&rho, &u, &v, &p};
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
    for (int a = 0; a < 4; ++a) {
        std::fprintf(f,
                     "<DataArray type=\"Float32\" Name=\"%s\" "
                     "format=\"appended\" offset=\"%llu\"/>\n",
                     names[a],
                     (unsigned long long)(a * (sizeof(block) + block)));
    }
    std::fprintf(f, "</CellData>\n</Piece>\n</ImageData>\n"
                    "<AppendedData encoding=\"raw\">_");
    for (int a = 0; a < 4; ++a) {
        std::fwrite(&block, sizeof(block), 1, f);
        std::fwrite(fields[a]->data(), sizeof(float), n, f);
    }
    std::fprintf(f, "\n</AppendedData>\n</VTKFile>\n");
    std::fclose(f);
}

} // namespace mm

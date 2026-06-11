#pragma once

// vtkOverlappingAMR (.vthb) writer: an index file referencing one .vti
// per AMR piece, so ParaView renders the composite hierarchy with patch
// outlines. AMR must provide coarseRef(), patches (with ci0/cj0),
// patchRef() and fineCells().

#include "io/VtiWriter.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace mm {

template <class AMR>
inline void writeVthb(const std::string& base, const AMR& amr) {
    namespace fs = std::filesystem;
    fs::create_directories(base);
    const std::string leaf = fs::path(base).filename().string();

    const GridRef c = amr.coarseRef();
    writeVti(base + "/coarse.vti", c);

    FILE* f = std::fopen((base + ".vthb").c_str(), "w");
    if (f == nullptr) throw std::runtime_error("cannot write " + base);
    std::fprintf(f,
                 "<VTKFile type=\"vtkOverlappingAMR\" version=\"1.1\" "
                 "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
                 "<vtkOverlappingAMR origin=\"%g %g 0\" "
                 "grid_description=\"XY\">\n",
                 double(c.x0), double(c.y0));

    std::fprintf(f,
                 "<Block level=\"0\" spacing=\"%g %g 1\">\n"
                 "<DataSet index=\"0\" amr_box=\"0 %d 0 %d 0 0\" "
                 "file=\"%s/coarse.vti\"/>\n</Block>\n",
                 double(c.dx), double(c.dy), c.nx - 1, c.ny - 1,
                 leaf.c_str());

    const int nf = amr.fineCells();
    std::fprintf(f, "<Block level=\"1\" spacing=\"%g %g 1\">\n",
                 double(c.dx) / 2, double(c.dy) / 2);
    for (std::size_t k = 0; k < amr.patches.size(); ++k) {
        const auto& p = amr.patches[k];
        char name[64];
        std::snprintf(name, sizeof(name), "patch_%04zu.vti", k);
        writeVti(base + "/" + name, amr.patchRef(p));
        std::fprintf(f,
                     "<DataSet index=\"%zu\" amr_box=\"%d %d %d %d 0 0\" "
                     "file=\"%s/%s\"/>\n",
                     k, 2 * p.ci0, 2 * p.ci0 + nf - 1, 2 * p.cj0,
                     2 * p.cj0 + nf - 1, leaf.c_str(), name);
    }
    std::fprintf(f, "</Block>\n</vtkOverlappingAMR>\n</VTKFile>\n");
    std::fclose(f);
}

} // namespace mm

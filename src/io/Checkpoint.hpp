#pragma once

// Binary checkpoint of an AMR hierarchy (coarse state + patch blocks +
// patch data + time + step counter). Restarting a run from a checkpoint
// reproduces the straight-through run bit for bit: the dt sequence and
// the regrid cadence are functions of the saved state.

#include "core/Grid.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mm {

namespace ckdetail {

inline constexpr std::uint32_t MAGIC = 0x4d4d434b; // "MMCK"
inline constexpr std::uint32_t VERSION = 2; // v2: NG = 3 ghost ring

struct Header {
    std::uint32_t magic, version;
    std::int32_t nx, ny, fineCells, npatches, stepCount;
    double t;
};

inline void wr(FILE* f, const void* p, std::size_t n) {
    if (std::fwrite(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint write failed");
}
inline void rd(FILE* f, void* p, std::size_t n) {
    if (std::fread(p, 1, n, f) != n)
        throw std::runtime_error("checkpoint read failed (truncated?)");
}

} // namespace ckdetail

template <class AMR>
void saveCheckpoint(const std::string& path, const AMR& amr, double t) {
    using namespace ckdetail;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
        throw std::runtime_error("cannot write checkpoint: " + path);

    const GridRef c = amr.coarseRef();
    const Header h{MAGIC,
                   VERSION,
                   c.nx,
                   c.ny,
                   amr.fineCells(),
                   std::int32_t(amr.patches.size()),
                   amr.stepCount(),
                   t};
    wr(f, &h, sizeof(h));
    wr(f, c.q, std::size_t(c.totx()) * c.toty() * sizeof(Cons));
    for (const auto& p : amr.patches) {
        const std::int32_t b[2] = {p.bi, p.bj};
        wr(f, b, sizeof(b));
        const GridRef g = amr.patchRef(p);
        wr(f, g.q, std::size_t(g.totx()) * g.toty() * sizeof(Cons));
    }
    std::fclose(f);
}

// Returns the saved time. The AMR must be freshly constructed with the
// same grid, block size and config as the saved run.
template <class AMR>
double loadCheckpoint(const std::string& path, AMR& amr) {
    using namespace ckdetail;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
        throw std::runtime_error("cannot open checkpoint: " + path);

    Header h{};
    rd(f, &h, sizeof(h));
    GridRef c = amr.coarseRef();
    if (h.magic != MAGIC || h.version != VERSION)
        throw std::runtime_error("not a machmallow checkpoint: " + path);
    if (h.nx != c.nx || h.ny != c.ny || h.fineCells != amr.fineCells())
        throw std::runtime_error(
            "checkpoint grid/block mismatch with the current config");

    rd(f, c.q, std::size_t(c.totx()) * c.toty() * sizeof(Cons));

    std::vector<std::pair<int, int>> blocks(h.npatches);
    std::vector<std::vector<Cons>> data(h.npatches);
    const int pt = h.fineCells + 2 * NG;
    for (int k = 0; k < h.npatches; ++k) {
        std::int32_t b[2];
        rd(f, b, sizeof(b));
        blocks[k] = {b[0], b[1]};
        data[k].resize(std::size_t(pt) * pt);
        rd(f, data[k].data(), data[k].size() * sizeof(Cons));
    }
    std::fclose(f);

    amr.restoreBlocks(blocks, h.stepCount);
    for (int k = 0; k < h.npatches; ++k) {
        GridRef g = amr.patchRef(amr.patches[k]);
        std::copy(data[k].begin(), data[k].end(), g.q);
    }
    return h.t;
}

} // namespace mm

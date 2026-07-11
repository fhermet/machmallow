// Lock-step CPU↔GPU for immersed bodies: the same configuration (2-level
// grid, solid mask of a Mach 2 cylinder) is advanced by Amr2 (CPU,
// reference) and AmrGpu (GPU), step by step with the SAME dt. The fields must
// match to fp32 tolerance (~1e-2, GPU sum reassociation). This is
// the gate that locks the GPU port of the mask (Metal kernels + mask-aware
// AMR chain).

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
#include "amr/AmrGpuML.hpp"
#include "amr/AmrML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "core/Boundary.hpp"
#include "physics/Euler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

constexpr Real CFL = Real(0.4);
const Real CX = Real(0.5), CY = Real(0.5), RAD = Real(0.15);

std::uint8_t solidCircle(Real x, Real y) {
    return (x - CX) * (x - CX) + (y - CY) * (y - CY) < RAD * RAD ? 1 : 0;
}

template <class G> void fillTrans(G& g) {
    fillTransmissiveLeft(g);
    fillTransmissiveRight(g);
    fillTransmissiveBottom(g);
    fillTransmissiveTop(g);
}
template <class G> void fillPatchTrans(G& g, unsigned s) {
    if (s & SideLeft) fillTransmissiveLeft(g);
    if (s & SideRight) fillTransmissiveRight(g);
    if (s & SideBottom) fillTransmissiveBottom(g);
    if (s & SideTop) fillTransmissiveTop(g);
}

template <class IC> void initGrid(auto& g, IC ic) {
    for (int j = NG; j < NG + g.ny; ++j)
        for (int i = NG; i < NG + g.nx; ++i)
            g.at(i, j) = ic(g.xc(i), g.yc(j));
}

} // namespace

// One lock-step comparison; returns true on PASS.
bool compare(MetalContext& ctx, bool subcycle, Real mu) {
    const int nx = 96, ny = 64, nsteps = 40;
    const Real lx = Real(1.5), ly = Real(1.0);

    AmrConfig a;
    a.blockC = 8;
    a.maxLevels = 2;
    a.tagThreshold = Real(0.08);
    a.regridEvery = 2;
    a.subcycle = subcycle;
    a.mu = mu;

    // uniform Mach 2 free-stream (rho=1.4, p=1 -> c=1, u=2)
    const auto ic = [](Real, Real) {
        return toCons(Prim{Real(1.4), Real(2), 0, Real(1)});
    };

    Amr2 cpu(nx, ny, 0, 0, lx, ly, a);
    cpu.fillPhysicalGhosts = [](Grid& g, double) { fillTrans(g); };
    cpu.fillPatchPhysical = [](Grid& g, double, unsigned s) {
        fillPatchTrans(g, s);
    };
    cpu.solidAt = solidCircle;
    cpu.init(ic);

    AmrGpu gpu(ctx, nx, ny, 0, 0, lx, ly, a);
    gpu.fillPhysicalGhosts = [](GridRef& g, double) { fillTrans(g); };
    gpu.fillPatchPhysical = [](GridRef& g, double, unsigned s) {
        fillPatchTrans(g, s);
    };
    gpu.solidAt = solidCircle;
    gpu.init(ic);

    double t = 0, maxRel = 0;
    bool patchMismatch = false;
    for (int s = 0; s < nsteps; ++s) {
        Real dt = cpu.maxStableDtAll(CFL);
        if (s < 10) dt *= Real(0.3); // gentle start (same on both)
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
        if (cpu.patches.size() != gpu.patches.size()) patchMismatch = true;
    }

    const GridRef gc = gpu.coarseRef();
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const Real* pa = &cpu.coarse.at(i, j).rho;
            const Real* pb = &gc.at(i, j).rho;
            for (int k = 0; k < 4; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }

    const bool ok = maxRel < 1e-2 && !patchMismatch;
    char tag[32];
    std::snprintf(tag, sizeof(tag), "%s%s",
                  subcycle ? "subcycled" : "single",
                  mu > 0 ? " +viscous" : "");
    std::printf("  %-18s patches CPU=%zu GPU=%zu | max rel diff %.3e  %s\n",
                tag, cpu.patches.size(), gpu.patches.size(), maxRel,
                ok ? "PASS" : "FAIL");
    return ok;
}

// Multi-level lock-step: AmrML (CPU) vs AmrGpuML (GPU), `levels` deep.
bool compareML(MetalContext& ctx, int levels) {
    const int nx = 96, ny = 64, nsteps = 30;
    const Real lx = Real(1.5), ly = Real(1.0);

    AmrConfig a;
    a.blockC = 8;
    a.maxLevels = levels;
    a.tagThreshold = Real(0.08);
    a.regridEvery = 2;
    a.subcycle = true;

    const auto ic = [](Real, Real) {
        return toCons(Prim{Real(1.4), Real(2), 0, Real(1)});
    };

    AmrML cpu(nx, ny, 0, 0, lx, ly, a);
    cpu.fillPhysicalGhosts = [](Grid& g, double) { fillTrans(g); };
    cpu.fillPatchPhysical = [](Grid& g, double, unsigned s) {
        fillPatchTrans(g, s);
    };
    cpu.solidAt = solidCircle;
    cpu.init(ic);

    AmrGpuML gpu(ctx, nx, ny, 0, 0, lx, ly, a);
    gpu.fillPhysicalGhosts = [](GridRef& g, double) { fillTrans(g); };
    gpu.fillPatchPhysical = [](GridRef& g, double, unsigned s) {
        fillPatchTrans(g, s);
    };
    gpu.solidAt = solidCircle;
    gpu.init(ic);

    double t = 0, maxRel = 0;
    for (int s = 0; s < nsteps; ++s) {
        Real dt = cpu.maxStableDtAll(CFL);
        if (s < 10) dt *= Real(0.3);
        cpu.step(dt, t);
        gpu.step(dt, t);
        t += dt;
    }
    const GridRef gc = gpu.coarseRef();
    for (int j = NG; j < NG + ny; ++j)
        for (int i = NG; i < NG + nx; ++i) {
            const Real* pa = &cpu.base.at(i, j).rho;
            const Real* pb = &gc.at(i, j).rho;
            for (int k = 0; k < 4; ++k)
                maxRel = std::max(maxRel,
                                  std::fabs(double(pa[k]) - pb[k]) /
                                      (std::fabs(double(pa[k])) + 1e-3));
        }
    const bool ok = maxRel < 1e-2;
    std::printf("  ML %d levels        | max rel diff %.3e  %s\n", levels,
                maxRel, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    MetalContext* ctx = nullptr;
    try {
        ctx = new MetalContext();
    } catch (const std::exception& e) {
        std::printf("no Metal device — gate skipped (%s)\nPASS\n",
                    e.what());
        return 0;
    }
    std::printf("Immersed lock-step CPU/GPU (Mach 2 cylinder, 96x64, 40 steps)\n");
    const bool ok = compare(*ctx, false, 0) & compare(*ctx, true, 0) &
                    compare(*ctx, false, Real(2e-3)) & // no-slip wall
                    compareML(*ctx, 3);                // multi-level (AmrML)
    delete ctx;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

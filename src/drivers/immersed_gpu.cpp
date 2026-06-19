// Lock-step CPU↔GPU pour les corps immergés : la même configuration (grille
// 2 niveaux, masque solide d'un cylindre Mach 2) est avancée par Amr2 (CPU,
// référence) et AmrGpu (GPU), pas à pas avec le MÊME dt. Les champs doivent
// coïncider à la tolérance fp32 (~1e-2, réassociation des sommes GPU). C'est
// la porte qui verrouille le portage GPU du masque (kernels Metal + chaîne
// AMR masque-aware).

#include "amr/Amr2.hpp"
#include "amr/AmrGpu.hpp"
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
                  subcycle ? "subcyclé" : "single",
                  mu > 0 ? " +visqueux" : "");
    std::printf("  %-18s patches CPU=%zu GPU=%zu | écart rel max %.3e  %s\n",
                tag, cpu.patches.size(), gpu.patches.size(), maxRel,
                ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    MetalContext* ctx = nullptr;
    try {
        ctx = new MetalContext();
    } catch (const std::exception& e) {
        std::printf("pas de device Metal — gate ignoré (%s)\nPASS\n",
                    e.what());
        return 0;
    }
    std::printf("Lock-step immergé CPU/GPU (cylindre Mach 2, 96x64, 40 pas)\n");
    const bool ok = compare(*ctx, false, 0) & compare(*ctx, true, 0) &
                    compare(*ctx, false, Real(2e-3)); // paroi no-slip
    delete ctx;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

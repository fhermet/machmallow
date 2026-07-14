#pragma once

// GPU (Metal) 1st-order cut-cell step, the embedded-boundary analogue of
// Euler2DGpu. Conserved state lives in a shared buffer (CPU fills ghosts / does
// I-O on the same memory). The embedded geometry (per-cell moments) is uploaded
// once via setGeometry(). One step = five kernels: aperture-weighted x/y face
// fluxes, conservative divergence D^c, hybrid divergence + gathered flux
// redistribution, positivity-floored update. Validated in lock-step against the
// CPU cutCellStepFluxed oracle.

#include "backend/metal/MetalContext.hpp"
#include "core/Grid.hpp"
#include "geometry/CutCell.hpp"

#include <cstdint>
#include <cstring>

namespace mm {

class CutCell2DGpu {
public:
    // Per-cell EB moments, matching struct CCMom in cutcell.metal.
    struct CCMom {
        float vol = 1;
        float apXhi = 1, apYhi = 1;
        float ebArea = 0, ebnx = 0, ebny = 0;
        float pad0 = 0, pad1 = 0;
    };
    struct Params {
        std::int32_t tx, ty, nx, ny;
        float dx, dy, dt;
    };

    CutCell2DGpu(MetalContext& ctx, int nx, int ny, Real dx, Real dy)
        : ctx_(ctx), nx_(nx), ny_(ny), tx_(nx + 2 * NG), ty_(ny + 2 * NG),
          dx_(dx), dy_(dy) {
        lib_ = ctx.compileLibrary("cutcell.metal");
        fluxX_ = ctx.makePipeline(lib_, "cc_flux_x");
        fluxY_ = ctx.makePipeline(lib_, "cc_flux_y");
        dc_ = ctx.makePipeline(lib_, "cc_dc");
        hybrid_ = ctx.makePipeline(lib_, "cc_hybrid");
        update_ = ctx.makePipeline(lib_, "cc_update");
        const std::size_t n = std::size_t(tx_) * ty_;
        const auto mk4 = [&] {
            return ctx.device()->newBuffer(n * sizeof(Cons),
                                           MTL::ResourceStorageModeShared);
        };
        q_ = mk4(); Fx_ = mk4(); Fy_ = mk4(); Dc_ = mk4(); D_ = mk4();
        geo_ = ctx.device()->newBuffer(n * sizeof(CCMom),
                                       MTL::ResourceStorageModeShared);
        auto* g = static_cast<CCMom*>(geo_->contents());
        for (std::size_t k = 0; k < n; ++k) g[k] = CCMom{}; // default full
    }
    ~CutCell2DGpu() {
        for (MTL::Buffer* b : {q_, Fx_, Fy_, Dc_, D_, geo_}) b->release();
        for (MTL::ComputePipelineState* p :
             {fluxX_, fluxY_, dc_, hybrid_, update_})
            p->release();
        lib_->release();
    }
    CutCell2DGpu(const CutCell2DGpu&) = delete;
    CutCell2DGpu& operator=(const CutCell2DGpu&) = delete;

    Cons* data() { return static_cast<Cons*>(q_->contents()); }
    GridRef ref(Real x0, Real y0) {
        return GridRef{nx_, ny_, x0, y0, dx_, dy_, data()};
    }

    // Upload the embedded-boundary geometry (tx*ty cells, ghosts included).
    void setGeometry(const cutcell::Geometry& G) {
        auto* g = static_cast<CCMom*>(geo_->contents());
        for (int j = 0; j < ty_; ++j)
            for (int i = 0; i < tx_; ++i) {
                const CellMoments& m = G.at(i, j);
                CCMom c;
                c.vol = float(m.vol);
                c.apXhi = float(m.apXhi);
                c.apYhi = float(m.apYhi);
                c.ebArea = float(m.eb.area);
                c.ebnx = float(m.eb.nx);
                c.ebny = float(m.eb.ny);
                g[std::size_t(j) * tx_ + i] = c;
            }
    }

    // One 1st-order forward-Euler cut-cell step. Ghosts filled by the caller.
    void step(Real dt) {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_), float(dt)};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, fluxX_, {q_, geo_, Fx_}, p, nx_ + 1, ny_);
        encode(cmd, fluxY_, {q_, geo_, Fy_}, p, nx_, ny_ + 1);
        encode(cmd, dc_, {q_, geo_, Fx_, Fy_, Dc_}, p, nx_, ny_);
        encode(cmd, hybrid_, {geo_, Dc_, D_}, p, nx_, ny_);
        encode(cmd, update_, {q_, geo_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

private:
    void encode(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                std::initializer_list<MTL::Buffer*> bufs, const Params& p,
                int w, int h) const {
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        int slot = 0;
        for (MTL::Buffer* b : bufs) enc->setBuffer(b, 0, slot++);
        enc->setBytes(&p, sizeof(p), slot);
        enc->dispatchThreads(MTL::Size(w, h, 1), MTL::Size(16, 16, 1));
        enc->endEncoding();
    }

    MetalContext& ctx_;
    int nx_, ny_, tx_, ty_;
    Real dx_, dy_;
    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *fluxX_ = nullptr, *fluxY_ = nullptr,
                              *dc_ = nullptr, *hybrid_ = nullptr,
                              *update_ = nullptr;
    MTL::Buffer *q_ = nullptr, *Fx_ = nullptr, *Fy_ = nullptr, *Dc_ = nullptr,
                *D_ = nullptr, *geo_ = nullptr;
};

static_assert(sizeof(Cons) == 4 * sizeof(float), "Cons must be float4");

} // namespace mm

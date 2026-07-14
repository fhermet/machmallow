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
        std::int32_t stride; // cells per pool slot (0 = plain)
        float mu = 0, kT = 0; // viscosity + heat conductivity (0 = inviscid)
    };

    // Enable the viscous (Stokes/Fourier + no-slip EB) terms in the 2nd-order
    // path. kT = mu*GAMMA/((GAMMA-1)*Pr) (pass heatConductivity(mu)).
    void setViscosity(Real mu, Real kT) { mu_ = mu; kT_ = kT; }

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
        for (MTL::Buffer* b : {qP_, FxP_, FyP_, DcP_, DP_, geoP_, slots_})
            if (b) b->release();
        for (MTL::Buffer* b : {Gdx_, Gdy_, qn_, GdxP_, GdyP_, qnP_})
            if (b) b->release();
        for (MTL::ComputePipelineState* p :
             {fluxX_, fluxY_, dc_, hybrid_, update_})
            p->release();
        for (MTL::ComputePipelineState* p :
             {fluxXP_, fluxYP_, dcP_, hybridP_, updateP_})
            if (p) p->release();
        for (MTL::ComputePipelineState* p :
             {grad_, fluxXo2_, fluxYo2_, dcO2_, rk2_})
            if (p) p->release();
        for (MTL::ComputePipelineState* p :
             {gradP_, fluxXo2P_, fluxYo2P_, dcO2P_, rk2P_})
            if (p) p->release();
        lib_->release();
    }
    CutCell2DGpu(const CutCell2DGpu&) = delete;
    CutCell2DGpu& operator=(const CutCell2DGpu&) = delete;

    Cons* data() { return static_cast<Cons*>(q_->contents()); }
    GridRef ref(Real x0, Real y0) {
        return GridRef{nx_, ny_, x0, y0, dx_, dy_, data()};
    }

    // Upload the embedded-boundary geometry (tx*ty cells, ghosts included).
    // (x0,y0) is the grid origin used to build G: pad0/pad1 store the EB
    // centroid offset from the cell centre (for 2nd-order EB reconstruction),
    // which is origin-independent, so the default 0 suffices at 1st order.
    void setGeometry(const cutcell::Geometry& G, Real x0 = 0, Real y0 = 0) {
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
                const Real xc = x0 + (Real(i - NG) + Real(0.5)) * dx_;
                const Real yc = y0 + (Real(j - NG) + Real(0.5)) * dy_;
                c.pad0 = float(m.eb.cx - xc);
                c.pad1 = float(m.eb.cy - yc);
                g[std::size_t(j) * tx_ + i] = c;
            }
    }

    // One 1st-order forward-Euler cut-cell step. Ghosts filled by the caller.
    void step(Real dt) {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_),
                       float(dt), 0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, fluxX_, {q_, geo_, Fx_}, p, nx_ + 1, ny_);
        encode(cmd, fluxY_, {q_, geo_, Fy_}, p, nx_, ny_ + 1);
        encode(cmd, dc_, {q_, geo_, Fx_, Fy_, Dc_}, p, nx_, ny_);
        encode(cmd, hybrid_, {geo_, Dc_, D_}, p, nx_, ny_);
        encode(cmd, update_, {q_, geo_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // ---- phase-split single-grid step -----------------------------------
    // For composite AMR use: the caller interleaves CPU cross-patch passes
    // (Dc ghost fill / redistribution scatter) between the GPU phases.
    // dcPhase computes the extensive fluxes (Fx/Fy) and D^c; hybridPhase the
    // hybrid divergence + gathered FRD; updatePhase the floored update.
    void dcPhase() {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_), 0, 0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, fluxX_, {q_, geo_, Fx_}, p, nx_ + 1, ny_);
        encode(cmd, fluxY_, {q_, geo_, Fy_}, p, nx_, ny_ + 1);
        encode(cmd, dc_, {q_, geo_, Fx_, Fy_, Dc_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    void hybridPhase() {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_), 0, 0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, hybrid_, {geo_, Dc_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    void updatePhase(Real dt) {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_),
                       float(dt), 0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, update_, {q_, geo_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    Cons* dcData() { return static_cast<Cons*>(Dc_->contents()); }
    Cons* dData() { return static_cast<Cons*>(D_->contents()); }
    const Cons* fxData() const { return static_cast<Cons*>(Fx_->contents()); }
    const Cons* fyData() const { return static_cast<Cons*>(Fy_->contents()); }

    // ---- 2nd-order single-grid step (SSP-RK2) ----------------------------
    // Enable the 2nd-order path: least-squares gradients + reconstructed face
    // / EB fluxes, advanced with SSP-RK2. Allocates the gradient + t^n buffers.
    void enableO2() {
        grad_ = ctx_.makePipeline(lib_, "cc_grad");
        fluxXo2_ = ctx_.makePipeline(lib_, "cc_flux_x_o2");
        fluxYo2_ = ctx_.makePipeline(lib_, "cc_flux_y_o2");
        dcO2_ = ctx_.makePipeline(lib_, "cc_dc_o2");
        rk2_ = ctx_.makePipeline(lib_, "cc_rk2");
        const std::size_t n = std::size_t(tx_) * ty_;
        const auto mk4 = [&] {
            return ctx_.device()->newBuffer(n * sizeof(Cons),
                                            MTL::ResourceStorageModeShared);
        };
        Gdx_ = mk4(); Gdy_ = mk4(); qn_ = mk4();
    }
    // 2nd-order FRD'd divergence D from the current state (grad -> flux ->
    // D^c -> hybrid). Ghosts filled by the caller.
    void divO2() {
        const Params p{tx_,  ty_,        nx_,        ny_, float(dx_),
                       float(dy_), 0, 0, float(mu_), float(kT_)};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, grad_, {q_, geo_, Gdx_, Gdy_}, p, nx_, ny_);
        encode(cmd, fluxXo2_, {q_, geo_, Gdx_, Gdy_, Fx_}, p, nx_ + 1, ny_);
        encode(cmd, fluxYo2_, {q_, geo_, Gdx_, Gdy_, Fy_}, p, nx_, ny_ + 1);
        encode(cmd, dcO2_, {q_, geo_, Gdx_, Gdy_, Fx_, Fy_, Dc_}, p, nx_, ny_);
        encode(cmd, hybrid_, {geo_, Dc_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    // RK2 stage 1: save q^n, then q <- floor(q + dt D). Call divO2() first.
    void rk2Stage1(Real dt) {
        std::memcpy(qn_->contents(), q_->contents(),
                    std::size_t(tx_) * ty_ * sizeof(Cons));
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_), float(dt),
                       0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, update_, {q_, geo_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    // RK2 stage 2: q <- floor(0.5 q^n + 0.5 (q + dt D)). Call divO2() (from
    // the stage-1 state) first.
    void rk2Stage2(Real dt) {
        const Params p{tx_, ty_, nx_, ny_, float(dx_), float(dy_), float(dt),
                       0};
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encode(cmd, rk2_, {q_, qn_, geo_, D_}, p, nx_, ny_);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // ---- pooled multi-patch layout (all slots advanced in one dispatch) ---
    // Every slot holds a full tx*ty grid; the pool kernels select a slot via
    // gid.z -> slots[gid.z]. This is the layout AmrGpu batches its patches in.
    void enablePool(int nSlots) {
        nSlots_ = nSlots;
        fluxXP_ = ctx_.makePipeline(lib_, "cc_flux_x_pool");
        fluxYP_ = ctx_.makePipeline(lib_, "cc_flux_y_pool");
        dcP_ = ctx_.makePipeline(lib_, "cc_dc_pool");
        hybridP_ = ctx_.makePipeline(lib_, "cc_hybrid_pool");
        updateP_ = ctx_.makePipeline(lib_, "cc_update_pool");
        const std::size_t n = std::size_t(tx_) * ty_;
        const auto mk4 = [&](std::size_t m) {
            return ctx_.device()->newBuffer(m * sizeof(Cons),
                                            MTL::ResourceStorageModeShared);
        };
        qP_ = mk4(n * nSlots); FxP_ = mk4(n * nSlots); FyP_ = mk4(n * nSlots);
        DcP_ = mk4(n * nSlots); DP_ = mk4(n * nSlots);
        geoP_ = ctx_.device()->newBuffer(n * nSlots * sizeof(CCMom),
                                         MTL::ResourceStorageModeShared);
        auto* g = static_cast<CCMom*>(geoP_->contents());
        for (std::size_t k = 0; k < n * nSlots; ++k) g[k] = CCMom{};
        slots_ = ctx_.device()->newBuffer(nSlots * sizeof(std::uint32_t),
                                          MTL::ResourceStorageModeShared);
    }
    Cons* poolData(int slot) {
        return static_cast<Cons*>(qP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }
    // (x0,y0) is the patch origin used to build G — needed for the EB-centroid
    // offset (pad0/pad1) that 2nd-order reconstruction uses. Origin-independent,
    // so the default 0 is fine at 1st order.
    void setPoolGeometry(int slot, const cutcell::Geometry& G, Real x0 = 0,
                         Real y0 = 0) {
        auto* g = static_cast<CCMom*>(geoP_->contents()) +
                  std::size_t(slot) * tx_ * ty_;
        for (int j = 0; j < ty_; ++j)
            for (int i = 0; i < tx_; ++i) {
                const CellMoments& m = G.at(i, j);
                const Real xc = x0 + (Real(i - NG) + Real(0.5)) * dx_;
                const Real yc = y0 + (Real(j - NG) + Real(0.5)) * dy_;
                g[std::size_t(j) * tx_ + i] =
                    CCMom{float(m.vol), float(m.apXhi), float(m.apYhi),
                          float(m.eb.area), float(m.eb.nx), float(m.eb.ny),
                          float(m.eb.cx - xc), float(m.eb.cy - yc)};
            }
    }
    // Advance the given slots in one pooled dispatch.
    void stepPool(Real dt, const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(dt);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, fluxXP_, {qP_, geoP_, FxP_}, p, nx_ + 1, ny_, z);
        encodeP(cmd, fluxYP_, {qP_, geoP_, FyP_}, p, nx_, ny_ + 1, z);
        encodeP(cmd, dcP_, {qP_, geoP_, FxP_, FyP_, DcP_}, p, nx_, ny_, z);
        encodeP(cmd, hybridP_, {geoP_, DcP_, DP_}, p, nx_, ny_, z);
        encodeP(cmd, updateP_, {qP_, geoP_, DP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // ---- phase-split pool step (composite cross-patch FRD on the CPU) ----
    void dcPhasePool(const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(0);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, fluxXP_, {qP_, geoP_, FxP_}, p, nx_ + 1, ny_, z);
        encodeP(cmd, fluxYP_, {qP_, geoP_, FyP_}, p, nx_, ny_ + 1, z);
        encodeP(cmd, dcP_, {qP_, geoP_, FxP_, FyP_, DcP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    void hybridPhasePool(const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(0);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, hybridP_, {geoP_, DcP_, DP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    void updatePhasePool(Real dt, const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(dt);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, updateP_, {qP_, geoP_, DP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    Cons* poolDc(int slot) {
        return static_cast<Cons*>(DcP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }
    Cons* poolD(int slot) {
        return static_cast<Cons*>(DP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }
    const Cons* poolFx(int slot) const {
        return static_cast<Cons*>(FxP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }
    const Cons* poolFy(int slot) const {
        return static_cast<Cons*>(FyP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }

    // ---- 2nd-order pooled composite (SSP-RK2) ----------------------------
    // Enable the pooled O2 path: gradient + t^n buffers and pool O2 pipelines.
    // enablePool(nSlots) must have been called first.
    void enableO2Pool() {
        gradP_ = ctx_.makePipeline(lib_, "cc_grad_pool");
        fluxXo2P_ = ctx_.makePipeline(lib_, "cc_flux_x_o2_pool");
        fluxYo2P_ = ctx_.makePipeline(lib_, "cc_flux_y_o2_pool");
        dcO2P_ = ctx_.makePipeline(lib_, "cc_dc_o2_pool");
        rk2P_ = ctx_.makePipeline(lib_, "cc_rk2_pool");
        const std::size_t n = std::size_t(tx_) * ty_ * nSlots_;
        const auto mk4 = [&] {
            return ctx_.device()->newBuffer(n * sizeof(Cons),
                                            MTL::ResourceStorageModeShared);
        };
        GdxP_ = mk4(); GdyP_ = mk4(); qnP_ = mk4();
    }
    // GPU: least-squares gradients for the active slots (interior).
    void gradPhasePool(const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(0);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, gradP_, {qP_, geoP_, GdxP_, GdyP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    // GPU: 2nd-order reconstructed fluxes + D^c for the active slots. Call
    // after gradPhasePool and the CPU gradient-ghost exchange.
    void dcO2PhasePool(const std::vector<int>& active) {
        setSlots_(active);
        Params p = poolParams_(0);
        p.mu = float(mu_);
        p.kT = float(kT_);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, fluxXo2P_, {qP_, geoP_, GdxP_, GdyP_, FxP_}, p, nx_ + 1,
                ny_, z);
        encodeP(cmd, fluxYo2P_, {qP_, geoP_, GdxP_, GdyP_, FyP_}, p, nx_,
                ny_ + 1, z);
        encodeP(cmd, dcO2P_, {qP_, geoP_, GdxP_, GdyP_, FxP_, FyP_, DcP_}, p,
                nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    // RK2 stage 1: save q^n for the active slots, then q <- floor(q + dt D).
    void rk2Stage1Pool(Real dt, const std::vector<int>& active) {
        for (int s : active) {
            const std::size_t off = std::size_t(s) * tx_ * ty_;
            std::memcpy(static_cast<Cons*>(qnP_->contents()) + off,
                        static_cast<Cons*>(qP_->contents()) + off,
                        std::size_t(tx_) * ty_ * sizeof(Cons));
        }
        updatePhasePool(dt, active);
    }
    // RK2 stage 2: q <- floor(0.5 q^n + 0.5 (q + dt D)) for the active slots.
    void rk2Stage2Pool(Real dt, const std::vector<int>& active) {
        setSlots_(active);
        const Params p = poolParams_(dt);
        const int z = int(active.size());
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeP(cmd, rk2P_, {qP_, qnP_, geoP_, DP_}, p, nx_, ny_, z);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    Cons* poolGdx(int slot) {
        return static_cast<Cons*>(GdxP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
    }
    Cons* poolGdy(int slot) {
        return static_cast<Cons*>(GdyP_->contents()) +
               std::size_t(slot) * tx_ * ty_;
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
    void setSlots_(const std::vector<int>& active) {
        auto* sl = static_cast<std::uint32_t*>(slots_->contents());
        for (std::size_t k = 0; k < active.size(); ++k)
            sl[k] = std::uint32_t(active[k]);
    }
    Params poolParams_(Real dt) const {
        return Params{tx_, ty_, nx_, ny_, float(dx_), float(dy_),
                      float(dt), tx_ * ty_};
    }
    void encodeP(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                 std::initializer_list<MTL::Buffer*> bufs, const Params& p,
                 int w, int h, int z) const {
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        int slot = 0;
        for (MTL::Buffer* b : bufs) enc->setBuffer(b, 0, slot++);
        enc->setBytes(&p, sizeof(p), slot++);
        enc->setBuffer(slots_, 0, slot);
        enc->dispatchThreads(MTL::Size(w, h, z), MTL::Size(16, 16, 1));
        enc->endEncoding();
    }

    MetalContext& ctx_;
    int nx_, ny_, tx_, ty_;
    Real dx_, dy_;
    Real mu_ = 0, kT_ = 0; // viscous terms (0 = inviscid)
    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *fluxX_ = nullptr, *fluxY_ = nullptr,
                              *dc_ = nullptr, *hybrid_ = nullptr,
                              *update_ = nullptr;
    MTL::Buffer *q_ = nullptr, *Fx_ = nullptr, *Fy_ = nullptr, *Dc_ = nullptr,
                *D_ = nullptr, *geo_ = nullptr;
    // 2nd-order path (allocated by enableO2)
    MTL::ComputePipelineState *grad_ = nullptr, *fluxXo2_ = nullptr,
                              *fluxYo2_ = nullptr, *dcO2_ = nullptr,
                              *rk2_ = nullptr;
    MTL::Buffer *Gdx_ = nullptr, *Gdy_ = nullptr, *qn_ = nullptr;
    // pooled multi-patch layout (allocated by enablePool)
    int nSlots_ = 0;
    MTL::ComputePipelineState *fluxXP_ = nullptr, *fluxYP_ = nullptr,
                              *dcP_ = nullptr, *hybridP_ = nullptr,
                              *updateP_ = nullptr;
    MTL::Buffer *qP_ = nullptr, *FxP_ = nullptr, *FyP_ = nullptr,
                *DcP_ = nullptr, *DP_ = nullptr, *geoP_ = nullptr,
                *slots_ = nullptr;
    // 2nd-order pooled path (allocated by enableO2Pool)
    MTL::ComputePipelineState *gradP_ = nullptr, *fluxXo2P_ = nullptr,
                              *fluxYo2P_ = nullptr, *dcO2P_ = nullptr,
                              *rk2P_ = nullptr;
    MTL::Buffer *GdxP_ = nullptr, *GdyP_ = nullptr, *qnP_ = nullptr;
};

static_assert(sizeof(Cons) == 4 * sizeof(float), "Cons must be float4");

} // namespace mm

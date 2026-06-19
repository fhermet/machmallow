#pragma once

// GPU implementation of the unsplit MUSCL-Hancock step. The conserved
// state lives in a shared (unified memory) buffer: the CPU fills ghost
// cells and does I/O on the very same memory the GPU computes on.

#include "backend/metal/MetalContext.hpp"
#include "core/Grid.hpp"
#include "physics/Reaction.hpp"
#include "physics/TwoGas.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>

namespace mm {

class Euler2DGpu {
public:
    // Cell scalar pair carried by the species kernels: (phi, Gamma).
    struct PhiG {
        float phi, G;
    };

    Euler2DGpu(MetalContext& ctx, int nx, int ny, Real dx, Real dy)
        : ctx_(ctx), nx_(nx), ny_(ny), tx_(nx + 2 * NG), ty_(ny + 2 * NG),
          dx_(dx), dy_(dy) {
        lib_ = ctx.compileLibrary("euler2d.metal");
        predictor_ = ctx.makePipeline(lib_, "predictor");
        fluxX_ = ctx.makePipeline(lib_, "flux_x");
        fluxY_ = ctx.makePipeline(lib_, "flux_y");
        update_ = ctx.makePipeline(lib_, "update_cons");
        wave_ = ctx.makePipeline(lib_, "wave_speed");

        const std::size_t bytes = std::size_t(tx_) * ty_ * sizeof(Cons);
        const auto mk = [&] {
            return ctx.device()->newBuffer(bytes,
                                           MTL::ResourceStorageModeShared);
        };
        q_ = mk(); xL_ = mk(); xR_ = mk(); yB_ = mk(); yT_ = mk();
        Fx_ = mk(); Fy_ = mk();
        // [0] = max sx, [1] = max sy, [2] = min rho (uint-ordered floats)
        smax_ = ctx.device()->newBuffer(3 * sizeof(std::uint32_t),
                                        MTL::ResourceStorageModeShared);
        // Immersed-solid mask (1 = solid), bound to the inviscid kernels.
        // Always allocated and zero by default → no-op unless populated.
        smask_ = ctx.device()->newBuffer(std::size_t(tx_) * ty_,
                                         MTL::ResourceStorageModeShared);
        std::memset(smask_->contents(), 0, std::size_t(tx_) * ty_);
    }

    ~Euler2DGpu() {
        for (MTL::Buffer* b : {q_, xL_, xR_, yB_, yT_, Fx_, Fy_, smax_, smask_})
            b->release();
        for (MTL::Buffer* b : {s_, fXb_, fYb_, sFx_, sFy_})
            if (b) b->release();
        for (MTL::Buffer* b : {u0_, FxA_, FyA_, s0_, sFxA_, sFyA_})
            if (b) b->release();
        for (MTL::ComputePipelineState* p : {wfluxX_, wfluxY_, rkUpd_})
            if (p) p->release();
        for (MTL::ComputePipelineState* p : {wfluxXY_, wfluxYY_, rkUpdY_})
            if (p) p->release();
        if (reactP_) reactP_->release();
        for (MTL::ComputePipelineState* p :
             {predictor_, fluxX_, fluxY_, update_, wave_})
            p->release();
        for (MTL::ComputePipelineState* p :
             {predictorY_, fluxXY_, fluxYY_, updateY_, waveY_})
            if (p) p->release();
        lib_->release();
    }

    Euler2DGpu(const Euler2DGpu&) = delete;
    Euler2DGpu& operator=(const Euler2DGpu&) = delete;

    // CPU-side view of the GPU state (ghosts included) for IC/BC/I-O.
    Cons* data() { return static_cast<Cons*>(q_->contents()); }
    GridRef ref(Real x0, Real y0) {
        return GridRef{nx_, ny_, x0, y0, dx_, dy_, data()};
    }

    // CFL time step from a GPU reduction (convective + viscous limits).
    Real maxStableDt(Real cfl) {
        zeroWave();
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeWave(cmd);
        cmd->commit();
        cmd->waitUntilCompleted();
        const auto [sx, sy] = waveSpeeds();
        Real dt = cfl * std::min(dx_ / sx, dy_ / sy);
        if (mu_ > 0)
            dt = std::min(dt, viscousDtLimit(cfl, dx_, dy_,
                                             mu_ / rhoMin()));
        return dt;
    }

    static Real viscousDtLimit(Real cfl, Real dx, Real dy, Real nu) {
        const Real nuEff =
            nu * std::max(Real(4.0 / 3.0), GAMMA / PRANDTL);
        return cfl * Real(0.5) /
               (nuEff * (1 / (dx * dx) + 1 / (dy * dy)));
    }

    // One full step. Ghosts must be filled (via data()) by the caller.
    void step(Real dt) {
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeStep(cmd, dt);
        cmd->commit();
        cmd->waitUntilCompleted();
    }

    // Composable pieces for callers building combined command buffers
    // (e.g. AMR coarse + patch pool in one submission).
    void zeroWave() {
        auto* sm = static_cast<std::uint32_t*>(smax_->contents());
        sm[0] = sm[1] = 0;
        sm[2] = std::bit_cast<std::uint32_t>(
            std::numeric_limits<float>::max());
    }
    void encodeWave(MTL::CommandBuffer* cmd) const {
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        const Params p = params(0);
        int b = 0;
        enc->setComputePipelineState(species_ ? waveY_ : wave_);
        enc->setBuffer(q_, 0, b++);
        if (species_) enc->setBuffer(s_, 0, b++);
        enc->setBuffer(smax_, 0, b++);
        enc->setBytes(&p, sizeof(p), b);
        dispatch(enc, nx_, ny_);
        enc->endEncoding();
    }
    std::pair<Real, Real> waveSpeeds() const {
        const auto* sm = static_cast<const std::uint32_t*>(smax_->contents());
        return {std::bit_cast<float>(sm[0]), std::bit_cast<float>(sm[1])};
    }
    Real rhoMin() const {
        const auto* sm = static_cast<const std::uint32_t*>(smax_->contents());
        return std::bit_cast<float>(sm[2]);
    }
    void encodeStep(MTL::CommandBuffer* cmd, Real dt) const {
        const Params p = params(dt);
        if (species_) {
            encode(cmd, predictorY_,
                   {q_, s_, xL_, xR_, yB_, yT_, fXb_, fYb_}, p, tx_ - 2,
                   ty_ - 2);
            encode(cmd, fluxXY_, {xL_, xR_, fXb_, q_, Fx_, sFx_}, p, nx_ + 1,
                   ny_);
            encode(cmd, fluxYY_, {yB_, yT_, fYb_, q_, Fy_, sFy_}, p, nx_,
                   ny_ + 1);
            encode(cmd, updateY_, {q_, s_, Fx_, Fy_, sFx_, sFy_}, p, nx_,
                   ny_);
            return;
        }
        encode(cmd, predictor_, {q_, xL_, xR_, yB_, yT_, smask_}, p,
               tx_ - 2, ty_ - 2);
        encode(cmd, fluxX_, {xL_, xR_, q_, Fx_, smask_}, p, nx_ + 1, ny_);
        encode(cmd, fluxY_, {yB_, yT_, q_, Fy_, smask_}, p, nx_, ny_ + 1);
        encode(cmd, update_, {q_, Fx_, Fy_, smask_}, p, nx_, ny_);
    }

    // Dynamic viscosity for the flux kernels (0 = inviscid Euler).
    void setViscosity(Real mu) { mu_ = mu; }
    void setGravity(Real gx, Real gy) { gx_ = gx; gy_ = gy; }

    // WENO5 + SSP-RK3 mode: per-stage kernels; the caller orchestrates
    // the three stages (ghosts are CPU-filled between them).
    void enableWeno() {
        weno_ = true;
        wfluxX_ = ctx_.makePipeline(lib_, "weno_flux_x");
        wfluxY_ = ctx_.makePipeline(lib_, "weno_flux_y");
        rkUpd_ = ctx_.makePipeline(lib_, "rk_update");
        const std::size_t bytes = std::size_t(tx_) * ty_ * sizeof(Cons);
        const auto mk = [&] {
            return ctx_.device()->newBuffer(
                bytes, MTL::ResourceStorageModeShared);
        };
        u0_ = mk(); FxA_ = mk(); FyA_ = mk();
    }
    // stage s of the SSP-RK3 step (ghosts must be current)
    void encodeWenoStage(MTL::CommandBuffer* cmd, Real dt,
                         int stage) const {
        static constexpr float A[3] = {0, 0.75f, float(1.0 / 3.0)};
        static constexpr float B[3] = {1, 0.25f, float(2.0 / 3.0)};
        static constexpr float W[3] = {float(1.0 / 6.0),
                                       float(1.0 / 6.0),
                                       float(2.0 / 3.0)};
        Params p = params(dt);
        p.rks = stage;
        p.rka = A[stage];
        p.rkb = B[stage];
        p.rkw = W[stage];
        encode(cmd, wfluxX_, {q_, Fx_, FxA_}, p, nx_ + 1, ny_);
        encode(cmd, wfluxY_, {q_, Fy_, FyA_}, p, nx_, ny_ + 1);
        encode(cmd, rkUpd_, {q_, u0_, Fx_, Fy_}, p, nx_, ny_);
    }
    // RK-weighted flux sums (what AMR refluxing consumes in WENO mode)
    const Cons* fxA() const {
        return static_cast<const Cons*>(FxA_->contents());
    }
    const Cons* fyA() const {
        return static_cast<const Cons*>(FyA_->contents());
    }

    // WENO5 + SSP-RK3 two-gas mode: combines the species scalar state
    // (phi, Gamma) with the per-stage WENO machinery. Allocates the RK
    // starts (u0, s0) and both the conserved and species flux sums.
    void enableWenoSpecies(const GasPair& gas) {
        gas_ = gas;
        species_ = true;
        wenoSpecies_ = true;
        wfluxXY_ = ctx_.makePipeline(lib_, "weno_flux_x_y");
        wfluxYY_ = ctx_.makePipeline(lib_, "weno_flux_y_y");
        rkUpdY_ = ctx_.makePipeline(lib_, "rk_update_y");
        // the wave kernel is scheme-independent (reads q + sc only)
        waveY_ = ctx_.makePipeline(lib_, "wave_y");
        const std::size_t n = std::size_t(tx_) * ty_;
        const auto mk4 = [&] {
            return ctx_.device()->newBuffer(
                n * sizeof(Cons), MTL::ResourceStorageModeShared);
        };
        s_ = ctx_.device()->newBuffer(n * sizeof(PhiG),
                                      MTL::ResourceStorageModeShared);
        s0_ = ctx_.device()->newBuffer(n * sizeof(PhiG),
                                       MTL::ResourceStorageModeShared);
        u0_ = mk4(); FxA_ = mk4(); FyA_ = mk4();
        sFx_ = mk4(); sFy_ = mk4(); sFxA_ = mk4(); sFyA_ = mk4();
    }
    void encodeWenoStageY(MTL::CommandBuffer* cmd, Real dt,
                          int stage) const {
        static constexpr float A[3] = {0, 0.75f, float(1.0 / 3.0)};
        static constexpr float B[3] = {1, 0.25f, float(2.0 / 3.0)};
        static constexpr float W[3] = {float(1.0 / 6.0),
                                       float(1.0 / 6.0),
                                       float(2.0 / 3.0)};
        Params p = params(dt);
        p.rks = stage;
        p.rka = A[stage];
        p.rkb = B[stage];
        p.rkw = W[stage];
        encode(cmd, wfluxXY_, {q_, s_, Fx_, FxA_, sFx_, sFxA_}, p,
               nx_ + 1, ny_);
        encode(cmd, wfluxYY_, {q_, s_, Fy_, FyA_, sFy_, sFyA_}, p, nx_,
               ny_ + 1);
        encode(cmd, rkUpdY_, {q_, s_, u0_, s0_, Fx_, Fy_, sFx_, sFy_}, p,
               nx_, ny_);
    }
    // RK-weighted species mass flux sum (.rho), refluxed in WENO mode.
    const Cons* sfxA() const {
        return static_cast<const Cons*>(sFxA_->contents());
    }
    const Cons* sfyA() const {
        return static_cast<const Cons*>(sFyA_->contents());
    }

    // Two-gas mode: allocates the scalar state (phi, Gamma) and face /
    // flux buffers and switches encodeStep/encodeWave to the species
    // kernels (which have neither viscosity nor gravity, like the CPU).
    void enableSpecies(const GasPair& gas) {
        gas_ = gas;
        species_ = true;
        predictorY_ = ctx_.makePipeline(lib_, "predictor_y");
        fluxXY_ = ctx_.makePipeline(lib_, "flux_x_y");
        fluxYY_ = ctx_.makePipeline(lib_, "flux_y_y");
        updateY_ = ctx_.makePipeline(lib_, "update_y");
        waveY_ = ctx_.makePipeline(lib_, "wave_y");
        const std::size_t n = std::size_t(tx_) * ty_;
        s_ = ctx_.device()->newBuffer(n * sizeof(PhiG),
                                      MTL::ResourceStorageModeShared);
        const auto mk4 = [&] {
            return ctx_.device()->newBuffer(
                n * sizeof(Cons), MTL::ResourceStorageModeShared);
        };
        fXb_ = mk4(); fYb_ = mk4(); sFx_ = mk4(); sFy_ = mk4();
    }
    PhiG* sData() { return static_cast<PhiG*>(s_->contents()); }
    MTL::Buffer* sBuffer() const { return s_; }

    // Single-step reaction source (needs species mode for the scalar
    // state). encodeReact applies one Strang half-step over dt.
    void enableReaction(const Reaction& r) {
        react_ = r;
        reactP_ = ctx_.makePipeline(lib_, "react");
    }
    void encodeReact(MTL::CommandBuffer* cmd, Real dt) const {
        Params p = params(dt);
        p.rA = react_.A; p.rEa = react_.Ea;
        p.rq = react_.q; p.rTign = react_.Tign;
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(reactP_);
        enc->setBuffer(q_, 0, 0);
        enc->setBuffer(s_, 0, 1);
        enc->setBytes(&p, sizeof(p), 2);
        dispatch(enc, nx_, ny_);
        enc->endEncoding();
    }
    const Reaction& reaction() const { return react_; }
    void react(Real dt) { // one reaction half-step (own command buffer)
        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        encodeReact(cmd, dt);
        cmd->commit();
        cmd->waitUntilCompleted();
    }
    // Species flux scalars (Fpx, Ssx, Fgx, 0) per face, CPU-visible.
    const Cons* sfx() const {
        return static_cast<const Cons*>(sFx_->contents());
    }
    const Cons* sfy() const {
        return static_cast<const Cons*>(sFy_->contents());
    }

    MTL::Buffer* qBuffer() const { return q_; }

    // Immersed-solid mask (1 = solid), CPU-visible, tx*ty bytes, ghosts
    // included. Default all-zero (no solid). The caller populates it.
    std::uint8_t* solidData() {
        return static_cast<std::uint8_t*>(smask_->contents());
    }

    // Step fluxes, CPU-visible (needed by AMR refluxing).
    const Cons* fx() const { return static_cast<const Cons*>(Fx_->contents()); }
    const Cons* fy() const { return static_cast<const Cons*>(Fy_->contents()); }

    struct Params {
        std::int32_t tx, ty, nx, ny;
        float dx, dy, dt;
        std::int32_t stride; // pool kernels only; 0 here
        float mu, kT;        // viscosity / heat conductivity (0 = Euler)
        float gx, gy;        // gravity (split source)
        float g1 = GAMMA, g2 = GAMMA; // two-gas gammas (species only)
        std::int32_t rks = 0;         // WENO/RK3 stage index
        float rka = 0, rkb = 1, rkw = 0; // RK update / flux weights
        float rA = 0, rEa = 0, rq = 0, rTign = 0; // reaction source
    };

private:
    Params params(Real dt) const {
        const float kT =
            mu_ > 0 ? mu_ * GAMMA / ((GAMMA - 1) * PRANDTL) : 0;
        return {tx_, ty_, nx_, ny_, dx_, dy_, dt, 0, mu_, kT, gx_, gy_,
                gas_.gamma1, gas_.gamma2};
    }

    static void dispatch(MTL::ComputeCommandEncoder* enc, int w, int h) {
        enc->dispatchThreads(MTL::Size(w, h, 1), MTL::Size(16, 16, 1));
    }

    // One encoder per kernel: Metal's hazard tracking orders them.
    void encode(MTL::CommandBuffer* cmd, MTL::ComputePipelineState* pso,
                std::initializer_list<MTL::Buffer*> bufs, const Params& p,
                int w, int h) const {
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        int slot = 0;
        for (MTL::Buffer* b : bufs) enc->setBuffer(b, 0, slot++);
        enc->setBytes(&p, sizeof(p), slot);
        dispatch(enc, w, h);
        enc->endEncoding();
    }

    MetalContext& ctx_;
    int nx_, ny_, tx_, ty_;
    Real dx_, dy_;
    Real mu_ = 0, gx_ = 0, gy_ = 0;
    bool species_ = false;
    GasPair gas_;
    Reaction react_;
    MTL::ComputePipelineState* reactP_ = nullptr;
    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *predictor_ = nullptr, *fluxX_ = nullptr,
                              *fluxY_ = nullptr, *update_ = nullptr,
                              *wave_ = nullptr;
    MTL::ComputePipelineState *predictorY_ = nullptr, *fluxXY_ = nullptr,
                              *fluxYY_ = nullptr, *updateY_ = nullptr,
                              *waveY_ = nullptr;
    MTL::Buffer *q_ = nullptr, *xL_ = nullptr, *xR_ = nullptr,
                *yB_ = nullptr, *yT_ = nullptr, *Fx_ = nullptr,
                *Fy_ = nullptr, *smax_ = nullptr, *smask_ = nullptr;
    MTL::Buffer *s_ = nullptr, *fXb_ = nullptr, *fYb_ = nullptr,
                *sFx_ = nullptr, *sFy_ = nullptr;
    bool weno_ = false, wenoSpecies_ = false;
    MTL::ComputePipelineState *wfluxX_ = nullptr, *wfluxY_ = nullptr,
                              *rkUpd_ = nullptr;
    MTL::ComputePipelineState *wfluxXY_ = nullptr, *wfluxYY_ = nullptr,
                              *rkUpdY_ = nullptr;
    MTL::Buffer *u0_ = nullptr, *FxA_ = nullptr, *FyA_ = nullptr;
    MTL::Buffer *s0_ = nullptr, *sFxA_ = nullptr, *sFyA_ = nullptr;
};

static_assert(sizeof(Cons) == 4 * sizeof(float),
              "Cons must match float4 layout");

} // namespace mm

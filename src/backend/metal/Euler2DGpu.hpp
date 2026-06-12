#pragma once

// GPU implementation of the unsplit MUSCL-Hancock step. The conserved
// state lives in a shared (unified memory) buffer: the CPU fills ghost
// cells and does I/O on the very same memory the GPU computes on.

#include "backend/metal/MetalContext.hpp"
#include "core/Grid.hpp"
#include "physics/TwoGas.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
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
    }

    ~Euler2DGpu() {
        for (MTL::Buffer* b : {q_, xL_, xR_, yB_, yT_, Fx_, Fy_, smax_})
            b->release();
        for (MTL::Buffer* b : {s_, fXb_, fYb_, sFx_, sFy_})
            if (b) b->release();
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
            encode(cmd, fluxXY_, {xL_, xR_, fXb_, Fx_, sFx_}, p, nx_ + 1,
                   ny_);
            encode(cmd, fluxYY_, {yB_, yT_, fYb_, Fy_, sFy_}, p, nx_,
                   ny_ + 1);
            encode(cmd, updateY_, {q_, s_, Fx_, Fy_, sFx_, sFy_}, p, nx_,
                   ny_);
            return;
        }
        encode(cmd, predictor_, {q_, xL_, xR_, yB_, yT_}, p, tx_ - 2,
               ty_ - 2);
        encode(cmd, fluxX_, {xL_, xR_, q_, Fx_}, p, nx_ + 1, ny_);
        encode(cmd, fluxY_, {yB_, yT_, q_, Fy_}, p, nx_, ny_ + 1);
        encode(cmd, update_, {q_, Fx_, Fy_}, p, nx_, ny_);
    }

    // Dynamic viscosity for the flux kernels (0 = inviscid Euler).
    void setViscosity(Real mu) { mu_ = mu; }
    void setGravity(Real gx, Real gy) { gx_ = gx; gy_ = gy; }

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
    // Species flux scalars (Fpx, Ssx, Fgx, 0) per face, CPU-visible.
    const Cons* sfx() const {
        return static_cast<const Cons*>(sFx_->contents());
    }
    const Cons* sfy() const {
        return static_cast<const Cons*>(sFy_->contents());
    }

    MTL::Buffer* qBuffer() const { return q_; }

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
    MTL::Library* lib_ = nullptr;
    MTL::ComputePipelineState *predictor_ = nullptr, *fluxX_ = nullptr,
                              *fluxY_ = nullptr, *update_ = nullptr,
                              *wave_ = nullptr;
    MTL::ComputePipelineState *predictorY_ = nullptr, *fluxXY_ = nullptr,
                              *fluxYY_ = nullptr, *updateY_ = nullptr,
                              *waveY_ = nullptr;
    MTL::Buffer *q_ = nullptr, *xL_ = nullptr, *xR_ = nullptr,
                *yB_ = nullptr, *yT_ = nullptr, *Fx_ = nullptr,
                *Fy_ = nullptr, *smax_ = nullptr;
    MTL::Buffer *s_ = nullptr, *fXb_ = nullptr, *fYb_ = nullptr,
                *sFx_ = nullptr, *sFy_ = nullptr;
};

static_assert(sizeof(Cons) == 4 * sizeof(float),
              "Cons must match float4 layout");

} // namespace mm

#pragma once

// Real-time view of a running AmrGpuML hierarchy: a Cocoa window with a
// CAMetalLayer; each frame draws one fullscreen triangle whose fragment
// samples the SIMULATION buffers directly (base grid, slot pool,
// per-level block->slot tables) — zero copies, the same unified memory
// the solver computes in. Keys: space = pause, q or close = quit.

#include "amr/AmrGpuML.hpp"
#include "backend/metal/MetalContext.hpp"
#include "render/CocoaShim.h"

#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <cstdint>
#include <string>

namespace mm {

class LiveView {
public:
    // Returns a non-usable view (ok() == false) when no window server
    // is available — callers degrade to a normal headless run.
    LiveView(MetalContext& ctx, const AmrGpuML& amr, int pixPerCell,
             const std::string& title, Real lo, Real hi, bool grid)
        : ctx_(ctx), amr_(amr), lo_(lo), hi_(hi),
          autoscale_(hi <= lo), grid_(grid) {
        if (autoscale_) { lo_ = Real(1e30); hi_ = Real(-1e30); }
        const GridRef b = amr.coarseRef();
        layer_ = static_cast<CA::MetalLayer*>(
            lvCreateWindow(b.nx * pixPerCell, b.ny * pixPerCell,
                           title.c_str(), ctx.device()));
        if (layer_ == nullptr) return;
        lib_ = ctx.compileLibrary("render.metal");
        MTL::Function* vs =
            lib_->newFunction(NS::String::string(
                "lv_vertex", NS::UTF8StringEncoding));
        MTL::Function* fs =
            lib_->newFunction(NS::String::string(
                "lv_fragment", NS::UTF8StringEncoding));
        MTL::RenderPipelineDescriptor* pd =
            MTL::RenderPipelineDescriptor::alloc()->init();
        pd->setVertexFunction(vs);
        pd->setFragmentFunction(fs);
        pd->colorAttachments()->object(0)->setPixelFormat(
            MTL::PixelFormatBGRA8Unorm);
        NS::Error* err = nullptr;
        pso_ = ctx.device()->newRenderPipelineState(pd, &err);
        pd->release();
        vs->release();
        fs->release();
        if (pso_ == nullptr) layer_ = nullptr;
    }

    ~LiveView() {
        if (pso_ != nullptr) pso_->release();
        if (lib_ != nullptr) lib_->release();
    }
    LiveView(const LiveView&) = delete;
    LiveView& operator=(const LiveView&) = delete;

    bool ok() const { return layer_ != nullptr; }

    // Pump events and draw one frame. Returns 0 = quit, 1 = running,
    // 2 = paused (callers should keep calling while paused).
    int frame() {
        if (!ok()) return 1;
        const int state = lvPumpEvents();
        if (state == 0) return 0;
        draw_();
        return state;
    }

private:
    struct RUniforms {
        std::int32_t nx0, ny0, blockC, levels, pTot, stride;
        float lo, hi;
        std::int32_t showGrid;
        std::int32_t ng; // ghost layers (must match core/Grid.hpp NG)
    };

    // Expand-only auto color range: when the case gave no explicit
    // range, track the running [min, max] of the base density so the
    // map covers the dynamics as they develop (e.g. a detonation whose
    // density is uniform at t=0 — a fixed IC-derived range would be
    // degenerate). Expand-only, so the colors never flicker.
    void autoRange_() {
        const GridRef b = amr_.coarseRef();
        Real mn = Real(1e30), mx = Real(-1e30);
        for (int j = NG; j < NG + b.ny; ++j)
            for (int i = NG; i < NG + b.nx; ++i) {
                const Real r = b.at(i, j).rho;
                mn = std::min(mn, r);
                mx = std::max(mx, r);
            }
        const Real pad = (mx - mn) * Real(0.03) + Real(1e-6);
        lo_ = std::min(lo_, mn - pad);
        hi_ = std::max(hi_, mx + pad);
    }

    void draw_() {
        CA::MetalDrawable* drawable = layer_->nextDrawable();
        if (drawable == nullptr) return;
        if (autoscale_) autoRange_();

        MTL::RenderPassDescriptor* rp =
            MTL::RenderPassDescriptor::alloc()->init();
        auto* att = rp->colorAttachments()->object(0);
        att->setTexture(drawable->texture());
        att->setLoadAction(MTL::LoadActionClear);
        att->setStoreAction(MTL::StoreActionStore);
        att->setClearColor(MTL::ClearColor(0, 0, 0, 1));

        MTL::CommandBuffer* cmd = ctx_.queue()->commandBuffer();
        MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(rp);
        enc->setRenderPipelineState(pso_);

        const GridRef b = amr_.coarseRef();
        const RUniforms u{b.nx,
                          b.ny,
                          amr_.renderBlockC(),
                          amr_.numLevels(),
                          amr_.renderPTot(),
                          amr_.renderStride(),
                          float(lo_),
                          float(hi_),
                          grid_ ? 1 : 0,
                          NG};
        enc->setFragmentBuffer(amr_.renderBaseQ(), 0, 0);
        enc->setFragmentBuffer(amr_.renderPoolQ(), 0, 1);
        enc->setFragmentBytes(&u, sizeof(u), 2);
        for (int slot = 3; slot <= 5; ++slot) {
            const int l = slot - 2;
            // unused level bindings get a harmless dummy buffer
            MTL::Buffer* tbl =
                amr_.numLevels() > 1
                    ? amr_.renderSlotTable(
                          std::min(l, amr_.numLevels() - 1))
                    : amr_.renderBaseQ();
            enc->setFragmentBuffer(tbl, 0, slot);
        }
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0),
                            NS::UInteger(3));
        enc->endEncoding();
        cmd->presentDrawable(drawable);
        cmd->commit();
        rp->release();
    }

    MetalContext& ctx_;
    const AmrGpuML& amr_;
    Real lo_, hi_;
    bool autoscale_;
    bool grid_;
    CA::MetalLayer* layer_ = nullptr;
    MTL::Library* lib_ = nullptr;
    MTL::RenderPipelineState* pso_ = nullptr;
};

} // namespace mm

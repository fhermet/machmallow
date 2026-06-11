#pragma once

#include <Metal/Metal.hpp>

#include <string>

namespace mm {

// Owns the Metal device and command queue, and compiles .metal sources at
// runtime (newLibrary from source), so no offline metal compiler is needed.
class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    MTL::Device* device() const { return device_; }
    MTL::CommandQueue* queue() const { return queue_; }

    // Compile a .metal file (path relative to MACHMALLOW_SHADER_DIR, or
    // absolute) and return the library. Caller releases.
    MTL::Library* compileLibrary(const std::string& path) const;

    // Build a compute pipeline for a kernel function of the library.
    // Caller releases.
    MTL::ComputePipelineState* makePipeline(MTL::Library* lib,
                                            const std::string& function) const;

private:
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
};

} // namespace mm
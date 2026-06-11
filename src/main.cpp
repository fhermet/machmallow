// Phase 0 driver: validate the Metal toolchain end to end with a saxpy
// benchmark — runtime shader compilation, shared (zero-copy) buffers,
// GPU timing, CPU reference, and a correctness check.

#include "backend/metal/MetalContext.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

void cpuSaxpy(float* y, const float* x, float a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

} // namespace

int main() {
    constexpr std::size_t N = 1u << 25; // 32M floats, 128 MB per buffer
    constexpr float A = 2.0f;
    constexpr int ITERS = 20;

    mm::MetalContext ctx;
    MTL::Device* dev = ctx.device();
    std::printf("GPU: %s\n", dev->name()->utf8String());
    std::printf("max working set: %.1f GB\n",
                double(dev->recommendedMaxWorkingSetSize()) / 1e9);

    MTL::Library* lib = ctx.compileLibrary("saxpy.metal");
    MTL::ComputePipelineState* pso = ctx.makePipeline(lib, "saxpy");

    // Shared storage: CPU and GPU see the same memory, no copies.
    MTL::Buffer* xBuf =
        dev->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* yBuf =
        dev->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared);
    auto* x = static_cast<float*>(xBuf->contents());
    auto* y = static_cast<float*>(yBuf->contents());

    for (std::size_t i = 0; i < N; ++i) {
        x[i] = float(i % 1000) * 0.001f;
        y[i] = 1.0f;
    }

    const auto runGpu = [&] {
        MTL::CommandBuffer* cmd = ctx.queue()->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(yBuf, 0, 0);
        enc->setBuffer(xBuf, 0, 1);
        enc->setBytes(&A, sizeof(A), 2);
        const uint32_t n32 = uint32_t(N);
        enc->setBytes(&n32, sizeof(n32), 3);
        const NS::UInteger tg =
            std::min<NS::UInteger>(pso->maxTotalThreadsPerThreadgroup(), 256);
        enc->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(tg, 1, 1));
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return cmd->GPUEndTime() - cmd->GPUStartTime();
    };

    // --- Correctness: one GPU pass vs one CPU pass on a copy -------------
    std::vector<float> yRef(y, y + N);
    runGpu();
    cpuSaxpy(yRef.data(), x, A, N);
    double maxErr = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        maxErr = std::max(maxErr, double(std::fabs(y[i] - yRef[i])));
    }
    std::printf("max |gpu - cpu| = %.3e  %s\n", maxErr,
                maxErr == 0.0 ? "(exact)" : "");
    if (maxErr > 1e-6) {
        std::fprintf(stderr, "FAIL: GPU result differs from CPU\n");
        return EXIT_FAILURE;
    }

    // --- GPU bandwidth: best of ITERS (read x, read+write y = 12 B/elem) -
    double gpuBest = 1e30;
    for (int it = 0; it < ITERS; ++it) {
        gpuBest = std::min(gpuBest, runGpu());
    }
    const double bytes = 3.0 * double(N) * sizeof(float);
    std::printf("GPU saxpy: %.3f ms, %.1f GB/s\n", gpuBest * 1e3,
                bytes / gpuBest / 1e9);

    // --- CPU reference (single thread) -----------------------------------
    double cpuBest = 1e30;
    for (int it = 0; it < 5; ++it) {
        const auto t0 = Clock::now();
        cpuSaxpy(y, x, A, N);
        cpuBest = std::min(cpuBest, seconds(t0, Clock::now()));
    }
    std::printf("CPU saxpy (1 thread): %.3f ms, %.1f GB/s\n", cpuBest * 1e3,
                bytes / cpuBest / 1e9);
    std::printf("speedup: %.1fx\n", cpuBest / gpuBest);

    yBuf->release();
    xBuf->release();
    pso->release();
    lib->release();
    return EXIT_SUCCESS;
}

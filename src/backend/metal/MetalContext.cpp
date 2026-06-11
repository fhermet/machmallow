#include "backend/metal/MetalContext.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mm {

namespace {

std::string readFile(const std::string& path) {
    std::filesystem::path p(path);
    if (p.is_relative()) {
        p = std::filesystem::path(MACHMALLOW_SHADER_DIR) / p;
    }
    std::ifstream in(p);
    if (!in) {
        throw std::runtime_error("cannot open shader file: " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[noreturn]] void throwNSError(const std::string& what, NS::Error* error) {
    std::string msg = what;
    if (error != nullptr) {
        msg += ": ";
        msg += error->localizedDescription()->utf8String();
    }
    throw std::runtime_error(msg);
}

} // namespace

MetalContext::MetalContext() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (device_ == nullptr) {
        throw std::runtime_error("no Metal device available");
    }
    queue_ = device_->newCommandQueue();
    if (queue_ == nullptr) {
        throw std::runtime_error("failed to create command queue");
    }
}

MetalContext::~MetalContext() {
    if (queue_ != nullptr) queue_->release();
    if (device_ != nullptr) device_->release();
}

MTL::Library* MetalContext::compileLibrary(const std::string& path) const {
    const std::string source = readFile(path);
    NS::String* src =
        NS::String::string(source.c_str(), NS::UTF8StringEncoding);

    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(true);

    NS::Error* error = nullptr;
    MTL::Library* lib = device_->newLibrary(src, options, &error);
    options->release();
    if (lib == nullptr) {
        throwNSError("shader compilation failed (" + path + ")", error);
    }
    return lib;
}

MTL::ComputePipelineState* MetalContext::makePipeline(
    MTL::Library* lib, const std::string& function) const {
    NS::String* name =
        NS::String::string(function.c_str(), NS::UTF8StringEncoding);
    MTL::Function* fn = lib->newFunction(name);
    if (fn == nullptr) {
        throw std::runtime_error("kernel function not found: " + function);
    }
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pso =
        device_->newComputePipelineState(fn, &error);
    fn->release();
    if (pso == nullptr) {
        throwNSError("pipeline creation failed (" + function + ")", error);
    }
    return pso;
}

} // namespace mm

#pragma once

// Minimal parallel-for on top of GCD (libdispatch): uses the pre-warmed
// system thread pool, so per-call overhead is negligible — exactly what
// the per-step AMR CPU work (many small independent patches) needs.

#include <cstddef>
#include <dispatch/dispatch.h>

namespace mm {

template <class F>
inline void parallelFor(std::size_t n, F&& f) {
    if (n == 0) return;
    if (n == 1) {
        f(std::size_t(0));
        return;
    }
    struct Ctx {
        F* f;
    } ctx{&f};
    dispatch_apply_f(n, DISPATCH_APPLY_AUTO, &ctx,
                     [](void* c, std::size_t i) {
                         (*static_cast<Ctx*>(c)->f)(i);
                     });
}

} // namespace mm

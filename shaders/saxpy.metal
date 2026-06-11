#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device float* y [[buffer(0)]],
                  device const float* x [[buffer(1)]],
                  constant float& a [[buffer(2)]],
                  constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]])
{
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

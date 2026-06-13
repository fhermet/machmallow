// Live view: one fullscreen triangle whose fragment walks the AMR
// hierarchy finest-first and samples the SIMULATION buffers directly
// (zero copy): per-level block->slot tables say which pool slot covers
// the pixel; otherwise the base grid is sampled. Density is mapped
// through a viridis-like gradient; optional patch outlines.

#include <metal_stdlib>
using namespace metal;

struct RUniforms {
    int nx0, ny0;     // base interior cells
    int blockC;       // block size (patch = 2*blockC cells)
    int levels;       // total levels incl. base (1..4)
    int pTot, stride; // pool slot layout
    float lo, hi;     // density color range
    int showGrid;     // 1 = draw patch outlines
    int ng;           // ghost layers (cell (i,j) lives at index i+ng)
};

struct VSOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VSOut lv_vertex(uint vid [[vertex_id]]) {
    // fullscreen triangle
    const float2 xy = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos = float4(xy * 2.0 - 1.0, 0, 1);
    o.uv = float2(xy.x, xy.y); // y already flips with clip space below
    o.pos.y = -o.pos.y;        // domain y up
    return o;
}

inline float3 colormap(float s) {
    s = saturate(s);
    // 5-stop viridis-like gradient
    const float3 c0 = float3(0.267, 0.005, 0.329);
    const float3 c1 = float3(0.229, 0.322, 0.546);
    const float3 c2 = float3(0.127, 0.567, 0.551);
    const float3 c3 = float3(0.369, 0.789, 0.383);
    const float3 c4 = float3(0.993, 0.906, 0.144);
    const float t = s * 4.0;
    if (t < 1.0) return mix(c0, c1, t);
    if (t < 2.0) return mix(c1, c2, t - 1.0);
    if (t < 3.0) return mix(c2, c3, t - 2.0);
    return mix(c3, c4, t - 3.0);
}

fragment float4 lv_fragment(VSOut in [[stage_in]],
                            device const float4* baseQ [[buffer(0)]],
                            device const float4* pool [[buffer(1)]],
                            constant RUniforms& U [[buffer(2)]],
                            device const uint* slots1 [[buffer(3)]],
                            device const uint* slots2 [[buffer(4)]],
                            device const uint* slots3 [[buffer(5)]])
{
    const int nf = 2 * U.blockC;
    float rho = 0.0;
    float edge = 0.0;
    bool found = false;

    for (int l = U.levels - 1; l >= 1 && !found; --l) {
        device const uint* tbl =
            l == 1 ? slots1 : (l == 2 ? slots2 : slots3);
        const int nxl = U.nx0 << l, nyl = U.ny0 << l;
        const int cx = clamp(int(in.uv.x * nxl), 0, nxl - 1);
        const int cy = clamp(int(in.uv.y * nyl), 0, nyl - 1);
        const uint slot = tbl[(cy / nf) * (nxl / nf) + cx / nf];
        if (slot == 0xffffffffu) continue;
        const int lx = cx % nf, ly = cy % nf;
        rho = pool[slot * U.stride + (ly + U.ng) * U.pTot +
                   (lx + U.ng)].x;
        if (U.showGrid != 0 &&
            (lx == 0 || ly == 0 || lx == nf - 1 || ly == nf - 1))
            edge = 0.25 + 0.15 * float(l);
        found = true;
    }
    if (!found) {
        const int cx = clamp(int(in.uv.x * U.nx0), 0, U.nx0 - 1);
        const int cy = clamp(int(in.uv.y * U.ny0), 0, U.ny0 - 1);
        rho = baseQ[(cy + U.ng) * (U.nx0 + 2 * U.ng) + cx + U.ng].x;
    }

    float3 c = colormap((rho - U.lo) / max(U.hi - U.lo, 1e-9f));
    c = mix(c, float3(0.0), edge);
    return float4(c, 1.0);
}

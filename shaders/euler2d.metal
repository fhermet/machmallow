// GPU port of the unsplit MUSCL-Hancock + HLLC step. Conserved state is a
// float4 (rho, mx, my, E), primitive is a float4 (rho, u, v, p) — same
// memory layout as the C++ Cons struct.
//
// Each kernel exists in two forms: the plain one operating on a single
// grid, and a `_pool` variant where gid.z selects a patch slot inside
// pooled buffers (all AMR patches advanced in one dispatch).

#include <metal_stdlib>
using namespace metal;

constant float GAMMA = 1.4f;
constant float RHO_FLOOR = 1e-10f;
constant float P_FLOOR = 1e-10f;
constant int NG = 2;

struct Params {
    int tx, ty;   // total cells including ghosts
    int nx, ny;   // interior cells
    float dx, dy, dt;
    int stride;   // cells per pool slot (0 for plain kernels)
    float mu, kT; // dynamic viscosity and heat conductivity (0 = Euler)
};

// ---- physics ------------------------------------------------------------

inline float4 toPrim(float4 q) {
    const float rho = max(q.x, RHO_FLOOR);
    const float u = q.y / rho;
    const float v = q.z / rho;
    const float p =
        max((GAMMA - 1.0f) * (q.w - 0.5f * rho * (u * u + v * v)), P_FLOOR);
    return float4(rho, u, v, p);
}

inline float4 toCons(float4 w) {
    const float ke = 0.5f * w.x * (w.y * w.y + w.z * w.z);
    return float4(w.x, w.x * w.y, w.x * w.z, w.w / (GAMMA - 1.0f) + ke);
}

inline float soundSpeed(float4 w) { return sqrt(GAMMA * w.w / w.x); }

inline float4 fluxX(float4 w) {
    const float4 q = toCons(w);
    return float4(q.y, q.y * w.y + w.w, q.z * w.y, (q.w + w.w) * w.y);
}

inline float4 fluxY(float4 w) {
    const float4 q = toCons(w);
    return float4(q.z, q.y * w.z, q.z * w.z + w.w, (q.w + w.w) * w.z);
}

// ---- numerics -----------------------------------------------------------

// Component-wise MC limited slope.
inline float4 limitedSlope(float4 qm, float4 q0, float4 qp) {
    const float4 dm = q0 - qm;
    const float4 dp = qp - q0;
    const float4 c = 0.5f * (dm + dp);
    const float4 lim = 2.0f * min(abs(dm), abs(dp));
    const float4 s = copysign(min(abs(c), lim), c);
    return select(s, float4(0.0f), dm * dp <= 0.0f);
}

inline float4 hllcSideFluxX(float4 w, float S, float Ss) {
    const float4 q = toCons(w);
    const float4 f = fluxX(w);
    const float coef = w.x * (S - w.y) / (S - Ss);
    const float4 qs = float4(
        coef, coef * Ss, coef * w.z,
        coef * (q.w / w.x + (Ss - w.y) * (Ss + w.w / (w.x * (S - w.y)))));
    return f + S * (qs - q);
}

inline float4 hllcFluxX(float4 L, float4 R) {
    const float cL = soundSpeed(L);
    const float cR = soundSpeed(R);

    const float rhoBar = 0.5f * (L.x + R.x);
    const float cBar = 0.5f * (cL + cR);
    const float pPvrs =
        0.5f * (L.w + R.w) - 0.5f * (R.y - L.y) * rhoBar * cBar;
    const float pStar = max(0.0f, pPvrs);

    const float qfL = pStar <= L.w
        ? 1.0f
        : sqrt(1.0f + (GAMMA + 1.0f) / (2.0f * GAMMA) * (pStar / L.w - 1.0f));
    const float qfR = pStar <= R.w
        ? 1.0f
        : sqrt(1.0f + (GAMMA + 1.0f) / (2.0f * GAMMA) * (pStar / R.w - 1.0f));
    const float SL = L.y - cL * qfL;
    const float SR = R.y + cR * qfR;

    const float Ss =
        (R.w - L.w + L.x * L.y * (SL - L.y) - R.x * R.y * (SR - R.y)) /
        (L.x * (SL - L.y) - R.x * (SR - R.y));

    if (SL >= 0.0f) return fluxX(L);
    if (Ss >= 0.0f) return hllcSideFluxX(L, SL, Ss);
    if (SR >= 0.0f) return hllcSideFluxX(R, SR, Ss);
    return fluxX(R);
}

inline float4 hllcFluxY(float4 L, float4 R) {
    const float4 f =
        hllcFluxX(float4(L.x, L.z, L.y, L.w), float4(R.x, R.z, R.y, R.w));
    return float4(f.x, f.z, f.y, f.w);
}

// ---- kernel bodies --------------------------------------------------------

inline void predictorBody(device const float4* q, device float4* xL,
                          device float4* xR, device float4* yB,
                          device float4* yT, constant Params& P, int base,
                          int i, int j) {
    const int id = base + j * P.tx + i;

    const float4 q0 = q[id];
    const float4 dqx = limitedSlope(q[id - 1], q0, q[id + 1]);
    const float4 dqy = limitedSlope(q[id - P.tx], q0, q[id + P.tx]);

    const float4 xl = q0 - 0.5f * dqx;
    const float4 xr = q0 + 0.5f * dqx;
    const float4 yb = q0 - 0.5f * dqy;
    const float4 yt = q0 + 0.5f * dqy;

    const float hx = 0.5f * P.dt / P.dx;
    const float hy = 0.5f * P.dt / P.dy;
    const float4 adv = hx * (fluxX(toPrim(xl)) - fluxX(toPrim(xr))) +
                       hy * (fluxY(toPrim(yb)) - fluxY(toPrim(yt)));

    xL[id] = xl + adv;
    xR[id] = xr + adv;
    yB[id] = yb + adv;
    yT[id] = yt + adv;
}

inline void fluxXBody(device const float4* xL, device const float4* xR,
                      device const float4* q, device float4* Fx,
                      constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    float4 F = hllcFluxX(toPrim(xR[id]), toPrim(xL[id + 1]));
    if (P.mu > 0.0f) {
        // Central viscous flux from t^n cell values (mirrors the CPU).
        const float4 w00 = toPrim(q[id]);
        const float4 w10 = toPrim(q[id + 1]);
        const float4 w0p = toPrim(q[id + P.tx]);
        const float4 w0m = toPrim(q[id - P.tx]);
        const float4 w1p = toPrim(q[id + P.tx + 1]);
        const float4 w1m = toPrim(q[id - P.tx + 1]);
        const float ux = (w10.y - w00.y) / P.dx;
        const float vx = (w10.z - w00.z) / P.dx;
        const float Tx = (w10.w / w10.x - w00.w / w00.x) / P.dx;
        const float uy = ((w0p.y + w1p.y) - (w0m.y + w1m.y)) / (4.0f * P.dy);
        const float vy = ((w0p.z + w1p.z) - (w0m.z + w1m.z)) / (4.0f * P.dy);
        const float txx = P.mu * ((4.0f / 3.0f) * ux - (2.0f / 3.0f) * vy);
        const float txy = P.mu * (uy + vx);
        const float ub = 0.5f * (w00.y + w10.y);
        const float vb = 0.5f * (w00.z + w10.z);
        F.y -= txx;
        F.z -= txy;
        F.w -= ub * txx + vb * txy + P.kT * Tx;
    }
    Fx[id] = F;
}

inline void fluxYBody(device const float4* yB, device const float4* yT,
                      device const float4* q, device float4* Fy,
                      constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    float4 F = hllcFluxY(toPrim(yT[id]), toPrim(yB[id + P.tx]));
    if (P.mu > 0.0f) {
        const float4 w00 = toPrim(q[id]);
        const float4 w01 = toPrim(q[id + P.tx]);
        const float4 wp0 = toPrim(q[id + 1]);
        const float4 wm0 = toPrim(q[id - 1]);
        const float4 wp1 = toPrim(q[id + P.tx + 1]);
        const float4 wm1 = toPrim(q[id + P.tx - 1]);
        const float uy = (w01.y - w00.y) / P.dy;
        const float vy = (w01.z - w00.z) / P.dy;
        const float Ty = (w01.w / w01.x - w00.w / w00.x) / P.dy;
        const float ux = ((wp0.y + wp1.y) - (wm0.y + wm1.y)) / (4.0f * P.dx);
        const float vx = ((wp0.z + wp1.z) - (wm0.z + wm1.z)) / (4.0f * P.dx);
        const float txy = P.mu * (uy + vx);
        const float tyy = P.mu * ((4.0f / 3.0f) * vy - (2.0f / 3.0f) * ux);
        const float ub = 0.5f * (w00.y + w01.y);
        const float vb = 0.5f * (w00.z + w01.z);
        F.y -= txy;
        F.z -= tyy;
        F.w -= ub * txy + vb * tyy + P.kT * Ty;
    }
    Fy[id] = F;
}

inline void updateBody(device float4* q, device const float4* Fx,
                       device const float4* Fy, constant Params& P,
                       int base, int i, int j) {
    const int id = base + j * P.tx + i;
    q[id] += (P.dt / P.dx) * (Fx[id - 1] - Fx[id]) +
             (P.dt / P.dy) * (Fy[id - P.tx] - Fy[id]);
}

// Max wave speed per direction: simdgroup reduction, then one atomic max
// per simdgroup. Positive float bit patterns compare like uints.
inline void waveBody(device const float4* q, device atomic_uint* smax,
                     constant Params& P, int base, int i, int j) {
    const float4 w = toPrim(q[base + j * P.tx + i]);
    const float c = soundSpeed(w);

    const float sx = simd_max(fabs(w.y) + c);
    const float sy = simd_max(fabs(w.z) + c);
    if (simd_is_first()) {
        atomic_fetch_max_explicit(&smax[0], as_type<uint>(sx),
                                  memory_order_relaxed);
        atomic_fetch_max_explicit(&smax[1], as_type<uint>(sy),
                                  memory_order_relaxed);
    }
}

// ---- plain kernels (one grid) ---------------------------------------------

kernel void predictor(device const float4* q [[buffer(0)]],
                      device float4* xL [[buffer(1)]],
                      device float4* xR [[buffer(2)]],
                      device float4* yB [[buffer(3)]],
                      device float4* yT [[buffer(4)]],
                      constant Params& P [[buffer(5)]],
                      uint2 g [[thread_position_in_grid]])
{
    predictorBody(q, xL, xR, yB, yT, P, 0, int(g.x) + 1, int(g.y) + 1);
}

kernel void flux_x(device const float4* xL [[buffer(0)]],
                   device const float4* xR [[buffer(1)]],
                   device const float4* q [[buffer(2)]],
                   device float4* Fx [[buffer(3)]],
                   constant Params& P [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]])
{
    fluxXBody(xL, xR, q, Fx, P, 0, int(g.x) + NG - 1, int(g.y) + NG);
}

kernel void flux_y(device const float4* yB [[buffer(0)]],
                   device const float4* yT [[buffer(1)]],
                   device const float4* q [[buffer(2)]],
                   device float4* Fy [[buffer(3)]],
                   constant Params& P [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]])
{
    fluxYBody(yB, yT, q, Fy, P, 0, int(g.x) + NG, int(g.y) + NG - 1);
}

kernel void update_cons(device float4* q [[buffer(0)]],
                        device const float4* Fx [[buffer(1)]],
                        device const float4* Fy [[buffer(2)]],
                        constant Params& P [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]])
{
    updateBody(q, Fx, Fy, P, 0, int(g.x) + NG, int(g.y) + NG);
}

kernel void wave_speed(device const float4* q [[buffer(0)]],
                       device atomic_uint* smax [[buffer(1)]],
                       constant Params& P [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]])
{
    waveBody(q, smax, P, 0, int(g.x) + NG, int(g.y) + NG);
}

// ---- pool kernels (all AMR patches in one dispatch) -----------------------

kernel void predictor_pool(device const float4* q [[buffer(0)]],
                           device float4* xL [[buffer(1)]],
                           device float4* xR [[buffer(2)]],
                           device float4* yB [[buffer(3)]],
                           device float4* yT [[buffer(4)]],
                           constant Params& P [[buffer(5)]],
                           device const uint* slots [[buffer(6)]],
                           uint3 g [[thread_position_in_grid]])
{
    predictorBody(q, xL, xR, yB, yT, P, int(slots[g.z]) * P.stride,
                  int(g.x) + 1, int(g.y) + 1);
}

kernel void flux_x_pool(device const float4* xL [[buffer(0)]],
                        device const float4* xR [[buffer(1)]],
                        device const float4* q [[buffer(2)]],
                        device float4* Fx [[buffer(3)]],
                        constant Params& P [[buffer(4)]],
                        device const uint* slots [[buffer(5)]],
                        uint3 g [[thread_position_in_grid]])
{
    fluxXBody(xL, xR, q, Fx, P, int(slots[g.z]) * P.stride,
              int(g.x) + NG - 1, int(g.y) + NG);
}

kernel void flux_y_pool(device const float4* yB [[buffer(0)]],
                        device const float4* yT [[buffer(1)]],
                        device const float4* q [[buffer(2)]],
                        device float4* Fy [[buffer(3)]],
                        constant Params& P [[buffer(4)]],
                        device const uint* slots [[buffer(5)]],
                        uint3 g [[thread_position_in_grid]])
{
    fluxYBody(yB, yT, q, Fy, P, int(slots[g.z]) * P.stride, int(g.x) + NG,
              int(g.y) + NG - 1);
}

kernel void update_pool(device float4* q [[buffer(0)]],
                        device const float4* Fx [[buffer(1)]],
                        device const float4* Fy [[buffer(2)]],
                        constant Params& P [[buffer(3)]],
                        device const uint* slots [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]])
{
    updateBody(q, Fx, Fy, P, int(slots[g.z]) * P.stride, int(g.x) + NG,
               int(g.y) + NG);
}

kernel void wave_pool(device const float4* q [[buffer(0)]],
                      device atomic_uint* smax [[buffer(1)]],
                      constant Params& P [[buffer(2)]],
                      device const uint* slots [[buffer(3)]],
                      uint3 g [[thread_position_in_grid]])
{
    waveBody(q, smax, P, int(slots[g.z]) * P.stride, int(g.x) + NG,
             int(g.y) + NG);
}

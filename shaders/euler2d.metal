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
constant int NG = 3;

struct Params {
    int tx, ty;   // total cells including ghosts
    int nx, ny;   // interior cells
    float dx, dy, dt;
    int stride;   // cells per pool slot (0 for plain kernels)
    float mu, kT; // dynamic viscosity and heat conductivity (0 = Euler)
    float gx, gy; // gravity (split source in the update kernel)
    float g1, g2; // two-gas gammas (species kernels only)
    int rks;            // WENO/RK3: stage index (0, 1, 2)
    float rka, rkb, rkw; // q = rka*u0 + rkb*(q + dt L); flux weight
    float rA, rEa, rq, rTign; // single-step reaction (source kernels)
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

    float4 fxl = xl + adv, fxr = xr + adv;
    float4 fyb = yb + adv, fyt = yt + adv;
    // Positivity fallback to first order (mirrors the CPU path).
    const auto bad = [](float4 c) {
        if (c.x <= RHO_FLOOR) return true;
        const float ke = 0.5f * (c.y * c.y + c.z * c.z) / c.x;
        return c.w - ke <= 0.0f;
    };
    if (bad(fxl) || bad(fxr) || bad(fyb) || bad(fyt)) {
        fxl = fxr = fyb = fyt = q0;
    }
    xL[id] = fxl;
    xR[id] = fxr;
    yB[id] = fyb;
    yT[id] = fyt;
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
    float4 qn = q[id] + (P.dt / P.dx) * (Fx[id - 1] - Fx[id]) +
                (P.dt / P.dy) * (Fy[id - P.tx] - Fy[id]);
    if (P.gx != 0.0f || P.gy != 0.0f) {
        // gravity split source; same arithmetic order as the CPU path
        const float rho = max(qn.x, RHO_FLOOR);
        const float mx0 = qn.y, my0 = qn.z;
        qn.y += P.dt * rho * P.gx;
        qn.z += P.dt * rho * P.gy;
        qn.w += 0.5f * P.dt *
                (P.gx * (mx0 + qn.y) + P.gy * (my0 + qn.z));
    }
    q[id] = qn;
}

// Max wave speed per direction + min density (viscous dt limit):
// simdgroup reduction, then one atomic per simdgroup. Positive float bit
// patterns compare like uints.
inline void waveBody(device const float4* q, device atomic_uint* smax,
                     constant Params& P, int base, int i, int j) {
    const float4 w = toPrim(q[base + j * P.tx + i]);
    const float c = soundSpeed(w);

    const float sx = simd_max(fabs(w.y) + c);
    const float sy = simd_max(fabs(w.z) + c);
    const float rmn = simd_min(w.x);
    if (simd_is_first()) {
        atomic_fetch_max_explicit(&smax[0], as_type<uint>(sx),
                                  memory_order_relaxed);
        atomic_fetch_max_explicit(&smax[1], as_type<uint>(sy),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&smax[2], as_type<uint>(rmn),
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

// ---- two-gas kernels (species transport, mirrors Muscl2DSpecies.hpp) ------
//
// Cell scalars ride in a float2 buffer s = (phi, Gamma); the predictor
// emits packed face buffers fX = (yxL, yxR, gxL, gxR) and
// fY = (yyB, yyT, gyB, gyT); the flux kernels emit packed scalar fluxes
// sFx = (Fpx, Ssx, Fgx, 0). Reconstruction is PRIMITIVE + Gamma with the
// face Gamma advanced by half dt (see the CPU file for the why).

inline float4 toPrimG(float4 q, float G) {
    const float rho = max(q.x, RHO_FLOOR);
    const float u = q.y / rho;
    const float v = q.z / rho;
    const float p = max((q.w - 0.5f * rho * (u * u + v * v)) / G, P_FLOOR);
    return float4(rho, u, v, p);
}
inline float4 toConsG(float4 w, float G) {
    const float ke = 0.5f * w.x * (w.y * w.y + w.z * w.z);
    return float4(w.x, w.x * w.y, w.x * w.z, w.w * G + ke);
}
inline float4 fluxXG(float4 w, float G) {
    const float4 q = toConsG(w, G);
    return float4(q.y, q.y * w.y + w.w, q.z * w.y, (q.w + w.w) * w.y);
}
inline float4 fluxYG(float4 w, float G) {
    const float4 q = toConsG(w, G);
    return float4(q.z, q.y * w.z, q.z * w.z + w.w, (q.w + w.w) * w.z);
}

inline float mcSlope1(float dm, float dp) {
    if (dm * dp <= 0.0f) return 0.0f;
    const float c = 0.5f * (dm + dp);
    return copysign(min(abs(c), 2.0f * min(abs(dm), abs(dp))), c);
}

// HLLC with per-side gamma; also returns the contact speed Ss.
inline float4 hllcFluxXG(float4 L, float gL, float4 R, float gR,
                         thread float& SsOut) {
    const float cL = sqrt(gL * L.w / L.x);
    const float cR = sqrt(gR * R.w / R.x);
    const float GL = 1.0f / (gL - 1.0f), GR = 1.0f / (gR - 1.0f);

    const float rhoBar = 0.5f * (L.x + R.x);
    const float cBar = 0.5f * (cL + cR);
    const float pPvrs =
        0.5f * (L.w + R.w) - 0.5f * (R.y - L.y) * rhoBar * cBar;
    const float pStar = max(0.0f, pPvrs);

    const float qfL = pStar <= L.w
        ? 1.0f
        : sqrt(1.0f + (gL + 1.0f) / (2.0f * gL) * (pStar / L.w - 1.0f));
    const float qfR = pStar <= R.w
        ? 1.0f
        : sqrt(1.0f + (gR + 1.0f) / (2.0f * gR) * (pStar / R.w - 1.0f));
    const float SL = L.y - cL * qfL;
    const float SR = R.y + cR * qfR;
    const float Ss =
        (R.w - L.w + L.x * L.y * (SL - L.y) - R.x * R.y * (SR - R.y)) /
        (L.x * (SL - L.y) - R.x * (SR - R.y));
    SsOut = Ss;

    const auto side = [&](float4 w, float G, float S) {
        const float4 q = toConsG(w, G);
        const float4 f =
            float4(q.y, q.y * w.y + w.w, q.z * w.y, (q.w + w.w) * w.y);
        const float coef = w.x * (S - w.y) / (S - Ss);
        const float4 qs = float4(
            coef, coef * Ss, coef * w.z,
            coef * (q.w / w.x +
                    (Ss - w.y) * (Ss + w.w / (w.x * (S - w.y)))));
        return f + S * (qs - q);
    };
    if (SL >= 0.0f) return fluxXG(L, GL);
    if (Ss >= 0.0f) return side(L, GL, SL);
    if (SR >= 0.0f) return side(R, GR, SR);
    return fluxXG(R, GR);
}

inline void predictorYBody(device const float4* q, device const float2* sc,
                           device float4* xL, device float4* xR,
                           device float4* yB, device float4* yT,
                           device float4* fX, device float4* fY,
                           constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    const int iw = id - 1, ie = id + 1;
    const int js = id - P.tx, jn = id + P.tx;

    const float4 q0 = q[id];
    const float hx = 0.5f * P.dt / P.dx;
    const float hy = 0.5f * P.dt / P.dy;
    const float Gmin = min(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    const float Gmax = max(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));

    const auto Yof = [&](int a) {
        return clamp(sc[a].x / max(q[a].x, RHO_FLOOR), 0.0f, 1.0f);
    };
    const float Y0 = Yof(id);
    const float dYx = mcSlope1(Y0 - Yof(iw), Yof(ie) - Y0);
    const float dYy = mcSlope1(Y0 - Yof(js), Yof(jn) - Y0);
    float4 fxv = clamp(float4(Y0 - 0.5f * dYx, Y0 + 0.5f * dYx, 0.0f, 0.0f),
                       0.0f, 1.0f);
    float4 fyv = clamp(float4(Y0 - 0.5f * dYy, Y0 + 0.5f * dYy, 0.0f, 0.0f),
                       0.0f, 1.0f);

    const float G0 = sc[id].y;
    const float dGx = mcSlope1(G0 - sc[iw].y, sc[ie].y - G0);
    const float dGy = mcSlope1(G0 - sc[js].y, sc[jn].y - G0);
    const float u0c = q0.y / max(q0.x, RHO_FLOOR);
    const float v0c = q0.z / max(q0.x, RHO_FLOOR);
    const float gAdv = hx * u0c * dGx + hy * v0c * dGy;
    fxv.z = clamp(G0 - 0.5f * dGx - gAdv, Gmin, Gmax);
    fxv.w = clamp(G0 + 0.5f * dGx - gAdv, Gmin, Gmax);
    fyv.z = clamp(G0 - 0.5f * dGy - gAdv, Gmin, Gmax);
    fyv.w = clamp(G0 + 0.5f * dGy - gAdv, Gmin, Gmax);

    // primitive slopes; face states rebuilt from face prims + face Gamma
    const float4 w0 = toPrimG(q0, G0);
    const float4 wm = toPrimG(q[iw], sc[iw].y);
    const float4 wp = toPrimG(q[ie], sc[ie].y);
    const float4 wb = toPrimG(q[js], sc[js].y);
    const float4 wt = toPrimG(q[jn], sc[jn].y);
    float4 dwx, dwy;
    for (int c = 0; c < 4; ++c) {
        dwx[c] = mcSlope1(w0[c] - wm[c], wp[c] - w0[c]);
        dwy[c] = mcSlope1(w0[c] - wb[c], wt[c] - w0[c]);
    }
    const auto face = [&](float4 w, float sgn, float4 d) {
        float4 f = w + sgn * 0.5f * d;
        f.x = max(f.x, RHO_FLOOR);
        f.w = max(f.w, P_FLOOR);
        return f;
    };
    const float4 pxl = face(w0, -1.0f, dwx);
    const float4 pxr = face(w0, +1.0f, dwx);
    const float4 pyb = face(w0, -1.0f, dwy);
    const float4 pyt = face(w0, +1.0f, dwy);

    const float4 adv = hx * (fluxXG(pxl, fxv.z) - fluxXG(pxr, fxv.w)) +
                       hy * (fluxYG(pyb, fyv.z) - fluxYG(pyt, fyv.w));
    float4 fxl = toConsG(pxl, fxv.z) + adv;
    float4 fxr = toConsG(pxr, fxv.w) + adv;
    float4 fyb = toConsG(pyb, fyv.z) + adv;
    float4 fyt = toConsG(pyt, fyv.w) + adv;

    const auto bad = [](float4 c) {
        if (c.x <= RHO_FLOOR) return true;
        const float ke = 0.5f * (c.y * c.y + c.z * c.z) / c.x;
        return c.w - ke <= 0.0f;
    };
    if (bad(fxl) || bad(fxr) || bad(fyb) || bad(fyt)) {
        fxl = fxr = fyb = fyt = q0;
        fxv = float4(Y0, Y0, G0, G0);
        fyv = float4(Y0, Y0, G0, G0);
    }
    xL[id] = fxl;
    xR[id] = fxr;
    yB[id] = fyb;
    yT[id] = fyt;
    fX[id] = fxv;
    fY[id] = fyv;
}

inline void fluxXYBody(device const float4* xL, device const float4* xR,
                       device const float4* fX, device float4* Fx,
                       device float4* sFx, constant Params& P, int base,
                       int i, int j) {
    const int id = base + j * P.tx + i;
    const int idp = id + 1;
    const float GL = fX[id].w, GR = fX[idp].z;
    float ss = 0.0f;
    const float4 F = hllcFluxXG(toPrimG(xR[id], GL), 1.0f + 1.0f / GL,
                                toPrimG(xL[idp], GR), 1.0f + 1.0f / GR,
                                ss);
    Fx[id] = F;
    sFx[id] = float4(F.x * (F.x > 0.0f ? fX[id].y : fX[idp].x), ss,
                     ss * (ss > 0.0f ? GL : GR), 0.0f);
}

inline void fluxYYBody(device const float4* yB, device const float4* yT,
                       device const float4* fY, device float4* Fy,
                       device float4* sFy, constant Params& P, int base,
                       int i, int j) {
    const int id = base + j * P.tx + i;
    const int idp = id + P.tx;
    const float GB = fY[id].w, GT = fY[idp].z;
    const float4 L = toPrimG(yT[id], GB);
    const float4 R = toPrimG(yB[idp], GT);
    float ss = 0.0f;
    const float4 F = hllcFluxXG(float4(L.x, L.z, L.y, L.w),
                                1.0f + 1.0f / GB,
                                float4(R.x, R.z, R.y, R.w),
                                1.0f + 1.0f / GT, ss);
    Fy[id] = float4(F.x, F.z, F.y, F.w);
    sFy[id] = float4(F.x * (F.x > 0.0f ? fY[id].y : fY[idp].x), ss,
                     ss * (ss > 0.0f ? GB : GT), 0.0f);
}

inline void updateYBody(device float4* q, device float2* sc,
                        device const float4* Fx, device const float4* Fy,
                        device const float4* sFx, device const float4* sFy,
                        constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    const int iw = id - 1, js = id - P.tx;
    const float lx = P.dt / P.dx, ly = P.dt / P.dy;
    const float Gmin = min(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    const float Gmax = max(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));

    q[id] = q[id] + lx * (Fx[iw] - Fx[id]) + ly * (Fy[js] - Fy[id]);
    float2 sv = sc[id];
    sv.x += lx * (sFx[iw].x - sFx[id].x) + ly * (sFy[js].x - sFy[id].x);
    sv.y = clamp(sv.y -
                     lx * (sFx[id].z - sFx[iw].z -
                           sv.y * (sFx[id].y - sFx[iw].y)) -
                     ly * (sFy[id].z - sFy[js].z -
                           sv.y * (sFy[id].y - sFy[js].y)),
                 Gmin, Gmax);
    sc[id] = sv;
}

inline void waveYBody(device const float4* q, device const float2* sc,
                      device atomic_uint* smax, constant Params& P,
                      int base, int i, int j) {
    const int id = base + j * P.tx + i;
    const float4 w = toPrimG(q[id], sc[id].y);
    const float c = sqrt((1.0f + 1.0f / sc[id].y) * w.w / w.x);

    const float sx = simd_max(fabs(w.y) + c);
    const float sy = simd_max(fabs(w.z) + c);
    const float rmn = simd_min(w.x);
    if (simd_is_first()) {
        atomic_fetch_max_explicit(&smax[0], as_type<uint>(sx),
                                  memory_order_relaxed);
        atomic_fetch_max_explicit(&smax[1], as_type<uint>(sy),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&smax[2], as_type<uint>(rmn),
                                  memory_order_relaxed);
    }
}

// ---- plain species kernels -------------------------------------------------

kernel void predictor_y(device const float4* q [[buffer(0)]],
                        device const float2* sc [[buffer(1)]],
                        device float4* xL [[buffer(2)]],
                        device float4* xR [[buffer(3)]],
                        device float4* yB [[buffer(4)]],
                        device float4* yT [[buffer(5)]],
                        device float4* fX [[buffer(6)]],
                        device float4* fY [[buffer(7)]],
                        constant Params& P [[buffer(8)]],
                        uint2 g [[thread_position_in_grid]])
{
    predictorYBody(q, sc, xL, xR, yB, yT, fX, fY, P, 0, int(g.x) + 1,
                   int(g.y) + 1);
}

kernel void flux_x_y(device const float4* xL [[buffer(0)]],
                     device const float4* xR [[buffer(1)]],
                     device const float4* fX [[buffer(2)]],
                     device float4* Fx [[buffer(3)]],
                     device float4* sFx [[buffer(4)]],
                     constant Params& P [[buffer(5)]],
                     uint2 g [[thread_position_in_grid]])
{
    fluxXYBody(xL, xR, fX, Fx, sFx, P, 0, int(g.x) + NG - 1,
               int(g.y) + NG);
}

kernel void flux_y_y(device const float4* yB [[buffer(0)]],
                     device const float4* yT [[buffer(1)]],
                     device const float4* fY [[buffer(2)]],
                     device float4* Fy [[buffer(3)]],
                     device float4* sFy [[buffer(4)]],
                     constant Params& P [[buffer(5)]],
                     uint2 g [[thread_position_in_grid]])
{
    fluxYYBody(yB, yT, fY, Fy, sFy, P, 0, int(g.x) + NG,
               int(g.y) + NG - 1);
}

kernel void update_y(device float4* q [[buffer(0)]],
                     device float2* sc [[buffer(1)]],
                     device const float4* Fx [[buffer(2)]],
                     device const float4* Fy [[buffer(3)]],
                     device const float4* sFx [[buffer(4)]],
                     device const float4* sFy [[buffer(5)]],
                     constant Params& P [[buffer(6)]],
                     uint2 g [[thread_position_in_grid]])
{
    updateYBody(q, sc, Fx, Fy, sFx, sFy, P, 0, int(g.x) + NG,
                int(g.y) + NG);
}

kernel void wave_y(device const float4* q [[buffer(0)]],
                   device const float2* sc [[buffer(1)]],
                   device atomic_uint* smax [[buffer(2)]],
                   constant Params& P [[buffer(3)]],
                   uint2 g [[thread_position_in_grid]])
{
    waveYBody(q, sc, smax, P, 0, int(g.x) + NG, int(g.y) + NG);
}

// ---- pool species kernels ---------------------------------------------------

kernel void predictor_y_pool(device const float4* q [[buffer(0)]],
                             device const float2* sc [[buffer(1)]],
                             device float4* xL [[buffer(2)]],
                             device float4* xR [[buffer(3)]],
                             device float4* yB [[buffer(4)]],
                             device float4* yT [[buffer(5)]],
                             device float4* fX [[buffer(6)]],
                             device float4* fY [[buffer(7)]],
                             constant Params& P [[buffer(8)]],
                             device const uint* slots [[buffer(9)]],
                             uint3 g [[thread_position_in_grid]])
{
    predictorYBody(q, sc, xL, xR, yB, yT, fX, fY, P,
                   int(slots[g.z]) * P.stride, int(g.x) + 1, int(g.y) + 1);
}

kernel void flux_x_y_pool(device const float4* xL [[buffer(0)]],
                          device const float4* xR [[buffer(1)]],
                          device const float4* fX [[buffer(2)]],
                          device float4* Fx [[buffer(3)]],
                          device float4* sFx [[buffer(4)]],
                          constant Params& P [[buffer(5)]],
                          device const uint* slots [[buffer(6)]],
                          uint3 g [[thread_position_in_grid]])
{
    fluxXYBody(xL, xR, fX, Fx, sFx, P, int(slots[g.z]) * P.stride,
               int(g.x) + NG - 1, int(g.y) + NG);
}

kernel void flux_y_y_pool(device const float4* yB [[buffer(0)]],
                          device const float4* yT [[buffer(1)]],
                          device const float4* fY [[buffer(2)]],
                          device float4* Fy [[buffer(3)]],
                          device float4* sFy [[buffer(4)]],
                          constant Params& P [[buffer(5)]],
                          device const uint* slots [[buffer(6)]],
                          uint3 g [[thread_position_in_grid]])
{
    fluxYYBody(yB, yT, fY, Fy, sFy, P, int(slots[g.z]) * P.stride,
               int(g.x) + NG, int(g.y) + NG - 1);
}

kernel void update_y_pool(device float4* q [[buffer(0)]],
                          device float2* sc [[buffer(1)]],
                          device const float4* Fx [[buffer(2)]],
                          device const float4* Fy [[buffer(3)]],
                          device const float4* sFx [[buffer(4)]],
                          device const float4* sFy [[buffer(5)]],
                          constant Params& P [[buffer(6)]],
                          device const uint* slots [[buffer(7)]],
                          uint3 g [[thread_position_in_grid]])
{
    updateYBody(q, sc, Fx, Fy, sFx, sFy, P, int(slots[g.z]) * P.stride,
                int(g.x) + NG, int(g.y) + NG);
}

kernel void wave_y_pool(device const float4* q [[buffer(0)]],
                        device const float2* sc [[buffer(1)]],
                        device atomic_uint* smax [[buffer(2)]],
                        constant Params& P [[buffer(3)]],
                        device const uint* slots [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]])
{
    waveYBody(q, sc, smax, P, int(slots[g.z]) * P.stride, int(g.x) + NG,
              int(g.y) + NG);
}

// ---- WENO5 + SSP-RK3 kernels (mirrors solver/Weno2D.hpp) ------------------
//
// WENO5 (Jiang-Shu) reconstruction of the primitive face states fed to
// HLLC (LLF splitting smeared contacts/shear; see the CPU file).
// The flux kernels also accumulate the RK-weighted flux sum (zeroed at
// stage 0) that the CPU refluxing reads; rk_update captures u0 at
// stage 0 and applies q = rka*u0 + rkb*(q + dt L).

inline float weno5rec(float v0, float v1, float v2, float v3, float v4) {
    const float EPS = 1e-6f;
    const float q0 = (2.0f * v0 - 7.0f * v1 + 11.0f * v2) / 6.0f;
    const float q1 = (-v1 + 5.0f * v2 + 2.0f * v3) / 6.0f;
    const float q2 = (2.0f * v2 + 5.0f * v3 - v4) / 6.0f;
    const float c13 = 13.0f / 12.0f;
    const float b0 = c13 * (v0 - 2.0f * v1 + v2) * (v0 - 2.0f * v1 + v2) +
                     0.25f * (v0 - 4.0f * v1 + 3.0f * v2) *
                         (v0 - 4.0f * v1 + 3.0f * v2);
    const float b1 = c13 * (v1 - 2.0f * v2 + v3) * (v1 - 2.0f * v2 + v3) +
                     0.25f * (v1 - v3) * (v1 - v3);
    const float b2 = c13 * (v2 - 2.0f * v3 + v4) * (v2 - 2.0f * v3 + v4) +
                     0.25f * (3.0f * v2 - 4.0f * v3 + v4) *
                         (3.0f * v2 - 4.0f * v3 + v4);
    const float w0 = 0.1f / ((EPS + b0) * (EPS + b0));
    const float w1 = 0.6f / ((EPS + b1) * (EPS + b1));
    const float w2 = 0.3f / ((EPS + b2) * (EPS + b2));
    return (w0 * q0 + w1 * q1 + w2 * q2) / (w0 + w1 + w2);
}

inline void wenoFluxBody(device const float4* q, device float4* F,
                         device float4* FA, constant Params& P, int base,
                         int i, int j, int str, bool xDir) {
    const int id = base + j * P.tx + i;
    float4 w[6];
    for (int k = 0; k < 6; ++k)
        w[k] = toPrim(q[id + (k - 2) * str]);
    float4 L, R;
    for (int m = 0; m < 4; ++m) {
        L[m] = weno5rec(w[0][m], w[1][m], w[2][m], w[3][m], w[4][m]);
        R[m] = weno5rec(w[5][m], w[4][m], w[3][m], w[2][m], w[1][m]);
    }
    L.x = max(L.x, RHO_FLOOR); L.w = max(L.w, P_FLOOR);
    R.x = max(R.x, RHO_FLOOR); R.w = max(R.w, P_FLOOR);
    float4 out = xDir ? hllcFluxX(L, R) : hllcFluxY(L, R);

    // viscous flux: same 2nd-order central stencil as the MUSCL path
    // (Stokes stress + Fourier heat flux), evaluated from this stage's
    // cell values; subtracted from the convective face flux
    if (P.mu > 0.0f) {
        const float c43 = 4.0f / 3.0f, c23 = 2.0f / 3.0f;
        if (xDir) {
            const float4 w00 = toPrim(q[id]);
            const float4 w10 = toPrim(q[id + 1]);
            const float4 w0p = toPrim(q[id + P.tx]);
            const float4 w0m = toPrim(q[id - P.tx]);
            const float4 w1p = toPrim(q[id + P.tx + 1]);
            const float4 w1m = toPrim(q[id - P.tx + 1]);
            const float ux = (w10.y - w00.y) / P.dx;
            const float vx = (w10.z - w00.z) / P.dx;
            const float Tx = (w10.w / w10.x - w00.w / w00.x) / P.dx;
            const float uy =
                ((w0p.y + w1p.y) - (w0m.y + w1m.y)) / (4.0f * P.dy);
            const float vy =
                ((w0p.z + w1p.z) - (w0m.z + w1m.z)) / (4.0f * P.dy);
            const float txx = P.mu * (c43 * ux - c23 * vy);
            const float txy = P.mu * (uy + vx);
            const float ub = 0.5f * (w00.y + w10.y);
            const float vb = 0.5f * (w00.z + w10.z);
            out.y -= txx;
            out.z -= txy;
            out.w -= ub * txx + vb * txy + P.kT * Tx;
        } else {
            const float4 w00 = toPrim(q[id]);
            const float4 w01 = toPrim(q[id + P.tx]);
            const float4 wp0 = toPrim(q[id + 1]);
            const float4 wm0 = toPrim(q[id - 1]);
            const float4 wp1 = toPrim(q[id + P.tx + 1]);
            const float4 wm1 = toPrim(q[id + P.tx - 1]);
            const float uy = (w01.y - w00.y) / P.dy;
            const float vy = (w01.z - w00.z) / P.dy;
            const float Ty = (w01.w / w01.x - w00.w / w00.x) / P.dy;
            const float ux =
                ((wp0.y + wp1.y) - (wm0.y + wm1.y)) / (4.0f * P.dx);
            const float vx =
                ((wp0.z + wp1.z) - (wm0.z + wm1.z)) / (4.0f * P.dx);
            const float txy = P.mu * (uy + vx);
            const float tyy = P.mu * (c43 * vy - c23 * ux);
            const float ub = 0.5f * (w00.y + w01.y);
            const float vb = 0.5f * (w00.z + w01.z);
            out.y -= txy;
            out.z -= tyy;
            out.w -= ub * txy + vb * tyy + P.kT * Ty;
        }
    }
    F[id] = out;
    FA[id] = (P.rks == 0 ? float4(0.0f) : FA[id]) + P.rkw * out;
}

inline void rkUpdateBody(device float4* q, device float4* u0,
                         device const float4* Fx, device const float4* Fy,
                         constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    const float4 qc = q[id];
    if (P.rks == 0) u0[id] = qc;
    const float4 adv = qc +
                       (P.dt / P.dx) * (Fx[id - 1] - Fx[id]) +
                       (P.dt / P.dy) * (Fy[id - P.tx] - Fy[id]);
    q[id] = P.rka * u0[id] + P.rkb * adv;
}

kernel void weno_flux_x(device const float4* q [[buffer(0)]],
                        device float4* Fx [[buffer(1)]],
                        device float4* FxA [[buffer(2)]],
                        constant Params& P [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]])
{
    wenoFluxBody(q, Fx, FxA, P, 0, int(g.x) + NG - 1, int(g.y) + NG, 1,
                 true);
}

kernel void weno_flux_y(device const float4* q [[buffer(0)]],
                        device float4* Fy [[buffer(1)]],
                        device float4* FyA [[buffer(2)]],
                        constant Params& P [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]])
{
    wenoFluxBody(q, Fy, FyA, P, 0, int(g.x) + NG, int(g.y) + NG - 1,
                 P.tx, false);
}

kernel void rk_update(device float4* q [[buffer(0)]],
                      device float4* u0 [[buffer(1)]],
                      device const float4* Fx [[buffer(2)]],
                      device const float4* Fy [[buffer(3)]],
                      constant Params& P [[buffer(4)]],
                      uint2 g [[thread_position_in_grid]])
{
    rkUpdateBody(q, u0, Fx, Fy, P, 0, int(g.x) + NG, int(g.y) + NG);
}

kernel void weno_flux_x_pool(device const float4* q [[buffer(0)]],
                             device float4* Fx [[buffer(1)]],
                             device float4* FxA [[buffer(2)]],
                             constant Params& P [[buffer(3)]],
                             device const uint* slots [[buffer(4)]],
                             uint3 g [[thread_position_in_grid]])
{
    wenoFluxBody(q, Fx, FxA, P, int(slots[g.z]) * P.stride,
                 int(g.x) + NG - 1, int(g.y) + NG, 1, true);
}

kernel void weno_flux_y_pool(device const float4* q [[buffer(0)]],
                             device float4* Fy [[buffer(1)]],
                             device float4* FyA [[buffer(2)]],
                             constant Params& P [[buffer(3)]],
                             device const uint* slots [[buffer(4)]],
                             uint3 g [[thread_position_in_grid]])
{
    wenoFluxBody(q, Fy, FyA, P, int(slots[g.z]) * P.stride,
                 int(g.x) + NG, int(g.y) + NG - 1, P.tx, false);
}

kernel void rk_update_pool(device float4* q [[buffer(0)]],
                           device float4* u0 [[buffer(1)]],
                           device const float4* Fx [[buffer(2)]],
                           device const float4* Fy [[buffer(3)]],
                           constant Params& P [[buffer(4)]],
                           device const uint* slots [[buffer(5)]],
                           uint3 g [[thread_position_in_grid]])
{
    rkUpdateBody(q, u0, Fx, Fy, P, int(slots[g.z]) * P.stride,
                 int(g.x) + NG, int(g.y) + NG);
}

// ---- two-gas WENO5 kernels (mirrors Weno2D.hpp wenoFluxesY) ----------------
//
// Reconstruct (rho,u,v,p) + Y + Gamma at each face, close with per-side
// gamma HLLC; species mass flux = HLLC mass flux upwinded by Y; Gamma
// rides quasi-conservatively on the contact speed. The flux kernel
// outputs the instantaneous conserved flux F and packed scalar flux
// sF = (Fp, Ss, Fg, 0), and accumulates the RK-weighted conserved flux
// (FA) and species mass flux (sFA, in .x) for the CPU refluxing.

inline void wenoFluxYBody(device const float4* q, device const float2* sc,
                          device float4* F, device float4* FA,
                          device float4* sF, device float4* sFA,
                          constant Params& P, int base, int i, int j,
                          int str, bool xDir) {
    const int id = base + j * P.tx + i;
    const float Gmin = min(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    const float Gmax = max(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    float4 w[6];
    float Yc[6], Gc[6];
    for (int k = 0; k < 6; ++k) {
        const int c = id + (k - 2) * str;
        Gc[k] = sc[c].y;
        w[k] = toPrimG(q[c], Gc[k]);
        Yc[k] = clamp(sc[c].x / max(q[c].x, RHO_FLOOR), 0.0f, 1.0f);
    }
    float4 L, R;
    for (int m = 0; m < 4; ++m) {
        L[m] = weno5rec(w[0][m], w[1][m], w[2][m], w[3][m], w[4][m]);
        R[m] = weno5rec(w[5][m], w[4][m], w[3][m], w[2][m], w[1][m]);
    }
    L.x = max(L.x, RHO_FLOOR); L.w = max(L.w, P_FLOOR);
    R.x = max(R.x, RHO_FLOOR); R.w = max(R.w, P_FLOOR);
    const float YL = clamp(weno5rec(Yc[0], Yc[1], Yc[2], Yc[3], Yc[4]),
                           0.0f, 1.0f);
    const float YR = clamp(weno5rec(Yc[5], Yc[4], Yc[3], Yc[2], Yc[1]),
                           0.0f, 1.0f);
    const float GL = clamp(weno5rec(Gc[0], Gc[1], Gc[2], Gc[3], Gc[4]),
                           Gmin, Gmax);
    const float GR = clamp(weno5rec(Gc[5], Gc[4], Gc[3], Gc[2], Gc[1]),
                           Gmin, Gmax);
    float ss = 0.0f;
    float4 f;
    if (xDir) {
        f = hllcFluxXG(L, 1.0f + 1.0f / GL, R, 1.0f + 1.0f / GR, ss);
    } else {
        const float4 Lr = float4(L.x, L.z, L.y, L.w);
        const float4 Rr = float4(R.x, R.z, R.y, R.w);
        const float4 fr =
            hllcFluxXG(Lr, 1.0f + 1.0f / GL, Rr, 1.0f + 1.0f / GR, ss);
        f = float4(fr.x, fr.z, fr.y, fr.w);
    }
    F[id] = f;
    const float Fp = f.x * (f.x > 0.0f ? YL : YR);
    const float Fg = ss * (ss > 0.0f ? GL : GR);
    sF[id] = float4(Fp, ss, Fg, 0.0f);
    FA[id] = (P.rks == 0 ? float4(0.0f) : FA[id]) + P.rkw * f;
    sFA[id] = (P.rks == 0 ? float4(0.0f) : sFA[id]) +
              P.rkw * float4(Fp, 0.0f, 0.0f, 0.0f);
}

inline void rkUpdateYBody(device float4* q, device float2* sc,
                          device float4* u0, device float2* s0,
                          device const float4* Fx, device const float4* Fy,
                          device const float4* sFx,
                          device const float4* sFy, constant Params& P,
                          int base, int i, int j) {
    const int id = base + j * P.tx + i;
    const int iw = id - 1, js = id - P.tx;
    const float lx = P.dt / P.dx, ly = P.dt / P.dy;
    const float Gmin = min(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    const float Gmax = max(1.0f / (P.g1 - 1.0f), 1.0f / (P.g2 - 1.0f));
    if (P.rks == 0) { u0[id] = q[id]; s0[id] = sc[id]; }
    const float4 adv = q[id] + lx * (Fx[iw] - Fx[id]) +
                       ly * (Fy[js] - Fy[id]);
    q[id] = P.rka * u0[id] + P.rkb * adv;
    const float2 sv = sc[id];
    const float phiAdv = sv.x + lx * (sFx[iw].x - sFx[id].x) +
                         ly * (sFy[js].x - sFy[id].x);
    const float GmAdv =
        sv.y - lx * (sFx[id].z - sFx[iw].z -
                     sv.y * (sFx[id].y - sFx[iw].y)) -
        ly * (sFy[id].z - sFy[js].z - sv.y * (sFy[id].y - sFy[js].y));
    sc[id] = float2(P.rka * s0[id].x + P.rkb * phiAdv,
                    clamp(P.rka * s0[id].y + P.rkb * GmAdv, Gmin, Gmax));
}

kernel void weno_flux_x_y(device const float4* q [[buffer(0)]],
                          device const float2* sc [[buffer(1)]],
                          device float4* Fx [[buffer(2)]],
                          device float4* FxA [[buffer(3)]],
                          device float4* sFx [[buffer(4)]],
                          device float4* sFxA [[buffer(5)]],
                          constant Params& P [[buffer(6)]],
                          uint2 g [[thread_position_in_grid]])
{
    wenoFluxYBody(q, sc, Fx, FxA, sFx, sFxA, P, 0, int(g.x) + NG - 1,
                  int(g.y) + NG, 1, true);
}

kernel void weno_flux_y_y(device const float4* q [[buffer(0)]],
                          device const float2* sc [[buffer(1)]],
                          device float4* Fy [[buffer(2)]],
                          device float4* FyA [[buffer(3)]],
                          device float4* sFy [[buffer(4)]],
                          device float4* sFyA [[buffer(5)]],
                          constant Params& P [[buffer(6)]],
                          uint2 g [[thread_position_in_grid]])
{
    wenoFluxYBody(q, sc, Fy, FyA, sFy, sFyA, P, 0, int(g.x) + NG,
                  int(g.y) + NG - 1, P.tx, false);
}

kernel void rk_update_y(device float4* q [[buffer(0)]],
                        device float2* sc [[buffer(1)]],
                        device float4* u0 [[buffer(2)]],
                        device float2* s0 [[buffer(3)]],
                        device const float4* Fx [[buffer(4)]],
                        device const float4* Fy [[buffer(5)]],
                        device const float4* sFx [[buffer(6)]],
                        device const float4* sFy [[buffer(7)]],
                        constant Params& P [[buffer(8)]],
                        uint2 g [[thread_position_in_grid]])
{
    rkUpdateYBody(q, sc, u0, s0, Fx, Fy, sFx, sFy, P, 0, int(g.x) + NG,
                  int(g.y) + NG);
}

kernel void weno_flux_x_y_pool(device const float4* q [[buffer(0)]],
                               device const float2* sc [[buffer(1)]],
                               device float4* Fx [[buffer(2)]],
                               device float4* FxA [[buffer(3)]],
                               device float4* sFx [[buffer(4)]],
                               device float4* sFxA [[buffer(5)]],
                               constant Params& P [[buffer(6)]],
                               device const uint* slots [[buffer(7)]],
                               uint3 g [[thread_position_in_grid]])
{
    wenoFluxYBody(q, sc, Fx, FxA, sFx, sFxA, P,
                  int(slots[g.z]) * P.stride, int(g.x) + NG - 1,
                  int(g.y) + NG, 1, true);
}

kernel void weno_flux_y_y_pool(device const float4* q [[buffer(0)]],
                               device const float2* sc [[buffer(1)]],
                               device float4* Fy [[buffer(2)]],
                               device float4* FyA [[buffer(3)]],
                               device float4* sFy [[buffer(4)]],
                               device float4* sFyA [[buffer(5)]],
                               constant Params& P [[buffer(6)]],
                               device const uint* slots [[buffer(7)]],
                               uint3 g [[thread_position_in_grid]])
{
    wenoFluxYBody(q, sc, Fy, FyA, sFy, sFyA, P,
                  int(slots[g.z]) * P.stride, int(g.x) + NG,
                  int(g.y) + NG - 1, P.tx, false);
}

kernel void rk_update_y_pool(device float4* q [[buffer(0)]],
                             device float2* sc [[buffer(1)]],
                             device float4* u0 [[buffer(2)]],
                             device float2* s0 [[buffer(3)]],
                             device const float4* Fx [[buffer(4)]],
                             device const float4* Fy [[buffer(5)]],
                             device const float4* sFx [[buffer(6)]],
                             device const float4* sFy [[buffer(7)]],
                             constant Params& P [[buffer(8)]],
                             device const uint* slots [[buffer(9)]],
                             uint3 g [[thread_position_in_grid]])
{
    rkUpdateYBody(q, sc, u0, s0, Fx, Fy, sFx, sFy, P,
                  int(slots[g.z]) * P.stride, int(g.x) + NG,
                  int(g.y) + NG);
}

// ---- reaction source (v1.5, mirrors physics/Reaction.hpp react()) ---------
// Per-cell constant-volume Strang half-step: progress lambda rides on
// sc.x = rho*lambda; internal energy is slaved to progress (e = e0 +
// q*(lambda-lambda0)), so we integrate lambda alone with an adaptive
// sub-cycled RK4 and set the energy algebraically. g1 == g2 (single gas).

inline float reactRate(float l, float e0, float lam0,
                       constant Params& P) {
    if (l >= 1.0f) return 0.0f;
    const float T = (P.g1 - 1.0f) * (e0 + P.rq * (l - lam0));
    if (T <= P.rTign) return 0.0f;
    return P.rA * (1.0f - l) * exp(-P.rEa / max(T, 1e-30f));
}

inline void reactBody(device float4* q, device float2* sc,
                      constant Params& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    float4 c = q[id];
    const float rho = max(c.x, RHO_FLOOR);
    const float ke = 0.5f * (c.y * c.y + c.z * c.z) / rho;
    const float e0 = (c.w - ke) / rho;
    const float lam0 = clamp(sc[id].x / rho, 0.0f, 1.0f);
    float lam = lam0;
    float t = 0.0f;
    int guard = 0;
    while (t < P.dt && ++guard < 100000) {
        const float rate = reactRate(lam, e0, lam0, P);
        float h = P.dt - t;
        if (rate > 0.0f) h = min(h, 0.05f / rate);
        const float k1 = reactRate(lam, e0, lam0, P);
        const float k2 = reactRate(lam + 0.5f * h * k1, e0, lam0, P);
        const float k3 = reactRate(lam + 0.5f * h * k2, e0, lam0, P);
        const float k4 = reactRate(lam + h * k3, e0, lam0, P);
        lam = clamp(lam + h / 6.0f * (k1 + 2 * k2 + 2 * k3 + k4), 0.0f,
                    1.0f);
        t += h;
    }
    c.w = rho * (e0 + P.rq * (lam - lam0)) + ke;
    q[id] = c;
    sc[id].x = rho * lam;
}

kernel void react(device float4* q [[buffer(0)]],
                  device float2* sc [[buffer(1)]],
                  constant Params& P [[buffer(2)]],
                  uint2 g [[thread_position_in_grid]])
{
    reactBody(q, sc, P, 0, int(g.x) + NG, int(g.y) + NG);
}

kernel void react_pool(device float4* q [[buffer(0)]],
                       device float2* sc [[buffer(1)]],
                       constant Params& P [[buffer(2)]],
                       device const uint* slots [[buffer(3)]],
                       uint3 g [[thread_position_in_grid]])
{
    reactBody(q, sc, P, int(slots[g.z]) * P.stride, int(g.x) + NG,
              int(g.y) + NG);
}

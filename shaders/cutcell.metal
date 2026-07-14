// GPU port of the 1st-order cut-cell (embedded-boundary) step: aperture-
// weighted HLLC face fluxes + slip-wall EB flux -> conservative divergence D^c,
// hybrid divergence + flux redistribution (FRD), positivity-floored update.
//
// The FRD is written in GATHER form: instead of a cut cell scattering its
// excess into its 3x3 neighbours (a write-conflict on the GPU), each cell
// gathers what its neighbours shed to it. Mathematically identical to the CPU
// scatter (cutCellHybridD), no atomics. Physics is copied verbatim from
// euler2d.metal so the GPU matches the CPU cut-cell oracle bit-for-bit.

#include <metal_stdlib>
using namespace metal;

constant float GAMMA = 1.4f;
constant float RHO_FLOOR = 1e-10f;
constant float P_FLOOR = 1e-10f;
constant float TINY = 1e-9f;
constant float SPEED_CAP = 50.0f;
constant int NG = 3;

struct CCP {
    int tx, ty, nx, ny;
    float dx, dy, dt;
    int stride; // cells per pool slot (0 for plain kernels)
};

// Per-cell embedded-boundary moments (matches C++ mm::gpu CCMom).
struct CCMom {
    float vol;                 // fluid volume fraction kappa
    float apXhi, apYhi;        // hi-face apertures (lo = neighbour's hi)
    float ebArea, ebnx, ebny;  // EB face length + normal (solid -> fluid)
    float pad0, pad1;
};

// ---- physics (verbatim from euler2d.metal) ------------------------------
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
inline float wallPressure(float4 W, float un) {
    const float c = soundSpeed(W);
    const float A = 2.0f / ((GAMMA + 1.0f) * W.x);
    const float B = (GAMMA - 1.0f) / (GAMMA + 1.0f) * W.w;
    const float m = (GAMMA - 1.0f) / (2.0f * GAMMA);
    float p = max(P_FLOOR, W.w + un * W.x * c);
    for (int it = 0; it < 30; ++it) {
        float f, df;
        if (p > W.w) {
            const float s = sqrt(A / (p + B));
            f = (p - W.w) * s;
            df = s * (1.0f - 0.5f * (p - W.w) / (p + B));
        } else {
            f = 2.0f * c / (GAMMA - 1.0f) * (pow(p / W.w, m) - 1.0f);
            df = 1.0f / (W.x * c) *
                 pow(p / W.w, -(GAMMA + 1.0f) / (2.0f * GAMMA));
        }
        const float step = (f - un) / df;
        p = max(P_FLOOR, p - step);
        if (fabs(step) < 1e-6f * (p + P_FLOOR)) break;
    }
    return p;
}
// slip-wall EB flux; (nox,noy) is the OUTWARD-from-fluid normal = -n_EB.
inline float4 ebWallFlux(float4 w, float nox, float noy) {
    const float un = w.y * nox + w.z * noy;
    const float ps = wallPressure(w, un);
    return float4(0.0f, ps * nox, ps * noy, 0.0f);
}
// positivity-preserving state (mirrors CPU floorState).
inline float4 floorState(float4 q) {
    float4 w = toPrim(q);
    const bool bad = !(w.x >= RHO_FLOOR) || !isfinite(w.w) ||
                     !isfinite(w.y) || !isfinite(w.z) ||
                     sqrt(w.y * w.y + w.z * w.z) > SPEED_CAP;
    if (bad) return toCons(float4(RHO_FLOOR, 0.0f, 0.0f, P_FLOOR));
    if (w.w < P_FLOOR) { w.w = P_FLOOR; return toCons(w); }
    return q;
}

// ---- kernel bodies (base = slot offset; 0 for a single grid) -------------

// (S, D^nc) for a cell from its 3x3 fluid neighbourhood.
inline void neighAvg(device const CCMom* geo, device const float4* Dc,
                     int id, int tx, thread float& S, thread float4& Dnc) {
    S = 0.0f;
    float4 w = float4(0.0f);
    for (int dj = -1; dj <= 1; ++dj)
        for (int di = -1; di <= 1; ++di) {
            const int nb = id + dj * tx + di;
            const float k = geo[nb].vol;
            if (k > TINY) { S += k; w += k * Dc[nb]; }
        }
    Dnc = w / S;
}

inline void ccFluxXBody(device const float4* q, device const CCMom* geo,
                        device float4* Fx, constant CCP& P, int base,
                        int i, int j) {
    const int id = base + j * P.tx + i;
    const float a = geo[id].apXhi;
    if (a <= TINY) { Fx[id] = float4(0.0f); return; }
    Fx[id] = (a * P.dy) * hllcFluxX(toPrim(q[id]), toPrim(q[id + 1]));
}
inline void ccFluxYBody(device const float4* q, device const CCMom* geo,
                        device float4* Fy, constant CCP& P, int base,
                        int i, int j) {
    const int id = base + j * P.tx + i;
    const float a = geo[id].apYhi;
    if (a <= TINY) { Fy[id] = float4(0.0f); return; }
    Fy[id] = (a * P.dx) * hllcFluxY(toPrim(q[id]), toPrim(q[id + P.tx]));
}
inline void ccDcBody(device const float4* q, device const CCMom* geo,
                     device const float4* Fx, device const float4* Fy,
                     device float4* Dc, constant CCP& P, int base,
                     int i, int j) {
    const int id = base + j * P.tx + i;
    const float kc = geo[id].vol;
    if (kc <= TINY) { Dc[id] = float4(0.0f); return; }
    float4 dU = (Fx[id - 1] - Fx[id]) + (Fy[id - P.tx] - Fy[id]);
    const CCMom m = geo[id];
    if (kc < 1.0f - TINY && m.ebArea > TINY) {
        const float4 w = toPrim(q[id]);
        dU -= m.ebArea * ebWallFlux(w, -m.ebnx, -m.ebny);
    }
    Dc[id] = (1.0f / (kc * P.dx * P.dy)) * dU;
}
inline void ccHybridBody(device const CCMom* geo, device const float4* Dc,
                         device float4* D, constant CCP& P, int base,
                         int i, int j) {
    const int id = base + j * P.tx + i;
    const float kc = geo[id].vol;
    if (kc <= TINY) { D[id] = float4(0.0f); return; }
    float Sc; float4 Dncc;
    neighAvg(geo, Dc, id, P.tx, Sc, Dncc);
    float4 Dv = kc * Dc[id] + (1.0f - kc) * Dncc;
    for (int dj = -1; dj <= 1; ++dj)
        for (int di = -1; di <= 1; ++di) {
            const int nb = id + dj * P.tx + di;
            const float kn = geo[nb].vol;
            if (kn <= TINY || kn >= 1.0f - TINY) continue;
            float Sn; float4 Dncn;
            neighAvg(geo, Dc, nb, P.tx, Sn, Dncn);
            Dv += (kn * (1.0f - kn) / Sn) * (Dc[nb] - Dncn);
        }
    D[id] = Dv;
}
inline void ccUpdateBody(device float4* q, device const CCMom* geo,
                         device const float4* D, constant CCP& P, int base,
                         int i, int j) {
    const int id = base + j * P.tx + i;
    if (geo[id].vol <= TINY) return; // covered: frozen
    q[id] = floorState(q[id] + P.dt * D[id]);
}

// ---- 2nd-order bodies (least-squares gradients + reconstruction) ---------
// Linear reconstruction of a primitive state at offset (ox,oy) from the cell.
inline float4 recon(float4 w, float4 gdx, float4 gdy, float ox, float oy) {
    return w + gdx * ox + gdy * oy;
}

// Least-squares primitive gradients over the fluid 3x3 neighbourhood, Barth-
// Jespersen limited (matches the CPU lsqGradients). Uniform grid, so the
// neighbour offset is exactly (di*dx, dj*dy).
inline void ccGradBody(device const float4* q, device const CCMom* geo,
                       device float4* Gdx, device float4* Gdy, constant CCP& P,
                       int base, int i, int j) {
    const int id = base + j * P.tx + i;
    if (geo[id].vol <= TINY) { Gdx[id] = float4(0.0f); Gdy[id] = float4(0.0f);
        return; }
    const float4 wc = toPrim(q[id]);
    float Sxx = 0.0f, Sxy = 0.0f, Syy = 0.0f;
    float4 bx = float4(0.0f), by = float4(0.0f);
    float4 wlo = wc, whi = wc;
    for (int dj = -1; dj <= 1; ++dj)
        for (int di = -1; di <= 1; ++di) {
            if (di == 0 && dj == 0) continue;
            const int nb = id + dj * P.tx + di;
            if (geo[nb].vol <= TINY) continue;
            const float dxk = float(di) * P.dx, dyk = float(dj) * P.dy;
            const float w = 1.0f / (dxk * dxk + dyk * dyk);
            Sxx += w * dxk * dxk; Sxy += w * dxk * dyk; Syy += w * dyk * dyk;
            const float4 wk = toPrim(q[nb]);
            const float4 dW = wk - wc;
            bx += w * dxk * dW; by += w * dyk * dW;
            wlo = min(wlo, wk); whi = max(whi, wk);
        }
    const float det = Sxx * Syy - Sxy * Sxy;
    float4 gdx = float4(0.0f), gdy = float4(0.0f);
    if (det > 1e-30f) {
        for (int c = 0; c < 4; ++c) {
            const float gx = (Syy * bx[c] - Sxy * by[c]) / det;
            const float gy = (Sxx * by[c] - Sxy * bx[c]) / det;
            float phi = 1.0f;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int nb = id + dj * P.tx + di;
                    if (geo[nb].vol <= TINY) continue;
                    const float d = gx * (float(di) * P.dx) +
                                    gy * (float(dj) * P.dy);
                    float lim = 1.0f;
                    if (d > 1e-30f) lim = min(1.0f, (whi[c] - wc[c]) / d);
                    else if (d < -1e-30f) lim = min(1.0f, (wlo[c] - wc[c]) / d);
                    phi = min(phi, lim);
                }
            gdx[c] = phi * gx; gdy[c] = phi * gy;
        }
    }
    Gdx[id] = gdx; Gdy[id] = gdy;
}

// 2nd-order aperture-weighted x/y face fluxes: reconstruct L/R to the face
// centre. Boundary faces use constant reconstruction (ox=0), matching the CPU
// (else an interior gradient against a zero-gradient ghost leaks at the wall).
inline void ccFluxXo2Body(device const float4* q, device const CCMom* geo,
                          device const float4* Gdx, device const float4* Gdy,
                          device float4* Fx, constant CCP& P, int base,
                          int i, int j) {
    const int id = base + j * P.tx + i;
    const float a = geo[id].apXhi;
    if (a <= TINY) { Fx[id] = float4(0.0f); return; }
    const bool bnd = (i < NG) || (i + 1 >= NG + P.nx);
    const float ox = bnd ? 0.0f : 0.5f * P.dx;
    const float4 wL = recon(toPrim(q[id]), Gdx[id], Gdy[id], ox, 0.0f);
    const float4 wR = recon(toPrim(q[id + 1]), Gdx[id + 1], Gdy[id + 1], -ox,
                            0.0f);
    Fx[id] = (a * P.dy) * hllcFluxX(wL, wR);
}
inline void ccFluxYo2Body(device const float4* q, device const CCMom* geo,
                          device const float4* Gdx, device const float4* Gdy,
                          device float4* Fy, constant CCP& P, int base,
                          int i, int j) {
    const int id = base + j * P.tx + i;
    const float a = geo[id].apYhi;
    if (a <= TINY) { Fy[id] = float4(0.0f); return; }
    const bool bnd = (j < NG) || (j + 1 >= NG + P.ny);
    const float oy = bnd ? 0.0f : 0.5f * P.dy;
    const float4 wB = recon(toPrim(q[id]), Gdx[id], Gdy[id], 0.0f, oy);
    const float4 wT = recon(toPrim(q[id + P.tx]), Gdx[id + P.tx],
                            Gdy[id + P.tx], 0.0f, -oy);
    Fy[id] = (a * P.dx) * hllcFluxY(wB, wT);
}

// 2nd-order conservative divergence: same face-flux balance as ccDcBody, but
// the EB wall flux is reconstructed to the boundary centroid (offset pad0,pad1
// = centroid - cell centre, filled by setGeometry).
inline void ccDcO2Body(device const float4* q, device const CCMom* geo,
                       device const float4* Gdx, device const float4* Gdy,
                       device const float4* Fx, device const float4* Fy,
                       device float4* Dc, constant CCP& P, int base,
                       int i, int j) {
    const int id = base + j * P.tx + i;
    const float kc = geo[id].vol;
    if (kc <= TINY) { Dc[id] = float4(0.0f); return; }
    float4 dU = (Fx[id - 1] - Fx[id]) + (Fy[id - P.tx] - Fy[id]);
    const CCMom m = geo[id];
    if (kc < 1.0f - TINY && m.ebArea > TINY) {
        const float4 w = recon(toPrim(q[id]), Gdx[id], Gdy[id], m.pad0, m.pad1);
        dU -= m.ebArea * ebWallFlux(w, -m.ebnx, -m.ebny);
    }
    Dc[id] = (1.0f / (kc * P.dx * P.dy)) * dU;
}

// SSP-RK2 combine: q^{n+1} = 0.5 q^n + 0.5 (q1 + dt D(q1)); q holds q1.
inline void ccRk2Body(device float4* q, device const float4* qn,
                      device const CCMom* geo, device const float4* D,
                      constant CCP& P, int base, int i, int j) {
    const int id = base + j * P.tx + i;
    if (geo[id].vol <= TINY) return;
    q[id] = floorState(0.5f * qn[id] + 0.5f * (q[id] + P.dt * D[id]));
}

// ---- plain kernels (one grid) -------------------------------------------
kernel void cc_flux_x(device const float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device float4* Fx [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    ccFluxXBody(q, geo, Fx, P, 0, int(g.x) + NG - 1, int(g.y) + NG);
}
kernel void cc_flux_y(device const float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device float4* Fy [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    ccFluxYBody(q, geo, Fy, P, 0, int(g.x) + NG, int(g.y) + NG - 1);
}
kernel void cc_dc(device const float4* q [[buffer(0)]],
                  device const CCMom* geo [[buffer(1)]],
                  device const float4* Fx [[buffer(2)]],
                  device const float4* Fy [[buffer(3)]],
                  device float4* Dc [[buffer(4)]],
                  constant CCP& P [[buffer(5)]],
                  uint2 g [[thread_position_in_grid]]) {
    ccDcBody(q, geo, Fx, Fy, Dc, P, 0, int(g.x) + NG, int(g.y) + NG);
}
kernel void cc_hybrid(device const CCMom* geo [[buffer(0)]],
                      device const float4* Dc [[buffer(1)]],
                      device float4* D [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    ccHybridBody(geo, Dc, D, P, 0, int(g.x) + NG, int(g.y) + NG);
}
kernel void cc_update(device float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device const float4* D [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    ccUpdateBody(q, geo, D, P, 0, int(g.x) + NG, int(g.y) + NG);
}

// ---- pool kernels (all AMR patches in one dispatch; gid.z = slot) -------
kernel void cc_flux_x_pool(device const float4* q [[buffer(0)]],
                           device const CCMom* geo [[buffer(1)]],
                           device float4* Fx [[buffer(2)]],
                           constant CCP& P [[buffer(3)]],
                           device const uint* slots [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]]) {
    ccFluxXBody(q, geo, Fx, P, int(slots[g.z]) * P.stride,
                int(g.x) + NG - 1, int(g.y) + NG);
}
kernel void cc_flux_y_pool(device const float4* q [[buffer(0)]],
                           device const CCMom* geo [[buffer(1)]],
                           device float4* Fy [[buffer(2)]],
                           constant CCP& P [[buffer(3)]],
                           device const uint* slots [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]]) {
    ccFluxYBody(q, geo, Fy, P, int(slots[g.z]) * P.stride,
                int(g.x) + NG, int(g.y) + NG - 1);
}
kernel void cc_dc_pool(device const float4* q [[buffer(0)]],
                       device const CCMom* geo [[buffer(1)]],
                       device const float4* Fx [[buffer(2)]],
                       device const float4* Fy [[buffer(3)]],
                       device float4* Dc [[buffer(4)]],
                       constant CCP& P [[buffer(5)]],
                       device const uint* slots [[buffer(6)]],
                       uint3 g [[thread_position_in_grid]]) {
    ccDcBody(q, geo, Fx, Fy, Dc, P, int(slots[g.z]) * P.stride,
             int(g.x) + NG, int(g.y) + NG);
}
kernel void cc_hybrid_pool(device const CCMom* geo [[buffer(0)]],
                           device const float4* Dc [[buffer(1)]],
                           device float4* D [[buffer(2)]],
                           constant CCP& P [[buffer(3)]],
                           device const uint* slots [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]]) {
    ccHybridBody(geo, Dc, D, P, int(slots[g.z]) * P.stride,
                 int(g.x) + NG, int(g.y) + NG);
}
kernel void cc_update_pool(device float4* q [[buffer(0)]],
                           device const CCMom* geo [[buffer(1)]],
                           device const float4* D [[buffer(2)]],
                           constant CCP& P [[buffer(3)]],
                           device const uint* slots [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]]) {
    ccUpdateBody(q, geo, D, P, int(slots[g.z]) * P.stride,
                 int(g.x) + NG, int(g.y) + NG);
}

// ---- 2nd-order plain kernels (one grid) ---------------------------------
kernel void cc_grad(device const float4* q [[buffer(0)]],
                    device const CCMom* geo [[buffer(1)]],
                    device float4* Gdx [[buffer(2)]],
                    device float4* Gdy [[buffer(3)]],
                    constant CCP& P [[buffer(4)]],
                    uint2 g [[thread_position_in_grid]]) {
    ccGradBody(q, geo, Gdx, Gdy, P, 0, int(g.x) + NG, int(g.y) + NG);
}
kernel void cc_flux_x_o2(device const float4* q [[buffer(0)]],
                         device const CCMom* geo [[buffer(1)]],
                         device const float4* Gdx [[buffer(2)]],
                         device const float4* Gdy [[buffer(3)]],
                         device float4* Fx [[buffer(4)]],
                         constant CCP& P [[buffer(5)]],
                         uint2 g [[thread_position_in_grid]]) {
    ccFluxXo2Body(q, geo, Gdx, Gdy, Fx, P, 0, int(g.x) + NG - 1,
                  int(g.y) + NG);
}
kernel void cc_flux_y_o2(device const float4* q [[buffer(0)]],
                         device const CCMom* geo [[buffer(1)]],
                         device const float4* Gdx [[buffer(2)]],
                         device const float4* Gdy [[buffer(3)]],
                         device float4* Fy [[buffer(4)]],
                         constant CCP& P [[buffer(5)]],
                         uint2 g [[thread_position_in_grid]]) {
    ccFluxYo2Body(q, geo, Gdx, Gdy, Fy, P, 0, int(g.x) + NG,
                  int(g.y) + NG - 1);
}
kernel void cc_dc_o2(device const float4* q [[buffer(0)]],
                     device const CCMom* geo [[buffer(1)]],
                     device const float4* Gdx [[buffer(2)]],
                     device const float4* Gdy [[buffer(3)]],
                     device const float4* Fx [[buffer(4)]],
                     device const float4* Fy [[buffer(5)]],
                     device float4* Dc [[buffer(6)]],
                     constant CCP& P [[buffer(7)]],
                     uint2 g [[thread_position_in_grid]]) {
    ccDcO2Body(q, geo, Gdx, Gdy, Fx, Fy, Dc, P, 0, int(g.x) + NG,
               int(g.y) + NG);
}
kernel void cc_rk2(device float4* q [[buffer(0)]],
                   device const float4* qn [[buffer(1)]],
                   device const CCMom* geo [[buffer(2)]],
                   device const float4* D [[buffer(3)]],
                   constant CCP& P [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]]) {
    ccRk2Body(q, qn, geo, D, P, 0, int(g.x) + NG, int(g.y) + NG);
}

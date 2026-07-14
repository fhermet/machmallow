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

// ---- kernels ------------------------------------------------------------

// aperture-weighted extensive x-face flux at cell (i,j)'s hi (right) face.
kernel void cc_flux_x(device const float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device float4* Fx [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    const int i = int(g.x) + NG - 1, j = int(g.y) + NG;
    const int id = j * P.tx + i;
    const float a = geo[id].apXhi;
    if (a <= TINY) { Fx[id] = float4(0.0f); return; }
    Fx[id] = (a * P.dy) * hllcFluxX(toPrim(q[id]), toPrim(q[id + 1]));
}

kernel void cc_flux_y(device const float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device float4* Fy [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    const int i = int(g.x) + NG, j = int(g.y) + NG - 1;
    const int id = j * P.tx + i;
    const float a = geo[id].apYhi;
    if (a <= TINY) { Fy[id] = float4(0.0f); return; }
    Fy[id] = (a * P.dx) * hllcFluxY(toPrim(q[id]), toPrim(q[id + P.tx]));
}

// conservative divergence D^c = dU/(kappa V), dU = sum of aperture-weighted
// face fluxes + EB wall flux.
kernel void cc_dc(device const float4* q [[buffer(0)]],
                  device const CCMom* geo [[buffer(1)]],
                  device const float4* Fx [[buffer(2)]],
                  device const float4* Fy [[buffer(3)]],
                  device float4* Dc [[buffer(4)]],
                  constant CCP& P [[buffer(5)]],
                  uint2 g [[thread_position_in_grid]]) {
    const int i = int(g.x) + NG, j = int(g.y) + NG;
    const int id = j * P.tx + i;
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

// hybrid divergence D = kappa D^c + (1-kappa) D^nc + gathered redistribution.
kernel void cc_hybrid(device const CCMom* geo [[buffer(0)]],
                      device const float4* Dc [[buffer(1)]],
                      device float4* D [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    const int i = int(g.x) + NG, j = int(g.y) + NG;
    const int id = j * P.tx + i;
    const float kc = geo[id].vol;
    if (kc <= TINY) { D[id] = float4(0.0f); return; }
    float Sc; float4 Dncc;
    neighAvg(geo, Dc, id, P.tx, Sc, Dncc);
    float4 Dv = kc * Dc[id] + (1.0f - kc) * Dncc;
    // gather: every cut neighbour nb (incl self) sheds dD to this cell.
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

kernel void cc_update(device float4* q [[buffer(0)]],
                      device const CCMom* geo [[buffer(1)]],
                      device const float4* D [[buffer(2)]],
                      constant CCP& P [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
    const int i = int(g.x) + NG, j = int(g.y) + NG;
    const int id = j * P.tx + i;
    if (geo[id].vol <= TINY) return; // covered: frozen
    q[id] = floorState(q[id] + P.dt * D[id]);
}

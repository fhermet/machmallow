#pragma once

namespace mm {

// Solver-wide floating point type. GPU path is float-only (Metal has no
// fp64), so the whole code runs in float32.
using Real = float;

// Conserved variables: rho, rho*u, rho*v, E (total energy per volume).
inline constexpr int NVARS = 4;

// Perfect gas. Per-case gamma can come later; constant for now.
inline constexpr Real GAMMA = Real(1.4);

// Prandtl number (viscous runs); R = 1, so T = p/rho and
// k = mu * gamma / ((gamma - 1) * Pr).
inline constexpr Real PRANDTL = Real(0.72);

// Positivity floors (float32 needs them around strong shocks).
inline constexpr Real RHO_FLOOR = Real(1e-10);
inline constexpr Real P_FLOOR = Real(1e-10);

} // namespace mm

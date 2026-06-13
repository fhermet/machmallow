// v1.5 first step: 0D constant-volume reactor — validates the stiff
// reaction integrator (react()) in isolation, before coupling it to the
// flow. Three exact references:
//   1. isothermal (q = 0): lambda(t) = 1 - (1-lambda0) exp(-K t),
//      K = A exp(-Ea/T0) — checks the integrator accuracy;
//   2. adiabatic equilibrium + energy conservation: the burn runs to
//      lambda = 1 and raises the temperature by exactly (gamma-1) q;
//   3. stiffness: a very large A with a coarse outer dt must stay
//      bounded and still reach equilibrium (sub-cycling works).

#include "core/Types.hpp"
#include "physics/Reaction.hpp"

#include <cmath>
#include <cstdio>

using namespace mm;

namespace {

bool gate1_isothermal() {
    // q = 0 -> T frozen -> constant rate K -> exact exponential.
    const Real T0 = 2.0, lam0 = 0;
    Reaction r{/*A*/ Real(3.0), /*Ea*/ Real(5.0), /*q*/ 0};
    const Real K = r.A * std::exp(-r.Ea / T0);
    Real e = T0 / (GAMMA - 1), lam = lam0;
    const Real tEnd = 2.0;
    react(e, lam, tEnd, r);
    const double exact = 1 - (1 - lam0) * std::exp(-double(K) * tEnd);
    const double err = std::fabs(double(lam) - exact);
    std::printf("gate 1 — isothermal reactor: lambda %.6f vs exact "
                "%.6f, err %.3e (gate 1e-5)\n",
                double(lam), exact, err);
    return err < 1e-5;
}

bool gate2_adiabatic() {
    // Full burn at constant volume: lambda -> 1 and T rises by (g-1)q.
    const Real T0 = 2.0, q = Real(8.0);
    Reaction r{Real(5.0), Real(6.0), q};
    Real e = T0 / (GAMMA - 1), lam = 0;
    const Real e0 = e;
    react(e, lam, Real(20.0), r); // long enough to equilibrate
    const double Tend = double(GAMMA - 1) * double(e);
    const double TexpEq = double(T0) + double(GAMMA - 1) * double(q);
    // energy conservation: e gained = q * lambda
    const double econs =
        std::fabs(double(e) - (double(e0) + double(q) * double(lam)));
    std::printf("gate 2 — adiabatic reactor: lambda %.6f (gate >0.999), "
                "T %.5f vs eq %.5f, energy-balance resid %.3e\n",
                double(lam), Tend, TexpEq, econs);
    return lam > Real(0.999) &&
           std::fabs(Tend - TexpEq) / TexpEq < 1e-4 && econs < 1e-5;
}

bool gate3_stiff() {
    // Very stiff: large A, coarse single outer dt. The sub-cycling must
    // stay bounded and reach equilibrium in one call.
    const Real T0 = 2.0, q = Real(8.0);
    Reaction r{Real(1e4), Real(6.0), q}; // ~2000x faster than gate 2
    Real e = T0 / (GAMMA - 1), lam = 0;
    react(e, lam, Real(1.0), r); // one coarse outer step
    const bool bounded = lam >= 0 && lam <= 1 && std::isfinite(double(e));
    const double TexpEq = double(T0) + double(GAMMA - 1) * double(q);
    const double Tend = double(GAMMA - 1) * double(e);
    std::printf("gate 3 — stiff reactor (A=1e4, dt=1): lambda %.6f, "
                "T %.5f vs eq %.5f, bounded %d\n",
                double(lam), Tend, TexpEq, int(bounded));
    return bounded && lam > Real(0.999) &&
           std::fabs(Tend - TexpEq) / TexpEq < 1e-4;
}

} // namespace

int main() {
    bool ok = true;
    ok = gate1_isothermal() && ok;
    ok = gate2_adiabatic() && ok;
    ok = gate3_stiff() && ok;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

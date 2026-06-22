#pragma once

// Fully declarative case definition: domain, named primitive states,
// geometric regions (with optional moving shock fronts), profile
// modifiers and per-side boundary conditions — everything parsed from
// the INI case file, no per-case C++.
//
//   [domain]   x0/x1/y0/y1
//   [grid]     nx/ny
//   [state.X]  rho/u/v/p
//   [ic]       default = X
//              region.N  = halfplane a b c [speed s] : X
//                        | band x|y lo hi : X
//                        | rect x0 x1 y0 y1 : X
//                        | circle cx cy r : X
//                        | sinex x0 amp lambda : X
//              perturb.N = u|v|rho|p sin periods amp
//                        | u|v|rho|p erf x0 width amp
//   [bc]       x|y = periodic
//              left|right|bottom|top =
//                  transmissive | reflective | noslip | analytic |
//                  inflow X   (noslip = viscous wall, needs mu > 0)
//                  [if x|y < val else <spec>]
//
// The key idea: `analytic` ghosts re-evaluate the (time-dependent)
// region stack at the ghost centers, so exact moving-shock BCs (DMR's
// top boundary) come for free from the same description as the IC.

#include "core/Boundary.hpp"
#include "physics/Reaction.hpp"
#include "physics/TwoGas.hpp"
#include "core/Config.hpp"
#include "core/Grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mm {

class CaseDef {
public:
    Real x0 = 0, y0 = 0, lx = 1, ly = 1;
    int nx = 64, ny = 64;
    bool periodicX = false, periodicY = false;
    Real gx = 0, gy = 0; // gravity ([physics] gravity = gx gy)

    static CaseDef parse(const Config& cfg) {
        CaseDef c;
        c.x0 = Real(cfg.getReal("domain.x0", 0));
        c.y0 = Real(cfg.getReal("domain.y0", 0));
        c.lx = Real(cfg.getReal("domain.x1", 1)) - c.x0;
        c.ly = Real(cfg.getReal("domain.y1", 1)) - c.y0;
        if (c.lx <= 0 || c.ly <= 0)
            throw std::runtime_error("[domain] x1/y1 must exceed x0/y0");
        c.nx = cfg.getInt("grid.nx", 64);
        c.ny = cfg.getInt("grid.ny", 64);

        if (cfg.has("physics.gravity")) {
            const auto g = tokens_(cfg.requireString("physics.gravity"));
            if (g.size() != 2)
                throw std::runtime_error(
                    "[physics] gravity = <gx> <gy>");
            c.gx = num_(g[0]);
            c.gy = num_(g[1]);
        }

        // two-gas cases: a [species] section names the two gammas and
        // each state may declare `gas = 2` (default 1)
        if (cfg.has("species.gamma1") || cfg.has("species.gamma2")) {
            c.species_ = true;
            c.gas_.gamma1 = Real(cfg.getReal("species.gamma1", 1.4));
            c.gas_.gamma2 = Real(cfg.getReal("species.gamma2", 1.4));
            if (c.gas_.gamma1 <= 1 || c.gas_.gamma2 <= 1)
                throw std::runtime_error(
                    "[species] gammas must be > 1");
        }

        // reacting flow: a [reaction] section (single-step Arrhenius)
        // makes the progress variable lambda ride on the species scalar
        // (single gas, gamma1 = gamma2); a state with `gas = 2` is the
        // burnt/ignited region (lambda = 1).
        if (cfg.has("reaction.q")) {
            c.species_ = true;
            c.react_ = true;
            const Real g = Real(cfg.getReal("reaction.gamma", 1.4));
            c.gas_ = GasPair{g, g};
            c.reaction_.A = Real(cfg.getReal("reaction.A", 0));
            c.reaction_.Ea = Real(cfg.getReal("reaction.Ea", 0));
            c.reaction_.q = Real(cfg.getReal("reaction.q", 0));
            c.reaction_.Tign = Real(cfg.getReal("reaction.Tign", 0));
            if (g <= 1) throw std::runtime_error(
                "[reaction] gamma must be > 1");
        }

        // named states: plain ones first, then derived (Rankine-Hugoniot
        // post-shock states referencing a plain state)
        std::vector<std::string> names;
        for (const std::string& key : cfg.sectionKeys("state")) {
            const std::string rest = key.substr(6); // after "state."
            const auto dot = rest.find('.');
            if (dot == std::string::npos) continue;
            const std::string name = rest.substr(0, dot);
            if (std::find(names.begin(), names.end(), name) ==
                names.end())
                names.push_back(name);
        }
        for (const std::string& name : names) {
            const int gasIdx =
                int(cfg.getInt("state." + name + ".gas", 1));
            if (gasIdx != 1 && gasIdx != 2)
                throw std::runtime_error("state '" + name +
                                         "': gas must be 1 or 2");
            if (gasIdx == 2 && !c.species_)
                throw std::runtime_error(
                    "state '" + name +
                    "': gas = 2 needs a [species] section");
            if (!cfg.has("state." + name + ".shock"))
                c.states_[name] = {
                    {Real(cfg.getReal("state." + name + ".rho", 1)),
                     Real(cfg.getReal("state." + name + ".u", 0)),
                     Real(cfg.getReal("state." + name + ".v", 0)),
                     Real(cfg.getReal("state." + name + ".p", 1))},
                    0, gasIdx};
        }
        for (const std::string& name : names)
            if (cfg.has("state." + name + ".shock")) {
                NamedState st = c.parseShockState_(
                    cfg.requireString("state." + name + ".shock"));
                if (cfg.has("state." + name + ".gas") &&
                    int(cfg.getInt("state." + name + ".gas", 1)) !=
                        st.gas)
                    throw std::runtime_error(
                        "state '" + name +
                        "': a shock state inherits its upstream gas");
                c.states_[name] = st;
            }
        for (const auto& [name, st] : c.states_)
            if (st.w.rho <= 0 || st.w.p <= 0)
                throw std::runtime_error("state '" + name +
                                         "': rho and p must be > 0");

        {
            const NamedState& d =
                c.lookupState_(cfg.requireString("ic.default"));
            c.def_ = d.w;
            c.defY_ = d.gas == 2 ? Real(1) : Real(0);
        }
        for (const std::string& key : numbered_(cfg, "ic.region."))
            c.regions_.push_back(c.parseRegion_(cfg.requireString(key)));
        for (const std::string& key : numbered_(cfg, "ic.perturb."))
            c.perturbs_.push_back(parsePerturb_(cfg.requireString(key)));

        // Immersed solids: static geometric regions flagged as solid
        // (same shape grammar as IC regions, but no state and no motion).
        // The solver masks these cells off and treats fluid/solid faces as
        // reflective walls (slip). See [solid] in CASE_FORMAT.md.
        for (const std::string& key : numbered_(cfg, "solid.region."))
            c.solids_.push_back(
                c.parseSolidRegion_(cfg.requireString(key)));

        c.periodicX = cfg.getString("bc.x", "") == "periodic";
        c.periodicY = cfg.getString("bc.y", "") == "periodic";
        const auto sideSpec = [&](const char* name, bool periodic) {
            if (periodic) return Side{};
            return c.parseSide_(
                cfg.getString(std::string("bc.") + name, "transmissive"));
        };
        c.sides_[0] = sideSpec("left", c.periodicX);
        c.sides_[1] = sideSpec("right", c.periodicX);
        c.sides_[2] = sideSpec("bottom", c.periodicY);
        c.sides_[3] = sideSpec("top", c.periodicY);
        return c;
    }

    // Primitive state of the case description at (x, y, t).
    Prim prim(Real x, Real y, double t) const {
        Prim w = def_;
        for (const Region& r : regions_)
            if (r.inside(x, y, t)) w = r.st;
        for (const Perturb& pb : perturbs_)
            pb.apply(w, x, y, x0, lx, gy);
        return w;
    }
    Cons state(Real x, Real y, double t) const {
        if (!species_) return toCons(prim(x, y, t));
        return toConsG(prim(x, y, t), gas_.Gamma(massFraction(x, y, t)));
    }

    // Mass fraction of gas 2 at (x, y, t): the region logic of prim()
    // (perturbations do not touch the composition).
    Real massFraction(Real x, Real y, double t) const {
        Real Y = defY_;
        for (const Region& r : regions_)
            if (r.inside(x, y, t)) Y = r.Y;
        return Y;
    }
    bool species() const { return species_; }
    const GasPair& gases() const { return gas_; }
    bool reacts() const { return react_; }
    const Reaction& reaction() const { return reaction_; }

    // Immersed solids: any declared [solid] region present?
    bool hasSolids() const { return !solids_.empty(); }
    // 1 if (x, y) falls inside a solid region, else 0 (static: no t).
    std::uint8_t solidAt(Real x, Real y) const {
        for (const Region& r : solids_)
            if (r.inside(x, y, 0)) return 1;
        return 0;
    }

    template <class G>
    void fillGhosts(G& g, double t) const {
        if (periodicX) fillPeriodicX(g);
        if (periodicY) fillPeriodicY(g);
        fillGhostSides(g, t,
                       (periodicX ? 0u : (SideLeft | SideRight)) |
                           (periodicY ? 0u : (SideBottom | SideTop)));
    }

    template <class G>
    void fillGhostSides(G& g, double t, unsigned mask) const {
        if (mask & SideLeft) side_(g, t, 0);
        if (mask & SideRight) side_(g, t, 1);
        if (mask & SideBottom) side_(g, t, 2);
        if (mask & SideTop) side_(g, t, 3);
    }

    // For the preflight printer: name, primitive state, shock speed
    // (0 unless RH-derived), plus the fastest signal speed of the case.
    struct StateInfo {
        std::string name;
        Prim w;
        Real shockSpeed;
    };
    std::vector<StateInfo> listStates() const {
        std::vector<StateInfo> out;
        for (const auto& [name, st] : states_)
            out.push_back({name, st.w, st.shockSpeed});
        return out;
    }
    Real maxSignalSpeed() const {
        Real s = 0;
        for (const auto& [name, st] : states_)
            s = std::max(s, std::max(std::fabs(st.w.u),
                                     std::fabs(st.w.v)) +
                                soundSpeedG(st.w, gammaOf_(st.gas)));
        return s;
    }

private:
    enum class Bc { Transmissive, Reflective, NoSlip, Analytic, Inflow };
    struct Spec {
        Bc type = Bc::Transmissive;
        Prim inflow{};
        Real inflowG = 1 / (GAMMA - 1); // EOS closure of the inflow gas
    };
    struct Side {
        Spec a;
        bool split = false;
        Real splitAt = 0; // on x for bottom/top, on y for left/right
        Spec b;
    };

    struct Region {
        enum class Shape {
            HalfPlane, BandX, BandY, Rect, Circle, SineX, Triangle
        } shape;
        Real p[6] = {0, 0, 0, 0, 0, 0};
        Real speed = 0, norm = 1; // half-plane front motion
        Prim st{};
        Real Y = 0; // mass fraction of gas 2 (two-gas cases)

        bool inside(Real x, Real y, double t) const {
            switch (shape) {
            case Shape::HalfPlane:
                return p[0] * x + p[1] * y <
                       p[2] + speed * norm * Real(t);
            case Shape::BandX: return x > p[0] && x < p[1];
            case Shape::BandY: return y > p[0] && y < p[1];
            case Shape::Rect:
                return x > p[0] && x < p[1] && y > p[2] && y < p[3];
            case Shape::Triangle: {
                // point-in-triangle (sommets (p0,p1),(p2,p3),(p4,p5)) :
                // les trois produits vectoriels de bord ont le même signe.
                const auto cross = [](Real ax, Real ay, Real bx, Real by,
                                      Real px, Real py) {
                    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
                };
                const Real d1 = cross(p[0], p[1], p[2], p[3], x, y);
                const Real d2 = cross(p[2], p[3], p[4], p[5], x, y);
                const Real d3 = cross(p[4], p[5], p[0], p[1], x, y);
                const bool neg = d1 < 0 || d2 < 0 || d3 < 0;
                const bool pos = d1 > 0 || d2 > 0 || d3 > 0;
                return !(neg && pos);
            }
            case Shape::SineX: // x < x0 + amp*cos(2*pi*y/lambda)
                return x < p[0] +
                               p[1] * std::cos(Real(2 * M_PI) * y / p[2]);
            default: {
                const Real dx = x - p[0], dy = y - p[1];
                return dx * dx + dy * dy < p[2] * p[2];
            }
            }
        }
    };

    struct Perturb {
        Prim::Var var = Prim::Var::Rho;
        // Sin: a=periods, b=amp | Erf: a=x0, b=width, c=amp
        // SinG (sin x gaussian in y): a=periods, b=amp, c=yc, d=sigma
        // Hydro (hydrostatic pressure): a=yref -> p += rho*gy*(y-yref)
        enum class Kind { Sin, Erf, SinG, Hydro } kind;
        Real a = 0, b = 0, c = 0, d = 0;

        void apply(Prim& w, Real x, Real y, Real x0, Real lx,
                   Real gy) const {
            switch (kind) {
            case Kind::Sin:
                w.add(var, b * Real(std::sin(2 * M_PI * double(a) *
                                             double((x - x0) / lx))));
                break;
            case Kind::Erf:
                w.add(var, c * Real(std::erf(double((x - a) / b))));
                break;
            case Kind::SinG: {
                const Real e = (y - c) / d;
                w.add(var,
                      b *
                          Real(std::sin(2 * M_PI * double(a) *
                                        double((x - x0) / lx))) *
                          Real(std::exp(-double(e) * e)));
                break;
            }
            case Kind::Hydro:
                w.p += w.rho * gy * (y - a);
                break;
            }
        }
    };

    // ---- parsing helpers --------------------------------------------------
    static std::vector<std::string> numbered_(const Config& cfg,
                                              const std::string& prefix) {
        std::vector<std::pair<int, std::string>> found;
        // probe indices 1..99 (sparse numbering allowed)
        for (int i = 1; i < 100; ++i) {
            const std::string key = prefix + std::to_string(i);
            if (cfg.has(key)) found.push_back({i, key});
        }
        std::sort(found.begin(), found.end());
        std::vector<std::string> out;
        for (auto& [i, k] : found) out.push_back(k);
        return out;
    }

    static std::vector<std::string> tokens_(const std::string& s) {
        std::istringstream in(s);
        std::vector<std::string> out;
        std::string tok;
        while (in >> tok) out.push_back(tok);
        return out;
    }
    static Real num_(const std::string& s) {
        char* end = nullptr;
        const double x = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0')
            throw std::runtime_error("case file: not a number: " + s);
        return Real(x);
    }

    struct NamedState {
        Prim w{};
        Real shockSpeed = 0; // lab-frame front speed (RH-derived states)
        int gas = 1;         // 1 or 2 (two-gas cases)
    };

    const NamedState& lookupState_(const std::string& name) const {
        const auto it = states_.find(name);
        if (it == states_.end())
            throw std::runtime_error("unknown state '" + name +
                                     "' (add a [state." + name +
                                     "] section)");
        return it->second;
    }

    // "shock = <upstream> mach <Ms> [+x|-x|+y|-y]": Rankine-Hugoniot
    // post-shock state for a shock moving at Mach Ms (relative to the
    // upstream gas) in the given direction. Stores the lab-frame shock
    // speed for `speed auto` fronts.
    NamedState parseShockState_(const std::string& spec) const {
        const auto toks = tokens_(spec);
        if (toks.size() < 3 || toks.size() > 4 || toks[1] != "mach")
            throw std::runtime_error(
                "shock state: expected '<state> mach <Ms> [+x|-x|+y|-y]'"
                ": " +
                spec);
        const NamedState& up = lookupState_(toks[0]);
        const Real Ms = num_(toks[2]);
        if (Ms <= 1)
            throw std::runtime_error("shock state: Ms must be > 1");
        const std::string dir =
            toks.size() == 4 ? toks[3] : std::string("+x");
        const Real sgn = (dir == "-x" || dir == "-y") ? Real(-1) : Real(1);
        const bool alongX = dir == "+x" || dir == "-x";
        if (!alongX && dir != "+y" && dir != "-y")
            throw std::runtime_error("shock state: bad direction " + dir);

        const Real ga = gammaOf_(up.gas);
        const Real c1 = soundSpeedG(up.w, ga);
        const Real M2 = Ms * Ms;
        NamedState s;
        s.w = up.w;
        s.gas = up.gas; // the shock runs in the upstream gas
        s.w.rho = up.w.rho * (ga + 1) * M2 / ((ga - 1) * M2 + 2);
        s.w.p = up.w.p * (1 + 2 * ga / (ga + 1) * (M2 - 1));
        const Real du = sgn * 2 * c1 / (ga + 1) * (Ms - 1 / Ms);
        const Real u1 = alongX ? up.w.u : up.w.v;
        if (alongX) s.w.u += du;
        else s.w.v += du;
        s.shockSpeed = u1 + sgn * Ms * c1; // lab frame
        return s;
    }

    Region parseRegion_(const std::string& spec) const {
        const auto toks = tokens_(spec);
        const auto colon = std::find(toks.begin(), toks.end(), ":");
        if (colon == toks.end() || colon + 1 == toks.end())
            throw std::runtime_error("region needs ': stateName' — " +
                                     spec);
        Region r;
        const NamedState& ns = lookupState_(*(colon + 1));
        r.st = ns.w;
        r.Y = ns.gas == 2 ? Real(1) : Real(0);
        const std::size_t n = std::size_t(colon - toks.begin());
        const auto need = [&](std::size_t want, const char* what) {
            if (n != want)
                throw std::runtime_error(std::string("malformed ") +
                                         what + " region: " + spec);
        };
        if (toks[0] == "halfplane") {
            if (n != 4 && n != 6)
                throw std::runtime_error("malformed halfplane: " + spec);
            r.shape = Region::Shape::HalfPlane;
            r.p[0] = num_(toks[1]);
            r.p[1] = num_(toks[2]);
            r.p[2] = num_(toks[3]);
            r.norm = std::sqrt(r.p[0] * r.p[0] + r.p[1] * r.p[1]);
            if (n == 6) {
                if (toks[4] != "speed")
                    throw std::runtime_error("expected 'speed': " + spec);
                if (toks[5] == "auto") {
                    // RH-derived state: front advances along the
                    // half-plane's own normal at the lab-frame shock
                    // speed.
                    if (ns.shockSpeed == 0)
                        throw std::runtime_error(
                            "speed auto needs a 'shock =' derived "
                            "state: " +
                            spec);
                    r.speed = std::fabs(ns.shockSpeed);
                } else {
                    r.speed = num_(toks[5]);
                }
            }
        } else if (toks[0] == "band") {
            need(4, "band");
            r.shape = toks[1] == "x" ? Region::Shape::BandX
                                     : Region::Shape::BandY;
            r.p[0] = num_(toks[2]);
            r.p[1] = num_(toks[3]);
        } else if (toks[0] == "rect") {
            need(5, "rect");
            r.shape = Region::Shape::Rect;
            for (int k = 0; k < 4; ++k) r.p[k] = num_(toks[1 + k]);
        } else if (toks[0] == "circle") {
            need(4, "circle");
            r.shape = Region::Shape::Circle;
            for (int k = 0; k < 3; ++k) r.p[k] = num_(toks[1 + k]);
        } else if (toks[0] == "sinex") {
            // sinex x0 amp lambda : X — everything left of the
            // cosine-perturbed interface x(y) = x0 + amp*cos(2 pi y/lambda)
            need(4, "sinex");
            r.shape = Region::Shape::SineX;
            for (int k = 0; k < 3; ++k) r.p[k] = num_(toks[1 + k]);
            if (r.p[2] == 0)
                throw std::runtime_error("sinex: lambda must be != 0");
        } else {
            throw std::runtime_error("unknown region shape: " + toks[0]);
        }
        return r;
    }

    // Solid region: the IC shape grammar without a state or motion —
    // "rect x0 x1 y0 y1 | circle cx cy r | halfplane a b c | band x|y lo hi
    //  | sinex x0 amp lambda". Reuses Region::inside (speed stays 0).
    Region parseSolidRegion_(const std::string& spec) const {
        const auto toks = tokens_(spec);
        const std::size_t n = toks.size();
        const auto need = [&](std::size_t want, const char* what) {
            if (n != want)
                throw std::runtime_error(std::string("malformed solid ") +
                                         what + " region: " + spec);
        };
        Region r;
        if (toks.empty()) throw std::runtime_error("empty solid region");
        if (toks[0] == "halfplane") {
            need(4, "halfplane");
            r.shape = Region::Shape::HalfPlane;
            r.p[0] = num_(toks[1]);
            r.p[1] = num_(toks[2]);
            r.p[2] = num_(toks[3]);
            r.norm = std::sqrt(r.p[0] * r.p[0] + r.p[1] * r.p[1]);
        } else if (toks[0] == "band") {
            need(4, "band");
            r.shape = toks[1] == "x" ? Region::Shape::BandX
                                     : Region::Shape::BandY;
            r.p[0] = num_(toks[2]);
            r.p[1] = num_(toks[3]);
        } else if (toks[0] == "rect") {
            need(5, "rect");
            r.shape = Region::Shape::Rect;
            for (int k = 0; k < 4; ++k) r.p[k] = num_(toks[1 + k]);
        } else if (toks[0] == "circle") {
            need(4, "circle");
            r.shape = Region::Shape::Circle;
            for (int k = 0; k < 3; ++k) r.p[k] = num_(toks[1 + k]);
        } else if (toks[0] == "sinex") {
            need(4, "sinex");
            r.shape = Region::Shape::SineX;
            for (int k = 0; k < 3; ++k) r.p[k] = num_(toks[1 + k]);
            if (r.p[2] == 0)
                throw std::runtime_error("solid sinex: lambda must be != 0");
        } else if (toks[0] == "triangle") {
            // triangle x1 y1 x2 y2 x3 y3 (sommets, ordre quelconque)
            need(7, "triangle");
            r.shape = Region::Shape::Triangle;
            for (int k = 0; k < 6; ++k) r.p[k] = num_(toks[1 + k]);
        } else {
            throw std::runtime_error("unknown solid region shape: " +
                                     toks[0]);
        }
        return r;
    }

    static Perturb parsePerturb_(const std::string& spec) {
        const auto toks = tokens_(spec);
        Perturb pb;
        const auto var = [&](const std::string& v) {
            if (v == "rho") return Prim::Var::Rho;
            if (v == "u") return Prim::Var::U;
            if (v == "v") return Prim::Var::V;
            if (v == "p") return Prim::Var::P;
            throw std::runtime_error("unknown perturb variable: " + v);
        };
        if (toks.size() == 4 && toks[1] == "sin") {
            pb.var = var(toks[0]);
            pb.kind = Perturb::Kind::Sin;
            pb.a = num_(toks[2]); // periods over the domain width
            pb.b = num_(toks[3]); // amplitude
            pb.c = 0;
        } else if (toks.size() == 6 && toks[1] == "sing") {
            pb.var = var(toks[0]);
            pb.kind = Perturb::Kind::SinG;
            pb.a = num_(toks[2]); // periods
            pb.b = num_(toks[3]); // amplitude
            pb.c = num_(toks[4]); // y center of the gaussian envelope
            pb.d = num_(toks[5]); // sigma
        } else if (toks.size() == 3 && toks[1] == "hydro") {
            if (toks[0] != "p")
                throw std::runtime_error(
                    "hydro perturb applies to p only");
            pb.var = Prim::Var::P;
            pb.kind = Perturb::Kind::Hydro;
            pb.a = num_(toks[2]); // reference height (p unchanged there)
        } else if (toks.size() == 5 && toks[1] == "erf") {
            pb.var = var(toks[0]);
            pb.kind = Perturb::Kind::Erf;
            pb.a = num_(toks[2]); // x0
            pb.b = num_(toks[3]); // width
            pb.c = num_(toks[4]); // amplitude
        } else {
            throw std::runtime_error("malformed perturb: " + spec);
        }
        return pb;
    }

    Spec parseSpec_(const std::vector<std::string>& toks, std::size_t i,
                    std::size_t end) const {
        Spec s;
        if (i >= end)
            throw std::runtime_error("empty boundary spec");
        const std::string& t = toks[i];
        if (t == "transmissive") s.type = Bc::Transmissive;
        else if (t == "reflective") s.type = Bc::Reflective;
        else if (t == "noslip") s.type = Bc::NoSlip;
        else if (t == "analytic") s.type = Bc::Analytic;
        else if (t == "inflow") {
            s.type = Bc::Inflow;
            if (i + 1 >= end)
                throw std::runtime_error("inflow needs a state name");
            const NamedState& ns = lookupState_(toks[i + 1]);
            s.inflow = ns.w;
            s.inflowG = 1 / (gammaOf_(ns.gas) - 1);
        } else
            throw std::runtime_error("unknown boundary type: " + t);
        return s;
    }

    Side parseSide_(const std::string& spec) const {
        const auto toks = tokens_(spec);
        Side side;
        const auto ifPos = std::find(toks.begin(), toks.end(), "if");
        if (ifPos == toks.end()) {
            side.a = parseSpec_(toks, 0, toks.size());
            return side;
        }
        // "<specA> if x|y < val else <specB>"
        const std::size_t i = std::size_t(ifPos - toks.begin());
        if (i + 4 > toks.size() || toks[i + 2] != "<")
            throw std::runtime_error("malformed split BC (expected 'if "
                                     "x < val else ...'): " +
                                     spec);
        const auto elsePos = std::find(toks.begin(), toks.end(), "else");
        if (elsePos == toks.end())
            throw std::runtime_error("split BC needs 'else': " + spec);
        side.a = parseSpec_(toks, 0, i);
        side.split = true;
        side.splitAt = num_(toks[i + 3]);
        side.b = parseSpec_(toks, std::size_t(elsePos - toks.begin()) + 1,
                            toks.size());
        return side;
    }

    // ---- ghost filling -----------------------------------------------------
    // dir: 0 left, 1 right, 2 bottom, 3 top. Column-/row-local types let
    // split sides pick the spec per ghost column.
    template <class G>
    void side_(G& g, double t, int dir) const {
        const Side& sd = sides_[dir];
        const bool xSide = dir < 2;
        const int n1 = xSide ? g.toty() : g.totx(); // sweep direction

        for (int s = 0; s < n1; ++s) {
            const Real coord = xSide ? g.yc(s) : g.xc(s);
            const Spec& sp =
                (sd.split && coord < sd.splitAt) ? sd.a
                : sd.split                       ? sd.b
                                                 : sd.a;
            for (int k = 0; k < NG; ++k) {
                int i, j, mi, mj; // ghost cell and mirror interior cell
                if (dir == 0) {
                    i = NG - 1 - k; j = s; mi = NG + k; mj = s;
                } else if (dir == 1) {
                    i = NG + g.nx + k; j = s;
                    mi = NG + g.nx - 1 - k; mj = s;
                } else if (dir == 2) {
                    i = s; j = NG - 1 - k; mi = s; mj = NG + k;
                } else {
                    i = s; j = NG + g.ny + k;
                    mi = s; mj = NG + g.ny - 1 - k;
                }
                switch (sp.type) {
                case Bc::Transmissive: {
                    // zero-gradient: copy nearest interior
                    const int ci = xSide ? (dir == 0 ? NG : NG + g.nx - 1)
                                         : mi;
                    const int cj = xSide ? mj
                                         : (dir == 2 ? NG : NG + g.ny - 1);
                    g.at(i, j) = g.at(ci, cj);
                    break;
                }
                case Bc::Reflective: {
                    Cons c = g.at(mi, mj);
                    if (xSide) c.mx = -c.mx;
                    else c.my = -c.my;
                    // Under gravity a mirrored pressure flips the
                    // hydrostatic gradient at the wall and pumps energy
                    // (boundary rows flip-flop until blow-up):
                    // extrapolate p hydrostatically instead.
                    const Real gn = xSide ? gx : gy;
                    if (gn != 0) {
                        Prim w = toPrim(c);
                        const Real dpos = xSide ? g.xc(i) - g.xc(mi)
                                                : g.yc(j) - g.yc(mj);
                        w.p = std::max(w.p + w.rho * gn * dpos, P_FLOOR);
                        c = toCons(w);
                    }
                    g.at(i, j) = c;
                    break;
                }
                case Bc::NoSlip: {
                    // viscous wall: mirror but flip BOTH momentum
                    // components -> zero wall velocity (adiabatic).
                    Cons c = g.at(mi, mj);
                    c.mx = -c.mx;
                    c.my = -c.my;
                    g.at(i, j) = c;
                    break;
                }
                case Bc::Analytic:
                    g.at(i, j) = state(g.xc(i), g.yc(j), t);
                    break;
                case Bc::Inflow:
                    g.at(i, j) = toConsG(sp.inflow, sp.inflowG);
                    break;
                }
            }
        }
    }

    Real gammaOf_(int gasIdx) const {
        if (!species_) return GAMMA;
        return gasIdx == 2 ? gas_.gamma2 : gas_.gamma1;
    }

    bool species_ = false;
    bool react_ = false;
    GasPair gas_;
    Reaction reaction_;
    Real defY_ = 0;
    std::map<std::string, NamedState> states_;
    Prim def_{};
    std::vector<Region> regions_;
    std::vector<Region> solids_; // immersed solid regions ([solid])
    std::vector<Perturb> perturbs_;
    Side sides_[4];
};

} // namespace mm

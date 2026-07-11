# Developer guide

How to contribute to the code: where to put what, the validation discipline,
the conventions and the pitfalls. See
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for the big picture and
[`docs/NUMERICS.md`](NUMERICS.md) for the schemes.

---

## 1. Where code goes

Layers with downward dependencies (only include upward):

| Directory | Role | Depends on the GPU? |
|---|---|---|
| `core/` | types, grid, ghosts, INI, parallel | no |
| `physics/` | Euler, two-gas, reaction | no |
| `numerics/` | HLLC, limiter, exact Riemann | no |
| `solver/` | single-grid schemes (MUSCL, WENO5, species) | no |
| `amr/` | AMR hierarchy (Amr2/AmrML CPU; AmrGpu/AmrGpuML hybrid) | hybrid only |
| `backend/` | Metal (MetalContext, Euler2DGpu) | yes |
| `io/`, `render/` | VTK/checkpoint/log; live view | render: yes |
| `cases/` | `CaseDef` (declarative case) | no |
| `drivers/` | executables (the `run` runner + validation suites) | per driver |
| `shaders/` | Metal kernels (`euler2d.metal`, `render.metal`) | — |

Everything is **header-only** (`.hpp`) except the Metal backend; the drivers
are the only `.cpp` files (along with `MetalContext.cpp`).

---

## 2. Validation discipline (the golden rule)

> Every functional addition comes with a **quantitative gate**, and we
> **commit each development once it is finished AND validated**.

For each feature:

1. **A quantitative gate**: a driver that returns `PASS`/`FAIL` (exit 0/1) on
   a numeric metric (L1 error, observed order, mass drift, CPU/GPU
   lock-step…). No "looks good".
2. **Ideally a declarative case** (`.ini`) for usage.
3. **CI coverage** (CPU or GPU suite).
4. **ROADMAP updated** in the same commit.
5. **Commit** to `main`, without tool attribution.

```mermaid
flowchart LR
    A["code the feature"] --> B["gate driver (numeric PASS/FAIL)"]
    B --> C["target it in CMakeLists"]
    C --> D["build + run the gate locally"]
    D --> E{"PASS?"}
    E -- no --> A
    E -- yes --> F["add to CI"]
    F --> G["update ROADMAP"]
    G --> H["commit (one per validated dev)"]
```

---

## 3. Recipe: add a driver / a gate

**a) Write `src/drivers/mycase.cpp`** — compute a metric and return `0`
(PASS) or `1` (FAIL):
```cpp
int main() {
    // ... set up a case, measure ...
    const bool ok = error < tol;
    std::printf("%s (err %.3e, tol %.3e)\n", ok ? "PASS" : "FAIL", err, tol);
    return ok ? 0 : 1;
}
```

**b) Declare the target in `CMakeLists.txt`** — two patterns:
```cmake
# pure CPU (core/physics/numerics/solver/amr CPU):
add_executable(mycase src/drivers/mycase.cpp)
target_include_directories(mycase PRIVATE src)

# needs the GPU (AmrGpu/AmrGpuML/Euler2DGpu):
add_executable(mycase src/drivers/mycase.cpp)
target_link_libraries(mycase PRIVATE mm_metal)   # brings src as include
# + mm_render if a live view
```
Then reconfigure: `cmake -B build && cmake --build build --target mycase`.

> ⚠️ **Always configure/build in `build/`** (`cmake -B build …`). Running
> `cmake .` at the root pollutes the repo with in-source artifacts.

**c) Add to CI** (`.github/workflows/ci.yml`) — the build compiles
everything; just add the execution line to the right suite:
```yaml
# CPU suite (GPU-less machine)          # GPU suite (Metal runner)
./build/mycase                          ./build/mycase
```

**d) Update `ROADMAP.md`** and **commit**.

---

## 4. Conventions & invariants

- **`Real = float` everywhere** — Metal has no `double`. The CPU aligns for
  the bit-for-bit lock-step and the zero-copy `float4` layout.
- **`NG = 3` ghosts** (`core/Grid.hpp`) — required by WENO5; MUSCL uses 2.
  Every consumer of the memory (including the **renderer**) must respect
  `NG`: the bit-identical check must cover *all* consumers (a rendering bug
  hardcoded to `NG=2` was a reminder).
- **CPU/GPU lock-step**: each GPU path has a bit-identical CPU reference. The
  CPU classes (`Amr2`, `AmrML`) do not touch Metal.
- **Physical BCs of edge patches: at the fine level** (`fillPatchPhysical`) —
  prolongating coarse ghosts breaks consistency at the boundaries.
- **Conservation gates** calibrated on the **measured fp32 rounding floor**
  (~1e-8/step per active patch), not on an ideal value; the discriminating
  test is the contrast with/without refluxing.
- **Square cells**: `nx/ny = (x1-x0)/(y1-y0)`; `nx,ny` multiples of
  `amr.block`.
- **Order measured in smooth regions**: TVD limiters cap the order at ~1 at
  smooth extrema, the midpoint face flux caps multi-D near 2, and the fp32
  floor caps everything at large N.

---

## 5. Working on the GPU

- The kernels live in `shaders/euler2d.metal`, **compiled at runtime**
  (`MetalContext::compileLibrary`) — no Xcode required.
- `Euler2DGpu` wraps device/pipelines and exposes `enableWeno/Species/
  Reaction`, `step`, `react`, `maxStableDt`. `AmrGpuML` orchestrates the
  hierarchy (a single **slot pool** for all patches).
- **Unified memory**: buffers are `StorageModeShared`; the CPU reads/writes
  the same bytes as the GPU, without a copy. Any layout change must stay
  consistent on both sides (and with the renderer).
- The pool has a **capacity** (`amr.max_patches`, default ~1/8 of the working
  set); its exhaustion raises an actionable error, not a crash.

The four AMR classes and their automatic selection: `Amr2`/`AmrGpu` (fast 2
levels); `AmrML`/`AmrGpuML` (arbitrary depth, two-gas, WENO5, reaction). A
physics change is done first on the CPU side (`AmrML`), then the GPU kernel is
ported and the lock-step is locked in.

---

## 6. Testing locally

```sh
cmake --build build -j                 # everything
./build/<gate>                          # one gate (exit 0 = PASS)
for c in cases/*.ini; do ./build/run --check "$c"; done   # cases parse
```

CI (`.github/workflows/ci.yml`) replays: a **CPU suite** (sod*, dmr*,
weno_suite, convergence, mms, reactor, blasius, casedef_test, `--check` of all
cases…) and a **GPU suite** (dmr_gpu, mlgpu_amr, detonation, hs_suite,
lock-steps…). The heavy studies (benchmark, deflagration) are manual (compiled
in CI, not executed).

---

## 7. What is out of scope

No MPI / multi-machine, no turbulence model (RANS/LES), no global implicit
solver, no "production" generality (unstructured meshes). On the other hand,
an **industrial-grade UX** *is* a goal (ease of use, code readability). See
[`ROADMAP.md`](../ROADMAP.md).

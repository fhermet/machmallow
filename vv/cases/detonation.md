# Chapman–Jouguet detonation — *validation vs exact $D_{CJ}$*

**Objective.** A 1D reactive-Euler detonation in a closed tube: once the
overdriven ignition relaxes (via the Taylor rarefaction), the leading shock
must settle at the exact **Chapman–Jouguet speed** $D_{CJ}$.

## Numerical setup
> Reactive Euler — single-step Arrhenius reaction (q = 10) + heat release,
> MUSCL-Hancock + HLLC, **Strang split** $R(\tfrac{dt}{2})\,\mathcal{H}(dt)\,R(\tfrac{dt}{2})$,
> CFL 0.4, **closed tube of length L = 16** (reflecting walls, hot ignition) —
> long enough for the overdrive to relax to CJ (see discussion). Run
> **uniform** and on a **3-level AMR** hierarchy (CPU *and* GPU, refining the
> reaction zone). $D_{CJ}$ solved exactly from Rankine–Hugoniot + the CJ
> tangency condition. Driver: `detonation <L>` (default L = 8 in CI).

## Results
![CJ detonation relaxation](../figures/detonation.png)

| Speed | Value | vs $D_{CJ}$ |
|---|---|---|
| $D_{CJ}$ exact | 4.6809 | — |
| uniform | 4.6986 | 0.4 % (gate 3 %) |
| 3-level AMR (CPU) | 4.6723 | -0.2 % (gate 5 %) |
| 3-level AMR (GPU) | 4.6723 | -0.2 % (lock-step) |

## Discussion
The strong ignition is **overdriven**: the front starts faster than CJ and
**relaxes asymptotically** toward $D_{CJ}$ as the trailing Taylor
rarefaction drains its support (the measured speed decays onto the dashed
line). The decay is slow, so the tube must be **long enough** — at **L = 16**
the uniform speed is within **+0.4 %** and the AMR within **−0.2 %** of
$D_{CJ}$, whereas a short L = 8 tube still reads +1.3 %. AMR further helps by
resolving the thin reaction zone (a sharper von Neumann spike); its GPU path
is bit-for-bit with the CPU. A transmissive boundary would act as an infinite
reservoir and keep the detonation permanently overdriven — hence the closed
tube.

---
*Part of the [V&V dossier](../README.md). Regenerate: `python3 vv/generate.py`. Source data: [`../data/`](../data/).*

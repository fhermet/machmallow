# machmallow — pistes futures

La feuille de route initiale (phases 0–7) est complète : solveur Euler +
Navier-Stokes 2D, AMR block-structured hybride CPU/GPU avec subcycling,
validé à chaque étage (voir README). Ce document liste les extensions
possibles, par thème et avec leur point d'entrée dans le code.

## Numérique

- [ ] **AMR multi-niveaux (3+)** — généraliser `Amr2`/`AmrGpu` (2 niveaux)
  en hiérarchie récursive : niveau L+1 raffine des blocs du niveau L.
  Le subcycling devient récursif (2 sous-pas par niveau). Les briques
  (prolongation, restriction, refluxing, ghosts θ-blendés) sont déjà
  par-niveau ; le travail est surtout structurel.
  *Entrée : `src/amr/Amr2.hpp`, `src/amr/AmrGpu.hpp`.*
- [ ] **Schémas d'ordre élevé** — WENO5 + RK3-SSP en option du
  MUSCL-Hancock. Demande 3 ghosts (NG=3) et des stencils plus larges →
  impact sur le ghost fill AMR et les kernels Metal.
  *Entrée : `src/numerics/`, `shaders/euler2d.metal`, `core/Grid.hpp` (NG).*
- [ ] **Ratio de raffinement 4** — réduit le nombre de niveaux nécessaires ;
  touche prolongation/restriction/refluxing (4 faces fines par face
  grossière) et le θ-blend (4 sous-pas).
- [x] ~~Tagging plus riche~~ — fait : critère vitesse
  (`AmrConfig.tagVelocity`, saut de vitesse normalisé par c locale) en
  plus du gradient de densité ; validé par `kh_amr` (cisaillement
  isochore raffiné, invisible au critère densité). Extensions possibles :
  gradient de pression, estimateur de Richardson.
- [ ] **Gamma / gaz paramétrables** — `GAMMA` est `constexpr`
  (`core/Types.hpp`) et dupliqué dans le shader ; passer en paramètre de
  cas (struct physique + `Params` GPU).

## Physique

- [x] ~~Plomberie visqueuse AMR~~ — `AmrConfig.mu` câblé partout (CPU et
  GPU, dt visqueux avec réduction ρ_min sur GPU). **Reste** : un cas de
  validation visqueux *avec raffinement* (couche limite plaque plane /
  Blasius) — le tagging actuel (gradient de ρ) ne raffine pas les
  écoulements isochores, voir « tagging plus riche ».
- [ ] **Parois no-slip** — BC réfléchissante actuelle = slip ; ajouter le
  miroir complet (u, v inversés) + paroi isotherme/adiabatique pour les
  cas NS muraux. *Entrée : `core/Boundary.hpp`.*
- [x] ~~Kelvin-Helmholtz + BCs périodiques~~ — fait : domaines périodiques
  complets sur AMR (wrap des ghosts de patchs, du refluxing aux raccords
  et des masques de bords ; conservation exacte vérifiée en domaine
  fermé par `kh_amr`), preset `kh` dans `run` (`cases/kh.ini`).
- [ ] **Rayleigh-Taylor** — demande les termes sources (gravité), voir
  ci-dessous ; les BCs et le tagging nécessaires existent désormais.
- [ ] **Termes sources** — gravité (RT), pour commencer en source splittée.

## Performance

- [ ] **Pipelining des soumissions GPU** — le pas hybride est synchrone
  (waitUntilCompleted) car le CPU remplit les ghosts entre deux pas.
  Piste : découpler ghost fill par dépendances de patchs, ou recouvrir
  le regrid/reflux CPU avec le pas GPU suivant via double buffering.
- [ ] **Ghost fill sur GPU** — kernels de copie sibling + prolongation ;
  supprimerait la section CPU restante (~10 % du pas) et une partie des
  syncs. Le regrid resterait CPU.
- [ ] **Kernels fusionnés** — predictor+flux en un kernel avec tuiles en
  threadgroup memory ; à profiler avant d'investir (le pas est déjà
  ~80 % calcul GPU utile).
- [ ] **Profiling Metal System Trace** — nécessite Xcode complet (les CLT
  seules n'ont pas `xctrace`) ; l'instrumentation chrono intégrée
  (`AmrGpu::timings`) couvre le besoin en attendant.

## Qualité / outillage

- [ ] **CI** — GitHub Actions runner macOS : build + suite de régression
  (`sod1d`, `sod2d`, `sod_amr`, `shear`, `dmr_gpu`, `dmr_amr` en petit).
  Tous les drivers retournent déjà un code d'échec exploitable.
- [x] ~~Configs de cas en fichiers~~ — fait : parseur INI header-only
  (`core/Config.hpp`) + driver générique `run` (`./build/run
  cases/dmr.ini`), presets sod/dmr/shear, backends cpu/hybrid, AMR et μ
  configurables. Extension naturelle : nouveaux presets dans
  `caseGeometry`/`wireCase` (src/drivers/run.cpp).
- [ ] **Rendu temps réel** — fenêtre Metal affichant ρ pendant la simu
  (les données sont déjà dans des buffers GPU partagés — il ne manque
  qu'une passe de rendu et une boucle d'événements).
- [ ] **Restart / checkpointing** — dump binaire de la hiérarchie pour
  reprendre un calcul long.

## Leçons à retenir (notes de conception)

- **BCs physiques des patchs de bord : toujours au niveau fin**
  (`fillPatchPhysical`). La prolongation des ghosts coarse aux bords du
  domaine casse la cohérence dès qu'une onde touche la frontière — bug
  découvert en phase 7, deux ordres de grandeur sur la conservation.
- Les portes de conservation en float32 se calibrent sur le **plancher
  d'arrondi mesuré** (~1e-8/pas par patch actif), pas sur une valeur
  idéale ; le test discriminant est le contraste avec/sans refluxing sur
  maillage figé traversé par les ondes.
- Benchmarks sur Apple Silicon : variance ±30 % sur les petits cas
  (gouverneur de fréquence GPU) — toujours best-of-N, et les gros cas
  sont plus fiables.

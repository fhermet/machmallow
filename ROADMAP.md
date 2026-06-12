# machmallow — pistes futures

La feuille de route initiale (phases 0–7) est complète : solveur Euler +
Navier-Stokes 2D, AMR block-structured hybride CPU/GPU avec subcycling,
validé à chaque étage (voir README). Ce document liste les extensions
possibles, par thème et avec leur point d'entrée dans le code.

## Numérique

- [x] ~~AMR multi-niveaux (3+)~~ — fait : `AmrML` (CPU) + `AmrGpuML`
  (hybride, un seul pool de slots pour tous les niveaux), récursion
  Berger-Colella complète (subcycling, ghosts θ-blendés, refluxing par
  paire, nesting forcé au regrid, cadence de regrid par niveau), clé
  `amr.levels` dans `run`. Validé : AmrML(2) ≡ Amr2 bit-exact ; DMR
  3 niveaux finest 1/1024 en 31 s (rouleaux KH de la ligne de glissement
  résolus) ; conservation périodique au plancher fp32 à 3 niveaux.
  **Restes** : checkpoint multi-niveaux ; perf des petits patchs profonds
  (sync par niveau par sous-pas) ; précision ratio ~1.4 vs uniforme-fin
  (churn de déraffinement + mémoire du contact → estimateur de
  Richardson).
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
- [x] ~~Rayleigh-Taylor + gravité~~ — fait : source splittée
  (`[physics] gravity`, énergie au point milieu, cell-locale donc
  refluxing intact), perturbations `p hydro` (équilibre hydrostatique)
  et `sing` (seed sinus × gaussienne), murs reflective hydrostatiques
  sous gravité (le miroir de pression inversait le gradient → pompage
  d'énergie jusqu'au blow-up, diagnostiqué via le journal CSV).
  `cases/rt.ini` : zone de mélange turbulente complète à t=12 (1/512).
  Gates : chute libre exacte (3e-7) + lock-step CPU/GPU gravité.
  Extension : autres sources (Coriolis, chauffage).

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

- [x] ~~CI~~ — fait : `.github/workflows/ci.yml` (macos-15) — build + les
  3 harnais CPU obligatoires + les harnais GPU avec sonde Metal (skip
  propre si le runner n'a pas de device, échec franc sinon). Attention :
  les minutes macOS comptent 10× sur les dépôts privés.
- [x] ~~Configs de cas en fichiers~~ — fait, puis **industrialisé** :
  solveur 100 % déclaratif (`cases/CaseDef.hpp`) — états nommés, IC par
  régions (fronts de choc mobiles inclus), perturbations, BCs par côté
  segmentables + `analytic` (réévaluation des régions au temps t : BC
  exacte du DMR sans C++). Plus de presets dans run.cpp. Outils :
  `run --check` (config effective + warnings clés inconnues),
  `run --list`, `cases/TEMPLATE.ini`, gate `casedef_test` (équivalence
  bit-près avec les presets historiques), smoke `--check` de tous les
  cas en CI. Puis : **états Rankine-Hugoniot dérivés** (`shock = etat
  mach M [dir]`, vitesse de front `speed auto`), **pré-vol** (états
  affichés, warning cellules non carrées avec correction suggérée,
  estimation pas/coût/temps) et **`run --preview`** (IC en .vti sans
  calcul). Extensions possibles : nouvelles formes de régions (ellipse,
  polygone), perturbations 2D/le long de y, diagnostics CSV (item
  dédié ci-dessous).
- [x] ~~Diagnostics intégraux CSV~~ — fait : `[diagnostics] every/file`
  → journal CSV du calcul (`io/Diagnostics.hpp`) : step, t, dt,
  cellules, patchs, extrema ρ/p, masse, énergies cinétique/totale,
  enstrophie, temps mur, débit, et **résidus par équation** (RMS de
  dU/dt sur la base restreinte : masse, qdm x/y, énergie — le suivi de
  convergence standard) — composite sur toute la hiérarchie
  (2 niveaux et ML), flush par ligne. Limitation documentée : la
  vorticité utilise des différences clampées par grille (biais aux
  coutures de patchs — tendances fiables, valeurs absolues aux
  interfaces raides sous-estimées). Extension possible : sondes
  ponctuelles, quantités au choix.
- [ ] **Rendu temps réel** — fenêtre Metal affichant ρ pendant la simu
  (les données sont déjà dans des buffers GPU partagés — il ne manque
  qu'une passe de rendu et une boucle d'événements).
- [x] ~~Restart / checkpointing~~ — fait : `io/Checkpoint.hpp` (dump
  binaire coarse + patchs + t + compteur de pas, `restoreBlocks` dans les
  deux classes AMR), clés `output.checkpoint` / `restart` dans `run`.
  Le run coupé/repris est bit-identique au run continu (porte dans
  `kh_amr`).

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

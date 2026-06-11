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
- [ ] **Tagging plus riche** — critère sur le gradient de pression ou
  l'estimateur de Richardson, en plus du gradient de densité.
  *Entrée : `regrid()` dans les deux classes AMR.*
- [ ] **Gamma / gaz paramétrables** — `GAMMA` est `constexpr`
  (`core/Types.hpp`) et dupliqué dans le shader ; passer en paramètre de
  cas (struct physique + `Params` GPU).

## Physique

- [ ] **Cas visqueux sur AMR** — la plomberie existe (les flux visqueux
  vivent dans les mêmes tableaux de faces, le refluxing les traite
  automatiquement ; `Params.mu` est câblé dans le pool). Il manque :
  un `mu` exposé dans `AmrConfig`, la contrainte de dt visqueuse côté
  `maxStableDtAll` GPU (réduction de ρ_min), et un cas de validation
  (couche limite sur plaque plane, blasius).
- [ ] **Parois no-slip** — BC réfléchissante actuelle = slip ; ajouter le
  miroir complet (u, v inversés) + paroi isotherme/adiabatique pour les
  cas NS muraux. *Entrée : `core/Boundary.hpp`.*
- [ ] **Instabilités KH / RT** — cas tests spectaculaires et sensibles à la
  diffusion numérique (étaient dans les options de départ) ; KH demande
  des BCs périodiques (nouveau type de bord, simple).
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
- [ ] **Configs de cas en fichiers** — le dossier `cases/` prévu au départ
  est vide ; les paramètres sont en dur dans les drivers. Un parseur
  TOML/INI simple suffirait.
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

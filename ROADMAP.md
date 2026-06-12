# machmallow — vision et roadmap

## Vision

machmallow est un **laboratoire personnel de CFD compressible sur Apple
Silicon**, à quatre facettes indissociables :

1. **Comprendre en construisant** — chaque méthode numérique est
   implémentée from scratch, lisible de bout en bout (mini-AMReX) ;
2. **Prouver par des références** — rien n'entre sans porte de
   validation quantitative (solutions exactes, théorie, conservation) ;
3. **Exploiter et montrer** — étudier la physique des instabilités
   compressibles avec une expérience native Mac (zéro-copie, temps réel) ;
4. **Utilisable comme un outil industriel** — simple et pratique : un
   fichier de cas suffit, le pré-vol attrape les erreurs avant le
   calcul, le journal suit la convergence, les sorties s'exploitent
   directement. La cible d'usage : poser un cas et obtenir un résultat
   fiable sans lire le code.

Règle directrice : **un ajout = un fichier de cas qui le pilote + une
porte qui le verrouille + une UX au niveau du reste.** Tout passe par le
système déclaratif et la CI.

## Acquis (v1.0 — fondations, complet)

Solveur Euler/Navier-Stokes 2D float32, MUSCL-Hancock + HLLC avec
positivité, AMR block-structured multi-niveaux hybride CPU/GPU
(subcycling Berger-Colella, refluxing, nesting, périodique), gravité,
cas 100 % déclaratifs (régions, fronts RH mobiles, BCs analytiques),
outillage complet (pré-vol, preview, journal CSV avec résidus,
checkpoint 2 niveaux, CI macOS) et validation : Sod/Toro T2-T5 vs
Riemann exact, acoustique (ordre TVD 4/3), vortex isentropique (2.35),
Sedov (0.490), couche erf visqueuse, chute libre, conservation au
plancher fp32, lock-steps CPU/GPU partout. Détails : git log.

## Jalons

### Fil rouge « outil industriel » *(transverse, un peu à chaque jalon)*
- [ ] Guide utilisateur (poser un cas en 10 minutes, lire le journal,
  exploiter les sorties) — distinct de la doc du code.
- [ ] Post-traitement fourni : script de tracé (champs + journal CSV)
  prêt à l'emploi, sans dépendre de ParaView pour le quotidien.
- [ ] Messages d'erreur systématiquement actionnables (fichier:ligne,
  correction suggérée) ; sondes ponctuelles ; checkpoint multi-niveaux
  (reprendre n'importe quel calcul).

### v1.1 — Voir et mesurer *(démo + labo)*
Le projet gagne ses yeux et son instrumentation scientifique.
- [x] ~~Rendu temps réel Metal~~ — fait : `[render] live = true` —
  fenêtre Cocoa programmatique (sans Xcode) + fragment shader qui
  échantillonne les buffers de SIMULATION par pixel (descente de la
  hiérarchie du plus fin à la base via les tables bloc→slot, zéro
  copie), colormap viridis, contours de patchs, pause (espace) et arrêt
  propre (q/fermeture). ~15 %% d'overhead sur le KH, résultats
  bit-identiques au run headless. Dégradation propre en headless (CI).
  Extension : mode schlieren, champ au choix.
- [x] ~~Taux de croissance RT vs théorie linéaire~~ — fait :
  amplitude de Fourier du mode semé à l'interface (immune aux
  harmoniques qui polluent l'énergie globale), fit LSQ sur la fenêtre
  linéaire → **σ = 1.440 vs √(Agk) = 1.447 (0.5 %)**, gate ±15 % dans
  `analytic_suite`. Constat documenté : le KH en feuille de vorticité
  n'est PAS gateable (mal posé, σ ∝ k — l'échelle résolue la plus fine
  gagne ; le seed uniforme en y se projette mal sur le mode propre
  localisé) — une gate KH propre demanderait un profil tanh + les
  valeurs propres de Michalke (backlog).
- **Sortie v1.1 : ATTEINTE** — démo live (KH/RT/bulle) + gate de
  croissance RT en CI.

### v1.2 — Physique des mélanges *(labo)*
De « densités différentes » à « gaz différents ».
- [x] ~~Cœur bi-gaz (grille uniforme CPU)~~ — fait : φ = ρY conservatif
  (flux de masse HLLC × Y amont — Y exactement constant où uniforme),
  Γ = 1/(γ−1) advecté quasi-conservativement par la vitesse de contact
  S* (l'EOS lit ce Γ : poids de mélange E/Γ cohérents), HLLC bi-γ,
  Riemann exact généralisé aux γ par côté. `species_suite` : advection
  d'interface |p−1| 1.0 %% (démarrage) / 0.6 %% (entretenu), Sod bi-γ à
  1.2e-3 de l'exact, masse d'espèce au plancher fp32. Reconstruction
  PRIMITIVE (ρ,u,v,p)+Γ avec faces Γ avancées du demi-pas (la
  désynchronisation E/Γ était la moitié du résidu) ; le reliquat vient
  du mélange E des états-étoiles HLLC — remède définitif = double-flux
  d'Abgrall-Karni (optionnel, plus tard).
- [x] ~~Plomberie AMR multi-espèces (CPU)~~ — fait : φ/Γ dans AmrML
  (ghosts sibling + prolongation θ-blendée, restriction, refluxing de φ
  avec back-out/fine-apply, dt espèces) ; single-gas bit-inchangé.
  Gate : Sod bi-γ sur AMR 3 niveaux subcyclé, L1 3.1e-3 vs exact,
  masse d'espèce 4.4e-5 (plancher fp32 relatif — φ porte ~9× moins de
  masse que ρ). Bug attrapé : les ghosts scalaires physiques écrasaient
  les 4 côtés (y compris intérieurs) → version masquée par côté.
- [x] ~~GPU multi-espèces~~ — fait : kernels Metal bi-γ en miroir de
  `step2DY` (reconstruction primitive + Γ de face advecté demi-dt,
  HLLC γ-par-côté, flux φ upwindé, transport quasi-conservatif de Γ),
  scalaires (φ, Γ) en float2 par cellule dans le pool de slots,
  plomberie complète AmrGpuML (ghosts θ-blendés, restriction, reflux
  de φ). Gate : Sod bi-γ 3 niveaux GPU en lock-step CPU complet —
  L1 2.62e-3 vs exact (= CPU à 1e-7 près), masse d'espèce 2.0e-6,
  écart CPU/GPU 1.6e-4 (fp32), patchs identiques.
- [x] ~~CaseDef bi-gaz + cas vitrines~~ — fait : section `[species]`
  (gamma1/gamma2), `gas = 2` par état, états RH calculés avec le γ du
  gaz amont, inflow/analytic fermés sur le bon Γ, nouvelle région
  `sinex` (interface cosinus). `bubble.ini` devient la vraie bulle He
  (γ 1.667 dans l'air, Haas & Sturtevant) et `rm.ini` ajoute le
  Richtmyer-Meshkov air/SF6 (γ 1.09, mode unique périodique). Les cas
  bi-gaz passent par AmrML/AmrGpuML à toute profondeur (les classes
  2 niveaux n'ont pas les champs espèces).
- [ ] **Suite multi-espèces** : gate quantitative Haas & Sturtevant
  (vitesses d'interface vs expérience), masse d'espèce dans le log de
  diagnostics, Y dans les sorties VTK ; double-flux Abgrall-Karni en
  option si les 0.6 %% gênent.
- [ ] **Cas** : vraie bulle d'hélium dans l'air (Haas & Sturtevant
  quantitatif), Richtmyer-Meshkov.
- **Sortie** : gate d'interface (pas d'oscillation de p), bulle He
  comparée aux vitesses caractéristiques publiées.

### v1.3 — Ordre élevé *(pédago + labo)*
Lever la limite TVD quantifiée par la suite analytique (ordre 4/3 aux
extrema lisses).
- [ ] **WENO5 + RK3-SSP** en option du MUSCL (NG = 3, kernels Metal,
  impact ghost fill AMR).
- **Sortie** : acoustique à l'ordre ~5 en lisse, interfaces RT/KH/RM
  visiblement plus fines à résolution égale.

### v1.4 — La troisième dimension *(démo + pédago)*
Le grand chantier, mené comme le multi-niveaux : CPU de référence →
portes de validation → GPU.
- [ ] Extension 3D du cœur (Grid, schéma, AMR, pool, CaseDef, rendu).
- **Sortie** : un cas 3D AMR ~100M cellules effectives sur le M4,
  visualisé en temps réel.

## Backlog (tiré dans un jalon quand il sert, jamais en direct)

Mode stationnaire (local time stepping — donnerait tout leur sens aux
résidus du journal) ; no-slip + Blasius/Couette ; tagging de Richardson ;
checkpoint multi-niveaux ; perf (pipelining GPU, ghost fill GPU, syncs
multi-niveaux) ; sondes ponctuelles ; régions ellipse/polygone ; Riemann
2D 4-quadrants ; ratio de raffinement 4 ; sources additionnelles
(Coriolis, chauffage) ; Metal System Trace (nécessite Xcode complet).

## Non-objectifs (pour rester net)

- Pas de MPI / multi-machine — un Mac, c'est le cadre du projet.
- Pas de modèles de turbulence (RANS/LES) ni de solveur implicite.
- Pas un code de production en *échelle* (maillages non structurés,
  généralité tous-azimuts) — mais une **UX de niveau industriel** est,
  elle, un objectif explicite (facette 4) : la simplicité d'usage prime
  sur la généralité, la lisibilité du code sur l'optimisation extrême.

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
- L'ordre de convergence se mesure **en régime lisse ET sur le bon
  régime de grilles** : les limiteurs TVD plafonnent à 4/3 aux extrema
  lisses (sinus), le raidissement non linéaire O(A²) plafonne les tests
  à retour-sur-IC aux grandes N, et le plancher fp32 plafonne tout en
  dessous de ~1e-6 — chaque gate doit savoir lequel des trois elle
  mesure.
- Murs réfléchissants sous gravité : extrapoler la pression
  hydrostatiquement dans les ghosts (le miroir inverse le gradient et
  pompe de l'énergie jusqu'au blow-up).
- Tagging : les stencils doivent lire les ghosts (jamais clamper aux
  coutures de patchs) ; le wrap périodique s'applique à TOUS les niveaux
  y compris les coordonnées parent de la prolongation ; la cadence de
  regrid doit être par niveau (buffer invariant d'échelle).

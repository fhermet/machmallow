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
- [ ] **Rendu temps réel Metal** : fenêtre affichant ρ (ou schlieren)
  pendant la simu, contours de patchs AMR, pause/reprise. Les données
  sont déjà dans des buffers GPU partagés — il manque la passe de rendu.
- [ ] **Taux de croissance RT/KH vs théorie linéaire** : fit du journal
  CSV (énergie cinétique ~ e^{2σt}) comparé à σ = √(Agk) (RT) et au
  taux KH ; nouvelle gate dans `analytic_suite`.
- **Sortie** : démo live du KH/RT + gate de croissance en CI.

### v1.2 — Physique des mélanges *(labo)*
De « densités différentes » à « gaz différents ».
- [ ] **Multi-espèces** : fraction massique advectée + γ(Y) de mélange
  (schéma quasi-conservatif d'Abgrall, sans oscillations de pression à
  l'interface) ; γ paramétrable par cas en sous-produit.
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

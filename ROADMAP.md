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
- [x] ~~Diagnostics & sorties bi-gaz~~ — fait : colonne `species_mass`
  dans le log CSV (conservée au plancher fp32 sur frontière fermée),
  champ `Y` dans les .vti/.vthb et pression fermée sur le Γ local
  (l'ancienne sortie fermait sur γ=1.4 partout : p faux de ~20 %% dans
  l'hélium).
- [x] ~~Gate quantitative Haas & Sturtevant~~ — fait : `hs_suite`
  rejoue l'expérience (choc Mach 1.22 dans l'air, cylindre He+28 %%
  d'air, γ 1.645, ρ/ρ_air 0.182) et mesure les vitesses d'interface
  sur l'axe (croisements Y=0.5, pentes LSQ sur les fenêtres des
  régimes mesurés par l'expérience). Résultats vs H&S 1987 :
  V_amont +6.7 %%, V_aval +5.6 %%, jet −0.7 %% (gates ±10 %% ;
  Quirk & Karni 1996 : +4.7/+0.7/−1.3 %%). bubble.ini aligné sur
  l'expérience (Mach 1.22, He contaminé). Leçon : l'interface amont
  ACCÉLÈRE vers le jet — la « vitesse initiale » de l'expérience se
  mesure juste après le passage du choc, pas sur une fenêtre tardive.
- [x] ~~Cas RM corrigé~~ — le champignon est non-linéaire (ka > 1) :
  choc rapproché de l'interface, t_end 8 (ka final 1.75, enroulement),
  BC gauche transmissive (une BC analytic réinjecte le choc réfléchi).
  Bug corrigé au passage : les ghosts scalaires de la base étaient
  transmissifs même sur les axes périodiques (wrap désormais).
- [ ] Double-flux Abgrall-Karni en option si les 0.6 %% gênent
  (backlog v1.2).
- **Sortie v1.2 : ATTEINTE** — bulle He (Haas & Sturtevant) et
  Richtmyer-Meshkov air/SF6 pilotées par fichier de cas, gate
  d'interface (pas d'oscillation de p) + gate quantitative des
  vitesses caractéristiques (`hs_suite`) en CI.

### v1.3 — Ordre élevé *(pédago + labo)*
Lever la limite TVD quantifiée par la suite analytique (ordre 4/3 aux
extrema lisses).
- [x] ~~NG = 3~~ — fait : 3 couches de ghosts partout (stencil WENO5),
  toutes les suites bit-identiques, checkpoint v2.
- [x] ~~Cœur WENO5 + RK3-SSP (grille uniforme, CPU)~~ — fait :
  FD-WENO5 Jiang-Shu, splitting Lax-Friedrichs local par composante,
  flux de faces conservés pour le refluxing à venir. `weno_suite` :
  onde d'entropie à l'ordre 3.02 (= cap RK3 exactement, 4.84 sur la
  paire grossière où le spatial domine, spatial seul ~4 en
  pré-asymptotique), vortex ordre 3.55 et dissipation 5.5× plus
  faible que MUSCL à 64², Sod borné sans over/undershoot (contact
  +19 %% vs HLLC — LLF diffuse le contact, attendu).
- [x] ~~WENO5 dans l'AMR (CPU)~~ — fait : `cfg.weno`, 3 étages RK
  synchrones par niveau avec ghosts re-remplis aux temps d'étage
  (θ = (k+c)/n), flux accumulés aux poids (1/6, 1/6, 2/3) consommés
  par le refluxing existant. Gate brutale : hiérarchie 2 niveaux
  tout-raffinée périodique = grille uniforme **bit-à-bit** (0 diff
  sur 40 pas). Sod 3 niveaux : L1 1.47× MUSCL sur la même hiérarchie
  (la prolongation 2ᵉ ordre plie les stencils WENO aux interfaces
  c-f — connu ; l'intérêt haut-ordre est dans l'intérieur lisse),
  conservation 2.6e-6.
- [x] ~~Kernels Metal WENO5 + `scheme = weno5`~~ — fait : kernels
  flux WENO (accumulation RK intégrée, zérotée à l'étage 0) + update
  RK (capture de u⁰ à l'étage 0), 3 allers-retours GPU par pas de
  niveau (ghosts CPU entre étages). Gate : Sod 3 niveaux GPU en
  lock-step CPU complet — L1 identique à 5 chiffres (4.5784e-3),
  écart 4e-4, patchs identiques. `scheme = weno5` dans les .ini
  (mono-gaz, non visqueux, classes ML à toute profondeur) ; KH
  4 niveaux WENO tourne en live. Leçon : la combinaison RK
  a·u⁰+b·(q+dtL) ne télescope pas aussi proprement en fp32 que
  l'update incrémental MUSCL — dérive de masse ~1e-5 sur domaine
  fermé (vs ~1e-8), c'est le plancher de la formulation, pas une
  fuite (le reflux pondéré est conservatif).
- [x] ~~LLF → faces HLLC~~ — l'utilisateur a vu juste : le splitting
  Lax-Friedrichs dissipe ∝ |u|+c sur TOUTES les ondes, y compris le
  cisaillement (∝ |u|) que HLLC résout quasi exactement — le KH WENO
  était VISIBLEMENT plus diffusé que MUSCL malgré l'ordre formel.
  Remède : reconstruction WENO5 des états primitifs de face + HLLC.
  Re-mesures : Sod bat désormais MUSCL (1.29e-3 vs 1.46e-3), vortex
  6× moins dissipé, KH +14-19 %% d'enstrophie retenue en phase
  d'enroulement. Leçons : (a) l'ordre 2D mesuré (~2.2) révèle la
  quadrature de face dim-par-dim que le LLF masquait — la valeur est
  la CONSTANTE ; (b) e64 de l'onde d'entropie = plancher temporel RK3
  (identique entre variantes à 5 chiffres) — chaque mesure d'ordre
  doit savoir quel plafond elle touche.
- **Sortie v1.3 : ATTEINTE** — vortex à l'ordre ≥3 et 6× moins
  dissipé que MUSCL, interfaces KH/RM visiblement plus fines à
  résolution égale ; WENO5 + HLLC du fichier de cas (`scheme = weno5`)
  jusqu'au GPU, en lock-step CPU.

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
- **KH inviscide raide = mal posé** (σ ∝ k, aucune coupure) : la
  structure fine des enroulements est du bruit de troncature amplifié,
  donc SCHÉMA-DÉPENDANTE. MUSCL et WENO5-HLLC divergent complètement en
  petits billows (motifs différents, pas seulement +14-19 %%
  d'enstrophie sur le rouleau primaire), et raffiner donne PLUS
  d'enroulements, jamais une convergence. Pour comparer deux schémas il
  faut une coupure physique — `mu > 0` ou une couche de cisaillement
  tanh d'épaisseur résolue (cf. Lecoanet 2016) — alors les deux
  convergent vers le même champ, WENO5 à plus basse résolution. Le
  splitting de flux (LLF) aggrave : il dissipe ∝ |u|+c sur TOUTES les
  ondes, donc lisse contacts et cisaillement que HLLC garde nets.

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

Ajout hors-jalon — **murs no-slip** : BC déclarative `noslip` (miroir
avec les deux composantes de quantité de mouvement inversées, paroi
adiabatique), validée par la **couche limite de Blasius** sur plaque
plane (driver `blasius` + `cases/blasius.ini`) : profil RMS 1.4 %,
δ99 −2 %, Cf +7 % vs théorie à Re_x ≈ 2700. Leçon : un Blasius propre
en compressible exige un flot libre épinglé en haut (ZPG) et un domaine
haut (~14 δ99) — un haut transmissif laisse le déplacement de la couche
accélérer le flot (Ue +16 %) et amincir la BL ; il faut aussi être à
Re_x assez élevé (δ ≪ x) pour le régime asymptotique. Un vrai
non-réfléchissant caractéristique reste en backlog.

## Jalons

### Fil rouge « outil industriel » *(transverse, un peu à chaque jalon)*
- [x] ~~Guide utilisateur (poser un cas en 10 minutes, lire le journal,
  exploiter les sorties)~~ — fait : `docs/GUIDE.md` (tutoriel pas-à-pas,
  exemple choc/bulle validé par `--check`). Doc d'architecture du code
  séparée dans `docs/ARCHITECTURE.md` (schémas Mermaid).
- [~] Post-traitement fourni : amorcé — `tools/plot_convergence.py`
  (courbes d'ordre), `tools/plot_benchmark.py` (apport GPU vs taille), et
  `tools/schlieren_video.py` : tracé des champs en **schlieren numérique**
  (|∇ρ|) depuis les `.vthb`, composé du niveau AMR le plus fin sans
  ParaView (compositing numpy bilinéaire cellule→point, ~1 s/frame, anti-
  escalier), palettes fond noir/blanc, caméra de suivi auto (DMR),
  surcouche pédagogique annotée (style LaTeX) et export MP4 via ffmpeg.
  Sortie frame-par-pas côté solveur via `[output] every = K`. Reste : le
  tracé du journal. Sans dépendre de ParaView pour le quotidien.
- [~] Messages d'erreur systématiquement actionnables (fichier:ligne,
  correction suggérée) — amorcé : saturation du pool de slots GPU
  (AmrGpuML) désormais signalée par une erreur claire (nombre de patchs,
  KB/patch, MB du pool, et les leviers : `amr.levels`, `tag_threshold`,
  `amr.max_patches`) au lieu d'un abort par `assert` désactivé en
  `-DNDEBUG`. Cap du pool configurable (`amr.max_patches`) et par défaut
  dimensionné sur la mémoire du device (~1/8 du working set). Reste :
  généraliser ce style à toutes les portes. Sondes ponctuelles ;
  checkpoint multi-niveaux (reprendre n'importe quel calcul).
- [x] ~~**Vérification par solutions manufacturées (MMS)**~~ — fait.
  L'étude de convergence (`convergence`) couvrait l'ordre EULER lisse
  (onde d'entropie, vortex isentropique) ; aucune porte ne vérifiait
  l'ordre de l'opérateur VISQUEUX. La MMS comble ça : solution
  manufacturée lisse périodique stationnaire (ρ, u, v, p sinusoïdaux),
  terme source S = div(F_Euler) − div(F_visqueux) calculé **en double**
  par différences finies d'ordre 4 (h=1e-3) sur les champs analytiques
  (« exact » à ~1e-12 devant le schéma testé), injecté dans le pas
  mono-grille `step2D(mu)`. On laisse relaxer vers l'état stationnaire et
  on mesure l'ordre de l'erreur L1 (densité) vs h sur N=16→128.
  - [x] driver `mms.cpp` + source manufacturé (FD ordre 4, double)
  - [x] mesure par erreur de solution stationnaire (robuste fp32 —
    la mesure par troncature `(U−U0)/dt` était bruitée par fp32)
  - [x] **opérateur visqueux (mu=0.01), les DEUX schémas : MUSCL 2.10,
    WENO5 1.97** (gate >1.8, PASS) — vérifie Navier-Stokes complet à
    l'ordre 2 ; le flux visqueux central (commun aux deux schémas)
    plafonne l'ordre à 2, comme attendu.
  - [x] **source de gravité** (split, MUSCL) : ordre 2.10 (PASS). La
    densité n'ayant pas de source de gravité, son ordre 2 confirme que le
    couplage gravité (qté de mouvement/énergie → vitesse/pression →
    densité via les flux) est consistant (un bug de signe ou de terme de
    travail casserait l'état stationnaire et l'ordre).
  - [x] inviscide stationnaire (mu=0) *informatif* : ~1 pour les deux
    schémas (MUSCL 1.08, WENO5 1.01). Sans viscosité physique, l'erreur
    de l'état stationnaire est fixée par la **viscosité numérique** des
    flux de face (1er ordre ici) — ce n'est PAS l'ordre transitoire de
    conception (MUSCL ~2, WENO5 ~4-5), lui vérifié par `convergence`
    (advection d'une solution exacte). Non gaté.
  - [x] **source de réaction** : pas de MMS — déjà vérifié plus
    rigoureusement par `reactor` (fonction `react()` vs solutions
    analytiques exactes : isotherme err 8e-8, équilibre adiabatique) et le
    couplage Strang réaction↔hydro par `detonation` (CJ à 0.8 %).
  - [x] driver `mms` ajouté à la suite CPU de la CI (Euler+visqueux ×2
    schémas + gravité)

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
- [x] ~~WENO5 visqueux (Navier-Stokes)~~ — fait : le flux de Stokes +
  Fourier (différences centrées 2ᵉ ordre, opérateur factorisé partagé
  avec MUSCL) s'ajoute aux flux de face HLLC à chaque étage RK, CPU +
  Metal + AMR. `scheme = weno5` accepte désormais `mu > 0`. Gates :
  couche de cisaillement erf vs solution exacte à l'ordre 2.15
  (`weno_suite`), parité CPU/GPU 4.4e-4 sur hiérarchie 2 niveaux
  raffinée (`mlgpu_amr` gate 5) ; KH visqueux 3 niveaux tourne en
  bout-en-bout. Note : avec u=0/v lisse l'opérateur visqueux est
  identique à MUSCL, donc l'écart résiduel (~1.5×) est le seul
  constant temporel RK3-vs-Hancock — la preuve de correction est
  l'ordre 2 vers l'exact.
- [x] ~~WENO5 bi-gaz (croise v1.2 + v1.3)~~ — fait : reconstruction
  WENO5 des états de face (ρ,u,v,p) + Y + Γ, HLLC γ-par-côté, flux
  d'espèce upwindé, Γ quasi-conservatif sur la vitesse de contact ;
  méthode des lignes donc pas de terme demi-pas (face-p et face-Γ
  synchrones par construction). CPU (cœur + AMR) et GPU (kernels Metal
  + pool). `scheme = weno5` accepte désormais `[species]`. Gates :
  Sod bi-γ uniforme L1 1.49e-3, AMR 3 niveaux L1 4.09e-3 (entre MUSCL
  3.1e-3 et WENO mono-gaz 4.27e-3), GPU en lock-step CPU complet
  (L1 = CPU à 4 chiffres, masse d'espèce 4.4e-6). Bug attrapé : le
  wave kernel manquait dans `enableWenoSpecies` (crash de
  `gpu.maxStableDtAll`), invisible car la gate pilotait le `dt` par le
  CPU → la gate exerce désormais aussi la réduction GPU.
- **Sortie v1.3 : ATTEINTE** — vortex à l'ordre ≥3 et 6× moins
  dissipé que MUSCL, interfaces KH/RM visiblement plus fines à
  résolution égale ; WENO5 + HLLC (Euler, Navier-Stokes ET bi-gaz) du
  fichier de cas (`scheme = weno5`) jusqu'au GPU, en lock-step CPU.

### v1.4 — La troisième dimension *(démo + pédago)*
Le grand chantier, mené comme le multi-niveaux : CPU de référence →
portes de validation → GPU.
- [ ] Extension 3D du cœur (Grid, schéma, AMR, pool, CaseDef, rendu).
- **Sortie** : un cas 3D AMR ~100M cellules effectives sur le M4,
  visualisé en temps réel.

### v1.5 — Écoulements réactifs (combustion) *(labo + pédago)*
Extension naturelle du multi-espèces (v1.2) : on ajoute un terme source
de réaction + dégagement de chaleur. Reste strictement 2D (orthogonal à
la dimension). Modèle volontairement simple — **réaction à une étape
d'Arrhenius** + variable d'avancement λ + chaleur de réaction q (« Euler
réactif ») ; PAS de chimie détaillée multi-espèces (hors-périmètre, type
CHEMKIN). Le terme source réutilise le précédent de la gravité (split
source) et l'EOS à Γ variable.
- [x] ~~Intégrateur de réaction raide (ODE par cellule)~~ — fait :
  `Reaction.hpp` (Arrhenius une étape) + `react()` RK4 sous-cyclé
  adaptatif. Énergie asservie à l'avancement (e = e0 + q·Δλ exact),
  donc conservatif par construction et sans incohérence de clamp.
  Driver `reactor` (réacteur 0D), gates : isotherme exact (err 8e-8),
  équilibre adiabatique (T = T0+(γ-1)q exact, énergie à 5e-7), raide
  (A=1e4, dt grossier → borné et convergé : le sous-cyclage gère la
  raideur). Reste : splitting de Strang pour COUPLER au flot (le risque
  de raideur, lui, est levé).
- [x] ~~Splitting de Strang + couplage chaleur/λ (1D)~~ — fait :
  R(dt/2)·A(dt)·R(dt/2), A = `step2DY` réutilisé avec γ1=γ2 (λ porté
  par φ=ρλ, Γ constant), R = `reactGrid` (réaction à volume constant
  par cellule : chaleur dans E, λ avancé). Driver `detonation`.
- [x] ~~Détonation de Chapman-Jouguet 1D~~ — fait : D_CJ exact résolu
  numériquement (Rankine-Hugoniot avec chaleur + tangence CJ ; limite
  forte → √(2(γ²−1)q) vérifiée). Tube fermé (paroi réfléchissante,
  allumage chaud) : la détonation surconduite **relaxe vers CJ** par la
  détente de Taylor — vitesse mesurée +5.6 %% → +2.5 %% → +1.4 %% en
  fenêtres successives, **+1.3 %% établie** vs D_CJ=4.68. Leçons : (a)
  bord transmissif = réservoir infini → surconduite permanente, il faut
  une paroi réfléchissante ; (b) Ea trop grand / A trop petit → la zone
  de réaction se découple du choc (échec de détonabilité) — il faut une
  réaction assez rapide et peu raide en T (Ea=8, zone ~10 mailles).
- [x] ~~AMR (CPU) : réaction dans la hiérarchie multi-niveaux~~ — fait :
  `reactLevel_` encadre le pas hyperbolique de chaque niveau (Strang
  par niveau, source locale donc sans interaction avec reflux/restrict),
  `cfg.react` (implique species, γ1=γ2). Gate : détonation CJ sur AMR
  3 niveaux, le raffinement (tagué sur le saut de densité) suit le front
  mobile, D_CJ préservé à **+0.8 %%** (vs +1.3 %% uniforme).
- [x] ~~GPU : réaction sur le GPU (kernel source par cellule)~~ — fait :
  kernel Metal `react`/`react_pool` (RK4 sous-cyclé adaptatif par
  thread, énergie asservie à λ), `Euler2DGpu::encodeReact` + réaction
  par niveau dans AmrGpuML (Strang autour du pas hyperbolique).
  Gate : détonation CJ sur AMR GPU 3 niveaux = **+0.8 %% (identique au
  CPU au chiffre près)** — lock-step parfait.
- [x] ~~`[reaction]` déclaratif + cellule de détonation 2D~~ — fait :
  section `[reaction]` (A, Ea, q, Tign, gamma) dans CaseDef ; la
  progression λ réutilise le scalaire d'espèce (`gas = 2` = zone
  allumée, λ=1). `cases/detonation.ini` : détonation 2D en canal (tube
  fermé, perturbation transverse, raffinée par AMR) qui tourne en live
  via `run`. La physique CJ est verrouillée par le driver `detonation`
  (0.8 %% sur AMR CPU et GPU).
- **Validation par étapes** (méthodo no-slip) : réacteur 0D ✓
  (isotherme exact, équilibre adiabatique, raideur) → détonation CJ 1D ✓
  (D_CJ à 1.3 %% uniforme, +0.8 %% sur AMR 3 niveaux) → reste la
  cellule de détonation 2D.
- [x] ~~Déflagration (chemin réactif visqueux)~~ — fait : `step2DY`
  porte le flux de Stokes+Fourier (mu > 0, opérateur partagé) + dt
  visqueux + `species + mu` autorisé. La flamme se propage par
  CONDUCTION (driver `deflagration` — étude manuelle, hors CI car
  dt ~ dx²/ν sur une longue propagation ; + `cases/deflagration.ini`) :
  subsonique (Mach 0.17), 3× plus rapide qu'avec la seule diffusion
  numérique (mu=0). q petit pour que la compression seule ne puisse pas
  allumer (T_brûlé > Tign de peu) → c'est la conduction qui mène.
  Leçon : T_brûlé = 1+(γ-1)q doit dépasser Tign sinon la flamme meurt ;
  le √(α) de Zeldovich est fragile (plancher de diffusion numérique à
  mu=0, quench à fort mu en marge mince) — gaté en « conduction-driven »
  plutôt qu'en scaling strict.
- [x] ~~Branche visqueuse des kernels species GPU~~ — fait : flux de
  Stokes+Fourier dans `flux_x_y`/`flux_y_y` (+ pool), `q` lié aux
  kernels. La déflagration tourne sur GPU (lock-step CPU à 1 %%) ;
  `cases/deflagration.ini` passe en backend hybride + live.
- **Sortie v1.5 : ATTEINTE** — détonation de Chapman-Jouguet à la
  vitesse théorique (uniforme +1.3 %%, AMR CPU/GPU +0.8 %% en
  lock-step), cellule de détonation 2D pilotée par fichier de cas et
  raffinée par AMR, en live. Reste optionnel : chimie multi-étapes
  (hors-périmètre), structure cellulaire quantitative.
- Effort : cœur CPU + gate CJ ~1-2 sessions ; intégration GPU/AMR
  complète ≈ taille du jalon multi-espèces.

### v1.6 — Corps immergés *(labo + démo)*
Des géométries dans l'écoulement, sans mailler le solide : un **masque
solide** sur la grille cartésienne (méthode ghost-cell / paroi
réfléchissante — pas de cellules coupées, donc pas de problème de petite
cellule). Les faces fluide↔solide reçoivent un flux de paroi
réfléchissante (vitesse normale mirroir, glissement). Première brique
posée :
- [x] **`step2D` masque-aware** — paramètre optionnel `const uint8_t*
  solid` (défaut `nullptr` → chemin inchangé, régressions
  `convergence`/`sod1d`/`mms` intactes) : cellules solides ignorées
  (prédicteur/update/gravité), pentes reconstruites en mirroir au contact
  d'un solide, flux de face fluide↔solide = paroi réfléchissante
  (`mx→−mx` / `my→−my`). MVP **non-visqueux** (no-slip avec `mu>0`
  reporté ; les flux visqueux ignorent encore le masque).
- [x] **gate `immersed`** — réflexion d'un choc plan sur une paroi
  immergée alignée, deux régimes : Ms=2 (post-choc subsonique) → **14.95
  vs 15.0 exact (0.33 %)** ; Ms=3 (post-choc **supersonique** vers la
  paroi, M1≈1.36) → **51.68 vs 51.67 (0.02 %)**. Face alignée ⇒
  vérification exacte. Ajouté à la suite CPU de la CI.
- [x] **flux de paroi exact** (`wallPressure`/`wallFluxX/Y` dans `Hllc`) —
  la paroi miroir + HLLC **fuit en supersonique** (l'estimation PVRS garde
  `SL = uL − cL·q > 0` et décentre tout le flux entrant : un corps
  supersonique devenait ~transparent). Remplacé par le flux de pression de
  paroi exact `(0, p*, 0, 0)`, `p*` résolu par Newton sur `f_W(p*) = u_n`
  (Toro) — correct en sub- ET supersonique. C'est ce qui débloque les
  arcs de choc (sans lui, l'arc se formait puis se vidait par la paroi).
- [x] **démos visuelles** — `cases/wc_step.ini` (marche Mach 3 de
  Woodward & Colella 1984 : arc de choc, réflexion de Mach, point triple,
  ligne de glissement, détente de coin) et `cases/cylinder_bowshock.ini`
  (cylindre Mach 2 : arc détaché en escalier, détente aux épaules,
  sillage). ρ de stagnation conforme (step 6.27, cylindre 4.36).
- [x] **masque déclaratif** dans le fichier de cas — section `[solid]`
  `region.N = rect|circle|halfplane|band|sinex …` (même grammaire
  géométrique que `[ic]`, sans état ni mouvement ; `CaseDef::solidAt`).
  Threadé dans le pas MUSCL de la grille de base d'`Amr2` ; garde-fou
  dans `run.cpp` (backend cpu, muscl mono-gaz, raffinement désactivé →
  grille de base seule). Gate `immersed_case` : `cases/shock_wall.ini`
  rejoue la réflexion de choc par le chemin déclaratif → **14.95 vs 15.0
  exact (0.33 %)**, ajouté à la suite CPU de la CI.
- [x] **intégration AMR** (`Amr2`, 2 niveaux) — masque par patch
  (`buildPatchSolid_`), pas de patch masque-aware, **restriction** sur les
  seules cellules filles fluides (cellules grossières solides figées),
  **refluxing** qui ne corrige jamais une cellule solide, **prolongation**
  en escalier constant au contact d'un solide, et **tagging du bord** du
  corps (cellules fluides au contact → raffinement de l'escalier). Tout est
  no-op sans `[solid]` (gates non-solides `dmr_amr`/`casedef_test`/`ml_amr`
  intacts). Gate `immersed_amr` : réflexion de choc raffinée (4 patches),
  pression de paroi **14.98 / 15.00 subcyclé** vs 15.0 exact (0.14 / 0.03 %)
  et cohérente avec la grille de base (0.19 %) ; teste single-rate ET
  subcyclé. Démos `cylinder_bowshock`/`wc_step` raffinées (bord + chocs).
- [x] **portage GPU** (`AmrGpu` hybride, lock-step) — masque solide dans
  les kernels Metal (`predictor`/`flux_x`/`flux_y`/`update` + variantes
  `_pool` : cellules solides figées, pentes en miroir, **flux de paroi
  exact `wallPressure` en MSL**), masque par slot (`smaskP_`) + masque
  coarse (`Euler2DGpu`), et chaîne AMR masque-aware CPU (restriction,
  reflux, prolongation, tagging) identique à `Amr2`. `AmrGpuML` reçoit un
  masque-zéro (kernels inviscides partagés). Gate `immersed_gpu` :
  lock-step CPU↔GPU sur un cylindre Mach 2 (single **5.9e-4** + subcyclé
  **1.1e-3** vs gate 1e-2, patches identiques). Démos en `backend=hybrid`
  (~5× : cylindre 56→10 s, marche 116→23 s). Non-solides intacts
  (`dmr_gpu` 9.8e-6, `mlgpu_amr`).
- [x] **validation chiffrée — choc oblique θ-β-M** : dièdre immergé
  (rampe 15°, Mach 2.5) → angle de choc mesuré **β = 38.3° vs 36.9° exact**
  (relation θ-β-M, biais d'escalier 1.4° qui décroît avec la résolution :
  6.1°→1.7°→1.4°). Gate `immersed_wedge` (CI CPU) + démo `cases/wedge.ini`
  (réflexion sur paroi haute, type entrée d'air). Reste l'arc détaché du
  cylindre vs corrélation de Billig (qualitatif pour l'instant).
- [ ] multi-niveaux solide (`AmrML`/`AmrGpuML`, profondeur > 2).
- [x] **efforts (traînée / portance par ∫ pression de paroi)** —
  `wallForce` (`solver/Forces.hpp`) intègre p·n sur les faces fluide/solide.
  Validé dans `immersed_wedge` : C_p de paroi du dièdre **2.447 vs 2.468
  exact (0.8 %, choc oblique)** et **portance d'un cylindre symétrique = 0**
  (|F_y/F_x| < 1e-3). Frottement visqueux non inclus (traînée de pression).
- [x] **no-slip visqueux** (flux visqueux masque-aware) — un voisin solide
  devient un ghost no-slip (deux vitesses inversées, ρ/p conservés →
  adiabatique) ; le cisaillement de paroi rend le mur adhérent (le flux
  convectif fournit la pression). Gate `immersed_noslip` : couche limite de
  Blasius sur une **plaque immergée** → profil **RMS 0.7 %** vs f', Cf
  **3 %** vs 0.664/√Re_x. **Porté sur GPU** (flux visqueux masque-aware dans
  les kernels Metal) : lock-step CPU↔GPU visqueux vérifié (`immersed_gpu`,
  écart 4e-4) — `mu > 0` + solide tourne en `hybrid`.
- [x] **multi-niveaux solide** (profondeur > 2) — masque threadé dans
  `AmrML` (CPU, profondeur arbitraire) : par-patch + base, restriction
  (filles fluides), refluxing (cellules solides épargnées), prolongation
  (escalier constant), tagging de bord — à chaque niveau, via une requête
  géométrique `solidAt(position)` (le parent d'un niveau peut être un patch,
  pas la base). Gate `immersed_amr` étendu : réflexion de choc à **3
  niveaux** → paroi 15.03 vs 15.0 exact (0.21 %, 16 patches). No-op sans
  `[solid]` (ml_amr / species / weno intacts). **Porté sur GPU**
  (`AmrGpuML`, masque par slot du pool + masque base, requêtes géométriques
  pour la plomberie) : lock-step CPU↔GPU à **3 niveaux** vérifié
  (`immersed_gpu`, écart 7.5e-4). Les solides tournent désormais en `hybrid`
  à profondeur arbitraire.
- [ ] cut-cells (supprimer l'escalier) — le seul item restant du facet.

## Backlog (tiré dans un jalon quand il sert, jamais en direct)

Mode stationnaire (local time stepping — donnerait tout leur sens aux
résidus du journal **et débloquerait les régimes internes de tuyère** ;
cf. leçon tuyère) ; **outflow/inflow non-réfléchissant caractéristique
(NSCBC)** — réclamé par le Blasius (haut ZPG) ET la tuyère/jet (cf.
leçons) ; tagging de Richardson ;
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
- **Le shader de rendu lit la même mémoire que le solveur** : le passage
  NG 2→3 (v1.3) a été vérifié bit-identique sur le chemin DONNÉES
  (sorties headless), mais le fragment shader du live codait l'offset
  ghost en dur (`+2`, largeur `nx0+4`) → tous les cas affichés brouillés
  (quel que soit le schéma) alors que les gates passaient. Offset ghost
  désormais passé en uniform (`ng`). Leçon : une vérif « bit-identique »
  doit couvrir TOUS les consommateurs de la mémoire, pas seulement le
  solveur — le rendu zéro-copie en est un.
- **Tuyère C-D : régimes & convergence stationnaire** (BC `backpressure`
  déclarative ajoutée : schedule en escalier `t0 p0 t1 p1 …`). Le balayage
  de contre-pression et les cas par régime ont révélé deux limites
  STRUCTURELLES (pas des bugs), chacune pointant un item de backlog :
  (a) la convergence vers un état STATIONNAIRE subsonique est très lente en
  explicite (le M au col grimpe sur des dizaines d'unités de temps, et le
  premier critique réel — amorçage — est plus bas que la prédiction 1D à
  cause du blocage de couche limite / immersed-boundary) → **mode
  stationnaire (local time stepping / pseudo-temps)** ; (b) en domaine
  ouvert, le jet débouchant dans un ambiant au repos crée une couche de
  mélange + des réflexions à l'outflow (`backpressure` transmissif en
  supersonique) qui empêchent un stationnaire propre et raidissent le pas
  de temps → **outflow non-réfléchissant caractéristique**. En prime : le
  choc droit transsonique est intrinsèquement instationnaire (train de
  chocs λ + décollement), et les régimes sur/sous-détendus ont un jet
  turbulent (KH) jamais figé — l'« établi » utile y est la structure de
  choc PROCHE-sortie, pas un champ gelé. Diagnostic clé : tracer le **champ
  de Mach + ligne sonique M=1**, pas la schlieren (qui sur-réagit au bruit
  AMR et au mélange de sortie — m'a fait conclure à tort à un choc).
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

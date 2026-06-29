# machmallow — roadmap chaîne YouTube

Mettre en valeur le solveur CFD via des vidéos **hero hypnotiques**.
Objectif : maximiser les vues, construire une audience, asseoir l'autorité
technique. Document vivant — on coche et on ajuste selon l'analytics.

## 1. Positionnement

**Niche** : à l'intersection de trois audiences qui se recouvrent peu et
qu'on capte d'un coup — *oddly satisfying* (boucles hypnotiques), *science
populaire* (physique des fluides), *dev/maker* (un solveur GPU codé from
scratch sur un Mac). Personne n'occupe bien ce triangle.

**Promesse de marque** : « la beauté cachée des fluides, simulée à la main
sur un Mac ». Chaque vidéo doit être soit **hypnotique** (on la regarde en
boucle), soit **épatante** (« comment c'est possible ? »), idéalement les
deux.

**Identité visuelle = signature reconnaissable** (déjà trouvée, validée
« vraiment hypnotisante ») :
- fond **noir**, palette **icefire** (cyan ↔ braise, centre sombre) ;
- mouvement organique, chocs en schlieren blanc quand ça sert ;
- un **watermark discret** constant (le nom de la chaîne) ;
- une **typo** unique pour titres/légendes (sans-serif, accents propres).
La cohérence palette = on reconnaît un short machmallow en 0,3 s.

**Nom de chaîne : `machmallow`** (décidé) — unique, doux/tech, cohérent
avec le projet/repo.

**Langue : visuel-first + anglais** (décidé). Les Shorts sont quasi sans
texte (boucle + musique = universel) ; le peu de texte, les titres et le
long-format sont en **anglais** → audience mondiale (~10× le marché FR).
C'est le levier #1 de portée.

## 2. Stratégie de format (le cœur de l'acquisition)

| | Shorts (9:16, ≤ 60 s) | Long-format (16:9, 6-12 min) |
|---|---|---|
| Rôle | **Découverte / acquisition** | **Conversion / rétention / $** |
| Pourquoi | boucle parfaite = rewatch ∞, l'algo pousse | profondeur, autorité, monétisable |
| Cadence | **5+ / semaine** (agressif) | **2 / mois** |
| Contenu | une boucle hypnotique + 1 phrase | récit, explication, build-in-public |

**Règle d'or des Shorts** : le mouvement DÉMARRE à la frame 1 (pas
d'intro), **boucle sans couture** (le spectateur ne voit pas le raccord →
il reste), une seule idée, une légende courte et grosse. Cross-post
systématique TikTok + Instagram Reels (même asset 9:16) = 3× la portée
pour 0 effort.

**Le long-format** capitalise l'audience Shorts : playlists, séries,
chapitres. C'est là que vit le récit « j'ai codé un solveur GPU » et les
explications scientifiques, et c'est ce qui débloque la monétisation.

## 3. Piliers de contenu

**Angle dominant (décidé) : hypnotique d'abord** — les boucles
satisfaisantes sont le moteur d'acquisition ; science et build-in-public
viennent en appui. Croissance la plus rapide.

1. **Boucles hypnotiques** (Shorts) — von Kármán, KH, RT, paires de
   tourbillons… mouvement pur, zéro blabla. Le moteur de vues.
2. **« C'est quoi ce truc ? »** (Shorts + long) — expliquer un phénomène
   en 45 s ou en 8 min (réutilise le format pédago tuyère déjà construit).
3. **Build-in-public** (long) — le solveur lui-même : Metal, AMR, GPU,
   from scratch sur M4. Public dev/maker, très fidélisant.
4. **Études & duels** (long) — balayages de régime, comparaisons, side-by-
   side (ex. le transitoire de tuyère, MUSCL vs WENO, raffinement AMR).

## 4. Backlog de vidéos (mappé sur ce que le solveur sait rendre)

### Shorts — boucles (priorité de production)
- [ ] **Allée de von Kármán** (cylindre, sillage subsonique visqueux) —
  *flagship*, boucle parfaite, icefire.
- [ ] **Kelvin-Helmholtz** — train de volutes (déjà rapide à produire).
- [ ] **Rayleigh-Taylor** — champignons qui descendent.
- [ ] **Arc de choc qui respire** (cylindre/triangle Mach 2) — déjà rendu.
- [ ] **Cellule de détonation** — motif en losanges qui pulse.
- [ ] **Forêt de cylindres** — interférences de sillages.
- [ ] **Bulle d'hélium + choc** (double champignon, densité bi-gaz).
- [ ] **Diamants de choc de tuyère** (le jet sous-détendu, déjà là).

### Shorts — « c'est quoi ? » (1 phénomène, 45 s)
- [ ] Pourquoi un jet de fusée a des **diamants de choc**.
- [ ] Pourquoi les drapeaux **claquent** (KH).
- [ ] Comment naît une **allée de tourbillons**.
- [ ] Qu'est-ce qu'un **disque de Mach**.

### Long-format
- [ ] **« J'ai codé un solveur CFD GPU from scratch sur un Mac »** — le
  récit fondateur (Metal, fp32, AMR, validation). Aimant à abonnés dev.
- [ ] **« Les instabilités fluides les plus hypnotiques »** — compilation
  commentée (recycle tous les Shorts en une pièce de rétention).
- [ ] **« Les régimes d'une tuyère »** — réutilise la vidéo pédago déjà
  construite (intro + arrêts commentés).
- [ ] **« Choc rencontre bulle »** — Richtmyer-Meshkov / Haas-Sturtevant,
  science spectaculaire validée par l'expérience.
- [ ] **AMR expliqué** — pourquoi/comment la grille se raffine toute seule
  (les boîtes de patchs qui suivent les chocs, très visuel).

## 5. Pipeline de production (technique)

Construire un outil `tools/hero.py` (au-dessus de `schlieren_video.py`) qui
sort, depuis un même rendu, les **deux formats prêts à poster** :
- **16:9** 1920×1080 (ou 4K) — long-format / YouTube classique.
- **9:16** 1080×1920 — Shorts / Reels / TikTok (recadrage centré +
  éventuel fond flouté/étendu).
- **boucle** : ping-pong OU vraie période détectée (von Kármán/KH sont
  périodiques → couper sur une période = boucle invisible).
- **brand** : watermark, légende grosse, carte de titre optionnelle.
- **musique** : piste libre de droits synchronisée (hors solveur).

Étapes type pour un hero :
1. choisir le cas (`.ini`) + résolution AMR ;
2. run GPU (hybrid), un job surveillé (protocole anti-crash : un seul gros
   job à la fois, surveiller la mémoire) ;
3. rendu vorticité icefire (+ overlay schlieren si chocs) ;
4. `hero.py` → 16:9 + 9:16 + boucle + brand ;
5. miniature (frame la plus spectaculaire + titre gros).

## 6. Roadmap par phases

### Phase 0 — Identité (avant toute publication)
- [x] Trancher le **nom** (`machmallow`).
- [x] **Avatar + bannière + wordmark** : la mascotte (un chamallow qui se
  prend un arc de choc détaché, icefire sur noir) générée par
  `tools/brand_avatar.py` — dessin vectoriel reproductible, zéro simu.
- [ ] **Intro-sting 1,5 s** (reste à faire).
- [x] Figer la **charte** (palette icefire, typo Space Grotesk, wordmark,
  tagline) dans `BRAND.md` + `tools/brand_avatar.py`.

### Phase 1 — Pipeline
- [ ] `tools/hero.py` : double format + boucle + brand, validé sur un cas.

### Phase 2 — Banque de boucles (le stock de lancement)
- [ ] Produire **15-20 Shorts** d'avance (cadence agressive = il faut un
  matelas) : von Kármán, KH, RT, arc de choc, détonation, bulle He,
  diamants tuyère, forêt de cylindres, paires de tourbillons… On ne lance
  pas à sec, et on garde toujours ~2 semaines d'avance.
- [ ] `hero.py` doit pouvoir **batcher** : une liste de cas → tous les
  Shorts d'un coup (la cadence agressive l'exige).

### Phase 3 — Lancement
- [ ] Publier 3 Shorts d'un coup + 1 long-format fondateur (le « build »).
- [ ] Cross-poster TikTok + Reels.

### Phase 4 — Cadence & itération
- [ ] Tenir **5+ Shorts/sem + 2 longs/mois** (agressif) — soutenable
  uniquement avec la banque d'avance + le batch de `hero.py`.
- [ ] Lire l'analytics (rétention, taux de boucle, source de trafic) et
  doubler ce qui marche ; décliner les Shorts gagnants en séries.

## 7. Tactiques de croissance (2026)

- **Hook < 1 s** : image la plus folle d'emblée, jamais d'écran de titre
  avant le mouvement.
- **Boucle sans couture** : le KPI Shorts est le rewatch — une boucle
  parfaite double le temps de visionnage.
- **Une vidéo = une idée** : pas de fourre-tout.
- **Séries + playlists** : « Instabilités », « Chocs », « Build » →
  binge-watch.
- **Titres curiosité** + miniatures à fort contraste (icefire sur noir
  ressort déjà très bien).
- **Audio tendance** sur les Shorts ; musique ambient/synthwave sur les
  longs (cohérent avec l'esthétique hypnotique).
- **Cross-post** TikTok/Reels/X dès J0.
- **CTA doux** : « la simu complète + le code → lien » (renvoie au repo,
  fidélise le public dev).

## 8. Métriques de succès

- Court terme (1-2 mois) : ≥ 1 Short qui dépasse 10k vues (signal de
  product-market fit du format).
- Moyen terme (6 mois) : cadence tenue, 1k abonnés (seuil monétisation),
  taux de rétention long-format > 40 %.
- Long terme : une série identifiable, une communauté dev autour du repo.

---
*Prochaine action proposée : Phase 1 — construire `tools/hero.py` (double
format + boucle), puis Phase 2 — produire le premier flagship (allée de
von Kármán).*

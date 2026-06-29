# machmallow — identité de marque (chaîne YouTube)

Spec décisive de l'habillage. Objectif : reconnaissable en 0,3 s, cohérent
sur tous les supports (Shorts, longs, miniatures, avatar, bannière).
Voir la stratégie dans `YOUTUBE.md`.

## Palette (le cœur de la marque)

Fond **noir pur** + colormap **icefire** (diverging cyan ↔ braise, centre
sombre) — validée « vraiment hypnotisante ». La constance de la palette EST
la marque.

| Rôle | Hex | Usage |
|------|-----|-------|
| Fond | `#000000` | partout (vidéos, miniatures, cartes) |
| Cyan (froid) | `#29C5E6` | vorticité négative, accents UI |
| Bleu profond | `#1B3A6B` | froid extrême |
| Braise (chaud) | `#F2622E` | vorticité positive |
| Or clair | `#F6C95B` | chaud extrême / highlights |
| Accent | `#9FD3FF` | liserés de légende, NPR, chiffres |
| Texte | `#FFFFFF` / `#E8E8E8` | titres / corps |

Colormap de référence : **`icefire`** (seaborn) pour la vorticité ;
schlieren blanc en overlay pour les chocs. Densité bi-gaz : palette
chaude séparée (bulle He) mais toujours sur fond noir.

## Typographie

- **Titres / wordmark** : sans-serif géométrique moderne — **Space
  Grotesk** (ou Montserrat). À installer ; sinon **DejaVu Sans Bold** en
  repli (déjà utilisé par les rendus, gère les accents).
- **Corps / légendes** : **Inter** (ou DejaVu Sans).
- **Touche dev** (cartes de code, build-in-public) : **JetBrains Mono**.
- Tout en **bas-de-casse** pour le wordmark (`machmallow`) — doux/tech.

## Wordmark & logo

- **Wordmark** : `machmallow` en bas-de-casse, Space Grotesk Medium, blanc.
- **Mark / glyphe** : une **spirale de vorticité** stylisée (un tourbillon
  icefire) — sert d'avatar et de favicon. On le DÉRIVE d'un vrai rendu
  (crop carré d'un tourbillon icefire) → zéro outil de design, 100 %
  cohérent avec le contenu.
- **Watermark vidéo** : `machmallow` en bas-droite, blanc, opacité ~0,6,
  halo noir léger, petit (présent mais discret) — déjà le gabarit utilisé
  dans `nozzle_pedago`.

## Avatar (cercle, lisible en tout petit)

Crop **carré** d'un tourbillon icefire bien net et symétrique sur fond
noir (un seul motif fort, pas de détail fin qui disparaît à 48 px).
Candidats : volute de DMR, tourbillon de sillage (cylindre), billow KH.

## Bannière (2560×1440, zone sûre centrale 1546×423)

Filets de courant **épais** balayant toute la largeur jusqu'à l'arc de
choc ; **wordmark** `machmallow` (décalé à gauche du centre) + tagline :
**"a hybrid CPU/GPU AMR CFD solver, coded on a Mac"** ; mascotte + choc à
droite. **Tout le texte ET le chamallow tiennent dans la zone sûre
1546×423** (garantis visibles sur mobile / ordinateur / TV).

## Carte de titre / intro-sting (1,5 s)

Noir → un **tourbillon icefire éclot** rapidement (0,8 s) → le wordmark
`machmallow` apparaît en fondu (0,5 s) → coupe sèche sur le contenu.
Produit en rendant une formation de tourbillon + overlay texte (réutilise
le pipeline). Court : ne PAS manger la rétention des Shorts (souvent on
l'omet sur les Shorts, on le garde pour les longs).

## Gabarit de miniature (Shorts 9:16 & longs 16:9)

- La frame la **plus spectaculaire** en plein cadre (icefire sur noir
  ressort déjà fort).
- Titre **3-5 mots** en gros (Space Grotesk Bold), blanc, gradient noir
  léger derrière pour la lisibilité, en haut OU en bas (pas au centre).
- Petit wordmark/mark en coin.
- Fort contraste, zéro surcharge — une idée par miniature.

## Règles d'or

1. Toujours **fond noir** + **icefire**. Jamais de fond clair sur la chaîne.
2. Le **mouvement avant le texte** (Shorts) ; le texte sert la lisibilité,
   pas la déco.
3. Watermark constant, discret, même position partout.
4. Une palette, deux polices max par visuel.

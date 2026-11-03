# 3 November 2026 — local museum demo

Readability cleanup: separate upload, prediction and layout functions; explicit model
unpacking; simpler CSS selectors; repaired French accents in project instructions.

Visual refresh: navy school banner, royal-blue controls and scores, pale-blue background,
white panels and responsive typography, inspired by Albert School and Mines Paris–PSL.
Styling is separated into `src/demo.css` and `.streamlit/config.toml`, both backed up.

Added `python main.py demo`: a localhost Streamlit interface for image upload,
top-three style scores, curated movement descriptions and a downloadable JSON result.
It reuses the existing checkpoint loader and preprocessing; no model architecture,
weights or training settings were changed. The demo handles camera orientation,
invalid uploads and missing weights, and labels collection examples explicitly.
See [DEMO.md](DEMO.md) for operation and limitations. The backup now includes the
curated descriptions in addition to source code and run artifacts.

# Organisation actuelle ? 30 octobre 2026

Deuxi?me rangement : code dans src/, un point d'entr?e main.py, configurations
regroup?es par mod?le, un README par mod?le comparant les versions, tests par
cat?gorie et r?sultats regroup?s par mod?le/version. Poids et mesures conserv?s.
La carte des anciens chemins est dans PATHS.json. Les instructions plus anciennes
ci-dessous d?crivent les ?tapes pr?c?dentes ; le README racine est la r?f?rence actuelle.

---

# Historique des modèles et de l'organisation

## 30 octobre 2026 — versions améliorées et comparaison statistique

Nouvelles options de configuration (valeurs par défaut identiques aux anciennes versions, qui ne changent
pas) : augmentation sur le train seulement, AdamW, scheduler cosinus, label smoothing, choix du checkpoint sur
le macro F1 de validation avec arrêt précoce, BatchNorm1d pour les MLP, features calculées à la main pour les
modèles scikit-learn. Nouvelle commande `python main.py compare` (test de McNemar exact sur les mêmes images
de test).

| Nouvelle version | Référence | Test (référence → nouvelle) | McNemar |
| --- | --- | --- | --- |
| M0 v2_features (`m0_features`) | M0 v1 | 37,78 % → **50,67 %** | p = 0,004, significatif |
| M1 v2_tuned (`m1_tuned`) | M1 v1 | 30,22 % → 37,33 % | p = 0,081, non significatif |
| M2 v3_regularised (`m2_regularised`) | M2 v2 | 41,78 % → 36,44 % | p = 0,16, non significatif |
| M3 v3_augmented (`m3_augmented`) | M3 v2 | 83,11 % → 79,11 % | p = 0,12, non significatif |
| M3 v4_frozen_augmented (`m3_frozen_augmented`) | M3 v1 | 75,11 % → 70,67 % | p = 0,053, non significatif |

Nouvelle commande `python main.py robustness` : sur les 225 images de test dégradées comme une photo de
visiteur (inclinaison, vue oblique, cadrage, lumière, flou, compression cumulés), M3 v3_augmented obtient
38,7 % contre 30,7 % pour v2 (p = 0,001, significatif) : l'augmentation ne gagne rien sur des scans propres
mais rend le modèle plus robuste aux conditions de la borne.

Au passage : M3 v1 gelé contre M3 v2 fine-tuné, 75,11 % → 83,11 %, p = 0,006 (gain du fine-tuning significatif).
Détails et motivations dans les README de `configs/M0` à `configs/M3`.

## 30 octobre 2026 — rangement sans réentraînement

- Tests déplacés dans `tests/`, documents dans `docs/`, utilitaires dans `tools/`.
- Versions regroupées dans `model_versions/<famille>/<version>/` : réglages,
  lancement et fiche expliquant les différences.
- Architecture, identifiants, formats de checkpoints et dossiers de runs conservés.
- Les nouveaux runs enregistrent automatiquement un README avec réglages et résultats.
- Les anciennes commandes `python train_m*.py` sont remplacées par les modules
  documentés dans le README racine. `python main.py train --model <id>` reste valable.
- La documentation ajoutée rétrospectivement ne signifie pas que le code actuel a
  produit chaque ancien fichier. Les mesures historiques restent inchangées.

## 30 octobre 2026 — M2 v2 ReLU

Même profondeur et largeurs que v1. Adam lr 0,01 → 0,0001 ; sigmoid → ReLU ;
dropout 0,3 → 0 ; standardisation par canal sur le train uniquement.
Motivation : saturation, gradients presque nuls et prédiction constante Cubisme.
Choix fondés sur les ablations de validation du groupe.

Run `m2_fixed_1793368290720728000` : 41,78 % test contre 20 % pour v1, cinq classes
prédites. Checkpoint époque 6 par perte de validation. Surapprentissage ensuite.
Voir la [fiche M2 v2](../configs/M2/README.md).

## Versions de référence conservées

- M0 v1 : régression logistique, C choisi sur validation ; 37,78 % test (run du groupe).
- M1 v1 : une couche ReLU, 64 px, Adam 0,001 ; 30,22 % test.
- M2 v1 : cinq sigmoïdes, Adam 0,01, dropout 0,3 ; 20 % test, tout Cubisme.
- M3 v1 : ResNet gelé, seule la tête apprend ; 75,11 % test.
- M3 v2 : tous les paramètres de ResNet ajustés, lr 0,0001 ; 83,11 % test.

Les anciens poids dans `models/` précèdent les splits corrigés et restent des archives.

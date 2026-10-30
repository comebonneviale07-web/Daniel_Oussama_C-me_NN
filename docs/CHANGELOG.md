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
| M3 v3_augmented (`m3_augmented`) | M3 v2 | en cours | |
| M3 v4_frozen_augmented (`m3_frozen_augmented`) | M3 v1 | en cours | |

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

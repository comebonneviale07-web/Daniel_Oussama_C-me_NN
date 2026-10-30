# Organisation actuelle ? 30 octobre 2026

Deuxi?me rangement : code dans src/, un point d'entr?e main.py, configurations
regroup?es par mod?le, un README par mod?le comparant les versions, tests par
cat?gorie et r?sultats regroup?s par mod?le/version. Poids et mesures conserv?s.
La carte des anciens chemins est dans PATHS.json. Les instructions plus anciennes
ci-dessous d?crivent les ?tapes pr?c?dentes ; le README racine est la r?f?rence actuelle.

---

# Historique des modèles et de l'organisation

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

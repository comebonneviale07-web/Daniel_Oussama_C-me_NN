# Poids historiques

Poids des anciens entraînements de l'équipe, avant le split partagé corrigé.
Ce dossier sert d'archive, pas de dossier de code.

Les versions du code sont dans `model_versions/`. Les checkpoints comparables sont
dans `runs/<identifiant>_<timestamp>/best.pt` (ou `best.joblib` pour M0).
Voir `docs/RESULTS.md` pour les modèles effectivement évalués.

Les trois fichiers `.pth` (environ 196 Mo) ne sont plus suivis par Git depuis le 31 octobre 2026 : trop
volumineux, et non comparables aux runs du split partagé. Ils restent sur les ordinateurs qui les avaient et
sont inclus dans les sauvegardes `python main.py backup` ; les anciens commits les contiennent encore.

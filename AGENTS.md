# Organisation du projet

- Un seul point d'entrée public : `python main.py <commande>`. Code dans `src/`.
- Configurations regroupées dans `configs/M0`, `M1`, `M2`, `M3`. Un JSON par version,
  un README par modèle comparant ses versions. Ne pas recréer de train.py par version.
- Pour toute modification substantielle retenue, ajouter une version et un ID stable
  dans `src/versions.py`. Expliquer réglages, différences, motivation, commande,
  résultats et limites dans le README du modèle et `docs/CHANGELOG.md`.
- Ne pas modifier la signification d'un ID utilisé par des checkpoints historiques.
- Tests par catégorie dans `tests/training`, `tests/evaluation`, `tests/prediction` ;
  lancer `python -m unittest discover -s tests -v` dans la .venv Python 3.13.
- Résultats dans `results/<modèle>/<version>/<run>/`, diagnostics et benchmarks dans
  leurs catégories. Chaque nouveau run reçoit automatiquement son README.
- Préserver les poids et mesures. Les chemins historiques restent documentés dans
  `docs/PATHS.json`. Sauvegardes : `python main.py backup`, dans `archives/backups/`.

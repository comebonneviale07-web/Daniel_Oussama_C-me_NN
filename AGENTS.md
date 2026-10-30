# Organisation du projet

- Un seul point d'entr?e public : `python main.py <commande>`. Code dans `src/`.
- Configurations regroup?es dans `configs/M0`, `M1`, `M2`, `M3`. Un JSON par version,
  un README par mod?le comparant ses versions. Ne pas recr?er de train.py par version.
- Pour toute modification substantielle retenue, ajouter une version et un ID stable
  dans `src/versions.py`. Expliquer r?glages, diff?rences, motivation, commande,
  r?sultats et limites dans le README du mod?le et `docs/CHANGELOG.md`.
- Ne pas modifier la signification d'un ID utilis? par des checkpoints historiques.
- Tests par cat?gorie dans `tests/training`, `tests/evaluation`, `tests/prediction` ;
  lancer `python -m unittest discover -s tests -v` dans la .venv Python 3.13.
- R?sultats dans `results/<mod?le>/<version>/<run>/`, diagnostics et benchmarks dans
  leurs cat?gories. Chaque nouveau run re?oit automatiquement son README.
- Pr?server les poids et mesures. Les chemins historiques restent document?s dans
  `docs/PATHS.json`. Sauvegardes : `python main.py backup`, dans `archives/backups/`.

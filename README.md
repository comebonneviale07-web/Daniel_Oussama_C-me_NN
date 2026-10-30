# Museum AI Guide

Classification de 1 500 peintures en cinq styles, avec Python 3.13 sur CPU.

## Les grandes cat?gories

```text
main.py      une seule commande pour utiliser le projet
src/         tout le code Python
configs/     mod?les M0, M1, M2, M3 ; versions JSON et un README par mod?le
tests/       tests : training / evaluation / prediction
results/     entra?nements class?s par mod?le puis version
docs/        r?sultats, diagnostic et historique
data/        métadonnées et splits partagés (images hors Git : voir data/README.md)
archives/    anciens poids et sauvegardes ZIP
```

Exemple : `configs/M2/` contient `v1_sigmoid.json`, `v2_relu.json` et un README
qui compare les deux versions. Les entra?nements correspondants sont regroup?s
dans `results/M2/v1_sigmoid/` et `results/M2/v2_relu/`.

## Utilisation depuis la racine du d?p?t

```bash
source .venv/Scripts/activate
python main.py train --model m2_fixed
python main.py predict photo.jpg
python main.py benchmark
python main.py compare --run-a <run> --run-b <run>
python main.py robustness --runs <run> <run>
python main.py diagnose --experiment prototype
python main.py backup
python -m unittest discover -s tests -v
```

Les images ne sont pas dans Git : les reconstruire une fois avec la commande de
[data/README.md](data/README.md) (mêmes 1 500 images pour tout le monde).

Pour installer : `py -3.13 -m venv .venv`, activer l'environnement, puis
`python -m pip install -r requirements-lock.txt`. Les options de chaque commande
sont accessibles avec `python main.py train --help`, par exemple.

Modèles : `m0` (régression logistique sur pixels), `m0_features` (régression logistique sur
couleurs, contours et texture), `m1` et `m1_tuned` (MLP une couche), `m2` (sigmoïdes, échec conservé),
`m2_fixed` (ReLU corrigé), `m2_regularised`, `m3_frozen`, `m3_frozen_augmented`, `m3` (fine-tuning) et
`m3_augmented` (fine-tuning + augmentation). Ne pas relancer tous les entraînements pour consulter leurs
résultats. `--lr`, `--epochs`, `--threads`, `--seed` restent disponibles.

## Résultats et recommandation

Sur les 225 images de test (même split pour tous) : le meilleur modèle sans réseau pré-entraîné est
`m0_features` (50,7 %) ; les ResNet-18 atteignent 70,7 à 83,1 % (`m3` : 83,1 %). Pour la borne, `m3_augmented`
(79,1 % sur scans propres, différence non significative avec `m3`) résiste significativement mieux aux photos
de visiteur simulées (38,7 % contre 30,7 %, p = 0,001). Comparer deux runs : `python main.py compare`.
Robustesse : `python main.py robustness`. Tableau complet, tests statistiques et limites :
[docs/RESULTS.md](docs/RESULTS.md).

## O? regarder

- [Catalogue des mod?les](docs/MODELS.md)
- [R?sultats mesur?s](docs/RESULTS.md)
- [Diagnostic](docs/DIAGNOSIS.md)
- [Exp?riences et benchmark](docs/EXPERIMENTS.md)
- [Historique des changements](docs/CHANGELOG.md)
- [Utilitaires et sauvegardes](docs/TOOLS.md)

Chaque mod?le a une fiche dans `configs/M*/README.md`. Chaque nouveau run re?oit
automatiquement un README avec sa version, ses r?glages et ses r?sultats.
Les anciens chemins d?plac?s sont list?s dans `docs/PATHS.json`. Les chemins
absolus des anciennes configurations d?crivent leur environnement d'origine.

Le split partag? est dans `data/splits/style5.json` : 1 050 train / 225 validation /
225 test. La normalisation de M2 corrig? utilise uniquement le train. Le split est
par image, pas par artiste. Les rapports sont dans Git ; les nouveaux checkpoints
et les ZIP restent locaux. Copier les sauvegardes sur un stockage ind?pendant.

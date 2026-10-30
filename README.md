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
data/        images, m?tadonn?es et splits partag?s
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
python main.py diagnose --experiment prototype
python main.py backup
python -m unittest discover -s tests -v
```

Pour installer : `py -3.13 -m venv .venv`, activer l'environnement, puis
`python -m pip install -r requirements-lock.txt`. Les options de chaque commande
sont accessibles avec `python main.py train --help`, par exemple.

Mod?les : `m0`, `m1`, `m2` (sigmoid historique), `m2_fixed` (ReLU corrig?),
`m3_frozen`, `m3` (fine-tuning). Ne pas relancer tous les entra?nements pour
consulter leurs r?sultats. `--lr`, `--epochs`, `--threads`, `--seed` restent disponibles.

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

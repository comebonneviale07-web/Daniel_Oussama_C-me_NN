# Mod?les

| Famille | Fiche et configurations | IDs |
| --- | --- | --- |
| M0 | [R?gression logistique](../configs/M0/README.md) | `m0` |
| M1 | [MLP simple](../configs/M1/README.md) | `m1` |
| M2 | [MLP profond : sigmoid et ReLU](../configs/M2/README.md) | `m2`, `m2_fixed` |
| M3 | [ResNet : gel? et fine-tuning](../configs/M3/README.md) | `m3_frozen`, `m3` |

Une fiche par mod?le, un JSON par version. Lancer avec
`python main.py train --model <id>`. Le code commun est dans `src/models.py`,
le registre dans `src/versions.py`, la boucle d'entra?nement dans `src/training.py`.
Les IDs anciens restent stables pour charger les checkpoints historiques.

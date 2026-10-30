# Donn?es et sauvegardes

Depuis la racine du d?p?t, avec la .venv activ?e :

```bash
python main.py dataset --label style --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art --per-class 300 --size 128 --out data/wikiart_style5
python main.py backup
```

Le dataset et ses restrictions sont d?crits dans `src/dataset.py`.
La sauvegarde ZIP v?rifi?e contient code, configurations, tests, documentation,
splits et tous les r?sultats (poids r?cents inclus). Elle est cr??e dans
`archives/backups/`. Copier ZIP et checksum vers un stockage ind?pendant.
Le dataset, la .venv et les anciens poids de `archives/legacy_models/` sont exclus.
`python main.py document-runs` documente les runs anciens sans toucher ? leurs mesures.

# Données

Les 1 500 images WikiArt (`data/wikiart_style5/<style>/*.jpg`) ne sont **pas dans Git** : leur licence est
inconnue et le script du cours demande de ne pas les publier. Git ne contient que :

- `wikiart_style5/metadata.csv` : une ligne par image (chemin, style, genre, artiste…) ;
- `wikiart_style5/subset_info.json` : la commande exacte, la révision du dataset et les shards utilisés ;
- `splits/style5.json` : le split partagé (1 050 train / 225 validation / 225 test) et l'empreinte des chemins.

## Reconstruire les images

Depuis la racine du dépôt (environ 4,2 Go téléchargés, 9 shards ; quelques minutes sur Colab, 10 à 40 min
sur une connexion domestique) :

```bash
python main.py dataset --label style --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art --per-class 300 --size 128 --out data/wikiart_style5
```

La révision du dataset est figée et la graine du plan est fixe : la commande redonne **exactement les mêmes
1 500 images**, avec les mêmes noms de fichiers (vérifié : plan identique à `metadata.csv`, mêmes shards que
`subset_info.json`). L'empreinte enregistrée dans `splits/style5.json` reste donc valide ; si une image
manquait ou différait, `python main.py train` s'arrêterait avec « Dataset changed ».

Ajouter `--dry-run` pour voir le plan sans rien télécharger. Une commande interrompue reprend où elle s'était
arrêtée. Au sein du groupe, partager plutôt un ZIP de `data/wikiart_style5/` sur un Drive privé.

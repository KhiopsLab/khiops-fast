# khiops-fast

Interface scikit-learn de [Khiops](https://khiops.org) dont l'objectif est d'utiliser
[khiops-kni](https://pypi.org/project/khiops-kni/) pour `predict`, `predict_proba` et `transform`,
au lieu de lancer le runner Khiops à chaque déploiement.

| khiops.sklearn | khiops_fast |
|---|---|
| `KhiopsClassifier` | `KhiopsFastClassifier` |
| `KhiopsRegressor` | `KhiopsFastRegressor` |
| `KhiopsEncoder` | `KhiopsFastEncoder` |

## État

**Étape 1 (fait)** : copie de `khiops.sklearn` (khiops 11.0.1.0) dans `src/khiops_fast/`, avec les classes renommées.
`dataset.py` inclut la correction locale des types `dbdate` / `dbtime` (détectés comme `Date` / `Time`).

**Étape 2 (fait)** : paramètre `deployment` du constructeur :

| Valeur | `predict` / `predict_proba` / `transform` |
|---|---|
| `"kni"` (défaut) | déploiement en mémoire avec khiops-kni ([kni_deploy.py](src/khiops_fast/kni_deploy.py)) |
| `"khiops"` | runner Khiops sur fichiers temporaires, identique à khiops.sklearn |

- `fit` utilise toujours le runner Khiops ; on peut changer de mode après apprentissage avec
  `set_params(deployment=...)`.
- Mono-table (pandas, numpy, sparse) et multi-table (étoile et flocon) : les tables sont sérialisées exactement
  comme les fichiers du mode `"khiops"`, les enregistrements secondaires sont envoyés à KNI pour chaque ligne
  de la table principale, et l'ordre des lignes d'entrée est conservé.
- KNI n'est pas réentrante : la bibliothèque est chargée une fois par processus et un verrou global sérialise
  les déploiements des différents threads. Plusieurs processus ne se bloquent pas entre eux.
- Pas de repli silencieux : une erreur KNI lève une `RuntimeError` avec l'étape et le journal KNI.
- Non supporté en mode `"kni"` : blocs de variables dans la sortie (utiliser `"khiops"`).

**Étape 3 (à faire)** : déploiement KNI en parallèle (pool de processus ayant chacun leur KNI).

## Installation

```bash
conda activate python312
pip install -e ".[dev]"
```

## Utilisation

```python
from khiops_fast import KhiopsFastClassifier

clf = KhiopsFastClassifier().fit(X_train, y_train)          # deployment="kni" par défaut
proba = clf.predict_proba(X_test)
proba_runner = clf.set_params(deployment="khiops").predict_proba(X_test)   # identique
```

Les paramètres, les formats d'entrée (pandas, numpy, sparse, multi-table
`{"main_table": ..., "additional_data_tables": ...}`) et les attributs appris sont ceux de khiops.sklearn.

## Tests

```bash
python -m pytest -q      # ~2 min
```

Les tests entraînent chaque estimateur et son équivalent khiops.sklearn sur les mêmes données (Adult, Iris,
Accidents multi-table étoile et flocon) et vérifient, pour les deux modes de déploiement, que les prédictions,
probabilités, scores et encodages sont identiques. Ils couvrent aussi les entrées numpy/sparse, les tables non
triées, les lignes sans enregistrement secondaire, les appels concurrents depuis plusieurs threads et le pickle.

Comparaison des temps de prédiction :

```bash
python benchmarks/bench_deploy.py --sizes 1 100 10000 100000
```

## Licence

Code dérivé de khiops-python, © Orange, sous licence BSD 3-Clause Clear (voir [LICENSE.md](LICENSE.md)).

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
`estimators.py`, `dataset.py` et `helpers.py` sont des copies ; seuls les noms de classes et les imports internes changent.
Le comportement est identique à khiops.sklearn (vérifié par les tests).
`dataset.py` inclut la correction locale des types `dbdate` / `dbtime` (détectés comme `Date` / `Time`).

**Étape 2 (à faire)** : remplacer, dans `predict`, `predict_proba` et `transform`, l'appel au runner Khiops
(`kh.deploy_model`) par un déploiement en mémoire avec khiops-kni.

## Installation

```bash
conda activate python312
pip install -e ".[dev]"
```

## Utilisation

```python
from khiops_fast import KhiopsFastClassifier

clf = KhiopsFastClassifier().fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

Les paramètres, les formats d'entrée (pandas, numpy, sparse, multi-table
`{"main_table": ..., "additional_data_tables": ...}`) et les attributs appris sont ceux de khiops.sklearn.

## Tests

```bash
python -m pytest -q      # ~2 min
```

Les tests entraînent chaque estimateur et son équivalent khiops.sklearn sur les mêmes données (Adult, Iris,
Accidents multi-table) et vérifient que les prédictions, probabilités, scores et encodages sont identiques.

## Licence

Code dérivé de khiops-python, © Orange, sous licence BSD 3-Clause Clear (voir [LICENSE.md](LICENSE.md)).

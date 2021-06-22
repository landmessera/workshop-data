## Übung: Optimization

!pip install sklearn

Packete importieren

import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle
from sklearn.model_selection import train_test_split

Einlesen der Datensets

with open('../output/bikebuyers/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)
print('Dataset geladen')

### Baseline erstellen

#### Task 1: Laden Transformer Pipeline

Lesen Sie die gespeicherte Transformer Pipeline aus der pickle-Datei '../output/bikebuyers/transformer_pipeline.pkl' und speichern die Pipeline in einer Variable namens transformer_pipeline.

# Hier den Code eingeben

````{Dropdown} Lösung Task 1

  ```{code-block} python
    # Erste Code-Zelle
    with open('../data/bikebuyers/transformer_pipeline.pkl', 'rb') as handle:
        transformer_pipeline = pickle.load(handle)
  ```
````

#### Task 2: Pipeline mit Klassifikator erstellen

Erstellen Sie die finale Pipeline, bestehend aus der Transformer Pipeline und anschließendem Predictor in Form eines K-Nearest-Neighbor Klassifikator. Speichern Sie die Pipeline in einer Variable namens 'full_pipeline'.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 2

  ```{code-block} python
    from sklearn.neighbors import KNeighborsClassifier
    full_pipeline = Pipeline(steps=[
        ('transformers', transformer_pipeline),
        ('predictor', KNeighborsClassifier(n_neighbors=2))
    ])

  ```
````

#### Task 3: Pipeline verwenden

* Trainieren Sie die Pipeline mit dem Trainingsdatenset durch aufrufen der fit()-Methode. 
* Evaluieren Sie das Modell mit dem Validierungsdatenset durch aufrufen der score()-Methode.

Welches Ergebnis erhalten Sie?

# Hier den Code eingeben.

````{Dropdown} Lösung Task 3

  ```{code-block} python
    full_pipeline.fit(datasets['X_train'], datasets['y_train'])
    full_pipeline.score(datasets['X_val'], datasets['y_val'])

  ```
````

### Parameter optimieren

#### Task 4: Grid Search vorbereiten

Erstellen Sie eine Instanz der Klasse [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) aus Scikit Learn. Verwenden folgende Paramter-Einstellungen:
* estimator:full_pipeline
* param_grid: 
    * factor values [1.0, 1.5, 2.0, 3.0]
    * n_neighbors: [2,3,4,5,6]
* cv: 10

# Hier den Code eingeben.

````{Dropdown} Lösung Task 4

  ```{code-block} python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'transformers__num__outlier_remover__factor': [1.0, 1.5, 2.0, 3.0],
        'predictor__n_neighbors': [2,3,4,5,6],
    }

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=10)

  ```
````

#### Task 5: Grid Search anwenden

* Rufen Sie die fit-Methode unter Verwendung der Trainingsdatensets auf.
* Geben Sie die beste Parameterkombination aus.
* Geben Sie das Ergebnis der besten Parameterkombination aus.

# Hier den Code eingeben

````{Dropdown} Lösung Task 5

  ```{code-block} python
    grid_search.fit(datasets['X_train'], datasets['y_train'])

    print("Best params:")
    print(grid_search.best_params_)
    
    print("Ergebnis mit der besten Parametereinstellung auf den Trainingsdaten:")
    print(f"{grid_search.best_score_:.3f}")
  ```
````

### Merkmale optimieren
#### Task xx: xxx

#### Task 6: 

Speichern Sie die Pipeline mit der besten Parametereinstellung in einer Pickle-Datei.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 6

  ```{code-block} python
    
  ```
````

#### Task 7: 

Speichern Sie die Pipeline mit der besten Parametereinstellung in einer Pickle-Datei.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 7

  ```{code-block} python
    
  ```
````

#### Task 8: Beste Pipeline speichern

Speichern Sie die Pipeline mit der besten Parametereinstellung in einer Pickle-Datei.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 8

  ```{code-block} python
    best_pipeline = grid_search.best_estimator_
    with open('../output/bikebuyers/pipeline.pkl', 'wb') as handle:
            pickle.dump(best_pipeline, handle)
  ```
````

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

with open('../output/bikebuyers/transformer_pipeline.pkl', 'rb') as handle:
    transformer_pipeline = pickle.load(handle)

full_pipeline = Pipeline(steps=[
    ('transformers', transformer_pipeline),
    ('predictor', KNeighborsClassifier(n_neighbors=2))
])

param_grid = {
    'transformers__num__outlier_remover__factor': [1.0, 1.5, 2.0, 3.0],
    'predictor__n_neighbors': [2,3,4,5,6],
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=10)
grid_search.fit(datasets['X_train'], datasets['y_train'])

print("Best params:")
print(grid_search.best_params_)

print("Ergebnis mit der besten Parametereinstellung auf den Trainingsdaten:")
print(f"{grid_search.best_score_:.3f}")

best_pipeline = grid_search.best_estimator_

with open('../output/bikebuyers/full_pipeline.pkl', 'wb') as handle:
    pickle.dump(best_pipeline, handle)
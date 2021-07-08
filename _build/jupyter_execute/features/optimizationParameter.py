# Pakete importieren
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from treeinterpreter import treeinterpreter
from sklearn.tree import DecisionTreeClassifier

# Transformierte Daten laden
with open('../output/titanic/datasets_transformed.pkl', 'rb') as handle:
    datasets_transformed = pickle.load(handle)

# Transformer laden
with open('../output/titanic/transformer_pipeline.pkl', 'rb') as handle:
    transformer_pipeline = pickle.load(handle)

# Datensets laden
with open('../output/titanic/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# Full Pipeline erstellen
full_pipeline = Pipeline(steps=[
    ('transformers', transformer_pipeline),
    ('predictor', DecisionTreeClassifier(random_state=0))
])

## Parameter optimieren

### Grid Search anwenden

Für jeden beliebigen Schritt in der Pipeline können Wertebereiche für die Parameter angegeben werden. Für das Anwendungsbeispiel werden folgende Wertebereiche gesetzt:
* der Faktor, der bei der IQR-Methode zur Ausreißererkennung verwendet wird, mit einem Wertebereich von [1.0, 1.5, 2.0, 3.0]
* der min_samples_split Parameter, der beim DecisionTreeClassifier-Algorithmus bestimmt wie viele Daten notwendig sind, um eine weitere Verzweigung zu erstellen, mit einem Wertebereich von [2, 3, 4, 5, 6]

Die Bezeichnung der Parameter folgt einer Regel: Vor die eigentliche Parameterbezeichnung wie z.B. "factor" werden die Pipeline-Namen mit doppeltem Unterstrich getrennt gestellt.

Der Aufbau der Pipeline:

```{figure} ../images/pipelineStructure.png
---
height: 250px
align: center
name: fig-pipelineStructure
---
```

Die Vollständige Bezeichnung lautet: "transformers__num__outlier_remover__factor"

Der Parameter param_grid stellt der GridSearchCV Klasse ein Dictionary bereit. Die Keys entsprechen den Bezeichnungen und die Values einer Liste von Werten, die getestet werden sollen.

from sklearn.model_selection import GridSearchCV

param_grid = {
    'transformers__num__outlier_remover__factor': [1.0, 1.5, 2.0, 3.0],
    'predictor__min_samples_split' : [2,3,4,5,6]
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=10)

Die Fit()-Methode testet alle Kombinationsmöglichkeiten und speichert die beste Parameterkombination im Attribut best_params_. 

grid_search.fit(datasets['X_train'], datasets['y_train'])
grid_search.best_params_

Über das Attribut best_score_ erhält man die Accuracy-Score der besten Parametereinstellung.

grid_search.best_score_

Das Ergebnis auf den Validierungsdaten lautet:

grid_search.score(datasets['X_val'], datasets['y_val'])

Im Vergleich zur vorherigen Einstellung, der Baseline, hat sich das Ergebnis um 1% verbessert.

```{figure} ../images/gridsearchDct.png
---
height: 250px
align: center
name: fig-gridsearchDct
---
```

Ersetzten der bisherigen Pipeline mit der besten Pipeline aus Grid Search.

best_pipeline_gridsearch = grid_search.best_estimator_

Zum Abschluss: Validieren der Pipeline mit dem Validierungsset. Das Ergebnis liefert wie erwartet den Accuracy Score von 0.752

best_pipeline_gridsearch.score(datasets['X_val'], datasets['y_val'])
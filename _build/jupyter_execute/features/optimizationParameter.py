## Parameter optimieren

### Grid Search anwenden

Für jeden beliebigen Schritt in der Pipeline können Wertebereiche für die Parameter angegeben werden. Für das Anwendungsbeispiel werden folgende Wertebereiche gesetzt:
* der Faktor, der bei der IQR-Methode zur Ausreißererkennung verwendet wird, mit einem Wertebereich von [1.0, 1.5, 2.0, 3.0]
* der n_neighbors Parameter, der beim K-Nearest-Neighbor-Algorithmus bestimmt wie viele Nachbarn berücksichtigt werden, mit einem Wertebereich von [2,3,4,5,6,7,8]

from sklearn.model_selection import GridSearchCV

param_grid = {
    'transformers__num__outlier_remover__factor': [1.0, 1.5, 2.0, 3.0],
    'predictor__n_neighbors': [2,3,4,5,6],
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=10)

Die Fit()-Methode testet alle Kombinationsmöglichkeiten und liefert die beste Parameterkombination.

grid_search.fit(datasets['X_train'], datasets['y_train'])

print(f"Best params:")
print(grid_search.best_params_)

Über das Attribut best_score_ wird die Accuracy abgefragt.

print(f"Ergebnis mit der besten Parametereinstellung auf den Trainingsdaten: {grid_search.best_score_:.3f}")

print(("Ergebnis auf den Validierungsdaten: %.3f"
       % grid_search.score(datasets['X_val'], datasets['y_val'])))

Das Ergebnis hat sich im Vergleich zur vorherigen Einstellung um 6% verbessert.

```{figure} ../images/gridSearch.png
---
height: 250px
align: center
name: fig-gridSearch
---
```

Ausgabe der besten Parameter über das Attribut best_params_.

grid_search.best_params_

Ersetzten der Pipeline mit den besten Parametern.

full_pipeline = grid_search.best_estimator_

Zum Abschluss: Validieren der Pipeline mit dem Validierungsset. Das Ergebnis liefert wie erwartet den Accuracy Score von 0.803

full_pipeline.score(datasets['X_val'], datasets['y_val'])
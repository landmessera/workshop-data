## Baseline erstellen

Pakete importieren

import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

Transformierte Daten laden

with open('../output/titanic/datasets_transformed.pkl', 'rb') as handle:
    datasets_transformed = pickle.load(handle)

### Machine Learning Verfahren anwenden

Beim Anwendungsbeispiel Titanic handelt es sich um eine Klassifikationsaufgabe. Die Anwendung eines K-Nearest Neighbors Klassifikators sieht dann wie folgt aus:

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(datasets_transformed['X_train'], datasets_transformed['y_train'])

Nach importieren der KNeighborsClassifier-Klasse, wird eine Instanz erstellt und der Klassifikator mit den transformierten Daten trainiert.

classifier.score(datasets_transformed['X_val'], datasets_transformed['y_val'])

Validiert wird durch aufrufen der score()-Methode und übergabe des Validierungsdatenset.

Das übliche Vorgehen beim Machine Learning ist experimentell. Man entwickelt zunächst eine Variante und erhält ein Ergebnis. In unserem Fall ein Accuracy Score von 0.74, bedeutet 74% der vorgehergesagten Werte sind richtig. Dieser Score dient als Basis für weitere Optimierungen. Es werden Veränderungen unterschiedlichster Art vorgenommen wie zum Beispiel 
* Anwendung weiterer Transformationsschritte
* Entfernen von Transformationsschritte
* Änderung der Transformationseinstellungen
* Hinzufügen von Merkmalen
* Entfernen von Merkmalen
* Modifizieren von Merkmalen
* Ändern des Machine Learning Algorithmus 
* Ändern der Hyperparameter

Nach jeder Änderung wird geprüft ob sich das Ergebnis, der Score, verbessert oder verschlechtert hat und entprechend die Änderung beibehalten oder verworfen. Häufig sind es sehr viele Experimente die durchgeführt werden müssen. Es fällt schwer den Überblick zu behalten und es ist aufwendig die Experimente manuell durchzuführen. Für die Automatisierung der Experimente für Hyperparameter kann die sogenannte Grid-Search[^footnote3] eingesetzt werden. Man gibt für jeden Hyperparamter eine begrenzte Menge von möglichen Werten die getestet werden soll. Grid-Search testet alle Kombinationen und gibt die Wertekombination mit den besten Ergebnisen aus.

Wie bereits zu Beginn dieses Abschnitts erwähnt, ist es möglich am Ende der Pipeline einen beliebigen Estimator einzusetzen anstatt ein Transformer. Ein beliebiger Estimator kann auch ein Predictor sein. So kann beim Anwendungsbeispiel Titanic einfach der Klassifikator am Ende der Pipeline eingefügt werden. Einer der Vorteile, wenn man die Vorverarbeitungsschritte und den Prediktor in einer Pipeline integriert ist, dass Grid-Search auch für die Vorverarbeitungsschritte eingesetzt werden kann.

Transformer Pipeline laden

with open('../output/titanic/transformer_pipeline.pkl', 'rb') as handle:
    transformer_pipeline = pickle.load(handle)

Datensets laden

with open('../output/titanic/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

### Klassifikator in eine Pipeline integrieren

Erstellt wird eine Pipeline, die im ersten Schritt die bereits erstellte Transformer-Pipeline enthält und im Anschluss den Klassifikator.

```{figure} ../images/pipelineFull.png
---
height: 180px
align: center
name: fig-pipelineFull
---
```

full_pipeline = Pipeline(steps=[
    ('transformers', transformer_pipeline),
    ('predictor', KNeighborsClassifier(n_neighbors=3))
])

Die Pipeline wird mit dem Trainingsdatenset trainiert und dem Validierungsset validiert.

full_pipeline.fit(datasets['X_train'], datasets['y_train'])
full_pipeline.score(datasets['X_val'], datasets['y_val'])

Das Ergebnis ist wie erwartet unverändert. Die Accuracy beträgt 0.74. Nun können mit der Grid Search Methode optimale Einstellungen gefunden werden.

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

## Merkmale optimieren

### Wichtige Merkmale erkennen

### Neue Merkmale hinzufügen

## Pipeline speichern

with open('../output/titanic/pipeline.pkl', 'wb') as handle:
    pickle.dump(full_pipeline, handle)

In diesem Abschnitt haben Sie die Kernelemente von Scikit Learn kennengelernt, wie man eigene Transformer erstellt und anwendet, wie Pipelines erstellt und genutzt werden können und wie die Suche nach optimalen Parametern für alle Schritte der Pipeline automatisiert werden kann. Nun sind Sie gefragt: Festigen Sie ihr Wissen durch die Anwendung von Transformationen am Datenset "Bike Buyers".

[^footnote1]: "API design for machine learning software: experiences from the scikit-learn project", L Buitinck, G Louppe, M Blondel, et. al.

[^footnote2]: siehe https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

[^footnote3]: siehe https://scikit-learn.org/stable/modules/grid_search.html
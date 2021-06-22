# Baseline erstellen

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

classifier.score(datasets_transformed['X_val'], datasets_transformed['y_val'])

Das übliche Vorgehen beim Machine Learning ist experimentell. Man entwickelt zunächst eine Variante und erhält ein Ergebnis. In unserem Fall ein Accuracy Score von 0.74, bedeutet 74% der vorgehergesagten Werte sind richtig. Dieser Score dient als Basis für weitere Optimierungen. Es werden Veränderungen unterschiedlichster Art vorgenommen wie zum Beispiel 
* Anwendung weiterer Transformationsschritte
* Entfernen von Transformationsschritte
* Änderung der Transformationseinstellungen
* Hinzufügen von Merkmalen
* Entfernen von Merkmalen
* Modifizieren von Merkmalen
* Ändern des Machine Learning Algorithmus 
* Ändern der Hyperparameter

Nach jeder Änderung wird geprüft ob sich das Ergebnis, der Score, verbessert oder verschlechtert hat und entprechend die Änderung beibehalten oder verworfen. Häufig sind es sehr viele Experimente die durchgeführt werden müssen. Es fällt schwer den Überblick zu behalten und ist aufwendig manuell durchzuführen. Für die Automatisierung der Experimente für Hyperparameter kann die sogenannte Grid-Search[^footnote3] eingesetzt werden. Man gibt für jeden Hyperparamter eine begrenzte Menge von möglichen Werten die getestet werden soll. Grid-Search testet alle Kombinationen und gibt die Wertekombination mit den besten Ergebnisen aus.

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
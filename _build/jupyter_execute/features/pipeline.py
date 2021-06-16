# Tranformationen koordinieren

In den bisherigen Kapiteln wurden die wichtigsten Transformationen für numerische und kategorische Daten vorgestellt und am Beispiel des Titanic Datensets auf den Trainingsdaten angewendet. In der Praxis werden die Transformationen in bestimmter Reihenfolge benötigt und mit unterschiedlichen Einstellungen erprobt. Außerdem müssen die Transformationen nicht nur auf den Trainingsdaten erfolgen, sondern auch auf den Validierungs- und Testdaten. 

Wichtig ist, dass die Bearbeitung der Trainings-, Validierungs-, und Testdaten stets getrennt erfolgt. Denn einer der größten Fehler die in der Anwendung von Machine Learning passieren ist die Durchmischung oder Beeinflussung der Datensets. Wenn zum Beispiel die Anpassung des Skalierungsverfahren nicht nur mit den Trainingsdaten, sondern mit den gesamten Daten stattfindet, haben auch die Testdaten Einfluss auf die Skalierung.

Um derartige Fehler zu vermeiden ist es hilfreich etwas Zeit in die Koordiniation der Transformationen zu investieren. Häufig wird dieser Teil vernachlässigt, da man möglichst schnell zur Anwendung der Machine Learning Modelle gelangen möchte und unterschiedliche Algorithmen testen. Doch meist liegt der Schlüssel zum Erfolg nicht in der Erprobung möglichst zahlreicher Algorithmen, sondern in der Vorverarbeitung der Daten. Ein solides Fundament in der Vorverarbeitung ermöglicht später eine qualitativ hochwertige Erprobung und führt meist zu besseren Ergebnissen.

## Crash Kurs Scikit Learn

Es wurden bereits an mehreren Stellen dieses Workshops Scikit Learn verwendet. Bisher wurden einzelne Methoden aufgerufen ohne den Aufbau und die Konzeption von Scikit Learn zu verstehen. Das genügt in den meisten Fällen auch. Um Transformationen zu koordinieren bietet Scikit Learn die Erstellung sogenannter Pipelines an. Eine Pipeline ist eine Klasse. Bei der Instanziierung werden die Transformationsschritte in Form einer Liste von Tuples übergeben. Ein Tuple enthält den Namen (frei wählbar) und ein Transformer oder Estimator. Was sind Transfomer und Estimatoren? An dieser Stelle ist es hilfreich etwas mehr über den Aufbau und das durchdachte Designkonzept von Scikit Learn zu erfahren.

Alle Objekte besitzen eine konsistente Schnittstelle. Es existieren drei Arten von Schnittstellen: Estimatoren, um Modelle zu erstellen und anzupassen, Prädiktoren, um Vorhersagen zu treffen und  Transformer, um Daten zu transformieren.[^footnote1]

### Estimatoren, Prädiktoren und Transformer

**Estimator**: Die  Estimator-Schnittstelle ist der Kern von Scikit Learn. Sie definiert die Art der Instanzieerung von Objekten und bietet eine Fit-Methode für das Lernen eines Modells. 


```{figure} ../images/estimator.png
---
height: 400px
align: center
name: fig-estimator
---
```

Ein Estimator, der ein Modell für die Lebensmittel Eier, Tomaten und Kartoffeln lernen soll, kann als Eingabe Eigenschaften der Lebensmittel wie z.B. Größe, Form und Farbe und die zugehörige Bezeichnung "Ei", "Tomate" oder "Kartoffel" über den Aufruf der Fit-Methode erhalten. Gelernt wird ein Modell, dass Eingaben auf die Zielgröße abbildet. Das gelernte Modell lautet dann: Ist das Objekt weiß handelt es sich um ein Ei, ist das Objekt rot handelt es sich um eine Tomate, ist das Objekt braun ist es eine Kartoffel. (Beispiel 1)

Ein Estimator kann auch lernen wie Daten verarbeitet werden sollen. Man kann sich das ähnlich wie bei einem Koch-Lehrling vorstellen, der Lebensmittel und Rezepte zur Verarbeitung erhält. Der ausgebildete Koch weiß, mit welchen Lebensmittel bestimmte Gerichte erstellt werden. (Beispiel 2)

Prädiktor- und Transformer-Schnittstellen sind Erweiterungen der Estimator Schnittstellen.


```{figure} ../images/estimatorExtended.png
---
height: 400px
align: center
name: fig-estimatorExtended
---
```

**Prädiktor**: Ein Prädikator ist definiert durch die Erweiterung um die Predict-Methode, die Vorhersagen auf Basis des gelernten Modells treffen kann. Ein Prädiktor der Beispiel 1 erweitert, kann durch Eingabe der Eigenschaften eines neuen Objekts, z.B. Farbe "braun", Größe "5 cm" und Form "oval" über den Aufruf der Predict-Methode, die Aussage treffen, dass es sich um eine Kartoffel handelt.

**Transformer**: Die Erweiterung ist in diesem Fall die Transform-Methode. Sie nimmt Eingabedaten entgegen und liefert die transformierten Daten zurück. Ein Transformer der Beispiel 2 erweitert, kann durch Eingabe einer Kartoffel über die Transform-Methode das Gericht Pommes liefern.


```{figure} ../images/transformerPredictor.png
---
height: 200px
align: center
name: fig-transformerPredictor
---
```

Scikit-Learn stellt eine ganze Reihe von Transformer bereit. Im Abschnitt zur Transformation von numerischen Daten wurden bereits die Transformer MinMaxScaler, StandardScaler und der KBinsDiscretizer verwendet. Trotz des großen Angebots an Transformer zur Datenvorverarbeitung von Scikit Learn[^footnote2] kommt es häufig vor, dass man weitere oder auf den Anwendungsfall spezifische Transformationen benötigt. In diesem Fall lassen sich einfach eigene Transformer erstellen.

### Eigene Transformer erstellen

Wie bereits erwähnt benötigt ein Transformer eine fit()- und transform()-Methode. Außerdem wird eine fit_transform()-Methode benötigt, die beide Methoden kombiniert. Als Beispiel wird die Ausreißererkennung und -entfernung wie sie im Abschnitt zur Transformation von numerischen Daten gezeigt wurde als Transformer implementiert. Man erstellt zunächst eine Klasse die von den Klassen BaseEstimator und TransformerMixin erben. Der BaseEstimator liefert die Möglichkeit die Methoden get_params() und set_params() zu nutzen, die TransformerMixin Klasse erstellt automatisch bei gegebenen fit()- und transform()-Methoden, die fit_transform()-Methode.

Die fit()-Methode muss im Fall Ausreißererkennung und -entfernung keine Aufgabe erfüllen. Der Inhalt der Methode bleibt leer. Der Rückgabewert entspricht dem Instanz selbst unverändert.

Die transform()-Methode enthält die in Abschnitt "Transformation > Numerische Daten > Ausreißer erkennen" beschriebenen Zeilen Code, um Ausreißer mit der IQR-Methode zu erkennen und mit dem Median-Wert zu ersetzen. Der Faktor wird in der __init__() Methode über den Parameter "factor" übergeben und gesetzt. Der Default-Wert beträgt 1.5.

import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = pd.DataFrame(X)
        q1 = X_.quantile(0.25)
        q3 = X_.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.factor * iqr)
        upper_bound = q3 + (self.factor * iqr)
        X_[((X_ < lower_bound) | (X_ > upper_bound))] = np.nan
        X_.fillna(X_.median(), inplace=True)
        return X_.values

### Transformer anwenden

```{figure} ../images/transformerOutlier.png
---
height: 200px
align: center
name: fig-transformerOutlier
---
````

Es wird ein Datenframe erstellt und mit Beispieldaten befüllt. Das Merkmal 'Größe' enthält einen Ausreißer 999 in der dritten Zeile, das Merkmal 'Gewicht' enthält keinen Ausreißer und das Merkmal 'Alter' enthält einen Ausreißer in der zweiten Zeile mit dem Wert '-16'.

X = pd.DataFrame({'Größe':[60,23,999,54],'Gewicht':[30,3,5,25],'Alter':[2,-16,10,4]}, index=[1,2,3,4])
X

Es wird eine Instanz der Klasse erstellen.

outlier_transformer = OutlierRemover()

Aufrufen der fit()-Methode.

outlier_transformer.fit(X)

Aufrufen der transform()-Methode.

res = outlier_transformer.transform(X)
pd.DataFrame(res, columns=X.columns)

Alternativ kann die fit_transform()-Methode aufgerufen werden.

res = outlier_transformer.fit_transform(X)
pd.DataFrame(res, columns=X.columns)

Die Ausreißer wurden erkannt und mit dem NaN-Wert ersetzt. Sie wissen jetzt, wie man Transformer von Scikit Learn anwendet und wie man eigene Transformer erstellt. Im nächsten Schritt wird gezeigt wie diese Transformer in einer Pipeline verwendet werden können.

### Pipelines erstellen

Die Klasse Pipeline aus Scikit-Learn unterstützen die Organisation von Transformationen. Bei der Instanziierung werden die Transformationsschritte in einer Liste von Tuples übergeben. Ein Tuple enthält den Namen (frei wählbar) und ein Transformer. Das letzte Element der Liste kann ein Tuple sein, dass anstatt eines Transformers einen Estimator enthält.


```{figure} ../images/pipelineGeneral.png
---
height: 250px
align: center
name: fig-pipelineGeneral
---
````

Packete importieren

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

Erstellen einer einfachen Pipeline, die den eigenen Transformer zur Ausreißerentfernung aufruft und anschließend eine Standardisierung vornimmt.

pipeline_numerical = Pipeline(steps=[
    ('outlier_remover', OutlierRemover()),
    ('scaler', StandardScaler())
])

pipeline_numerical.fit_transform(X)

Die Werte der bisherigen Daten waren ausschließlich numerisch. Wie bereits aus den vorigen Kapiteln bekannt ist, sind bestimmte Transformationen für entsprechende Datentypen notwendig. Für dieses Handling kann der ColumnTransformer zum Einsatz kommen. Beim Instanziieren des ColumnTransformers werden dem Parameter "transformers" eine Liste von Tuples (name, transformer, columns) übergeben, wobei der Name frei wählbar ist, 'transformer' ein einzelner Transformer oder eine Pipeline sein kann und 'columns' eine Liste der Merkmale darstellt, die transformiert werden sollen.

Erweitern der Beispieldaten um kategorische Daten:

X['Tierart'] = ['Hund', 'Maus', 'Maus', 'Hund']
X['Gemütszustand'] = ['glücklich', 'traurig', 'neutral', 'traurig']
X

Eine Pipeline für kategorische Daten erstellen.

```{figure} ../images/transformerOneHot.png
---
height: 180px
align: center
name: fig-transformerOneHot
---
```

pipeline_categorical = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

Erstellen einer Instanz des ColumnTransformer, wobei die erste Pipeline für numerische Daten und die zweite erstellte Pipeline für kategorische Daten verwendet werden soll.

features_numerical = ['Größe', 'Gewicht', 'Alter']
features_categorical = ['Tierart', 'Gemütszustand']

preprocessor = ColumnTransformer(
    transformers = [
        (
            'numeric', 
            pipeline_numerical,
            features_numerical
        ),
        (
            'categorical', 
            pipeline_categorical,
            features_categorical
        )
    ])

res = preprocessor.fit_transform(X)
pd.DataFrame(res)

Ermitteln der neuen Spaltenbezeichnungen.

feature_categorical_onehot = preprocessor.transformers_[1][1]['onehot'].get_feature_names(features_categorical)
list(feature_categorical_onehot)

Spaltenbezeichnungen einfügen.

pd.DataFrame(res, columns=features_numerical+list(feature_categorical_onehot))

## Pipeline für Transformationen am Beispiel Titanic

Laden der Datensets aus Pickle File

with open('datasets.pickle', 'rb') as handle:
    datasets = pickle.load(handle)

Speichern des Trainingsdatenset in der Variable X_train und Anzeigen der ersten Zeilen.

X_train = datasets['X_train']
X_train.head()

### Pipeline erstellen

Erstellen des Column Transformer unter Verwendung der bereits erstellten Pipelines für numerische und kategorsiche Daten.

features_numerical = ['Age', 'SibSp', 'Parch', 'Fare']
features_categorical = ['Pclass', 'Sex', 'Embarked']

transformer_pipeline = ColumnTransformer(
    transformers = [
        (
            'num', 
            pipeline_numerical,
            features_numerical
        ),
        (
            'cat',
            pipeline_categorical,
            features_categorical
        )
    ])

Transformationen anwenden.

res = transformer_pipeline.fit_transform(X_train)

Neue Spaltenbezeichnungen aufrufen und in der Variable features_categorcial_onehot speichern.

feature_categorical_onehot = transformer_pipeline.transformers_[1][1]['onehot'].get_feature_names(features_categorical)
list(feature_categorical_onehot)

Ergebnis anzeigen:

pd.DataFrame(res, columns=features_numerical+list(feature_categorical_onehot))

### Umgang mit weiteren Datensets

Bisher wurde nur das Trainingsdatenset transformiert. Validierungs- und Testset müssen ebenfalls transformiert werden. Wichtig ist dabei, dass nur transformiert und nicht trainiert wird. Das scheint selbstverständlich, ist jedoch eine häufige Fehlerquelle. Training im Kontext von Machine Learning bedeutet, dass etwas aus Daten gelernt wird. Im Fall der Transformationen findet zum Beispiel ein Training statt, wenn man eine Standardisierung vornimmt. Es wird gelernt auf welchen Bereich die Daten skaliert werden sollen. Das Training darf ausschließlich mit den Trainingsdaten stattfinden. Die Skalierung selbst, also die Transformation der Daten findet auf den Trainings-, Validierungs- und Testdaten statt. 

Wie wendet man jetzt die Pipeline korrekt auf die anderen Datensets an? 
* Trainingsdatenset: 
    * fit_transform() Methode aufrufen
* Validierungsdatenset: 
    * transform()-Methode aufrufen
* Testdatenset: 
    * transform()-Methode aufrufen

Datensets in Variablen speichern.

X_train = datasets['X_train']
y_train = datasets['y_train']
X_val = datasets['X_val']
y_val = datasets['y_val']
X_test = datasets['X_test']
y_test = datasets['y_test']

X_train_transformed = transformer_pipeline.fit_transform(X_train)
X_val_transformed = transformer_pipeline.transform(X_val)
X_test_transformed = transformer_pipeline.transform(X_test)

feature_categorical_onehot = transformer_pipeline.transformers_[1][1]['onehot'].get_feature_names(features_categorical)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=features_numerical+list(feature_categorical_onehot))
X_val_transformed = pd.DataFrame(X_val_transformed, columns=features_numerical+list(feature_categorical_onehot))
X_test_transformed = pd.DataFrame(X_test_transformed, columns=features_numerical+list(feature_categorical_onehot))

X_train_transformed.head()

X_val_transformed.head()

X_test_transformed.head()

Die erste Pipeline ist erstellt und die Transformationen auf alle Datensets angewendet. Man kann an dieser Stelle die transformierten Datensets für Machine Learning Verfahren verwenden. 

### Machine Learning Verfahren anwenden

Beim Anwendungsbeispiel Titanic handelt es sich um eine Klassifikationsaufgabe. Die Anwendung eines K-Nearest Neighbors Klassifikators sieht dann wie folgt aus:

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train_transformed, y_train)

classifier.score(X_val_transformed, y_val)

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

full_pipeline.fit(X_train, y_train)
full_pipeline.score(X_val, y_val)

Das Ergebnis ist wie erwartet unverändert. Die Accuracy beträgt 0.74. Nun können mit der Grid Search Methode optimale Einstellungen gefunden werden.

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

grid_search.fit(X_train, y_train)

print(f"Best params:")
print(grid_search.best_params_)

Über das Attribut best_score_ wird die Accuracy abgefragt.

print(f"Ergebnis mit der besten Parametereinstellung auf den Trainingsdaten: {grid_search.best_score_:.3f}")

print(("Ergebnis auf den Validierungsdaten: %.3f"
       % grid_search.score(X_val, y_val)))

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

full_pipeline.score(X_val, y_val)

In diesem Abschnitt haben Sie die Kernelemente von Scikit Learn kennengelernt, wie man eigene Transformer erstellt und anwendet, wie Pipelines erstellt und genutzt werden können und wie die Suche nach optimalen Parametern für alle Schritte der Pipeline automatisiert werden kann. Nun sind Sie gefragt: Festigen Sie ihr Wissen durch die Anwendung von Transformationen am Datenset "Bike Buyers".

[^footnote1]: "API design for machine learning software: experiences from the scikit-learn project", L Buitinck, G Louppe, M Blondel, et. al.

[^footnote2]: siehe https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

[^footnote3]: siehe https://scikit-learn.org/stable/modules/grid_search.html
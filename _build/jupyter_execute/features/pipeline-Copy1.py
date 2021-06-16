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

Die transform()-Methode enthält die in Abschnitt "Transformation > Numerische Daten > Ausreißer erkennen" beschriebenen Zeilen Code, um Ausreißer mit der IQR-Methode zu erkennen und mit dem NaN-Wert zu ersetzen. Der Faktor wird in der __init__() Methode über den Parameter "factor" übergeben und gesetzt. Der Default-Wert beträgt 1.5.

from sklearn.base import BaseEstimator, TransformerMixin
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def remove_outliers(self, X, y=None):
        X_ = pd.Series(X).copy()
        q1 = X_.quantile(0.25)
        q3 = X_.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.factor * iqr)
        upper_bound = q3 + (self.factor * iqr)
        X_.loc[((X_ < lower_bound) | (X_ > upper_bound))] = np.nan 
        return pd.Series(X_)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(self.remove_outliers)

### Transformer anwenden

Es wird ein Datenframe erstellt und mit Beispieldaten befüllt. Das Merkmal 'a' enthält einen Ausreißer 9999 in der dritten Zeile, das Merkmal 'b' enthält keinen Ausreißer und das Merkmal 'c' enthält einen Ausreißer in der zweiten Zeile mit dem Wert '-16'.

X = pd.DataFrame({'a':[1000,2000,9999,1500],'b':[1,0,1,2],'c':[2,-16,1,0]}, index=[1,2,3,4])
X

Es wird eine Instanz der Klasse erstellen.

outlier_transformer = OutlierRemover()

Aufrufen der fit()-Methode.

outlier_transformer.fit(X)

Aufrufen der transform()-Methode.

outlier_transformer.transform(X)

Alternativ kann die fit_transform()-Methode aufgerufen werden.

outlier_transformer.fit_transform(X)

Die Ausreißer wurden erkannt und mit dem NaN-Wert ersetzt. Sie wissen jetzt, wie man Transformer von Scikit Learn anwendet und wie man eigene Transformer erstellt. Im nächsten Schritt wird gezeigt wie diese Transformer in einer Pipeline verwendet werden können.

### Pipelines erstellen

Die Klasse Pipeline aus Scikit-Learn unterstützen die Organisation von Transformationen. Bei der Instanziierung werden die Transformationsschritte in einer Liste von Tuples übergeben. Ein Tuple enthält den Namen (frei wählbar) und ein Transformer. Das letzte Element der Liste kann ein Tuple sein, dass anstatt eines Transformers einen Estimator enthält.

Packete importieren

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

Laden der Datensets aus Pickle File

with open('datasets.pickle', 'rb') as handle:
    datasets = pickle.load(handle)

Erstellen einer einfachen Pipeline, die den eigenen Transformer zur Ausreißererkennung und -entfernung aufruft und anschließend eine Min-Max-Skalierung vornimmt.

sc = StandardScaler()

sc.fit_transform(X)

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('outlierRemover', OutlierRemover())
])

numeric_features = ['a', 'b', 'c']

preprocessor = ColumnTransformer(
    transformers = [
        (
            'num', 
            pipeline,
            numeric_features
        )
    ])

preprocessor.fit_transform(X)

pipeline.fit_transform(X)



numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

X_train = datasets['X_train']
X_train.head()

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers = [
        (
            'num', 
            numeric_transformer,
            numeric_features
        ),
        (
            'cat',
            categorical_transformer,
            categorical_features
        )
    ])

res = preprocessor.fit_transform(X_train)

# create new columns
new_columns = numeric_features
for feature in categorical_features:
    new_columns += X_train[feature].unique().tolist()

df_res = pd.DataFrame(res, columns=new_columns)

df_res.head()

## Outlier Detection Transformer erstellen

X = pd.Series(X).copy()
q1 = X.quantile(0.25)
q3 = X.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (self.factor * iqr)
upper_bound = q3 + (self.factor * iqr)
X.loc[((X < lower_bound) | (X > upper_bound))] = np.nan 

[^footnote1]: "API design for machine learning software: experiences from the scikit-learn project", L Buitinck, G Louppe, M Blondel, et. al.

[^footnote2]: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
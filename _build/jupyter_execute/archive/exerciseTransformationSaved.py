# Übung: Tranformation

!pip install sklearn

Packete importieren

import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle
from sklearn.model_selection import train_test_split

## Numerische Daten

### Task 1: Einlesen der Datensets

Lesen Sie die gespeicherte Datensets aus der pickle-Datei '../output/bikebuyers/datasets.pkl' aus und geben Sie die ersten fünf Zeilen der Merkmale im Trainingsdatenset (X_train) aus.

!ls -la ../data/bikebuyers/

f = open('../data/bikebuyers/datasets.pkl', 'rb')
datasets = pickle.load(f)

datasets['X_train']

Geben Sie die ersten fünf Zeilen der Zielgrößen im Trainingsdatenset (y_train) aus.

# Hier den Code eingeben

````{Dropdown} Lösung Task 1

  ```{code-block} python
    # Erste Code-Zelle
    with open('../data/bikebuyers/datasets.pkl', 'rb') as handle:
        datasets = pickle.load(handle)
        
    datasets['X_train'].head()
    
    # Zweite Code-Zelle
    datasets['y_train'].head()
  ```
````

### Task 2: Ausreißer erkennen

Ermitteln Sie mit der IQR-Methode die Ausreißer, ersetzen diese mit dem NaN-Wert (np.nan) und geben Sie die Anzahl der Ausreißer pro Merkmal aus. Verwenden Sie den Faktor 1.5 bei der IQR-Methode.

# Hier den Code eingeben.

Ersetzen Sie die NaN-Werte mit dem Mittelwert.

# Hier den Code eingeben.

````{Tip}
Schritte zur Lösung:
* Eine Variable factor erstellen und mit dem Wert 1.5 belegen.
* Mit Hilfe der [quantile()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html)-Methode das 25%-Quantil bestimmen und in einer Variable q1 speichern.
* Mit Hilfe der [quantile()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html)-Methode das 75%-Quantil bestimmen und in einer Variable q3 speichern.
* Die Differenz von q3 und q1 berechnen und in einer Variable namens iqr speichern.
* Die Untere Grenze berechnen durch die Differenz von q1 und dem Faktor multipliziert mit iqr.
* Alle Werte die außerhalb des Bereichs liegen (definierte Grenzen) mit dem NaN-Wert belegen.
* Mit Hilfe der Methode [isna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isna.html) alle NaN-Werte identifizieren und mit Hilfe der Methode sum() die Anzahl pro Merkmal ausgeben.
* Die Methode [fillna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) hilft bei dem Ersetzen der NaN-Werte.
* Die [mean()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mean.html)-Methode ermittelt Mittelwerte eines DataFrames.
````

````{Dropdown} Lösung Task 2

  ```{code-block} python
    # Erste Code-Zelle
    X_ = pd.DataFrame(datasets['X_train'])
    factor = 1.5
    q1 = X_.quantile(0.25)
    q3 = X_.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    X_[((X_ < lower_bound) | (X_ > upper_bound))] = np.nan
    X_.isna().sum()
    
    # Zweite‚ Code-Zelle
    X_.fillna(X_.mean(), inplace=True)
  ```
````

### Task 3: Outlier Remover Transformer erstellen

Erstellen Sie eine Klasse "OutlierRemoverExtended", welche das Transformer-Interface von Scikit Learn abbildet und von den Klassen BaseEstimator und TranformerMixin ableitet. Bei der Instanziierung der Klasse sollen zwei Parameter gesetzt werden können:
* factor, Default-Wert 1.5
* strategy, Default-Wert 'median'

Die fit()-Methode soll zwei numpy-Arrays als Parameter (X und y), keine Funktion enthalten und nur die Instanz selbst zurückgeben. Die Transformer-Methode soll die gleichen Parameter wie die fit()-Methode erhalten, Ausreißer mit der IQR-Methode unter Verwendung des factor-Paramters erkennen und mit dem Mittelwert oder dem Median ersetzen. Welche Art zum Einsatz kommt, wird mit dem Paramter strategy bestimmt. Valide Werte sind 'median' und 'mean'.

# Hier den Code eingeben

````{Dropdown} Lösung Task 3

  ```{code-block} python
    from sklearn.base import BaseEstimator, TransformerMixin
    class OutlierRemoverExtended(BaseEstimator, TransformerMixin):
        def __init__(self, factor=1.5, strategy='median'):
            self.factor = factor
            self.strategy = strategy

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

            if self.strategy == 'median':
                X_.fillna(X_.median(), inplace=True)
            elif self.strategy == 'mean':
                X_.fillna(X_.mean(), inplace=True)
            else:
                raise ValueError('Invalid value for strategy paramter. Valid values are median or mean.')

            return X_.values
  ```
````

### Task 4: Transformer anwenden

* Erstellen Sie eine Insatnz der Klasse OutlierRemoverExtended.
* Wenden sie die fit_transform()-Methode auf das Trainingsdatenset an und speichern das Ergebnis in einer Variable.
* Erstellen Sie ein Pandas DataFrame, übergeben sie dabei die transformierten Werte und geben das Ergebnis aus.

# Hier den Code eingeben

````{Dropdown} Lösung Task 4

  ```{code-block} python
    outlier_remover = OutlierRemoverExtended()
    res = outlier_remover.fit_transform(datasets['X_train'])
    pd.DataFrame(res)
  ```
````

### Task 5: Pipeline für numerische Daten erstellen

Erstellen Sie eine Pipeline mit Hilfe der [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)-Klasse von Scikit Learn. Schritte der Pipeline:
1. Der OutlierRemoverExtended-Transformer aus Task 3  
2. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) von Scikit Learn

Speichern Sie die Pipeline in einer Variable namens pipeline_numerical.

````{Dropdown} Lösung Task 5

  ```{code-block} python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline_numerical = Pipeline(steps=[
        ('outlier_remover', OutlierRemoverExtended()),
        ('scaler', StandardScaler())
    ])
  ```
````

## Kategorische Daten

### Task 6: Pipeline für kategorische Daten erstellen

Erstellen Sie eine Pipeline mit Hilfe der [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)-Klasse von Scikit Learn. Schritte der Pipeline:
1. [OneHot-Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) von Scikit Learn  

Speichern Sie die Pipeline in einer Variable namens pipeline_categorical.

# Hier den Code eingeben

````{Dropdown} Lösung Task 6

  ```{code-block} python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    pipeline_categorical = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
  ```
````

## Transformationen koordinieren

### Task 7: Pipelines zusammenfügen und Merkmale zuweisen

* Bezeichnungen der numerischen Features in einer Liste mit Elementen vom Typ String namens 'features_numerical' speichern.
* Bezeichnungen der kategorischen Features in einer Liste mit Elmenten vom Typ String namens 'features_categorical' speichern.
* Eine Instanz des [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) aus Scikit Learn erstellen. Dabei dem Parameter transformers die Tuples 
    * ('num', pipeline_numerical, features_numerical)
    * ('cat', pipeline_categorical, features_categorical)  
übergeben

# Hier den Code eingeben

````{Dropdown} Lösung Task 7

  ```{code-block} python
    from sklearn.compose import ColumnTransformer

    features_numerical = ['Income', 'Age', 'Cars', 'Children']
    features_categorical = [
        'Marital Status', 
        'Gender', 
        'Education', 
        'Occupation', 
        'Home Owner', 
        'Commute Distance',
        'Region'
    ]

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
  ```
````

### Task 8: Transformer-Pipeline anwenden

* Die Transformer Pipeline auf dem Trainingsdatenset(X_train) anwenden durch aufrufen der Methode fit_transform() und speichern des Rückgabewert vom Typ Numpy-Array in einer Variable namens 'res'.
* Aus dem Numpy-Array res ein Pandas DataFrame erstellen.


# Hier den Code eingeben

````{Dropdown} Lösung Task 8

  ```{code-block} python
    res = transformer_pipeline.fit_transform(datasets['X_train'])
    pd.DataFrame(res)
  ```
````

### Task 9: Merkmalsbezeichnungen hinzufügen

* Die neuen Feature-Bezeichnungen aus der Transformer Pipeline des Step 'onehot' über die Methode [get_feature_names()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) abfragen und in einer Variable namens feature_categorical_onehot speichern.
* Aus dem Numpy-Array res und den Merkmalsbezeichnungen (feature_catgorical_onehot und features_numerical) ein Pandas DataFrame erstellen und ausgeben.

# Hier den Code eingeben

````{Dropdown} Lösung Task 9

  ```{code-block} python
    feature_categorical_onehot = transformer_pipeline\
        .transformers_[1][1]['onehot']\
        .get_feature_names(features_categorical)
        
    pd.DataFrame(res, columns=features_numerical+list(feature_categorical_onehot))

  ```
````

### Task 10: Auf alle Datensets anwenden

Wenden Sie die Transformationen auf alle Datensets (Training, Validierung und Test) an und speichern das Ergebnis in den Variablen X_train_transformed, X_val_transformed und X_test_transformed.

# Hier den Code eingeben.

Erstellen Sie aus jedem transformierten Datenset ein Pandas Datenframe inklusive der Spaltenbezeichnungen und speichern diese in den gleichen Variablen. Die neuen Spaltenbezeichungen der kategorischen Daten können sie aus der Variable feature_categorical_onehot des vorherigen Tasks auslesen.

# Hier den Code eingeben.

Geben Sie die ersten Zeilen des Pandas Dataframe X_train_transformed aus:

# Hier den Code eingeben.

````{Dropdown} Lösung Task 10

  ```{code-block} python
    # Erste Code-Zelle
    X_train_transformed = transformer_pipeline.fit_transform(datasets['X_train'])
    X_val_transformed = transformer_pipeline.transform(datasets['X_val'])
    X_test_transformed = transformer_pipeline.transform(datasets['X_test'])
    
    # Zweite Code-Zelle
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=features_numerical+list(feature_categorical_onehot))
    X_val_transformed = pd.DataFrame(X_val_transformed, columns=features_numerical+list(feature_categorical_onehot))
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=features_numerical+list(feature_categorical_onehot))
    
    # Dritte Code-Zelle
    X_train_transformed.head()

  ```
````

### Task 11: Pipeline mit Klassifikator erstellen

Erstellen Sie die finale Pipeline, bestehend aus der Transformer Pipeline und anschließendem Predictor in Form eines K-Nearest-Neighbor Klassifikator. Speichern Sie die Pipeline in einer Variable namens 'full_pipeline'.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 11

  ```{code-block} python
    from sklearn.neighbors import KNeighborsClassifier
    full_pipeline = Pipeline(steps=[
        ('transformers', transformer_pipeline),
        ('predictor', KNeighborsClassifier(n_neighbors=2))
    ])

  ```
````

### Task 12: Pipeline verwenden

* Trainieren Sie die Pipeline mit dem Trainingsdatenset durch aufrufen der fit()-Methode. 
* Evaluieren Sie das Modell mit dem Validierungsdatenset durch aufrufen der score()-Methode.

Welches Ergebnis erhalten Sie?

# Hier den Code eingeben.

````{Dropdown} Lösung Task 12

  ```{code-block} python
    full_pipeline.fit(datasets['X_train'], datasets['y_train'])
    full_pipeline.score(datasets['X_val'], datasets['y_val'])

  ```
````

### Task 13: Grid Search vorbereiten

Erstellen Sie eine Instanz der Klasse [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) aus Scikit Learn. Verwenden folgende Paramter-Einstellungen:
* estimator:full_pipeline
* param_grid: 
    * factor values [1.0, 1.5, 2.0, 3.0]
    * n_neighbors: [2,3,4,5,6]
* cv: 10

# Hier den Code eingeben.

````{Dropdown} Lösung Task 12

  ```{code-block} python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'transformers__num__outlier_remover__factor': [1.0, 1.5, 2.0, 3.0],
        'predictor__n_neighbors': [2,3,4,5,6],
    }

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=10)

  ```
````

### Task 13: Grid Search anwenden

* Rufen Sie die fit-Methode unter Verwendung der Trainingsdatensets auf.
* Geben Sie die beste Parameterkombination aus.
* Geben Sie das Ergebnis der besten Parameterkombination aus.

# Hier den Code eingeben

````{Dropdown} Lösung Task 13

  ```{code-block} python
    grid_search.fit(datasets['X_train'], datasets['y_train'])

    print("Best params:")
    print(grid_search.best_params_)
    
    print("Ergebnis mit der besten Parametereinstellung auf den Trainingsdaten:")
    print(f"{grid_search.best_score_:.3f}")
  ```
````

### Task 14: Beste Pipeline speichern

Speichern Sie die Pipeline mit der besten Parametereinstellung in einer Pickle-Datei.

# Hier den Code eingeben.

````{Dropdown} Lösung Task 14

  ```{code-block} python
    best_pipeline = grid_search.best_estimator_
    with open('../output/bikebuyers/pipeline.pkl', 'wb') as handle:
            pickle.dump(best_pipeline, handle)
  ```
````

import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load data
with open('../output/bikebuyers/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# Create custom transformer
class OutlierRemoverExtended(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5, strategy='median'):
        self.factor = factor
        self.strategy = strategy

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

        if self.strategy == 'median':
            X_.fillna(X_.median(), inplace=True)
        elif self.strategy == 'mean':
            X_.fillna(X_.mean(), inplace=True)
        else:
            raise ValueError('Invalid value for strategy paramter. Valid values are median or mean.')

        return X_.values

# Numerical Pipeline
pipeline_numerical = Pipeline(steps=[
    ('outlier_remover', OutlierRemoverExtended()),
    ('scaler', StandardScaler())
])

# Categorical Pipeline
pipeline_categorical = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer
features_numerical = ['Income', 'Age', 'Cars', 'Children']
features_categorical = [
    'Marital Status', 
    'Gender', 
    'Education', 
    'Occupation', 
    'Home Owner', 
    'Commute Distance',
    'Region'
]

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

# transform datsets
X_train_transformed = transformer_pipeline.fit_transform(datasets['X_train'])
X_val_transformed = transformer_pipeline.transform(datasets['X_val'])
X_test_transformed = transformer_pipeline.transform(datasets['X_test'])

feature_categorical_onehot = transformer_pipeline\
    .transformers_[1][1]['onehot']\
    .get_feature_names(features_categorical)

# Zweite Code-Zelle
X_train_transformed = pd.DataFrame(X_train_transformed, columns=features_numerical+list(feature_categorical_onehot))
X_val_transformed = pd.DataFrame(X_val_transformed, columns=features_numerical+list(feature_categorical_onehot))
X_test_transformed = pd.DataFrame(X_test_transformed, columns=features_numerical+list(feature_categorical_onehot))

print(X_train_transformed.head())

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

with open('../output/bikebuyers/pipeline.pkl', 'wb') as handle:
        pickle.dump(best_pipeline, handle)
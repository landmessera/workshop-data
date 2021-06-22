# Optimization

Visualisierung der Daten um Erkenntnisse zu gewinnen
* Suche nach Korrelkationen (scatter matrix von pandas?)

import pandas as pd
import numpy as np
import pickle

with open('datasets.pickle', 'rb') as handle:
    datasets = pickle.load(handle)

X_train = datasets['X_train']
X_train.head()

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

with open('../output/titanic/pipeline.pkl', 'rb') as handle:
    pipeline = pickle.load(handle)

pipeline.fit_predict(datasets['X_train'], datasets['y_train'])

# Merkmalsoptimierung

Most important features / Nützliche Features identifizieren

Neue Features erstellen
* Experimentieren mit Kombinationen von Merkmalen

Clustering mit K-Means

PCA
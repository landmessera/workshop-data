
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

sibsp_ix, parch_ix = 1,2

class FamilySize(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = pd.DataFrame(X)
        X_["FamilySize"] = X_.iloc[:,sibsp_ix] + X_.iloc[:,parch_ix] + 1
        X_['Single'] = X_['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        X_['SmallF'] = X_['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
        X_['MedF'] = X_['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
        X_['LargeF'] = X_['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
        return X_.values

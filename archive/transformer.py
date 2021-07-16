
import pandas as pd
import numpy as np
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

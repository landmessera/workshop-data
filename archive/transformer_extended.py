
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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

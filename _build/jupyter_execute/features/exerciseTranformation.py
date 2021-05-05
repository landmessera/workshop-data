# Pipeline erstellen

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

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
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

# Pakete importieren
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from treeinterpreter import treeinterpreter
from sklearn.tree import DecisionTreeClassifier

# Transformierte Daten laden
with open('../output/titanic/datasets_transformed.pkl', 'rb') as handle:
    datasets_transformed = pickle.load(handle)

# Datensets laden
with open('../output/titanic/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# Merkmale optimieren

Eine weitere Möglichkeit Optimierungen durchzuführen ist die Bearbeitung der Merkmale. Hierbei wird Domänenwissen eingebracht um z.B. neue Merkmale aus Kombinationen von mehreren Merkmalen zu erstellen oder eine Diskretisierung der Werte durchzuführen. Das Domänenwissen kann über Experten aus dem Anwendungsbereich eingeholt werden oder bzw. meist in Kombination durch Einarbeiten in den Anwenungsbereich. 

Zunächst versucht man die Daten besser zu verstehen zum Beispiel durch Visualisierungen oder Statistiken. Im ersten Schritt versucht man zu erkennen welche Merkmale wichtig sind?

## Wichtige Merkmale erkennen

Eine Korrelationsmatrix, zeigt wie stark Merkmale korrelieren. Fügt man die Zielwerte, im Beispiel Titanic die Spalte "Survived" hinzu, kann nicht nur die Korrelation zwischen Merkmalen, sondern auch die Korrelation der Merkmale mit dem Zielwert abgefragt werden.

Erstellen eines temporären Datenframe und hinzufügen der Zielwerte in eine Spalte namens "Survived".

datasets_transformed['X_train'].shape

datasets['y_train'].shape

df_temp = datasets_transformed['X_train'].copy()
df_temp['Survived'] = datasets_transformed['y_train'].values

Pandas liefert die corr()-Methode um eine Korrelationsmatrix zu erstellen. Visualisiert wird die Matrix mit Hilfe der [heatmap()-Methode der Bibliothek Seaborn](https://seaborn.pydata.org/generated/seaborn.heatmap.html) und Matplotlib.

plt.subplots(figsize=(20,15))
sns.heatmap(df_temp.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")



- Signifikante Korrelationen mit der Zielgröße "Survived"
* Sex_female(-0.55) und Sex_male (0.55)
* Pclass_3 (0.32) und Pclass_1(0.28)
* Fare (0.23)

- Sex_male und Sex_female -> 1.0 da Merkmale aus einem kategorischen Merkmal mit zwei Klassen erstellt worden sind 
- Pclass_1 und Fare -> 0.61 da für die erste Klasse sehr hohe Ticketpreise bezahlt wurden
- Parch und SibSp -> Geschwister oder Ehepartner an Bord oft mit Familie, daher Korrelation mit Eltern oder/und Kinder

# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp",y="Survived",data=df_temp, kind="bar", height=6, palette = "muted")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Wahrscheinlichkeit zu überleben")

# Explore SibSp feature vs Survived
g = sns.catplot(x="Pclass_3",y="Survived",data=df_temp, kind="bar", height=6, palette = "muted")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Wahrscheinlichkeit zu überleben")

-> hohe Werte (viele Geschwister oder/und Ehepartner an Bord) führen zu einer geringeren Wahrscheinlichkeit zu überleben. Niedrige Werte -> geringe Überlebenswahrscheinlichkeit
-> Was ist hier der Hintergrund? Es liegt nahe, dass Passagiere nur gemeinsam mit den Familienangehörigen das Schiff verlassen wollten. Annahme: Je größer die Familie umso schwieriger ein Rettungsboot zu finden. Ist eine Person allein an Bord, war es zwar einfacher einen Platz im Rettungsboot zu finden, jedoch der Weg dorthin musste allein ohne Hilfe erfolgen.


# Explore SibSp feature vs Survived
g = sns.catplot(x="Parch",y="Survived",data=df_temp, kind="bar", height=6, palette = "muted")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Wahrscheinlichkeit zu überleben")

df_temp['Age'].hist()

bins = [0, 1, 16, 30, 64,100]
labels = [1,2,3,4,5]
df_temp['Age_binned'] = pd.cut(df_temp['Age'], bins=bins, labels=labels)

# Explore SibSp feature vs Survived
g = sns.catplot(x="Age_binned",y="Survived",data=df_temp, kind="bar", height=6, palette = "muted")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Wahrscheinlichkeit zu überleben")

df_temp.corr()['Survived']

### Neue Merkmale erstellen

Erstellen eines Family Size und Age Transformer

#%%writefile transformer_family.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

age_ix, sibsp_ix, parch_ix = 0, 1, 2

class FamilySize(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.feature_names_new = []
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = pd.DataFrame(X)
        X_["FamilySize"] = X_.iloc[:,sibsp_ix] + X_.iloc[:,parch_ix] + 1
        X_['Single'] = X_['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        #X_['SmallF'] = X_['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
        #X_['MedF'] = X_['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
        #X_['LargeF'] = X_['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
        #self.feature_names_new = ['FamilySize', 'Single', 'SmallF', 'MedF', 'LargeF']
        self.feature_names_new = ['FamilySize', 'Single']
        return X_.values
    
    def get_feature_names(self):
        return self.feature_names+self.feature_names_new

#from transformer_family import FamilySize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from transformer import OutlierRemover

## Neue Merkmale prüfen

features_numerical = ['Age', 'SibSp', 'Parch', 'Fare']
features_categorical = ['Pclass', 'Sex', 'Embarked']

pipe_numerical = Pipeline(
    steps=[
        ('outlier_remover', OutlierRemover(factor=3.0)),
        ('familysize', FamilySize(feature_names=features_numerical)),
        ('scaler', StandardScaler())
    ],
    verbose=True
)

pipe_categorical = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(drop='if_binary'))
    ],
    verbose=True
)

transformer_pipe = ColumnTransformer(
    transformers = [
        (
            'num', 
            pipe_numerical,
            features_numerical
        ),
        (
            'cat',
            pipe_categorical,
            features_categorical
        )
    ]
)

full_pipeline_fe1 = Pipeline(steps=[
    ('transformers', transformer_pipe),
    ('predictor', DecisionTreeClassifier(
        random_state=0,
        min_samples_split=5
    ))]
)

full_pipeline_fe1.fit(datasets['X_train'], datasets['y_train'])
full_pipeline_fe1.score(datasets['X_val'], datasets['y_val'])

transformer_pipe.transformers_[0][1]['familysize'].get_feature_names()

Ergebnis hat sich um 0.006 verbessert.

## Weitere Transformation hinzufügen

#%%writefile transformer_age.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AgeBinned(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = pd.DataFrame(X)
        bins = [0, 1, 16, 30, 64,100]
        labels = [1,2,3,4,5]
        X_[age_ix] = pd.cut(X_[age_ix], bins=bins, labels=labels)
        return X_.values

transformer_pipe.transformers_[0][1]['familysize'].get_feature_names()

features_numerical = ['Age', 'SibSp', 'Parch', 'Fare']
features_categorical = ['Pclass', 'Sex', 'Embarked']

pipe_numerical = Pipeline(
    steps=[
        ('outlier_remover', OutlierRemover(factor=3.0)),
        ('familysize', FamilySize(feature_names=features_numerical)),
        ('agebinned', AgeBinned()),
        ('scaler', StandardScaler())
    ],
    verbose=True
)

pipe_categorical = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(drop='if_binary'))
    ],
    verbose=True
)

transformer_pipe = ColumnTransformer(
    transformers = [
        (
            'num', 
            pipe_numerical,
            features_numerical
        ),
        (
            'cat',
            pipe_categorical,
            features_categorical
        )
    ]
)

full_pipeline_fe2 = Pipeline(steps=[
    ('transformers', transformer_pipe),
    ('predictor', DecisionTreeClassifier(
        random_state=0,
        min_samples_split=5
    ))]
)

full_pipeline_fe2.fit(datasets['X_train'], datasets['y_train'])
full_pipeline_fe2.score(datasets['X_val'], datasets['y_val'])

Das Ergebnis hat sich um 0.03 verbessert.

## Finaler Test mit dem Testdatenset

Ergebnis vor den Parameter- und Merklmalsoptimierungen auf dem Validierungsdatenset

full_pipeline.score(datasets['X_val'], datasets['y_val'])

Ergebnis nach den Parameter- und Merklmalsoptimierungen auf dem Validierungsdatenset

full_pipeline_fe2.score(datasets['X_val'], datasets['y_val'])

Ergebnis nach den Parameter- und Merkmalsoptimierungen auf dem Testdatenset

full_pipeline.score(datasets['X_test'], datasets['y_test'])

Ergebnis nach den Parameter- und Merkmalsoptimierungen auf dem Testdatenset

full_pipeline_fe2.score(datasets['X_test'], datasets['y_test'])

-> Ergebnis besser nach Optimierung  
-> Anhand des Testdatenset -> verallgemeinert besser

## Welche Merkmale sind entscheidend?

clf = full_pipeline_fe2.named_steps['predictor']

importances = clf.feature_importances_
importances

features_categorical_transformed = list(transformer_pipe.transformers_[1][1]['onehot'].get_feature_names(features_categorical))

features_numerical_transformed = transformer_pipe.transformers_[0][1]['familysize'].get_feature_names()

feature_names = features_numerical_transformed + features_categorical_transformed

feature_names

importances = pd.Series(importances, index=feature_names)
importances.sort_values(ascending=False)

# Pipeline speichern

with open('../output/titanic/pipeline.pkl', 'wb') as handle:
    pickle.dump(full_pipeline_fe2, handle)

In diesem Abschnitt haben Sie die Kernelemente von Scikit Learn kennengelernt, wie man eigene Transformer erstellt und anwendet, wie Pipelines erstellt und genutzt werden können und wie die Suche nach optimalen Parametern für alle Schritte der Pipeline automatisiert werden kann. Nun sind Sie gefragt: Festigen Sie ihr Wissen durch die Anwendung von Transformationen am Datenset "Bike Buyers".

[^footnote1]: "API design for machine learning software: experiences from the scikit-learn project", L Buitinck, G Louppe, M Blondel, et. al.

[^footnote2]: siehe https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

[^footnote3]: siehe https://scikit-learn.org/stable/modules/grid_search.html
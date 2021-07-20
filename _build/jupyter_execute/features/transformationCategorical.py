# Kategorische Daten

Die meisten Machine Learning Algorithmen können nur mit numerischen Daten umgehen. Bei Kategorischen Daten handelt es sich jedoch häufig um Text. Ein Merkmal "Tierart" enthält z.B. die Werte "Hund", "Maus" und "Katze". Ein Merkmal, das den Gemütszustand beschreibt, kann Werte wie z.B. "traurig", "neutral" und "glücklich" enthalten. In beiden Fällen ist eine **Umwandlung in numerische Daten** notwendig. Beim Merkmal "Tierart" handelt es sich um nominale kategorische Daten. Bedeutet, die Werte lassen sich nicht ordnen. Im Gegenteil dazu ist das Merkmal "Gemütszustand" ordinal. Die Werte lassen sich ordnen. Aufsteigend: traurig, neutral, glücklich. 


```{figure} ../images/nominalOrdinal.png
---
height: 200px
align: left
name: fig-nominalOrdinal
---
```

Bei einer Transformation der Merkmalswerte in numerische Daten muss diese Eigenschaft berücksichtigt werden. Im Folgenden werden die Transformationsmethoden für nominale und ordinale Daten behandelt.

import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle

## Nominale Daten

Die einfachste Transformation, von textbasierten zu numerischen kategorischen Merkmalen, ist das Ersetzen einer Kategorie mit einer bestimmten Zahl. Zum Beispiel für "Hund" die 1, für "Maus" die 2 und für "Katze" die 3. Einige ML-Algorithmen gehen jedoch davon aus, dass sich **zwei benachbarte Werte ähnlicher** sind als weiter entfernte Werte. Bei nominalen Daten trifft diese Annahme nicht zu. Es muss sicher gestellt werden, dass der Algorithmus hier **keine Fehlinterpretation** vornimmt. Lösung ist das sogenannte **One-Hot-Encoding**.  

### One-Hot-Encoding

Beim One-Hot-Encoding wird für **jeden möglichen Wert eine Spalte** mit binären Werten erstellt. 1 bedeutet, der Wert liegt vor, 0 bedeutet der Wert liegt nicht vor. Das Merkmal Tierart mit den drei möglichen Kategorien "Hund", "Maus" und "Katze" wird mit dem One-Hot-Encoding in drei Spalten transformiert. Für die Umsetzung bietet sich die [get_dummies()](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)-Methode von Pandas an.

Erstellen und Anzeigen der Beispieldaten:

data = {
    'Tierart': ['Hund', 'Maus', 'Maus', 'Hund', 'Katze', 'Hund'], 
    'Gemütszustand': ['glücklich', 'traurig', 'neutral', 'traurig', 'glücklich', 'glücklich']
}
df = pd.DataFrame(data)
df

One-Hot-Encoding des nominalen Merkmal "Tierart".

pd.get_dummies(df['Tierart'])

## Ordinale Daten

Für ordinale kategorische Daten kann die Transformation durch Ersetzen der Texte mit einer Zahl erfolgen. Eine Möglichkeit der Umsetzung ist der Aufruf der [factorize()](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html)-Methode von Pandas.

df['Gemütszustand'].factorize()

Die Liste der Kategorien gibt an wie das Mapping stattgefunden hat. "glücklich" wurde mit der Zahl 0 ersetzt, "traurig" mit der Zahl 1 und "neutral" mit der Zahl 2. Jetzt würden die meisten Machine Learning Algorithmen interpretieren dass "glücklich" ähnlicher "traurig" ist als "neutral". Die einfach anzuwendene Factorize-Methode kann funktionieren, muss aber nicht. **Kontrollieren** Sie nach der Anwendung ob die Zahlenwerte auch der **Reihenfolge der Ordnung** entsprechen. Alternativ kann die [replace()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html)-Methode verwendet und das Mapping **explizit** bestimmt werden.

mapping = {
    "glücklich": 0,
    "neutral": 1,
    "traurig": 2
}

df['Gemütszustand'].replace(mapping)

```{important}
Die Ordnung muss sich in den zugewiesenen Zahlen wiederfinden. 
````

## Transformation kategorischer Merkmale am Beispiel Titanic

Laden der Datensets aus Pickle File.

with open('../output/titanic/datasets_or.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

Die verbleibenden kategorischen Merkmale des Titanic Datensets sind die Ticketklasse (Pclass), das Geschlecht (Sex) und der Zustiegsort (Embarked). 

X_train = datasets['X_train'].copy()
X_train[['Pclass', 'Sex', 'Embarked']]

### Ticketklasse

Bei der Ticketklasse handelt es sich um ordinale Daten: 1 ist die höchste Klasse, 2 die mittlere und 3 die niedrigste Klasse. Über das Attribut dtype kann der Datentyp abgefragt werden. 

X_train['Pclass'].dtype

Es handelt sich bereits um Integer Werte. In diesem Fall muss keine Transformation vorgenommen werden.

### Gechlecht

Das Merkmal Geschlecht enthält nominale Kategorien. Es wird das One-Hot Encoding angewendet.

pd.get_dummies(X_train['Sex'])

### Zustiegsort

Der Zustiegsort enthält ebenso nominale Kategorien. Es wird das One-Hot Encoding angewendet.

pd.get_dummies(X_train['Embarked'])

Sie kennen nun die wichtigsten Methoden um numerische und kategorische Daten für Machine Learning Algorithmen aufzubereiten. Im nächsten Abschnitt wird gezeigt, wie die Transformationen angewendet werden. 


(missing-data)=
# Umgang mit fehlenden Daten

Die meisten Machine Learning-Algorithmen können nicht mit fehlenden Werten umgehen. Die Daten müssen daher auf fehlende Daten geprüft und modifiziert werden. Möglicher Umgang mit fehlenden Werten:
1. Datensätze entfernen
2. Ersetzen durch Median/Mittelwert/Mode
3. Eine eigene Kategorie zuweisen
4. Fehlende Werte schätzen

Die Ansätze werden anhand des Titanic-Datensets erläutert.

Importieren der verwendeten Python Module

import pandas as pd
import numpy as np

Laden der Daten

data = pd.read_csv("../data/titanic/train.csv")
data.head()

## 1. Datensätze entfernen
Diese Methode wird sehr häufig verwendet, wenn eine ausreichende Menge an Daten vorliegen. Zeilen, die fehlende Werte enthalten werden entfernt. 

data_tmp = data.dropna()
data_tmp.isnull().sum()

## 2. Ersetzen durch Mittelwert/Median/Mode

Kann angewendet werden, wenn es sich um ein Merkmal mit numerischen Daten handelt. Der Mittelwert, Median oder Mode des Merkmals wird berechnet und die fehlenden Werte mit dem berechneten Wert ersetzt. Es handelt sich bei dieser Methode um eine Annäherung mittels einem statistischen Ansatz. Eine weiter Möglichkeit ist die berechnung des Durchschnitts der Nachbarwerte. Dieses Vorgehen funktioniert besser, wenn die Daten linear sind. 

Beim Mode-Wert handelt es sich um den am häufigsten vorkommenden Wert. Der Median, ist das 50%-Quantil und sollte verwendet werden, wenn die Verteilung der Daten rechts- oder linksschief ist.


data['Age'].head(10)

data['Age'].mean()

data['Age'].replace(np.NaN, data['Age'].mean()).head(10)

Alternativ: Berechnung des Median oder Mode-Wert

data['Age'].median()

data['Age'].mode()

## 3. Eine eigene Kategorie zuweisen

Diese Methode kann bei kategorischen Daten angewendet werden. Für die fehlenden Werte wird eine eigene Kategorie erstellt wie z.B. "Unbekannt" und die fehlenden Werte mit dieser Bezeichnung ersetzt.

data['Cabin'].head(10)

data['Cabin'].fillna('Unbekannt').head(10)

## 4. Die fehlenden Werte schätzen

Unter Verwendung der Merkmale, bei denen keine fehlenden Werte vorkommen, können die fehlenden Werte mit Hilfe von Machine Learning Algorithmen geschätzt werden. Im Folgenden ein Beispiel mit Linearer Regression die fehlenden Werte des Merkmals Alter zu ersetzten:

from sklearn.linear_model import LinearRegression
model = LinearRegression()

data_uncleaned = data[['PassengerId', 'Pclass', 'Survived', 'SibSp', 'Parch','Fare', 'Age']].dropna()
data_cleaned = data_uncleaned.dropna()

# Alle Merkmale bis auf das Alter
train_x = data_cleaned.iloc[:,:6]

# Nur das Alter
train_y = data_cleaned.iloc[:,6]

# Trainieren mit den vorhandenen Daten
model.fit(train_x, train_y)

# 
test_x = data_uncleaned.iloc[:,:6]
age_predicted= pd.DataFrame(model.predict(test_x))

# nur die fehlenden Werte ersetzen
data_uncleaned.Age.fillna(age_predicted.iloc[0], inplace=True)

data_uncleaned['Age'].head(10)

## Welche Methode sollte verwendet werden?

Finden Sie heraus warum die Daten fehlen:

Häufig erhalten wir die Daten und haben diese nicht selbst aufgezeichnet. Wir können daher nicht sicher sagen, warum die Daten fehlen. Meist lässt sich dies schätzen. Stellen Sie sich die Frage  

**"Fehlt der Wert weil er nicht aufgezeichnet wurde oder weil er nicht existiert"?**

Wenn ein Wert fehlt, weil er nicht existiert macht es keinen Sinn den Wert zu schätzen. In diesem Fall ist es besser den Datensatz (die Zeile) zu verwerfen oder die Lücke mit einem "Nicht vorhanden" (NaN)-Wert zu belegen. 

Fehlt ein Wert weil er nicht aufgezeichnet wurde, macht es Sinn diesen zu schätzen. Entweder durch statistische Analyse der restlichen Werte in der Spalte oder durch Machine Learning Algorithmen unter Verwendung der anderen Spalten.

## Anwendung am Bespiel Datenset Titanic

Ausgabe der Anzahl fehlender Werte pro Merkmal:

missingValuesCount = data.isnull().sum()
total = missingValuesCount.sort_values(ascending=False)
percent = (missingValuesCount/data.isnull().count()*100).sort_values(ascending=False)
missingData = pd.concat([total, percent], axis=1, keys=['Gesamt', 'Prozent'])
missingData

Beobachtungen:
* Der Großteil (77,1%) der Kabinen-Werte (Cabin) fehlen, entspricht 687 Datensätze
* 19,9% der Altersangaben fehlen, entspricht 177 Datensätze
* 0,2% der Zustiegsorte (Embarked) fehlen, entspricht nur 2 Datensätze

**Zustiegsorte (Embarked)**  
Jeder der Passagiere hatte einen Zustiegsort. Bedeutet, die fehlenden Werte existieren, wurden jedoch nicht erfasst.In diesem Fall macht es Sinn die Daten zu schätzen. 

Da es sich bei den fehlenden Zustiegsorten lediglich um 2 Datensätze handelt, können diese einfach mit den häufigsten Werten aufgefüllt werden. 

data['Embarked'].describe()

data['Embarked'].mode()[0]

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

**Alter (Age)**  
Jeder Passagier hat ein Alter. Die fehlenden Werte existieren, wurden jedoch nicht erfasst. Eine Schätzung macht auch in diesem Fall Sinn.

Das fehlende Alter kann zunächst mit Mittelwert oder Median ersetzt werden. Insbesondere wenn viele Ausreißer enthalten sind oder die Verteilung rechts- oder linksschief ist, eignet sich der Median. Sieht man sich die Verteilung der Daten an (siehe Diagramm unten), zeigt sich eine rechtsschiefe Verteilung. In diesem Fall eignet sich der Median besser als der Mittelwert. 

data['Age'].hist()

data['Age'].median()

data['Age'].fillna(data['Age'].median(), inplace=True)

data.isnull().sum()

**Kabine (Cabin)**

data

Aufgrund der sehr hohen Anzahl an fehlendenen Kabinen-Werten (77,1%) und der Annahme, dass der Kabinenname vermutlich wenig darüber aussagt, ob ein Passagier überlebt hat, wird die Spalte Kabine aus den Daten entfernt.


drop_column = ['Cabin']
data.drop(drop_column, axis=1, inplace = True)

#### Entfernen von nicht relevanten Merkmalen

Bei den Merkmalen PassengerId, Name und Ticket wird angenommen, das es sich um zufällige eindeutige Identifikatoren handelt, die keinen Einfluss auf die Ergebnisvariable haben. Daher werden sie von der Analyse ausgeschlossen.

drop_column = ['PassengerId','Ticket','Name']
data.drop(drop_column, axis=1, inplace = True)

data

## Speichern der aufbereiteten Daten

data.to_pickle('../output/preparedData.pkl')

Das Pickle-Format kann wie folgt eingelesen werden:

df = pd.read_pickle('../output/preparedData.pkl')

df


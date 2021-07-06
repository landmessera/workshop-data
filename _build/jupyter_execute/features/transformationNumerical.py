# Numerische Daten

# Packete importieren
import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# Settings
#colors = ["#D0755E", "#B35097", "#68349A", "#E54E78", "#F66F6F"]

# Laden der Datensets aus Pickle File
with open('../output/titanic/datasets.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

## Ausreißer erkennen

```{admonition} Was sind Ausreißer?
Ausreißer sind Daten die sich stark von den anderen Daten unterscheiden. Das Vorkommen ist selten und hebt sich in irgendeiner Art von den anderen Daten ab. 
```

# Create sample data
from sklearn.ensemble import IsolationForest
X, y_true = make_blobs(n_samples=500, centers=1, cluster_std=3.20, random_state=5)
X_append, y_true_append = make_blobs(n_samples=100,centers=1, cluster_std=8,random_state=5)
X = np.vstack([X,X_append])
y_true = np.hstack([y_true, [1 for _ in y_true_append]])
X = X[:, ::-1] 

# Detect Outlier
sns.set_style("whitegrid", {'axes.grid' : False})
preds = IsolationForest(random_state=0).fit_predict(X)
list(preds).count(-1)
df = pd.DataFrame(preds, columns=['outlier'])
df['x'] = X[:,0]
df['y'] = X[:,1]
df['tmp'] = X[:,1].shape[0]*[1]

#customPalette = sns.set_palette(sns.color_palette(colors,n_colors=5))
d = {
    1: "#D0755E",
    -1: "#68349A"
}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.scatterplot(data=df, x="x", y="y", hue="tmp", palette=d, ax=axes[0], legend=False)
sns.scatterplot(data=df, x="x", y="y", hue="outlier", palette=d, ax=axes[1], legend=False)

fig.tight_layout()
plt.show()

In der linken Grafik sind eine Menge von Daten dargestellt. Jeder Punkt stellt ein Datensatz dar, der aus einem x- und einem y-Wert besteht. Auf der rechten Seite wurde eine Ausreißererkennung durchgeführt. Ausreißer sind mit lilaner Farbe gekennzeichnet.

Ursachen für Ausreißer:
* Eingabefehler: Wenn z.B. Menschen Fehler bei der Eingabe von Hand gemacht haben.
* Messfehler: Beim Aufzeichnen von Daten sind Fehler entstanden.
* Beschädigung: Nachdem die Daten erfasst wurden, sind Fehler aufgetreteten, zum Beispiel während der Verarbeitung.
* Natürlich: In diesem Fall handelt sich um keinen Fehler.

**Warum möchte man Ausreißer finden?**   
Im Machine Learning Kontext können Ausreißer das Training eines Modells negativ beeinflussen. Dies kann sich in einem verzerrten Modell äußern. Infolge dessen liefert das Modell nicht die gewünschten Ergebnisse. Nicht alle Ausreißer bringen eine negative Auswirkung mit sich. Wenn die Ursache der Ausreißer natürlich ist, also kein Fehler, spiegeln die Ausreißer die Realität wieder und sollten nicht entfernt werden. Bei Anwendungsfällen wie z.B. der Katzenklappe, sind Momente, in denen eine Katze mit Beute vor der Klappe steht wesentlich seltener als Momente, in denen die Katze ohne Beute oder keine Katze vor der Klappe sichtbar ist. Die Zustände "Katze mit Beute" könnten als Ausreißer erkannt werden. In diesem Fall sind jedoch gerade diese Ausreißer von Bedeutung. Der Zustand "Katze mit Beute" ist kein Fehler und sollte nicht aus den Daten entfernt werden.

```{tip}
Stellen Sie sich die Frage:
**Sind die Outlier durch Fehler begründet?**   
Ja -> Ausreißer entfernen  
Nein -> Ausreißer im Datenset belassen  
```

Beispiel: Das Merkmal Alter im Titanic Datenset  
Für das Merkmal Alter eines Menschen erwartet man einen Wert zwischen 0 und 122. Diesen Wertebereich kennt man aufgrund von "Fachwissen": Der älteste Mensch wurde bisher 122 Jahre alt. Außerdem kann man den den Wertebereich weiter eingrenzen, wenn man annimmt, dass es sehr unwahrscheinlich ist, dass ein Mensch der älter als 100 Jahre ist, eine Reise über den Atlantik antritt. Alle Werte die außerhalb des Wertebereichs von 0 und 100 liegen, bezeichnet man in diesem Fall als Fehler.

Die Verteilung und Ausreißer von numerischen Daten lassen sich sehr gut aus einem Boxplot ablesen. Ein Boxplot ist eine Methode zur grafischen Darstellung numerischen Daten durch ihre Quartile. Die Box erstreckt sich von den Q1- bis Q3-Quartilwerten der Daten, mit einer Linie am Median (Q2). Die Whisker erstrecken sich von den Rändern der Box, um den Bereich der Daten anzuzeigen. Standardmäßig erstrecken sie sich nicht weiter als 1,5 * IQR (IQR = Q3 - Q1) von den Rändern der Box und enden beim am weitesten entfernten Datenpunkt innerhalb dieses Intervalls. Ausreißer werden als separate Punkte gezeichnet.

todo: Bild mit Boxplot erklärung

vgl: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html

Die Boxplots der numerischen Merkmale des Datensets Titanic, vermitteln einen ersten Eindruck der Verteilung der Daten.

# define numeric features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']

# get train dataset and select numeric features
X_train = datasets['X_train'].copy()

# Boxplot of numerical features
X_train[numeric_features].plot(kind="box",subplots=True,figsize=(15,5),title="Boxplots der numerischen Merkmale");

Boxplots verwenden in der Standardeinstellung wie wir sie in der Abbildung sehen die Interquartile Range (IQR) - Methode für die Ausreißererkennung. Die IQR-Methode funktioniert wie folgt:
1. Das 25%-Quantil (Q1) berechnen
2. Das 75%-Quantil (Q3) berechnen
3. Q1 von Q3 subtrahieren (ergibt die Höhe der Box im Boxplot) = iqr
4. Die untere Grenze berechnen durch Q1 - (Faktor * iqr)
5. Die obere Grenze berechnen durch Q3 - (Faktor * iqr)

Wobei der Faktor variabel ist. Die Standardeinstellung beträgt 1,5.

Mit folgenden Codezeilen lassen sich die Ausreißer von den Daten des Merkmals "Alter" erkennen und anzeigen:

feature = 'Age'
factor = 1.5

s = pd.Series(X_train[feature]).copy()
q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (factor * iqr)
upper_bound = q3 + (factor * iqr)
print('Q1-Quantil: ',q1)
print('Q3-Quantil: ',q3)
print('lower_bound: ',lower_bound)
print('upper_bound: ',upper_bound)

outliers = s.loc[((s < lower_bound) | (s > upper_bound))].copy()
print('Anzahl der Ausreißer: ',outliers.shape[0])
print('Ausreißer')
outliers.sort_values()

Schaut man sich die Werte der 37 Ausreißer an, stellt man fest, dass es sich ausschließlich um valide Werte handelt. Jeder Wert befindet sich innerhalb unseres erwarteten Wertebereichs zwischen 0 und 100. Die Ausreißer sind also natürlicher Art. Es gibt wenige Menschen an Bord der Titanic, die jünger als 2,5 Jahre sind und wenige die älter als 54,5 Jahre sind. Die Werte der Ausreißer sind jedoch realistisch und sollten nicht entfernt werden.

**Wie verhält es sich mit den Ausreißern der restlichen numerischen Merkmale?**  
Das Merkmal **SibSp** enthält die Anzahl der Geschwister oder Ehepartner, die mit an Bord der Titanic sind. Die Werte der Ausreißer betragen 3,4,5 und 8. 

Beim Merkmal **Parch** handelt es sich um die Anzahl der Eltern oder Kinder, die mit an Bord sind. Die Werte der Ausreißer betragen 1-6. 

Der Wertebereich der Ausreißer des **Ticketpreis** beträgt 65 - 512. Der höchste Wert 512 liegt weit über dem zweithöchsten Wert von 263. Der einzelne sehr hohe Betrag von 512 könnte der Preis für eine Luxuskabine sein.

```{admonition} Umgang mit Ausreißer im Titanic Datenset
Es handelt sich bei allen Ausreißern der numerischen Merkmalen um realistische Werte und sollten daher nicht entfernt werden.
````

Häufig verwendete Methoden zum Entfernen von Outlier:
* IQR-Methode
* DBScan Clustering
* Isolation Forest

Die Isolation Forest Methode und das DBScan Clustering können angewendet werden, wenn die Ausreißererkunng auf mehrdimensionalen Daten durchgeführt werden soll. Im Fall des Titanic Datenset werden die einzelnen Merkmale getrennt betrachtet, daher handelt es sich nicht um multivariate sondern um univariate Daten.

## Skalierung

Die Skalierung ist eine der wichtigsten Transformationen. Die meisten Machine Learning Algorithmen funktionieren nicht gut, wenn sich die Wertebereiche der Merkmale unterscheiden. Das ist bei den numerischen Merkmalen des Titanic Datenset der Fall: 
* Das Alter der Passagiere liegt zwischen 0,67 und 80
* Die Anzahl der Geschwister oder Ehepartner reicht von 0 bis 8

Um die numerischen Merkmale auf die gleiche Skala zu bringen gibt es zwei übliche Verfahren:
1. Min-Max-Skalierung
2. Standardisierung

### Min-Max-Skalierung
Die Min-Max-Skalierung, auch Normalisierung genannt, ist eine Verschiebung der Werte in den Wertebereich von 0 und 1. Hierzu wird zuerst der Minimalwert von allen Werten subtrahiert und anschließend alle Werte durch die Differenz von Maximal- und Minimalwert geteilt. 

Erzeugen von Beispieldaten, wobei der Minimalwert 10 und der Maximalwert 19 beträgt:

x = np.arange(10,20)
x

Subtrahieren des Minimalwertes und teilen durch die Differenz von Maximal- und Minimalwert:

x_scaled = (x - x.min())/(x.max()-x.min())
x_scaled

Ausgabe der neuen Minimal- und Maximalwerte

x_scaled.min()

x_scaled.max()

#### Min-Max Skalierung am Beispiel Titanic 

In der folgenden Abbildung sieht man die Werte des Merkmals "Alter" in einem Violin- und Swarmplot dargestellt. Ein Violinplot ist ähnlich wie ein Boxplot und stellt die Verteilung der Daten dar. Im Vergleich zum Boxplot wird die Verteilung nicht mit den Median-Werten dargestellt, sondern mit einem gespiegelten Kernel-Dichtediagramm. Der Swarmplot zeigt alle Datenpunkte einer Datenmenge. Daten mit dem gleichen Wert, werden nicht überlappt dargestellt sondern so gestapelt, dass jeder Datenpunkt sichtbar ist. Der Wertebereich liegt zwischen 0 und 80.

ax = sns.violinplot(data=X_train, x="Age", inner=None, color='#D0755E')
ax = sns.swarmplot(data=X_train, x="Age", size=1.8, orient="h", color="white")
ax.set(ylabel="")

Scikit-learn bietet einen MinMaxScaler an, mit dem sich die Min-Max-Skalierung einfach auf die Merkmale in einem Pandas Dataframe anwenden lassen. Als Rückgabe erhält man ein numpy-Array mit den skalierten Werten. Die Anwendung:
* MinMaxScaler importieren
* MinMaxScaler-Objekt erstellen
* der fit_transform Methode die Daten übergeben


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
df_scaled = pd.DataFrame(X_train_scaled, columns=X_train[numeric_features].columns)

Violoin- und Swarmplot zeigt, dass die Verteilung der sklalierten Werte gleich geblieben ist. Lediglich der Wertebereich hat sich auf 0 bis 1 geändert.

ax = sns.violinplot(data=df_scaled, x="Age", inner=None, color='#D0755E')
ax = sns.swarmplot(data=df_scaled, x="Age", size=1.8, orient="h", color="white")
ax.set(ylabel="")

### Standardisierung

Bei der Standardisierung wird zuerst der Mittelwert abgezogen. Das bewirkt, dass der Mittelwert der neuen Werte Null beträgt. In einem zweiten Schritt wird durch die Varianz geteilt. Daraus folgt eine Varianz von 1 bei den neuen Werten. 

Erzeugen von Beispieldaten, wobei der Minimalwert 10 beträgt und der Maximalwert 19:

x = np.arange(10,20)
x

Skalierung durch Standardisierung
* Mittelwert bestimmen
* Standardabweichung bestimmen
* Mittelwert subtrahieren und durch die Standardabweichung teilen

Mittelwert:

x.mean()

Standardabweichung:

x.std()

Mittelwert subtrahiereun und durch die Standardabweichung teilen

x_scaled = (x - x.mean())/x.std()
x_scaled

Neuer Mittelwert:

round(x_scaled.mean(),2)

Neue Standardabweichung:

round(x_scaled.std(),2)

#### Standardisierung am Beispiel Titanic

Die StandardScaler-Methode von Scikit-Learn kann auf einen Pandas Dataframe angewendet werden. Als Rückgabe erhält man ein numpy-Array mit den skalierten Werten. Die Anwendung:
* StandardScaler importieren
* StandardScaler-Objekt erstellen
* fit_transform Methode die Daten übergeben

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
df_scaled = pd.DataFrame(X_train_scaled, columns=X_train[numeric_features].columns)

ax = sns.violinplot(data=df_scaled, x="Age", inner=None, color='#D0755E')
ax = sns.swarmplot(data=df_scaled, x="Age", size=1.8, orient="h", color="white")
ax.set(ylabel="")

Die Werte des Merkmals "Alter" wurden auf einen Wertebereich von -2,24 bis 3.92 skaliert. Der Mittelwert beträgt 0 und die Standardabweichung 1.

Das [Preprocessing Modul](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) der Scikit-learn Bibliothek bietet eine Reihe weiterere Transformationen an. Die Verwendung ist meist sehr ähnlich. Im wesentlichen sind lediglich die drei Schritte "Importieren", "Objekt erzeugen" und "fit_transform-Methode aufrufen" notwendig um die Preprocessing-Methoden anzuwenden.

### Standardisierung oder Min-Max-Skalierung?

Die Standardabweichung hat im Vergleich zur Min-Max-Skalierung den Vorteil, dass Sie besser mit Ausreißer umgehen kann. Angenommen von 1000 Passagieren ist ein Passagier 110 Jahre alt, der Rest zwischen 20 und 30. Bei der Min-Max-Skalierung werden alle Werte zwischen 20 und 30 in den Bereich von 0-0.1 gequetscht. Bei der Standardisierung ändert sich durch den variablen Wertebereich nicht viel. Einige Algorithmen können nur mit Werten zwischen 0 und 1 umgehen, wie z.B. Neuronale Netze. In diesem Fall ist die Min-Max-Skalierung vorteilhaft.

```{Note}
Die meisten Machine Learning Verfahren funktionieren besser , wenn eine Skalierung der Daten vorgenommen wird. Welches Skalierungsverfahren eingesetzt werden sollte hängt von den Daten und den Anforderungen des Algorithmus ab.
```

## Diskretisierung

Die Diskretisierung, auch Binning genannt, transformiert kontinuierliche in diskrete Werte. Hierzu wird ein Set von zusammenhängenden Intervallen definiert, sogenannte Bins (Behälter). Jeder Behälter erhält einen Namen oder eine Regel wie die Elemente dieser Behälter benannt werden. Das kann zum Beispiel der Mittelwert der enthaltenen Elemente sein. Anschließend wird jeder kontinuierliche Wert einem der Behälter zugewiesen und mit der Bezeichnung ersetzt.

Erstellen von Beispieldaten:

x = [2, 6, 7, 12, 15, 18, 19, 34, 35, 40, 50]

### Diskretisierung in diskrete numerische Daten

Bestimmen von Bins:
* kleiner als 10
* 10-30
* größer als 30

Bezeichnung: Mittelwert der zugewiesenen Elemente

Umsetzung:

import statistics

bin1 = []
bin2 = []
bin3 = []
x_discretized = []

for value in x:
    if value < 10:
        bin1.append(value)
    elif (10 <= value <= 30):
        bin2.append(value)
    else:
        bin3.append(value)
        
print('Bins: ', bin1, bin2, bin3)
print('Mittelwert der Elemente der Bins: ', 
      statistics.mean(bin1), 
      ',',
      statistics.mean(bin2), 
      ',',
      statistics.mean(bin3)
)

for value in x:
    if value < 10:
        x_discretized.append(statistics.mean(bin1))
    elif (10 <= value <= 30):
        x_discretized.append(statistics.mean(bin2))
    else:
        x_discretized.append(statistics.mean(bin3))
        
print('Diskretisierte Werte: ', x_discretized)

### Diskretisierung in kategorische Daten

Bestimmen von Bins und Bezeichnung:
* kleiner als 10: 'niedrig'
* 10-30: 'mittel'
* größer als 30 : 'hoch'

Umsetzung:

x_discretized = []
for value in x:
    if value < 10:
        x_discretized.append('niedrig')
    elif (10 <= value <= 30):
        x_discretized.append('mittel')
    else:
        x_discretized.append('hoch')
        
x_discretized

Es gibt verschiedene Arten um die Bins zu definieren. Über
1. die gleiche Breite: z.B. Werte von 0-10, 10-20, 30-40 usw.
2. die gleiche Anzahl von Elementen: z.B. Quantile
3. ein Clustering: z.B. Durchführen eines kmeans-Clustering
 
Die Scikit-Learn Bibliothek liefert eine Methode namens KBinsDiscretizer, die eine Diskretisierung durchführt. Über den Parameter "strategy" bestimmt man die Art wie die Bins erstellt werden. Mögliche Werte:
* "uniform": 1. Art, gleiche Breite
* "quantile": 2. Art, gleiche Anzahl
* "kmeans": 3. Art, Clustering


### Diskretisierung am Beispiel Titanic

Die Diskretisierung wird am Merkmal "Fare", dem Ticketpreis veranschaulicht. Mit dem Violin- und Swarmplot erhält man einen ersten Eindruck von den Daten.

ax = sns.violinplot(data=X_train, x="Fare", inner=None, color='#68349A')
ax = sns.swarmplot(data=X_train, x="Fare", size=1, orient="h", color="white")
ax.set(ylabel="")

Der Violinplot zeigt, dass die Verteilung der Daten rechtsschief ist.

#### Diskretisierung mit gleicher Breite der Bins

from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')

X_train_fare_scaled = discretizer.fit_transform(X_train[['Fare']].values)
df_scaled = pd.DataFrame(X_train_fare_scaled, columns=X_train[['Fare']].columns)

Ausgabe der letzten fünf Zeilen der skalierten Werte:

df_scaled.tail()

Bezeichnung der Bins:

df_scaled['Fare'].unique()

Anzahl der Elemente pro Bin:

df_scaled['Fare'].value_counts().plot(kind='bar')

#### Diskretisierung mit gleicher Anzahl Elemente in Bins

from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

X_train_fare_scaled = discretizer.fit_transform(X_train[['Fare']].values)
df_scaled = pd.DataFrame(X_train_fare_scaled, columns=X_train[['Fare']].columns)

df_scaled['Fare'].value_counts().plot(kind='bar')

#### Diskretisierung mit Clustering

from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')

X_train_fare_scaled = discretizer.fit_transform(X_train[['Fare']].values)
df_scaled = pd.DataFrame(X_train_fare_scaled, columns=X_train[['Fare']].columns)
df_scaled

df_scaled['Fare'].value_counts().plot(kind='bar')

Bei der Anwendung einer Diskretisierung ist zu beachten, dass Informationen verloren gehen. Das kann von Vorteil sein, wenn die verlorenen Informationen für den Anwendungsfall nicht relevant sind. Außerdem kann eine Diskretisierung eine vereinfachte Interpretation der Daten ermöglichen. Bewirkt eine Diskretisierung jedoch einen Verlust von relevanten Informationen, wirkt sich das negativ auf das Endergebnis aus.  

Mit Domänenwissen kann zunächst abgeschätzt werden, ob eine Diskretisierung erfolgsversprechend ist. Am Ende muss jedoch immer eine Erprobung stattfinden um sicher gehen zu können, dass eine Diskretisierung die Performance des Algorithmus steigert. Hierzu führt man Experimente mit unterschiedlichen Transformationsmöglichkeiten durch und analysiert das Ergebnis. Mehr dazu im Abschnitt "Feature Engineering". 

Im nächsten Abschnitt werden Transformationen für kategorische Daten behandelt.
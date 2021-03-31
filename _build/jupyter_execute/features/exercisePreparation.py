# Übung: Aufbereitung

## Überblick verschaffen

### Python Packete importieren und Daten laden

Python Packete importieren

import pandas as pd
import numpy as np

Daten laden

# read in all our data
data = pd.read_csv("../data/bikebuyers/bike_buyers.csv")

### Anzeigen der ersten Datensätze

```{note}
Die [head-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) liefert die ersten Zeilen eines Pandas Datenframe.

```

# Enter your code here

Lösung: Anzeigen der ersten Datensätze

data.head(7)

Beobachtungen aus den ersten Datensätzen
* <mark> [?] </mark>
* <mark> [?] </mark>
* <mark> [?] </mark>

```{Dropdown} Lösung: Beobachtungen aus den ersten Datensätzen
* Die Merkmale Gender und Home Owner enthalten fehlende Wert
* Die meisten Merkmale enthalten nicht-numerische Daten
* Die numerischen Merkmale unterscheiden sich stark in den Wertebereichen
```

# Geben Sie hier den Code ein, 
# um die Informationen über die Daten zu erhalten.

Das Datenset besteht aus  
<mark> [?] </mark> Datensätzen  
<mark> [?] </mark> Merkmalen  
  
Die Zielvariable ist: <mark> [?] </mark>  
      
<mark> [?] </mark> Merkmale sind vom Typ Float  
<mark> [?] </mark> Merkmale sind vom Typ Integer  
<mark> [?] </mark> Merkmale sind Objekte  

```{note}
Die [Info-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) liefert eine Zusammenfassung des Pandas Datenframe.

```

```{Dropdown} Lösung  
Das Datenset besteht aus   
1000 Datensätzen  
12 Merkmalen  
  
Die Zielvariable ist: Purchased Bike   
  
4 Merkmale sind vom Typ Float  
1 Merkmal vom Typ Integer  
7 Merkmale sind Objekte  
```

data.info()

#### Beschreibung der Merkmale:
* **ID**: Eindeutige Id des Kunden
* **Marital Status**: Familienstand [Verheirated, Single]
* **Income**: Jahreseinkommen 
* **Children**: Anzahl an Kinder
* **Education**: Bildungsstand  
[Bachelor, Partial College, High School, Graduate Degree, Partial High School]
* **Occupation**: Berufsstand  
[Professional, Skilled Manual, Clerical, Management, Manual]
* **Home Owner**: Hausbesitzer  [True, False]
* **Cars**: Anzahl an Autos
* **Commute Distance**: Pendeldistanz
* **Region**: Region [North America, Europe, Pacific]
* **Age**: Alter 
* **Fahrrad gekauft** [True, False]

### Datentypen bestimmen

Numerische Daten:
* Diskret
    * <mark> [?] </mark>
    * <mark> [...] </mark>
* Kontinuierlich
    * <mark> [?] </mark>
    * <mark> [...] </mark>
    
Kategorische Daten:
* Nominal
    * <mark> [?] </mark>
    * <mark> [...] </mark>
* Ordinal
    * <mark> [?] </mark>
    * <mark> [...] </mark>

```{Dropdown} Lösung: Datentypen bestimmen
Numerische Daten:
* Diskret
    * Cars
    * Children
* Kontinuierlich
    * Income
    
Kategorische Daten:
* Nominal
    * ID
    * Marital Status
    * Gender
    * Income
    * Occupation
    * Home Owner
* Ordinal
    * Education
    * Commute Distance
```

### Deskriptive Statistiken

```{note}
Die [Describe-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)  liefert deskriptive Statistiken des Pandas Datenframe.

```

# Enter your code here

Lösung

data.describe()

#### Beobachtungen

* Das Einkommen der meisten Kunden (75%) liegt unter <mark> [?] </mark>.
* Die Standardabweichung des Einkommen beträgt <mark> [?] </mark>.
* 50% der Kunden haben mehr als <mark> [?] </mark> Kinder.
* Die Kunden besitzen im Durchschnitt <mark> [?] </mark> Autos.
* Das Alter der Kunden variiert zwischen <mark> [?] </mark> und <mark> [?] </mark>.
* 25% der Kunden sind älter als <mark> [?] </mark>.

```{Dropdown} Lösung: Beobachtungen
* Das Einkommen der meisten Kunden (75%) liegt unter <mark> [?] </mark>.
* Die Standardabweichung des Einkommen beträgt 31067,82.
* 50% der Kunden haben mehr als <mark> [?] </mark> Kinder.
* Die Kunden besitzen im Durchschnitt <mark> [?] </mark> Autos.
* Das Alter der Kunden variiert zwischen 25 und 89.
* 25% der Kunden sind älter als <mark> [?] </mark>.
````

## Fehlende Werte korrigieren

Geben Sie die absoulte Zahl der fehlenden Werte pro Spalte aus.

```{note}
Die [isnull-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.isnull.html)  liefert für jeden Wert des Pandas Datenframe einen Boolean-Wert, true oder false. True bedeutet, es handelt sich um einen fehlenden Wert.

Die [sum-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html) liefert die Anzahl der True Werte pro Spalte (bei Default-Einstellung)

```

# Enter your code here

Lösung

data.isnull()

missingValuesCount = data.isnull().sum()
missingValuesCount

Geben Sie den absoluten und prozent-Wert pro Spalte an.

# Enter your code here

Lösung

total = missingValuesCount.sort_values(ascending=False)
percent = (missingValuesCount/data.isnull().count()*100).sort_values(ascending=False)
missingData = pd.concat([total, percent], axis=1, keys=['Gesamt', 'Prozent'])
missingData

Notieren Sie Ihre Beobachtungen:

Beobachtungen:
* <mark> [?] </mark>
* <mark> [?] </mark>

```{Dropdown} Lösung: Beobachtungen
* Die fehlenden Werte pro Spalte sind sehr gering und liegen zwischen 0.4 und 1.1 Prozent.
* Alle fehlende Werte (Cars, Age, Children, Marital Status, Income, Home Owner) existieren vermutlich, wurden jedoch nicht erfasst.
```

### Fehlende Werte ersetzen/entfernen

Entscheiden Sie sich bei jedem Merkmal für eine Methode und wenden diese auf die Datensätze, welche fehlende Werte enthalten an.

```{note}
Die [hist-Methode] (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html)  kann auf eine Spalte des Datenframes angewendet werden um die Verteilung der Daten zu erhalten.

```

#### Gender

# Enter your code here

Beim Gender Merkmal handelt es sich um kategorische Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen.

data['Gender'].hist()

data['Gender'].mode()[0]

data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

#### Cars

# Enter your code here

Beim Cars Merkmal handelt es sich um numerisch diskrete Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen.

data['Cars'].hist()

data['Cars'].mode()[0]

data['Cars'].fillna(data['Cars'].mode()[0], inplace=True)

#### Age

# Enter your code here

```{Dropdown} Lösung: 
Beim Age Merkmal handelt es sich um numerische Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte zu ersetzen. Aufgrund der rechtsschiefen Verteilung der Daten bietet sich der Median als Ersatzwert an.
```

data['Age'].hist()

data['Age'].median()

data['Age'].fillna(data['Age'].median(), inplace=True)

#### Children

# Enter your code here

```{Dropdown} Lösung: 
Beim Merkmal Children handelt es sich um diskrete Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 
```

data['Children'].hist()

data['Children'].mode()[0]

data['Children'].fillna(data['Children'].mode()[0], inplace=True)

#### Marital Status

# Enter your code here

```{Dropdown} Lösung: 
Beim Merkmal Children handelt es sich um nominale Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 
```

data['Marital Status'].hist()

data['Marital Status'].mode()[0]

data['Marital Status'].fillna(data['Marital Status'].mode()[0], inplace=True)

#### Income

# Enter your code here

```{Dropdown} Lösung: 
Beim Merkmal Income handelt es sich um kontinuierliche Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem Mittelwert zu ersetzen. 
```

data['Income'].hist()

data['Income'].mean()

data['Income'].fillna(data['Income'].mean(), inplace=True)

#### Home Owner

# Enter your code here

```{Dropdown} Lösung: 
Beim Merkmal Owner handelt es sich um nominale Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 
```

data['Home Owner'].hist()

data['Home Owner'].fillna(data['Home Owner'].mode()[0], inplace=True)

#### Überprüfen der fehlenden Werte

Geben Sie erneut die Anzahl der fehlenden Werte aus. Es sollten jetzt keine fehlende Werte existieren.

# Enter your code here

missingValuesCount = data.isnull().sum()
missingValuesCount

### Ergebnis speichern

Speichern Sie die aufbereiteten Daten als Pickle-Datei ab.

```{Note}
Die [to_pickle-Methode](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html) ermöglicht eine einfache Speicherung des Pandas Datenframe im .pkl-Format
```

# Enter your code here

data.to_pickle('../output/preparedData.pkl')
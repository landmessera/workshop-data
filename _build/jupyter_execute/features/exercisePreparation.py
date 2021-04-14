# Übung: Aufbereitung

(start)=
## Überblick verschaffen

### Python Packete importieren und Daten laden

Python Packete importieren

# Packete importieren
import pandas as pd
import numpy as np

# Daten von CSV in Pandas Dataframe laden
data = pd.read_csv("../data/bikebuyers/bike_buyers.csv")
print(str(data.shape[0]) +' Datensätze geladen')

### Task 1: Anzeigen der ersten Datensätze

In der Variable data liegen die Daten in Form eines Panda Dataframe vor. Geben Sie die ersten 7 Datensätze dieses Pandas Datenframe aus. 

# Enter your code here

```{tip}
Die [head-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html) liefert die ersten Zeilen eines Pandas Datenframe.

```

````{Dropdown} Lösung Task 1
  ```{code-block} python
  data.head(7)
  ```
`````

### Task 2: Beobachtungen aus den ersten 7 Datensätzen
* Die Merkmale Gender und Home Owner enthalten <mark> [?] </mark>  
* Die meisten Merkmale enthalten <mark> [?] </mark>  
* Die numerischen Merkmale unterscheiden sich stark in <mark> [?] </mark> 

```{Dropdown} Lösung Task 2
* Die Merkmale Gender und Home Owner enthalten fehlende Werte
* Die meisten Merkmale enthalten nicht-numerische Daten
* Die numerischen Merkmale unterscheiden sich stark in den Wertebereichen
```

### Task 3: Datenset beschreiben

# Geben Sie hier den Code ein, 
# um die Informationen über die Daten zu erhalten.

Das Datenset besteht aus  
<mark> [?] </mark> Datensätzen  
<mark> [?] </mark> Merkmalen  
  
Die Zielvariable ist: <mark> [?] </mark>  
      
<mark> [?] </mark> Merkmale sind vom Typ Float  
<mark> [?] </mark> Merkmale sind vom Typ Integer  
<mark> [?] </mark> Merkmale sind Objekte  

```{tip}
Die [Info-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html) liefert eine Zusammenfassung des Pandas Datenframe.

```

````{Dropdown} Lösung Task 3
Informationen erhalten über den Aufruf der Info-Methode
  ```{code-block} python
  data.info()
  ```
Das Datenset besteht aus   
1000 Datensätzen  
12 Merkmalen  
  
Die Zielvariable ist: Purchased Bike   
  
4 Merkmale sind vom Typ Float  
1 Merkmal vom Typ Integer  
7 Merkmale sind Objekte  
````

### Beschreibung der Merkmale:
* **ID**: Eindeutige Id des Kunden
* **Marital Status**: Familienstand [Verheirated, Single]
* **Gender**: Geschlecht [Male, Female]
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

### Task 4: Datentypen bestimmen

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

```{Dropdown} Lösung Task 4
Numerische Daten:
* Diskret
    * Cars
    * Children
* Kontinuierlich
    * Income
    * Age
    
Kategorische Daten:
* Nominal
    * ID
    * Marital Status
    * Gender
    * Occupation
    * Home Owner
    * Region
* Ordinal
    * Education
    * Commute Distance
```

###  Task 5: Deskriptive Statistiken erstellen

# Enter your code here

```{tip}
Die [Describe-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)  liefert deskriptive Statistiken des Pandas Datenframe.

```

````{Dropdown} Lösung Task 5
  ```{code-block} python
  data.describe()
  ```
`````

#### Task 6: Beobachtungen notieren

* Das Einkommen der meisten Kunden (75%) liegt unter <mark> [?] </mark>.
* Die Standardabweichung des Einkommen beträgt <mark> [?] </mark>.    
* 50% der Kunden haben mehr als <mark> [?] </mark> Kinder.  
* Die Kunden besitzen im Durchschnitt <mark> [?] </mark> Autos.  
* Das Alter der Kunden variiert zwischen <mark> [?] </mark> und <mark> [?] </mark>.  
* 25% der Kunden sind älter als <mark> [?] </mark>.  

```{Dropdown} Lösung Task 6
* Das Einkommen der meisten Kunden (75%) liegt unter <mark> 24470,75 </mark>.
* Die Standardabweichung des Einkommen beträgt <mark> 31067,82 </mark>.
* 50% der Kunden haben mehr als <mark> 2 </mark> Kinder.
* Die Kunden besitzen im Durchschnitt <mark> 1,46 </mark> Autos.
* Das Alter der Kunden variiert zwischen <mark> 25 </mark> und <mark> 89 </mark>.
* 25% der Kunden sind älter als <mark> 35 </mark>.
````

## Fehlende Werte korrigieren

### Task 7: Fehlende Werte ermitteln (absolut)

Ermitteln Sie die absoulte Zahl der fehlenden Werte pro Spalte und speichern Sie diese in einer Variablen namens "missingValuesCount".

# Enter your code here

```{tip}
Die [isnull-Methode](https://pandas.pydata.org/docs/reference/api/pandas.isnull.html)  liefert für jeden Wert des Pandas Datenframe einen Boolean-Wert, true oder false. True bedeutet, es handelt sich um einen fehlenden Wert.

Die [sum-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sum.html) liefert die Anzahl der True Werte pro Spalte in einer Pandas Series (bei Default-Einstellung)

```

````{Dropdown} Lösung Task 7
  ```{code-block} python
  missingValuesCount = data.isnull().sum()
  missingValuesCount
  ```
````

### Task 8: Fehlende Werte ermitteln (absolut und prozentual)

Geben Sie den absoluten und prozentualen-Wert pro Spalte an. Implementieren Sie folgende Schritte
* Sortieren Sie das zuvor gespeicherte Ergebnis (missingValuesCount) in absteigender Reihenfolge und speichern Sie das Ergebnis in einer neuen Variable namens "total".
* Berechnen Sie den prozentualen Anteil der fehlenden Werte an der Gesamtzahl der der Datensätze, sortieren Sie das Ergebnis absteigend und speichern es in einer neuen Variable namens "percentage".
* Erstellen Sie einen Pandas Datenframe mit 2 Spalten. Die erste Spalte mit der Bezeichnung "Gesamt" und den Daten der Variable total und die zweite Spalte mit der Bezeichnung "Prozentual" und den Daten der Variable percentage.
* Speichern Sie den Pandas Datenframe in einer neuen Variable namens "missingData".
* Geben Sie den erstellten Pandas Datenframe aus.

# Enter your code here

```{tip}
Die [sort_values-Methode](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html) sortiert die Pandas Series anhand der enthaltenen Werte.

Die [shape-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) liefert die Dimensionen eines Pandas Datenframe in einem Tuple, wobei der erste Wert, der Anzahl der Zeilen entspricht und der zweite Wert, der Anzahl der Spalten.

Unter Verwendung der [DataFrame-Methode](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) und der Übergabe eines Dictionaries, wobei die Keys der Spaltenbezeichnung und die Values der Pandas Series entsprechen, lassen sich Pandas Dataframes erstellen.

```

````{Dropdown} Lösung Task 7
  ```{code-block} python
  total = missingValuesCount.sort_values(ascending=False)
  percentage = (missingValuesCount/data.shape[0]*100).sort_values(ascending=False)
  missingData = pd.DataFrame({'Gesamt':total, 'Prozentual':percentage})
  missingData
  ```
````

### Task 9: Notieren Sie Ihre Beobachtungen

Was fällt Ihnen bei den ermittelten Werten aus Task 8 auf? Ist die Anzahl der fehlenden Werte groß oder klein? Was ist könnte die Ursache für die fehlenden Werte sein? Existieren die Werte und wurden nicht erfasst?

Beobachtungen:
* <mark> [?] </mark>
* <mark> [?] </mark>

```{Dropdown} Lösung Task 9
* Die fehlenden Werte pro Spalte sind sehr gering und liegen zwischen 0.4 und 1.1 Prozent.
* Alle fehlende Werte (Cars, Age, Children, Marital Status, Income, Home Owner) existieren vermutlich, wurden jedoch nicht erfasst.
```

### Task 10: Fehlende Werte ersetzen/entfernen

Entscheiden Sie sich bei jedem Merkmal für eine Methode und wenden diese auf die Datensätze, welche fehlende Werte enthalten an.

``` {Admonition} Wiederholung
Möglicher Umgang mit fehlenden Werten:
1. Datensätze entfernen
2. Ersetzen durch Median/Mittelwert/Mode
3. Eine eigene Kategorie zuweisen
4. Fehlende Werte schätzen

{doc}`missingData` anhand eines Beispiel-Datensets.

```

```{tip}
Die [hist-Methode](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html)  kann auf eine Spalte des Datenframes angewendet werden um die Verteilung der Daten zu erhalten.

```

#### a) Gender

# Enter your code here

````{Dropdown} Lösung Task 10a

Beim Gender Merkmal handelt es sich um kategorische Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen.

  ```{code-block} python
  data['Gender'].hist()
  
  data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
  ```
````

#### b) Cars

# Enter your code here

````{Dropdown} Lösung Task 10b

Beim Cars Merkmal handelt es sich um numerisch diskrete Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen.

  ```{code-block} python
  data['Cars'].hist()
  
  data['Cars'].fillna(data['Cars'].mode()[0], inplace=True)
  ```
````

#### c) Age

# Enter your code here

````{Dropdown} Lösung Task 10c

Beim Age Merkmal handelt es sich um numerische Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte zu ersetzen. Aufgrund der rechtsschiefen Verteilung der Daten bietet sich der Median als Ersatzwert an.

  ```{code-block} python
  data['Age'].hist()
  
  data['Age'].fillna(data['Age'].median(), inplace=True)
  ```
````

#### d) Children

# Enter your code here

````{Dropdown} Lösung Task 10d

Beim Merkmal Children handelt es sich um diskrete Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 

  ```{code-block} python
  data['Children'].hist()
  
  data['Children'].fillna(data['Children'].mode()[0], inplace=True)
  ```
````

#### e) Marital Status

# Enter your code here

````{Dropdown} Lösung Task 10e

Beim Merkmal Marital Status handelt es sich um nominale Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 

  ```{code-block} python
  data['Marital Status'].hist()
  
  data['Marital Status'].fillna(data['Marital Status'].mode()[0], inplace=True)
  ```
````

#### f) Income

# Enter your code here

````{Dropdown} Lösung Task 10f

Beim Merkmal Income handelt es sich um kontinuierliche Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem Mittelwert zu ersetzen.

  ```{code-block} python
  data['Income'].hist()
  
  data['Income'].fillna(data['Income'].mean(), inplace=True)
  ```
````

#### g) Home Owner

# Enter your code here

````{Dropdown} Lösung Task 10g

Beim Merkmal Owner handelt es sich um nominale Daten. Aufgrund der geringen Anzahl an fehlenden Werten in Bezug auf die Gesamtanzahl des Datensets, macht es Sinn die fehlenden Datensätze zu entfernen oder die fehlenden Werte mit dem häufigsten vorkommenden Wert zu ersetzen. 

  ```{code-block} python
  data['Home Owner'].hist()
  
  data['Home Owner'].fillna(data['Home Owner'].mode()[0], inplace=True)
  ```
````

### Task 11: Überprüfen der fehlenden Werte

Geben Sie erneut die Anzahl der fehlenden Werte aus. Es sollten jetzt keine fehlende Werte existieren.

# Enter your code here

````{Dropdown} Lösung Task 11

  ```{code-block} python
  missingValuesCount = data.isnull().sum()
  missingValuesCount
  ```
````

### Task 12: Ergebnis speichern

Speichern Sie die aufbereiteten Daten als Pickle-Datei unter '../output/preparedData.pkl' ab.

# Enter your code here

```{Tip}
Die [to_pickle-Methode](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html) ermöglicht eine einfache Speicherung des Pandas Datenframe im .pkl-Format
```

````{Dropdown} Lösung Task 11
  ```{code-block} python
  data.to_pickle('../output/preparedData.pkl')
  ```
````
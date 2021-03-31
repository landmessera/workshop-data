# Überblick verschaffen

## Python Module importieren und laden der Daten

Python Module importieren

import pandas as pd
import numpy as np

Daten laden

data = pd.read_csv("../data/titanic/train.csv")

## Anzeigen der ersten Datensätze

data.head(7)

Beobachtungen aus den ersten Datensätzen:
* bei den Merkmalen Alter und Kabine fehlen Werte (NaN)
* einige der Merkmale sind nicht numerisch
* Merkmale unterscheiden sich teils stark in den Wertebereichen

data.info()

Das Datenset besteht aus 891 Datensätzen, 11 Merkmalen und der Zielvariable (survived). Zwei der Merkmale sind vom Typ Float, 5 sind Integer und 5 sind Objekte.

Beschreibung der Merkmale:
* PassengerId: Eindeutige Id eines Passagiers
* Survived: Das Schiffsunglück überlebt ( 1 oder 0, für ja oder nein)
* Pclass: Ticketklasse
* Name: Passagiername
* Sex: Geschlecht
* Age: Alter
* SibSp: Geschwister oder Ehepartner mit an Board der Titanic
* Parch: Eltern oder Kinder an Board der Titanic
* fare: Ticketpreis
* Cabin: Kabinennummer
* Embarked: Ort des Zustiegs

## Datentypen

### Numerische Daten 
(wenn man den Mittelwert berechnen kann)

- Diskrete Daten (wenn man es zählen kann)
- Kontinuierliche Daten (wenn man es messen kann, Länge, Gewicht, Temperatur etc.)

### Kategorische Daten
- Nominale Daten (wenn man es benennen kann, z.B. Farbe: rot, blau)
- Ordinale Daten (wenn man es ordnen kann)

Kategorische Daten in numerischer Form:
- Hausnummern,
- Telefonnummern
- Geburtsdatum
- Postleitzahlen

### Anwendung: Datentypen bestimmen 

Numerische Daten:
* Diskret
    * Sibsp
    * Parch
* Kontinuierlich
    * Age
    * Fare

Kategorische Daten:
* Nominal
    * PassengerId
    * Name
    * Ticket
    * Cabin
    * Survived
    * Sex 
    * Embarked
* Ordinal
    * Pclass

### Wissen testen

Die Variablen Land (Deutschland, Frankreich, Schweden) und Fläche (in Quadratkilometer) sind

```{dropdown} beide kategorisch
Falsch.
```
```{dropdown} beide numerisch
Falsch.
```
```{dropdown} numerisch und kategorisch
Falsch.
```
```{dropdown} kategorisch und numerisch
Richtig.
```

Die Variablen Monat (Januar, Februar, März) und monatlicher Umsatz (in Euro) sind

```{dropdown} beide kategorisch
<font color='red'>Falsch.</font>
```
```{dropdown} beide numerisch
<font color='red'>Falsch.</font>
```
```{dropdown} kategorisch und numerisch
<font color='green'>Richtig.</font>
```
```{dropdown} numerisch und kategorisch
<font color='red'>Falsch.</font>
```

Die Variablen Wasserverbrauch (Liter pro Tag) und Hausgröße (in m^2) sind

```{dropdown} beide kategorisch
<font color='red'>Falsch.</font>
```
```{dropdown} beide numerisch
<font color='green'>Richtig.</font>
```
```{dropdown} kategorisch und numerisch
<font color='red'>Falsch.</font>
```
```{dropdown} numerisch und kategorisch
<font color='red'>Falsch.</font>
```

Der Wasserverbrauch von 250 Haushalten wurde in niedrig, mittel und hoch eingestuft und die Hausgröße in klein, mittel und groß. Die Variablen Wasserhaushalt und Hausgröße sind in diesem Fall'

```{dropdown} beide kategorisch
<font color='green'>Richtig.</font>
```
```{dropdown} beide numerisch
<font color='red'>Falsch.</font>
```
```{dropdown} kategorisch und numerisch
<font color='red'>Falsch.</font>
```
```{dropdown} numerisch und kategorisch
<font color='red'>Falsch.</font>
    

## Grundlegende deskriptive Statistiken

data.describe()

Fakten aus den einfachen deskriptiven Statistiken:
* 38% der Passagiere im Datenset haben überlebt
* Das Alter der Passagiere variiert zwischen 0,4 und 80 Jahren.
* 75% der Passagiere haben weniger als 31,00 bezahlt  
* 25% der Passagiere haben mehr als 31,00 bezahlt
* der durchschnittliche Ticketpreis beträgt 32


### Wissen testen

Welche Aussage trifft zu?

```{dropdown} 25% der Passagiere sind mit Eltern oder Kinder gereist
<font color='red'>Falsch.</font>
```
```{dropdown} 50% der Passagiere sind mit Eltern oder Kinder gereist
<font color='red'>Falsch.</font>
```
```{dropdown} die meisten Passagiere (> 75%) sind nicht mit Eltern oder Kinder gereist
<font color='green'>Richtig.</font>
```

Welche Aussage trifft zu?
```{dropdown} 25% der Passagiere sind mit Ehepartner oder Geschwister gereist
<font color='green'>Richtig.</font>
```
```{dropdown} 50% der Passagiere sind mit Ehepartner oder Geschwister gereist
<font color='red'>Falsch.</font>
```
```{dropdown} die meisten Passagiere (> 75%) sind mit Ehepartner oder Geschwister gereist
<font color='red'>Falsch.</font>
```


# Überblick verschaffen

Um einen ersten Eindruck von Daten zu erhalten, schaut man sich am Besten ein paar Datensätze an und versucht diese zu verstehen. Welche Informationen relevant sind, hängt vom Anwendungsfall ab. Einige Informationen sind jedoch bei den meisten Anwendungsfällen relevant:

* Welche Merkmale gibt es?
* Liegen Zielwerte vor? 
* Welche Datentypen liegen vor?
* Gibt es fehlende Werte?
* Wie unterscheiden sich die Wertebereiche der Merkmale?
* Wie viele Datensätze gibt es?
* Was kann man für Aussagen auf Basis von grundlegenden Statistiken treffen?

Am Beispiel des Titanic Datensets wird das Vorgehen veranschaulicht. Es wird das Python Modul Pandas verwendet.

## Modul importieren und Daten laden

Python Modul Pandas importieren

import pandas as pd

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
* SibSp: Anzahl Geschwister oder Ehepartner mit an Board der Titanic
* Parch: Anzahl Eltern oder Kinder an Board der Titanic
* fare: Ticketpreis in US-Dollar
* Cabin: Kabinennummer
* Embarked: Ort des Zustiegs

## Datentypen

Die Bestimmung der Datentypen ist sehr wichtig. Der Datentyp entscheidet welche Vorverarbeitungsschritte notwendig und möglich sind. Zunächst wird unterschieden in **numerische und kategorische** Daten. Numerische Daten liegen vor, wenn man einen Mittelwert berechnen kann. Bei den numerischen Daten wird weiter unterschieden in **diskrete oder kontinuierliche** Daten. Daten sind diskret, wenn man die Werte **zählen** kann. Kann man die Werte **messen**, wie z.B. Länge, Gewicht oder Temperatur, dann handelt es sich um kategorische Daten. Kategorische Daten werden unterteilt in **nominale und ordinale Daten**. Wenn man das Merkmal **benennen** kann, wie z.B. Farbe: rot, blau etc., dann handelt es sich um nominale Daten. Ordinale Daten liegen vor, wenn man die Werte ordnen kann. 

### Beispiel Titantic: Datentypen bestimmen

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

### Wissen prüfen

Die Variablen Land (Deutschland, Frankreich, Schweden) und Fläche (in Quadratkilometer) sind

```{dropdown} beide kategorisch
<font color='red'>Falsch.</font>
```
```{dropdown} beide numerisch
<font color='red'>Falsch.</font>
```
```{dropdown} numerisch und kategorisch
<font color='red'>Falsch.</font>
```
```{dropdown} kategorisch und numerisch
<font color='green'>Richtig.</font>
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


Mit Hilfe der [describe()-Methode](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) lassen sich sehr einfach die grundlegenden Statistiken der Merkmale ausgeben. Aus den Ergebnissen können Aussagen über die Merkmale getroffen werden, die ein besseres Verständnis der Daten schaffen.

data.describe()

Fakten aus den einfachen deskriptiven Statistiken:
* 38% der Passagiere im Datenset haben überlebt
* Das Alter der Passagiere variiert zwischen 0,4 und 80 Jahren.
* 75% der Passagiere haben weniger als 31\$ bezahlt  
* 25% der Passagiere haben mehr als 31\$ bezahlt
* der durchschnittliche Ticketpreis beträgt 32\$


### Wissen prüfen

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
```{dropdown} 50% der Passagiere sind mit Ehepartner oder Geschwister gereist
<font color='red'>Falsch.</font>
```
```{dropdown} 75% der Passagiere sind mit Ehepartner oder Geschwister gereist
<font color='green'>Falsch.</font>
```
```{dropdown} die meisten Passagiere (> 75%) sind ohne Ehepartner oder Geschwister gereist
<font color='red'>Richtig.</font>
```


# Datenexploration und Cleaning

import ipywidgets as widgets
import sys
from IPython.display import display
from IPython.display import clear_output

def create_multipleChoice_widget(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternativ = widgets.Box(
        [
            widgets.RadioButtons(
                options = radio_options,
                layout={'width': 'max-content'},
                description = '',
                disabled = False
            )
        ]
    )
    
    description_out = widgets.Output()
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternativ.children[0].value)
        if a==correct_answer_index:
            s = '\x1b[6;30;42m' + "Richtig." + '\x1b[0m' +"\n" #green color
        else:
            s = '\x1b[5;30;41m' + "Falsch. " + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="Absenden")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternativ, check, feedback_out])
    

Q1 = create_multipleChoice_widget(
    'Die Variablen Land (Deutschland, Frankreich, Schweden) und Fläche (in Quadratkilometer) sind',
    ['beide kategorisch','beide numerisch','numerisch und kategorisch', 'kategorisch und numerisch', 'kategorisch nominal und kategorisch ordinal'],
    'kategorisch und numerisch')
Q2 = create_multipleChoice_widget(
    'Die Variablen Monat (Januar, Februar, März) und monatlicher Umsatz (in Euro) sind',
    ['beide kategorisch','beide numerisch','numerisch und kategorisch', 'kategorisch und numerisch', 'kategorisch nominal und kategorisch ordinal'],
    'kategorisch und numerisch')
Q3 = create_multipleChoice_widget(
    'Die Variablen Wasserverbrauch (Liter pro Tag) und Hausgröße (in m^2) sind',
    ['beide kategorisch','beide numerisch','numerisch und kategorisch', 'kategorisch und numerisch', 'kategorisch nominal und kategorisch ordinal'],
    'beide numerisch')
Q4 = create_multipleChoice_widget(
    'Der Wasserverbrauch von 250 Haushalten wurde in niedrig, mittel und hoch eingestuft und die Hausgröße in klein, mittel und groß. Die Variablen Wasserhaushalt und Hausgröße sind in diesem Fall',
    ['beide kategorisch','beide numerisch','numerisch und kategorisch', 'kategorisch und numerisch', 'kategorisch nominal und kategorisch ordinal'],
    'beide kategorisch')

bQ1 = create_multipleChoice_widget(
    'Welche Aussage trifft zu?',
    [
        '50% der Passagiere sind mit Eltern oder Kinder gereist',
        '25% der Passagiere sind mit Eltern oder Kinder gereist',
        'die meisten Passagiere (> 75%) sind nicht mit Eltern oder Kinder gereist'
    ],
    'die meisten Passagiere (> 75%) sind nicht mit Eltern oder Kinder gereist')
bQ2 = create_multipleChoice_widget(
    'Welche Aussage trifft zu?',
    [
        '50% der Passagiere sind mit Ehepartner oder Geschwister gereist',
        '25% der Passagiere sind mit Ehepartner oder Geschwister gereist',
        'die meisten Passagiere (> 75%) sind mit Ehepartner oder Geschwister gereist'
    ],
    '25% der Passagiere sind mit Ehepartner oder Geschwister gereist')


Ziele:
* Einen ersten Eindruck von den Daten gewinnen  
* Grundlegendes Verständnis der Daten 
* Grundlegende Statistiken erheben und visualisieren
* Fehlende Daten ermitteln
* Umgang mit fehlenden Daten

## Überblick verschaffen

Zunächst werden die Bibliotheken und Daten geladen.

Um die Techniken zu erklären wird das Datenset "Titanic" verwendet. Die Techniken werden im Anschluss auf dem Datenset "Bike Buyers" angewendet.

#### Das Datenset Titanic

<div><img src="./pics/titanic.jpeg" style="width: 400px; float: left; margin-top: 10px; margin-right: 10px;"/></div>
<div style="font-size: 7px; float: left;">Quelle: https://de.wikipedia.org/wiki/RMS_Titanic#/media/Datei:RMS_Titanic_3.jpg</div>


<div style="font-size: 14px; float: left;">Der Untergang der Titanic ist eines der berüchtigtsten Schiffsunglücke der Geschichte.

Am 15. April 1912, während ihrer Jungfernfahrt, sank die weithin als "unsinkbar" geltende RMS Titanic nach der Kollision mit einem Eisberg. Unglücklicherweise gab es nicht genügend Rettungsboote für alle an Bord, was zum Tod von 1502 der 2224 Passagiere und der Besatzung führte.

Obwohl das Überleben auch ein gewisses Glückselement beinhaltete, scheint es, dass einige Gruppen von Menschen eher überlebten als andere.

Die Aufgabe des Datensets lautet: Erstellen eines Vorhersagemodell, das die Frage beantwortet: "Welche Arten von Menschen überlebten mit größerer Wahrscheinlichkeit?" unter Verwendung von Passagierdaten (d. h. Name, Alter, Geschlecht, sozioökonomische Klasse usw.).</div>

#### Python Module importieren

import pandas as pd
import numpy as np

#### Daten laden

data = pd.read_csv("./data/titanic/train.csv")

#### Anzeigen der ersten Datensätze

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

Datentypen

Numerische Daten (wenn man den Mittelwert berechnen kann)

- Diskrete Daten (wenn man es zählen kann)
- Kontinuierliche Daten (wenn man es messen kann, Länge, Gewicht, Temperatur etc.)

Kategorische Daten
- Nominale Daten (wenn man es benennen kann, z.B. Farbe: rot, blau)
- Ordinale Daten (wenn man es ordnen kann)

Categorical data in numerischer Form:
- Hausnummern,
- Telefonnummern
- Geburtsdatum
- Postleitzahlen

#### Numerische und nicht-numerische Daten

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

##### Wissen testen

display(Q1)
display(Q2)
display(Q3)
display(Q4)

#### Grundlegende deskriptive Statistiken

data.describe()

Fakten aus den einfachen deskriptiven Statistiken:
* 38% der Passagiere im Datenset haben überlebt
* Das Alter der Passagiere variiert zwischen 0,4 und 80 Jahren.
* 75% der Passagiere haben weniger als 31,00 bezahlt  
* 25% der Passagiere haben mehr als 31,00 bezahlt
* der durchschnittliche Ticketpreis beträgt 32
* 25% der Passagiere hatten Ehepartner oder Geschwister an Board


#### Wissen testen

display(bQ1)
display(bQ2)

#### Fehlende Werte ermitteln

Die meisten Machine Learning-Algorithmen können nicht mit fehlenden Werten umgehen. Die Daten müssen daher auf fehlende Daten geprüft und modifiziert werden. Möglicher Umgang mit fehlenden Werten:

##### 1. Datensätze entfernen
Diese Methode wird sehr häufig verwendet, wenn eine ausreichende Menge an Daten vorliegen. Zeilen, die fehlende Werte enthalten werden entfernt. 

data_tmp = data.dropna()
data_tmp.isnull().sum()

##### 2. Ersetzen durch Mittelwert/Median/Mode

Kann angewendet werden, wenn es sich um ein Merkmal mit numerischen Daten handelt. Der Mittelwert, Median oder Mode des Merkmals wird berechnet und die fehlenden Werte mit dem berechneten Wert ersetzt. Es handelt sich bei dieser Methode um eine Annäherung mittels einem statistischen Ansatz. Eine weiter Möglichkeit ist die berechnung des Durchschnitts der Nachbarwerte. Dieses Vorgehen funktioniert besser, wenn die Daten linear sind. 

Beim Mode-Wert handelt es sich um den am häufigsten vorkommenden Wert. Der Median, ist das 50%-Quantil und sollte verwendet werden, wenn die Verteilung der Daten rechts- oder linksschief ist.


data['Age'].head(10)

data['Age'].mean()

data['Age'].replace(np.NaN, data['Age'].mean()).head(10)

Alternativ: Berechnung des Median oder Mode-Wert

data['Age'].median()

data['Age'].mode()

##### 3. Eine eigene Kategorie zuweisen

Diese Methode kann bei kategorischen Daten angewendet werden. Für die fehlenden Werte wird eine eigene Kategorie erstellt wie z.B. "Unbekannt" und die fehlenden Werte mit dieser Bezeichnung ersetzt.

data['Cabin'].head(10)

data['Cabin'].fillna('Unbekannt').head(10)

##### 4. Die fehlenden Werte schätzen

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

##### Welche Methode sollte verwendet werden?

Finden Sie heraus warum die Daten fehlen:

Häufig erhalten wir die Daten und haben diese nicht selbst aufgezeichnet. Wir können daher nicht sicher sagen, warum die Daten fehlen. Meist lässt sich dies schätzen. Stellen Sie sich die Frage  

**"Fehlt der Wert weil er nicht aufgezeichnet wurde oder weil er nicht existiert"?**

Wenn ein Wert fehlt, weil er nicht existiert macht es keinen Sinn den Wert zu schätzen. In diesem Fall ist es besser den Datensatz (die Zeile) zu verwerfen oder die Lücke mit einem "Nicht vorhanden" (NaN)-Wert zu belegen. 

Fehlt ein Wert weil er nicht aufgezeichnet wurde, macht es Sinn diesen zu schätzen. Entweder durch statistische Analyse der restlichen Werte in der Spalte oder durch Machine Learning Algorithmen unter Verwendung der anderen Spalten.

##### Anwendung am Bespiel Datenset Titanic

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

data['Embarked'].fillna('S', inplace=True)

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


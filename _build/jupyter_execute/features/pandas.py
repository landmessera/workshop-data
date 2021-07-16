# Pandas

Pandas 
stellt umfangreiche Datenstrukturen und Funktionen zur Verfügung, um die Arbeit mit strukturierten Daten zu erleichtern.[^footnote1]  Die wichtigste Datenstruktur ist der sogenannte DataFrame. Es handelt sich um eine 2-dimensionale Struktur. Man kann sich den DatenFrame wie ein Excel-Sheet vorstellen. Die Daten werden in einer Tabelle erfasst, Zeilen und Spalten können benannt werden. Diese Kennzeichnung der Zeilen und Spalten sind ein wesentlicher Vorteil gegenüber anderen Formaten in der Programmierung wie z.B. einer einfachen Matrix. Numpy ist z.B. ein Python-Packet, dass für den Umgang mit numerischen Daten sehr beliebt ist und ebenfalls zahlreiche Funktionen bietet. Die zentrale Struktur ist hier das Numpy-Array. Man kann sich das Array als eine Abbildung von Matrizen vorstellen. Hier fehlt die Möglichkeit Spalten- und Zeilenbezeichnung vergeben zu können. 

Pandas Datenformat
* Dataframe
* Series

Pandas Frame erstellen
Erste Zeilen ausgeben
Shape ausgeben
Info - Summary
Spalten ausgeben
Index ausgeben
Eine Zeile auswählen
Mehrere Zeilen auswählen
Filter Rows by Column Value
Plot
Plot Scatter
Create new Columns
Mean of column
Median of two columns
describe
count()
Werte sortieren
concat

Daten aus CSV lesen
DataFrame speichern
DataFrame auslesen


data = pd.read_csv("../data/titanic/train.csv")

[^footnote1]: "Python for Data Analysis", Wes McKinney


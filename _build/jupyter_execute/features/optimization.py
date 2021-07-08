## Ziele

* Eine Baseline erstellen
* Eine Pipeline für die Organisation nutzen
* Optimale Parametereinstellungen finden
* Daten verstehen und kombinieren
* Wichtige Merkmale erkennen
* Auswahl geeigneter Transformationen 

## Inhalt

In den vorigen Abschnitten wurde eine Basis für die Anwendung von Machine Learning Verfahren geschaffen. Die Daten wurden aufbereitet und Transformationen angewendet. Doch woher weiß man ob die Transformationen gut waren? Und was bedeutet in diesem Kontext gut? 

Transformationen werden aus zwei Gründen angewendet, um
1. die Daten in eine Form zu bringen, die den **Anforderungen** von Machine Learning Algorithmen entspricht.

2. die **Leistung** zu steigern.

Der erste Punkt lässt sich leicht prüfen indem ein gewünschter Machine Learning Algorithmus auf den Daten ausgeführt wird. Der zweite Punkt erfordert ein **Leistungsmaß**. Hierzu wird ein Leistungsmaß wie z.B. im Fall einer Klassifikationsaufgabe die Accuracy (Genaugkeitsmaß) gewählt. Der Algorithmus wird mit dem Trainingsdatenset trainiert und mit dem Validierungsdatenset validiert. Das Ergebnis ist ein sogenannter **Score**, der die Accuracy abbildet. 

```{admonition} Was ist ein guter Score?
Bei einer Klassifikationsaufgabe in zwei Klassen sagt ein Accuracy-Score größer als 0,5 aus, dass es sich nicht um eine Zufallsentscheidung handelt. Also mehr als 50% der Daten wurden richtig klassifiziert. Welcher Accuracy-Score ausreichend ist, **hängt stark vom Anwendungsfall ab**. So kann es sein, dass z.B. bei einer Fehlererkennung in der Produktion ein Accuracy-Score von 0.80 akzeptabel ist. In der Medizin, bei der Erkennung von Krebs wird ein Accurcy-Score von 0.98 benötigt. Fehlentscheidungen sind in diesem Bereich nur in geringem Ausmaß zulässig.
```

Das erste Ergebnis dient als Basis für Optimierungen, es ist die sogenannte **Baseline**. Jede Änderung im Vorgehen wird anschließend validiert und mit der Baseline verglichen. Ist die Baseline besser, wird die Änderung verworfen, ist die Baseline schlechter, wird die Änderung behalten.

In diesem Abschnitt wird am Beispiel Titanic gezeigt, wie man eine **Baseline erstellt, Optimierung der Parameter und Merkmale durchführt und diese validiert**.


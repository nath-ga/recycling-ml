# README.md

# Bilderkennung zur Materialklassifikation mit CNN und PyTorch

Dieses Portfolio-Projekt zeigt, wie mit Convolutional Neural Networks (CNNs) in PyTorch eine automatische Klassifikation von Müllmaterialien anhand von Bildern erfolgen kann. Ziel ist die Entwicklung eines Prototyps, der sich perspektivisch in automatisierte Sortiersysteme von Recyclinganlagen integrieren lässt.

## Projektziel

Aufbau eines bildbasierten Klassifikators, der sechs Materialarten zur Recyclingvorbereitung erkennen und trennen kann:

- **cardboard**
- **glass**
- **metal**
- **paper**
- **plastic**
- **trash**

Dieser Prototyp legt die Grundlage für Machine-Learning-gestützte Sortieranlagen, in denen z. B. ein Roboterarm das korrekt erkannte Material vom Förderband entnimmt.

## Modell & Technik

- **Modellarchitektur:** ResNet18 (pretrained)
- **Framework:** PyTorch
- **Frontend:** Streamlit-Demo zum Hochladen und Auswerten einzelner Bilder
- **Trainingsdaten:** [Garbage Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)  
  (manuelle Aufteilung in Train/Test)

## Ablauf

1. Bilder vorbereitet und in `/data/train` und `/data/test` unterteilt  
2. Zwei Trainingsvarianten:
   - Variante A: Bilder wurden **verzerrend skaliert** (Standard-Resize auf 224×224 px)
   - Variante B: Bilder wurden **mit Padding** auf 224×224 px gebracht, um Verzerrung zu vermeiden  
3. Beide Modelle wurden trainiert und anschließend ausgewertet
4. Fehlklassifizierungen wurden analysiert
5. Zu Testzwecken wurde eine Streamlit-Demo aufgebaut (lokale Anwendung)

## Ergebnisse

| Variante        | Accuracy | Besonderheiten                        |
|----------------|----------|----------------------------------------|
| Ohne Padding   | 87.20 %  | Teilweise sehr verzerrte Bilder        |
| Mit Padding    | 87.40 %  | Bessere Stabilität & Bildtreue         |

**Fehlklassifikationen** traten häufig bei stark gezoomten Bildausschnitten ohne Kontext auf, z. B. einem Flaschenhals oder einer Kartonecke ohne erkennbare Kanten. Farb- und Texturinformationen könnten bei der Weiterentwicklung helfen.

## Learnings

- Padding statt verzerrendem Resize verbessert die Bildqualität für das Modell
- CNNs wie ResNet18 sind auch mit überschaubaren Daten gut einsetzbar
- Die Analyse zeigt: Kontextelemente (Rand, Umriss, Materialstruktur) sind oft entscheidend
- Die Streamlit-Demo ist ein effektives Werkzeug zur Präsentation und Testung

## Demo starten

Lokale Ausführung (nach Installation der Abhängigkeiten):

```bash
streamlit run app.py
```
Die Demo erlaubt das Hochladen eines Bildes und zeigt: 
Die vorhergesagte Klasse
Die Wahrscheinlichkeiten aller Klassen als Balkendiagramm (s.a. Präsentation)

## Ordnerstruktur 

```
recycling-ml/
│
├── data/                     # Bilder nach Klassen (train/test)
├── models/                   # Gespeicherte Modelle (.pt)
├── outputs/                  # Diagramme & Ergebnisbilder
├── src/                      # Trainings- & Evaluierungsskripte
│   ├── train_model.py
│   ├── train_model_padded.py
│   └── evaluate_and_log_predictions.py
├── app.py                    # Streamlit-Demo
├── .gitignore
└── README.md
```

## Datenquelle & Nutzungshinweis

Die verwendeten Bilddaten stammen aus dem öffentlichen Garbage Classification Dataset auf Kaggle https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification.
Lizenzhinweis:
„Data files © Original Authors“ – die Rechte an den Bildern liegen bei den ursprünglichen Ersteller:innen.
Die Nutzung im Rahmen dieses Projekts erfolgt ausschließlich zu nicht-kommerziellen Demonstrations- und Lernzwecken.

## Lizenz

Dieses Projekt steht unter der [Creative Commons BY-NC 4.0 Lizenz](https://creativecommons.org/licenses/by-nc/4.0/).  
Das bedeutet: Sie dürfen den Code gerne verwenden, verändern und weitergeben – aber nicht kommerziell, und unter Nennung der Urheberin Nathalie Gassert.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Hinweis: Die verwendeten Bilddaten unterliegen gesonderten Rechten – siehe Abschnitt *Datenquelle & Nutzungshinweis*.

##  Status

Projekt abgeschlossen – keine weitere Modelloptimierung geplant, da es sich um ein Portfolio-Projekt handelt.
Ergebnisse und Screenshots wurden dokumentiert.

Autorin: Nathalie Gassert
GitHub Portfolio github.com/nath-ga | LinkedIn linkedin.com/in/nathalie-gassert/
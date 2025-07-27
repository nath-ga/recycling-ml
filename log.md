# 📓 Projektlog: Recycling-ML – Bilderkennung zur Materialklassifikation

## 🌱 Projektstart
- Ziel: Prototypisches Machine-Learning-Modell zur Klassifikation von Abfällen (z. B. Glas, Metall, Papier)
- Datensatz: Garbage Classification (Kaggle)
- Train/Test-Split: 80/20 automatisch per Skript (`01_split_dataset.py`)
- Struktur angepasst für PyTorch: `data/train/` und `data/test/` mit Unterordnern pro Klasse

---

## 🧠 Erste Erkenntnisse

### Bildgröße & Verzerrung

- Ursprünglich wurden Bilder mit `transforms.Resize((224, 224))` verarbeitet.
- Problem erkannt: Dadurch werden rechteckige Bilder **verzerrt**, was Formen unnatürlich verändert.
- Entscheidung: Umstieg auf **Padding + Resize**.

```python
# Statt:
transforms.Resize((224, 224))

# Neu:
transforms.Lambda(pad_to_square),
transforms.Resize((224, 224))

Padding-Funktion sorgt für gleichmäßige quadratische Bilder ohne Verzerrung, zentriert mit weißem Rand.

| Version           | Methode                   | Epochen | Lernrate | Batchgröße | Zielgröße |
| ----------------- | ------------------------- | ------- | -------- | ---------- | --------- |
| `model_plain.pt`  | Resize (verzerrt)         | 5       | 0.0003   | 32         | 224×224   |
| `model_padded.pt` | Padding → Resize (sauber) | 5       | 0.0003   | 32         | 224×224   |

🧪 Nächste Schritte
Modelle vergleichen (evaluate_models.py):

Accuracy

Confusion Matrix

ggf. Fehlklassifikationen visualisieren

Danach: Streamlit-Demo vorbereiten

📝 To Do / Ideen
Trainingszeiten dokumentieren

Streamlit-Frontend vorbereiten

Modellgröße prüfen (<100 MB für GitHub?)

Vergleich mit anderen Architekturen (z. B. MobileNetV2?)

Stand: 26. Juli 2025

---

## 📊 Vergleich: Modell ohne vs. mit Padding

Nach dem Training beider Modelle wurde die Test-Performance auf 508 Bildern verglichen.

### 🔎 Accuracy

| Modell                | Accuracy | Bemerkung                           |
|-----------------------|----------|-------------------------------------|
| Ohne Padding          | 87.20 %  | deutliche Schwankung im Training    |
| Mit Padding           | 87.40 %  | stabileres Training, bessere Recall-Werte |

---

### 🧾 Klassifikationsbericht – ohne Padding

| Klasse     | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| cardboard  | 1.00      | 0.80   | 0.89     | 81      |
| glass      | 0.93      | 0.85   | 0.89     | 101     |
| metal      | 0.81      | 0.88   | 0.84     | 82      |
| paper      | 0.88      | 0.90   | 0.89     | 119     |
| plastic    | 0.84      | 0.90   | 0.87     | 97      |
| trash      | 0.70      | 0.93   | 0.80     | 28      |

---

### 🧾 Klassifikationsbericht – mit Padding

| Klasse     | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| cardboard  | 0.92      | 0.99   | 0.95     | 81      |
| glass      | 0.89      | 0.82   | 0.86     | 101     |
| metal      | 0.82      | 0.88   | 0.85     | 82      |
| paper      | 0.94      | 0.85   | 0.89     | 119     |
| plastic    | 0.90      | 0.85   | 0.87     | 97      |
| trash      | 0.62      | 0.93   | 0.74     | 28      |

---

### 🖼️ Confusion Matrices

![Ohne Padding](./fe472e18-5e10-466c-b2ab-9311e110cb10.png)
*Modell ohne Padding*

![Mit Padding](./084adb95-5774-40d5-bbfb-4eed308adf2e.png)
*Modell mit Padding*

---

### 💡 Fazit:

- **Gesamt-Accuracy nahezu identisch**, aber:
  - Das Padding-Modell liefert **gleichmäßigere Recall-Werte**
  - Besonders **„cardboard“ profitiert deutlich** von Padding
  - **Trash-Klasse** wird in beiden Modellen gut erkannt, trotz kleiner Datenbasis
- Training mit Padding wirkt **stabiler und plausibler** – visuell und metrisch.

> Empfehlung: Künftig **Padding-Strategie verwenden**, besonders bei gemischten oder unproportionalen Bildformaten.

... Erkenntnis zum Bildkontext z. B. fehlender Rand bei Nahaufnahmen führt zu Fehlern, Hinweis auf die Bildqualität zu achten

## Streamlit-Demo (model_padded.pt)

- Erfolgreich integriert: Bild-Upload, Vorhersageklasse, Balkendiagramm mit Wahrscheinlichkeiten
- Testweise 20 Bilder hochgeladen (verschiedene Kategorien)
- **Ergebnis: Alle Vorhersagen waren falsch**
  - Hinweis auf begrenzte Generalisierbarkeit des Modells
  - Wahrscheinlich durch nahe Ausschnitte / fehlenden Kontext im Bild
- Keine Nachbesserung geplant – diese Schwäche wird im README dokumentiert
- Modell bleibt als funktionsfähiger Prototyp zur Demonstration bestehen


Stand 27.07.25

# Prüfungsaufgabe 2: Unit-Testing und Logging für Deep Learning

Meine Umsetzung der Prüfungsaufgabe 2 im Modul "Data Science und Engineering mit Python". Als Grundlage dient die Deep Learning Übung (Banknoten-Klassifizierung) aus dem Udemy-Kurs.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/7-Aufgabe2-Deep-Learning/main?filepath=Deep_Learning_Solution.ipynb)

## Überblick

Vorhersage, ob eine Banknote echt oder gefälscht ist, anhand von Merkmalen aus Bildern. Modell: MLPClassifier / Neuronales Netz (scikit-learn).

Nach dem Ansatz aus dem Artikel [Unit Testing and Logging for Data Science](https://medium.com/data-science/unit-testing-and-logging-for-data-science-d7fb8fd5d) habe ich `my_logger` und `my_timer` Funktionen implementiert und damit zwei Python Unit-Tests geschrieben.

## Dateien

| Datei | Beschreibung |
|---|---|
| `Deep_Learning_Solution.ipynb` | Haupt-Notebook mit der Implementierung des neuronalen Netzes |
| `bank_note_data.csv` | Original-Datensatz (1372 Einträge) |
| `model_logic.py` | Kernlogik (StandardScaler + MLPClassifier) mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Python Unit-Tests für `predict()` und `fit()` |
| `generate_test_data.py` | Skript zur Erzeugung der Trainings- und Testdaten |
| `train_data.csv` | Trainingsdaten (960 Zeilen) |
| `test_data.csv` | Testdaten (412 Zeilen) |
| `training.log` | Log-File mit Trainingsereignissen |

## Logging

Die Funktionen `my_logger` und `my_timer` aus dem Artikel sind als Dekoratoren in `model_logic.py` umgesetzt. Sie loggen Funktionsaufrufe (Argumente) und Ausführungszeiten in `training.log`.

```python
@my_logger
@my_timer
def fit_model(X_train, y_train):
    ...
```

Beispiel-Ausgabe in `training.log`:
```text
2026-02-18 14:30:01,123 - INFO - Ran with args: (...), and kwargs: {}
2026-02-18 14:30:01,456 - INFO - fit_model ran in: 0.1607 sec
```

## Testfälle

**Testfall 1 - predict():** Das neuronale Netz wird auf `train_data.csv` trainiert und die Accuracy auf `test_data.csv` geprüft. Ziel: Accuracy > 0.95.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der repräsentativen Normzeit (0.5s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 1.0000
.
[Test fit()] Gemessene Dauer: 0.1607s (Limit: 0.6000s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.330s

OK
```

## Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`

## Notebook ausführen

1. Auf den **Binder-Badge** oben klicken.
2. `Deep_Learning_Solution.ipynb` öffnen.
3. Alle Zellen ausführen (*Run > Run All Cells*).
4. **Erwartete Ergebnisse:**
   - Analyse der Banknoten-Merkmale
   - Feature-Skalierung mit StandardScaler
   - Training des neuronalen Netzes
   - Classification Report mit sehr hoher Precision und Recall
   - Accuracy von ca. **0.99 - 1.00**

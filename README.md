#  FHIR Laboratory Dashboard

Ein Streamlit-Dashboard zur Anzeige und Analyse von Laborwerten aus einem FHIR-Server.

Die Anwendung lädt Patienten und deren Observations, filtert Laborwerte und zeigt deren zeitlichen Verlauf interaktiv an.

---

## Funktionen

- Verbindung zu einem FHIR Server
- Patienten auswählen
- Laborwerte anzeigen
- Zeitverlauf als Diagramm
- Filter nach Datum, Jahr und Monat
- Automatische oder manuelle Unterbrechung von Messreihen
- Export der Daten als CSV

---

## Installation

Python installieren (empfohlen: ≥ 3.9)

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt

Dann die Anwendung starten mit:
streamlit run timeseries_dashboard.py


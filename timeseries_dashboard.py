import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import date
from typing import Any, Dict, List

# ----------------------------
# Page / App config
# ----------------------------
st.set_page_config(page_title="FHIR Time Series Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🧪 Laboratory Time Series Dashboard (FHIR)")

# ============================
# Helper: robust FHIR fetch (pagination + retry)
# ============================
def _safe_fhir_fetch(url: str, resource_type: str) -> List[Dict[str, Any]]:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    accumulated: List[Dict[str, Any]] = []
    current_url = url

    while current_url:
        try:
            resp = session.get(current_url, headers={"Accept": "application/fhir+json"}, timeout=30)
            resp.raise_for_status()
            # .json() parst diesen JSON-Text in eine native Python-Datenstruktur (in diesem Fall ein dict, also ein Dictionary)
            bundle = resp.json() 
        except Exception as e:
            st.error(f"Fehler beim Abruf von {current_url}: {e}")
            break
        # prüfe ob es sich um ein Bundle handelt
        if isinstance(bundle, dict) and bundle.get("resourceType") == "Bundle" and "entry" in bundle:
            for entry in bundle["entry"]:
                resource = entry.get("resource")
                if isinstance(resource, dict) and resource.get("resourceType") == resource_type:
                    accumulated.append(resource)
        #Paginierungs-Logik
        next_url = None
        if isinstance(bundle, dict):
            for link in bundle.get("link", []) or []:
                if link.get("relation") == "next" and link.get("url"):
                    next_url = link["url"]
                    break
        current_url = next_url

    return accumulated

@st.cache_data(ttl=300)
def safe_fhir_fetch_cached(url: str, resource_type: str) -> List[Dict[str, Any]]:
    return _safe_fhir_fetch(url, resource_type)

# ============================
# Filter: Nur Laborwerte
# ============================
def is_labor_observation(resource: Dict[str, Any]) -> bool:
    """Filtert Observations, die als Laborwerte gelten."""
    if not isinstance(resource, dict):
        return False
    #Ein Laborwert braucht ein Datum
    if not resource.get("issued"):
        return False

    # Prüfe valueQuantity (direkt oder in Komponenten)
    #Prüft, ob die Observation direkt einen numerischen Wert hat
    has_numeric = False
    vq = resource.get("valueQuantity")
    if isinstance(vq, dict) and vq.get("value") is not None:
        has_numeric = True
    else:
        for comp in resource.get("component", []) or []:
            vqc = comp.get("valueQuantity")
            if isinstance(vqc, dict) and vqc.get("value") is not None:
                has_numeric = True
                break
    if not has_numeric:
        return False

    # Kategorie prüfen
    cats = []
    for cat in resource.get("category", []) or []:
        for c in (cat.get("coding") or []):
            cats.append((c.get("code") or "").lower())
            cats.append((c.get("display") or "").lower())

    is_lab = any("laboratory" in c for c in cats)
    return is_lab

# ============================
# Normalisierung für Laborwerte
# ============================
def normalize_fhir_observations(obs_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    def _to_float(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", ".").strip()
        try:
            return float(s)
        except Exception:
            return None

    for res in obs_list:
        if not is_labor_observation(res):
            continue

        obs_name = None
        for c in (res.get("code", {}).get("coding") or []):
            if isinstance(c, dict):
                obs_name = c.get("display") or c.get("code") or obs_name
        obs_name = obs_name or res.get("code", {}).get("text") or "Unknown Observation"

        issued = res.get("issued")

        # Hauptwert
        vq = res.get("valueQuantity")
        if isinstance(vq, dict) and vq.get("value") is not None:
            rows.append({
                "id": res.get("id"),
                "observation_name": obs_name,
                "value": _to_float(vq.get("value")),
                "unit": vq.get("unit") or vq.get("code"),
                "date": issued,
            })

        # Komponenten (z. B. Panelwerte)
        for comp in res.get("component", []) or []:
            comp_name = None
            for cc in (comp.get("code", {}).get("coding") or []):
                if isinstance(cc, dict):
                    comp_name = cc.get("display") or cc.get("code") or comp_name
            comp_name = comp_name or comp.get("code", {}).get("text")

            vqc = comp.get("valueQuantity")
            if isinstance(vqc, dict) and vqc.get("value") is not None:
                rows.append({
                    "id": res.get("id"),
                    "observation_name": comp_name or obs_name,
                    "value": _to_float(vqc.get("value")),
                    "unit": vqc.get("unit") or vqc.get("code"),
                    "date": issued,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    #Konvertiert die Datums-Spalte, die derzeit aus Text besteht, in echte Datums-Objekte.
    #errors="coerce": Wenn ein Datum ungültig ist (z. B. "Text"), wird es in NaT (Not a Time) umgewandelt, anstatt einen Fehler zu werfen.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    #Datenbereinigung: Entfernt Zeilen mit fehlenden Daten in den "date"- oder "value"-Spalten und sortiert die Daten nach Datum.
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df

# ============================
# Sidebar / Inputs
# ============================
st.sidebar.title("⚙️ Konfiguration")
fhir_base = st.sidebar.text_input("FHIR Server Base URL:", value="http://localhost:8081/fhir")
fetch_btn = st.sidebar.button("Fetch Patients")

st.sidebar.markdown("---")
st.sidebar.subheader("Startdatum (optional)")
start_date = st.sidebar.date_input("Startdatum (ab):", value=None)

# ============================
# Fetch patients
# ============================
if fetch_btn:
    fetch_url = f"{fhir_base}/Patient"
    if start_date:
        sep = "?" if "?" not in fetch_url else "&"
        fetch_url = f"{fetch_url}{sep}_lastUpdated=gt{start_date.isoformat()}"
    with st.spinner("Hole Patienten..."):
        patients = safe_fhir_fetch_cached(fetch_url, "Patient")
    if patients:
        st.success(f"{len(patients)} Patienten geladen.")
        #session_state speichern, so dass Patients für die gesamte Sitzung des Benutzers verfügbar bleiben.
        #json_normalize: Wandelt verschachtelte JSON-Daten in ein flaches Tabellenformat um.(name.family, name.given)
        st.session_state["patients"] = pd.json_normalize(patients, sep=".")
        dfp = st.session_state["patients"]
        if "name" in dfp.columns:
            def _extract_name(n):
                if isinstance(n, list) and n:
                    first = n[0]
                    family = first.get("family")
                    given = first.get("given", [None])[0] if isinstance(first.get("given"), list) else first.get("given")
                    return pd.Series([family, given])
                return pd.Series([None, None])
            names = dfp["name"].apply(_extract_name)
            names.columns = ["name.family", "name.given"]
            st.session_state["patients"] = pd.concat([dfp.drop(columns=["name"], errors="ignore"), names], axis=1)
    else:
        st.warning("Keine Patienten gefunden oder Abruf fehlgeschlagen.")
        st.session_state["patients"] = pd.DataFrame()

if "patients" not in st.session_state:
    st.session_state["patients"] = pd.DataFrame()

if st.session_state["patients"].empty:
    st.info("Bitte zuerst Patienten laden (Button in der Sidebar).")
    st.stop()

# ============================
# Patient selection
# ============================
df_pat = st.session_state["patients"].copy()
patient_count = len(df_pat)

# KPI oben: Wie viele Patienten erfasst
st.markdown("### 👥 Patientenübersicht")
col1, col2 = st.columns([1, 3])
col1.metric("📋 Anzahl erfasster Patienten", f"{patient_count}")
col2.write("Wähle unten einen Patienten aus, um Laborwerte anzuzeigen.")

if "name.given" in df_pat.columns and "name.family" in df_pat.columns:
    df_pat["full_name"] = (df_pat["name.given"].fillna("") + " " + df_pat["name.family"].fillna("")).str.strip()
else:
    df_pat["full_name"] = df_pat.get("id", "").astype(str)
df_pat["display"] = df_pat.apply(lambda r: f"{r['full_name']} (ID:{str(r.get('id',''))[:8]})", axis=1)

patient_choice = st.sidebar.selectbox("Patient auswählen:", df_pat["display"].tolist())
selected = df_pat[df_pat["display"] == patient_choice].iloc[0]
patient_id = selected.get("id")

st.subheader(f"🧍 Patient: {selected.get('full_name','-')} — ID: {patient_id}")

# ============================
# Fetch Observations (nur Laborwerte)
# ============================
obs_url = f"{fhir_base}/Observation?subject=Patient/{patient_id}"
with st.spinner("Hole Observations..."):
    observations = safe_fhir_fetch_cached(obs_url, "Observation")

if not observations:
    st.warning("Keine Observations gefunden.")
    st.stop()

# Labor-Filterung & Statistik
total_count = len(observations)
lab_obs = [o for o in observations if is_labor_observation(o)]
lab_count = len(lab_obs)
excluded = total_count - lab_count
percent = (lab_count / total_count * 100) if total_count > 0 else 0
st.info(f"Gesamt: {total_count} Observations | Laborwerte: {lab_count} ({percent:.1f}%) | Ausgeschlossen: {excluded}")

df_obs = normalize_fhir_observations(lab_obs)
if df_obs.empty:
    st.warning("Keine Laborwerte mit numerischem valueQuantity gefunden.")
    st.stop()

# ============================
# Auswahl der Observations
# ============================
available = sorted(df_obs["observation_name"].dropna().unique().tolist())
selected_obs = st.sidebar.multiselect("Laborwerte auswählen:", options=available, default=[])

if not selected_obs:
    st.info("Bitte mindestens einen Laborwert auswählen (Sidebar).")
    st.stop()

df_filtered = df_obs[df_obs["observation_name"].isin(selected_obs)].copy()
if start_date:
    df_filtered = df_filtered[df_filtered["date"].dt.date >= start_date]

if df_filtered.empty:
    st.warning("Keine Daten nach Filterung vorhanden.")
    st.stop()

# ============================
# KPIs
# ============================
total_count = len(df_filtered)
unique_obs = int(df_filtered["observation_name"].nunique())

k1, k2= st.columns([1, 1])
k1.metric("🔢 Messwerte (gefiltert)", f"{total_count}")
k2.metric("🧪 Laborwerte", f"{unique_obs}")

# ============================
# Statistik pro Observation
# ============================
stats = df_filtered.groupby("observation_name")["value"].agg(count="count", mean="mean", min="min", max="max").reset_index()
stats[["mean", "min", "max"]] = stats[["mean", "min", "max"]].round(2)
st.markdown("### 📊 Statistik pro Laborwert")
st.dataframe(stats, width=True)

# ============================
# Plot (mit korrekt formatiertem Hover)
# ============================
fig = px.line(
    df_filtered,
    x="date",
    y="value",
    color="observation_name",
    markers=True,
    labels={"date": "Datum", "value": "Wert", "observation_name": "Laborwert"},
    title="Zeitverlauf der ausgewählten Laborwerte",
)
fig.update_traces(
    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y} %{customdata[0]}<br>%{fullData.name}",
    customdata=df_filtered[["unit"]].values
)
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=30, label="30d", step="day", stepmode="backward"),
            dict(count=90, label="90d", step="day", stepmode="backward"),
            dict(step="all")
        ]),
        type="date"
    ),
    legend_title_text="Laborwert"
)
st.plotly_chart(fig, width=True)

# ============================
# Rohdaten + Export
# ============================
st.markdown("### 🗂️ Gefilterte Rohdaten")
st.dataframe(df_filtered.sort_values("date").reset_index(drop=True), width=True)
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("📥 CSV herunterladen", csv, file_name=f"laborwerte_patient_{patient_id}.csv", mime="text/csv")

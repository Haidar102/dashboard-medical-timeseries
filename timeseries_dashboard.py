import streamlit as st
import pandas as pd
import plotly.express as px
import json
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import date

# --- Page Setup ---
st.set_page_config(page_title="FHIR Time Series Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("⏱️ Time Series Dashboard (FHIR)")

# ==========================================
# FHIR FETCH
# ==========================================
def _safe_fhir_fetch(url, resource_type):
    """Ruft FHIR-Daten ab und folgt Pagination."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    accumulated = []
    current_url = url

    while current_url:
        try:
            response = session.get(current_url, headers={"Accept": "application/fhir+json"}, timeout=30)
            response.raise_for_status()
            bundle = response.json()

            if bundle.get("resourceType") == "Bundle" and "entry" in bundle:
                for entry in bundle["entry"]:
                    resource = entry.get("resource")
                    if resource and resource.get("resourceType") == resource_type:
                        accumulated.append(resource)

            # Pagination
            next_url = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_url = link["url"]
                    break
            current_url = next_url

        except Exception as e:
            st.error(f"Fehler beim Abruf: {e}")
            break

    return accumulated


@st.cache_data(ttl=300)
def safe_fhir_fetch_cached(url, resource_type):
    return _safe_fhir_fetch(url, resource_type)

# ==========================================
# PATIENT NORMALIZATION
# ==========================================
def normalize_fhir_patients(patients_list):
    if not patients_list:
        return pd.DataFrame()

    df = pd.json_normalize(patients_list, sep='.')

    # Name extrahieren
    def extract_name(name_list):
        if isinstance(name_list, list) and name_list:
            n = name_list[0]
            family = n.get('family')
            given = n.get('given', [None])[0] if isinstance(n.get('given'), list) else n.get('given')
            return pd.Series([family, given], index=['name.family', 'name.given'])
        return pd.Series([None, None], index=['name.family', 'name.given'])

    if 'name' in df.columns:
        name_comp = df['name'].apply(extract_name)
        df = pd.concat([df.drop('name', axis=1), name_comp], axis=1)

    df = df.rename(columns={'meta.lastUpdated': 'lastUpdated'})
    df = df.drop(columns=['meta','identifier','meta.versionId','managingOrganization.reference'], errors='ignore')

    cols = ['id', 'name.family', 'name.given', 'birthDate', 'gender', 'lastUpdated']
    others = [c for c in df.columns if c not in cols]
    df = df[[c for c in cols if c in df.columns] + others]
    return df

# ==========================================
# OBSERVATION NORMALIZATION
# ==========================================
def normalize_fhir_observations(obs_list):
    if not obs_list:
        return pd.DataFrame()

    df = pd.json_normalize(obs_list, sep='.')

    # Namen extrahieren
    def obs_name(row):
        if isinstance(row.get('code.coding'), list) and row['code.coding']:
            c = row['code.coding'][0]
            return c.get('display') or c.get('code') or row.get('code.text')
        return row.get('code.text')

    df['observation_name'] = df.apply(obs_name, axis=1)

    # Werte extrahieren
    def extract_val(row):
        vq = row.get('valueQuantity')
        if isinstance(vq, dict):
            return pd.Series([vq.get('value'), vq.get('unit')], index=['value', 'unit'])
        return pd.Series([
            row.get('valueQuantity.value') or row.get('value'),
            row.get('valueQuantity.unit') or row.get('unit')
        ], index=['value', 'unit'])

    vals = df.apply(extract_val, axis=1)
    df = pd.concat([df, vals], axis=1)

    # Datum extrahieren
    def get_date(row):
        for k in ['issued', 'effectiveDateTime', 'effective.dateTime']:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None

    df['date'] = pd.to_datetime(df.apply(get_date, axis=1), errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df = df.drop(columns=['code','meta','subject','resourceType','status','valueQuantity'], errors='ignore')
    df = df[['id','observation_name','value','unit','date'] + [c for c in df.columns if c not in ['id','observation_name','value','unit','date']]]
    return df

# ==========================================
# SIDEBAR CONFIG
# ==========================================
st.sidebar.title("⚙️ Konfiguration")
fhir_base = st.sidebar.text_input("FHIR Server Base URL:", value="http://localhost:8081/fhir")
fetch_btn = st.sidebar.button("Fetch Patients")

st.sidebar.markdown("---")
st.sidebar.subheader("Zeitraum-Filter (optional)")
start_date = st.sidebar.date_input("Startdatum (ab):", value=None)

# ==========================================
# FETCH PATIENTS
# ==========================================
if fetch_btn:
    fetch_url = f"{fhir_base}/Patient"
    if start_date:
        fetch_url += f"?_lastUpdated=gt{start_date.isoformat()}"

    with st.spinner("Lade Patientendaten..."):
        patients = safe_fhir_fetch_cached(fetch_url, "Patient")

    if patients:
        st.success(f"{len(patients)} Patienten abgerufen.")
        st.session_state["patients"] = normalize_fhir_patients(patients)
    else:
        st.warning("Keine Patienten gefunden.")

if "patients" not in st.session_state:
    st.session_state["patients"] = pd.DataFrame()

# ==========================================
# WENN PATIENTEN VORHANDEN
# ==========================================
if not st.session_state["patients"].empty:
    df_pat = st.session_state["patients"].copy()

    df_pat["full_name"] = (
        df_pat["name.given"].fillna("") + " " + df_pat["name.family"].fillna("")
    ).str.strip()

    df_pat["display"] = df_pat.apply(
        lambda r: f"{r['full_name']} (ID: {str(r['id'])[:8]})", axis=1
    )

    patient_choice = st.sidebar.selectbox("Patient auswählen:", df_pat["display"])

    sel_row = df_pat[df_pat["display"] == patient_choice].iloc[0]
    sel_id = sel_row["id"]

    st.subheader(f"Patient: {sel_row['full_name']} — ID: {sel_id}")

    obs_url = f"{fhir_base}/Observation?subject=Patient/{sel_id}"
    st.info(f"Abrufe Observations von: `{obs_url}`")

    with st.spinner("Lade Observations..."):
        observations = safe_fhir_fetch_cached(obs_url, "Observation")

    if not observations:
        st.warning("Keine Observations gefunden.")
        st.stop()

    df_obs = normalize_fhir_observations(observations)
    if df_obs.empty:
        st.warning("Keine gültigen Observationsdaten.")
        st.stop()

    # ---------------------------------
    # Filter: Observation & Zeitraum
    # ---------------------------------
    available = sorted(df_obs["observation_name"].dropna().unique().tolist())
    selected_obs = st.sidebar.multiselect(
        "Observation-Typen:", options=available, default=available[:3]
    )

    if selected_obs:
        df_filtered = df_obs[df_obs["observation_name"].isin(selected_obs)]
    else:
        df_filtered = df_obs.copy()

    if start_date:
        df_filtered = df_filtered[df_filtered["date"].dt.date >= start_date]

    df_filtered = df_filtered.dropna(subset=["date", "value"])

    if df_filtered.empty:
        st.warning("Keine Daten im gewählten Zeitraum.")
        st.stop()

    # ==========================================
    # PLOT
    # ==========================================
    df_filtered = df_filtered.sort_values("date")

    fig = px.line(
        df_filtered,
        x="date",
        y="value",
        color="observation_name",
        markers=True,
        title="Zeitverlauf der Observations",
        labels={"date": "Datum", "value": "Wert", "observation_name": "Observation"},
    )
    fig.update_traces(
        hovertemplate="%{x}<br>%{y} %{customdata[0]}<br>%{fullData.name}",
        customdata=df_filtered[["unit"]].values,
    )
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(count=90, label="90d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TABELLENANSICHT + EXPORT
    # ==========================================
    st.subheader("📊 Datenansicht")
    st.dataframe(df_filtered, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 CSV herunterladen",
        csv,
        file_name=f"observations_patient_{sel_id}.csv",
        mime="text/csv",
    )

else:
    st.info("Bitte zuerst Patienten laden (Button in der Sidebar).")

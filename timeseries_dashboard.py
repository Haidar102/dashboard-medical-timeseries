import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from requests.adapters import HTTPAdapter, Retry
from datetime import date

# --- Page ---
st.set_page_config(page_title="FHIR Time Series Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("⏱️ Time Series Dashboard (FHIR)")

# ==========================================
# FHIR FETCH (Pagination + Retry)
# ==========================================
def _safe_fhir_fetch(url, resource_type):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500,502,503,504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    accumulated = []
    current_url = url

    while current_url:
        try:
            resp = session.get(current_url, headers={"Accept":"application/fhir+json"}, timeout=30)
            resp.raise_for_status()
            bundle = resp.json()

            if bundle.get("resourceType") == "Bundle" and "entry" in bundle:
                for entry in bundle["entry"]:
                    res = entry.get("resource")
                    if res and res.get("resourceType") == resource_type:
                        accumulated.append(res)

            # pagination next link
            next_url = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next" and link.get("url"):
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
# Normalizer: Patients
# ==========================================
def normalize_fhir_patients(patients_list):
    if not patients_list:
        return pd.DataFrame()
    df = pd.json_normalize(patients_list, sep='.')
    def extract_name(n):
        if isinstance(n, list) and n:
            first = n[0]
            family = first.get('family')
            given = first.get('given')[0] if isinstance(first.get('given'), list) and first.get('given') else first.get('given')
            return pd.Series([family, given], index=['name.family','name.given'])
        return pd.Series([None, None], index=['name.family','name.given'])
    if 'name' in df.columns:
        name_comp = df['name'].apply(extract_name)
        df = pd.concat([df.drop('name',axis=1), name_comp], axis=1)
    df = df.rename(columns={'meta.lastUpdated':'lastUpdated'})
    df = df.drop(columns=['meta','identifier','meta.versionId','managingOrganization.reference'], errors='ignore')
    priority = ['id','name.family','name.given','birthDate','gender','lastUpdated']
    others = [c for c in df.columns if c not in priority]
    return df[[c for c in priority if c in df.columns] + others]

# ==========================================
# Normalizer: Observations
# ==========================================
def normalize_fhir_observations(obs_list):
    if not obs_list:
        return pd.DataFrame()
    df = pd.json_normalize(obs_list, sep='.')

    def obs_name(row):
        cod = row.get('code.coding')
        if isinstance(cod, list) and cod:
            first = cod[0]
            if isinstance(first, dict):
                return first.get('display') or first.get('code') or row.get('code.text')
        return row.get('code.text') or None

    df['observation_name'] = df.apply(obs_name, axis=1)

    def extract_val(row):
        vq = row.get('valueQuantity')
        if isinstance(vq, dict):
            return pd.Series([vq.get('value'), vq.get('unit')], index=['value','unit'])
        return pd.Series([
            row.get('valueQuantity.value') or row.get('value'),
            row.get('valueQuantity.unit') or row.get('unit')
        ], index=['value','unit'])

    vals = df.apply(extract_val, axis=1)
    df = pd.concat([df, vals], axis=1)

    def get_date(row):
        for k in ('issued','effectiveDateTime','effective.dateTime'):
            if k in row and pd.notna(row[k]):
                return row[k]
        return None

    df['date'] = pd.to_datetime(df.apply(get_date, axis=1), errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df = df.drop(columns=['code','meta','subject','resourceType','status','valueQuantity'], errors='ignore')
    cols = ['id','observation_name','value','unit','date']
    others = [c for c in df.columns if c not in cols]
    return df[[c for c in cols if c in df.columns] + others]

# ==========================================
# Sidebar: Config + Fetch
# ==========================================
st.sidebar.title("⚙️ Konfiguration")
fhir_base = st.sidebar.text_input("FHIR Server Base URL:", value="http://localhost:8081/fhir")
fetch_btn = st.sidebar.button("Fetch Patients")

st.sidebar.markdown("---")
st.sidebar.subheader("Startdatum (optional)")
start_date = st.sidebar.date_input("Startdatum (ab):", value=None)

# Fetch patients on demand
if fetch_btn:
    fetch_url = f"{fhir_base}/Patient"
    if start_date:
        sep = "?" if "?" not in fetch_url else "&"
        fetch_url = f"{fetch_url}{sep}_lastUpdated=gt{start_date.isoformat()}"
    with st.spinner("Hole Patienten..."):
        patients = safe_fhir_fetch_cached(fetch_url, "Patient")
    if patients:
        st.success(f"{len(patients)} Patienten geladen.")
        st.session_state['patients'] = normalize_fhir_patients(patients)
    else:
        st.warning("Keine Patienten gefunden oder Abruf fehlgeschlagen.")

if 'patients' not in st.session_state:
    st.session_state['patients'] = pd.DataFrame()

# ==========================================
# Main: wenn Patienten vorhanden
# ==========================================
if not st.session_state['patients'].empty:
    df_pat = st.session_state['patients'].copy()
    # full name
    if 'name.given' in df_pat.columns and 'name.family' in df_pat.columns:
        df_pat['full_name'] = (df_pat['name.given'].fillna('') + ' ' + df_pat['name.family'].fillna('')).str.strip()
    else:
        df_pat['full_name'] = df_pat.get('id', '').astype(str)

    df_pat['display'] = df_pat.apply(lambda r: f"{r['full_name']} (ID:{str(r['id'])[:8]})", axis=1)
    patient_choice = st.sidebar.selectbox("Patient auswählen:", df_pat['display'].tolist())
    selected_row = df_pat[df_pat['display'] == patient_choice].iloc[0]
    patient_id = selected_row['id']

    st.subheader(f"Patient: {selected_row['full_name']} — ID: {patient_id}")

    # fetch observations
    obs_url = f"{fhir_base}/Observation?subject=Patient/{patient_id}"
    st.info(f"Abrufe Observations: `{obs_url}`")
    with st.spinner("Hole Observations..."):
        observations = safe_fhir_fetch_cached(obs_url, "Observation")

    if not observations:
        st.warning("Keine Observations für diesen Patienten.")
        st.stop()

    df_obs = normalize_fhir_observations(observations)
    if df_obs.empty:
        st.warning("Keine verwertbaren Observations.")
        st.stop()

    # Filter: observation types
    available = sorted(df_obs['observation_name'].dropna().unique().tolist())
    selected_obs = st.sidebar.multiselect("Observation-Typen:", options=available, default=available[:3] if available else [])

    if selected_obs:
        df_filtered = df_obs[df_obs['observation_name'].isin(selected_obs)].copy()
    else:
        df_filtered = df_obs.copy()

    # optionaler Start-Datum-Filter (wie ursprünglich: nur applied wenn gesetzt)
    if start_date:
        # Vergleiche nur datum ohne timezone
        df_filtered = df_filtered.dropna(subset=['date']).copy()
        df_filtered = df_filtered[df_filtered['date'].dt.date >= start_date]

    # drop rows ohne date oder value
    df_filtered = df_filtered.dropna(subset=['date','value']).copy()

    if df_filtered.empty:
        st.warning("Keine Daten nach Filtern vorhanden.")
        st.stop()

    # --------------------------
    # KPI Bereich
    # --------------------------
    total_count = len(df_filtered)
    unique_obs = df_filtered['observation_name'].nunique()
    first_date = df_filtered['date'].min()
    last_date = df_filtered['date'].max()
    overall_mean = df_filtered['value'].mean()

    k1, k2, k3, k4 = st.columns([1.2,1,1,1])
    k1.metric("🔢 Messwerte (Zeilen)", value=f"{total_count}")
    k2.metric("📌 Observation-Typen", value=f"{unique_obs}")
    k3.metric("🗓️ Erstes Datum", value=first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else "—")
    k4.metric("🗓️ Letztes Datum", value=last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else "—")

    # kleine zweite Zeile mit Durchschnitt (format)
    c1, c2 = st.columns([1,2])
    c1.metric("📊 Durchschnittswert (gesamt)", value=f"{overall_mean:.2f}" if pd.notna(overall_mean) else "—")
    # Anzeige: kurze Beschreibung
    c2.write("Filter angewendet: " + (", ".join(selected_obs) if selected_obs else "alle") + (f"; ab {start_date}" if start_date else ""))

    # --------------------------
    # Per-Observation Statistik (count, mean, min, max)
    # --------------------------
    stats = df_filtered.groupby('observation_name')['value'].agg(['count','mean','min','max']).reset_index()
    stats['mean'] = stats['mean'].round(2)
    stats['min'] = stats['min'].round(2)
    stats['max'] = stats['max'].round(2)
    st.markdown("### 🔎 Statistik pro Observation")
    st.dataframe(stats, use_container_width=True)

    # --------------------------
    # Plot
    # --------------------------
    df_plot = df_filtered.sort_values('date')
    fig = px.line(
        df_plot,
        x='date',
        y='value',
        color='observation_name',
        markers=True,
        title="Zeitverlauf der ausgewählten Observations",
        labels={'date':'Datum','value':'Wert','observation_name':'Observation'}
    )
    if 'unit' in df_plot.columns:
        fig.update_traces(hovertemplate='%{x}<br>%{y} %{customdata[0]}<br>%{fullData.name}', customdata=df_plot[['unit']].values)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=7,label="7d",step="day",stepmode="backward"),
                dict(count=30,label="30d",step="day",stepmode="backward"),
                dict(count=90,label="90d",step="day",stepmode="backward"),
                dict(step="all")
            ]),
            type="date"
        ),
        legend_title_text='Observation'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # Datenansicht + Export
    # --------------------------
    st.markdown("### 🗂️ Gefilterte Rohdaten")
    st.dataframe(df_filtered.sort_values('date').reset_index(drop=True), use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 CSV herunterladen", csv, file_name=f"observations_patient_{patient_id}.csv", mime='text/csv')

else:
    st.info("Bitte zuerst Patienten laden (Button in der Sidebar).")

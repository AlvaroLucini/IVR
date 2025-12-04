#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import pandas as pd
import streamlit as st
import altair as alt

# =========================
# RUTAS BÁSICAS
# =========================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BASE_DIR / "test_results"              # carpeta con los JSON
CSV_SCENARIOS = BASE_DIR / "config" / "scenarios.csv" # config de escenarios

st.set_page_config(
    page_title="IVR Tester - Estadísticas escenarios",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Resultados IVR por escenario")
st.caption(
    "📈 Para cada escenario, una barra verde (éxitos) y una roja (fallos), "
    "a partir de los JSON de test_results."
)

# =========================
# HELPERS
# =========================

def format_seconds_hhmmss(seconds: float | int | None) -> str:
    """Convierte segundos (float) a HH:MM:SS."""
    if seconds is None:
        return "N/A"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "N/A"
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# =========================
# LOOKUP DE ESCENARIOS
# =========================

def load_scenarios_lookup() -> pd.DataFrame:
    """
    Lee config/scenarios.csv y devuelve:
    - scenario_id
    - scenario_title (TITLE)
    - mission_text  (MISSION_TEXT)
    solo para escenarios ACTIVE=TRUE (si existe la columna).
    """
    if not CSV_SCENARIOS.exists():
        st.warning(f"No se ha encontrado el archivo de escenarios: {CSV_SCENARIOS}")
        return pd.DataFrame()

    df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")

    if "ACTIVE" in df.columns:
        df = df[df["ACTIVE"].str.upper() == "TRUE"].copy()

    if df.empty:
        return pd.DataFrame()

    df["SCENARIO_ID"] = df["SCENARIO_ID"].astype(str)

    out = pd.DataFrame({
        "scenario_id": df["SCENARIO_ID"],
        "scenario_title": df.get("TITLE", ""),
        "mission_text": df.get("MISSION_TEXT", ""),
    })

    return out


scenarios_lookup = load_scenarios_lookup()

# =========================
# CARGA DE TODOS LOS JSON
# =========================

def load_all_results() -> pd.DataFrame:
    if not RESULTS_ROOT.exists():
        st.warning(f"No existe la carpeta de resultados: {RESULTS_ROOT}")
        return pd.DataFrame()

    records = []

    for day_dir in sorted(RESULTS_ROOT.iterdir()):
        if not day_dir.is_dir():
            continue

        for f in day_dir.glob("*.json"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                st.warning(f"No se pudo leer {f}: {e}")
                continue

            records.append({
                "test_id":           data.get("test_id", ""),
                "timestamp_utc":     data.get("timestamp_utc", ""),
                "scenario_id":       data.get("scenario_id", ""),
                "scenario_title":    data.get("scenario_title", ""),
                "result":            data.get("result", ""),
                "expected_queue_id": data.get("expected_queue_id", ""),
                "reached_queue_id":  data.get("reached_queue_id", ""),
                "end_node_id":       data.get("end_node_id", ""),
                "end_node_type":     data.get("end_node_type", ""),
                "duration_seconds":  data.get("duration_seconds", None),
                "num_steps":         data.get("num_steps", None),
                "__source_file":     str(f.relative_to(BASE_DIR)),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    df["scenario_id"] = df["scenario_id"].astype(str)
    df["result"] = df["result"].astype(str).str.upper().str.strip()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    return df


df = load_all_results()

# =========================
# SOLO BOTÓN EN LA BARRA IZQUIERDA
# =========================

with st.sidebar:
    if st.button("🔄 Recargar resultados"):
        st.rerun()

# =========================
# SI NO HAY DATOS
# =========================

if df.empty:
    st.info("Todavía no hay resultados para mostrar.")
    st.stop()

# =========================
# MAPEO RESULTADOS → Éxito / Fallo
# =========================

SUCCESS_VALUES = {
    "OK",
    "SUCCESS",
    "EXPECTED_QUEUE",
    "CORRECT_QUEUE",
    "RIGHT_QUEUE",
}

def map_result_label(x: str) -> str:
    s = str(x).upper().strip()
    if s in SUCCESS_VALUES:
        return "Éxito"
    return "Fallo"

df["resultado_label"] = df["result"].apply(map_result_label)

# =========================
# AGREGADO POR ESCENARIO
# =========================

agg = (
    df.groupby(["scenario_id", "resultado_label"])
      .size()
      .reset_index(name="count")
)

# =========================
# KPI TESTS Y ESCENARIOS
# =========================

# --- KPIs de TESTS ---
total_tests = int(len(df))
tests_success = int((df["resultado_label"] == "Éxito").sum())
tests_fail = total_tests - tests_success

st.markdown("### Resumen global (tests)")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("🔵 Tests totales", total_tests)
with c2:
    st.metric("🟢 Tests con éxito", tests_success)
with c3:
    st.metric("🔴 Tests con fallo", tests_fail)

# --- Duración media ---
dur_series = df["duration_seconds"]
if dur_series.notna().any():
    avg_seconds = float(dur_series.dropna().mean())
else:
    avg_seconds = None

avg_str = format_seconds_hhmmss(avg_seconds)

st.markdown("### Métricas de duración")

c_dur, _, _ = st.columns(3)
with c_dur:
    st.metric("⏱ Duración media de los tests (HH:MM:SS)", avg_str)

st.markdown("---")

# --- KPIs de ESCENARIOS ---
total_scenarios = int(df["scenario_id"].nunique())
scenarios_with_success = int(
    agg[agg["resultado_label"] == "Éxito"]["scenario_id"].nunique()
)
scenarios_without_success = max(total_scenarios - scenarios_with_success, 0)

st.markdown("### Resumen por escenarios")

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("🔵 Escenarios ejecutados", total_scenarios)
with c5:
    st.metric("🟢 Escenarios con algún éxito", scenarios_with_success)
with c6:
    st.metric("🔴 Escenarios sin ningún éxito", scenarios_without_success)

st.markdown("---")

# =========================
# GRÁFICA DE BARRAS
# =========================

st.subheader("Resultados por escenario")

max_count = int(agg["count"].max())

chart = (
    alt.Chart(agg)
    .mark_bar()
    .encode(
        x=alt.X("scenario_id:N", title="Escenario"),
        xOffset="resultado_label:N",
        y=alt.Y(
            "count:Q",
            title="Número de tests",
            scale=alt.Scale(domain=(0, max_count + 0.5)),
            axis=alt.Axis(
                values=list(range(0, max_count + 1)),
                format="d",
            ),
        ),
        color=alt.Color(
            "resultado_label:N",
            scale=alt.Scale(
                domain=["Fallo", "Éxito"],
                range=["red", "green"],
            ),
            legend=alt.Legend(title="Resultado"),
        ),
        tooltip=[
            alt.Tooltip("scenario_id:N", title="Escenario"),
            alt.Tooltip("resultado_label:N", title="Resultado"),
            alt.Tooltip("count:Q", title="Nº de tests"),
        ],
    )
    .properties(height=500)
)

st.altair_chart(chart, width="stretch")

# =========================
# TABLA RESUMEN
# =========================

st.subheader("Detalle numérico por escenario")

tabla = agg.pivot_table(
    index="scenario_id",
    columns="resultado_label",
    values="count",
    fill_value=0,
).reset_index()

# Añadimos % de éxito por escenario
if "Éxito" in tabla.columns and "Fallo" in tabla.columns:
    total_por_escenario = tabla["Éxito"] + tabla["Fallo"]
    # evitar división por 0
    tasa_exito = (tabla["Éxito"] / total_por_escenario.replace({0: pd.NA})).fillna(0) * 100
    tabla["pct_exito"] = tasa_exito.round(1)
    tabla.rename(columns={"pct_exito": "% éxito"}, inplace=True)

if not scenarios_lookup.empty:
    tabla = tabla.merge(
        scenarios_lookup,
        on="scenario_id",
        how="left",
    )
    metric_cols = [
        c for c in tabla.columns
        if c not in ("scenario_id", "scenario_title", "mission_text")
    ]
    # Dejamos el orden: ID, título, misión, Fallo, Éxito, % éxito
    orden_metricas = []
    for col in ["Fallo", "Éxito", "% éxito"]:
        if col in metric_cols:
            orden_metricas.append(col)
    # añadimos cualquier otra métrica que pudiera aparecer
    for col in metric_cols:
        if col not in orden_metricas:
            orden_metricas.append(col)

    nueva_orden = ["scenario_id", "scenario_title", "mission_text"] + orden_metricas
    tabla = tabla[nueva_orden]

st.dataframe(tabla, width="stretch")

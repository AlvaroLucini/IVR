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
RESULTS_ROOT = BASE_DIR / "test_results"   # 👈 carpeta que ya tienes

st.set_page_config(
    page_title="IVR Tester - Estadísticas escenarios",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Resultados IVR por escenario")
st.caption(
    "Para cada escenario, una barra verde (éxitos) y una roja (fallos), "
    "a partir de los JSON de test_results."
)

# =========================
# CARGA DE TODOS LOS JSON
# =========================

def load_all_results() -> pd.DataFrame:
    if not RESULTS_ROOT.exists():
        st.warning(f"No existe la carpeta de resultados: {RESULTS_ROOT}")
        return pd.DataFrame()

    records = []

    # Recorremos todas las subcarpetas tipo 2025-12-03, etc.
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

            # Extraemos los campos que nos interesan; si falta alguno, lo dejamos vacío
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

    # Normalizamos tipos básicos
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["result"] = df["result"].astype(str).str.upper().str.strip()

    # Parseamos timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")

    return df


df = load_all_results()

# =========================
# CONTROLES LATERALES
# =========================

with st.sidebar:
    st.header("⚙️ Filtros")

    # Filtro por fecha (si hay timestamp)
    if not df.empty and "timestamp_utc" in df.columns:
        if df["timestamp_utc"].notna().any():
            min_dt = df["timestamp_utc"].min()
            max_dt = df["timestamp_utc"].max()

            rango = st.date_input(
                "Rango de fechas:",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
            )
            if isinstance(rango, tuple) and len(rango) == 2:
                d1, d2 = rango
                mask = (
                    (df["timestamp_utc"].dt.date >= d1)
                    & (df["timestamp_utc"].dt.date <= d2)
                )
                df = df[mask]

    # Filtro por escenario
    if not df.empty:
        escenarios = sorted(df["scenario_id"].unique())
        seleccion = st.multiselect(
            "Escenarios:",
            options=escenarios,
            default=escenarios,
        )
        df = df[df["scenario_id"].isin(seleccion)]

    st.write("---")
# 👇 dentro del with st.sidebar:
    if st.button("🔄 Recargar resultados"):
        st.rerun()


# =========================
# SI NO HAY DATOS
# =========================

if df.empty:
    st.info("Todavía no hay resultados o los filtros no devuelven nada.")
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
    # todo lo que no sea “éxito”, lo tratamos como fallo por ahora
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
# GRÁFICA DE BARRAS
# =========================

st.subheader("Resultados por escenario")

chart = (
    alt.Chart(agg)
    .mark_bar()
    .encode(
        x=alt.X("scenario_id:N", title="Escenario"),
        xOffset="resultado_label:N",            # barras lado a lado
        y=alt.Y("count:Q", title="Número de tests"),
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

st.altair_chart(chart, use_container_width=True)

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

st.dataframe(tabla, use_container_width=True)

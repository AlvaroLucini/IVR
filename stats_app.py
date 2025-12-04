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
CSV_NODES = BASE_DIR / "config" / "ivr_nodes.csv"     # nodos IVR (para nombres)

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

# ===== ESTILOS PERSONALIZADOS (TIPOGRAFÍA) =====
st.markdown(
    """
    <style>
    /* Valores de los KPIs (st.metric) */
    div[data-testid="stMetricValue"] {
        font-size: 2.6rem;
        font-weight: 700;
    }

    /* Etiquetas de los KPIs */
    div[data-testid="stMetricLabel"] {
        font-size: 1.0rem;
    }

    /* Tamaño de letra de la tabla */
    div[data-testid="stDataFrame"] table {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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


def parse_route_json(route_str: str):
    """Convierte el string route_json en lista de pasos."""
    if not route_str:
        return []
    try:
        return json.loads(route_str)
    except Exception:
        return []


def build_route_str(steps, node_labels: dict[str, str]) -> str:
    """
    Construye una cadena legible de la ruta.
    Usa SOLO la etiqueta del nodo (NODE_LABEL) si existe; si no, el NODE_ID.
    Ejemplo: "Menú principal [3] → Autoservicio Resolution [2] → Contacto agente [1]"
    """
    parts = []
    for step in steps:
        node_id = str(step.get("node_id", ""))
        digit = str(step.get("digit", ""))

        label = node_labels.get(node_id, "").strip()
        display = label if label else node_id

        if digit and digit not in ("None",):
            part = f"{display} [{digit}]"
        else:
            part = display

        parts.append(part)

    return " → ".join(parts)


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
# LOOKUP DE NODOS
# =========================

def load_node_labels() -> dict[str, str]:
    """
    Lee ivr_nodes.csv y devuelve un dict: NODE_ID -> NODE_LABEL.
    """
    labels: dict[str, str] = {}
    if not CSV_NODES.exists():
        st.warning(f"No se ha encontrado el archivo de nodos: {CSV_NODES}")
        return labels

    df = pd.read_csv(CSV_NODES, dtype=str).fillna("")
    for _, row in df.iterrows():
        node_id = str(row.get("NODE_ID", "")).strip()
        if not node_id:
            continue
        labels[node_id] = str(row.get("NODE_LABEL", "")).strip()

    return labels


NODE_LABELS = load_node_labels()

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
                "test_id":             data.get("test_id", ""),
                "timestamp_utc":       data.get("timestamp_utc", ""),
                "scenario_id":         data.get("scenario_id", ""),
                "scenario_title":      data.get("scenario_title", ""),
                "result":              data.get("result", ""),
                "expected_queue_id":   data.get("expected_queue_id", ""),
                "reached_queue_id":    data.get("reached_queue_id", ""),
                "reached_queue_name":  data.get("reached_queue_name", ""),
                "end_node_id":         data.get("end_node_id", ""),
                "end_node_type":       data.get("end_node_type", ""),
                "duration_seconds":    data.get("duration_seconds", None),
                "num_steps":           data.get("num_steps", None),
                "route_json":          data.get("route_json", ""),
                "__source_file":       str(f.relative_to(BASE_DIR)),
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

# --- Duración media global ---
dur_series = df["duration_seconds"]
if dur_series.notna().any():
    avg_seconds = float(dur_series.dropna().mean())
else:
    avg_seconds = None
avg_str = format_seconds_hhmmss(avg_seconds)

# ===== Fila 1: resumen global (tests + duración) =====
st.markdown("### Resumen global")

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.metric("🔵 Tests totales", total_tests)
with g2:
    st.metric("🟢 Tests con éxito", tests_success)
with g3:
    st.metric("🔴 Tests con fallo", tests_fail)
with g4:
    st.metric("⏱ Duración media (HH:MM:SS)", avg_str)

st.markdown("---")

# --- KPIs de ESCENARIOS ---
total_scenarios = int(df["scenario_id"].nunique())
scenarios_with_success = int(
    agg[agg["resultado_label"] == "Éxito"]["scenario_id"].nunique()
)
scenarios_without_success = max(total_scenarios - scenarios_with_success, 0)

# ===== Fila 2: resumen por escenarios =====
st.markdown("### Resumen por escenarios")

e1, e2, e3 = st.columns(3)
with e1:
    st.metric("🔵 Escenarios ejecutados", total_scenarios)
with e2:
    st.metric("🟢 Escenarios con algún éxito", scenarios_with_success)
with e3:
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
# TABLA RESUMEN GLOBAL POR ESCENARIO
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
    tasa_exito = (tabla["Éxito"] / total_por_escenario.replace({0: pd.NA})).fillna(0) * 100
    tabla["pct_exito"] = tasa_exito.round(1)
    tabla.rename(columns={"pct_exito": "% éxito"}, inplace=True)

# Tiempo medio por escenario
if "duration_seconds" in df.columns:
    dur_por_escenario = (
        df.groupby("scenario_id")["duration_seconds"]
          .mean()
          .reset_index(name="avg_duration_seconds")
    )
    tabla = tabla.merge(dur_por_escenario, on="scenario_id", how="left")
    tabla["Duración media"] = tabla["avg_duration_seconds"].apply(format_seconds_hhmmss)
    tabla.drop(columns=["avg_duration_seconds"], inplace=True)

# Añadir info de scenarios.csv y ordenar columnas
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

    orden_metricas = []
    for col in ["Fallo", "Éxito", "% éxito", "Duración media"]:
        if col in metric_cols:
            orden_metricas.append(col)
    for col in metric_cols:
        if col not in orden_metricas:
            orden_metricas.append(col)

    nueva_orden = ["scenario_id", "scenario_title", "mission_text"] + orden_metricas
    tabla = tabla[nueva_orden]

st.dataframe(tabla, width="stretch")

# =========================
# DETALLE POR ESCENARIO: COLA CORRECTA + LLAMADAS
# =========================

st.markdown("---")
st.subheader("Detalle de rutas por escenario")

# Opciones de escenario para el desplegable
escenarios_disponibles = sorted(df["scenario_id"].unique())

def scenario_label(sid: str) -> str:
    row = scenarios_lookup[scenarios_lookup["scenario_id"] == sid]
    if not row.empty:
        title = row.iloc[0]["scenario_title"]
        return f"{sid} - {title}"
    return sid

selected_scenario = st.selectbox(
    "Selecciona un escenario para ver sus rutas:",
    options=escenarios_disponibles,
    format_func=scenario_label,
)

df_scenario = df[df["scenario_id"] == selected_scenario].copy()

if df_scenario.empty:
    st.info("No hay resultados para este escenario.")
else:
    # -------- Cola correcta (esperada) + ruta de éxito --------
    st.markdown("#### Cola correcta")

    # Cola esperada (puede venir repetida; nos quedamos con valores únicos)
    exp_q = (
        df_scenario["expected_queue_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(exp_q) == 0:
        cola_esperada = "No definida"
    else:
        cola_esperada = ", ".join(sorted(exp_q))

    st.markdown(f"**Cola esperada:** `{cola_esperada}`")

    df_ok = df_scenario[df_scenario["resultado_label"] == "Éxito"].copy()

    if df_ok.empty:
        # No hay éxitos → mostramos aviso y tabla vacía
        st.warning("Este escenario no tiene tests con Éxito todavía.")
        st.markdown("#### Llamadas de este escenario")
        empty_cols = ["test_id", "timestamp_utc", "resultado_label", "reached_queue_id", "ruta"]
        st.dataframe(pd.DataFrame(columns=empty_cols), width="stretch")
    else:
        # Tomamos la ruta de éxito más frecuente
        vc = df_ok["route_json"].value_counts()
        best_route_json = vc.index[0]
        steps = parse_route_json(best_route_json)
        correct_route_str = build_route_str(steps, NODE_LABELS)

        st.markdown(f"**Ruta de éxito más frecuente:** {correct_route_str}")

        st.markdown("#### Llamadas de este escenario")

        # Construimos una columna con la ruta seguida en cada test (más sintética)
        def row_route_str(route_str: str) -> str:
            steps_local = parse_route_json(route_str)
            return build_route_str(steps_local, NODE_LABELS)

        df_scenario["ruta"] = df_scenario["route_json"].apply(row_route_str)

        # Construimos una columna con la ruta seguida en cada test (más sintética)
    def row_route_str(route_str: str) -> str:
        steps_local = parse_route_json(route_str)
        return build_route_str(steps_local, NODE_LABELS)

    df_scenario["ruta"] = df_scenario["route_json"].apply(row_route_str)

    # 🔵🟢🟥 Columna de icono de estado (éxito / fallo)
    df_scenario["estado"] = df_scenario["resultado_label"].map(
        {"Éxito": "🟢", "Fallo": "🔴"}
    ).fillna("⚪")

    # Tabla de llamadas (sin reached_queue_name ni end_node_type)
    cols_preferencia = [
        "estado",          # 👈 nuevo icono
        "test_id",
        "timestamp_utc",
        "resultado_label",
        "reached_queue_id",
        "ruta",
    ]
    cols_presentes = [c for c in cols_preferencia if c in df_scenario.columns]

    tabla_llamadas = (
        df_scenario[cols_presentes]
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    st.dataframe(tabla_llamadas, width="stretch")

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

# =========================
# ESTILOS PERSONALIZADOS
# =========================

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

    /* Tamaño de letra de las tablas */
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
      - expected_queue_id
      - expected_alt_queue_ids
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
        "expected_queue_id": df.get("EXPECTED_QUEUE_ID", ""),
        "expected_alt_queue_ids": df.get("EXPECTED_ALT_QUEUE_IDS", ""),
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


df_raw = load_all_results()

# =========================
# COMPROBAR QUE HAY DATA
# =========================

if df_raw.empty:
    st.info("Todavía no hay resultados para mostrar.")
    st.stop()

# =========================
# NORMALIZAR reached_queue_id PARA SMS (sobre df_raw)
# =========================

if "end_node_type" in df_raw.columns:
    end_type_upper = df_raw["end_node_type"].astype(str).str.upper()
    rq_stripped = df_raw["reached_queue_id"].astype(str).str.strip()
    mask_sms = (end_type_upper == "SMS") & (rq_stripped == "")
    df_raw.loc[mask_sms, "reached_queue_id"] = "SMS"

# =========================
# MAPEO RESULTADOS → Éxito / Fallo (función global)
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


# =========================
# CONTROLES LATERALES (RECARGAR + RANGO DE FECHAS PRINCIPAL)
# =========================

has_dates = "timestamp_utc" in df_raw.columns and df_raw["timestamp_utc"].notna().any()
if has_dates:
    min_date = df_raw["timestamp_utc"].min().date()
    max_date = df_raw["timestamp_utc"].max().date()
else:
    min_date = max_date = None

with st.sidebar:
    if st.button("🔄 Recargar resultados"):
        st.rerun()

    st.markdown("---")
    st.markdown("### Filtro de fechas (Resumen)")

    if has_dates:
        date_range = st.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="main_date_range",
        )

        if isinstance(date_range, (tuple, list)):
            if len(date_range) == 2:
                start_date_main, end_date_main = date_range
            elif len(date_range) == 1:
                start_date_main = end_date_main = date_range[0]
            else:
                start_date_main = end_date_main = None
        else:
            start_date_main = end_date_main = date_range
    else:
        st.info("Los JSON no tienen timestamp_utc, no se puede filtrar por fecha.")
        start_date_main = end_date_main = None

# Aplicamos el filtro de fechas principal
if has_dates and start_date_main and end_date_main:
    mask_main = df_raw["timestamp_utc"].dt.date.between(start_date_main, end_date_main)
    df = df_raw[mask_main].copy()
else:
    df = df_raw.copy()

# =========================
# TABS
# =========================

tab_resumen, tab_comp = st.tabs(["Resumen", "COMPARATIVA"])

# =====================================================
# TAB 1: RESUMEN (igual que antes pero usando df filtrado)
# =====================================================

with tab_resumen:

    if df.empty:
        st.info("No hay resultados en el rango seleccionado.")
    else:
        # Etiqueta de resultado
        df["resultado_label"] = df["result"].apply(map_result_label)

        # -------- AGREGADO POR ESCENARIO --------
        agg = (
            df.groupby(["scenario_id", "resultado_label"])
              .size()
              .reset_index(name="count")
        )

        # -------- KPI TESTS Y ESCENARIOS --------
        total_tests = int(len(df))
        tests_success = int((df["resultado_label"] == "Éxito").sum())
        tests_fail = total_tests - tests_success

        dur_series = df["duration_seconds"]
        if dur_series.notna().any():
            avg_seconds = float(dur_series.dropna().mean())
        else:
            avg_seconds = None
        avg_str = format_seconds_hhmmss(avg_seconds)

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

        total_scenarios = int(df["scenario_id"].nunique())
        scenarios_with_success = int(
            agg[agg["resultado_label"] == "Éxito"]["scenario_id"].nunique()
        )
        scenarios_without_success = max(total_scenarios - scenarios_with_success, 0)

        st.markdown("### Resumen por escenarios")

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("🔵 Escenarios ejecutados", total_scenarios)
        with e2:
            st.metric("🟢 Escenarios con algún éxito", scenarios_with_success)
        with e3:
            st.metric("🔴 Escenarios sin ningún éxito", scenarios_without_success)

        st.markdown("---")

        # -------- GRÁFICA DE BARRAS --------
        st.subheader("Resultados por escenario")

        if not agg.empty:
            max_count = int(agg["count"].max())
        else:
            max_count = 0

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

        st.altair_chart(chart, use_container_width=True)

        # -------- TABLA RESUMEN GLOBAL POR ESCENARIO --------
        st.subheader("Detalle numérico por escenario")

        tabla = agg.pivot_table(
            index="scenario_id",
            columns="resultado_label",
            values="count",
            fill_value=0,
        ).reset_index()

        if "Éxito" in tabla.columns and "Fallo" in tabla.columns:
            total_por_escenario = tabla["Éxito"] + tabla["Fallo"]
            tasa_exito = (tabla["Éxito"] / total_por_escenario.replace({0: pd.NA})).fillna(0) * 100
            tabla["pct_exito"] = tasa_exito.round(1)
            tabla.rename(columns={"pct_exito": "% éxito"}, inplace=True)

        if "duration_seconds" in df.columns:
            dur_por_escenario = (
                df.groupby("scenario_id")["duration_seconds"]
                  .mean()
                  .reset_index(name="avg_duration_seconds")
            )
            tabla = tabla.merge(dur_por_escenario, on="scenario_id", how="left")
            tabla["Duración media"] = tabla["avg_duration_seconds"].apply(format_seconds_hhmmss)
            tabla.drop(columns=["avg_duration_seconds"], inplace=True)

        if not scenarios_lookup.empty:
            tabla = tabla.merge(
                scenarios_lookup[["scenario_id", "scenario_title", "mission_text"]],
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

        st.dataframe(tabla, use_container_width=True)

        # -------- DETALLE POR ESCENARIO --------
        st.markdown("---")
        st.subheader("Detalle de rutas por escenario")

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
            st.info("No hay resultados para este escenario en el rango seleccionado.")
        else:
            st.markdown("#### Cola correcta")

            cfg_row = scenarios_lookup[scenarios_lookup["scenario_id"] == selected_scenario]
            if not cfg_row.empty:
                cfg = cfg_row.iloc[0]
                primary_q = str(cfg.get("expected_queue_id", "")).strip()
                alt_q = str(cfg.get("expected_alt_queue_ids", "")).strip()
            else:
                primary_q = ""
                alt_q = ""

            if primary_q or alt_q:
                partes = []
                if primary_q:
                    partes.append(f"**Principal:** `{primary_q}`")
                if alt_q:
                    partes.append(f"**Alternativas válidas:** `{alt_q}`")
                cola_txt = "  \n".join(partes)
                st.markdown(cola_txt)
            else:
                st.markdown("_Cola esperada no definida en scenarios.csv_")

            df_scenario["resultado_label"] = df_scenario["result"].apply(map_result_label)
            df_ok = df_scenario[df_scenario["resultado_label"] == "Éxito"].copy()

            if not df_ok.empty:
                vc = df_ok["route_json"].value_counts()
                best_route_json = vc.index[0]
                steps = parse_route_json(best_route_json)
                correct_route_str = build_route_str(steps, NODE_LABELS)

                st.markdown(f"**Ruta de éxito más frecuente:** {correct_route_str}")
            else:
                st.warning("Este escenario no tiene tests con Éxito en el rango seleccionado.")

            st.markdown("#### Llamadas de este escenario")

            def row_route_str(route_str: str) -> str:
                steps_local = parse_route_json(route_str)
                return build_route_str(steps_local, NODE_LABELS)

            df_scenario["ruta"] = df_scenario["route_json"].apply(row_route_str)

            df_scenario["estado"] = df_scenario["resultado_label"].map(
                {"Éxito": "🟢", "Fallo": "🔴"}
            ).fillna("⚪")

            cols_preferencia = [
                "estado",
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

            st.dataframe(tabla_llamadas, use_container_width=True)

# =====================================================
# TAB 2: COMPARATIVA (dos intervalos + heatmap)
# =====================================================

with tab_comp:

    st.markdown(
        "Compara dos intervalos de fechas. "
        "TEST #1 (izquierda) vs TEST #2 (derecha). "
        "Las columnas de diferencia muestran en verde las mejoras y en rojo los empeoramientos."
    )

    if not has_dates:
        st.info("Los JSON no tienen timestamp_utc, no se puede hacer comparativa por fechas.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            range1 = st.date_input(
                "Intervalo TEST #1",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="cmp_range_1",
            )
        with col_b:
            range2 = st.date_input(
                "Intervalo TEST #2",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="cmp_range_2",
            )

        def parse_range(r):
            if isinstance(r, (tuple, list)):
                if len(r) == 2:
                    return r[0], r[1]
                elif len(r) == 1:
                    return r[0], r[0]
                else:
                    return None, None
            else:
                return r, r

        start1, end1 = parse_range(range1)
        start2, end2 = parse_range(range2)

        def filter_by_dates(df_base, d1, d2):
            if d1 and d2:
                m = df_base["timestamp_utc"].dt.date.between(d1, d2)
                return df_base[m].copy()
            return df_base.copy()

        df1 = filter_by_dates(df_raw, start1, end1)
        df2 = filter_by_dates(df_raw, start2, end2)

        # --- función de agregado por escenario para la comparativa ---
        def aggregate_scenarios(df_subset: pd.DataFrame) -> pd.DataFrame:
            if df_subset.empty:
                return pd.DataFrame(
                    columns=["scenario_id", "mission_text", "Fallo", "Éxito",
                             "pct_exito", "avg_duration_seconds", "Duración media"]
                )

            tmp = df_subset.copy()
            tmp["resultado_label"] = tmp["result"].apply(map_result_label)

            agg_local = (
                tmp.groupby(["scenario_id", "resultado_label"])
                   .size()
                   .reset_index(name="count")
            )

            tabla_local = agg_local.pivot_table(
                index="scenario_id",
                columns="resultado_label",
                values="count",
                fill_value=0,
            ).reset_index()

            # Aseguramos columnas
            if "Fallo" not in tabla_local.columns:
                tabla_local["Fallo"] = 0
            if "Éxito" not in tabla_local.columns:
                tabla_local["Éxito"] = 0

            total_esc = tabla_local["Fallo"] + tabla_local["Éxito"]
            tabla_local["pct_exito"] = (
                (tabla_local["Éxito"] / total_esc.replace({0: pd.NA})).fillna(0) * 100
            )

            # Duración media
            dur_loc = (
                tmp.groupby("scenario_id")["duration_seconds"]
                   .mean()
                   .reset_index(name="avg_duration_seconds")
            )
            tabla_local = tabla_local.merge(dur_loc, on="scenario_id", how="left")
            tabla_local["Duración media"] = tabla_local["avg_duration_seconds"].apply(
                format_seconds_hhmmss
            )

            # Mission text
            if not scenarios_lookup.empty:
                tabla_local = tabla_local.merge(
                    scenarios_lookup[["scenario_id", "mission_text"]],
                    on="scenario_id",
                    how="left",
                )
            else:
                tabla_local["mission_text"] = ""

            return tabla_local[
                ["scenario_id", "mission_text", "Fallo", "Éxito",
                 "pct_exito", "avg_duration_seconds", "Duración media"]
            ]

        summary1 = aggregate_scenarios(df1)
        summary2 = aggregate_scenarios(df2)

        if summary1.empty and summary2.empty:
            st.info("No hay datos en ninguno de los dos intervalos seleccionados.")
        else:
            merged = pd.merge(
                summary1,
                summary2,
                on=["scenario_id", "mission_text"],
                how="outer",
                suffixes=("_1", "_2"),
            ).fillna(0)

            # Preparamos columnas de visualización
            disp = pd.DataFrame()
            disp["scenario_id"] = merged["scenario_id"]
            disp["mission_text"] = merged["mission_text"]

            # TEST #1
            disp["Fallo_1"] = merged["Fallo_1"].astype(int)
            disp["Éxito_1"] = merged["Éxito_1"].astype(int)
            disp["% éxito_1"] = merged["pct_exito_1"]
            disp["Duración media_1"] = merged["Duración media_1"]

            # TEST #2
            disp["Fallo_2"] = merged["Fallo_2"].astype(int)
            disp["Éxito_2"] = merged["Éxito_2"].astype(int)
            disp["% éxito_2"] = merged["pct_exito_2"]
            disp["Duración media_2"] = merged["Duración media_2"]

            # Diferencias (2 - 1)
            disp["Δ % éxito (2-1)"] = merged["pct_exito_2"] - merged["pct_exito_1"]
            # Positivo = mejora (menos duración en 2)
            disp["Δ duración (s, +mejor si >0)"] = (
                merged["avg_duration_seconds_1"] - merged["avg_duration_seconds_2"]
            )

            # Formato y heatmap
            def color_diff(val):
                try:
                    v = float(val)
                except Exception:
                    return ""
                if v > 0:
                    return "background-color: rgba(0, 150, 0, 0.5); color: white;"
                elif v < 0:
                    return "background-color: rgba(200, 0, 0, 0.55); color: white;"
                else:
                    return "background-color: rgba(120,120,120,0.3); color: white;"

            styler = (
                disp.style
                .format({
                    "% éxito_1": "{:.1f}",
                    "% éxito_2": "{:.1f}",
                    "Δ % éxito (2-1)": "{:+.1f}",
                    "Δ duración (s, +mejor si >0)": "{:+.0f}",
                })
                .applymap(color_diff, subset=["Δ % éxito (2-1)", "Δ duración (s, +mejor si >0)"])
            )

            st.dataframe(styler, use_container_width=True)

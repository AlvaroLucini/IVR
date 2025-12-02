#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

# =========================
# CONFIG BÁSICA
# =========================

BASE_DIR = Path(__file__).resolve().parent
CSV_NODES = BASE_DIR / "config" / "ivr_nodes.csv"
CSV_SCENARIOS = BASE_DIR / "config" / "scenarios.csv"
AUDIO_DIR = BASE_DIR / "audio"

st.set_page_config(
    page_title="IVR POC Tester",
    page_icon="📞",
    layout="centered",
)


# =========================
# MODELOS
# =========================

@dataclass
class IVRNode:
    id: str
    label: str
    prompt_text: str
    audio_url: Optional[str]
    queue_id: Optional[str]
    queue_name: Optional[str]
    options: Dict[str, str]  # DTMF -> next_node_id


@dataclass
class Scenario:
    id: str
    title: str
    mission_text: str
    entry_node_id: str
    expected_queue_id: Optional[str]
    active: bool


# =========================
# LOADERS
# =========================

@st.cache_data
def load_nodes() -> Dict[str, IVRNode]:
    df = pd.read_csv(CSV_NODES, dtype=str).fillna("")

    nodes: Dict[str, IVRNode] = {}

    # Detectar dinámicamente columnas OPT_?_NEXT_NODE
    opt_cols = [c for c in df.columns if c.startswith("OPT_") and c.endswith("_NEXT_NODE")]

    for _, row in df.iterrows():
        node_id = row["NODE_ID"].strip()
        if not node_id:
            continue

        label = row.get("NODE_LABEL", "").strip()
        prompt = row.get("PROMPT_TEXT", "").strip()
        audio_url = row.get("AUDIO_URL", "").strip() or None
        queue_id = row.get("QUEUE_ID", "").strip() or None
        queue_name = row.get("QUEUE_NAME", "").strip() or None

        options: Dict[str, str] = {}

        # Mapeamos OPT_0_NEXT_NODE -> DTMF "0", OPT_1_NEXT_NODE -> "1", etc.
        for col in opt_cols:
            target = row.get(col, "").strip()
            if not target:
                continue

            # Extraer el dígito de "OPT_X_NEXT_NODE"
            # Ej: "OPT_1_NEXT_NODE" -> "1"
            try:
                digit = col.split("_")[1]  # ["OPT", "1", "NEXT", "NODE"]
            except Exception:
                continue

            options[digit] = target

        nodes[node_id] = IVRNode(
            id=node_id,
            label=label or node_id,
            prompt_text=prompt,
            audio_url=audio_url,
            queue_id=queue_id,
            queue_name=queue_name,
            options=options,
        )

    return nodes


@st.cache_data
def load_scenarios() -> Dict[str, Scenario]:
    df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")

    scenarios: Dict[str, Scenario] = {}

    for _, row in df.iterrows():
        sid = row["SCENARIO_ID"].strip()
        if not sid:
            continue

        active_flag = row.get("ACTIVE", "").strip().upper()
        active = active_flag in ("TRUE", "1", "YES", "Y")

        scenarios[sid] = Scenario(
            id=sid,
            title=row.get("TITLE", "").strip() or sid,
            mission_text=row.get("MISSION_TEXT", "").strip(),
            entry_node_id=row.get("ENTRY_NODE_ID", "").strip(),
            expected_queue_id=row.get("EXPECTED_QUEUE_ID", "").strip() or None,
            active=active,
        )

    return scenarios


# =========================
# STATE HELPERS
# =========================

def init_state():
    if "scenario_id" not in st.session_state:
        st.session_state["scenario_id"] = None

    if "ivr_current_node_id" not in st.session_state:
        st.session_state["ivr_current_node_id"] = None

    if "ivr_last_played_node_id" not in st.session_state:
        st.session_state["ivr_last_played_node_id"] = None

    if "ivr_path" not in st.session_state:
        st.session_state["ivr_path"] = []  # lista de nodos visitados


def start_scenario(scenario: Scenario):
    st.session_state["scenario_id"] = scenario.id
    st.session_state["ivr_current_node_id"] = scenario.entry_node_id or None
    st.session_state["ivr_last_played_node_id"] = None
    st.session_state["ivr_path"] = []
    rerun()


def go_to_node(next_node_id: Optional[str]):
    """
    Cambia al nodo indicado.
    Si next_node_id es vacío/None, se considera fin de flujo.
    """
    if not next_node_id:
        st.session_state["ivr_current_node_id"] = None
        st.session_state["ivr_last_played_node_id"] = None
        rerun()
        return

    st.session_state["ivr_current_node_id"] = next_node_id
    st.session_state["ivr_last_played_node_id"] = None

    path = st.session_state.get("ivr_path", [])
    path = path + [next_node_id]
    st.session_state["ivr_path"] = path

    rerun()


def rerun():
    # Compatibilidad con distintas versiones de Streamlit
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================
# RENDER NODO
# =========================

def render_node(
    node: IVRNode,
    scenario: Optional[Scenario] = None,
):
    st.markdown(f"### Nodo: `{node.id}`")
    st.write(f"**Etiqueta:** {node.label}")

    if scenario:
        with st.expander("🎯 Contexto del escenario", expanded=True):
            st.write(f"**Escenario:** {scenario.title}")
            if scenario.mission_text:
                st.caption(scenario.mission_text)
            if scenario.expected_queue_id:
                st.caption(f"Queue esperada: `{scenario.expected_queue_id}`")

    st.markdown("---")

    # 1) Audio – se reproduce solo la primera vez que entramos al nodo
    if st.session_state.get("ivr_last_played_node_id") != node.id:
        if node.audio_url:
            audio_source = resolve_audio_source(node.audio_url)
            if audio_source:
                st.audio(str(audio_source))
            else:
                st.warning(f"⚠️ Audio no encontrado: {node.audio_url}")
        else:
            st.info("Este nodo no tiene audio asociado.")

        st.session_state["ivr_last_played_node_id"] = node.id

    # 2) Texto del prompt (lo que “diría” la IVR)
    if node.prompt_text:
        st.write(node.prompt_text)

    # 3) Si el nodo tiene queue asociada, la mostramos
    if node.queue_id:
        st.markdown(
            f"> 🧵 **Queue destino:** `{node.queue_id}`"
            + (f" — {node.queue_name}" if node.queue_name else "")
        )

    st.markdown("---")

    # 4) Teclado DTMF – en cuanto pulses algo, se corta el audio (porque cambiamos de nodo)
    st.markdown("#### Teclado DTMF")

    buttons: Dict[str, bool] = {}

    # Fila 1: 1 2 3
    cols = st.columns(3)
    buttons["1"] = cols[0].button("1")
    buttons["2"] = cols[1].button("2")
    buttons["3"] = cols[2].button("3")

    # Fila 2: 4 5 6
    cols = st.columns(3)
    buttons["4"] = cols[0].button("4")
    buttons["5"] = cols[1].button("5")
    buttons["6"] = cols[2].button("6")

    # Fila 3: 7 8 9
    cols = st.columns(3)
    buttons["7"] = cols[0].button("7")
    buttons["8"] = cols[1].button("8")
    buttons["9"] = cols[2].button("9")

    # Fila 4: * 0 #
    cols = st.columns(3)
    buttons["*"] = cols[0].button("*")
    buttons["0"] = cols[1].button("0")
    buttons["#"] = cols[2].button("#")

    pressed_digit = next((dtmf for dtmf, pressed in buttons.items() if pressed), None)

    if pressed_digit is not None:
        next_node_id = node.options.get(pressed_digit)
        if not next_node_id:
            st.warning(f"No hay destino configurado para la opción `{pressed_digit}` en este nodo.")
        go_to_node(next_node_id)


def resolve_audio_source(audio_url: str) -> Optional[Path | str]:
    """
    Si es un path relativo, lo resolvemos sobre AUDIO_DIR.
    Si parece una URL (http/https), lo devolvemos tal cual.
    """
    audio_url = audio_url.strip()
    if not audio_url:
        return None

    if audio_url.startswith("http://") or audio_url.startswith("https://"):
        return audio_url

    candidate = AUDIO_DIR / audio_url
    if candidate.exists():
        return candidate

    return None


# =========================
# MAIN UI
# =========================

def main():
    init_state()

    st.title("📞 IVR POC Tester (Web)")

    nodes = load_nodes()
    scenarios = load_scenarios()

    # ---- Sidebar: selección de escenario ----
    st.sidebar.header("Escenario")

    active_scenarios = {sid: s for sid, s in scenarios.items() if s.active}
    if not active_scenarios:
        st.sidebar.error("No hay escenarios activos en scenarios.csv")
        return

    # Mapeo para el selectbox
    scenario_labels = {
        sid: f"{s.id} — {s.title}" for sid, s in active_scenarios.items()
    }

    default_scenario_id = (
        st.session_state.get("scenario_id")
        if st.session_state.get("scenario_id") in active_scenarios
        else list(active_scenarios.keys())[0]
    )

    selected_label = st.sidebar.selectbox(
        "Selecciona escenario",
        options=list(scenario_labels.values()),
        index=list(scenario_labels.keys()).index(default_scenario_id),
    )

    # Invertimos para recuperar el ID a partir del label
    label_to_id = {v: k for k, v in scenario_labels.items()}
    selected_scenario_id = label_to_id[selected_label]
    selected_scenario = active_scenarios[selected_scenario_id]

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("▶️ Iniciar / reiniciar"):
        start_scenario(selected_scenario)
    if col_btn2.button("⏹ Finalizar"):
        st.session_state["ivr_current_node_id"] = None
        st.session_state["ivr_last_played_node_id"] = None
        st.session_state["ivr_path"] = []
        rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("POC IVR: simulación de flujo y audio con DTMF en navegador.")

    # ---- Contenido principal ----

    current_node_id = st.session_state.get("ivr_current_node_id")

    if not current_node_id:
        st.info("Pulsa **Iniciar / reiniciar** en la barra lateral para comenzar el escenario.")
        return

    node = nodes.get(current_node_id)
    if not node:
        st.error(f"Nodo actual desconocido: `{current_node_id}` (revisa ivr_nodes.csv).")
        return

    # Render nodo
    render_node(node, scenario=selected_scenario)

    # Trayectoria (log visual simple)
    with st.expander("🧭 Camino recorrido"):
        path = st.session_state.get("ivr_path", [])
        if not path:
            st.write("Solo has visitado el nodo inicial.")
        else:
            st.write(" → ".join(path))


if __name__ == "__main__":
    main()

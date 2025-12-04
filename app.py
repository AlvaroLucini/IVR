#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import random
import base64
import json
from uuid import uuid4
import time  # ⏱️ para la cuenta atrás

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================
# CONFIG PÁGINA
# =========================

st.set_page_config(page_title="IVR Tester", page_icon="📞", layout="centered")

# =========================
# MODO DEBUG (para ti)
# =========================

DEBUG_MODE = bool(st.secrets.get("debug_mode", False))

# =========================
# MODO MANTENIMIENTO
# =========================
# Si MAINTENANCE_MODE está a True (o en secrets), la app muestra solo
# un mensaje + imagen y NO permite lanzar tests (salvo que estés en DEBUG).
MAINTENANCE_MODE = bool(st.secrets.get("maintenance_mode", False))

MAINTENANCE_MESSAGE = (
    "Estamos actualizando la IVR para testar una nueva estructura. "
    "En breves momentos volveremos a abrir la plataforma para testarla. 🙏"
)

# Duración de la cuenta atrás (en segundos) cuando se entra en mantenimiento
# Ajusta este valor cada vez que cierres la plataforma (por ejemplo, 30 * 60 = 30 minutos)
MAINTENANCE_COUNTDOWN_SECONDS = 30 * 60


# =========================
# RUTAS BÁSICAS
# =========================

BASE_DIR = Path(__file__).resolve().parent
CSV_NODES = BASE_DIR / "config" / "ivr_nodes.csv"
CSV_SCENARIOS = BASE_DIR / "config" / "scenarios.csv"

# Posibles ubicaciones del tono de llamada
RING_PATHS = [
    BASE_DIR / "audio" / "ringtone.wav",
    BASE_DIR / "tts_audio" / "ringtone.wav",
]

# Imagen de mantenimiento (ajusta nombre/ruta si quieres)
MAINTENANCE_IMAGE_PATH = BASE_DIR / "maintenance.png"


# =========================
# DEBUG SECRETS
# =========================

try:
    if DEBUG_MODE:
        st.sidebar.write("DEBUG secrets keys:", list(st.secrets.keys()))
        if "github" in st.secrets:
            st.sidebar.write("DEBUG github keys:", list(st.secrets["github"].keys()))
except Exception as e:
    if DEBUG_MODE:
        st.sidebar.write(f"DEBUG st.secrets error: {e}")


# =========================
# HELPERS AUDIO
# =========================

def _audio_path_to_src(path: Path) -> tuple[str, str]:
    """Convierte Path local a data:URL base64 + mime."""
    audio_bytes = path.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode("utf-8")

    ext = path.suffix.lower()
    if ext == ".wav":
        mime = "audio/wav"
    elif ext in (".mp3", ".mpeg"):
        mime = "audio/mpeg"
    elif ext == ".ogg":
        mime = "audio/ogg"
    else:
        mime = "audio/wav"

    src = f"data:{mime};base64,{b64}"
    return src, mime


def get_node_audio_source(node: dict) -> tuple[str | None, str | None]:
    """
    Devuelve (src, mime) para el audio del nodo,
    o (None, None) si no hay audio configurado.

    - src: data:URL base64 o URL http/https
    - mime: 'audio/... ' o 'url' si src es URL remota
    """
    node_id = node["NODE_ID"]

    # Candidatos locales: audio/{NODE_ID}_es.wav, audio/{NODE_ID}.wav, tts_audio/{NODE_ID}_es.wav
    audio_es_path = BASE_DIR / "audio" / f"{node_id}_es.wav"
    audio_plain_path = BASE_DIR / "audio" / f"{node_id}.wav"
    tts_path = BASE_DIR / "tts_audio" / f"{node_id}_es.wav"

    for p in (audio_es_path, audio_plain_path, tts_path):
        if p.exists():
            try:
                return _audio_path_to_src(p)
            except Exception:
                continue

    # AUDIO_URL como último recurso (solo http/https)
    audio_url = str(node.get("AUDIO_URL", "")).strip()
    if audio_url and audio_url.lower().startswith(("http://", "https://")):
        return audio_url, "url"

    return None, None


def play_ring_and_prompt(node: dict) -> bool:
    """
    Reproduce primero el tono de llamada y luego el prompt del nodo
    usando JavaScript en el navegador. No muestra reproductores visibles.
    """
    # Tono
    ring_file = next((p for p in RING_PATHS if p.exists()), None)
    ring_src = None
    ring_mime = None
    if ring_file is not None:
        try:
            ring_src, ring_mime = _audio_path_to_src(ring_file)
        except Exception:
            ring_src, ring_mime = None, None

    # Prompt del nodo
    prompt_src, prompt_mime = get_node_audio_source(node)

    # Si no hay ni tono ni prompt, no hacemos nada
    if ring_src is None and prompt_src is None:
        return False

    # Caso típico: ambos en base64 (tono y prompt)
    if prompt_src is not None and prompt_mime != "url":
        ring_audio_html = ""
        if ring_src is not None:
            ring_audio_html = f"""
            <audio id="ivr_ring" preload="auto">
                <source src="{ring_src}" type="{ring_mime}">
            </audio>
            """
        prompt_audio_html = f"""
        <audio id="ivr_prompt" preload="auto">
            <source src="{prompt_src}" type="{prompt_mime}">
        </audio>
        """
        script = """
        <script>
        (function() {
            var ring = document.getElementById("ivr_ring");
            var prompt = document.getElementById("ivr_prompt");
            if (!prompt) return;
            function playPrompt() {
                var pp = prompt.play();
                if (pp !== undefined) {
                    pp.catch(function(err) {
                        console.log("Error reproduciendo prompt:", err);
                    });
                }
            }
            if (ring) {
                var rp = ring.play();
                if (rp !== undefined) {
                    rp.catch(function(err) {
                        console.log("Error reproduciendo ring:", err);
                        // Si falla el tono, lanzamos directamente el prompt
                        playPrompt();
                    });
                }
                ring.onended = function() {
                    playPrompt();
                };
            } else {
                playPrompt();
            }
        })();
        </script>
        """
        html = ring_audio_html + prompt_audio_html + script
        components.html(html, height=0, width=0)
        return True
    else:
        # Si el prompt viene de una URL http/https
        ring_audio_html = ""
        if ring_src is not None:
            ring_audio_html = f"""
            <audio id="ivr_ring" preload="auto">
                <source src="{ring_src}" type="{ring_mime}">
            </audio>
            """
        prompt_src_final = prompt_src if prompt_src is not None else ""
        prompt_audio_html = f"""
        <audio id="ivr_prompt" preload="auto">
            <source src="{prompt_src_final}" type="audio/mpeg">
        </audio>
        """
        script = """
        <script>
        (function() {
            var ring = document.getElementById("ivr_ring");
            var prompt = document.getElementById("ivr_prompt");
            if (!prompt) return;
            function playPrompt() {
                var pp = prompt.play();
                if (pp !== undefined) {
                    pp.catch(function(err) {
                        console.log("Error reproduciendo prompt URL:", err);
                    });
                }
            }
            if (ring) {
                var rp = ring.play();
                if (rp !== undefined) {
                    rp.catch(function(err) {
                        console.log("Error reproduciendo ring:", err);
                        playPrompt();
                    });
                }
                ring.onended = function() {
                    playPrompt();
                };
            } else {
                playPrompt();
            }
        })();
        </script>
        """
        html = ring_audio_html + prompt_audio_html + script
        components.html(html, height=0, width=0)
        return True


def play_node_audio(node: dict) -> bool:
    """
    Reproduce solo el audio del nodo (sin tono) vía JS autoplay,
    sin reproductor visible.
    Se puede llamar tantas veces como se quiera (por ejemplo, con '*').
    """
    src, mime = get_node_audio_source(node)
    if src is None:
        return False

    # Caso base64/local (no URL remota)
    if mime != "url":
        html = f"""
        <audio id="ivr_prompt_only" preload="auto">
            <source src="{src}" type="{mime}">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById("ivr_prompt_only");
            if (!audio) return;

            try {{
                audio.pause();
                audio.currentTime = 0;
            }} catch (e) {{
                console.log("Error reseteando audio local:", e);
            }}

            var pp = audio.play();
            if (pp !== undefined) {{
                pp.catch(function(err) {{
                    console.log("Error reproduciendo prompt:", err);
                }});
            }}
        }})();
        </script>
        """
        components.html(html, height=0, width=0)
        return True

    # Caso URL http/https
    else:
        html = f"""
        <audio id="ivr_prompt_only_url" preload="auto">
            <source src="{src}" type="audio/mpeg">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById("ivr_prompt_only_url");
            if (!audio) return;

            try {{
                audio.pause();
                audio.currentTime = 0;
            }} catch (e) {{
                console.log("Error reseteando audio URL:", e);
            }}

            var pp = audio.play();
            if (pp !== undefined) {{
                pp.catch(function(err) {{
                    console.log("Error reproduciendo prompt URL:", err);
                }});
            }}
        }})();
        </script>
        """
        components.html(html, height=0, width=0)
        return True


def pick_next_scenario_least_executed() -> dict | None:
    """
    Elige el siguiente escenario entre los que tienen menos EXECUTIONS.
    Si hay varios empatados al mínimo, elige uno al azar.
    Devuelve un dict (fila del CSV) o None si no hay escenarios.
    """
    df = load_scenarios_df_for_selection()
    if df is None or df.empty:
        return None

    min_exec = df["EXECUTIONS_INT"].min()
    candidatos = df[df["EXECUTIONS_INT"] == min_exec]

    # Elegimos uno aleatorio entre los de menor EXECUTIONS
    fila = candidatos.sample(1).iloc[0]
    return fila.to_dict()


# =========================
# CARGA DE CONFIGURACIÓN
# =========================

def load_nodes():
    """
    Carga ivr_nodes.csv y construye el diccionario NODES.
    NODE_TYPE puede ser: MENU / QUEUE / SMS / TRANSFER / ACCOUNT
    """
    if not CSV_NODES.exists():
        st.error(f"No se encuentra el archivo de nodos: {CSV_NODES}")
        st.stop()

    df = pd.read_csv(CSV_NODES, dtype=str).fillna("")

    nodes: dict[str, dict] = {}

    for _, row in df.iterrows():
        node_id = str(row.get("NODE_ID", "")).strip()
        if not node_id:
            continue

        node_label = str(row.get("NODE_LABEL", "") or "")
        node_type = str(row.get("NODE_TYPE", "") or "")  # MENU / QUEUE / SMS / TRANSFER / ACCOUNT
        is_entry_flag = str(row.get("IS_ENTRY", "") or "").strip().upper() == "YES"
        prompt_text = str(row.get("PROMPT_TEXT", "") or "")
        audio_url = str(row.get("AUDIO_URL", "") or "")
        queue_id = str(row.get("QUEUE_ID", "") or "")
        queue_name = str(row.get("QUEUE_NAME", "") or "")

        next_map: dict[str, str] = {}
        for d in range(10):
            col_name = f"OPT_{d}_NEXT_NODE"
            next_map[str(d)] = str(row.get(col_name, "") or "").strip()

        nodes[node_id] = {
            "NODE_ID":     node_id,
            "NODE_LABEL":  node_label,
            "NODE_TYPE":   node_type,
            "IS_ENTRY":    is_entry_flag,
            "PROMPT_TEXT": prompt_text,
            "AUDIO_URL":   audio_url,
            "QUEUE_ID":    queue_id,
            "QUEUE_NAME":  queue_name,
            "NEXT":        next_map,
        }

    if DEBUG_MODE:
        st.sidebar.write("DEBUG NODE_IDs:", list(nodes.keys()))

    return nodes


def load_scenarios():
    if not CSV_SCENARIOS.exists():
        st.error(f"No se encuentra el archivo de escenarios: {CSV_SCENARIOS}")
        st.stop()

    df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")

    required_cols = ["SCENARIO_ID", "TITLE", "MISSION_TEXT",
                     "ENTRY_NODE_ID", "EXPECTED_QUEUE_ID"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Falta la columna '{col}' en scenarios.csv")
            st.stop()

    # Columnas opcionales para finales múltiples
    if "EXPECTED_ALT_QUEUE_IDS" not in df.columns:
        df["EXPECTED_ALT_QUEUE_IDS"] = ""
    if "EXPECTED_OK_TYPES" not in df.columns:
        df["EXPECTED_OK_TYPES"] = ""

    # Nueva: asegurar EXECUTIONS
    if "EXECUTIONS" not in df.columns:
        df["EXECUTIONS"] = "0"

    scenarios = df.to_dict(orient="records"])
    scenarios = [s for s in scenarios if s.get("ACTIVE", "").strip().upper() == "TRUE"]
    return scenarios


NODES = load_nodes()
SCENARIOS = load_scenarios()

# Nodo raíz para '#'
if "ROOT" in NODES:
    ROOT_NODE_ID = "ROOT"
else:
    ROOT_NODE_ID = next(
        (nid for nid, n in NODES.items() if n.get("IS_ENTRY")),
        None,
    )


def load_scenarios_df_for_selection() -> pd.DataFrame:
    """
    Carga scenarios.csv en un DataFrame, asegurando:
      - Solo escenarios ACTIVE=TRUE
      - Columna EXECUTIONS existente e interpretada como int
    """
    if not CSV_SCENARIOS.exists():
        st.error(f"No se encuentra el archivo de escenarios: {CSV_SCENARIOS}")
        return pd.DataFrame()

    df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")

    # Filtramos activos
    df = df[df.get("ACTIVE", "").str.upper() == "TRUE"].copy()
    if df.empty:
        return df

    # Aseguramos columna EXECUTIONS
    if "EXECUTIONS" not in df.columns:
        df["EXECUTIONS"] = "0"

    # Columna numérica para poder ordenar
    df["EXECUTIONS_INT"] = (
        pd.to_numeric(df["EXECUTIONS"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return df
    # (bloque debug aquí nunca se ejecuta por el return anterior; lo dejo tal cual lo tenías)


# =========================
# SESIÓN
# =========================

def init_session():
    ss = st.session_state
    if "test_active" not in ss:
        ss.test_active = False
        ss.scenario = None
        ss.current_node_id = None
        ss.route = []
        ss.start_ts = None
        ss.finished = False
        ss.result = None
        ss.last_action = None      # repeat / goto_root / invalid / error / None
        ss.last_message = ""
        ss.last_played_node_id = None
        ss.did_initial_ring = False   # si ya se ha reproducido ring+prompt inicial
        ss.end_audio_played = False   # audio del nodo final reproducido
        ss.account_buffer = ""        # buffer para nodos ACCOUNT


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def start_new_test():
    """Elige un escenario (el de menos EXECUTIONS) y arranca directamente en IVR."""
    scenario = pick_next_scenario_least_executed()
    if scenario is None:
        st.error("No hay escenarios activos en config/scenarios.csv")
        return

    entry_node_id = str(scenario["ENTRY_NODE_ID"]).strip()

    if entry_node_id not in NODES:
        st.error(
            f"ENTRY_NODE_ID '{entry_node_id}' no existe en ivr_nodes.csv. "
            f"IDs disponibles: {list(NODES.keys())}"
        )
        return

    ss = st.session_state
    ss.test_active = True
    ss.scenario = scenario
    ss.current_node_id = entry_node_id
    ss.route = []
    ss.start_ts = datetime.now(timezone.utc).isoformat()
    ss.finished = False
    ss.result = None
    ss.last_action = None
    ss.last_message = ""
    ss.last_played_node_id = None
    ss.did_initial_ring = False
    ss.end_audio_played = False
    ss.account_buffer = ""


# =========================
# LÓGICA ESPECIAL ACCOUNT
# =========================
# (… aquí sigue TODO tu código de ACCOUNT, handle_key, send_result_to_github,
#  increment_scenario_executions y finish_test SIN cambios …)
# =========================
# UI
# =========================

def render_keypad():
    st.markdown("### Teclado")

    def make_button(label, col):
        if label == "*":
            display_label = r"\*"
        elif label == "#":
            display_label = r"\#"
        else:
            display_label = label

        col.button(
            display_label,
            use_container_width=True,
            key=f"btn_{label}",
            on_click=handle_key,
            args=(label,),
        )

    layout = [["1", "2", "3"],
              ["4", "5", "6"],
              ["7", "8", "9"],
              ["*", "0", "#"]]

    for row in layout:
        cols = st.columns(3)
        for i, key in enumerate(row):
            make_button(key, cols[i])

    # Nota de ayuda solo en modo debug
    if DEBUG_MODE:
        st.caption("⭐ '*' repite el mensaje del nodo actual · '#' vuelve al menú principal")

    # Botón de timeout para nodos ACCOUNT
    ss = st.session_state
    current_node = NODES.get(ss.current_node_id) if "current_node_id" in ss else None
    node_type = str(current_node.get("NODE_TYPE", "")).strip().upper() if current_node else ""

    if node_type == "ACCOUNT":
        st.button(
            "⏱️ Simular que el cliente no pulsa nada",
            use_container_width=True,
            key="btn_timeout_account",
            on_click=handle_account_timeout,
        )


def main():
    # 🔧 MODO MANTENIMIENTO
    # Si está activado y NO estamos en DEBUG, solo mostramos mensaje+imagen+cuenta atrás.
    if MAINTENANCE_MODE and not DEBUG_MODE:
        st.title("📞 IVR Tester (simulador de IVR)")
        st.subheader("🔧 Plataforma en actualización")
        st.info(MAINTENANCE_MESSAGE)

        # Imagen centrada si existe
        try:
            if MAINTENANCE_IMAGE_PATH.exists():
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(str(MAINTENANCE_IMAGE_PATH), use_column_width=True)
        except Exception as e:
            if DEBUG_MODE:
                st.sidebar.error(f"Error mostrando imagen de mantenimiento: {e}")

        # ⏳ Cuenta atrás
        st.markdown("### ⏱️ Cuenta atrás para la reapertura")

        # Guardamos en sesión la hora objetivo solo la primera vez
        ss = st.session_state
        if "maintenance_deadline" not in ss:
            now_ts = datetime.now(timezone.utc).timestamp()
            ss.maintenance_deadline = now_ts + MAINTENANCE_COUNTDOWN_SECONDS

        placeholder = st.empty()

        while True:
            now_ts = datetime.now(timezone.utc).timestamp()
            remaining = int(ss.maintenance_deadline - now_ts)

            if remaining <= 0:
                placeholder.markdown(
                    "⏱️ El tiempo de mantenimiento ha finalizado. "
                    "Si la plataforma sigue cerrada, prueba a recargar la página en unos instantes."
                )
                break

            hrs, rem = divmod(remaining, 3600)
            mins, secs = divmod(rem, 60)
            placeholder.markdown(f"**{hrs:02d}:{mins:02d}:{secs:02d}**")
            time.sleep(1)

        return

    init_session()
    ss = st.session_state

    st.title("📞 IVR Tester (simulador de IVR)")

    # =========================
    # ESTADO INICIAL (sin test activo)
    # =========================
    if not ss.test_active:
        st.write(
            "Pulsa el botón para recibir una misión aleatoria y probar la IVR como si fueras un cliente."
        )

        if st.button("🎬 Empezar nuevo test"):
            start_new_test()
            st.rerun()   # un solo clic: arranca test y se repinta

        return  # <-- este return va DENTRO del if not ss.test_active

    # =========================
    # TEST EN CURSO
    # =========================
    scenario = ss.scenario
    current_node = NODES.get(ss.current_node_id)

    if not scenario or not current_node:
        st.error("Error de estado interno. Puedes reiniciar el test.")
        if st.button("Reiniciar todo"):
            reset_session()
            st.rerun()
        return

    # Bloque: misión
    st.subheader("📝 Tu misión")
    if DEBUG_MODE:
        st.write(f"**{scenario['TITLE']}**")
    st.info(scenario["MISSION_TEXT"])

    # =========================
    # TEST TERMINADO
    # =========================
    if ss.finished and ss.result:
        st.subheader("✅ Test finalizado")

        # Reproducir audio del nodo final (QUEUE/SMS/TRANSFER) una vez
        end_type = ss.result.get("end_node_type")
        if end_type in ("QUEUE", "SMS", "TRANSFER") and not ss.get("end_audio_played", False):
            end_node_id = ss.result.get("end_node_id")
            node_for_audio = NODES.get(end_node_id, current_node)
            if node_for_audio:
                play_node_audio(node_for_audio)
            ss.end_audio_played = True

        # Vista TESTER
        st.success("La misión ha finalizado. Gracias por completar el test. 🙌")

        # Vista DEBUG
        if DEBUG_MODE:
            result_type = ss.result["result"]
            st.markdown("### 🔍 Detalles internos (solo debug)")
            st.write(f"Resultado lógico: `{result_type}`")
            st.write(
                f"- Cola esperada principal: `{ss.result['expected_queue_id']}`\n"
                f"- Cola alcanzada: `{ss.result['reached_queue_id']}` "
                f"({ss.result.get('queue_name','')})\n"
                f"- Nodo final: `{ss.result.get('end_node_id','')}` "
                f"({ss.result.get('end_node_type','')})"
            )

            with st.expander("Ruta seguida (JSON)"):
                st.json(ss.route)

            if "last_result_row" in ss:
                with st.expander("Último registro enviado a GitHub"):
                    st.json(ss.last_result_row)

        st.divider()
        if st.button("🔁 Empezar otro test"):
            start_new_test()
            st.rerun()   # un solo clic para empezar otro

        return  # aquí sí termina la función

    # =========================
    # AUDIO INICIAL / CAMBIO DE NODO
    # =========================
    if not ss.did_initial_ring:
        if DEBUG_MODE:
            st.subheader("☎️ Llamando a la IVR...")
            st.caption("Escuchas el tono de llamada y, a continuación, el mensaje del menú correspondiente.")
        play_ring_and_prompt(current_node)
        ss.did_initial_ring = True
        ss.last_played_node_id = current_node["NODE_ID"]
    else:
        if DEBUG_MODE:
            st.subheader("📟 Llamada IVR (simulada)")

        # Si cambiamos de nodo, reproducimos el audio una vez
        if ss.last_played_node_id != current_node["NODE_ID"]:
            play_node_audio(current_node)
            ss.last_played_node_id = current_node["NODE_ID"]

    # Texto del nodo solo en debug
    if DEBUG_MODE:
        prompt_text = current_node.get("PROMPT_TEXT", "")
        st.write(f"🗣️ {prompt_text}")

    # Mensajes de estado
    if ss.last_message:
        if ss.last_action in ("invalid",):
            st.warning(ss.last_message)
        elif ss.last_action in ("repeat", "goto_root"):
            st.info(ss.last_message)
        elif ss.last_action == "error":
            st.error(ss.last_message)

    # Teclado
    render_keypad()

    st.divider()
    if st.button("❌ Cancelar test"):
        reset_session()
        st.rerun()


if __name__ == "__main__":
    main()

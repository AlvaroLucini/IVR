#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import random
import base64
import json
from uuid import uuid4
import time  # por si en el futuro quieres usarlo

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
MAINTENANCE_MODE = bool(st.secrets.get("maintenance_mode", True))

MAINTENANCE_MESSAGE = (
    "Estamos actualizando la IVR para testar una nueva estructura. "
    "En breves momentos volveremos a abrir la plataforma para testarla. 🙏"
)

# ⏳ Duración de la cuenta atrás (en segundos)
# Cámbialo cuando cierres la plataforma (p. ej. 30*60 = 30 minutos)
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

# Imagen de mantenimiento
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

    scenarios = df.to_dict(orient="records")
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
    # (el bloque debug que tenías aquí no se ejecuta nunca por el return, lo dejo igual)


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

def handle_account_key(key: str, current_node: dict):
    """
    Lógica especial para nodos de tipo ACCOUNT:
    - 0-9: se acumulan en un buffer de cuenta.
    - *  : confirma la cuenta y pasa a OPT_0_NEXT_NODE (o ROOT si no está).
    - #  : vuelve a ROOT.
    """
    ss = st.session_state
    now = datetime.now(timezone.utc).isoformat()
    node_id = current_node["NODE_ID"]

    if "account_buffer" not in ss:
        ss.account_buffer = ""

    # '#' => volver a ROOT
    if key == "#":
        ss.route.append({
            "step": len(ss.route) + 1,
            "node_id": node_id,
            "digit": "#",
            "timestamp": now,
            "event": "ACCOUNT_BACK_TO_ROOT",
            "account_buffer": ss.account_buffer,
        })
        ss.current_node_id = ROOT_NODE_ID
        ss.account_buffer = ""
        ss.last_action = "goto_root"
        ss.last_message = ""
        ss.last_played_node_id = None
        return

    # Dígitos => acumular
    if key in "0123456789":
        ss.account_buffer += key
        ss.route.append({
            "step": len(ss.route) + 1,
            "node_id": node_id,
            "digit": key,
            "timestamp": now,
            "event": "ACCOUNT_DIGIT",
            "account_buffer": ss.account_buffer,
        })
        ss.last_action = "account_digit"
        ss.last_message = ""
        # Nos quedamos en el mismo nodo
        return

    # '*' => confirmar número y pasar a OPT_0_NEXT_NODE (o ROOT)
    if key == "*":
        dest_id = current_node["NEXT"].get("0") or ROOT_NODE_ID

        ss.route.append({
            "step": len(ss.route) + 1,
            "node_id": node_id,
            "digit": "*",
            "timestamp": now,
            "event": "ACCOUNT_OK",
            "account_buffer": ss.account_buffer,
        })

        ss.account_buffer = ""
        ss.current_node_id = dest_id
        ss.last_action = None
        ss.last_message = ""
        ss.last_played_node_id = None

        new_node = NODES.get(dest_id)
        if not new_node:
            ss.last_action = "error"
            ss.last_message = f"Nodo destino '{dest_id}' no definido en ivr_nodes.csv."
            return

        node_type2 = str(new_node.get("NODE_TYPE", "")).strip().upper()
        if node_type2 in ("QUEUE", "SMS", "TRANSFER"):
            finish_test(new_node)
        return

    # Cualquier otra cosa -> no válida
    ss.last_action = "invalid"
    ss.last_message = "Opción no válida en este menú."
    return


def handle_account_timeout():
    """Simula que el cliente no pulsa nada en un nodo ACCOUNT (timeout)."""
    ss = st.session_state
    current_node = NODES.get(ss.current_node_id)
    if not current_node:
        return

    node_type = str(current_node.get("NODE_TYPE", "")).strip().upper()
    if node_type != "ACCOUNT":
        return

    now = datetime.now(timezone.utc).isoformat()
    node_id = current_node["NODE_ID"]
    dest_id = current_node["NEXT"].get("0") or ROOT_NODE_ID

    ss.route.append({
        "step": len(ss.route) + 1,
        "node_id": node_id,
        "digit": "TIMEOUT",
        "timestamp": now,
        "event": "ACCOUNT_TIMEOUT",
        "account_buffer": ss.get("account_buffer", ""),
    })

    ss.account_buffer = ""
    ss.current_node_id = dest_id
    ss.last_action = None
    ss.last_message = ""
    ss.last_played_node_id = None

    new_node = NODES.get(dest_id)
    if not new_node:
        ss.last_action = "error"
        ss.last_message = f"Nodo destino '{dest_id}' no definido en ivr_nodes.csv."
        return

    node_type2 = str(new_node.get("NODE_TYPE", "")).strip().upper()
    if node_type2 in ("QUEUE", "SMS", "TRANSFER"):
        finish_test(new_node)


# =========================
# MANEJO DE TECLAS GENERAL
# =========================

def handle_key(key: str):
    ss = st.session_state

    if not ss.test_active or ss.finished:
        return

    current_node = NODES.get(ss.current_node_id)
    if not current_node:
        ss.last_action = "error"
        ss.last_message = "Nodo actual no encontrado en la configuración."
        return

    node_type_current = str(current_node.get("NODE_TYPE", "")).strip().upper()

    # Nodos ACCOUNT tienen lógica especial
    if node_type_current == "ACCOUNT":
        handle_account_key(key, current_node)
        return

    # '*': repetir mensaje del nodo actual (solo prompt, sin tono)
    if key == "*":
        ss.route.append(
            {
                "step": len(ss.route) + 1,
                "node_id": current_node["NODE_ID"],
                "digit": "*",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        ss.last_action = "repeat"
        ss.last_message = "Repitiendo el mensaje del nodo."

        # Reproducimos directamente el audio del nodo aquí
        play_node_audio(current_node)
        # No tocamos last_played_node_id para no interferir con cambios de nodo
        return

    # '#': ir al ROOT
    if key == "#":
        if not ROOT_NODE_ID or ROOT_NODE_ID not in NODES:
            ss.last_action = "error"
            ss.last_message = "No se ha encontrado el nodo raíz (ROOT) en la configuración."
            return

        ss.route.append(
            {
                "step": len(ss.route) + 1,
                "node_id": current_node["NODE_ID"],
                "digit": "#",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        ss.current_node_id = ROOT_NODE_ID
        ss.last_action = "goto_root"
        ss.last_message = "Has vuelto al menú principal."
        ss.last_played_node_id = None
        return

    # 0..9
    if key not in "0123456789":
        ss.last_action = "error"
        ss.last_message = "Tecla no reconocida."
        return

    # Opciones “corridas”: la opción del 1 está en OPT_0, la del 2 en OPT_1, etc.
    d = int(key)
    if d == 0:
        lookup_digit = "0"
    else:
        lookup_digit = str(d - 1)

    next_id = current_node["NEXT"].get(lookup_digit, "")

    if not next_id:
        ss.last_action = "invalid"
        ss.last_message = "Opción no válida en este menú."
        return

    ss.route.append(
        {
            "step": len(ss.route) + 1,
            "node_id": current_node["NODE_ID"],
            "digit": key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    ss.current_node_id = next_id
    new_node = NODES.get(next_id)

    if not new_node:
        ss.last_action = "error"
        ss.last_message = f"Nodo destino '{next_id}' no definido en ivr_nodes.csv."
        return

    # Si el nuevo nodo es ACCOUNT, vaciamos el buffer
    if str(new_node.get("NODE_TYPE", "")).strip().upper() == "ACCOUNT":
        ss.account_buffer = ""

    ss.last_action = None
    ss.last_message = ""
    ss.last_played_node_id = None  # nuevo nodo -> reproducir audio

    # Si llegamos a una cola, a un nodo SMS o a un nodo TRANSFER, cerramos el test
    node_type = str(new_node["NODE_TYPE"]).strip().upper()
    if node_type in ("QUEUE", "SMS", "TRANSFER"):
        finish_test(new_node)


# =========================
# ENVÍO RESULTADOS A GITHUB
# =========================

def send_result_to_github(row: dict):
    """
    Envía el resultado de un test a GitHub como un JSON individual:
    test_results/YYYY-MM-DD/<test_id>.json
    """
    if DEBUG_MODE:
        st.sidebar.write("DEBUG: entrando en send_result_to_github")

    try:
        gh_conf = st.secrets["github"]
        token = gh_conf["token"]
        repo = gh_conf["repo"]
        branch = gh_conf.get("branch", "main")
    except Exception as e:
        if DEBUG_MODE:
            st.sidebar.error(f"DEBUG: st.secrets['github'] no disponible: {e}")
        return

    # Carpeta y nombre de archivo
    ts = row.get("timestamp_utc")
    if not ts:
        ts = datetime.now(timezone.utc).isoformat()
        row["timestamp_utc"] = ts

    date_str = ts[:10]  # YYYY-MM-DD
    test_id = row.get("test_id")
    if not test_id:
        test_id = str(uuid4())
        row["test_id"] = test_id

    path = f"test_results/{date_str}/{test_id}.json"
    if DEBUG_MODE:
        st.sidebar.write(f"DEBUG: path en GitHub: {path}")

    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"

    content_bytes = json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8")
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    payload = {
        "message": f"Add IVR test result {test_id}",
        "content": content_b64,
        "branch": branch,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if DEBUG_MODE:
            st.sidebar.write(f"DEBUG: GitHub status {resp.status_code}")
        if resp.status_code not in (200, 201):
            if DEBUG_MODE:
                st.sidebar.error(f"GitHub error: {resp.status_code} - {resp.text[:200]}")
        else:
            if DEBUG_MODE:
                st.sidebar.success(f"Resultado guardado en GitHub: {path}")
    except Exception as e:
        if DEBUG_MODE:
            st.sidebar.error(f"Error enviando resultado a GitHub: {e}")


def increment_scenario_executions(scenario_id: str):
    """
    Suma 1 a EXECUTIONS del escenario dado.

    Prioridad:
      1) Leer y actualizar scenarios.csv desde GitHub (fuente de verdad).
      2) Si GitHub falla, usar el CSV local como fallback.

    Solo se modifica EXECUTIONS; el resto de columnas (TITLE, MISSION_TEXT, etc.)
    se respetan tal cual estén en el CSV origen.
    """
    from io import StringIO  # import local para no tocar cabecera

    global SCENARIOS

    sid = str(scenario_id).strip()
    if not sid:
        if DEBUG_MODE:
            st.sidebar.error("DEBUG EXEC: scenario_id vacío al incrementar EXECUTIONS.")
        return

    if DEBUG_MODE:
        st.sidebar.write(f"DEBUG EXEC: increment_scenario_executions('{sid}')")

    df = None
    sha = None
    gh_conf = st.secrets.get("github", None)

    # =========================
    # 1) INTENTAR LEER DESDE GITHUB
    # =========================
    if gh_conf:
        try:
            token = gh_conf["token"]
            repo = gh_conf["repo"]
            branch = gh_conf.get("branch", "main")

            rel_path = "config/scenarios.csv"
            api_base = f"https://api.github.com/repos/{repo}/contents"
            get_url = f"{api_base}/{rel_path}?ref={branch}"

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            }

            resp_get = requests.get(get_url, headers=headers, timeout=15)

            if resp_get.status_code == 200:
                data_get = resp_get.json()
                sha = data_get.get("sha")
                content_b64 = data_get.get("content", "")

                csv_bytes = base64.b64decode(content_b64)
                csv_text = csv_bytes.decode("utf-8")

                df = pd.read_csv(StringIO(csv_text), dtype=str).fillna("")

                if DEBUG_MODE:
                    st.sidebar.write("DEBUG EXEC: scenarios.csv cargado desde GitHub.")
            else:
                if DEBUG_MODE:
                    st.sidebar.warning(
                        f"DEBUG EXEC: no se pudo leer scenarios.csv desde GitHub "
                        f"(status {resp_get.status_code}). Se usará CSV local."
                    )
        except Exception as e:
            if DEBUG_MODE:
                st.sidebar.error(f"DEBUG EXEC: error leyendo scenarios.csv desde GitHub: {e}")

    # =========================
    # 2) FALLBACK: LEER CSV LOCAL SI HACE FALTA
    # =========================
    if df is None:
        try:
            df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")
            if DEBUG_MODE:
                st.sidebar.write("DEBUG EXEC: scenarios.csv cargado desde disco local.")
        except Exception as e:
            if DEBUG_MODE:
                st.sidebar.error(f"DEBUG EXEC: no se pudo leer scenarios.csv local: {e}")
            return

    # =========================
    # 3) ACTUALIZAR COLUMNA EXECUTIONS
    # =========================
    if "SCENARIO_ID" not in df.columns:
        if DEBUG_MODE:
            st.sidebar.error("DEBUG EXEC: scenarios.csv no tiene columna SCENARIO_ID.")
        return

    if "EXECUTIONS" not in df.columns:
        df["EXECUTIONS"] = "0"

    df["SCENARIO_ID"] = df["SCENARIO_ID"].astype(str)
    df["SCENARIO_ID_STRIP"] = df["SCENARIO_ID"].str.strip()

    if DEBUG_MODE:
        st.sidebar.write(
            "DEBUG EXEC: SCENARIO_ID distintos en CSV: "
            f"{sorted(df['SCENARIO_ID_STRIP'].unique().tolist())}"
        )

    mask = df["SCENARIO_ID_STRIP"] == sid

    if not mask.any():
        if DEBUG_MODE:
            st.sidebar.warning(f"DEBUG EXEC: SCENARIO_ID '{sid}' no encontrado en scenarios.csv.")
        df = df.drop(columns=["SCENARIO_ID_STRIP"])
        return

    before_vals = df.loc[mask, "EXECUTIONS"].tolist()
    if DEBUG_MODE:
        st.sidebar.write(f"DEBUG EXEC: valores EXECUTIONS antes = {before_vals}")

    execs = (
        pd.to_numeric(df.loc[mask, "EXECUTIONS"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df.loc[mask, "EXECUTIONS"] = (execs + 1).astype(str)

    after_vals = df.loc[mask, "EXECUTIONS"].tolist()
    if DEBUG_MODE:
        st.sidebar.write(f"DEBUG EXEC: valores EXECUTIONS después = {after_vals}")

    df = df.drop(columns=["SCENARIO_ID_STRIP"])

    # =========================
    # 4) GUARDAR EN DISCO LOCAL
    # =========================
    try:
        df.to_csv(CSV_SCENARIOS, index=False)
        if DEBUG_MODE:
            st.sidebar.success("DEBUG EXEC: scenarios.csv actualizado localmente.")
    except Exception as e:
        if DEBUG_MODE:
            st.sidebar.error(f"DEBUG EXEC: no se pudo guardar scenarios.csv local: {e}")
        # aunque falle el save local, seguimos intentando subir a GitHub

    # =========================
    # 5) SUBIR A GITHUB (SI TENEMOS CONF)
    # =========================
    if gh_conf:
        try:
            token = gh_conf["token"]
            repo = gh_conf["repo"]
            branch = gh_conf.get("branch", "main")

            rel_path = "config/scenarios.csv"
            api_base = f"https://api.github.com/repos/{repo}/contents"
            put_url = f"{api_base}/{rel_path}"

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            }

            content_bytes = Path(CSV_SCENARIOS).read_bytes()
            content_b64 = base64.b64encode(content_bytes).decode("utf-8")

            payload = {
                "message": f"Update scenarios EXECUTIONS for {sid}",
                "content": content_b64,
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha  # necesario para actualizar un fichero existente

            resp_put = requests.put(put_url, headers=headers, json=payload, timeout=20)

            if DEBUG_MODE:
                st.sidebar.write(
                    f"DEBUG EXEC: resultado PUT scenarios.csv GitHub: "
                    f"{resp_put.status_code} - {resp_put.text[:200]}"
                )

            if resp_put.status_code not in (200, 201) and DEBUG_MODE:
                st.sidebar.error(
                    "DEBUG EXEC: fallo al subir scenarios.csv actualizado a GitHub. "
                    "El reparto interno de escenarios sigue funcionando."
                )
        except Exception as e:
            if DEBUG_MODE:
                st.sidebar.error(f"DEBUG EXEC: excepción al subir scenarios.csv a GitHub: {e}")

    # =========================
    # 6) REFRESCAR SCENARIOS EN MEMORIA
    # =========================
    try:
        SCENARIOS = load_scenarios()
    except Exception as e:
        if DEBUG_MODE:
            st.sidebar.error(f"DEBUG EXEC: error recargando SCENARIOS: {e}")


def finish_test(end_node: dict):
    ss = st.session_state
    scenario = ss.scenario

    node_type = str(end_node.get("NODE_TYPE", "")).strip().upper()

    # ---- Leer expectativas del escenario ----
    expected_queue_id = (scenario.get("EXPECTED_QUEUE_ID") or "").strip()

    alt_raw = (scenario.get("EXPECTED_ALT_QUEUE_IDS") or "").strip()
    ok_types_raw = (scenario.get("EXPECTED_OK_TYPES") or "").strip()

    ok_queues = {q.strip() for q in ([expected_queue_id] + alt_raw.split("|")) if q.strip()}
    ok_types = {t.strip().upper() for t in ok_types_raw.split("|") if t.strip()}

    reached_queue_id = end_node.get("QUEUE_ID", "")
    queue_name = end_node.get("QUEUE_NAME", "")

    # ---- Reglas de éxito ----
    is_ok_queue = (node_type == "QUEUE") and reached_queue_id in ok_queues
    is_ok_type = node_type in ok_types

    if is_ok_queue or is_ok_type:
        result = "SUCCESS"
    else:
        if node_type == "QUEUE":
            result = "WRONG_QUEUE"
        else:
            result = f"WRONG_END_{node_type or 'UNKNOWN'}"

    ss.finished = True
    ss.result = {
        "result": result,
        "expected_queue_id": expected_queue_id,
        "reached_queue_id": reached_queue_id,
        "queue_name": queue_name,
        "end_node_id": end_node.get("NODE_ID", ""),
        "end_node_type": node_type,
    }
    ss.end_audio_played = False  # aún no hemos lanzado el audio del nodo final

    # ✅ IMPORTANTÍSIMO: actualizar EXECUTIONS SIEMPRE, aunque luego falle el log
    scenario_id = scenario.get("SCENARIO_ID")
    if scenario_id:
        if DEBUG_MODE:
            st.sidebar.write(f"DEBUG EXEC: incrementando EXECUTIONS para {scenario_id}")
        increment_scenario_executions(scenario_id)

    # Registro en GitHub (esto ya es "extra", no debe bloquear el contador)
    try:
        start_ts_str = ss.start_ts
        start_dt = datetime.fromisoformat(start_ts_str)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        end_dt = datetime.now(timezone.utc)
        duration_seconds = (end_dt - start_dt).total_seconds()

        route = ss.route or []
        route_json = json.dumps(route, ensure_ascii=False)

        row = {
            "test_id": str(uuid4()),
            "timestamp_utc": end_dt.isoformat(),
            "scenario_id": scenario["SCENARIO_ID"],
            "scenario_title": scenario["TITLE"],
            "entry_node_id": scenario["ENTRY_NODE_ID"],
            "expected_queue_id": expected_queue_id,
            "reached_queue_id": reached_queue_id,
            "reached_queue_name": queue_name,
            "end_node_id": end_node.get("NODE_ID", ""),
            "end_node_type": node_type,
            "result": result,
            "duration_seconds": round(duration_seconds, 3),
            "num_steps": len(route),
            "route_json": route_json,
        }

        send_result_to_github(row)
        ss.last_result_row = row

    except Exception as e:
        if DEBUG_MODE:
            st.sidebar.error(f"Error guardando el resultado del test (GitHub/log): {e}")


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
    # Si está activado y NO estamos en DEBUG, solo mostramos mensaje+imagen.
    if MAINTENANCE_MODE and not DEBUG_MODE:
        st.title("📞 IVR Tester (simulador de IVR)")
        st.subheader("🔧 Plataforma en actualización - Test #3")
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

        # (Sin cuenta atrás: solo mensaje estático)
        return

    init_session()
    ss = st.session_state

    st.title("📞 IVR Tester - Test #3")

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





















#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import random
import base64

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components   # 👈 AÑADE ESTA LÍNEA


# =========================
# CONFIG PÁGINA
# =========================

st.set_page_config(page_title="IVR Tester", page_icon="📞", layout="centered")


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


# =========================
# HELPERS AUDIO
# =========================

def looks_like_audio_ref(s: str) -> bool:
    """Heurística para saber si AUDIO_URL parece fichero/URL de audio."""
    if not s:
        return False
    s = str(s).strip()
    if not s:
        return False

    if s.lower().startswith(("http://", "https://")):
        return True
    if "/" in s or "\\" in s:
        return True
    for ext in (".wav", ".mp3", ".ogg", ".m4a"):
        if s.lower().endswith(ext):
            return True
    return False


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


def play_hidden_audio(path: Path, loop: bool = False) -> bool:
    """Reproduce un audio local sin controles usando <audio autoplay>."""
    if not path.exists():
        return False

    try:
        src, mime = _audio_path_to_src(path)
    except Exception:
        return False

    loop_attr = "loop" if loop else ""
    html = f"""
    <audio autoplay {loop_attr}>
        <source src="{src}" type="{mime}">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)
    return True


def play_hidden_audio_url(url: str, loop: bool = False, mime: str = "audio/mpeg") -> bool:
    """Reproduce una URL HTTP(S) sin controles usando <audio autoplay>."""
    if not url:
        return False

    url = str(url).strip()
    if not url:
        return False

    loop_attr = "loop" if loop else ""
    html = f"""
    <audio autoplay {loop_attr}>
        <source src="{url}" type="{mime}">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)
    return True


def play_ringtone_once():
    """Intenta reproducir el ringtone una vez (oculto)."""
    ring_file = next((p for p in RING_PATHS if p.exists()), None)
    if ring_file:
        st.caption(f"Tono → {ring_file.relative_to(BASE_DIR)}")
        play_hidden_audio(ring_file)
    else:
        st.caption("Simulando tonos de llamada… (añade ringtone.wav en audio/ o tts_audio/).")


# ========== AQUÍ ESTÁ EL CAMBIO IMPORTANTE ==========
def play_node_audio(node: dict) -> bool:
    """
    Reproduce el audio asociado a un nodo.

    Prioridad:
      1) audio/{NODE_ID}_es.wav
      2) audio/{NODE_ID}.wav
      3) tts_audio/{NODE_ID}_es.wav
      4) AUDIO_URL

    Intento 1: usar <audio> con JS (audio.play()) en un componente oculto.
    Fallback: reproductor visible st.audio por si el navegador bloquea el autoplay.
    """
    node_id = node["NODE_ID"]

    # --- Local paths candidatos ---
    candidatos: list[Path] = []
    audio_es_path = BASE_DIR / "audio" / f"{node_id}_es.wav"
    candidatos.append(audio_es_path)

    audio_plain_path = BASE_DIR / "audio" / f"{node_id}.wav"
    candidatos.append(audio_plain_path)

    tts_path = BASE_DIR / "tts_audio" / f"{node_id}_es.wav"
    candidatos.append(tts_path)

    # 1–3) Ficheros locales
    for p in candidatos:
        if p.exists():
            st.caption(f"Reproduciendo audio: {p.relative_to(BASE_DIR)}")

            # Leemos y generamos data:URL para inyectar en un componente HTML+JS
            try:
                audio_bytes = p.read_bytes()
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                ext = p.suffix.lower()
                if ext == ".wav":
                    mime = "audio/wav"
                elif ext in (".mp3", ".mpeg"):
                    mime = "audio/mpeg"
                elif ext == ".ogg":
                    mime = "audio/ogg"
                else:
                    mime = "audio/wav"

                html = f"""
                <audio id="ivr_prompt_audio" preload="auto">
                    <source src="data:{mime};base64,{b64}" type="{mime}">
                </audio>
                <script>
                (function() {{
                    var audio = document.getElementById("ivr_prompt_audio");
                    if (!audio) return;
                    // Intento de reproducir inmediatamente
                    var playPromise = audio.play();
                    if (playPromise !== undefined) {{
                        playPromise.catch(function(err) {{
                            console.log("Autoplay bloqueado o error:", err);
                        }});
                    }}
                }})();
                </script>
                """
                # Componente oculto (altura 0) para que se ejecute el JS en el cliente
                components.html(html, height=0, width=0)
            except Exception as e:
                st.caption(f"No se pudo inyectar audio vía JS: {e}")

            # Fallback: reproductor visible por si el navegador bloquea todo
            try:
                st.audio(audio_bytes, format=mime)
            except Exception:
                pass

            return True

    # 4) AUDIO_URL
    audio_url = str(node.get("AUDIO_URL", "")).strip()
    if looks_like_audio_ref(audio_url):
        st.caption(f"Reproduciendo AUDIO_URL: {audio_url}")
        # Para URLs no hacemos base64, dejamos que el navegador resuelva
        html = f"""
        <audio id="ivr_prompt_audio_url" preload="auto">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById("ivr_prompt_audio_url");
            if (!audio) return;
            var playPromise = audio.play();
            if (playPromise !== undefined) {{
                playPromise.catch(function(err) {{
                    console.log("Autoplay bloqueado o error (URL):", err);
                }});
            }}
        }})();
        </script>
        """
        components.html(html, height=0, width=0)
        # Fallback visible
        st.audio(audio_url)
        return True

    buscados = ", ".join(str(p.relative_to(BASE_DIR)) for p in candidatos)
    st.caption(f"Sin audio para {node_id}. Buscados: {buscados} y AUDIO_URL='{audio_url}'")
    return False



# =========================
# CARGA DE CONFIGURACIÓN
# =========================

@st.cache_data
def load_nodes():
    if not CSV_NODES.exists():
        st.error(f"No se encuentra el archivo de nodos: {CSV_NODES}")
        st.stop()

    df = pd.read_csv(CSV_NODES, dtype=str, index_col=0).fillna("")

    nodes = {}
    for idx, row in df.iterrows():
        node_id = str(idx).strip()  # ROOT, L1_PEDIDOS, ...

        node_label = row.get("NODE_ID", "")
        node_type = row.get("NODE_LABEL", "")
        is_entry_flag = str(row.get("NODE_TYPE", "")).strip().upper() == "YES"

        prompt_text = row.get("PROMPT_TEXT", "")
        if not str(prompt_text).strip():
            prompt_text = row.get("IS_ENTRY", "")

        next_map = {}
        for d in range(10):
            col_name = f"OPT_{d}_NEXT_NODE"
            if col_name in row.index:
                next_map[str(d)] = row[col_name]
            else:
                next_map[str(d)] = ""

        nodes[node_id] = {
            "NODE_ID":      node_id,
            "NODE_LABEL":   node_label,
            "NODE_TYPE":    node_type,          # MENU / QUEUE
            "IS_ENTRY":     is_entry_flag,
            "PROMPT_TEXT":  prompt_text,
            "AUDIO_URL":    row.get("AUDIO_URL", ""),
            "QUEUE_ID":     row.get("QUEUE_ID", ""),
            "QUEUE_NAME":   row.get("QUEUE_NAME", ""),
            "NEXT":         next_map,
        }

    return nodes


@st.cache_data
def load_scenarios():
    if not CSV_SCENARIOS.exists():
        st.error(f"No se encuentra el archivo de escenarios: {CSV_SCENARIOS}")
        st.stop()

    df = pd.read_csv(CSV_SCENARIOS, dtype=str).fillna("")

    required_cols = ["SCENARIO_ID", "TITLE", "MISSION_TEXT", "ENTRY_NODE_ID", "EXPECTED_QUEUE_ID"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Falta la columna '{col}' en scenarios.csv")
            st.stop()

    scenarios = df.to_dict(orient="records")
    scenarios = [s for s in scenarios if s.get("ACTIVE", "").strip().upper() == "TRUE"]
    return scenarios


NODES = load_nodes()
SCENARIOS = load_scenarios()

# Nodo raíz para '#'
if "ROOT" in NODES:
    ROOT_NODE_ID = "ROOT"
else:
    ROOT_NODE_ID = next((nid for nid, n in NODES.items() if n.get("IS_ENTRY")), None)


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
        ss.phase = "idle"          # idle | ringing | ivr | done
        ss.last_played_node_id = None


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def start_new_test():
    """Elige un escenario y pasa a fase 'ringing'."""
    if not SCENARIOS:
        st.error("No hay escenarios activos en config/scenarios.csv")
        return

    scenario = random.choice(SCENARIOS)
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
    ss.start_ts = datetime.utcnow().isoformat()
    ss.finished = False
    ss.result = None
    ss.last_action = None
    ss.last_message = ""
    ss.phase = "ringing"
    ss.last_played_node_id = None


def handle_key(key: str):
    ss = st.session_state

    if not ss.test_active or ss.finished:
        return
    if ss.phase != "ivr":
        return

    current_node = NODES.get(ss.current_node_id)
    if not current_node:
        ss.last_action = "error"
        ss.last_message = "Nodo actual no encontrado en la configuración."
        return

    # '*': repetir mensaje del nodo actual
    if key == "*":
        ss.route.append(
            {
                "step": len(ss.route) + 1,
                "node_id": current_node["NODE_ID"],
                "digit": "*",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        ss.last_action = "repeat"
        ss.last_message = "Repitiendo el mensaje del nodo."
        ss.last_played_node_id = None
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
                "timestamp": datetime.utcnow().isoformat(),
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

    next_id = current_node["NEXT"].get(key, "")

    if not next_id:
        ss.last_action = "invalid"
        ss.last_message = "Opción no válida en este menú."
        return

    ss.route.append(
        {
            "step": len(ss.route) + 1,
            "node_id": current_node["NODE_ID"],
            "digit": key,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    ss.current_node_id = next_id
    new_node = NODES.get(next_id)

    if not new_node:
        ss.last_action = "error"
        ss.last_message = f"Nodo destino '{next_id}' no definido en ivr_nodes.csv."
        return

    ss.last_action = None
    ss.last_message = ""
    ss.last_played_node_id = None

    if str(new_node["NODE_TYPE"]).strip().upper() == "QUEUE":
        finish_test(new_node)


def finish_test(queue_node: dict):
    ss = st.session_state
    scenario = ss.scenario

    expected_queue_id = scenario["EXPECTED_QUEUE_ID"]
    reached_queue_id = queue_node.get("QUEUE_ID", "")

    if reached_queue_id == expected_queue_id:
        result = "SUCCESS"
    else:
        result = "WRONG_QUEUE"

    ss.finished = True
    ss.phase = "done"
    ss.result = {
        "result": result,
        "expected_queue_id": expected_queue_id,
        "reached_queue_id": reached_queue_id,
        "queue_name": queue_node.get("QUEUE_NAME", ""),
    }


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

    st.caption("⭐ '*' repite el mensaje · '#' vuelve al menú principal")


def main():
    init_session()
    ss = st.session_state

    st.title("📞 IVR Tester (simulador de IVR)")

    # Estado inicial
    if not ss.test_active:
        ss.phase = "idle"
        st.write(
            "Pulsa el botón para recibir una misión aleatoria y probar la IVR como si fueras un cliente."
        )

        if st.button("🎬 Empezar nuevo test"):
            start_new_test()
        return

    scenario = ss.scenario
    current_node = NODES.get(ss.current_node_id)

    if not scenario or not current_node:
        st.error("Error de estado interno. Puedes reiniciar el test.")
        if st.button("Reiniciar todo"):
            reset_session()
        return

    # Bloque: misión
    st.subheader("📝 Tu misión")
    st.write(f"**{scenario['TITLE']}**")
    st.info(scenario["MISSION_TEXT"])

    # Test terminado
    if ss.finished and ss.result:
        st.subheader("✅ Gracias por completar la prueba")

        if ss.result["result"] == "SUCCESS":
            st.success(
                f"Llegaste a la cola correcta: `{ss.result['reached_queue_id']}` "
                f"({ss.result.get('queue_name','')})."
            )
        else:
            st.error(
                "La cola alcanzada no coincide con la esperada.\n\n"
                f"- Esperada: `{ss.result['expected_queue_id']}`\n"
                f"- Alcanzada: `{ss.result['reached_queue_id']}` "
                f"({ss.result.get('queue_name','')})"
            )

        with st.expander("Ver ruta seguida (para análisis interno)"):
            st.json(ss.route)

        st.divider()
        if st.button("🔁 Empezar otro test"):
            reset_session()

        return

    # ===== FASE RINGING =====
    if ss.phase == "ringing":
        st.subheader("☎️ Llamando a la IVR...")
        play_ringtone_once()
        st.caption("Escuchas el tono de llamada…")

        if st.button("📞 Descolgar teléfono"):
            ss.phase = "ivr"
            ss.last_played_node_id = None

        st.divider()
        if st.button("❌ Cancelar test"):
            reset_session()

        return

    # ===== FASE IVR =====
    st.subheader("📟 Llamada IVR (simulada)")

    if ss.last_played_node_id != current_node["NODE_ID"]:
        play_node_audio(current_node)
        ss.last_played_node_id = current_node["NODE_ID"]

    prompt_text = current_node.get("PROMPT_TEXT", "")
    st.write(f"🗣️ {prompt_text}")

    if ss.last_message:
        if ss.last_action in ("invalid",):
            st.warning(ss.last_message)
        elif ss.last_action in ("repeat", "goto_root"):
            st.info(ss.last_message)
        elif ss.last_action == "error":
            st.error(ss.last_message)

    render_keypad()

    st.divider()
    if st.button("❌ Cancelar test"):
        reset_session()


if __name__ == "__main__":
    main()



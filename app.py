#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import random
import base64

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # para inyectar JS


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
    """
    src, mime = get_node_audio_source(node)
    if src is None:
        return False

    # Caso base64 local
    if mime != "url":
        html = f"""
        <audio id="ivr_prompt_only" preload="auto">
            <source src="{src}" type="{mime}">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById("ivr_prompt_only");
            if (!audio) return;
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
    else:
        # URL http/https
        html = f"""
        <audio id="ivr_prompt_only_url" preload="auto">
            <source src="{src}" type="audio/mpeg">
        </audio>
        <script>
        (function() {{
            var audio = document.getElementById("ivr_prompt_only_url");
            if (!audio) return;
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
    ROOT_NODE_ID = next(
        (nid for nid, n in NODES.items() if n.get("IS_ENTRY")),
        None,
    )


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


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def start_new_test():
    """Elige un escenario y arranca directamente en IVR."""
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
    ss.last_played_node_id = None
    ss.did_initial_ring = False   # aún no hemos hecho ring+prompt


def handle_key(key: str):
    ss = st.session_state

    if not ss.test_active or ss.finished:
        return

    current_node = NODES.get(ss.current_node_id)
    if not current_node:
        ss.last_action = "error"
        ss.last_message = "Nodo actual no encontrado en la configuración."
        return

    # '*': repetir mensaje del nodo actual (solo prompt, sin tono)
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
    ss.last_played_node_id = None  # nuevo nodo -> reproducir audio

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

    st.caption("⭐ '*' repite el mensaje del nodo actual · '#' vuelve al menú principal")


def main():
    init_session()
    ss = st.session_state

    st.title("📞 IVR Tester (simulador de IVR)")

    # Estado inicial
    if not ss.test_active:
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

    # ===== LÓGICA DE AUDIO INICIAL (ring + ROOT) =====
    if not ss.did_initial_ring:
        st.subheader("☎️ Llamando a la IVR...")
        st.caption("Escuchas el tono de llamada y, a continuación, el mensaje del menú principal.")
        play_ring_and_prompt(current_node)
        ss.did_initial_ring = True
        ss.last_played_node_id = current_node["NODE_ID"]
    else:
        st.subheader("📟 Llamada IVR (simulada)")
        # Reproducir prompt solo cuando cambiamos de nodo o repetimos
        if ss.last_played_node_id != current_node["NODE_ID"]:
            play_node_audio(current_node)
            ss.last_played_node_id = current_node["NODE_ID"]

    # Texto del mensaje del nodo
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


if __name__ == "__main__":
    main()

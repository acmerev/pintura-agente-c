"""
UI Streamlit del agente Pintumex.

Arranque (desde la raíz del proyecto):
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

import streamlit as st
import chromadb

from src.agent.graph import MODEL, run_agent

st.set_page_config(
    page_title="Pintumex · Agente",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo: alto contraste — texto oscuro sobre fondo claro
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Bebas+Neue&display=swap');

  :root {
    --ink: #111111;
    --paint: #c45c26;
    --paint-dark: #7a2f0a;
    --muted: #3d3832;
    --surface: #ffffff;
  }

  html, body, .stApp, [data-testid="stAppViewContainer"] {
    color: var(--ink) !important;
    font-family: "DM Sans", sans-serif;
  }

  .stApp {
    background:
      radial-gradient(ellipse 80% 50% at 10% -10%, #e8d5c4 0%, transparent 55%),
      radial-gradient(ellipse 60% 40% at 100% 0%, #d4e0d8 0%, transparent 50%),
      linear-gradient(180deg, #f3efe8 0%, #ebe4da 100%);
  }

  /* Texto principal: forzar contraste */
  .stApp p, .stApp span, .stApp label, .stApp li,
  .stApp .stMarkdown, .stApp [data-testid="stMarkdownContainer"],
  .stApp [data-testid="stMarkdownContainer"] p,
  .stApp [data-testid="stWidgetLabel"] p,
  .stApp [data-testid="stCaptionContainer"],
  .stApp [data-testid="stCaptionContainer"] p {
    color: var(--ink) !important;
  }

  h1, h2, h3, .brand {
    font-family: "Bebas Neue", sans-serif !important;
    letter-spacing: 0.04em;
    color: var(--ink) !important;
  }

  /* Chat: fondo sólido + texto negro */
  [data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #d4cfc6 !important;
    color: var(--ink) !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] span,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] div {
    color: var(--ink) !important;
  }

  /* Input del chat */
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] div {
    color: var(--ink) !important;
  }

  /* Sidebar oscura con texto claro legible */
  [data-testid="stSidebar"] {
    background: #1a1714 !important;
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] li,
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #f5f0e8 !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    background: var(--paint) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
  }
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] [data-baseweb="input"] input,
  [data-testid="stSidebar"] [data-baseweb="input"] {
    color: #111111 !important;
    background: #ffffff !important;
  }
  [data-testid="stSidebar"] [data-testid="stMetricValue"],
  [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #f5f0e8 !important;
  }

  .brand-wrap { margin-bottom: 0.5rem; }
  .brand {
    font-size: 3.2rem;
    line-height: 1;
    margin: 0;
    color: #111111 !important;
  }
  .tagline {
    color: #3d3832 !important;
    font-size: 1.05rem;
    margin: 0.25rem 0 1.25rem;
    font-weight: 500;
  }

  .meta-chip {
    display: inline-block;
    background: #f0e0d4;
    color: #7a2f0a !important;
    padding: 0.25rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.4rem;
    border: 1px solid #d4a888;
  }

  /* Alertas legibles */
  [data-testid="stAlert"] p {
    color: #111111 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _status() -> dict[str, bool]:
    data = ROOT / "data"
    chroma_dir = ROOT / "chroma_db"
    # Conteo real de vectores (no solo "carpeta existe")
    precios_count = 0
    fichas_count = 0
    if chroma_dir.exists():
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            try:
                precios_count = client.get_collection("precios").count()
            except Exception:  # noqa: BLE001
                precios_count = 0
            try:
                fichas_count = client.get_collection("fichas").count()
            except Exception:  # noqa: BLE001
                fichas_count = 0
        except Exception:  # noqa: BLE001
            precios_count = 0
            fichas_count = 0
    return {
        "precios_pdf": (data / "precios_actuales.pdf").exists(),
        "ficha_pdf": (data / "ficha_tecnica.pdf").exists(),
        "chroma_indexado": (precios_count > 0 and fichas_count > 0),
        "precios_count": precios_count,
        "fichas_count": fichas_count,
    }


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Estado")
    st.caption(f"Modelo: Cohere · `{MODEL}`")
    status = _status()
    st.write(f"{'✅' if status['precios_pdf'] else '❌'} precios_actuales.pdf")
    st.write(f"{'✅' if status['ficha_pdf'] else '❌'} ficha_tecnica.pdf")
    st.write(
        f"{'✅' if status['chroma_indexado'] else '❌'} Chroma indexado "
        f"(precios: {status['precios_count']}, fichas: {status['fichas_count']})"
    )
    if not status["chroma_indexado"]:
        st.warning("Ejecuta: `python -m src.rag.ingest`")

    st.divider()
    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main ---
st.markdown(
    """
<div class="brand-wrap">
  <p class="brand">PINTUMEX</p>
  <p class="tagline">Agente de precios, fichas técnicas y cálculo de pintura</p>
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route"):
            st.markdown(
                f'<span class="meta-chip">ruta: {msg["route"]}</span>',
                unsafe_allow_html=True,
            )
        meta = msg.get("meta") or {}
        chips = []
        if meta.get("litros_necesarios") is not None:
            chips.append(f"litros: {meta['litros_necesarios']}")
        if meta.get("costo_estimado") is not None:
            chips.append(f"costo: ${meta['costo_estimado']:,.2f}")
        if chips:
            st.markdown(
                " ".join(f'<span class="meta-chip">{c}</span>' for c in chips),
                unsafe_allow_html=True,
            )

prompt = st.chat_input("Pregunta por un producto, precio o cuánta pintura necesitas…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando agente…"):
            try:
                result = run_agent(prompt, thread_id="streamlit")
                reply = result.get("reply") or "(Sin respuesta)"
                st.markdown(reply)
                if result.get("route"):
                    st.markdown(
                        f'<span class="meta-chip">ruta: {result["route"]}</span>',
                        unsafe_allow_html=True,
                    )
                chips = []
                if result.get("litros_necesarios") is not None:
                    chips.append(f"litros: {result['litros_necesarios']}")
                if result.get("costo_estimado") is not None:
                    chips.append(f"costo: ${result['costo_estimado']:,.2f}")
                if chips:
                    st.markdown(
                        " ".join(
                            f'<span class="meta-chip">{c}</span>' for c in chips
                        ),
                        unsafe_allow_html=True,
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "route": result.get("route"),
                        "meta": {
                            "litros_necesarios": result.get("litros_necesarios"),
                            "costo_estimado": result.get("costo_estimado"),
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                err = f"Error al consultar el agente: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )

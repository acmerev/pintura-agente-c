"""
Grafo LangGraph: Router → nodos de herramienta → respuesta.

Flujo:
    START → router → (precios | fichas | calculo | general) → responder → END
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_cohere import ChatCohere
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState, Route
from src.agent.tools import calcular_litros, estimar_costo, get_retriever
from src.rag.retriever import PaintRetriever

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=True)

MODEL = os.getenv("COHERE_MODEL", "command-r-08-2024")
MAX_RETRIES = int(os.getenv("COHERE_MAX_RETRIES", "3"))


def _llm() -> ChatCohere:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta COHERE_API_KEY en .env. El agente ya no usa Gemini."
        )
    return ChatCohere(model=MODEL, temperature=0, cohere_api_key=api_key)


def _invoke_llm(messages: list[BaseMessage]) -> Any:
    """Invoca Cohere con reintentos ante 429 / cuota agotada."""
    llm = _llm()
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            text = str(exc)
            if "429" in text or "rate" in text.lower() or "quota" in text.lower():
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise
    assert last_err is not None
    raise last_err


def _last_user_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return str(msg.content)
    return ""

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_producto(text: str) -> str | None:
    """
    Heurística simple: intenta quedarnos con el nombre del producto.
    No busca ser perfecta; solo evita que el agente pida el dato si ya está.
    """
    t = text.strip()
    # Quita frases comunes
    t = re.sub(r"(?i)\b(precio|precios|costo|cuesta|valor|quiero|seria|serían|deseo)\b", " ", t)
    t = re.sub(r"(?i)\b(de|del|la|el|los|las|para)\b", " ", t)
    t = re.sub(r"(?i)\b(1\s*l(itro)?|litro|lata|gal(o|ón)|cubeta|bid[oó]n)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    return t if len(t) >= 4 else None


def _expand_price_query(query: str) -> str:
    """Amplía búsquedas: '1 litro' también busca 0.946 L / 1 L, etc."""
    q = query
    extra: list[str] = []
    if re.search(r"(?i)\b1\s*l(itro)?\b", q):
        extra += ["0.946 L", "1 L", "946 ml"]
    if re.search(r"(?i)\b4\s*l(itros)?\b", q):
        extra += ["3.785 L", "4 L"]
    if re.search(r"(?i)\b(19|18)\s*l(itros)?\b", q):
        extra += ["18 L", "19 L"]
    if extra:
        q = f"{q} {' '.join(extra)}"
    return q


def _best_snippet_for_price(ctx: str, query: str) -> str:
    """Ventana de líneas alrededor del nombre del producto en el contexto."""
    text = ctx.replace("\r", "")
    q = _norm(query)
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 3]
    if not words:
        return ctx

    lines = text.splitlines()
    scored: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        lnorm = _norm(line)
        score = sum(1 for w in words if w in lnorm)
        if score <= 0:
            continue
        if re.search(r"(?i)(dry|lux|esmal|vinil|primario|sellavin)", line):
            score += 2
        scored.append((score, i))

    if not scored:
        return ctx

    scored.sort(reverse=True)
    _, idx = scored[0]
    # Más contexto: precios suelen estar ANTES o DESPUÉS del nombre en PDFs tabulares
    start = max(0, idx - 8)
    end = min(len(lines), idx + 12)
    return "\n".join(lines[start:end]).strip() or ctx


def _extract_price_table_llm(producto: str, ctx: str, presentacion_pedida: str | None) -> str:
    """
    Pide al LLM que arme una tabla clara producto→presentación→precio
    a partir del texto desordenado del PDF.
    """
    system = """Extraes precios de listas de pintura Pintumex.
El texto del PDF suele venir desordenado (tablas rotas).
Reglas:
- Devuelve SOLO hechos que estén en el texto. No inventes.
- Relaciona el PRODUCTO pedido con sus PRESENTACIONES y PRECIOS ($).
- Equivalencias: 0.946 L ≈ 1 L; 3.785 L ≈ 4 L; 18 L ≈ 19 L.
- CUIDADO: no mezcles precios del producto ANTERIOR o SIGUIENTE en el PDF.
- Si aparece un código distinto (ej. 470SR) junto al producto, aclara si es otro ítem.
- Si el usuario pide una presentación, priorízala y di el precio con $.
- Si hay varias, lista: Presentación — $precio
- Si no puedes asociar con certeza, dilo y muestra 2-3 candidatos cercanos.
- NUNCA pidas m², rendimiento ni manos.
Responde en español, breve (máx 8 líneas)."""
    human = (
        f"Producto buscado: {producto or '(no claro)'}\n"
        f"Presentación pedida: {presentacion_pedida or '(cualquiera)'}\n\n"
        f"Texto del PDF:\n{ctx[:4500]}"
    )
    result = _invoke_llm(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    return str(result.content).strip()


def _detect_presentacion(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(0\.946|1)\s*l", t) or "litro" in t and "4" not in t:
        if re.search(r"\b1\s*l|\b0\.946|un litro|1 litro", t):
            return "1 L / 0.946 L"
    if re.search(r"\b(3\.785|4)\s*l", t):
        return "4 L / 3.785 L"
    if re.search(r"\b(18|19)\s*l", t):
        return "19 L / 18 L"
    if re.search(r"\b200\s*l", t):
        return "200 L"
    return None


def _extract_rendimiento_from_context(ctx: str) -> float | None:
    t = ctx.replace("\r", "")
    patterns = [
        r"(?i)rendimiento[^0-9]{0,20}(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\s*(?:/|por|x)\s*l",
        r"(?i)(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\s*(?:/|por|x)\s*l(?:itro)?",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def _extract_precio_from_structured(structured: str) -> float | None:
    """Toma un precio monetario del resumen ($183.00), no volúmenes (0.946 L)."""
    # Preferir montos con signo $
    for m in re.finditer(r"\$\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)", structured):
        alt = m.group(1).replace(",", "")
        try:
            val = float(alt)
            if val >= 10:  # evita capturar 0.946
                return val
        except ValueError:
            continue
    # Fallback: "precio ... 183"
    m = re.search(
        r"(?i)(?:precio|cuesta|vale)\D{0,20}(\d{2,3}(?:,\d{3})*(?:\.\d{2})?|\d{2,5}(?:\.\d{2})?)",
        structured,
    )
    if m:
        return float(m.group(1).replace(",", ""))
    return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """Eres un clasificador de intenciones para un agente de pintura (Pintumex).
Clasifica el mensaje del usuario en UNA sola categoría:

- precios: pregunta por precios, listas, costos unitarios, presentaciones comerciales
- fichas: pregunta por fichas técnicas, rendimiento (m²/L), usos, preparación, dilución
- calculo: quiere saber cuánta pintura comprar / litros / m² / manos / presupuesto de obra
- general: saludo, fuera de dominio, o consulta ambigua

Responde SOLO con JSON válido: {"route":"precios"|"fichas"|"calculo"|"general"}
"""


def router_node(state: AgentState) -> dict:
    text = _last_user_text(state)
    lowered = text.lower()

    # Heurística primero: evita que "precio de 1 litro" caiga en cálculo
    if any(w in lowered for w in ("precio", "precios", "cuesta", "lista de precios")):
        # Solo cálculo si además pide metros / cuánta pintura / presupuesto de obra
        if any(w in lowered for w in ("m2", "m²", "metro", "cuánta", "cuanta", "cuanto", "cuánto", "litros necesito", "presupuesto")):
            route: Route = "calculo"
        else:
            route = "precios"
            return {"route": route}

    result = _invoke_llm(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=text),
        ]
    )
    route = "general"
    raw = str(result.content).strip()
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            candidate = data.get("route", "general")
            if candidate in ("precios", "fichas", "calculo", "general"):
                route = candidate  # type: ignore[assignment]
    except json.JSONDecodeError:
        if any(w in lowered for w in ("precio", "cuesta", "lista", "$", "costo")):
            route = "precios"
        elif any(w in lowered for w in ("rendimiento", "ficha", "diluir", "aplicar")):
            route = "fichas"
        elif any(w in lowered for w in ("m2", "m²", "metro", "cuánta", "cuanta", "calcular")):
            route = "calculo"

    return {"route": route}


def route_after_router(
    state: AgentState,
) -> Literal["node_precios", "node_fichas", "node_calculo", "node_general"]:
    mapping = {
        "precios": "node_precios",
        "fichas": "node_fichas",
        "calculo": "node_calculo",
        "general": "node_general",
    }
    return mapping.get(state.get("route") or "general", "node_general")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Nodos de herramienta / recuperación
# ---------------------------------------------------------------------------

def node_precios(state: AgentState) -> dict:
    query = _last_user_text(state)
    search_q = _expand_price_query(query)
    docs = get_retriever().search_precios(search_q, k=8)
    ctx = PaintRetriever.format_docs(docs, label="precios")
    producto = _extract_producto(query)
    presentacion = _detect_presentacion(query)
    snippet = _best_snippet_for_price(ctx, query)
    structured = _extract_price_table_llm(producto or query, snippet or ctx, presentacion)
    return {
        "retrieved_context": (
            f"=== PRECIOS EXTRAÍDOS ===\n{structured}\n\n"
            f"=== TEXTO PDF (referencia) ===\n{snippet or ctx}"
        ),
        "producto": producto,
        "precio_unitario": _extract_precio_from_structured(structured),
    }


def node_fichas(state: AgentState) -> dict:
    query = _last_user_text(state)
    docs = get_retriever().search_fichas(query)
    ctx = PaintRetriever.format_docs(docs, label="fichas")
    return {"retrieved_context": ctx}


def _extract_numbers(text: str) -> dict[str, float | int | None]:
    """Heurística ligera para sacar m², rendimiento y manos del mensaje."""
    m2 = None
    rendimiento = None
    manos = None

    m2_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|metros?\s*cuadrados?)",
        text,
        re.IGNORECASE,
    )
    if m2_match:
        m2 = float(m2_match.group(1).replace(",", "."))

    rend_match = re.search(
        r"rendimiento[:\s]*(\d+(?:[.,]\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not rend_match:
        rend_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\s*/\s*l",
            text,
            re.IGNORECASE,
        )
    if not rend_match:
        # "10 m2 por litro" / "10 m2 x litro"
        rend_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\s*(?:por|x)\s*l(?:itro)?",
            text,
            re.IGNORECASE,
        )
    if rend_match:
        rendimiento = float(rend_match.group(1).replace(",", "."))

    manos_match = re.search(r"(\d+)\s*manos?", text, re.IGNORECASE)
    if manos_match:
        manos = int(manos_match.group(1))

    return {
        "metros_cuadrados": m2,
        "rendimiento_m2_por_litro": rendimiento,
        "manos": manos,
    }


def node_calculo(state: AgentState) -> dict:
    """Recupera ficha (rendimiento) + precios, y calcula si hay datos suficientes."""
    query = _last_user_text(state)
    search_q = _expand_price_query(query)
    both = get_retriever().search_both(search_q, k=6)
    ctx_fichas = PaintRetriever.format_docs(both["fichas"], label="fichas")
    ctx_precios = PaintRetriever.format_docs(both["precios"], label="precios")

    nums = _extract_numbers(query)
    producto = _extract_producto(query)
    presentacion = _detect_presentacion(query)
    rend_ctx = _extract_rendimiento_from_context(ctx_fichas)

    # Extrae precios del PDF (texto desordenado) con LLM
    snippet = _best_snippet_for_price(ctx_precios, query)
    structured = _extract_price_table_llm(producto or query, snippet or ctx_precios, presentacion)
    precio_ctx = _extract_precio_from_structured(structured)

    ctx = (
        f"=== PRECIOS EXTRAÍDOS ===\n{structured}\n\n"
        f"=== FICHAS ===\n{ctx_fichas}\n\n"
        f"=== TEXTO PRECIOS ===\n{snippet or ctx_precios}"
    )

    updates: dict = {
        "retrieved_context": ctx,
        "producto": producto,
        "metros_cuadrados": nums["metros_cuadrados"],
        "rendimiento_m2_por_litro": nums["rendimiento_m2_por_litro"] or rend_ctx,
        "manos": nums["manos"] or 2,
        "precio_unitario": precio_ctx,
    }

    m2 = nums["metros_cuadrados"]
    rend = nums["rendimiento_m2_por_litro"] or rend_ctx
    manos = int(nums["manos"] or 2)

    if m2 and rend:
        calc = calcular_litros(m2, rend, manos=manos)
        updates["litros_necesarios"] = calc["litros_sugeridos"]
        updates["bidones_sugeridos"] = calc["litros_sugeridos"]

        precio_match = re.search(
            r"\$?\s*(\d+(?:[.,]\d+)?)\s*(?:por\s*litro|/l(?:itro)?)",
            query,
            re.IGNORECASE,
        )
        precio: float | None = None
        if precio_match:
            precio = float(precio_match.group(1).replace(",", "."))
        elif precio_ctx is not None:
            precio = precio_ctx

        if precio is not None:
            cost = estimar_costo(calc["litros_sugeridos"], precio)
            updates["precio_unitario"] = precio
            updates["costo_estimado"] = cost["costo_por_litros"]
            updates["bidones_sugeridos"] = cost["envases_necesarios"]

    return updates


def node_general(state: AgentState) -> dict:
    return {"retrieved_context": None}


# ---------------------------------------------------------------------------
# Respuesta final
# ---------------------------------------------------------------------------

RESPONDER_SYSTEM = """Eres el asistente comercial-técnico de Pintumex.
Responde en español, claro, corto y útil.

Reglas estrictas:
- Usa SOLO el contexto (bloque PRECIOS EXTRAÍDOS / FICHAS) y variables del estado.
- No inventes precios ni rendimientos.
- Pregunta lo MÍNIMO. Un cliente normalmente NO sabe rendimiento técnico.

Por ruta:
- precios:
  * Contesta con el precio y presentaciones del bloque PRECIOS EXTRAÍDOS.
  * Si pidió "1 litro", usa 1 L o 0.946 L (equivalente).
  * NUNCA pidas m², rendimiento, manos ni "más datos" si ya hay precio.
  * Si faltara solo la presentación, pregunta UNA cosa: ¿1L, 4L o 19L?
- fichas: da rendimiento/uso desde el contexto.
- calculo:
  * Si hay m² + rendimiento (del usuario o ficha), calcula litros.
  * Manos por defecto = 2 si no las dijo.
  * Si pide costo y hay precio en PRECIOS EXTRAÍDOS, úsalo.
  * Solo pregunta el dato que falte (uno a la vez).
"""


def responder_node(state: AgentState) -> dict:
    user = _last_user_text(state)
    route = state.get("route") or "general"
    ctx = state.get("retrieved_context") or "(sin contexto)"

    calc_block = {
        "producto": state.get("producto"),
        "metros_cuadrados": state.get("metros_cuadrados"),
        "rendimiento_m2_por_litro": state.get("rendimiento_m2_por_litro"),
        "manos": state.get("manos"),
        "litros_necesarios": state.get("litros_necesarios"),
        "bidones_sugeridos": state.get("bidones_sugeridos"),
        "precio_unitario": state.get("precio_unitario"),
        "costo_estimado": state.get("costo_estimado"),
    }

    prompt = (
        f"Ruta detectada: {route}\n\n"
        f"Contexto recuperado:\n{ctx}\n\n"
        f"Variables de cálculo:\n{json.dumps(calc_block, ensure_ascii=False, indent=2)}\n\n"
        f"Pregunta del usuario:\n{user}\n\n"
        "Responde ya con la información disponible. No pidas datos innecesarios."
    )

    result = _invoke_llm(
        [
            SystemMessage(content=RESPONDER_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    return {"messages": [AIMessage(content=str(result.content))]}


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("node_precios", node_precios)
    graph.add_node("node_fichas", node_fichas)
    graph.add_node("node_calculo", node_calculo)
    graph.add_node("node_general", node_general)
    graph.add_node("responder", responder_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("node_precios", "responder")
    graph.add_edge("node_fichas", "responder")
    graph.add_edge("node_calculo", "responder")
    graph.add_edge("node_general", "responder")
    graph.add_edge("responder", END)

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def reset_agent() -> None:
    """Fuerza recompilar el grafo (útil tras cambios en caliente)."""
    global _agent
    _agent = None


def run_agent(message: str, *, thread_id: str = "default") -> dict:
    """Ejecuta una ronda del agente y devuelve respuesta + metadatos."""
    agent = get_agent()
    initial: AgentState = {
        "messages": [HumanMessage(content=message)],
        "route": None,
        "retrieved_context": None,
        "producto": None,
        "metros_cuadrados": None,
        "rendimiento_m2_por_litro": None,
        "manos": None,
        "litros_necesarios": None,
        "bidones_sugeridos": None,
        "precio_unitario": None,
        "costo_estimado": None,
    }
    final = agent.invoke(
        initial,
        config={"configurable": {"thread_id": thread_id}},
    )
    last = final["messages"][-1]
    return {
        "reply": last.content if hasattr(last, "content") else str(last),
        "route": final.get("route"),
        "metros_cuadrados": final.get("metros_cuadrados"),
        "rendimiento_m2_por_litro": final.get("rendimiento_m2_por_litro"),
        "manos": final.get("manos"),
        "litros_necesarios": final.get("litros_necesarios"),
        "bidones_sugeridos": final.get("bidones_sugeridos"),
        "precio_unitario": final.get("precio_unitario"),
        "costo_estimado": final.get("costo_estimado"),
    }

"""Estado compartido del agente LangGraph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


Route = Literal["precios", "fichas", "calculo", "general"]


class AgentState(TypedDict):
    """Mensajes del chat + variables de cálculo de pintura."""

    messages: Annotated[list[AnyMessage], add_messages]

    # Resultado del router
    route: Route | None

    # Contexto recuperado de Chroma (texto formateado)
    retrieved_context: str | None

    # Variables de cálculo (m² × rendimiento)
    producto: str | None
    metros_cuadrados: float | None
    rendimiento_m2_por_litro: float | None
    manos: int | None
    litros_necesarios: float | None
    bidones_sugeridos: float | None
    precio_unitario: float | None
    costo_estimado: float | None

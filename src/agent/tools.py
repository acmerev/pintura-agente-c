"""Herramientas de cálculo y búsqueda para el agente de pintura."""

from __future__ import annotations

import math
from typing import Any

from langchain_core.tools import tool

from src.rag.retriever import PaintRetriever

_retriever: PaintRetriever | None = None


def get_retriever() -> PaintRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PaintRetriever()
    return _retriever


def calcular_litros(
    metros_cuadrados: float,
    rendimiento_m2_por_litro: float,
    manos: int = 2,
    merma: float = 0.10,
) -> dict[str, float]:
    """
    Calcula litros necesarios.

    Fórmula base:
        litros = (m² × manos / rendimiento) × (1 + merma)

    - rendimiento_m2_por_litro: m² que cubre 1 litro por mano
    - merma: factor de desperdicio (default 10%)
    """
    if metros_cuadrados <= 0:
        raise ValueError("metros_cuadrados debe ser > 0")
    if rendimiento_m2_por_litro <= 0:
        raise ValueError("rendimiento_m2_por_litro debe ser > 0")
    if manos < 1:
        raise ValueError("manos debe ser >= 1")

    litros_base = (metros_cuadrados * manos) / rendimiento_m2_por_litro
    litros = litros_base * (1 + merma)
    # Redondeo hacia arriba a 0.5 L (presentaciones típicas)
    litros_redondeados = math.ceil(litros * 2) / 2

    return {
        "metros_cuadrados": metros_cuadrados,
        "rendimiento_m2_por_litro": rendimiento_m2_por_litro,
        "manos": float(manos),
        "merma": merma,
        "litros_exactos": round(litros, 3),
        "litros_sugeridos": litros_redondeados,
    }


def estimar_costo(
    litros: float,
    precio_por_litro: float,
    presentacion_litros: float = 19.0,
) -> dict[str, float]:
    """Estima costo y cantidad de envases a partir de litros y precio/L."""
    if litros <= 0 or precio_por_litro < 0:
        raise ValueError("litros y precio_por_litro deben ser válidos")
    if presentacion_litros <= 0:
        raise ValueError("presentacion_litros debe ser > 0")

    envases = math.ceil(litros / presentacion_litros)
    costo = litros * precio_por_litro
    costo_por_envases = envases * presentacion_litros * precio_por_litro

    return {
        "litros": litros,
        "precio_por_litro": precio_por_litro,
        "presentacion_litros": presentacion_litros,
        "envases_necesarios": float(envases),
        "costo_por_litros": round(costo, 2),
        "costo_por_envases": round(costo_por_envases, 2),
    }


@tool
def tool_calcular_pintura(
    metros_cuadrados: float,
    rendimiento_m2_por_litro: float,
    manos: int = 2,
    merma: float = 0.10,
) -> dict[str, Any]:
    """Calcula cuántos litros de pintura se necesitan según m², rendimiento y manos."""
    return calcular_litros(metros_cuadrados, rendimiento_m2_por_litro, manos, merma)


@tool
def tool_estimar_costo(
    litros: float,
    precio_por_litro: float,
    presentacion_litros: float = 19.0,
) -> dict[str, Any]:
    """Estima el costo total y número de envases dados litros y precio por litro."""
    return estimar_costo(litros, precio_por_litro, presentacion_litros)


@tool
def tool_buscar_precios(query: str) -> str:
    """Busca precios y presentaciones en la lista de precios actual."""
    docs = get_retriever().search_precios(query)
    return PaintRetriever.format_docs(docs, label="precios")


@tool
def tool_buscar_fichas(query: str) -> str:
    """Busca rendimiento, usos y datos técnicos en fichas técnicas."""
    docs = get_retriever().search_fichas(query)
    return PaintRetriever.format_docs(docs, label="fichas")


AGENT_TOOLS = [
    tool_calcular_pintura,
    tool_estimar_costo,
    tool_buscar_precios,
    tool_buscar_fichas,
]

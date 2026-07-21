"""
API FastAPI — punto de entrada del agente de pintura.

Arranque (desde la raíz del proyecto):
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    python -m src.main
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar este archivo directo desde el IDE (python src/main.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.graph import run_agent
from src.agent.tools import calcular_litros, estimar_costo

load_dotenv(ROOT / ".env")

app = FastAPI(
    title="Pintumex Paint Agent",
    description="Agente RAG + cálculo de pintura (precios / fichas / m²)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    route: str | None = None
    metros_cuadrados: float | None = None
    rendimiento_m2_por_litro: float | None = None
    manos: int | None = None
    litros_necesarios: float | None = None
    bidones_sugeridos: float | None = None
    precio_unitario: float | None = None
    costo_estimado: float | None = None


class CalcRequest(BaseModel):
    metros_cuadrados: float = Field(..., gt=0)
    rendimiento_m2_por_litro: float = Field(..., gt=0)
    manos: int = Field(2, ge=1)
    merma: float = Field(0.10, ge=0, le=0.5)
    precio_por_litro: float | None = Field(None, ge=0)
    presentacion_litros: float = Field(19.0, gt=0)


@app.get("/health")
def health():
    data_dir = Path(ROOT / "data")
    return {
        "status": "ok",
        "precios_pdf": (data_dir / "precios_actuales.pdf").exists(),
        "ficha_pdf": (data_dir / "ficha_tecnica.pdf").exists(),
        "chroma_db": (ROOT / "chroma_db").exists(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        result = run_agent(body.message, thread_id=body.thread_id)
        return ChatResponse(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/calcular")
def calcular(body: CalcRequest):
    """Endpoint directo de cálculo sin pasar por el LLM."""
    try:
        calc = calcular_litros(
            body.metros_cuadrados,
            body.rendimiento_m2_por_litro,
            manos=body.manos,
            merma=body.merma,
        )
        out: dict = {"calculo": calc}
        if body.precio_por_litro is not None:
            out["costo"] = estimar_costo(
                calc["litros_sugeridos"],
                body.precio_por_litro,
                presentacion_litros=body.presentacion_litros,
            )
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

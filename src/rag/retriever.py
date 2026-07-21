"""Retriever sobre las dos colecciones Chroma: precios y fichas técnicas."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", ROOT / "chroma_db"))
EMBEDDING_MODEL = os.getenv(
    "COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0"
)

CollectionName = Literal["precios", "fichas"]


class PaintRetriever:
    """Busca en una o ambas colecciones vectoriales."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        k: int = 4,
    ) -> None:
        self.persist_directory = Path(persist_directory or PERSIST_DIR)
        self.k = k
        self._embeddings = CohereEmbeddings(model=EMBEDDING_MODEL)
        self._stores: dict[str, Chroma] = {}

    def _store(self, collection: CollectionName) -> Chroma:
        if collection not in self._stores:
            self._stores[collection] = Chroma(
                collection_name=collection,
                embedding_function=self._embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._stores[collection]

    def search(
        self,
        query: str,
        collection: CollectionName,
        k: int | None = None,
    ) -> list[Document]:
        return self._store(collection).similarity_search(query, k=k or self.k)

    def search_precios(self, query: str, k: int | None = None) -> list[Document]:
        return self.search(query, "precios", k=k)

    def search_fichas(self, query: str, k: int | None = None) -> list[Document]:
        return self.search(query, "fichas", k=k)

    def search_both(
        self,
        query: str,
        k: int | None = None,
    ) -> dict[str, list[Document]]:
        """Consulta ambas colecciones y agrupa resultados por nombre."""
        return {
            "precios": self.search_precios(query, k=k),
            "fichas": self.search_fichas(query, k=k),
        }

    @staticmethod
    def format_docs(docs: list[Document], label: str = "") -> str:
        if not docs:
            return f"(Sin resultados{f' en {label}' if label else ''})"
        parts = []
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source_file", "?")
            page = doc.metadata.get("page", "?")
            header = f"[{label} #{i} | {src} p.{page}]" if label else f"[#{i}]"
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

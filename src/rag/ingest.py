"""
Ingesta de PDFs hacia ChromaDB con dos colecciones independientes.

Uso:
    python -m src.rag.ingest
    python -m src.rag.ingest --only precios
    python -m src.rag.ingest --only fichas
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", ROOT / "chroma_db"))
EMBEDDING_MODEL = os.getenv(
    "COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0"
)

# precios: cambio frecuente → chunks pequeños (SKUs, precios unitarios)
# fichas: cambio lento → chunks un poco más grandes (rendimiento, usos)
COLLECTIONS = {
    "precios": {
        "file": DATA_DIR / "precios_actuales.pdf",
        "chunk_size": 500,
        "chunk_overlap": 80,
    },
    "fichas": {
        "file": DATA_DIR / "ficha_tecnica.pdf",
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
}


def get_embeddings() -> CohereEmbeddings:
    return CohereEmbeddings(model=EMBEDDING_MODEL)


def process_documents(
    file_path: Path | str,
    collection_name: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    persist_directory: Path | str | None = None,
) -> Chroma:
    """Carga un PDF, lo trocea y lo indexa en una colección Chroma."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Coloca el PDF en data/ antes de ingerir."
        )

    persist = Path(persist_directory or PERSIST_DIR)
    persist.mkdir(parents=True, exist_ok=True)

    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["collection"] = collection_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(persist),
    )
    print(
        f"[ingest] {collection_name}: {len(splits)} chunks desde {path.name} -> {persist}"
    )
    return vectorstore


def ingest_all(only: str | None = None) -> None:
    targets = COLLECTIONS if only is None else {only: COLLECTIONS[only]}
    for name, cfg in targets.items():
        process_documents(
            cfg["file"],
            name,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestar PDFs a ChromaDB")
    parser.add_argument(
        "--only",
        choices=list(COLLECTIONS.keys()),
        help="Ingestar solo una colección (útil al reemplazar precios)",
    )
    args = parser.parse_args()
    ingest_all(only=args.only)


if __name__ == "__main__":
    main()

"""Embeds each asset's inspection_note into a Chroma vector store, so the
agent can answer fuzzy/descriptive questions (e.g. "reports mentioning
corrosion") via semantic search instead of exact SQL filters.

Run after db/build_dataset.py: python db/build_notes_index.py
"""
import sys
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

from agent import config  # noqa: E402


def load_asset_rows() -> list[dict]:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    con.execute("LOAD spatial;")
    cursor = con.execute(
        "SELECT id, asset_type, lon, lat, condition_score, last_inspection_date, inspection_note FROM assets"
    )
    columns = [c[0] for c in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    con.close()
    return rows


BATCH_SIZE = 50  # Ollama's embed endpoint is unreliable with very large single batches


def build_notes_index() -> int:
    rows = load_asset_rows()
    docs = [
        Document(
            page_content=r["inspection_note"],
            metadata={
                "asset_id": r["id"],
                "asset_type": r["asset_type"],
                "lon": r["lon"],
                "lat": r["lat"],
                "condition_score": r["condition_score"],
                "last_inspection_date": str(r["last_inspection_date"]),
            },
        )
        for r in rows
    ]

    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        embedding_function=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    store.delete_collection()
    store = Chroma(
        embedding_function=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        store.add_documents(batch)
        print(f"  embedded {min(start + BATCH_SIZE, len(docs))}/{len(docs)}")

    return len(docs)


if __name__ == "__main__":
    count = build_notes_index()
    print(f"Indexed {count} inspection notes into {config.CHROMA_DIR}")

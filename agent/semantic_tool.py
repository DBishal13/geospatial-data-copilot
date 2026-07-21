"""Semantic-search tool (phase 3): finds inspection notes similar in meaning
to a free-text description, using a Chroma index of note embeddings built by
db/build_notes_index.py — for fuzzy questions structured SQL can't express."""
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from agent import config


def _load_vectorstore() -> Chroma:
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No inspection-notes index found at '{config.CHROMA_DIR}'. "
            "Run `python db/build_notes_index.py` first."
        )
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL, keep_alive=config.OLLAMA_KEEP_ALIVE
    )
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def make_semantic_tool(shared_state: dict):
    @tool
    def search_inspection_notes(description: str, k: int = 8) -> str:
        """Finds assets whose inspection notes are semantically similar to a
        free-text description (e.g. "corrosion", "leaning pole", "vegetation
        encroachment"). Use this for fuzzy/descriptive questions about
        inspection findings that a structured SQL filter can't express."""
        try:
            store = _load_vectorstore()
        except FileNotFoundError as e:
            return str(e)

        matches = store.similarity_search(description, k=k)
        rows = [
            {
                "id": m.metadata["asset_id"],
                "asset_type": m.metadata["asset_type"],
                "lon": m.metadata["lon"],
                "lat": m.metadata["lat"],
                "condition_score": m.metadata["condition_score"],
                "inspection_note": m.page_content,
            }
            for m in matches
        ]
        shared_state["sql"] = None
        shared_state["rows"] = rows

        if not rows:
            return "No inspection notes matched that description."
        lines = "\n".join(f"- asset {r['id']} ({r['asset_type']}): {r['inspection_note']}" for r in rows)
        return f"Found {len(rows)} matching inspection note(s):\n{lines}"

    return search_inspection_notes

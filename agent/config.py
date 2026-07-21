"""Environment-driven settings. Every value has a default so the project
runs out of the box without a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# How long Ollama keeps a model resident in RAM after use. Without this,
# Ollama's default (5 minutes idle) evicts the model between requests during
# normal demo/dev usage, and a cold reload costs ~20s+ on CPU vs ~1s warm —
# by far the biggest latency lever on hardware with no GPU.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

DUCKDB_PATH = ROOT_DIR / os.getenv("DUCKDB_PATH", "data/geocopilot.duckdb")
CHROMA_DIR = ROOT_DIR / os.getenv("CHROMA_DIR", "data/chroma")
CHROMA_COLLECTION = "inspection_notes"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

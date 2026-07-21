"""FastAPI backend exposing a single /query endpoint that runs the
tool-calling geospatial agent and returns its answer, the SQL it ran, the
result rows, and an optional rendered map.

Run from the project root: uvicorn api.main:app --reload
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent.agent import ask  # noqa: E402
from agent.db import run_select  # noqa: E402
from agent.map_tool import build_map_html  # noqa: E402
from agent.semantic_tool import make_semantic_tool  # noqa: E402

app = FastAPI(title="Geospatial Data Copilot API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sql: str | None = None
    rows: list[dict] = []
    map_html: str | None = None


class OverviewResponse(BaseModel):
    map_html: str | None = None
    count: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/overview-map", response_model=OverviewResponse)
def overview_map():
    """Fast, agent-free default map of the whole seeded dataset — no LLM
    call, so it's near-instant. Used to populate the frontend's map canvas
    before the user has asked anything."""
    result = run_select("SELECT id, asset_type, lon, lat, condition_score FROM assets", row_limit=2000)
    map_html = build_map_html(result["rows"], cluster=True)
    return OverviewResponse(map_html=map_html, count=result["total_count"])


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = ask(req.question, extra_tools=[make_semantic_tool])
    return QueryResponse(
        answer=result["answer"],
        sql=result.get("sql"),
        rows=result.get("rows", []),
        map_html=result.get("map_html"),
    )

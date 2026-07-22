# Geospatial Data Copilot

A natural-language interface over spatial/infrastructure data. Ask a question in
plain English about San Francisco's utility infrastructure (poles, transmission
towers, transformers) — a local LLM agent translates it into a spatial SQL
query, runs it, and answers with a natural-language summary and an optional
interactive map.

Runs entirely locally via [Ollama](https://ollama.com) — no API keys, no
usage cost.

## Why this project

Most LLM portfolio projects are generic "chat with your PDF" RAG demos. This
one applies the same agent/tool-calling techniques to spatial and
infrastructure data instead — closer to real GIS/utility data work than a
tutorial clone.

## Tech stack note: DuckDB instead of PostgreSQL + PostGIS

The original design called for PostgreSQL + PostGIS. On this Windows dev
machine, PostGIS has no admin-free, Docker-free install path (conda-forge
doesn't ship a `postgis` build for `win-64`; the only real options were
Docker Desktop or an elevated native installer). Rather than misrepresent the
stack, this project uses **DuckDB + its `spatial` extension** instead:

- Zero-install: `pip install duckdb`, no server, no admin rights, no Docker.
- The SQL surface is deliberately PostGIS-like: `GEOMETRY` columns,
  `ST_Point`, `ST_Distance_Sphere`, `ST_X`/`ST_Y` — the agent's generated SQL
  reads almost identically to what it would generate against real PostGIS.
- Swapping back to PostgreSQL + PostGIS later mainly means changing the
  connection layer in `agent/db.py` and `db/build_dataset.py` — the schema,
  prompts, and tool logic would need only minor changes.

## Core architecture

```
User question (natural language)
        |
LangGraph ReAct agent (Ollama llama3.1:8b) with tools:
  - query_infrastructure_db: NL -> DuckDB SQL -> rows (structured filters/counts)
  - render_map:               NL -> DuckDB SQL -> rows -> Folium map (self-contained)
  - search_inspection_notes:  NL -> Chroma vector search over inspection notes
        |
FastAPI: POST /query (agent) + GET /overview-map (fast, agent-free)
        |
Streamlit UI: sidebar chat drives a persistent map canvas in the main area
```

The frontend is a **canvas + chat** layout, not a linear chat log: the map
lives in the main area and only updates when a query actually returns
spatial results, while the conversation scrolls independently in the
sidebar. On first load (before any question is asked) the canvas shows a
fast overview map of the whole seeded dataset via `GET /overview-map` —
a direct DuckDB query + Folium render with no LLM call involved, so the
canvas is never empty and never blocked on model latency.

Each tool independently runs its own NL-to-SQL pipeline rather than reading
another tool's cached output. This is a deliberate fix for a real bug found
during development: LangGraph executes multiple tool calls from the same
agent turn **concurrently** via a thread pool, so a tool that depended on a
sibling tool's side effect could read stale/empty state. Making every tool
self-contained eliminates that race entirely (see `agent/map_tool.py`).

There's no separate "summary" tool — the top-level agent's own final message,
grounded in the tool outputs it already saw, serves as the natural-language
answer.

## Dataset

Real asset locations come from OpenStreetMap (`power=pole|tower|transformer`
nodes for downtown/Mission/SoMa San Francisco, ~1,500 assets, fetched live via
the Overpass API in `db/build_dataset.py`). OSM has no inspection or
condition-tracking data, so those fields are synthesized on top of the real
coordinates: install date, last inspection date, a 0–100 condition/damage
score (100 = critical), a same-scale score from ~1 year ago (for trend
queries), and a free-text inspection note. **Do not treat the non-geometric
fields as real utility data** — only the asset locations are real.

## Repo structure

```
db/        schema.sql, build_dataset.py (OSM fetch + synthesis), build_notes_index.py (embeddings)
agent/     config.py, db.py (safe SQL exec), sql_tool.py, map_tool.py, semantic_tool.py, agent.py
api/       FastAPI app exposing POST /query
frontend/  Streamlit chat + map UI
.env.example
```

## Setup

1. **Install [Ollama](https://ollama.com/download)**, then pull the models:
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
2. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Build the dataset** (fetches live OSM data, ~1,500 assets):
   ```
   python db/build_dataset.py
   ```
4. **Build the semantic search index** (embeds inspection notes into Chroma):
   ```
   python db/build_notes_index.py
   ```
5. **Run the backend and frontend** (two terminals, from the project root):
   ```
   uvicorn api.main:app --port 8000
   streamlit run frontend/app.py
   ```

The DuckDB file is a single writer at a time — stop the API before re-running
`db/build_dataset.py`.

## Example queries (verified working)

**"Show me all poles inspected in the last 6 months with a high damage score."**
Generates a filtered `SELECT ... WHERE asset_type = 'utility_pole' AND
last_inspection_date >= current_date - INTERVAL 6 MONTH AND condition_score
>= 70`. On the current seeded data this genuinely returns 0 rows (verified
directly against the database) — the agent correctly reports "no poles
match" rather than fabricating results.

**"How many assets are within 2 miles of San Francisco City Hall?"**
Resolves the landmark to its listed coordinates and uses
`ST_Distance_Sphere(geom, ST_Point(lon, lat)) <= meters`. Answers with the
*true* total count even when the row preview is capped (a real bug fixed
during development — see below).

**"Which asset type has the highest average condition score decline this
year?"** Uses the synthesized `condition_score_last_year` column:
`avg(condition_score - condition_score_last_year) GROUP BY asset_type`.

**"Show me the 10 poles in the worst condition on a map."**
Remember the scale is a *damage* score (higher = worse), so this is `ORDER
BY condition_score DESC LIMIT 10` — renders an interactive Folium map with
color-coded markers (green/orange/red by condition score) and popups showing
each asset's details.

**"Find inspection notes similar to reports mentioning corrosion."**
Semantic search (phase 3) over inspection notes embedded with
`nomic-embed-text`, using Chroma's vector index — a purely keyword filter
would miss paraphrases like "heavy rust and metal thinning."

## Bugs found and fixed during development

Documented here because they're the more interesting part of building an
agent against a *local* model — this is not a curated success story:

1. **Tool argument confusion**: the top-level agent occasionally tried to
   write SQL itself and pass it as the tool's `question` argument instead of
   plain English. Fixed with an explicit Pydantic `args_schema`, a runtime
   guard that rejects SQL-looking input, and reinforced prompt language.
2. **Concurrent tool-call race**: `render_map` and `query_infrastructure_db`
   called in the same turn run in parallel threads under LangGraph's
   `ToolNode`; `render_map` could read empty shared state before the SQL
   tool finished. Fixed by making every tool fetch its own data instead of
   depending on a sibling's side effect.
3. **Semantic search returning irrelevant results**: with only ~18 note
   templates spread across ~1,500 assets, huge blocks of exact-duplicate
   embedding vectors broke Chroma's approximate HNSW index. Fixed by adding
   randomized detail suffixes to notes (216 effective combinations) and by
   batching embedding calls (a single 1,485-document embed request was also
   unreliable against local Ollama).
4. **Silent count truncation**: `agent/db.py` caps result previews at 200
   rows for context-window safety, but the agent was reporting that capped
   row count as if it were the true total. Fixed by always computing an
   exact `total_count` via a wrapping `COUNT(*)` subquery, independent of
   the preview cap.
5. **Ordering semantics**: "worst condition" was initially sorted ascending
   because a smaller-model default association (low number = bad, like a
   grade) conflicts with this schema's damage-score convention (high number
   = bad). Fixed with an explicit rule in the SQL-generation prompt.

## Definition of done

- [x] Local (DuckDB spatial, see tech-stack note) database seeded with real
      OSM asset geometry + synthetic asset-management data
- [x] Agent correctly answers all four example queries via generated SQL
- [x] Map renders for spatial results
- [x] Semantic (vector-search) query works end to end
- [x] README documents setup, architecture, dataset provenance, and example
      queries

## GitHub Pages demo

A static landing page (project overview, example queries, and a live overview
map of the seeded dataset) is built and published to GitHub Pages on pushes to
`main` via a GitHub Actions workflow (`.github/workflows/deploy-pages.yml`).
The CI job generates `docs/index.html` and `docs/overview_map.html` and
publishes them via the native `actions/upload-pages-artifact` +
`actions/deploy-pages` flow (repository Pages source: **GitHub Actions**).

This static page is a demo/landing page only — GitHub Pages can't run the
actual FastAPI + Streamlit + Ollama app, so the interactive chat isn't hosted
there. See "Run it yourself" on the page (or the Setup section above) to try
the real thing.

## Release process

Work happens on `dev`; changes land on `main` via pull request (which is what
triggers the Pages deploy). Tagged releases use semantic versioning
(`vMAJOR.MINOR.PATCH`), each with GitHub Release notes summarizing what
changed. See [Releases](https://github.com/DBishal13/geospatial-data-copilot/releases)
for the version history.

Locally, you can reproduce the built site with:

```
pip install -r requirements.txt
python scripts/build_static_site.py
# Open docs/index.html in your browser
```

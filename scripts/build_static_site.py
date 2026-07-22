#!/usr/bin/env python3
"""Builds a small static site (docs/) with an overview map HTML for GitHub Pages.

Runs the dataset build, queries the DB for overview rows, renders a folium
map via the existing `agent.map_tool.build_map_html`, and writes the result
to `docs/overview_map.html` with a simple `index.html` that embeds it.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db import build_dataset as build_dataset_module
from agent.db import run_select
from agent.map_tool import build_map_html


def main():
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # Build dataset (fetch OSM data and populate DuckDB)
    print("Building demo dataset (this may take a minute)...")
    build_dataset_module.main()

    print("Querying dataset for overview rows...")
    result = run_select("SELECT id, asset_type, lon, lat, condition_score FROM assets", row_limit=2000)
    map_html = build_map_html(result["rows"], cluster=True)

    if map_html is None:
        raise SystemExit("No plottable points found; aborting static site build.")

    overview_path = docs / "overview_map.html"
    print(f"Writing overview map to {overview_path} ...")
    overview_path.write_text(map_html, encoding="utf-8")

    repo_url = "https://github.com/DBishal13/geospatial-data-copilot"
    index_html = INDEX_TEMPLATE.replace("__REPO_URL__", repo_url)
    (docs / "index.html").write_text(index_html, encoding="utf-8")
    print("Static site build complete. docs/index.html and overview_map.html created.")


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Geospatial Data Copilot</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.55;
    color: #1b1f24;
    background: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    body { color: #e6e8eb; background: #14171a; }
    .card, .qa { background: #1c2024; border-color: #2c3238; }
    .tag { background: #2c3238; color: #cfd6dd; }
    a { color: #7ab7ff; }
    .btn { background: #3b82f6; }
  }
  a { color: #2563eb; }
  main { max-width: 880px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  header.hero { margin-bottom: 2rem; }
  h1 { font-size: 1.85rem; margin: 0 0 0.4rem; }
  .tagline { font-size: 1.05rem; opacity: 0.85; margin: 0 0 1rem; }
  .tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.25rem; }
  .tag { background: #eef1f4; color: #43494f; border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; }
  .btn { display: inline-block; background: #2563eb; color: white !important; text-decoration: none;
         padding: 0.55rem 1.1rem; border-radius: 8px; font-weight: 600; font-size: 0.95rem; }
  section { margin: 2.25rem 0; }
  h2 { font-size: 1.2rem; border-bottom: 1px solid rgba(127,127,127,0.25); padding-bottom: 0.4rem; }
  .card {
    border: 1px solid rgba(127,127,127,0.25); border-radius: 10px;
    overflow: hidden; background: #fafbfc;
  }
  .card iframe { display: block; width: 100%; height: 480px; border: 0; }
  .card figcaption { font-size: 0.85rem; opacity: 0.75; padding: 0.6rem 0.9rem; margin: 0; }
  .qa { border: 1px solid rgba(127,127,127,0.25); border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 0.9rem; background: #fafbfc; }
  .qa .q { font-weight: 600; margin-bottom: 0.35rem; }
  .qa .q::before { content: "Q: "; opacity: 0.5; }
  .qa .a::before { content: "→ "; opacity: 0.5; }
  .qa .a { opacity: 0.85; font-size: 0.95rem; }
  ol, ul { padding-left: 1.3rem; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88rem; }
  pre { background: rgba(127,127,127,0.12); padding: 0.75rem 1rem; border-radius: 8px; overflow-x: auto; }
  footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid rgba(127,127,127,0.25); font-size: 0.85rem; opacity: 0.7; }
</style>
</head>
<body>
<main>
  <header class="hero">
    <h1>Geospatial Data Copilot</h1>
    <p class="tagline">A natural-language interface over spatial/infrastructure data — ask a question in
    plain English about San Francisco's utility infrastructure and get back a SQL-grounded
    answer, an interactive map, or a semantic search result.</p>
    <div class="tags">
      <span class="tag">LangGraph ReAct agent</span>
      <span class="tag">Ollama (local LLM)</span>
      <span class="tag">DuckDB + spatial extension</span>
      <span class="tag">FastAPI</span>
      <span class="tag">Streamlit</span>
      <span class="tag">Chroma vector search</span>
    </div>
    <a class="btn" href="__REPO_URL__">View the code on GitHub</a>
  </header>

  <section>
    <h2>Live demo map</h2>
    <p>This is a real render of the seeded dataset (~1,500 power assets — poles, transmission
    towers, transformers — fetched live from OpenStreetMap for downtown/Mission/SoMa San
    Francisco). This overview loads instantly with no LLM call involved; the full app renders
    maps like this on demand in response to a chat question.</p>
    <figure class="card">
      <iframe src="./overview_map.html" title="Overview map of seeded assets" loading="lazy"></iframe>
      <figcaption>Color-coded by condition/damage score. Click a marker for asset details.</figcaption>
    </figure>
  </section>

  <section>
    <h2>What you can ask it</h2>
    <p>The full chat agent runs locally against Ollama and isn't hosted here (GitHub Pages only
    serves static files — see "Run it yourself" below). These are real, verified queries from
    development against the seeded dataset:</p>

    <div class="qa">
      <div class="q">Show me all poles inspected in the last 6 months with a high damage score.</div>
      <div class="a">Generates a filtered SQL query; on the seeded data this genuinely returns
      zero rows, and the agent correctly reports "no poles match" instead of fabricating results.</div>
    </div>
    <div class="qa">
      <div class="q">How many assets are within 2 miles of San Francisco City Hall?</div>
      <div class="a">Resolves the landmark to coordinates and uses a spatial distance query
      (<code>ST_Distance_Sphere</code>), answering with the true total count even when the row
      preview is capped for context-window safety.</div>
    </div>
    <div class="qa">
      <div class="q">Which asset type has the highest average condition score decline this year?</div>
      <div class="a">Aggregates <code>condition_score - condition_score_last_year</code> grouped
      by asset type.</div>
    </div>
    <div class="qa">
      <div class="q">Show me the 10 poles in the worst condition on a map.</div>
      <div class="a">Renders an interactive Folium map with color-coded markers and popups —
      like the live demo map above, but scoped to a chat question.</div>
    </div>
    <div class="qa">
      <div class="q">Find inspection notes similar to reports mentioning corrosion.</div>
      <div class="a">Semantic search over inspection notes embedded with <code>nomic-embed-text</code>,
      via Chroma's vector index — catches paraphrases like "heavy rust and metal thinning" that a
      keyword filter would miss.</div>
    </div>
  </section>

  <section>
    <h2>Run it yourself</h2>
    <p>The interactive chat + map UI runs entirely locally (no API keys, no usage cost) via
    <a href="https://ollama.com">Ollama</a>. To try it:</p>
    <pre>git clone __REPO_URL__
cd geospatial-data-copilot
pip install -r requirements.txt

ollama pull llama3.1:8b
ollama pull nomic-embed-text

python db/build_dataset.py
python db/build_notes_index.py

uvicorn api.main:app --port 8000
streamlit run frontend/app.py</pre>
    <p>Full setup details, architecture notes, and known-bugs-fixed writeup are in the
    <a href="__REPO_URL__#readme">README</a>.</p>
  </section>

  <footer>
    Built as a portfolio project applying agent/tool-calling techniques to spatial and
    infrastructure data. <a href="__REPO_URL__">Source on GitHub</a>.
  </footer>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()

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

    index_html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Geospatial Data Copilot — Demo Map</title>
      <style>body{{margin:0;padding:0}} iframe{{border:0;width:100vw;height:100vh}}</style>
    </head>
    <body>
      <iframe src="./overview_map.html" title="Overview map" />
    </body>
    </html>
    """

    (docs / "index.html").write_text(index_html, encoding="utf-8")
    print("Static site build complete. docs/index.html and overview_map.html created.")


if __name__ == "__main__":
    main()

"""Map-rendering tool: renders spatial results as an interactive Folium map,
returned as an HTML string so it can cross the FastAPI -> Streamlit boundary
as JSON.

Self-contained by design: it takes its own plain-English question and runs
the NL-to-SQL pipeline itself rather than reading another tool's cached
output — LangGraph executes same-turn tool calls concurrently via a thread
pool, so a tool that depended on a sibling tool's side effect could read it
before that sibling has run."""
import folium
from folium.plugins import MarkerCluster
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent.sql_tool import generate_and_run_sql, is_probably_sql

SCORE_COLORS = [(30, "green"), (65, "orange"), (101, "red")]


def _color_for_score(score) -> str:
    if score is None:
        return "blue"
    for threshold, color in SCORE_COLORS:
        if score < threshold:
            return color
    return "red"


def build_map_html(points: list[dict], cluster: bool = False) -> str | None:
    """Renders a list of {lat, lon, ...} rows as a Folium (Leaflet) map and
    returns the HTML, or None if there are no plottable points. Shared by
    the render_map tool and the fast, agent-free overview map."""
    points = [p for p in points if p.get("lon") is not None and p.get("lat") is not None]
    if not points:
        return None

    avg_lat = sum(p["lat"] for p in points) / len(points)
    avg_lon = sum(p["lon"] for p in points) / len(points)
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles="OpenStreetMap")
    target = MarkerCluster(disableClusteringAtZoom=17).add_to(fmap) if cluster else fmap

    for p in points:
        score = p.get("condition_score")
        popup_lines = [f"{k}: {v}" for k, v in p.items() if k != "geom"]
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=6,
            color=_color_for_score(score),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup("<br>".join(popup_lines), max_width=300),
        ).add_to(target)

    return fmap.get_root().render()


class RenderMapInput(BaseModel):
    natural_language_question: str = Field(
        description=(
            "The user's question restated in plain English, focused on what to "
            "map — for example 'the 10 utility poles with the worst condition "
            "scores'. This must be plain English, NEVER SQL; this tool writes "
            "its own SQL to fetch the locations to plot."
        )
    )


def make_map_tool(shared_state: dict, llm: ChatOllama):
    @tool(args_schema=RenderMapInput)
    def render_map(natural_language_question: str) -> str:
        """Renders a map of the assets relevant to the user's question, if they
        have lon/lat coordinates. Call this whenever the question is about
        locations, distances, or asks to "show"/"map" results — pass the same
        question (or the relevant part of it) as plain English. This tool is
        self-contained: it fetches its own data via SQL, so call it in ITS OWN
        separate turn, never bundled together with query_infrastructure_db in
        one turn."""
        if is_probably_sql(natural_language_question):
            return (
                "Error: this tool takes a plain-English question, not SQL. "
                "Re-call it with the user's question restated in English."
            )

        sql, result = generate_and_run_sql(llm, natural_language_question)
        rows, total_count = result["rows"], result["total_count"]
        shared_state["sql"] = sql
        shared_state["rows"] = rows

        if total_count == 0:
            return f"SQL executed:\n{sql}\n\nResult: 0 rows. No matching assets were found — nothing to map."

        map_html = build_map_html(rows)
        if map_html is None:
            return f"SQL executed:\n{sql}\n\nResult: {total_count} row(s), but no lon/lat columns — nothing to map."

        shared_state["map_html"] = map_html
        points = [r for r in rows if r.get("lon") is not None and r.get("lat") is not None]
        preview = points[:15]
        note = (
            f" (map + preview show {len(points)} of {total_count} total — use total_count "
            "when reporting a count, not the number plotted)"
            if result["truncated"]
            else ""
        )
        return f"SQL executed:\n{sql}\n\nResult: {total_count} row(s) total. Map rendered.{note}\n{preview}"

    return render_map

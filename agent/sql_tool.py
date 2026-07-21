"""Text-to-SQL tool: turns a natural-language question into a DuckDB SQL
query (using a dedicated, schema-grounded LLM prompt), executes it, and
returns a compact text summary the agent can reason over."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent import config
from agent.db import SCHEMA_DESCRIPTION, run_select

NL2SQL_SYSTEM = f"""You are an expert DuckDB SQL analyst for a utility infrastructure database.

{SCHEMA_DESCRIPTION}

Rules:
- Write exactly one DuckDB SELECT statement that answers the question.
- Never select the `geom` column directly — select `lon` and `lat` instead.
- For "within N miles/meters of <place>" questions, convert miles to meters
  (1 mile = 1609.34 m) and use: ST_Distance_Sphere(geom, ST_Point(lon, lat)) <= meters
- If the place is one of the known landmarks listed above, you MUST use its
  exact listed coordinates — do not substitute your own general knowledge of
  where that place is.
- If the question mentions a map, showing results, or locations (even
  alongside a count), select individual rows including id, lon, lat rather
  than collapsing to COUNT(*) — a count can always be read off the number of
  rows returned, but a map needs per-row coordinates.
- "condition score" / "damage score" with no other qualifier always means
  the CURRENT `condition_score` column. Only use `condition_score_last_year`
  when the question explicitly says "last year", "a year ago", "previously",
  or asks about decline/trend/change over time.
- "high" damage/condition score means condition_score >= 70. "Low" means
  condition_score < 30.
- Remember the scale is a DAMAGE score, not a quality grade: higher number =
  worse condition. "Worst condition" / "most damaged" means ORDER BY
  condition_score DESC. "Best condition" means ORDER BY condition_score ASC.
- Respond with ONLY the SQL query. No explanation, no markdown code fences.

Examples:

Q: Show me all poles inspected in the last 6 months with a high damage score.
SQL: SELECT id, asset_type, lon, lat, condition_score, last_inspection_date FROM assets WHERE asset_type = 'utility_pole' AND last_inspection_date >= current_date - INTERVAL 6 MONTH AND condition_score >= 70 ORDER BY condition_score DESC

Q: How many assets are within 2 miles of San Francisco City Hall?
SQL: SELECT count(*) AS asset_count FROM assets WHERE ST_Distance_Sphere(geom, ST_Point(-122.4193, 37.7793)) <= 2 * 1609.34

Q: How many assets are within 2 miles of San Francisco City Hall? Show me a map.
SQL: SELECT id, asset_type, lon, lat, condition_score FROM assets WHERE ST_Distance_Sphere(geom, ST_Point(-122.4193, 37.7793)) <= 2 * 1609.34

Q: Which asset type has the highest average condition score decline this year?
SQL: SELECT asset_type, avg(condition_score - condition_score_last_year) AS avg_decline FROM assets GROUP BY asset_type ORDER BY avg_decline DESC

Q: Show me the 10 poles in the worst condition.
SQL: SELECT id, asset_type, lon, lat, condition_score, inspection_note FROM assets ORDER BY condition_score DESC LIMIT 10
"""


def _clean_sql(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text.strip().rstrip(";").strip()


class QueryInfrastructureInput(BaseModel):
    natural_language_question: str = Field(
        description=(
            "The user's question, restated in plain English — for example "
            "'how many utility poles have a condition score above 70'. "
            "This must be plain English, NEVER SQL. Do not write any SQL "
            "yourself; this tool writes the SQL for you from your English "
            "question."
        )
    )


def is_probably_sql(text: str) -> bool:
    return bool(re.match(r"(?is)^\s*(select|with)\b", text))


def generate_and_run_sql(llm: ChatOllama, natural_language_question: str) -> tuple[str, dict]:
    """Shared NL-to-SQL pipeline used by both the SQL tool and the map tool,
    so each tool is self-contained and safe to run independently — LangGraph
    executes same-turn tool calls concurrently via a thread pool, so tools
    must never depend on another tool's side effects to be correct."""
    sql = _clean_sql(llm.invoke([SystemMessage(NL2SQL_SYSTEM), HumanMessage(natural_language_question)]).content)
    result = run_select(sql)
    return sql, result


def make_sql_tool(shared_state: dict, llm: ChatOllama):
    @tool(args_schema=QueryInfrastructureInput)
    def query_infrastructure_db(natural_language_question: str) -> str:
        """Answers a question about utility infrastructure assets (utility poles,
        transmission towers, transformers) by generating and running a DuckDB SQL
        query against the assets table. Use this for any question involving counts,
        filters, locations, dates, or condition/damage scores. The argument must be
        a plain-English question (never SQL) — this tool writes the SQL itself.
        Returns the SQL used and a preview of the result rows."""
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
            return f"SQL executed:\n{sql}\n\nResult: 0 rows. No matching assets were found."

        preview = rows[:15]
        note = (
            f" (showing first {len(preview)} of {total_count} total — use this total_count "
            "when reporting a count, not the number of rows shown)"
            if result["truncated"]
            else ""
        )
        return f"SQL executed:\n{sql}\n\nResult: {total_count} row(s) total.{note}\n{preview}"

    return query_infrastructure_db

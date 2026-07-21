"""Wires the tool-calling agent: a LangGraph ReAct agent over a local Ollama
model, with tools for structured spatial SQL querying and map rendering."""
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agent import config
from agent.map_tool import make_map_tool
from agent.sql_tool import make_sql_tool

SYSTEM_PROMPT = """You are a geospatial infrastructure data copilot. You answer
questions about utility assets (poles, transmission towers, transformers) in
San Francisco using the tools available to you.

Rules:
- If the question is about locations, distances, "show me", or "map" style
  requests, call ONLY render_map (it fetches its own data and answers counts
  too) — do not also call query_infrastructure_db for the same question.
- Otherwise, for questions about counts, filters, dates, or condition/damage
  scores with no map/location intent, call query_infrastructure_db.
- Either tool's argument must be the user's question restated in plain
  English. Never write SQL yourself and never pass SQL as a tool argument —
  the tools write the SQL for you.
- For fuzzy/descriptive questions about inspection notes (e.g. "reports
  mentioning corrosion", "similar to X"), call search_inspection_notes.
- Never describe a tool call in words (e.g. "Now I will call X") instead of
  actually calling it — always issue a real tool call, silently, then wait
  for its result.
- Only after all required tool calls are done, give one concise final
  natural-language answer using the actual numbers/rows returned by the
  tools. Never invent data the tools didn't return. If a tool returned zero
  rows, say so plainly.
"""


def build_agent(extra_tools=None):
    """Builds a fresh agent + shared result cache for a single request."""
    shared_state: dict = {}
    orchestrator_llm = ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )
    # Separate, deterministic model instance for NL-to-SQL generation, shared
    # by the SQL tool and the (self-contained) map tool.
    sql_llm = ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )

    tools = [make_sql_tool(shared_state, sql_llm), make_map_tool(shared_state, sql_llm)]
    if extra_tools:
        tools.extend(t(shared_state) for t in extra_tools)

    agent = create_react_agent(orchestrator_llm, tools, prompt=SYSTEM_PROMPT)
    return agent, shared_state


def ask(question: str, extra_tools=None) -> dict:
    agent, shared_state = build_agent(extra_tools)
    result = agent.invoke({"messages": [("user", question)]})
    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "sql": shared_state.get("sql"),
        "rows": shared_state.get("rows", []),
        "map_html": shared_state.get("map_html"),
    }

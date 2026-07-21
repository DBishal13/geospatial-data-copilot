"""Streamlit chat + map UI for the Geospatial Data Copilot. Talks to the
FastAPI backend over HTTP. Run: streamlit run frontend/app.py

Layout: the map is the persistent main canvas (it survives across turns and
only updates when a query actually returns spatial results); the chat lives
in the sidebar so it never pushes the map around or gets buried in scroll."""
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from agent import config  # noqa: E402

st.set_page_config(page_title="Geospatial Data Copilot", page_icon="🗺️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_map_html" not in st.session_state:
    st.session_state.latest_map_html = None
if "latest_map_question" not in st.session_state:
    st.session_state.latest_map_question = None


@st.cache_data(show_spinner=False)
def fetch_overview_map():
    """Fast, agent-free default map (no LLM call) so the canvas is never
    empty — fetched once per Streamlit process and cached."""
    resp = requests.get(f"{config.API_BASE_URL}/overview-map", timeout=30)
    resp.raise_for_status()
    return resp.json()

EXAMPLES = [
    "Show me all poles inspected in the last 6 months with a high damage score.",
    "How many assets are within 2 miles of San Francisco City Hall?",
    "Which asset type has the highest average condition score decline this year?",
    "Find inspection notes similar to reports mentioning corrosion.",
    "Show me the 10 poles in the worst condition on a map.",
]

with st.sidebar:
    st.title("🗺️ Geospatial Copilot")
    st.caption(
        "Real asset locations from OpenStreetMap; inspection dates, condition "
        "scores, and notes are synthetic (OSM has no such history)."
    )

    with st.expander("Example questions"):
        for ex in EXAMPLES:
            st.markdown(f"- {ex}")

    chat_box = st.container(height=420, border=True)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sql"):
                    with st.expander("SQL used"):
                        st.code(msg["sql"], language="sql")

    question = st.chat_input("Ask a question...")

st.title("San Francisco Utility Infrastructure")

if st.session_state.latest_map_question:
    st.caption(f'Map showing: "{st.session_state.latest_map_question}"')
    st.components.v1.html(st.session_state.latest_map_html, height=650, scrolling=True)
else:
    try:
        overview = fetch_overview_map()
    except requests.RequestException as e:
        overview = None
        st.error(f"Couldn't reach the API at {config.API_BASE_URL}: {e}")

    if overview and overview.get("map_html"):
        st.caption(f"Showing all {overview['count']} seeded assets — ask a question to filter this view.")
        st.components.v1.html(overview["map_html"], height=650, scrolling=True)
    elif overview:
        st.info("No assets in the database yet — run `python db/build_dataset.py` to seed it.")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking (running local model + spatial query)..."):
        try:
            resp = requests.post(f"{config.API_BASE_URL}/query", json={"question": question}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            data = None
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Couldn't reach the API at {config.API_BASE_URL}: {e}"}
            )

    if data:
        st.session_state.messages.append(
            {"role": "assistant", "content": data["answer"], "sql": data.get("sql")}
        )
        if data.get("map_html"):
            st.session_state.latest_map_html = data["map_html"]
            st.session_state.latest_map_question = question

    st.rerun()

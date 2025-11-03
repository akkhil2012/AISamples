"""
Streamlit frontend for the AI-Powered Error Resolution System.

Assumptions
-----------
1. The FastAPI backend is running locally on the host/port defined in
   `st.secrets["backend_url"]` (default: "http://localhost:8000").
2. All MCP servers are already launched by the backend startup event.
3. `streamlit run frontend/streamlit_app.py` is executed from the project
   root so relative imports work.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import networkx as nx
import pandas as pd
import requests
import streamlit as st
from networkx.readwrite import json_graph
from streamlit.components.v1 import html

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BACKEND_URL: str = st.secrets.get("backend_url", "http://localhost:8000")
SEARCH_ENDPOINT: str = f"{BACKEND_URL}/api/v1/search"
GRAPH_ENDPOINT: str = f"{BACKEND_URL}/api/v1/graph/{{query_id}}"
EXPORT_ENDPOINT: str = f"{BACKEND_URL}/api/v1/export/{{query_id}}?format=pdf"

st.set_page_config(
    page_title="AI-Powered Error Resolution",
    layout="wide",
    page_icon="🔍",
)

# -----------------------------------------------------------------------------
# Sidebar – about & settings
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info(
    "Configure backend URL in `.streamlit/secrets.toml` if different from\n"
    "default `http://localhost:8000`."
)

backend_input = st.sidebar.text_input("Backend URL", BACKEND_URL)
if backend_input != BACKEND_URL:
    BACKEND_URL = backend_input.rstrip("/")
    SEARCH_ENDPOINT = f"{BACKEND_URL}/api/v1/search"
    GRAPH_ENDPOINT = f"{BACKEND_URL}/api/v1/graph/{{query_id}}"
    EXPORT_ENDPOINT = f"{BACKEND_URL}/api/v1/export/{{query_id}}?format=pdf"

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _fetch(url: str, method: str = "get", **kwargs):
    """Thin wrapper around `requests` that raises on non-200."""

    func = requests.get if method.lower() == "get" else requests.post
    resp = func(url, timeout=60, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} – {resp.text}")
    return resp.json()


def _render_graph(graph_json: dict[str, list]):
    """Render an interactive network graph with vis.js inside Streamlit."""

    graph_data = json.dumps(graph_json)
    vis_code = f"""
    <div id="mynetwork" style="height:600px;border:1px solid #ddd;"></div>
    <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <script>
      const container = document.getElementById('mynetwork');
      const data = {nodes: new vis.DataSet({graph_json['nodes']}),
                    edges: new vis.DataSet({graph_json['edges']})};
      const options = {{ layout: {{improvedLayout:true}}, physics: {{stabilization: true}} }};
      new vis.Network(container, data, options);
    </script>
    """

    html(vis_code, height=620)


# -----------------------------------------------------------------------------
# Main UI – input form
# -----------------------------------------------------------------------------
st.title("🔍 AI-Powered Error Resolution System")

with st.form(key="error_form"):
    col1, col2, col3 = st.columns(3)

    severity_level = col1.selectbox("Error Severity Level", ["P1", "P2", "P3"])
    error_code = col2.text_input("Error Code", placeholder="e.g. ERR_CONNECTION_TIMEOUT")
    environment = col3.selectbox("Environment", ["Dev", "Test", "Prod"])

    error_description = st.text_area(
        "Error Description", height=120,
        placeholder="Describe the error, logs, stack-trace …"
    )

    col4, col5 = st.columns(2)
    application_name = col4.text_input("Application Name", placeholder="Service / App")
    applicable_pool = col5.text_input("Applicable Pool", placeholder="Cluster / Pool")

    submitted = st.form_submit_button("🔎 Search for Solutions")

if submitted:
    if not error_description:
        st.warning("The *Error Description* field is required.")
        st.stop()

    payload = {
        "severity_level": severity_level,
        "error_code": error_code,
        "error_description": error_description,
        "application_name": application_name,
        "environment": environment,
        "applicable_pool": applicable_pool,
    }

    with st.spinner("Processing with AI & searching sources …"):
        try:
            response = _fetch(SEARCH_ENDPOINT, method="post", json=payload)
        except Exception as exc:
            st.error(f"❌ Search failed – {exc}")
            st.stop()

    # ------------------------------------------------------------------
    # Results tabs
    # ------------------------------------------------------------------
    st.success("✅ Search complete")

    # Expanded query terms ------------------------------------------------
    with st.expander("🔧 Expanded Query Details"):
        st.json(response.get("expanded_query", {}), expanded=False)

    # Top recommendations -------------------------------------------------
    st.subheader("🏆 Top 3 Recommended Resolutions")
    rec_df = pd.json_normalize(response.get("top_recommendations", []))
    if not rec_df.empty:
        st.dataframe(rec_df[[
            "rank", "title", "source", "confidence_score", "reasoning"
        ]], hide_index=True)
    else:
        st.info("No ranked recommendations available.")

    # Per-source results ---------------------------------------------------
    st.subheader("📚 Source Results")
    res_tabs = st.tabs(list(response["search_results"].keys()))
    for tab, (source, results) in zip(res_tabs, response["search_results"].items()):
        with tab:
            df = pd.json_normalize(results)
            if df.empty:
                st.write("No results.")
            else:
                st.dataframe(df[[
                    "title", "snippet", "relevance", "url"
                ]], height=300, hide_index=True)

    # Visual workflow graph ----------------------------------------------
    st.subheader("🗺️ Search Workflow Graph")
    with st.spinner("Rendering graph …"):
        graph_json = _fetch(GRAPH_ENDPOINT.format(query_id=response["query_id"]))
        _render_graph(graph_json)

    # Export button -------------------------------------------------------
    with st.spinner():
        export_info = _fetch(EXPORT_ENDPOINT.format(query_id=response["query_id"]))
    pdf_url = f"{BACKEND_URL}{export_info['download_url']}"
    st.download_button("💾 Download PDF Report", pdf_url, "report.pdf")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    textwrap.dedent(
        """
        <br><hr>
        <small>
        Built with ❤️ using Streamlit, FastAPI, Mistral AI & Model-Context-Protocol.<br>
        © 2025 Your Company. All rights reserved.
        </small>
        """),
    unsafe_allow_html=True,
)

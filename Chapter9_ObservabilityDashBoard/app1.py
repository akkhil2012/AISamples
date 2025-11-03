# app.py
# Streamlit + single-file demo: "Orchestrating proactive workflows with Observability Systems"
# - D3 force-directed graph for SaaS services
# - Select services and trigger parallel runs
# - 5-step pipeline:
#   1) Semantic cache
#   2) Local SLM reasoning (simulated via TF-IDF similarity)
#   3) Vector DB / Embeddings similarity search (simulated via TF-IDF)
#   4) MCP tools over proprietary data (simulated Splunk logs)
#   5) AWS Bedrock fallback (stub)
# - Generates a per-service PDF and displays it inline

import base64
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------- Page & State ----------------------------------

st.set_page_config(page_title="Proactive Observability Orchestrator", layout="wide")

SERVICES = ["Service1", "Service2", "Service3"]
DEFAULT_LINKS = [
    ("Service1", "Service2"),
    ("Service2", "Service3"),
]

# Session state
if "semantic_cache" not in st.session_state:
    # Cache key: f"{service}::{normalized_prompt}"
    st.session_state.semantic_cache: Dict[str, Dict] = {}

if "last_status" not in st.session_state:
    # status: idle | running | done | error
    st.session_state.last_status = {s: "idle" for s in SERVICES}

if "last_pdfs" not in st.session_state:
    # map service -> bytes (pdf)
    st.session_state.last_pdfs: Dict[str, bytes] = {}

if "last_results" not in st.session_state:
    # store structured result logs for each service
    st.session_state.last_results: Dict[str, Dict] = {}


# ----------------------------- Sample Data ------------------------------------

# Simulated “local SLM knowledge” per service (Step 2)
LOCAL_KB = {
    "Service1": [
        "Service1 handles real-time transaction anomalies and alerts with minimal latency.",
        "Optimize thresholding for P1 incidents in production environments.",
        "Service1 supports semantic enrichment of Splunk logs before triage."
    ],
    "Service2": [
        "Service2 is responsible for user authentication and session lifecycle.",
        "Audit failures when token expiration is misaligned with gateway.",
        "Adaptive rate limits can reduce authentication spikes."
    ],
    "Service3": [
        "Service3 performs data export to data lake and delta tables.",
        "Backpressure on Kafka streams can cause batch delays.",
        "Partitioning and compaction strategies improve throughput."
    ],
}

# Simulated “vector DB / embeddings” knowledge (Step 3)
VECTOR_DB = {
    "Service1": [
        "Use similarity search on incident titles and metadata to suggest past resolutions.",
        "Correlate 5xx spikes with deploy timestamps to detect causal regressions.",
    ],
    "Service2": [
        "Map repeated login failures to IP reputation and device fingerprint.",
        "SSO misconfiguration often shows as 403 spikes immediately after IdP changes.",
    ],
    "Service3": [
        "Delta table vacuum misconfigurations lead to storage bloat.",
        "Event-time watermarking prevents late data causing reprocessing churn.",
    ],
}

# Simulated proprietary Splunk logs (Step 4)
SPLUNK_LOGS = {
    "Service1": [
        "2025-08-02 P1 spike in /payments/authorize correlates with release r-4501.",
        "Circuit breaker opened for upstream gateway timeouts."
    ],
    "Service2": [
        "2025-08-15 Increased 401s from mobile clients after certificate rotation.",
        "SSO callback mismatch detected for tenant xzy-prod."
    ],
    "Service3": [
        "2025-08-21 Export lag grew to 45m due to Kafka partition imbalance.",
        "Delta write failures due to schema evolution mismatch in bronze table."
    ],
}


# ----------------------------- Utilities --------------------------------------

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def jaccard_rank(a: str, b: str) -> float:
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    if not sa or not sb: return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union else 0.0

def tfidf_best_match(query: str, docs: List[str]) -> Tuple[float, Optional[str]]:
    if not docs:
        return 0.0, None
    corpus = docs + [query]
    vect = TfidfVectorizer()
    X = vect.fit_transform(corpus)
    query_vec = X[-1]
    doc_vecs = X[:-1]
    sims = (doc_vecs @ query_vec.T).toarray().ravel()
    idx = int(np.argmax(sims))
    return float(sims[idx]), docs[idx]


# ----------------------------- Step Functions ---------------------------------

def step1_semantic_cache(service: str, prompt: str) -> Tuple[bool, Dict]:
    """Check semantic cache and compute rank."""
    key = f"{service}::{normalize(prompt)}"
    if key in st.session_state.semantic_cache:
        hit = st.session_state.semantic_cache[key]
        return True, {
            "step": 1,
            "decision": "cache_hit",
            "rank": hit["rank"],
            "answer": hit["answer"],
        }

    # If not found, we still compute a rank against a tiny service seed to judge "nearby"
    seed = " ".join(LOCAL_KB.get(service, []))
    rank = jaccard_rank(prompt, seed)
    return False, {
        "step": 1,
        "decision": "cache_miss",
        "rank": rank,
    }


def step2_local_model(service: str, prompt: str) -> Optional[Dict]:
    """Simulate a local SLM call by TF-IDF match into service-local KB."""
    score, text = tfidf_best_match(prompt, LOCAL_KB.get(service, []))
    if score >= 0.75:
        return {
            "step": 2,
            "decision": "local_model_answer",
            "score": score,
            "answer": text,
        }
    return None


def step3_vector_db(service: str, prompt: str) -> Optional[Dict]:
    """Simulated vector DB / embeddings similarity search."""
    score, text = tfidf_best_match(prompt, VECTOR_DB.get(service, []))
    if score >= 0.55:
        return {
            "step": 3,
            "decision": "vector_db_hit",
            "score": score,
            "answer": text,
        }
    return None


def step4_mcp_agents(service: str, prompt: str) -> Optional[Dict]:
    """Simulate MCP tools over proprietary Splunk logs with keyword match."""
    logs = SPLUNK_LOGS.get(service, [])
    p = normalize(prompt)
    for line in logs:
        if any(tok in normalize(line) for tok in p.split()):
            return {
                "step": 4,
                "decision": "proprietary_data_match",
                "log": line,
                "answer": f"From proprietary logs: {line}",
            }
    return None


def step5_bedrock_fallback(service: str, prompt: str) -> Dict:
    """Stub for AWS Bedrock call. Replace with boto3 Bedrock-runtime if desired."""
    # Example: integrate real call with credentials (left as a stub)
    return {
        "step": 5,
        "decision": "bedrock_fallback",
        "answer": f"[Bedrock Stub] Reasoned generic answer for {service}:\n"
                  f"Summarized approach for: '{prompt}'. Consider checking recent deploys, "
                  f"correlating error spikes with changes, and running targeted diagnostics."
    }


def persist_cache(service: str, prompt: str, answer: str, rank: float):
    key = f"{service}::{normalize(prompt)}"
    st.session_state.semantic_cache[key] = {
        "answer": answer,
        "rank": rank,
        "ts": time.time()
    }


# ----------------------------- Orchestration ----------------------------------

def orchestrate_service(service: str, prompt: str) -> Dict:
    """Runs the 5-step chain for a service and returns a structured result."""
    steps_log: List[Dict] = []

    # Step 1: Semantic cache
    hit, info1 = step1_semantic_cache(service, prompt)
    steps_log.append(info1)
    if hit:
        final_answer = info1["answer"]
        # (Cache already had rank)
        return {
            "service": service,
            "final_answer": final_answer,
            "steps": steps_log,
            "from_step": 1,
        }

    # Step 2: Local model (simulated)
    info2 = step2_local_model(service, prompt)
    if info2:
        steps_log.append(info2)
        final_answer = info2["answer"]
        persist_cache(service, prompt, final_answer, steps_log[0]["rank"])
        return {
            "service": service,
            "final_answer": final_answer,
            "steps": steps_log,
            "from_step": 2,
        }

    # Step 3: Vector DB (simulated)
    info3 = step3_vector_db(service, prompt)
    if info3:
        steps_log.append(info3)
        final_answer = info3["answer"]
        persist_cache(service, prompt, final_answer, steps_log[0]["rank"])
        return {
            "service": service,
            "final_answer": final_answer,
            "steps": steps_log,
            "from_step": 3,
        }

    # Step 4: MCP over proprietary data (simulated)
    info4 = step4_mcp_agents(service, prompt)
    if info4:
        steps_log.append(info4)
        final_answer = info4["answer"]
        persist_cache(service, prompt, final_answer, steps_log[0]["rank"])
        return {
            "service": service,
            "final_answer": final_answer,
            "steps": steps_log,
            "from_step": 4,
        }

    # Step 5: Bedrock fallback (stub)
    info5 = step5_bedrock_fallback(service, prompt)
    steps_log.append(info5)
    final_answer = info5["answer"]
    persist_cache(service, prompt, final_answer, steps_log[0]["rank"])
    return {
        "service": service,
        "final_answer": final_answer,
        "steps": steps_log,
        "from_step": 5,
    }


# ----------------------------- PDF Builder ------------------------------------

def build_pdf(service: str, prompt: str, result: Dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Service Report: {service}", ln=1)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Prompt:\n{prompt}")
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Resolution Flow:", ln=1)

    pdf.set_font("Arial", "", 11)
    for step in result.get("steps", []):
        title = f"Step {step['step']} - {step.get('decision','')}"
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, title)
        pdf.set_font("Arial", "", 11)
        detail_lines = []
        for k, v in step.items():
            if k in ("step", "decision"):
                continue
            detail_lines.append(f"{k}: {v}")
        if detail_lines:
            pdf.multi_cell(0, 6, "\n".join(detail_lines))
        pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Final Answer:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, result["final_answer"])

    # Output to bytes
    pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
    return pdf_bytes


def embed_pdf(pdf_bytes: bytes, height: int = 420):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.components.v1.html(iframe, height=height + 20)


# ----------------------------- D3 Graph ---------------------------------------

def render_d3(nodes: List[Dict], links: List[Tuple[str, str]]):
    # Build the D3 HTML content with inline script (v7)
    link_json = [{"source": s, "target": t} for (s, t) in links]
    import json
    
    # Prepare the JavaScript code with proper escaping
    js_code = """
    <div id="graph" style="width:100%; height:500px; border:1px solid #eee; border-radius:12px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    // Initialize variables
    const width = document.getElementById('graph').clientWidth;
    const height = 500;

    // Data from Python - using JSON.parse to safely parse the JSON strings
    const nodes = JSON.parse('{nodes_escaped}');
    const links = JSON.parse('{links_escaped}');

    // Color mapping for node status
    const colorMap = {{
        "idle": "#A3A3A3",
        "running": "#3B82F6",
        "done": "#10B981",
        "error": "#EF4444"
    }};

    // Set up the simulation
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(function(d) {{ return d.id; }}).distance(80))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(40));

    // Create SVG container
    const svg = d3.select("#graph")
        .append("svg")
        .attr("width", "100%")
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto;");

    // Create links
    const link = svg.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke-width", 1.5);

    // Create nodes
    const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    // Add circles to nodes
    node.append("circle")
        .attr("r", 15)
        .attr("fill", function(d) {{ return colorMap[d.status] || "#888"; }})
        .attr("stroke", "#1f2937")
        .attr("stroke-width", 1.5);

    // Add text labels to nodes
    node.append("text")
        .text(function(d) {{ return d.id; }})
        .attr("x", 28)
        .attr("y", 5)
        .attr("font-size", "12px")
        .attr("font-family", "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto");

    // Create legend
    const legend = svg.append("g").attr("transform", "translate(10, 10)");
    const statuses = ["idle", "running", "done", "error"];
    
    // Add legend items
    statuses.forEach(function(s, i) {
        var yPos = i * 18;
        var g = legend.append("g").attr("transform", "translate(0, " + yPos + ")");
        g.append("circle").attr("r", 6).attr("cx", 6).attr("cy", 6).attr("fill", colorMap[s]);
        g.append("text").attr("x", 18).attr("y", 10).text(s).attr("font-size","11px");
    });

    // Update positions on each tick
    simulation.on("tick", function() {
        link
            .attr("x1", function(d) {{ return d.source.x; }})
            .attr("y1", function(d) {{ return d.source.y; }})
            .attr("x2", function(d) {{ return d.target.x; }})
            .attr("y2", function(d) {{ return d.target.y; }});

        node.attr("transform", function(d) {{ 
            return "translate(" + d.x + "," + d.y + ")"; 
        }});
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    </script>
    """.format(
        nodes_escaped=json.dumps(nodes).replace("'", "\\'"),
        links_escaped=json.dumps(link_json).replace("'", "\\'")
    )

    st.components.v1.html(js_code, height=520)


def refresh_graph(status_map: Dict[str, str], services: List[str]):
    nodes = [{"id": s, "status": status_map.get(s, "idle")} for s in services]
    render_d3(nodes, DEFAULT_LINKS)


# ----------------------------- UI ---------------------------------------------

st.title("🔭 Proactive Observability Orchestrator")
st.caption("Single-file Streamlit demo: D3 graph + parallel triggers + 5-step reasoning chain + per-node PDF output.")

col_graph, col_ctrl = st.columns([3, 2])

with col_graph:
    refresh_graph(st.session_state.last_status, SERVICES)

with col_ctrl:
    st.subheader("Trigger Services")
    prompt = st.text_area(
        "Request / Prompt",
        placeholder="Describe the issue, question, or action. E.g., 'Investigate 5xx spike after last deploy for payments.'",
        height=120,
    )

    selected = st.multiselect(
        "Select one or many services",
        options=SERVICES,
        default=["Service1", "Service2"]
    )

    run_parallel = st.button("▶️ Run Selected Services in Parallel", use_container_width=True)

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown(
        "- The graph shows node status (idle/running/done/error).\n"
        "- Each run generates a PDF per service with the full step trace and final answer.\n"
        "- Steps are simulated locally for a single-file demo; swap in your real APIs as needed."
    )

# ----------------------------- Execution --------------------------------------

def run_services(services: List[str], prompt_text: str):
    # Mark running
    for s in services:
        st.session_state.last_status[s] = "running"
    refresh_graph(st.session_state.last_status, SERVICES)

    results: Dict[str, Dict] = {}
    pdfs: Dict[str, bytes] = {}

    with ThreadPoolExecutor(max_workers=len(services)) as ex:
        futures = {ex.submit(orchestrate_service, s, prompt_text): s for s in services}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                result = fut.result()
                results[s] = result
                pdfs[s] = build_pdf(s, prompt_text, result)
                st.session_state.last_status[s] = "done"
            except Exception as e:
                results[s] = {"service": s, "error": str(e), "steps": []}
                st.session_state.last_status[s] = "error"

            refresh_graph(st.session_state.last_status, SERVICES)

    # store
    st.session_state.last_results.update(results)
    st.session_state.last_pdfs.update(pdfs)


if run_parallel:
    if not prompt.strip():
        st.error("Please enter a prompt before running.")
    elif not selected:
        st.error("Please select at least one service.")
    else:
        run_services(selected, prompt)


# ----------------------------- Output Area ------------------------------------

st.markdown("## 📄 Service Outputs")
for s in SERVICES:
    with st.expander(f"{s} — {st.session_state.last_status.get(s, 'idle').upper()}", expanded=False):
        res = st.session_state.last_results.get(s)
        pdf_bytes = st.session_state.last_pdfs.get(s)

        if res:
            st.markdown("**Step Trace**")
            for step in res.get("steps", []):
                st.markdown(f"- **Step {step.get('step')}** · *{step.get('decision','')}*")
            st.markdown("**Final Answer**")
            st.code(res.get("final_answer", "(no answer)"))

        if pdf_bytes:
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"{s}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            embed_pdf(pdf_bytes, height=420)


# ----------------------------- Integration Hooks ------------------------------
# If you have real FastAPI services per node, replace `orchestrate_service` internals
# with HTTP calls to your endpoints. Example sketch (not executed in this demo):
#
# import httpx
# def call_fastapi(service_url: str, payload: dict) -> dict:
#     with httpx.Client(timeout=30) as client:
#         r = client.post(service_url, json=payload)
#         r.raise_for_status()
#         return r.json()
#
# Then in orchestrate_service(), call your endpoint instead of local steps.
# Keep everything in this single file for simplicity.

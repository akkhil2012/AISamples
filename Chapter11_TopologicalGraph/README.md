## Chapter 11 – Topological Graph Anomaly Dashboard

Interactive Streamlit dashboards and utilities to visualize service/node topologies, explore logs/metrics CSVs, and detect anomalies (Isolation Forest). Includes a D3 force-graph view and modern Plotly/NetworkX-based UIs.

### Highlights

- Build network graphs from CSVs (nodes/edges/metrics)
- Visualize topology with NetworkX + Plotly or D3 (`d3_force_graph.html`)
- Run anomaly detection (Isolation Forest) on per-node metrics/log features
- Export or generate reports (via ReportLab utilities)

### Files

- `app.py` – Entry point (often Streamlit) to launch a basic dashboard
- `streamlit_modern_network_graph.py` – Modern Streamlit UI for graph + metrics
- `simple_network_graph.py` – Minimal example of graph creation/plot
- `create_graph.py` – Helpers to build a `networkx.Graph` from CSV data
- `AnomalyDetectionIsolationForest.py` – Isolation Forest anomaly detection code
- `anomaly_dashboard.py`, `anomaly_dashboard_latest.py` – Anomaly-focused dashboards/variants
- `backend_utils.py` – Shared utilities (data loading, processing, caching)
- `config.py` – Configuration (paths, thresholds, UI flags)
- `d3_force_graph.html` – Static D3 force-directed graph template
- `data/` – Sample CSVs for nodes/log metrics (Splunk-like, test datasets)
- `test.py` – Quick script/tests for data/graph routines
- `requirements.txt` – Python dependencies

### Prerequisites

- Python 3.9+

### Install

```bash
pip install -r requirements.txt
```

### Sample data

Sample CSVs live in `data/` and at repo root (`test.csv`, `node_0_test.csv`). Adjust paths in `config.py` or app arguments as needed.

### Run a dashboard

Modern Streamlit graph UI:
```bash
streamlit run streamlit_modern_network_graph.py
```

Basic app:
```bash
streamlit run app.py
```

Anomaly dashboards:
```bash
streamlit run anomaly_dashboard.py
# or
streamlit run anomaly_dashboard_latest.py
```

Open the printed URL (typically `http://localhost:8501`).

### Typical workflow

1. Configure data sources in `config.py` (CSV paths, column names, thresholds)
2. Launch a dashboard (modern or anomaly variant)
3. Explore the topology (hover/select nodes) and review per-node metrics
4. Run anomaly scoring (Isolation Forest) and inspect flagged nodes
5. Export or snapshot results (optional report generation utilities)

### Customization

- Update `create_graph.py` to map your CSV schema to nodes/edges/attributes
- Tune anomaly parameters in `AnomalyDetectionIsolationForest.py`
- Replace `d3_force_graph.html` with your own D3 template if preferred

### Troubleshooting

- Empty graph: verify CSV paths and column names in `config.py`
- Plotly rendering issues: ensure Streamlit and Plotly versions meet `requirements.txt`
- Anomaly model errors: check numeric column types and missing values in your CSVs



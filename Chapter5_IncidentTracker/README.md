## Chapter 5 – Incident Tracker (LLM-Assisted)

End-to-end incident tracking demo with:
- Backend for incident query processing, anomaly detection, and video report generation
- Frontend (Streamlit) to visualize incident graphs and interact with the system
- Optional MCP servers for Confluence and Microsoft Teams integration

### Structure

```
IncidentTracker/
├── backend/
│   ├── main.py                 # FastAPI (or similar) entrypoint for backend APIs
│   ├── query_processor.py      # Natural-language → query/routing logic
│   ├── mistral_client.py       # LLM client (Mistral) helpers
│   ├── videoCreater.py         # Creates short incident summary videos
│   └── anomalyDetection/
│       ├── AnomalyDetectionClassical.py  # Classical anomaly detection utilities
│       └── TestDatageneration.py         # Synthetic data generator for tests
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI for incident tracking
│   └── graph_builder.py        # Builds node/edge graph views for incidents
├── config/
│   ├── config.py               # Centralized configuration
│   ├── env-example.txt         # Example env vars
│   └── requirements.txt        # Backend deps
└── mcp_servers/
    ├── confluence_server.py    # MCP server to interact with Confluence
    └── teams_server.py         # MCP server to interact with MS Teams
```

### Prerequisites

- Python 3.9+
- Recommended: virtual environment
- Optional: API keys/credentials for LLM (Mistral), Confluence, Microsoft Teams

### Setup

1) Create `.env` (or export env vars) based on `IncidentTracker/config/env-example.txt`.

2) Install backend dependencies:
```bash
pip install -r IncidentTracker/config/requirements.txt
```

3) (Optional) Install frontend-specific libs if not included in backend requirements (Streamlit is typically included):
```bash
pip install streamlit
```

### Running

- Backend API (from `IncidentTracker/backend/`):
```bash
python main.py
```

- Frontend UI (from `IncidentTracker/frontend/`):
```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL (typically `http://localhost:8501`) to explore incidents and graphs.

### Key Components

- `query_processor.py`: Parses user queries, routes to anomaly detection, retrieval, or LLM summarization.
- `mistral_client.py`: Wraps calls to the Mistral API for summarization/insight.
- `anomalyDetection/AnomalyDetectionClassical.py`: Classical algorithms (e.g., Isolation Forest variants, thresholds) for detecting anomalies in metrics/logs.
- `anomalyDetection/TestDatageneration.py`: Utilities to generate synthetic time series/log-like data for demos.
- `videoCreater.py`: Programmatically renders short video summaries of incidents (e.g., timeline + narration/labels).
- `frontend/graph_builder.py`: Assembles incident graphs (nodes/edges, severities) for visualization.

### MCP Integrations (Optional)

- `mcp_servers/confluence_server.py`: Exposes operations to create/update pages with incident summaries.
- `mcp_servers/teams_server.py`: Posts updates to Microsoft Teams channels/threads.

Configure credentials via env vars or config before enabling these.

### Development Tips

- Start small: run backend locally, test endpoints (e.g., with cURL or a REST client), then open the Streamlit UI.
- Use `env-example.txt` as a template; avoid committing real secrets.
- For anomaly experiments, use `TestDatageneration.py` to craft synthetic spikes and verify detection.

### Troubleshooting

- Backend won’t start: validate required env vars in `config/config.py` and your `.env`.
- LLM errors: ensure Mistral API key is present and model names are correct.
- Streamlit UI blank: check backend URL configuration and console logs.
- Confluence/Teams failures: verify tokens/scopes and network access.

### License

Educational sample. Review and harden before production use.



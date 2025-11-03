## Chapter 7 – Semantic Cache and AI Service Dashboard

A FastAPI backend demonstrating a semantic cache for AI tool responses, storage to ChromaDB, optional MCP server integrations, and utilities like PDF report generation. Includes a startup script and a Streamlit frontend entry (adjust as needed).

### Features

- Semantic caching with embedding similarity (Sentence-Transformers)
- Pluggable AI tools via an MCP-style manager (mock tools included)
- ChromaDB storage of responses + metadata
- PDF report generation endpoint
- Health and service discovery endpoints
- Startup script to run backend and frontend together

### Files

- `fastapi_backend.py` – FastAPI app exposing:
  - `GET /` – basic status
  - `GET /health` – component health (cache size, Chroma status, tool count)
  - `POST /api/process` – main entry: check cache → call tool → cache + store
  - `POST /api/download` – generate PDF from content
  - `GET /api/services` – list available mock services
- `advanced_semantic_Cache.py` – Advanced cache with TTL, LRU eviction, persistence, analytics helpers
- `macp_integration.py` – Example MCP server interfaces, mock proprietary/external integrations, manager and initialization
- `requirements.txt` – Dependencies (FastAPI, Uvicorn, sentence-transformers, chromadb, etc.)
- `startup.sh` – Convenience script to install deps and launch services

### Prerequisites

- Python 3.9+
- For embeddings: model downloads require internet access (`all-MiniLM-L6-v2`)

### Install

```bash
pip install -r requirements.txt
```

### Run the backend

```bash
python fastapi_backend.py
```

Backend defaults to `http://localhost:8000`.

### Optional: Use the startup script

Runs backend and then a Streamlit app (you may need to add/adjust `streamlit_app.py`).

```bash
bash startup.sh
```

### Try the API

Health:
```bash
curl http://localhost:8000/health
```

List services:
```bash
curl http://localhost:8000/api/services
```

Process (semantic cache → tool → store):
```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "service_id": "semantic_search",
    "prompt": "best practices for vector databases",
    "user_id": "demo_user"
  }'
```

Generate PDF:
```bash
curl -X POST http://localhost:8000/api/download \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Semantic search results...",
    "filename": "ai_report"
  }' \
  -o ai_report.pdf
```

### How semantic cache works (backend)

1. Request hits `/api/process` with `service_id` and `prompt`.
2. Cache attempts a semantic match for the same service using embeddings and a similarity threshold.
3. On cache hit, returns cached response immediately.
4. On miss, calls the mapped MCP tool, caches the result, and stores it to ChromaDB with metadata.

### Customization tips

- Swap out mock MCP tools in `fastapi_backend.py` or wire real ones via `macp_integration.py`.
- Tune similarity threshold, cache size, TTLs in the advanced cache module.
- Persist cache by providing a `persistence_file` path (see `AdvancedSemanticCache`).
- Configure CORS and security before production use.

### Troubleshooting

- Model download errors: ensure internet access or pre-download models.
- ChromaDB failures: verify installation and permissions.
- PDF generation issues: check `/tmp` write access.



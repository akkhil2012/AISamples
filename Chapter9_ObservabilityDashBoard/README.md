## Chapter 9 – Observability Dashboard

Lightweight observability/demo dashboards (Streamlit/FastAPI-style scripts) to visualize metrics, experiment with prompts, and prototype LLM-assisted insights.

### Files

- `app.py` – Primary dashboard application (e.g., Streamlit or FastAPI) for visualizing signals/metrics and summaries.
- `app1.py` – Alternate or experimental dashboard variant (additional views or layouts).
- `prompt.txt` – Reference prompt(s) used for LLM-based summaries/alerts.

### Prerequisites

- Python 3.9+
- Recommended: virtual environment
- If an LLM is used by the apps, set the relevant API key in your environment.

### Install

Install common dependencies you use across chapters (Streamlit/FastAPI/etc.). If this chapter has no dedicated requirements file, install minimal deps you need, for example:

```bash
pip install streamlit fastapi uvicorn requests python-dotenv
```

### Run

- Streamlit-style run (if `app.py` is a Streamlit app):
```bash
streamlit run app.py
```

- Alternate app:
```bash
streamlit run app1.py
```

- FastAPI-style run (if `app.py` exposes an API):
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

Open the printed URL (typically `http://localhost:8501` for Streamlit or `http://localhost:8000` for FastAPI).

### Suggested workflow

1. Start `app.py` and confirm the base dashboard loads.
2. Adjust `prompt.txt` to tweak LLM summaries/alerts.
3. If applicable, point the dashboard to your data sources (logs, metrics APIs) via environment variables or a config block in the code.
4. Compare `app.py` vs `app1.py` to choose your preferred layout/flow.

### Configuration

- Environment variables (examples):
  - `OPENAI_API_KEY` or other model provider keys
  - `DATA_API_URL`, `LOGS_API_URL` for your observability backends

Set these in your shell or a `.env` file and load via `python-dotenv` if needed.

### Tips

- If charts don’t render, verify the data source returns valid JSON and the expected schema.
- For LLM prompts, iterate on `prompt.txt` and keep outputs short for quick scanning.
- Use separate terminal panes to run backend APIs and the Streamlit UI.

### License

Educational sample. Adapt to your org’s observability stack and secure credentials before production use.



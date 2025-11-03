# Lead Generation App (Chapter 4)

Streamlit UI that sends a lead-generation query to an n8n webhook and renders AI-enriched leads as markdown.

## What it does

- You enter a query like: "AI startups hiring in India".
- The app posts it to your n8n workflow at `/webhook/scout-leads`.
- Displays the returned leads (as markdown) in the page.

## Files

- `app.py` – Streamlit app with a single input + button and webhook call.
- `requirements.txt` – Minimal deps: `streamlit`, `requests`.

## Prerequisites

- Python 3.9+
- n8n running locally or hosted with a `scout-leads` webhook workflow.
  - Quick start (Docker):
    ```bash
    docker run -it --rm -p 5678:5678 n8nio/n8n
    ```

## Configure

Update the webhook URL in `app.py` if needed:

```python
webhook_url = "http://localhost:5678/webhook/scout-leads"
```

## Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL (usually `http://localhost:8501`).

## Expected n8n response

The app expects a JSON array with an object containing a `markdown` field:

```json
[
  {
    "markdown": "## Leads\n- Company A — website — note\n- Company B — website — note"
  }
]
```

Optionally, your workflow can also return a `file_url` (code includes commented UI for it).

## Troubleshooting

- Connection errors: Ensure n8n is running and the webhook URL is reachable.
- Empty output: Make sure your workflow sets `markdown` in the final response.
- HTTP 4xx/5xx: Inspect n8n execution logs for node errors.


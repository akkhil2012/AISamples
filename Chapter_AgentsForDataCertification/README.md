# GenAI-Powered Data Certification Engine

Automated data certification workflow for **completeness checks**, **drift detection**, and **audit-ready compliance reports**.

Implementation file: `data_certification_agents.py`

## Features

- **Completeness checks**
  - required columns
  - null-rate threshold validation
  - duplicate primary-key detection
- **Drift detection with PyTorch**
  - feature mean-shift checks on numeric columns
  - autoencoder reconstruction error ratio between baseline and current data
- **Compliance validation**
  - PII masking checks
  - region allowlist enforcement
  - retention-window checks
- **GenAI summary with Mistral AI**
  - Uses `mistral-large-latest` (via API) when `MISTRAL_API_KEY` is available
  - Falls back to deterministic local summary when key/dependency is unavailable

## Requirements

- Python 3.9+
- `pandas`
- `numpy`
- `torch`
- `requests` (optional, needed for Mistral API calls)

Install:

```bash
pip install pandas numpy torch requests
```

## Run

```bash
python data_certification_agents.py
```

The script generates demo baseline/current datasets and prints a JSON report with:

- run metadata (`run_id`, `timestamp_utc`)
- decision (`PASS`, `CONDITIONAL`, `FAIL`)
- overall weighted score
- detailed outputs for completeness, drift, and compliance checks
- audit summary text (Mistral-generated or local fallback)

## Configuration

Use `default_config()` as the starting point. Tune:

- quality null thresholds
- drift feature list and sensitivity
- compliance policy (PII columns, allowed regions, retention days)
- scoring weights and decision thresholds
- Mistral model selection

## Notes

- The demo intentionally injects quality/compliance/drift issues into current data so you can observe a non-PASS certification path.
- To use Mistral summary generation, set:

```bash
export MISTRAL_API_KEY=<your_key>
```

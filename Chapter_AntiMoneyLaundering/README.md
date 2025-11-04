## Anti-Money-Laundering (AML) Agents Playground

Interactive Streamlit demo showcasing simple, explainable agents for AML workflows: rule-based transaction scoring, graph insight, compliance decisions, and SAR drafting.

### What’s inside

- `app.py` (aka `aml_agents_app.py`):
  - TransactionAgent: rule-based risk scoring with per-factor contributions
  - GraphAgent: builds an account ↔ beneficiary graph and highlights hubs/beneficiaries
  - ComplianceAgent: applies thresholds to decide APPROVE/FLAG/ESCALATE
  - SARAgent: generates a draft Suspicious Activity Report for flagged items

### Requirements

- Python 3.9+
- Packages: `streamlit pandas networkx matplotlib numpy`

Install:
```bash
pip install streamlit pandas networkx matplotlib numpy
```

### Run

```bash
streamlit run app.py
# or, if you rename as in the header comment
streamlit run aml_agents_app.py
```

Open the printed URL (typically `http://localhost:8501`).

### Using the app

1) In the sidebar, tune rule weights and thresholds:
   - Large amount threshold/weight
   - High-risk countries and weight
   - Risky methods and weight
   - Rapid repeat threshold/weight
   - Structuring window sum threshold and per-10k weight
   - Decision thresholds: FLAG and ESCALATE

2) (Optional) Upload a CSV with columns:
   `txn_id, account_id, beneficiary_id, amount, currency, country, method, timestamp`
   - Timestamp should be parseable (ISO format recommended).

3) Click “Run Agents” to score transactions and render:
   - Scored transactions table with decisions
   - Per-transaction explainability (factor contributions)
   - Account ↔ Beneficiary graph (highlighting high-sum beneficiaries)
   - Flagged list and collapsible SAR drafts

### Sample data

If no CSV is uploaded, the app uses a small built-in sample covering normal and high-risk patterns (large amounts, high‑risk countries, structuring-like patterns, repeats).

### Notes and next steps

- Replace rule-based scoring with hybrid approaches (rules + learned features)
- Use a graph database and/or GNNs for deeper network analysis
- Add audit logs, model versioning, human-in-the-loop review
- Gate any GenAI-driven SAR generation with policy and manual checks



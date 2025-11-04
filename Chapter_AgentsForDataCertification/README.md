# Agents
## Data Certification Agents

Agents that collaborate to certify data quality and compliance across schema, quality, anomalies, lineage, and policy checks. Orchestrates all signals into an overall score and PASS/CONDITIONAL/FAIL decision.

### File

- `data_certification_agents.py` — Single-script demo containing:
  - SchemaAgent: required columns, dtypes, constraints, PK uniqueness
  - QualityAgent: nulls, ranges, uniqueness, referential integrity
  - AnomalyAgent: IsolationForest (if available) or z‑score fallback
  - LineageAgent: transform catalog validation and FX consistency check
  - PolicyAgent: PII masking, region allowlist, retention (demo)
  - CertificationOrchestrator: runs all agents, weights scores, decides outcome

### Requirements

- Python 3.9+
- Required: `pandas`, `numpy`
- Optional (for IsolationForest): `scikit-learn`

Install:
```bash
pip install pandas numpy scikit-learn
```

### Run

```bash
python data_certification_agents.py
```

The script prints a certification report with agent scores, overall score, decision, and findings.

### Demo data and metadata

`demo_dataset()` builds a small synthetic dataset and associated metadata dict with:
- Schema expectations (required columns, dtypes, constraints, primary key)
- Data quality config (null thresholds, ranges, reference integrity)
- Anomaly config (numeric columns, contamination, z-score cutoff)
- Lineage mappings and a simple FX rule check for `amount_usd`
- Policy rules (PII column list, allowed regions, masking required, retention)

### Tuning and experiments

- To observe certification transitions, un-comment hints in `__main__`:
  - Mask PII (replace emails with `***`)
  - Align `amount_usd` with declared FX rules
- Adjust agent weights and thresholds in `CertificationOrchestrator`
- Extend agents with org-specific rules (e.g., stricter referential integrity or policy checks)

### Output

Console report includes:
- Agent scores (0–1) and per-agent findings
- Overall score (weighted) and decision: PASS / CONDITIONAL / FAIL
- Notes describing next steps to reach PASS

Placeholder content for the agents chapter.

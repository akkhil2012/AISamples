"""GenAI-powered data certification engine.

This module implements an automated certification workflow with:
- Completeness checks (nulls, required columns, duplicate keys)
- Drift detection using PyTorch
- Audit-ready compliance report generation using Mistral AI (with local fallback)

Run:
    python data_certification_agents.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import requests
except ImportError:  # pragma: no cover - optional for LLM report generation
    requests = None


@dataclass
class CompletenessResult:
    score: float
    missing_columns: List[str]
    null_rate_by_column: Dict[str, float]
    null_rate_violations: Dict[str, float]
    duplicate_key_count: int
    findings: List[str] = field(default_factory=list)


@dataclass
class DriftResult:
    score: float
    drift_detected: bool
    feature_drift: Dict[str, float]
    reconstruction_error_train: float
    reconstruction_error_current: float
    findings: List[str] = field(default_factory=list)


@dataclass
class ComplianceResult:
    score: float
    pii_violations: int
    region_violations: int
    retention_violations: int
    findings: List[str] = field(default_factory=list)


@dataclass
class CertificationReport:
    run_id: str
    timestamp_utc: str
    decision: str
    overall_score: float
    completeness: CompletenessResult
    drift: DriftResult
    compliance: ComplianceResult
    raw_summary: str


class DriftAutoencoder(nn.Module):
    """Simple feedforward autoencoder for tabular drift scoring."""

    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        hidden_dim = max(8, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class DataCertificationEngine:
    """Runs completeness, drift, and compliance checks and emits an audit report."""

    def __init__(self, config: Dict):
        self.config = config

    def run(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> CertificationReport:
        completeness = self._run_completeness_checks(current_df)
        drift = self._run_drift_detection(baseline_df, current_df)
        compliance = self._run_compliance_checks(current_df)

        overall_score = (
            self.config["weights"]["completeness"] * completeness.score
            + self.config["weights"]["drift"] * drift.score
            + self.config["weights"]["compliance"] * compliance.score
        )

        decision = self._decision(overall_score, completeness, drift, compliance)
        summary = self._generate_audit_summary(decision, overall_score, completeness, drift, compliance)

        return CertificationReport(
            run_id=f"cert-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            decision=decision,
            overall_score=float(round(overall_score, 4)),
            completeness=completeness,
            drift=drift,
            compliance=compliance,
            raw_summary=summary,
        )

    def _run_completeness_checks(self, df: pd.DataFrame) -> CompletenessResult:
        required_cols = self.config["schema"]["required_columns"]
        null_threshold = self.config["quality"]["max_null_rate"]
        key_col = self.config["schema"]["primary_key"]

        missing_columns = [col for col in required_cols if col not in df.columns]
        null_rates = {col: float(df[col].isna().mean()) for col in df.columns}
        null_rate_violations = {
            col: rate
            for col, rate in null_rates.items()
            if col in required_cols and rate > null_threshold
        }

        duplicate_key_count = int(df[key_col].duplicated().sum()) if key_col in df.columns else len(df)

        score = 1.0
        findings: List[str] = []

        if missing_columns:
            findings.append(f"Missing required columns: {missing_columns}")
            score -= min(0.4, 0.1 * len(missing_columns))

        if null_rate_violations:
            findings.append(f"Null-rate violations: {null_rate_violations}")
            score -= min(0.3, 0.05 * len(null_rate_violations))

        if duplicate_key_count > 0:
            findings.append(f"Duplicate primary keys found: {duplicate_key_count}")
            score -= min(0.3, duplicate_key_count / max(1, len(df)))

        if not findings:
            findings.append("Completeness checks passed.")

        return CompletenessResult(
            score=max(0.0, round(score, 4)),
            missing_columns=missing_columns,
            null_rate_by_column={k: round(v, 6) for k, v in null_rates.items()},
            null_rate_violations={k: round(v, 6) for k, v in null_rate_violations.items()},
            duplicate_key_count=duplicate_key_count,
            findings=findings,
        )

    def _run_drift_detection(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
        feature_cols = self.config["drift"]["numeric_columns"]
        drift_threshold = self.config["drift"]["mean_shift_threshold"]

        baseline = baseline_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        current = current_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

        mean = baseline.mean(axis=0)
        std = baseline.std(axis=0)
        std[std == 0] = 1.0

        baseline_norm = (baseline - mean) / std
        current_norm = (current - mean) / std

        feature_drift = {
            col: float(abs(current_norm[:, i].mean() - baseline_norm[:, i].mean()))
            for i, col in enumerate(feature_cols)
        }

        model = DriftAutoencoder(input_dim=len(feature_cols), latent_dim=min(4, len(feature_cols)))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        x_train = torch.tensor(baseline_norm)
        model.train()
        for _ in range(self.config["drift"].get("epochs", 60)):
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, x_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_recon = model(torch.tensor(baseline_norm))
            curr_recon = model(torch.tensor(current_norm))
            train_err = float(nn.functional.mse_loss(train_recon, torch.tensor(baseline_norm)).item())
            curr_err = float(nn.functional.mse_loss(curr_recon, torch.tensor(current_norm)).item())

        reconstruction_ratio = curr_err / max(train_err, 1e-8)
        drift_detected = any(v > drift_threshold for v in feature_drift.values()) or reconstruction_ratio > 1.5

        score = 1.0
        findings: List[str] = []

        if drift_detected:
            findings.append(
                "Drift detected in one or more numeric features or reconstruction error ratio exceeded threshold."
            )
            drift_intensity = min(1.0, sum(feature_drift.values()) / max(1, len(feature_drift)))
            score -= min(0.6, 0.3 * drift_intensity + 0.3 * min(1.0, reconstruction_ratio - 1.0))
        else:
            findings.append("No material drift detected.")

        return DriftResult(
            score=max(0.0, round(score, 4)),
            drift_detected=drift_detected,
            feature_drift={k: round(v, 6) for k, v in feature_drift.items()},
            reconstruction_error_train=round(train_err, 6),
            reconstruction_error_current=round(curr_err, 6),
            findings=findings,
        )

    def _run_compliance_checks(self, df: pd.DataFrame) -> ComplianceResult:
        policy = self.config["compliance"]
        pii_cols = policy["pii_columns"]
        pii_mask_token = policy["mask_token"]
        allowed_regions = set(policy["allowed_regions"])
        max_retention_days = policy["max_retention_days"]

        findings: List[str] = []
        pii_violations = 0

        for col in pii_cols:
            if col in df.columns:
                unmasked = df[col].dropna().astype(str).str.contains("@|[0-9]{3}-[0-9]{2}-[0-9]{4}", regex=True)
                pii_violations += int(unmasked.sum())

        region_violations = 0
        if "region" in df.columns:
            region_violations = int((~df["region"].isin(allowed_regions)).sum())

        retention_violations = 0
        if "event_date" in df.columns:
            event_ts = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
            age_days = (pd.Timestamp.now(tz="UTC") - event_ts).dt.days
            retention_violations = int((age_days > max_retention_days).fillna(False).sum())

        score = 1.0
        if pii_violations > 0:
            findings.append(f"PII masking violations detected: {pii_violations}")
            score -= min(0.5, 0.02 * pii_violations)
        if region_violations > 0:
            findings.append(f"Region policy violations detected: {region_violations}")
            score -= min(0.3, 0.02 * region_violations)
        if retention_violations > 0:
            findings.append(f"Retention violations detected: {retention_violations}")
            score -= min(0.3, 0.02 * retention_violations)

        if not findings:
            findings.append(f"Compliance checks passed (PII token expected: '{pii_mask_token}').")

        return ComplianceResult(
            score=max(0.0, round(score, 4)),
            pii_violations=pii_violations,
            region_violations=region_violations,
            retention_violations=retention_violations,
            findings=findings,
        )

    def _decision(
        self,
        overall_score: float,
        completeness: CompletenessResult,
        drift: DriftResult,
        compliance: ComplianceResult,
    ) -> str:
        if compliance.pii_violations > 0:
            return "FAIL"
        if overall_score >= self.config["decision"]["pass_threshold"] and not drift.drift_detected:
            return "PASS"
        if overall_score >= self.config["decision"]["conditional_threshold"]:
            return "CONDITIONAL"
        return "FAIL"

    def _generate_audit_summary(
        self,
        decision: str,
        overall_score: float,
        completeness: CompletenessResult,
        drift: DriftResult,
        compliance: ComplianceResult,
    ) -> str:
        payload = {
            "decision": decision,
            "overall_score": round(overall_score, 4),
            "completeness": completeness.__dict__,
            "drift": drift.__dict__,
            "compliance": compliance.__dict__,
        }

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key or requests is None:
            return self._local_summary(payload)

        try:
            prompt = (
                "You are a data governance auditor. Generate an audit-ready certification summary with: "
                "executive status, key failures, risk implications, and remediation steps. "
                f"Input JSON:\n{json.dumps(payload, indent=2)}"
            )
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.config["llm"]["model"],
                    "messages": [
                        {"role": "system", "content": "You create concise audit reports."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._local_summary(payload)

    @staticmethod
    def _local_summary(payload: Dict) -> str:
        lines = [
            f"Certification Decision: {payload['decision']} (score={payload['overall_score']})",
            "",
            "Key Findings:",
            f"- Completeness: {'; '.join(payload['completeness']['findings'])}",
            f"- Drift: {'; '.join(payload['drift']['findings'])}",
            f"- Compliance: {'; '.join(payload['compliance']['findings'])}",
            "",
            "Recommended Remediation:",
            "1. Fix mandatory schema/data completeness violations and deduplicate primary keys.",
            "2. Re-baseline feature distributions or investigate upstream pipeline changes for drift.",
            "3. Mask PII, enforce region allowlists, and purge stale records beyond retention window.",
        ]
        return "\n".join(lines)


def build_demo_data(seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create baseline and current datasets with intentional issues for demonstration."""
    rng = np.random.default_rng(seed)

    n_base, n_curr = 500, 500
    baseline = pd.DataFrame(
        {
            "record_id": np.arange(1, n_base + 1),
            "amount": rng.normal(100, 15, n_base).round(2),
            "quantity": rng.integers(1, 8, n_base),
            "score": rng.normal(0.75, 0.08, n_base).clip(0, 1),
            "region": rng.choice(["US", "EU", "APAC"], n_base, p=[0.4, 0.35, 0.25]),
            "customer_email": ["***" for _ in range(n_base)],
            "event_date": pd.Timestamp.now(tz="UTC") - pd.to_timedelta(rng.integers(1, 60, n_base), unit="D"),
        }
    )

    current = pd.DataFrame(
        {
            "record_id": np.arange(10001, 10001 + n_curr),
            "amount": rng.normal(128, 28, n_curr).round(2),
            "quantity": rng.integers(1, 12, n_curr),
            "score": rng.normal(0.62, 0.12, n_curr).clip(0, 1),
            "region": rng.choice(["US", "EU", "APAC", "LATAM"], n_curr, p=[0.35, 0.3, 0.25, 0.1]),
            "customer_email": np.where(rng.random(n_curr) < 0.1, "user@example.com", "***"),
            "event_date": pd.Timestamp.now(tz="UTC") - pd.to_timedelta(rng.integers(1, 240, n_curr), unit="D"),
        }
    )

    current.loc[current.sample(frac=0.06, random_state=seed).index, "amount"] = np.nan
    current.loc[current.sample(frac=0.03, random_state=seed + 1).index, "score"] = np.nan

    return baseline, current


def default_config() -> Dict:
    return {
        "schema": {
            "required_columns": [
                "record_id",
                "amount",
                "quantity",
                "score",
                "region",
                "customer_email",
                "event_date",
            ],
            "primary_key": "record_id",
        },
        "quality": {"max_null_rate": 0.05},
        "drift": {
            "numeric_columns": ["amount", "quantity", "score"],
            "mean_shift_threshold": 0.30,
            "epochs": 80,
        },
        "compliance": {
            "pii_columns": ["customer_email"],
            "mask_token": "***",
            "allowed_regions": ["US", "EU", "APAC"],
            "max_retention_days": 180,
        },
        "weights": {"completeness": 0.35, "drift": 0.30, "compliance": 0.35},
        "decision": {"pass_threshold": 0.9, "conditional_threshold": 0.7},
        "llm": {"model": "mistral-large-latest"},
    }


def report_to_dict(report: CertificationReport) -> Dict:
    return {
        "run_id": report.run_id,
        "timestamp_utc": report.timestamp_utc,
        "decision": report.decision,
        "overall_score": report.overall_score,
        "completeness": report.completeness.__dict__,
        "drift": report.drift.__dict__,
        "compliance": report.compliance.__dict__,
        "audit_summary": report.raw_summary,
    }


if __name__ == "__main__":
    base_df, curr_df = build_demo_data()
    engine = DataCertificationEngine(default_config())
    report = engine.run(base_df, curr_df)

    print("=" * 80)
    print("GENAI DATA CERTIFICATION REPORT")
    print("=" * 80)
    print(json.dumps(report_to_dict(report), indent=2, default=str))

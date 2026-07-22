# severity_router.py
# pip install langchain-openai pydantic
#
# export OPENROUTER_API_KEY=sk-or-v1-...
# python severity_router.py

import os
import re
import json
import time
from typing import Literal, Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# =====================================================================
# 1. SEVERITY MODEL
# =====================================================================

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

class SeverityDecision(BaseModel):
    """Structured output the LLM classifier must produce."""
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity of the user's request"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in this classification, 0 to 1"
    )
    reason: str = Field(description="One-line justification")


def escalate(severity: str, levels: int = 1) -> str:
    """Move severity up N levels (capped at critical)."""
    idx = min(SEVERITY_ORDER.index(severity) + levels, len(SEVERITY_ORDER) - 1)
    return SEVERITY_ORDER[idx]

# =====================================================================
# 2. LAYER 1 — RULE TRIPWIRES (deterministic, checked first, no ML)
# =====================================================================

CRITICAL_PATTERNS = [
    r"\bdata\s*breach\b",
    r"\bhacked\b|\bcompromised\b",
    r"\bproduction\s+(is\s+)?down\b|\boutage\b",
    r"\bunauthorized\s+(access|transaction)\b",
    r"\bfraud\b",
    r"\bcannot\s+access\s+my\s+account\b.*\bmoney\b",
]

HIGH_PATTERNS = [
    r"\bnot\s+working\b|\bbroken\b|\bfailing\b",
    r"\berror\s+\d{3}\b",          # HTTP-style error codes
    r"\bpayment\s+failed\b",
]

def rule_layer(prompt: str) -> str | None:
    """Return a severity if a tripwire fires, else None."""
    text = prompt.lower()
    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, text):
            return "critical"
    for pattern in HIGH_PATTERNS:
        if re.search(pattern, text):
            return "high"
    return None

# =====================================================================
# 3. LAYER 2 — LLM CLASSIFIER (structured output, label only)
# =====================================================================

llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_retries=3,
    temperature=0,                  # deterministic-ish classification
)

classifier = llm.with_structured_output(SeverityDecision)

CLASSIFIER_INSTRUCTIONS = """You classify support requests by severity.

Definitions:
- critical: security incidents, data loss, fraud, safety risk, total outage
- high: a core feature or service is broken/unusable for the user
- medium: something is degraded, slow, or partially working
- low: general question, feature request, feedback, how-to

Classify the following request."""

def llm_classifier_layer(prompt: str) -> SeverityDecision:
    return classifier.invoke(f"{CLASSIFIER_INSTRUCTIONS}\n\nRequest: {prompt}")

# =====================================================================
# 4. LAYER 3 — CONFIDENCE SAFETY NET
# =====================================================================

CONFIDENCE_FLOOR = 0.6   # below this, over-escalate rather than guess

def apply_confidence_net(decision: SeverityDecision) -> tuple[str, bool]:
    """If the classifier is unsure, escalate one level. Returns (severity, was_escalated)."""
    if decision.confidence < CONFIDENCE_FLOOR:
        return escalate(decision.severity), True
    return decision.severity, False

# =====================================================================
# 5. DISPATCH — deterministic routing table (mock backends)
# =====================================================================

def faq_bot(prompt: str) -> str:
    return f"[FAQ-BOT] Here's a help article for: '{prompt[:50]}'"

def standard_support(prompt: str) -> str:
    return f"[SUPPORT-SLM] Troubleshooting steps generated for: '{prompt[:50]}'"

def priority_pipeline(prompt: str) -> str:
    return f"[PRIORITY-RAG] Ticket created, diagnostics running for: '{prompt[:50]}'"

def incident_backend(prompt: str) -> str:
    return f"[INCIDENT] 🚨 Escalated to on-call, incident ticket opened: '{prompt[:50]}'"

ROUTES: dict[str, Callable[[str], str]] = {
    "low":      faq_bot,
    "medium":   standard_support,
    "high":     priority_pipeline,
    "critical": incident_backend,
}

# =====================================================================
# 6. AUDIT LOG — every decision becomes training data for Option A later
# =====================================================================

LOG_FILE = "routing_log.jsonl"

def log_decision(record: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# =====================================================================
# 7. THE ROUTER — glue it all together
# =====================================================================

def route(prompt: str) -> str:
    record = {"ts": time.time(), "prompt": prompt}

    # Layer 1: rules
    rule_severity = rule_layer(prompt)
    if rule_severity:
        record.update({"layer": "rule", "severity": rule_severity,
                       "confidence": 1.0, "escalated": False})
        log_decision(record)
        print(f"  ⚡ rule tripwire → {rule_severity}")
        return ROUTES[rule_severity](prompt)

    # Layer 2: LLM classifier
    decision = llm_classifier_layer(prompt)
    print(f"  🤖 classifier → {decision.severity} "
          f"(conf={decision.confidence:.2f}) | {decision.reason}")

    # Layer 3: confidence safety net
    severity, escalated = apply_confidence_net(decision)
    if escalated:
        print(f"  ⬆️  low confidence → escalated to {severity}")

    record.update({"layer": "llm", "severity": severity,
                   "raw_severity": decision.severity,
                   "confidence": decision.confidence,
                   "escalated": escalated, "reason": decision.reason})
    log_decision(record)
    return ROUTES[severity](prompt)

# =====================================================================
# 8. DEMO
# =====================================================================

if __name__ == "__main__":
    test_prompts = [
        "How do I change my profile picture?",
        "The dashboard is loading really slowly since yesterday",
        "Payment failed three times and I got error 502",
        "I think my account was hacked, there are transactions I didn't make",
        "hmm something feels off with my data maybe",   # vague → tests confidence net
    ]

    for p in test_prompts:
        print(f"\nUser: {p}")
        result = route(p)
        print(f"  → {result}")

    print(f"\n📋 All decisions logged to {LOG_FILE} (your future training data)")
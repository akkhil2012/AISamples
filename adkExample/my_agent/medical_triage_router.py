# medical_triage_router.py
# pip install langchain-openai pydantic
#
# export OPENROUTER_API_KEY=sk-or-v1-...
# python medical_triage_router.py
#
# NOTE: Demo of a routing PATTERN for patient portal messages.
# Not a medical device; a real deployment needs clinical validation,
# regulatory review (SaMD), and clinician-in-the-loop oversight.

import os
import re
import json
import time
from typing import Literal, Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# =====================================================================
# 1. TRIAGE MODEL
# =====================================================================

SEVERITY_ORDER = ["routine", "moderate", "urgent", "emergency"]

class TriageDecision(BaseModel):
    """Structured output the LLM classifier must produce."""
    severity: Literal["routine", "moderate", "urgent", "emergency"] = Field(
        description="Clinical urgency of the patient's message"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this triage level, 0 to 1"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Any concerning symptoms mentioned, verbatim from the message"
    )
    reason: str = Field(description="One-line clinical justification")


def escalate(severity: str, levels: int = 1) -> str:
    idx = min(SEVERITY_ORDER.index(severity) + levels, len(SEVERITY_ORDER) - 1)
    return SEVERITY_ORDER[idx]

# =====================================================================
# 2. LAYER 1 — RED-FLAG TRIPWIRES (deterministic, never rely on ML alone)
#     These encode "never-miss" symptoms. In production this list comes
#     from clinical protocols (e.g., triage guidelines), owned by
#     clinicians and version-controlled — not by engineering.
# =====================================================================

EMERGENCY_PATTERNS = [
    r"\bchest\s+(pain|pressure|tightness)\b",
    r"\b(can'?t|cannot|difficulty|trouble)\s+breath",
    r"\bshortness\s+of\s+breath\b",
    r"\b(face|arm|speech)\s+droop", 
    r"\bslurred\s+speech\b",
    r"\bsudden\s+(numbness|weakness|confusion|vision\s+loss)\b",   # stroke signs
    r"\bunconscious\b|\bpassed\s+out\b|\bfaint(ed|ing)\b",
    r"\bseizure\b|\bconvulsion",
    r"\bsevere\s+bleeding\b|\bcoughing\s+(up\s+)?blood\b|\bvomiting\s+blood\b",
    r"\banaphyla|\bthroat\s+(closing|swelling)\b",
    r"\boverdose\b|\bswallowed\b.*\b(pills|poison|chemical)",
    r"\bsevere\s+allergic\s+reaction\b",
]

URGENT_PATTERNS = [
    r"\bhigh\s+fever\b|\bfever\b.*\b(103|104|40\s*°?c)\b",
    r"\bsevere\s+(pain|headache|abdominal)\b|\bworst\s+headache\b",
    r"\bdehydrat",
    r"\bblood\s+in\s+(urine|stool)\b",
    r"\bpersistent\s+vomiting\b",
    r"\b(deep\s+cut|wound)\b.*\bstitches\b",
    r"\bpregnan\w+\b.*\b(bleeding|pain|contractions)\b",
]

def rule_layer(message: str) -> tuple[str, str] | None:
    """Return (severity, matched_pattern) if a red flag fires, else None."""
    text = message.lower()
    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, text):
            return "emergency", pattern
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, text):
            return "urgent", pattern
    return None

# =====================================================================
# 3. LAYER 2 — LLM TRIAGE CLASSIFIER (label only, never advice)
# =====================================================================

llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_retries=3,
    temperature=0,
)

classifier = llm.with_structured_output(TriageDecision)

CLASSIFIER_INSTRUCTIONS = """You are a triage classifier for a patient portal.
You do NOT give medical advice. You only assign an urgency level.

Definitions:
- emergency: possible life/limb threat — needs emergency services NOW
  (cardiac, stroke signs, breathing difficulty, severe bleeding,
   anaphylaxis, overdose, loss of consciousness)
- urgent: needs clinician attention within hours (high fever, severe pain,
  signs of serious infection, concerning symptoms in pregnancy)
- moderate: needs a response within 1-2 days (worsening chronic symptoms,
  medication side effects, new non-severe symptoms)
- routine: administrative or non-time-sensitive (appointment scheduling,
  prescription refills, general health questions, test result queries)

CRITICAL RULE: When in doubt between two levels, ALWAYS choose the higher one.
Under-triage can harm patients; over-triage only costs clinician time.

Classify the following patient message."""

def llm_classifier_layer(message: str) -> TriageDecision:
    return classifier.invoke(f"{CLASSIFIER_INSTRUCTIONS}\n\nPatient message: {message}")

# =====================================================================
# 4. LAYER 3 — CONFIDENCE SAFETY NET (more aggressive than support version)
# =====================================================================

CONFIDENCE_FLOOR = 0.75          # higher floor than the support-ticket router

def apply_confidence_net(decision: TriageDecision) -> tuple[str, bool]:
    """Uncertain triage always rounds UP. Red flags found by the LLM
    also force at least 'urgent' even if it labeled lower."""
    severity = decision.severity
    escalated = False

    if decision.red_flags and SEVERITY_ORDER.index(severity) < SEVERITY_ORDER.index("urgent"):
        severity, escalated = "urgent", True

    if decision.confidence < CONFIDENCE_FLOOR:
        severity, escalated = escalate(severity), True

    return severity, escalated

# =====================================================================
# 5. DISPATCH — mock clinical backends
# =====================================================================

def self_service_portal(msg: str) -> str:
    return f"[PORTAL] Routed to self-service (refills/scheduling/FAQs): '{msg[:50]}'"

def nurse_queue(msg: str) -> str:
    return f"[NURSE-QUEUE] Added to nurse review queue, 24-48h SLA: '{msg[:50]}'"

def clinician_priority(msg: str) -> str:
    return f"[CLINICIAN] ⚠️ Priority flag — clinician callback within hours: '{msg[:50]}'"

def emergency_protocol(msg: str) -> str:
    return (f"[EMERGENCY] 🚨 Patient shown emergency guidance "
            f"(call local emergency number / go to ER immediately) "
            f"+ care team alerted in parallel: '{msg[:50]}'")

ROUTES: dict[str, Callable[[str], str]] = {
    "routine":   self_service_portal,
    "moderate":  nurse_queue,
    "urgent":    clinician_priority,
    "emergency": emergency_protocol,
}

# =====================================================================
# 6. AUDIT LOG — in a real clinical system this is a compliance
#     requirement, not just training data. Log message HASH, not text,
#     if PHI handling policies require it.
# =====================================================================

LOG_FILE = "triage_log.jsonl"

def log_decision(record: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# =====================================================================
# 7. THE TRIAGE ROUTER
# =====================================================================

def route(message: str) -> str:
    record = {"ts": time.time(), "message": message}

    # Layer 1: red-flag rules — bypass ML entirely
    rule_hit = rule_layer(message)
    if rule_hit:
        severity, pattern = rule_hit
        record.update({"layer": "rule", "severity": severity,
                       "matched": pattern, "confidence": 1.0, "escalated": False})
        log_decision(record)
        print(f"  ⚡ red-flag tripwire ({pattern}) → {severity}")
        return ROUTES[severity](message)

    # Layer 2: LLM triage classifier
    decision = llm_classifier_layer(message)
    print(f"  🤖 classifier → {decision.severity} "
          f"(conf={decision.confidence:.2f}) | flags={decision.red_flags} | {decision.reason}")

    # Layer 3: round-up safety net
    severity, escalated = apply_confidence_net(decision)
    if escalated:
        print(f"  ⬆️  safety net → escalated to {severity}")

    record.update({"layer": "llm", "severity": severity,
                   "raw_severity": decision.severity,
                   "confidence": decision.confidence,
                   "red_flags": decision.red_flags,
                   "escalated": escalated, "reason": decision.reason})
    log_decision(record)
    return ROUTES[severity](message)

# =====================================================================
# 8. DEMO
# =====================================================================

if __name__ == "__main__":
    test_messages = [
        "I need to refill my blood pressure medication",
        "I've had a mild cough for a week and it's not going away",
        "My chest feels tight and my left arm is numb",
        "My 2-year-old has had a 104 fever since last night",
        "I've been feeling a bit strange since starting the new medication",  # vague → net
    ]

    for m in test_messages:
        print(f"\nPatient: {m}")
        result = route(m)
        print(f"  → {result}")

    print(f"\n📋 Triage decisions logged to {LOG_FILE}")
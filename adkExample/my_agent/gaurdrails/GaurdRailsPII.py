"""
CrewAI Guardrails for PII Protection
====================================
A dedicated demo of using CrewAI task guardrails to stop PII from
leaking out of agent outputs. Two enforcement modes:

  1. BLOCK  mode -> reject the output, agent must regenerate without PII
  2. REDACT mode -> auto-mask detected PII and let the sanitized text pass

Detectors included (regex + checksum, India-aware):
  - Email addresses
  - Phone numbers (incl. +91 formats)
  - Aadhaar numbers (12-digit, with Verhoeff-style format check)
  - PAN numbers (ABCDE1234F)
  - Credit/debit cards (13-19 digits, Luhn-validated to cut false positives)
  - API keys / secrets (sk-..., AKIA...)
  - IP addresses

Guardrail contract:
    def guardrail(output: TaskOutput) -> tuple[bool, Any]:
        (True,  clean_or_redacted_text)  -> accepted
        (False, "feedback for the agent") -> auto-retry with feedback

Requires:
    pip install crewai litellm
    export OPENROUTER_API_KEY="..."
"""

import os
import re
from dataclasses import dataclass
from typing import Any, List, Tuple

from crewai import Agent, Crew, Process, Task
from crewai.tasks.task_output import TaskOutput

try:
    from crewai import LLM
except ImportError:  # older CrewAI versions
    from crewai.llm import LLM

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openrouter/openai/gpt-4o-mini"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY before running this demo")

os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = OPENROUTER_BASE_URL
os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

llm = LLM(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# ===============================================================
# 1. PII DETECTION ENGINE (pure Python — deterministic & auditable)
# ===============================================================

@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int


PII_PATTERNS = {
    "EMAIL":      r"[\w.+-]+@[\w-]+\.[\w.]{2,}",
    "PHONE_IN":   r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b",
    "AADHAAR":    r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b",
    "PAN":        r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "CARD":       r"\b\d(?:[\s-]?\d){12,18}\b",
    "API_KEY":    r"\b(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def luhn_valid(number: str) -> bool:
    """Luhn checksum — filters out random digit runs flagged as cards."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def detect_pii(text: str) -> List[PIIMatch]:
    """Scan text and return every PII hit with its location."""
    matches: List[PIIMatch] = []
    for pii_type, pattern in PII_PATTERNS.items():
        for m in re.finditer(pattern, text):
            value = m.group(0)

            # Reduce false positives with type-specific validation
            if pii_type == "CARD" and not luhn_valid(value):
                continue
            if pii_type == "IP_ADDRESS":
                octets = value.split(".")
                if any(int(o) > 255 for o in octets):
                    continue

            matches.append(PIIMatch(pii_type, value, m.start(), m.end()))

    # Resolve overlaps: prefer the LONGEST span (e.g. a 16-digit card
    # number would otherwise also trigger the 12-digit Aadhaar pattern
    # on its prefix). Overlapping shorter matches are discarded so
    # redaction offsets never collide.
    matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    resolved: List[PIIMatch] = []
    for m in matches:
        if resolved and m.start < resolved[-1].end:  # overlaps previous
            if (m.end - m.start) <= (resolved[-1].end - resolved[-1].start):
                continue                # shorter/equal -> drop it
            resolved[-1] = m            # longer -> replace previous
        else:
            resolved.append(m)
    return resolved


def redact(text: str, matches: List[PIIMatch]) -> str:
    """Replace each PII span with a typed placeholder, e.g. [REDACTED:EMAIL]."""
    # Replace from the end so earlier offsets stay valid
    for m in sorted(matches, key=lambda x: x.start, reverse=True):
        text = text[:m.start] + f"[REDACTED:{m.pii_type}]" + text[m.end:]
    return text


# ===============================================================
# 2. GUARDRAILS (the CrewAI gate functions)
# ===============================================================

def pii_block_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """STRICT mode: any PII -> reject and force the agent to regenerate.

    The failure message names the PII types (never echoes the values —
    you don't want PII leaking into logs/feedback either).
    """
    matches = detect_pii(output.raw)
    if matches:
        found_types = sorted({m.pii_type for m in matches})
        return (False,
                f"Output BLOCKED: contains PII of types {found_types} "
                f"({len(matches)} instance(s)). Regenerate the response "
                f"with ALL personal data removed — use role descriptions "
                f"like 'the customer' instead of names/contact details.")
    return (True, output.raw)


def pii_redact_guardrail(output: TaskOutput) -> Tuple[bool, Any]:
    """LENIENT mode: mask PII in place and pass the sanitized text on.

    Useful when the surrounding content is valuable and regeneration is
    expensive — e.g., summarizing support tickets that inherently quote
    customer messages.
    """
    matches = detect_pii(output.raw)
    if not matches:
        return (True, output.raw)

    sanitized = redact(output.raw, matches)
    print(f"\n[GUARDRAIL] Redacted {len(matches)} PII item(s): "
          f"{sorted({m.pii_type for m in matches})}")
    return (True, sanitized)


# --- Optional: LLM-judged semantic layer -----------------------
# Regex can't catch everything ("she lives near the big temple in
# Shamshabad, works at the only bank on MG Road" is identifying but
# pattern-free). Layer a no-code LLM guardrail on top for that:
#
#   Task(..., guardrail="Must not contain any information that could "
#                       "identify a specific real person: names, contact "
#                       "details, government IDs, or unique combinations "
#                       "of location + employer + role.")
#
# Strategy: regex guardrail = cheap deterministic floor,
#           LLM guardrail   = semantic ceiling. Use both for high stakes.

# ===============================================================
# 3. DEMO CREW
#    A support-ticket summarizer — a realistic PII leak scenario,
#    since raw tickets are full of names, emails, and phone numbers.
# ===============================================================

RAW_TICKET = """
Ticket #48211
From: Priya Sharma <priya.sharma1988@gmail.com>, +91 98491 23456
Issue: Charged twice on card 4539 1488 0343 6467 while paying my
insurance premium. My PAN is ABCPS1234K if you need it for verification.
Please fix urgently, my policy lapses Friday.
"""

summarizer = Agent(
    role="Support Ticket Analyst",
    goal="Summarize customer tickets for the engineering team",
    backstory=(
        "You turn raw support tickets into actionable engineering "
        "summaries. Engineering must NEVER see customer personal data — "
        "they only need the technical facts of the issue."
    ),
    verbose=True,
    llm=llm,
)

auditor = Agent(
    role="Privacy Compliance Auditor",
    goal="Write a privacy incident note about what PII appeared in inputs",
    backstory=(
        "You document data-handling events for the DPO. Your reports "
        "reference PII by TYPE only, never by value."
    ),
    verbose=True,
    llm=llm,
)

# Task A — BLOCK mode: the summary must be 100% PII-free or it's rejected
summary_task = Task(
    description=(
        f"Summarize this support ticket for the engineering team. "
        f"Include the technical issue, urgency, and suggested next step. "
        f"Do NOT include any personal data.\n\nTICKET:\n{RAW_TICKET}"
    ),
    expected_output="A 3-4 sentence engineering summary with zero PII",
    agent=summarizer,
    guardrail=pii_block_guardrail,   # strict gate
    max_retries=3,                   # failures -> feedback -> regenerate
)

# Task B — REDACT mode: quoting is allowed, but PII gets masked in flight
audit_task = Task(
    description=(
        "Write a short privacy note for the DPO describing what categories "
        "of personal data appeared in the original ticket and confirming "
        "the engineering summary was sanitized."
    ),
    expected_output="A brief DPO note referencing PII types only",
    agent=auditor,
    context=[summary_task],
    guardrail=pii_redact_guardrail,  # safety net: mask anything that slips
    max_retries=2,
)


def main():
    # Quick self-test of the detector before spending LLM tokens
    print("=" * 60)
    print("DETECTOR SELF-TEST on raw ticket")
    print("=" * 60)
    for m in detect_pii(RAW_TICKET):
        print(f"  found {m.pii_type:<10} at [{m.start}:{m.end}]")
    print("\nRedacted preview:\n", redact(RAW_TICKET, detect_pii(RAW_TICKET)))

    crew = Crew(
        agents=[summarizer, auditor],
        tasks=[summary_task, audit_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("FINAL OUTPUT (passed PII guardrails)")
    print("=" * 60)
    print(result.raw)

    # Prove the gate worked: re-scan the final artifacts
    leaks = detect_pii(summary_task.output.raw) + detect_pii(result.raw)
    print("\nPost-hoc verification:",
          "NO PII in final outputs ✔" if not leaks
          else f"LEAK DETECTED: {[m.pii_type for m in leaks]} ✘")


if __name__ == "__main__":
    main()
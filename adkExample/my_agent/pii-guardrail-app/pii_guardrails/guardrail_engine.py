"""
PII Guardrail Engine
--------------------
Pure-Python (stdlib only) PII detection + redaction engine so it can run
fully on-device inside an Android APK (no cloud egress).

Detects (India-first, extensible):
    - EMAIL
    - PHONE_IN          (Indian mobile numbers, +91 / 0 prefixed / bare 10-digit)
    - AADHAAR           (12 digits, validated with Verhoeff checksum)
    - PAN               (ABCDE1234F)
    - CREDIT_CARD       (13-19 digits, validated with Luhn)
    - IFSC              (bank branch codes)
    - UPI_ID            (name@bank)
    - GSTIN             (15-char GST number, embeds a PAN)
    - PASSPORT_IN       (A1234567)
    - IP_ADDRESS        (IPv4)
    - DOB               (dd/mm/yyyy, dd-mm-yyyy)

Policy actions per entity type: ALLOW | REDACT | BLOCK
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Pattern, Tuple


# --------------------------------------------------------------------------
# Checksum validators
# --------------------------------------------------------------------------

def luhn_valid(number: str) -> bool:
    """Luhn checksum for credit/debit card numbers."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# Verhoeff tables (used by UIDAI for Aadhaar checksums)
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """Verhoeff checksum — Aadhaar numbers must pass this."""
    digits = re.sub(r"\D", "", number)
    if len(digits) != 12 or digits[0] in "01":
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


# --------------------------------------------------------------------------
# Entity definitions
# --------------------------------------------------------------------------

class Action(str, Enum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


@dataclass
class EntitySpec:
    name: str
    pattern: Pattern
    validator: Optional[Callable[[str], bool]] = None
    severity: str = "MEDIUM"          # LOW | MEDIUM | HIGH | CRITICAL


ENTITY_SPECS: List[EntitySpec] = [
    EntitySpec(
        "AADHAAR",
        re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
        validator=verhoeff_valid,
        severity="CRITICAL",
    ),
    EntitySpec(
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        validator=luhn_valid,
        severity="CRITICAL",
    ),
    EntitySpec(
        "PAN",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        severity="HIGH",
    ),
    EntitySpec(
        "GSTIN",
        re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"),
        severity="HIGH",
    ),
    EntitySpec(
        "EMAIL",
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        severity="MEDIUM",
    ),
    EntitySpec(
        "UPI_ID",
        re.compile(
            r"\b[a-zA-Z0-9.\-_]{2,}@(?:ybl|upi|oksbi|okaxis|okhdfcbank|okicici|"
            r"paytm|apl|ibl|axl)\b"
        ),
        severity="HIGH",
    ),
    EntitySpec(
        "PHONE_IN",
        re.compile(r"(?:(?<=\s)|^|(?<=[:,(]))(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}\b"),
        severity="MEDIUM",
    ),
    EntitySpec(
        "IFSC",
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        severity="MEDIUM",
    ),
    EntitySpec(
        "PASSPORT_IN",
        re.compile(r"\b[A-PR-WY][1-9]\d{6}\b"),
        severity="HIGH",
    ),
    EntitySpec(
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
        severity="LOW",
    ),
    EntitySpec(
        "DOB",
        re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}\b"),
        severity="MEDIUM",
    ),
]


# --------------------------------------------------------------------------
# Findings & results
# --------------------------------------------------------------------------

@dataclass
class Finding:
    entity: str
    value: str
    start: int
    end: int
    severity: str
    action: Action

    def masked(self) -> str:
        """Show last 4 chars only, e.g. XXXX-XXXX-1234."""
        raw = self.value
        keep = 4 if len(raw) > 4 else 0
        return "X" * (len(raw) - keep) + raw[-keep:] if keep else "X" * len(raw)


@dataclass
class GuardrailResult:
    original_text: str
    sanitized_text: str
    findings: List[Finding] = field(default_factory=list)
    decision: Action = Action.ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision == Action.BLOCK

    def summary(self) -> str:
        if not self.findings:
            return "No PII detected. Input allowed."
        lines = [f"Decision: {self.decision.value}  |  {len(self.findings)} finding(s)"]
        for f in self.findings:
            lines.append(
                f"  - {f.entity:<12} [{f.severity:<8}] {f.masked()}  -> {f.action.value}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Policy + Engine
# --------------------------------------------------------------------------

DEFAULT_POLICY: Dict[str, Action] = {
    "AADHAAR": Action.BLOCK,        # never let Aadhaar reach the model
    "CREDIT_CARD": Action.BLOCK,
    "PAN": Action.REDACT,
    "GSTIN": Action.REDACT,
    "EMAIL": Action.REDACT,
    "UPI_ID": Action.REDACT,
    "PHONE_IN": Action.REDACT,
    "IFSC": Action.REDACT,
    "PASSPORT_IN": Action.REDACT,
    "IP_ADDRESS": Action.REDACT,
    "DOB": Action.REDACT,
}


class PIIGuardrail:
    """
    Usage:
        guard = PIIGuardrail()
        result = guard.apply("My PAN is ABCDE1234F")
        result.sanitized_text  -> "My PAN is <PAN_REDACTED>"
    """

    def __init__(
        self,
        policy: Optional[Dict[str, Action]] = None,
        redaction_style: str = "tag",     # "tag" | "mask" | "hash"
    ):
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.redaction_style = redaction_style

    # -- detection ---------------------------------------------------------

    def detect(self, text: str) -> List[Finding]:
        findings: List[Finding] = []
        claimed: List[Tuple[int, int]] = []
        for spec in ENTITY_SPECS:
            for m in spec.pattern.finditer(text):
                span = (m.start(), m.end())
                # skip spans already claimed by a higher-priority entity
                if any(s < span[1] and span[0] < e for s, e in claimed):
                    continue
                value = m.group()
                if spec.validator and not spec.validator(value):
                    continue
                action = self.policy.get(spec.name, Action.REDACT)
                findings.append(
                    Finding(spec.name, value, span[0], span[1], spec.severity, action)
                )
                claimed.append(span)
        findings.sort(key=lambda f: f.start)
        return findings

    # -- redaction ---------------------------------------------------------

    def _replacement(self, f: Finding) -> str:
        if self.redaction_style == "mask":
            return f.masked()
        if self.redaction_style == "hash":
            digest = hashlib.sha256(f.value.encode()).hexdigest()[:10]
            return f"<{f.entity}:{digest}>"
        return f"<{f.entity}_REDACTED>"

    def apply(self, text: str) -> GuardrailResult:
        findings = self.detect(text)
        if any(f.action == Action.BLOCK for f in findings):
            return GuardrailResult(
                original_text=text,
                sanitized_text="",
                findings=findings,
                decision=Action.BLOCK,
            )
        sanitized = text
        # replace from the end so offsets stay valid
        for f in sorted(findings, key=lambda f: f.start, reverse=True):
            if f.action == Action.REDACT:
                sanitized = sanitized[: f.start] + self._replacement(f) + sanitized[f.end :]
        decision = Action.REDACT if findings else Action.ALLOW
        return GuardrailResult(text, sanitized, findings, decision)


if __name__ == "__main__":
    demo = (
        "Hi, I'm Ravi. Email ravi.k@example.com, phone +91 98765 43210. "
        "My PAN is ABCDE1234F, card 4111 1111 1111 1111, "
        "Aadhaar 2341 5678 9012, IFSC HDFC0001234."
    )
    guard = PIIGuardrail()
    res = guard.apply(demo)
    print(res.summary())
    print("\nSanitized:", res.sanitized_text or "(blocked)")

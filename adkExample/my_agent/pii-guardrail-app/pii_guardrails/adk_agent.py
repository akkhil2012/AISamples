"""
Google ADK agent wired with PII guardrails.

The guardrail is enforced at two ADK interception points:

1. before_model_callback  -> sanitizes/blocks user input BEFORE it reaches
                             the LLM (Gemini). Blocked inputs never leave
                             the device/process.
2. after_model_callback   -> scans the model's OUTPUT so the agent can't
                             echo PII back (e.g. from tool results).

Run locally / server-side:
    pip install google-adk
    adk web            (from the parent directory)   or:
    python -m pii_guardrails.adk_agent "My PAN is ABCDE1234F, help me file taxes"
"""

from __future__ import annotations

import sys
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from .guardrail_engine import PIIGuardrail, Action

guard = PIIGuardrail(redaction_style="tag")


# ---------------------------------------------------------------------------
# Callback 1: input guardrail (runs before every LLM call)
# ---------------------------------------------------------------------------

def pii_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Sanitize the latest user message; block the call entirely on CRITICAL PII."""
    contents = llm_request.contents or []

    for content in reversed(contents):
        if content.role != "user" or not content.parts:
            continue
        for part in content.parts:
            if not getattr(part, "text", None):
                continue

            result = guard.apply(part.text)

            # Audit trail in session state (survives across turns)
            audit = callback_context.state.get("pii_audit", [])
            audit.append(
                {
                    "decision": result.decision.value,
                    "entities": [f.entity for f in result.findings],
                }
            )
            callback_context.state["pii_audit"] = audit

            if result.blocked:
                blocked_types = ", ".join(
                    sorted({f.entity for f in result.findings if f.action == Action.BLOCK})
                )
                # Returning an LlmResponse SKIPS the model call completely.
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                text=(
                                    "⛔ Request blocked by PII guardrail. "
                                    f"Critical identifiers detected: {blocked_types}. "
                                    "Please remove them and try again."
                                )
                            )
                        ],
                    )
                )

            if result.decision == Action.REDACT:
                part.text = result.sanitized_text  # model only ever sees redacted text
        break  # only inspect the most recent user turn

    return None  # proceed to the model


# ---------------------------------------------------------------------------
# Callback 2: output guardrail (runs after every LLM response)
# ---------------------------------------------------------------------------

def pii_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Redact any PII the model produced (e.g. leaked from tools/context)."""
    if not llm_response.content or not llm_response.content.parts:
        return None

    changed = False
    for part in llm_response.content.parts:
        if getattr(part, "text", None):
            result = guard.apply(part.text)
            if result.decision != Action.ALLOW:
                part.text = result.sanitized_text or "[output withheld by guardrail]"
                changed = True

    return llm_response if changed else None


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="pii_guarded_assistant",
    model="gemini-2.0-flash",
    description="Assistant that never sees or emits raw PII.",
    instruction=(
        "You are a helpful assistant. Input you receive has already been "
        "sanitized: tokens like <PAN_REDACTED> or <EMAIL_REDACTED> mark "
        "removed personal data. Never ask users to reveal the original "
        "values behind redaction tokens."
    ),
    before_model_callback=pii_before_model_callback,
    after_model_callback=pii_after_model_callback,
)


if __name__ == "__main__":
    # Quick smoke test of just the guardrail layer (no LLM call needed)
    text = " ".join(sys.argv[1:]) or "My PAN is ABCDE1234F and card is 4111 1111 1111 1111"
    print(guard.apply(text).summary())

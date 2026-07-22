# Fraud Case Investigation Pipeline — ADK Session State + output_key demo
#
# Scenario:
#   A customer reports suspected fraud. Instead of one monolithic prompt,
#   a SequentialAgent runs a 4-stage investigation pipeline. Each stage
#   writes its result into SESSION STATE via `output_key`, and the next
#   stage reads it via `{template}` injection in its instruction.
#
#   Stage 1  IntakeAgent      -> state["incident_details"]
#            (unstructured complaint -> structured incident JSON)
#   Stage 2  RiskScoringAgent -> state["risk_assessment"]
#            (reads {incident_details}, produces risk level + rationale)
#   Stage 3  ActionAgent      -> state["recommended_actions"]
#            (reads both, decides block-card / provisional-credit / monitor)
#   Stage 4  CaseNoteAgent    -> state["case_note"]
#            (reads all three, writes the audit-ready case note)
#
#   This is prompt chaining implemented as infrastructure: output_key is
#   the wire between stages, session state is the shared blackboard.
#
#   The demo then sends a SECOND message in the SAME session ("Also, I got
#   a suspicious SMS this morning") to show that state from the first run
#   persists and the pipeline can build on it — per-conversation working
#   memory in action.
#
# Setup:
#   pip install google-adk
#   export GOOGLE_API_KEY="..."      # (swap models to LiteLlm/OpenRouter if preferred)
#   python fraud_session_pipeline.py

import json
import asyncio
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part

MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------
# Stage 1: Intake — unstructured complaint -> structured incident
# ---------------------------------------------------------------

intake_agent = LlmAgent(
    name="IntakeAgent",
    model=MODEL,
    instruction=(
        "You are a fraud-intake specialist at a bank. "
        "Extract a structured incident record from the customer's message. "
        "Respond ONLY with compact JSON, no markdown fences, with keys: "
        "channel (card|upi|netbanking|unknown), "
        "amount (number or null), "
        "currency (string or null), "
        "merchant_or_beneficiary (string or null), "
        "customer_aware_of_transaction (true|false), "
        "possible_credential_compromise (true|false), "
        "summary (one sentence). "
        "If the customer mentions multiple incidents, include an "
        "'additional_signals' list."
    ),
    output_key="incident_details",          # -> state["incident_details"]
)

# ---------------------------------------------------------------
# Stage 2: Risk scoring — reads incident_details from state
# ---------------------------------------------------------------

risk_agent = LlmAgent(
    name="RiskScoringAgent",
    model=MODEL,
    instruction=(
        "You are a fraud risk analyst. Assess this incident:\n"
        "{incident_details}\n\n"                      # <- injected from state
        "Respond ONLY with compact JSON, no markdown fences, with keys: "
        "risk_level (LOW|MEDIUM|HIGH|CRITICAL), "
        "confidence (0.0-1.0), "
        "key_risk_factors (list of short strings), "
        "fraud_typology (e.g. 'card-not-present fraud', 'phishing-led "
        "account takeover', 'friendly fraud / dispute', 'unknown'). "
        "Rules of thumb: unauthorized transaction the customer was unaware "
        "of => at least HIGH. Credential compromise signals (phishing SMS, "
        "OTP sharing, suspicious links) escalate risk one level."
    ),
    output_key="risk_assessment",           # -> state["risk_assessment"]
)

# ---------------------------------------------------------------
# Stage 3: Action decision — reads BOTH prior stage outputs
# ---------------------------------------------------------------

action_agent = LlmAgent(
    name="ActionAgent",
    model=MODEL,
    instruction=(
        "You are a fraud-operations decision agent.\n"
        "Incident: {incident_details}\n"
        "Risk assessment: {risk_assessment}\n\n"
        "Decide immediate actions. Respond ONLY with compact JSON, "
        "no markdown fences, with keys: "
        "immediate_actions (ordered list chosen from: block_card, "
        "freeze_upi_mandates, force_password_reset, revoke_sessions, "
        "provisional_credit, file_fraud_case, enhanced_monitoring, "
        "customer_callback), "
        "sla_minutes (integer — time allowed before actions must complete), "
        "requires_human_approval (true|false — true for provisional_credit "
        "or anything CRITICAL), "
        "customer_message (2 calm, reassuring sentences to send the "
        "customer now)."
    ),
    output_key="recommended_actions",       # -> state["recommended_actions"]
)

# ---------------------------------------------------------------
# Stage 4: Case note — reads all three, writes the audit narrative
# ---------------------------------------------------------------

case_note_agent = LlmAgent(
    name="CaseNoteAgent",
    model=MODEL,
    instruction=(
        "You are writing the official case note for a fraud case file. "
        "It must be complete enough that an auditor six months from now "
        "understands what was reported, how it was assessed, and why "
        "actions were taken.\n"
        "Incident: {incident_details}\n"
        "Risk assessment: {risk_assessment}\n"
        "Actions: {recommended_actions}\n\n"
        "Write a professional case note of at most 120 words in plain "
        "prose (no JSON, no bullet points). Include the risk level and "
        "the rationale linking evidence to actions."
    ),
    output_key="case_note",                 # -> state["case_note"]
)

# ---------------------------------------------------------------
# The pipeline: SequentialAgent runs stages in order over ONE session,
# so each stage sees the state written by the previous ones.
# ---------------------------------------------------------------

fraud_pipeline = SequentialAgent(
    name="FraudInvestigationPipeline",
    sub_agents=[intake_agent, risk_agent, action_agent, case_note_agent],
)

# ---------------------------------------------------------------
# Execution
# ---------------------------------------------------------------

def show_state(state: dict, title: str) -> None:
    print(f"\n================ {title} ================")
    for key in ("incident_details", "risk_assessment",
                "recommended_actions", "case_note"):
        if key in state:
            value = state[key]
            # Pretty-print JSON stages; case_note is prose
            try:
                value = json.dumps(json.loads(value), indent=2)
            except (TypeError, ValueError):
                pass
            print(f"\n--- state['{key}'] ---\n{value}")


async def run_turn(runner: Runner, user_id: str, session_id: str,
                   text: str) -> None:
    print(f"\n>>> CUSTOMER: {text}")
    message = Content(role="user", parts=[Part(text=text)])
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if event.is_final_response():
            print(f"<<< Pipeline finished (final agent: {event.author})")


async def main():
    app_name, user_id, session_id = "fraud_app", "customer_042", "case_7781"

    session_service = InMemorySessionService()
    runner = Runner(
        agent=fraud_pipeline,
        app_name=app_name,
        session_service=session_service,
    )

    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Snapshot BEFORE anything runs — state is empty
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(f"Initial state: {session.state}")     # -> {}

    # ---- Turn 1: the fraud report ------------------------------
    await run_turn(
        runner, user_id, session_id,
        "There's a debit of Rs 48,500 on my credit card from a merchant "
        "called 'LUXGADGETS-SG' that I have never heard of. I did not "
        "make this purchase and I still have my card with me.",
    )

    # Re-fetch (the old `session` variable is a stale snapshot!)
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    show_state(session.state, "STATE AFTER TURN 1")

    # ---- Turn 2: SAME session — new evidence arrives -----------
    # The pipeline re-runs, but conversation history + prior state give
    # the intake agent context: this should now surface credential
    # compromise and escalate the risk level vs. turn 1.
    await run_turn(
        runner, user_id, session_id,
        "One more thing — this morning I got an SMS with a link saying my "
        "card reward points were expiring, and I clicked it and entered "
        "my card details before realizing it looked fake.",
    )

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    show_state(session.state, "STATE AFTER TURN 2 (escalated)")

    print("\nNote: each output_key was OVERWRITTEN by turn 2 — session "
          "state keeps the latest value per key. The full history of both "
          "turns lives in session.events, not session.state.")


if __name__ == "__main__":
    asyncio.run(main())
from google.adk.agents import LlmAgent, SequentialAgent

try:
    from google.adk.models.lite_llm import LiteLlm
except ImportError as exc:
    raise ImportError(
        "LiteLlm is required for OpenRouter provider models. "
        "Install it with `pip install google-adk[extensions]` or `pip install litellm>=1.75.5`."
    ) from exc

# ---------------------------------------------------------------------
# OpenRouter model
# ---------------------------------------------------------------------

OPENROUTER_MODEL = LiteLlm(model="openrouter/openai/gpt-4o-mini")

# =====================================================================
# Agent 1 : Fraud Detection Analyst
# =====================================================================

fraud_detector = LlmAgent(
    name="FraudDetectionAgent",
    model=OPENROUTER_MODEL,
    description="Analyzes customer account access attempts for potential fraud.",
    output_key="fraud_analysis",
    instruction="""
You are an AI Fraud Detection Analyst working for a retail bank.

The user will provide details of an account login or account access attempt.

Analyze the request using common fraud indicators such as:

- Multiple failed login attempts
- Login from an unfamiliar country
- Login from a TOR/VPN network
- Impossible travel
- New device
- Suspicious IP reputation
- Unusual transaction immediately after login
- Device fingerprint mismatch
- Login outside customer's normal behavior
- Account takeover indicators

Return ONLY a JSON object.

Example:

{
  "risk_score": 92,
  "risk_level": "HIGH",
  "fraud_detected": true,
  "reasons": [
      "Login from Russia",
      "Device never seen before",
      "Five failed password attempts"
  ]
}
""",
)

# =====================================================================
# Agent 2 : Banking Decision Engine
# =====================================================================

decision_agent = LlmAgent(
    name="BankingDecisionAgent",
    model=OPENROUTER_MODEL,
    description="Determines the banking action based on fraud analysis.",
    output_key="decision_result",
    instruction="""
You are the Fraud Decision Engine for a bank.

Read the fraud analysis stored in:

fraud_analysis

Determine the appropriate action.

Decision rules:

• LOW risk
    → ALLOW_LOGIN

• MEDIUM risk
    → REQUIRE_MFA

• HIGH risk
    → BLOCK_ACCOUNT
    → Freeze online banking session
    → Alert Fraud Operations
    → Notify customer

Return ONLY the following JSON:

{
   "decision":"ALLOW_LOGIN | REQUIRE_MFA | BLOCK_ACCOUNT",
   "confidence":95,
   "reason":"Explanation",
   "recommended_actions":[
       "...",
       "...",
       "..."
   ]
}
""",
)

# =====================================================================
# Sequential Pipeline
# =====================================================================

banking_fraud_pipeline = SequentialAgent(
    name="BankingFraudDetectionPipeline",
    description="Detects fraudulent account access and determines the appropriate banking action.",
    sub_agents=[
        fraud_detector,
        decision_agent,
    ],
)

# Root Agent

root_agent = banking_fraud_pipeline
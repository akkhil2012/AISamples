from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# OpenRouter provider model via LiteLlm. Requires litellm support.
MODEL = LiteLlm(model="openrouter/openai/gpt-4o-mini")

# =====================================================================
# Custom Banking Action Agent
# =====================================================================

class AccountProtectionAgent(BaseAgent):
    """
    Executes banking operations after fraud has been detected.
    """

    name: str = "AccountProtectionAgent"
    description: str = (
        "Protects customer accounts by blocking access, freezing "
        "transactions, and notifying the fraud operations team."
    )

    async def _run_async_impl(
        self,
        context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:

        # -----------------------------------------------------------------
        # Replace these print statements with actual banking APIs
        # -----------------------------------------------------------------

        print("✓ Blocking customer account...")
        print("✓ Freezing outgoing transactions...")
        print("✓ Enabling Multi-Factor Authentication...")
        print("✓ Creating fraud investigation case...")
        print("✓ Sending SMS notification to customer...")

        yield Event(
            author=self.name,
            content="""
Fraud mitigation completed successfully.

Actions Performed
-----------------
• Customer account blocked
• Outgoing transactions frozen
• Multi-Factor Authentication enabled
• Fraud investigation case created
• Customer notified via SMS
""",
        )


# =====================================================================
# Fraud Investigation Agent
# =====================================================================

fraud_investigator = LlmAgent(
    name="FraudInvestigator",
    model=MODEL,
    description="Analyzes suspicious customer login attempts.",
    instruction="""
You are an AI Fraud Analyst working for a retail bank.

Analyze the customer's login request.

Consider the following fraud indicators:

• Login from a foreign country
• VPN or Proxy detected
• New or unknown device
• Multiple failed login attempts
• Large money transfer
• Transfer to a newly added beneficiary
• Impossible travel
• Unusual customer behavior

Classify the request as:

• LOW Risk
• MEDIUM Risk
• HIGH Risk

If the risk is HIGH,
delegate execution to the AccountProtectionAgent.

Provide a concise explanation of your reasoning.
""",
)

# =====================================================================
# Banking Coordinator
# =====================================================================

banking_coordinator = LlmAgent(
    name="BankingCoordinator",
    model=MODEL,
    description="Coordinates fraud analysis and banking protection workflows.",
    instruction="""
You are the Banking Workflow Coordinator.

Your responsibilities are:

1. Delegate fraud analysis to FraudInvestigator.

2. Review the fraud assessment.

3. If the risk level is HIGH,
   invoke AccountProtectionAgent.

4. Return a consolidated response to the customer support team.
""",
    sub_agents=[
        fraud_investigator,
        AccountProtectionAgent(),
    ],
)

# =====================================================================
# Validate Agent Hierarchy
# =====================================================================

assert fraud_investigator.parent_agent == banking_coordinator

print("✅ Banking agent hierarchy created successfully.")
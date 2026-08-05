from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event


# =====================================================================
# Custom Banking Agent
# =====================================================================

class AccountProtectionAgent(BaseAgent):
    """
    Performs account protection actions after fraud is detected.
    """

    name: str = "AccountProtectionAgent"
    description: str = (
        "Executes banking protection actions such as blocking an "
        "account or requiring MFA."
    )

    async def _run_async_impl(
        self,
        context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:

        # In a real banking application this would call
        # IAM systems, fraud services, or account APIs.

        yield Event(
            author=self.name,
            content="""
Account Protection Action Completed

Action: BLOCK_ACCOUNT

Additional Actions:
- Freeze online banking
- Require MFA on next login
- Notify customer by SMS
- Create Fraud Investigation Case
""",
        )


# =====================================================================
# Fraud Investigation Agent
# =====================================================================

fraud_investigator = LlmAgent(
    name="FraudInvestigator",
    model="gemini-2.5-flash",
    description="Analyzes suspicious customer login attempts.",
    instruction="""
You are a Banking Fraud Analyst.

Review the customer's login attempt.

Look for:

- New device
- Foreign login
- VPN usage
- Multiple failed logins
- Large transaction
- Impossible travel

Determine whether the login is:

- LOW Risk
- MEDIUM Risk
- HIGH Risk

If HIGH risk, recommend invoking the
AccountProtectionAgent.
""",
)

# =====================================================================
# Banking Coordinator
# =====================================================================

coordinator = LlmAgent(
    name="BankingCoordinator",
    model="gemini-2.5-flash",
    description="Coordinates fraud investigation and account protection.",
    instruction="""
You are the banking orchestration agent.

Your responsibilities:

1. Delegate fraud analysis to FraudInvestigator.

2. If fraud risk is HIGH,
   delegate to AccountProtectionAgent.

3. Return a consolidated response to the user.
""",
    sub_agents=[
        fraud_investigator,
        AccountProtectionAgent(),
    ],
)

# =====================================================================
# Verify Hierarchy
# =====================================================================

assert fraud_investigator.parent_agent == coordinator

print("✅ Banking Agent hierarchy created successfully.")
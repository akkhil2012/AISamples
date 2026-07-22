# router_example.py
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- Specialist tools ---
def get_exchange_rate(currency: str) -> dict:
    """Returns the USD exchange rate for a currency code."""
    rates = {"INR": 87.2, "EUR": 0.92, "JPY": 148.5}
    rate = rates.get(currency.upper())
    return {"status": "success", "usd_rate": rate} if rate else {"status": "error"}

def check_loan_eligibility(income_usd: float) -> dict:
    """Checks loan eligibility based on annual income in USD."""
    return {"eligible": income_usd >= 30000, "max_loan": income_usd * 4}

# --- Specialist agents ---
forex_agent = Agent(
    name="forex_agent",
    model="gemini-2.0-flash",
    description="Handles currency exchange rates and forex questions.",
    instruction="Answer currency and exchange rate questions using your tool.",
    tools=[get_exchange_rate],
)

loan_agent = Agent(
    name="loan_agent",
    model="gemini-2.0-flash",
    description="Handles loan eligibility and lending questions.",
    instruction="Answer loan eligibility questions using your tool.",
    tools=[check_loan_eligibility],
)

general_agent = Agent(
    name="general_agent",
    model="gemini-2.0-flash",
    description="Handles greetings and anything not related to forex or loans.",
    instruction="Answer general questions politely. Keep it brief.",
)

# --- Router / coordinator ---
router = Agent(
    name="router",
    model="gemini-2.0-flash",
    instruction=(
        "You are a routing coordinator for a financial services app. "
        "Route currency/forex questions to forex_agent, "
        "loan questions to loan_agent, and everything else to general_agent. "
        "Do not answer questions yourself — always delegate."
    ),
    sub_agents=[forex_agent, loan_agent, general_agent],
)

# --- Runner ---
async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=router, app_name="fin_app", session_service=session_service)
    await session_service.create_session(
        app_name="fin_app", user_id="akhil", session_id="s1"
    )

    queries = [
        "What's the USD rate for INR?",
        "Am I eligible for a loan with $50,000 income?",
        "Hi, what can you help me with?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        content = types.Content(role="user", parts=[types.Part(text=q)])
        async for event in runner.run_async(
            user_id="akhil", session_id="s1", new_message=content
        ):
            # Watch the routing happen
            if event.author and event.is_final_response() and event.content:
                print(f"[{event.author}]: {event.content.parts[0].text}")

asyncio.run(main())

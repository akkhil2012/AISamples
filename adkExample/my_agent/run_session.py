# run_session.py
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- Tool ---
def get_exchange_rate(currency: str) -> dict:
    """Returns the USD exchange rate for a given currency code."""
    rates = {"INR": 87.2, "EUR": 0.92, "JPY": 148.5}
    rate = rates.get(currency.upper())
    if rate is None:
        return {"status": "error", "message": f"Unknown currency: {currency}"}
    return {"status": "success", "currency": currency.upper(), "usd_rate": rate}

# --- Agent ---
root_agent = Agent(
    name="exchange_agent",
    model="gemini-2.0-flash",
    instruction="Help users with currency exchange rates using your tool.",
    tools=[get_exchange_rate],
)

# --- Session + Runner setup ---
APP_NAME = "exchange_app"
USER_ID = "akhil"
SESSION_ID = "session_001"

session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

async def chat(query: str):
    content = types.Content(role="user", parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"Agent: {event.content.parts[0].text}")

async def main():
    # Session must be created before first use
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # Multi-turn: same session_id means the agent remembers context
    await chat("What's the USD rate for INR?")
    await chat("And how does that compare to JPY?")

    # Inspect what's stored in the session
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(f"\nEvents in session: {len(session.events)}")

if __name__ == "__main__":
    asyncio.run(main())

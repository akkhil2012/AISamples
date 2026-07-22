"""
Demo 2: Google ADK + InMemorySessionService — session-based memory
==================================================================
The ADK Runner + SessionService pattern automatically:
  1. Stores every event (user msg, agent reply) in the session
  2. Replays that history to the LLM on each new turn
  3. Keeps sessions isolated per (app_name, user_id, session_id)

Turn 1: "My name is Akki, I live in Hyderabad"
Turn 2: "What is my name and where do I live?"  --> Model remembers!

Requires:
    pip install google-adk litellm
    export OPENROUTER_API_KEY="your-key"
"""

import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "session_memory_demo"
USER_ID = "akki"
SESSION_ID = "session_001"
MODEL = LiteLlm(model="openrouter/openai/gpt-4o-mini")

os.environ.setdefault("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------
# 1. Define the agent
# ---------------------------------------------------------------
agent = LlmAgent(
    name="memory_agent",
    model=MODEL,
    instruction=(
        "You are a helpful assistant. Answer using details the user "
        "shared earlier in this conversation whenever relevant."
    ),
)

# ---------------------------------------------------------------
# 2. Create the session service + runner
#    InMemorySessionService = dict-backed storage, lives for the
#    lifetime of the Python process (perfect for demos/tests).
#    Swap for DatabaseSessionService / VertexAiSessionService in prod.
# ---------------------------------------------------------------
session_service = InMemorySessionService()

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,   # <-- THE key difference vs demo 1
)


async def talk(user_text: str) -> str:
    """Send one user turn through the Runner.

    The Runner automatically:
      - loads session history from InMemorySessionService
      - appends it to the LLM request
      - saves the new user + agent events back into the session
    """
    message = types.Content(role="user", parts=[types.Part(text=user_text)])

    final_reply = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_reply = event.content.parts[0].text
    return final_reply


async def main():
    # Sessions must be created explicitly before use
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    print("=" * 60)
    print("DEMO: GOOGLE ADK WITH InMemorySessionService (stateful)")
    print("=" * 60)

    # ---- Turn 1 ----
    turn_1 = "Hi! My name is Akki and I live in Hyderabad."
    print(f"\n[USER  - Turn 1]: {turn_1}")
    print(f"[AGENT - Turn 1]: {await talk(turn_1)}")

    # ---- Turn 2: SAME session_id --> history is replayed ----
    turn_2 = "What is my name and which city do I live in?"
    print(f"\n[USER  - Turn 2]: {turn_2}")
    print(f"[AGENT - Turn 2]: {await talk(turn_2)}")

    # ---- Peek inside the session to prove events are stored ----
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\n" + "=" * 60)
    print(f"OBSERVATION: Agent remembered! Session holds "
          f"{len(session.events)} stored events:")
    for i, event in enumerate(session.events, 1):
        text = ""
        if event.content and event.content.parts and event.content.parts[0].text:
            text = event.content.parts[0].text[:60].replace("\n", " ")
        print(f"  {i}. [{event.author}] {text}...")
    print("=" * 60)
    print("The Runner + SessionService did the history-stitching that")
    print("we had to do MANUALLY in script 01. Same LLM, same API —")
    print("the 'memory' lives entirely in the session layer.")


if __name__ == "__main__":
    asyncio.run(main())
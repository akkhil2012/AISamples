import asyncio

from dotenv import load_dotenv

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from FraudAttempt import root_agent

load_dotenv()

APP_NAME = "banking_demo"
USER_ID = "user123"
SESSION_ID = "session001"


async def main():

    # Create session service
    session_service = InMemorySessionService()

    # Create a session (must be awaited)
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # Create runner
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    # Inline prompt
    prompt = """
Customer ID: 123456

Login Attempt

Registered Country: India

Current Country: Russia

New Device: Yes

VPN: Yes

Failed Login Attempts: 6

Transaction:
Transfer ₹4,80,000 to a newly added beneficiary.
"""

    user_message = Content(
        role="user",
        parts=[Part(text=prompt)],
    )

    print("\nRunning Banking Fraud Pipeline...\n")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
import os
import json
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4.1-mini"


##############################################################
# Generic OpenRouter Async Client
##############################################################

class OpenRouterClient:

    def __init__(self):

        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    async def chat(self, system_prompt, user_prompt):

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=120) as client:

            response = await client.post(
                OPENROUTER_URL,
                headers=self.headers,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]


##############################################################
# Base Agent
##############################################################

class Agent:

    def __init__(self, name, system_prompt):

        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenRouterClient()

    async def execute(self, payload):

        response = await self.client.chat(

            self.system_prompt,

            json.dumps(payload, indent=2)

        )

        return {

            "agent": self.name,

            "response": response

        }


##############################################################
# Intent Agent
##############################################################

intent_agent = Agent(

    "IntentAgent",

    """
You are an Intent Classification Agent.

Return:

- user intent
- task type
- complexity

Be concise.
"""

)

##############################################################
# Research Agent
##############################################################

research_agent = Agent(

    "ResearchAgent",

    """
You are a Research Agent.

Read the user request.

Return useful background information,
important concepts,
and possible solution directions.

Be concise.
"""

)


##############################################################
# Orchestrator
##############################################################

class Orchestrator:

    async def execute(self, user_prompt):

        payload = {

            "user_prompt": user_prompt,

            "request_id": "12345",

            "metadata": {

                "application": "AgentDemo",

                "version": "1.0"

            }

        }

        print("\nSending JSON Payload\n")

        print(json.dumps(payload, indent=4))

        print("\nRunning agents asynchronously...\n")

        intent_task = asyncio.create_task(

            intent_agent.execute(payload)

        )

        research_task = asyncio.create_task(

            research_agent.execute(payload)

        )

        results = await asyncio.gather(

            intent_task,

            research_task

        )

        final = {

            "request": payload,

            "agent_results": results

        }

        return final


##############################################################
# Main
##############################################################

async def main():

    orchestrator = Orchestrator()

    while True:

        prompt = input("\nUser > ")

        if prompt.lower() == "exit":

            break

        result = await orchestrator.execute(prompt)

        print("\n" + "=" * 60)

        print("FINAL JSON")

        print("=" * 60)

        print(json.dumps(result, indent=4))


if __name__ == "__main__":

    asyncio.run(main())
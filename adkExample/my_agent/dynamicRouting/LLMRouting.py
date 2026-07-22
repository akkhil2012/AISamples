import os
import re
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -----------------------------
# Models available on OpenRouter
# -----------------------------
MODELS = {
    "FAST": "google/gemma-3-4b-it",
    "REASONING": "deepseek/deepseek-r1",
    "CODING": "qwen/qwen3-coder",
    "VISION": "google/gemma-3-27b-it",
    "GENERAL": "openai/gpt-4.1-mini",
}


class Route(Enum):
    FAST = "FAST"
    REASONING = "REASONING"
    CODING = "CODING"
    VISION = "VISION"
    GENERAL = "GENERAL"


def detect_route(prompt: str) -> Route:
    """
    Simple rule-based router.
    Replace this later with an intent-classification LLM if desired.
    """

    prompt_lower = prompt.lower()

    coding_keywords = [
        "python",
        "java",
        "kotlin",
        "bug",
        "code",
        "compile",
        "function",
        "algorithm",
        "sql",
        "javascript",
        "c++",
        "android",
        "exception",
        "stacktrace",
    ]

    reasoning_keywords = [
        "analyze",
        "compare",
        "reason",
        "why",
        "architecture",
        "design",
        "tradeoff",
        "explain deeply",
        "pros and cons",
    ]

    vision_keywords = [
        "image",
        "photo",
        "picture",
        "diagram",
        "identify",
        "ocr",
    ]

    if any(k in prompt_lower for k in coding_keywords):
        return Route.CODING

    if any(k in prompt_lower for k in reasoning_keywords):
        return Route.REASONING

    if any(k in prompt_lower for k in vision_keywords):
        return Route.VISION

    if len(prompt.split()) < 15:
        return Route.FAST

    return Route.GENERAL


def call_model(model: str, prompt: str):
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def main():

    print("=" * 60)
    print("Dynamic LLM Router using OpenRouter")
    print("=" * 60)

    while True:

        prompt = input("\nYou: ")

        if prompt.lower() in ["exit", "quit"]:
            break

        route = detect_route(prompt)

        model = MODELS[route.value]

        print(f"\nSelected Route : {route.value}")
        print(f"Selected Model : {model}")
        print("-" * 60)

        answer = call_model(model, prompt)

        print(answer)


if __name__ == "__main__":
    main()
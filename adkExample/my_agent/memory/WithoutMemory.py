"""
Demo 1: LLMs are STATELESS — no session storage
================================================
Each API call is completely independent. The model has NO memory
of previous calls unless YOU manually resend the history.

Turn 1: "My name is Akki, I live in Hyderabad"
Turn 2: "What is my name and where do I live?"  --> Model has no idea.

Requires:
    pip install openai
    export OPENROUTER_API_KEY="your-key"
"""

import os
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "openai/gpt-4o-mini"


def ask(prompt: str) -> str:
    """Each call = a brand-new request. Nothing is carried over."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def main():
    print("=" * 60)
    print("DEMO: LLM WITHOUT SESSION STORAGE (stateless)")
    print("=" * 60)

    # ---- Turn 1: give the model some personal context ----
    turn_1 = "Hi! My name is Akki and I live in Hyderabad."
    print(f"\n[USER - Turn 1]: {turn_1}")
    print(f"[LLM  - Turn 1]: {ask(turn_1)}")

    # ---- Turn 2: separate API call — context is GONE ----
    turn_2 = "What is my name and which city do I live in?"
    print(f"\n[USER - Turn 2]: {turn_2}")
    print(f"[LLM  - Turn 2]: {ask(turn_2)}")

    print("\n" + "=" * 60)
    print("OBSERVATION: The model cannot answer Turn 2.")
    print("Why? HTTP APIs to LLMs are stateless. The server does not")
    print("store your conversation. 'Memory' in chatbots is an illusion")
    print("created by the CLIENT resending the full history every turn.")
    print("=" * 60)

    # ---- Bonus: the manual workaround (what chat apps do) ----
    print("\nBONUS: Manual fix — resend history yourself:")
    stitched = (
        f"Previous conversation:\nUser: {turn_1}\n\n"
        f"New question: {turn_2}"
    )
    print(f"[LLM with manual history]: {ask(stitched)}")
    print("\nThis manual stitching is exactly what a SessionService")
    print("automates for you (see script 02).")


if __name__ == "__main__":
    main()
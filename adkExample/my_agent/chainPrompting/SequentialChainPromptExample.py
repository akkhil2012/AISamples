
#Original Prompt
# Intent Extraction
# Requirement Analysis
# Context Enrichment
# Prompt Optimization
# Final Answer Generation

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4.1-mini"


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def run_chain(user_prompt):

    stages = []

    print("=" * 70)
    print("Original Prompt")
    print("=" * 70)
    print(user_prompt)

    # ----------------------------------------------------
    # Stage 1
    # ----------------------------------------------------

    stage1 = call_llm(f"""
Identify the user's intent.

Return:
- Primary Goal
- Domain
- Expected Output

Prompt:

{user_prompt}
""")

    stages.append(("Intent Identification", stage1))

    # ----------------------------------------------------
    # Stage 2
    # ----------------------------------------------------

    stage2 = call_llm(f"""
Using the intent below, identify all requirements.

Intent

{stage1}

Original Prompt

{user_prompt}

Return:

- Functional requirements
- Non-functional requirements
- Missing information
""")

    stages.append(("Requirement Analysis", stage2))

    # ----------------------------------------------------
    # Stage 3
    # ----------------------------------------------------

    stage3 = call_llm(f"""
Improve the original prompt.

Intent

{stage1}

Requirements

{stage2}

Original Prompt

{user_prompt}

Produce an optimized prompt that is
clear,
specific,
and complete.
""")

    stages.append(("Prompt Improvement", stage3))

    # ----------------------------------------------------
    # Stage 4
    # ----------------------------------------------------

    stage4 = call_llm(f"""
Answer this optimized prompt.

{stage3}
""")

    stages.append(("Final Response", stage4))

    print()

    for title, content in stages:

        print("=" * 70)

        print(title)

        print("=" * 70)

        print(content)

        print()


def main():

    while True:

        prompt = input("\nEnter Prompt (exit to quit): ")

        if prompt.lower() == "exit":
            break

        run_chain(prompt)


if __name__ == "__main__":
    main()

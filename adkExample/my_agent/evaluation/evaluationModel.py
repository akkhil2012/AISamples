import json
import os
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Candidate model under evaluation
CANDIDATE_MODEL = "openai/gpt-4.1-mini"

# Judge model
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"


def call_model(model, prompt):
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def generate_answer(question):

    prompt = f"""
Answer the following question accurately.

Question:
{question}
"""

    return call_model(CANDIDATE_MODEL, prompt)


def evaluate(question, reference, answer):

    judge_prompt = f"""
You are an expert AI evaluator.

Question:
{question}

Expected Answer:
{reference}

Candidate Answer:
{answer}

Evaluate the candidate.

Return ONLY valid JSON.

{{
"correctness":0-10,
"completeness":0-10,
"reasoning":0-10,
"clarity":0-10,
"hallucination":0-10,
"overall":0-10,
"feedback":"..."
}}
"""

    result = call_model(JUDGE_MODEL, judge_prompt)

    return json.loads(result)


def main():

    with open("dataset.json") as f:
        dataset = json.load(f)

    report = []

    for sample in dataset:

        print(f"Evaluating {sample['id']}")

        answer = generate_answer(sample["question"])

        score = evaluate(
            sample["question"],
            sample["reference_answer"],
            answer
        )

        score["question"] = sample["question"]
        score["answer"] = answer

        report.append(score)

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print()

    print("=" * 60)

    print("Average Scores")

    metrics = [
        "correctness",
        "completeness",
        "reasoning",
        "clarity",
        "hallucination",
        "overall",
    ]

    for metric in metrics:
        avg = mean([r[metric] for r in report])
        print(f"{metric:15} : {avg:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
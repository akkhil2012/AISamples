# reflection_example.py
# pip install langchain langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # uses GOOGLE_API_KEY
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_retries=3,
)


# --- Generator ---
generator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical writer. Write a concise answer to the user's request. "
     "If critique is provided, revise your previous draft to address every point."),
    ("user",
     "Request: {request}\n\n"
     "Previous draft (empty if first attempt):\n{draft}\n\n"
     "Critique (empty if first attempt):\n{critique}"),
])
generator = generator_prompt | llm

# --- Critic / Reflector ---
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict reviewer. Evaluate the draft for accuracy, clarity, "
     "and completeness. List specific, actionable issues. "
     "If the draft is good enough, respond with exactly: APPROVED"),
    ("user", "Request: {request}\n\nDraft:\n{draft}"),
])
critic = critic_prompt | llm

# --- Reflection loop ---
def reflect(request: str, max_iterations: int = 3) -> str:
    draft, critique = "", ""

    for i in range(max_iterations):
        draft = generator.invoke({
            "request": request, "draft": draft, "critique": critique
        }).content
        print(f"\n--- Draft {i+1} ---\n{draft[:300]}...")

        critique = critic.invoke({"request": request, "draft": draft}).content
        print(f"\n--- Critique {i+1} ---\n{critique[:300]}")

        if critique.strip().startswith("APPROVED"):
            print(f"\n✅ Approved after {i+1} iteration(s)")
            return draft

    print(f"\n⚠️ Max iterations reached, returning last draft")
    return draft

if __name__ == "__main__":
    final = reflect(
        "Explain the purpose of taking ibrufin and dolo 650 in under 150 words."
    )
    print(f"\n=== FINAL ===\n{final}")

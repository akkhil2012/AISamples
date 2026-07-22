# Embedding-Based Semantic Routing — Banking Fraud Triage (OpenRouter edition)
#
# What changed vs. the Gemini version:
#   1. EMBEDDINGS: google-genai client  ->  OpenAI SDK pointed at OpenRouter
#      (https://openrouter.ai/api/v1/embeddings, OpenAI-compatible schema).
#      Default model: openai/text-embedding-3-small (cheap, solid English
#      performance). Swap to qwen/qwen3-embedding-8b etc. by changing one
#      constant — no code changes, that's the point of OpenRouter.
#   2. AGENT LLMs: ADK's native gemini-2.0-flash  ->  ADK's LiteLlm wrapper
#      with "openrouter/<provider>/<model>" strings. One env var
#      (OPENROUTER_API_KEY) now powers both routing and execution.
#
# Everything else (fail-safe fraud bias, threshold gate, audit log) is
# unchanged — the governance layer is provider-agnostic by design.
#
# Setup:
#   pip install google-adk litellm openai numpy
#   export OPENROUTER_API_KEY="sk-or-v1-..."
#   python fraud_semantic_routing_openrouter.py

import os
import uuid
import json
import asyncio
import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI                      # OpenAI SDK, repointed at OpenRouter
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # ADK's multi-provider bridge
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types                  # still used for message Content types

# ---------------------------------------------------------------
# 0. OpenRouter configuration — single key for everything
# ---------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]  # fail fast if missing

# Embedding model for ROUTING (see https://openrouter.ai/models?output_modalities=embeddings)
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Chat model for the SPECIALIST AGENTS. LiteLLM syntax: "openrouter/<provider>/<model>"
# Cheap + fast is fine here; the agents just call tools.
AGENT_MODEL = LiteLlm(model="openrouter/openai/gpt-4o-mini")

# One OpenAI-compatible client for embeddings, pointed at OpenRouter
embed_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# ---------------------------------------------------------------
# 1. Tool functions — simulate specialist banking actions
# ---------------------------------------------------------------

def fraud_report_handler(request: str) -> str:
    """Handles suspected fraud: blocks card, opens fraud case, alerts customer."""
    print("-------- FRAUD REPORT Handler Called (time-critical path) --------")
    case_id = f"FR-{uuid.uuid4().hex[:8].upper()}"
    return (f"Fraud case {case_id} opened for: '{request}'. "
            f"Simulated: card blocked, transactions frozen, "
            f"fraud-ops team notified, customer callback scheduled.")


def dispute_handler(request: str) -> str:
    """Handles transaction disputes and chargeback requests."""
    print("-------- Dispute Handler Called --------")
    dispute_id = f"DP-{uuid.uuid4().hex[:8].upper()}"
    return (f"Dispute {dispute_id} filed for: '{request}'. "
            f"Simulated: provisional credit assessed, merchant contacted, "
            f"resolution window 7-10 business days.")


def security_handler(request: str) -> str:
    """Handles phishing reports, credential resets, and account security."""
    print("-------- Security Handler Called --------")
    return (f"Security action for: '{request}'. "
            f"Simulated: sessions revoked, password reset link issued, "
            f"phishing sample forwarded to security team.")


def info_handler(request: str) -> str:
    """Handles general banking information requests."""
    print("-------- Info Handler Called --------")
    return (f"Information request: '{request}'. "
            f"Simulated: answer retrieved from banking knowledge base.")


def unclear_handler(request: str) -> str:
    """Fallback when no route clears the confidence threshold."""
    print("-------- Unclear Handler Called (below threshold) --------")
    return (f"Could not confidently route: '{request}'. "
            f"Escalated to human agent queue for manual triage.")


fraud_tool = FunctionTool(fraud_report_handler)
dispute_tool = FunctionTool(dispute_handler)
security_tool = FunctionTool(security_handler)
info_tool = FunctionTool(info_handler)

# ---------------------------------------------------------------
# 2. Specialist agents — same agents, OpenRouter-served models
# ---------------------------------------------------------------

fraud_agent = Agent(
    name="FraudReport",
    model=AGENT_MODEL,
    instruction=(
        "You are a fraud-response specialist. The customer is reporting "
        "suspected fraud. Treat every request as time-critical. "
        "Always call the fraud tool immediately. Be calm and reassuring."
    ),
    description="Handles suspected fraud: unauthorized transactions, stolen cards, account takeover.",
    tools=[fraud_tool],
)

dispute_agent = Agent(
    name="DisputeAgent",
    model=AGENT_MODEL,
    instruction=(
        "You are a dispute-resolution specialist for transaction disputes "
        "and chargebacks (merchant issues, double charges, refunds). "
        "Always call the dispute tool."
    ),
    description="Handles transaction disputes and chargebacks against merchants.",
    tools=[dispute_tool],
)

security_agent = Agent(
    name="SecurityAgent",
    model=AGENT_MODEL,
    instruction=(
        "You are an account-security specialist for phishing reports, "
        "suspicious messages, and credential concerns. "
        "Always call the security tool."
    ),
    description="Handles phishing, suspicious communications, and credential security.",
    tools=[security_tool],
)

info_agent = Agent(
    name="InfoAgent",
    model=AGENT_MODEL,
    instruction=(
        "You answer general banking questions (fees, limits, products, "
        "branches). Always call the info tool."
    ),
    description="Answers general banking information questions.",
    tools=[info_tool],
)

# ---------------------------------------------------------------
# 3. Semantic Router with fraud-aware fail-safe bias + audit log
# ---------------------------------------------------------------

@dataclass
class Route:
    name: str
    agent: Agent
    exemplars: List[str]
    exemplar_embeddings: Optional[np.ndarray] = field(default=None, repr=False)


class FraudAwareSemanticRouter:
    """
    Embedding-based router with two fraud-specific policies:

    1. Threshold gate: best score < `threshold` -> fallback (human queue).
    2. Fail-safe margin: if the fraud route is within `fraud_margin` of the
       winning score (and itself clears the threshold), override to fraud.
       Asymmetric cost: false positive = minutes; false negative = money +
       regulatory exposure.

    Every decision is appended to an audit log (regulated-industry posture).

    NOTE on models: text-embedding-3-* has no query/document task-type split
    (unlike Gemini's asymmetric RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT), so
    exemplars and queries are embedded identically. Thresholds are NOT
    portable across embedding models — recalibrate when you swap models.
    """

    def __init__(self,
                 routes: List[Route],
                 threshold: float = 0.45,
                 fraud_route_name: str = "FraudReport",
                 fraud_margin: float = 0.05):
        self.routes = routes
        self.threshold = threshold
        self.fraud_route_name = fraud_route_name
        self.fraud_margin = fraud_margin
        self.audit_log: List[dict] = []
        self._precompute_exemplar_embeddings()

    # ---- embeddings (OpenRouter, OpenAI-compatible) ------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Batch-embed via OpenRouter and L2-normalize (dot == cosine)."""
        response = embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,                # batch input supported
            encoding_format="float",
        )
        # Preserve input order (API returns items with .index)
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors = np.array([d.embedding for d in ordered], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _precompute_exemplar_embeddings(self) -> None:
        for route in self.routes:
            route.exemplar_embeddings = self._embed(route.exemplars)
            print(f"[Router] {len(route.exemplars)} exemplars embedded "
                  f"for route '{route.name}' via {EMBEDDING_MODEL}")

    # ---- decision ----------------------------------------------

    def route(self, query: str) -> tuple[Optional[Route], dict]:
        query_vec = self._embed([query])[0]

        scores = {}
        for route in self.routes:
            sims = route.exemplar_embeddings @ query_vec
            scores[route.name] = float(np.max(sims))

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        fraud_score = scores.get(self.fraud_route_name, 0.0)

        decision = best_name
        override_reason = None

        # Policy 1: confidence gate
        if best_score < self.threshold:
            decision = None
            override_reason = f"below_threshold({best_score:.3f}<{self.threshold})"

        # Policy 2: fail-safe fraud bias on near-ties
        elif (best_name != self.fraud_route_name
              and fraud_score >= self.threshold
              and (best_score - fraud_score) <= self.fraud_margin):
            decision = self.fraud_route_name
            override_reason = (f"fraud_failsafe(margin="
                               f"{best_score - fraud_score:.3f}"
                               f"<={self.fraud_margin})")

        entry = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "query": query,
            "embedding_model": EMBEDDING_MODEL,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "threshold": self.threshold,
            "raw_winner": best_name,
            "final_decision": decision or "FALLBACK_HUMAN_QUEUE",
            "override_reason": override_reason,
        }
        self.audit_log.append(entry)

        if decision is None:
            return None, entry
        winner = next(r for r in self.routes if r.name == decision)
        return winner, entry


# ---------------------------------------------------------------
# 4. Route definitions — exemplars are the routing table
# ---------------------------------------------------------------

routes = [
    Route(
        name="FraudReport",
        agent=fraud_agent,
        exemplars=[
            "There's a transaction on my account I didn't make.",
            "My card was charged in another country, I never traveled there.",
            "Someone withdrew money from my account without my permission.",
            "I think my card has been stolen and used.",
            "There are purchases on my statement that aren't mine.",
            "My account shows a transfer I never authorized.",
            "I lost my card and now there are charges on it.",
            "Someone accessed my account and moved money out.",
        ],
    ),
    Route(
        name="DisputeAgent",
        agent=dispute_agent,
        exemplars=[
            "The merchant charged me twice for the same order.",
            "I cancelled my subscription but was still billed.",
            "I returned the product but haven't received my refund.",
            "The store charged me more than the listed price.",
            "I was billed for a service I never received.",
            "I want to dispute a charge from an online store.",
        ],
    ),
    Route(
        name="SecurityAgent",
        agent=security_agent,
        exemplars=[
            "I received a suspicious SMS asking for my OTP.",
            "I got an email claiming to be from the bank asking for my password.",
            "I think I clicked a phishing link, what should I do?",
            "Someone called pretending to be from the bank and asked for my PIN.",
            "I want to reset my password because I think it's compromised.",
            "I'm getting OTP messages for logins I didn't attempt.",
        ],
    ),
    Route(
        name="InfoAgent",
        agent=info_agent,
        exemplars=[
            "What is the daily ATM withdrawal limit?",
            "What are the charges for an international wire transfer?",
            "What's the interest rate on your savings account?",
            "Where is the nearest branch?",
            "How do I update my registered mobile number?",
            "What documents do I need to open a current account?",
        ],
    ),
]

# ---------------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------------

async def run_agent(runner: InMemoryRunner, request: str) -> str:
    user_id = "customer_001"
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    final_result = ""
    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=request)]),
    ):
        if event.is_final_response() and event.content:
            if getattr(event.content, "text", None):
                final_result = event.content.text
            elif event.content.parts:
                final_result = "".join(
                    p.text for p in event.content.parts if p.text
                )
            break
    return final_result


async def handle_request(router: FraudAwareSemanticRouter,
                         runners: dict[str, InMemoryRunner],
                         request: str) -> str:
    print(f"\n=== Incoming customer message: '{request}' ===")

    route, audit_entry = router.route(request)
    print(f"[Router] scores={audit_entry['scores']} "
          f"decision={audit_entry['final_decision']} "
          f"override={audit_entry['override_reason']}")

    if route is None:
        return unclear_handler(request)

    result = await run_agent(runners[route.name], request)
    print(f"Final Response: {result}")
    return result


async def main():
    print("--- Banking Fraud Triage: Semantic Routing via OpenRouter ---")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Agent model:     openrouter/openai/gpt-4o-mini\n")

    router = FraudAwareSemanticRouter(
        routes,
        threshold=0.45,          # calibrated for text-embedding-3-small — see notes
        fraud_route_name="FraudReport",
        fraud_margin=0.05,
    )

    runners = {
        "FraudReport": InMemoryRunner(fraud_agent),
        "DisputeAgent": InMemoryRunner(dispute_agent),
        "SecurityAgent": InMemoryRunner(security_agent),
        "InfoAgent": InMemoryRunner(info_agent),
    }

    test_messages = [
        "Act as a banking fraud triage assistant. Review each customer message carefully and determine the most appropriate handling path. Classify the complaint as one of the following: fraud report, dispute, security concern, or general information request. If the message indicates unauthorized transactions, suspicious charges, or possible account compromise, treat it as a high-priority fraud case and route it to the fraud team. If the message is unclear or lacks enough context, escalate it to a human agent for manual review. Respond with a clear routing decision, a brief explanation, and an appropriate next step"
    ]

    for msg in test_messages:
        await handle_request(router, runners, msg)

    print("\n=== ROUTING AUDIT LOG ===")
    print(json.dumps(router.audit_log, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
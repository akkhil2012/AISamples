import os
import asyncio

from google.adk.agents import (
    LlmAgent,
    ParallelAgent,
    SequentialAgent,
)
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ---------------------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------------------

# Ensure this is set before running the example:
#   export OPENROUTER_API_KEY="sk-or-v1-..."
os.environ.setdefault("OPENROUTER_API_KEY", "")

OPENROUTER_MODEL = LiteLlm(model="openrouter/openai/gpt-4o-mini")

# =====================================================================
# 1. Researcher Agents
# =====================================================================

# ---------------------------------------------------------------------
# Portfolio Strategy Researcher
# ---------------------------------------------------------------------

researcher_agent_1 = LlmAgent(
    name="PortfolioStrategyResearcher",
    model=OPENROUTER_MODEL,
    description="Researches asset allocation and portfolio strategy trends.",
    output_key="portfolio_strategy_result",
    instruction="""
You are an AI Research Assistant specializing in asset and wealth management.

Research the latest trends in portfolio strategy for high-net-worth individuals, including allocation across equities, fixed income, and alternative assets.

Summarize the key insights in 1–2 concise sentences.

Output only the summary.
""",
)

# ---------------------------------------------------------------------
# Risk & Diversification Researcher
# ---------------------------------------------------------------------

researcher_agent_2 = LlmAgent(
    name="RiskDiversificationResearcher",
    model=OPENROUTER_MODEL,
    description="Researches risk management and diversification strategies.",
    output_key="risk_diversification_result",
    instruction="""
You are an AI Research Assistant specializing in asset risk management.

Research current best practices for diversification and downside protection in wealth management portfolios.

Summarize the key insights in 1–2 concise sentences.

Output only the summary.
""",
)

# ---------------------------------------------------------------------
# Wealth Planning Researcher
# ---------------------------------------------------------------------

researcher_agent_3 = LlmAgent(
    name="WealthPlanningResearcher",
    model=OPENROUTER_MODEL,
    description="Researches wealth planning and client advisory trends.",
    output_key="wealth_planning_result",
    instruction="""
You are an AI Research Assistant specializing in wealth planning for asset management clients.

Research the current state of wealth planning advice, client goals alignment, and emerging services for affluent investors.

Summarize the key insights in 1–2 concise sentences.

Output only the summary.
""",
)

# =====================================================================
# 2. Parallel Research Agent
# =====================================================================

parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    description=(
        "Runs multiple research agents concurrently to gather "
        "information from different sustainability domains."
    ),
    sub_agents=[
        researcher_agent_1,
        researcher_agent_2,
        researcher_agent_3,
    ],
)

# =====================================================================
# 3. Research Synthesis Agent
# =====================================================================

merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=OPENROUTER_MODEL,
    description=(
        "Combines research findings from parallel agents into a "
        "single structured report grounded only on their outputs."
    ),
    instruction="""
You are an AI Assistant responsible for synthesizing multiple research
summaries into a single structured report.

Your response MUST be based exclusively on the input summaries provided
below.

Do NOT introduce any external knowledge, assumptions, or additional
facts.

----------------------------
Input Summaries
----------------------------

### Portfolio Strategy
{portfolio_strategy_result}

### Risk & Diversification
{risk_diversification_result}

### Wealth Planning
{wealth_planning_result}

----------------------------
Output Format
----------------------------

## Summary of Asset Wealth Management Insights

### Portfolio Strategy Findings
(Based on PortfolioStrategyResearcher's findings)

Summarize only the portfolio strategy input provided.

### Risk & Diversification Findings
(Based on RiskDiversificationResearcher's findings)

Summarize only the risk and diversification input provided.

### Wealth Planning Findings
(Based on WealthPlanningResearcher's findings)

Summarize only the wealth planning input provided.

### Overall Conclusion

Provide a brief 1–2 sentence conclusion connecting the findings above.

Output only the structured report.
""",
)

# =====================================================================
# 4. Sequential Pipeline
# =====================================================================

sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    description=(
        "Coordinates parallel research and produces a final "
        "synthesized report."
    ),
    sub_agents=[
        parallel_research_agent,
        merger_agent,
    ],
)

# =====================================================================
# Root Agent
# =====================================================================

root_agent = sequential_pipeline_agent

# =====================================================================
# Run example
# =====================================================================

async def main() -> None:
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="parallel_research_demo", session_service=session_service)

    await session_service.create_session(
        app_name="parallel_research_demo",
        user_id="user_1",
        session_id="session_1",
    )

    prompt = "Run the parallel sustainability research pipeline and provide the final structured report."
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_response = ""

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    print("\n=== Final Structured Report ===")
    print(final_response)

if __name__ == "__main__":
    asyncio.run(main())

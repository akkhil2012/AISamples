## What this script does

This file defines and runs an AI pipeline for asset wealth management research.

### Core behavior

1. Defines three specialized research agents:
   - `PortfolioStrategyResearcher`
     - focuses on asset allocation and portfolio strategy trends
   - `RiskDiversificationResearcher`
     - focuses on risk management and diversification best practices
   - `WealthPlanningResearcher`
     - focuses on wealth planning and client advisory trends

2. Runs those three agents in parallel using `ParallelAgent`
   - all three research agents execute concurrently
   - each produces a short summary in its own domain

3. Synthesizes the results with a follow-up agent
   - `SynthesisAgent` receives all three summaries
   - it creates one structured report with:
     - portfolio strategy findings
     - risk/diversification findings
     - wealth planning findings
     - an overall conclusion

4. Wraps the whole flow in a sequential pipeline
   - `SequentialAgent` ensures:
     - first run parallel research
     - then run the synthesis step

5. Provides a runnable entrypoint
   - `main()` creates an ADK `Runner`
   - sends a single prompt to the pipeline
   - prints the final structured report

### Technical details

- Uses OpenRouter via `LiteLlm(model="openrouter/openai/gpt-4o-mini")`
- Does not use any external Google Search tool
- Uses `InMemorySessionService` and `Runner` to execute the agent pipeline

### Summary

Functionally, it is an end-to-end AI workflow that:
- gathers three parallel asset-management insights,
- combines them into one executive-style report,
- and prints that report when run.

## run the client script thats is:
 # python run.py


## What this script does

Functionally, it defines a two-step fraud detection pipeline.

### 1. Fraud detection analysis
- `fraud_detector` is an `LlmAgent`
- It receives account access or login attempt details
- It evaluates fraud indicators like:
  - failed logins
  - unfamiliar country or VPN/TOR access
  - impossible travel
  - new device or unusual behavior
- It returns only a JSON object with:
  - `risk_score`
  - `risk_level`
  - `fraud_detected`
  - `reasons`

### 2. Decision engine
- `decision_agent` is another `LlmAgent`
- It reads the output from `fraud_analysis`
- It applies decision rules:
  - `LOW` → `ALLOW_LOGIN`
  - `MEDIUM` → `REQUIRE_MFA`
  - `HIGH` → `BLOCK_ACCOUNT`
- It returns only a JSON object with:
  - `decision`
  - `confidence`
  - `reason`
  - `recommended_actions`

### 3. Pipeline orchestration
- `banking_fraud_pipeline` is a `SequentialAgent`
- It runs `fraud_detector` first, then `decision_agent`
- `root_agent` is set to this pipeline

### 4. Model backend
- The script is configured to use an OpenRouter model:
  - `openrouter/openai/gpt-4o-mini`
- That means it expects `LiteLlm` support for provider-style OpenRouter models

### Summary
This script is a fraud triage workflow:
- first classify risk from an access attempt,
- then choose the appropriate banking response,
- with both steps expressed as agents in a sequential pipeline.
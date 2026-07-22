# PII Guardrail App — Google ADK + Android APK

An input guardrail that detects, redacts, or blocks PII **before it ever reaches an LLM**, enforced through Google ADK's callback system, with an Android app front-end where the detection engine runs fully on-device (zero cloud egress for raw PII).

## Architecture

```
┌─────────────── Android APK (Kivy) ────────────────┐
│  main.py                                          │
│    └── pii_guardrails/guardrail_engine.py         │
│        pure-Python detector (Verhoeff, Luhn,      │
│        regex) — runs 100% on-device               │
│              │ sanitized text only (optional)     │
└──────────────┼────────────────────────────────────┘
               ▼
┌─────────── Server / laptop (ADK) ─────────────────┐
│  pii_guardrails/adk_agent.py                      │
│    Agent(model="gemini-2.0-flash")                │
│      before_model_callback ── input guardrail     │
│      after_model_callback  ── output guardrail    │
└───────────────────────────────────────────────────┘
```

**Why this split:** `google-adk` depends on grpc/protobuf toolchains that don't cross-compile cleanly through python-for-android, and shipping cloud-LLM SDKs inside the APK defeats the data-minimization goal anyway. So the trust boundary is the device: the guardrail engine (stdlib-only) lives in the APK, and only sanitized text is optionally forwarded to the ADK agent. Defense in depth: the ADK agent runs the *same* guardrail again in its callbacks, so even direct API callers are covered.

## Detected entities

| Entity | Validation | Default action |
|---|---|---|
| Aadhaar | Verhoeff checksum | **BLOCK** |
| Credit/debit card | Luhn checksum | **BLOCK** |
| PAN, GSTIN, Passport, UPI ID | pattern | REDACT |
| Email, Indian phone, IFSC, IPv4, DOB | pattern | REDACT |

Policy is a plain dict — override per deployment:

```python
from pii_guardrails.guardrail_engine import PIIGuardrail, Action
guard = PIIGuardrail(policy={"EMAIL": Action.ALLOW}, redaction_style="hash")
```

## Run the ADK agent (desktop/server)

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-adk
export GOOGLE_API_KEY=your_key          # AI Studio key
adk web                                  # from the project root; pick pii_guardrails
# or headless REST server for the Android app to call:
adk api_server --host 0.0.0.0 --port 8000
```

Try: *"My PAN is ABCDE1234F, help me draft a tax email"* → the model receives `<PAN_REDACTED>`.
Try: *"My card is 4111 1111 1111 1111"* → the LLM call is skipped entirely; the guardrail answers.

## Run the app on desktop (quick test)

```bash
pip install kivy
python main.py
```

## Build the APK

Buildozer needs Linux (native, WSL2, or the Docker image below).

```bash
# 1. System deps (Ubuntu/WSL2)
sudo apt update && sudo apt install -y git zip unzip openjdk-17-jdk \
    python3-pip autoconf libtool pkg-config zlib1g-dev \
    libncurses5-dev libtinfo6 cmake libffi-dev libssl-dev

# 2. Buildozer
pip3 install --user buildozer cython==0.29.36

# 3. Build (first run downloads Android SDK/NDK — ~30 min)
cd pii-guardrail-app
buildozer android debug

# APK appears at:
#   bin/piiguardrail-1.0.0-arm64-v8a_armeabi-v7a-debug.apk

# 4. Install on a connected phone
buildozer android deploy run
```

No-setup alternative (Docker):

```bash
docker run --rm -v "$PWD":/home/user/hostcwd kivy/buildozer android debug
```

Release build for Play Store: `buildozer android release`, then sign with `apksigner` using your keystore.

## Connecting the APK to the ADK agent (optional)

1. Start the server: `adk api_server --host 0.0.0.0 --port 8000`
2. In `main.py`, set `ADK_SERVER_URL = "http://<server-ip>:8000/run"`
   (use `http://10.0.2.2:8000/run` from the Android emulator)
3. Rebuild. Blocked inputs are never forwarded; redacted ones are.

## Project layout

```
pii-guardrail-app/
├── main.py                        # Kivy UI → APK entry point
├── buildozer.spec                 # APK packaging config
├── requirements-server.txt        # google-adk (server side only)
└── pii_guardrails/
    ├── __init__.py
    ├── guardrail_engine.py        # detector/redactor (stdlib only)
    └── adk_agent.py               # ADK agent + before/after model callbacks
```

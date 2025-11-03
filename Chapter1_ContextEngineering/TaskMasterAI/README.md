## TaskMasterAI – Context and Memory Demos

Three progressively richer examples that extract tasks/entities from text, persist them in memory, and generate context-aware summaries.

### Files
- `simpleTaskManagerAI.py`: Minimal task list with a context summary string.
- `tasmasterSample.py`: Rich demo with Tasks, Entities, priority queue, and reports.
- `taskManagerClient.py`: Tiny client sending two messages to demonstrate memory.

### Requirements
- Python 3.8+
- No external dependencies

### Quick Start

Run the minimal example:
```bash
python simpleTaskManagerAI.py
```

Run the rich demo (recommended):
```bash
python tasmasterSample.py
```

Run the tiny client:
```bash
python taskManagerClient.py
```

### What Each Script Demonstrates

- `simpleTaskManagerAI.py`
  - Defines a `Task` dataclass
  - Stores tasks in-memory and produces a concise context summary
  - Prints a prompt-ready block combining the summary and a user query

- `tasmasterSample.py`
  - `Task` and `Entity` models with timestamps and attributes
  - Regex-based extraction from messages into structured tasks/entities
  - Persistent memory store with a priority queue and conversation history
  - Functions for: processing messages, generating context-aware responses, simulating context overflow, and producing a status report
  - `main()` runs a small banking workflow scenario end-to-end

- `taskManagerClient.py`
  - Instantiates `TaskmasterAI`
  - Sends two messages to show memory retention
  - Prints preserved context and the latest response

### Example Snippets

Create and run the AI (rich demo):
```python
from tasmasterSample import TaskmasterAI

ai = TaskmasterAI()
ai.process_message("I need to buy a new laptop")
response = ai.process_message("What's my budget for that?")
print(ai.simulate_context_overflow())
print(response)
```

Expected output (abridged):
```text
I've identified 1 new task(s) from your message:
- buy a new laptop

Current context summary:
- Total tracked tasks: ...
```

### Notes
- All data lives in-memory; restart resets state.
- Regex extraction is intentionally simple and for demo purposes.



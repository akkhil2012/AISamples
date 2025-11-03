## Molecular Context

Build structured prompt contexts from instructions and examples. Supports simple input-output and chain-of-thought styles.

### Files
- `molecules.py`: Exposes `create_molecular_context(...)` to assemble the context string.
- `clientMolecule.py`: Minimal client showing how to call the function.

### Install
No external dependencies. Requires Python 3.8+.

### Usage
```python
from molecules import create_molecular_context

context = create_molecular_context(
    instruction="Translate the following English sentences to French:",
    examples=[
        {"input": "Hello, how are you?", "output": "Bonjour, comment ça va?"},
        {"input": "What time is it?", "output": "Quelle heure est-il?"}
    ],
    new_input="Where is the nearest restaurant?"
)

print(context)
```

### Output shape (input-output)
```text
Translate the following English sentences to French:

Input: Hello, how are you?
Output: Bonjour, comment ça va?

Input: What time is it?
Output: Quelle heure est-il?

Input: Where is the nearest restaurant?
Output:
```

### Chain-of-Thought format
Pass `format_type="chain-of-thought"` and include `thinking` in each example.

```python
from molecules import create_molecular_context

context = create_molecular_context(
    instruction="Solve the following math problems step by step:",
    examples=[
        {
            "input": "What is 5 + 3?",
            "thinking": "I need to add 5 and 3 together.",
            "output": "8"
        },
        {
            "input": "What is 10 - 4?",
            "thinking": "I need to subtract 4 from 10.",
            "output": "6"
        }
    ],
    new_input="What is 7 * 6?",
    format_type="chain-of-thought"
)

print(context)
```

### Quick tests

- Run the built-in examples in `molecules.py`:
```bash
python molecules.py
```

- Run the minimal client:
```bash
python clientMolecule.py
```

Both commands will print the assembled context to stdout.



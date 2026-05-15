

import ollama


MODEL      = "llama3.2"   # swap to "mistral", "phi3", "gemma2", etc.
EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text
 
 
# ── 1. Basic chat completion ─────────────────────────────────
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain RAG in one sentence."}],
)
print(response["message"]["content"])
 
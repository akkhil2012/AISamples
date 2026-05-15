import ollama


MODEL      = "llama3.2"   # swap to "mistral", "phi3", "gemma2", etc.
EMBED_MODEL = "nomic-embed-text"  

for chunk in ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)

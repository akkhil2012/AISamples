import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"

response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Local Inference & Training:"}],
)
print(response["message"]["content"])
print("Prompt tokens :", response.get("prompt_eval_count"))
print("Response tokens:", response.get("eval_count"))

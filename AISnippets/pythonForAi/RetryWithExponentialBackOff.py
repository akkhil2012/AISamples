import time, random
from ollama import ResponseError
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
def call_with_retry(prompt, retries=4):
    for attempt in range(retries):
        try:
            return ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except ResponseError as e:
            wait = (2 ** attempt) + random.random()
            print(f"Error: {e}. Retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

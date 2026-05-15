BLOCKED_KEYWORDS = {"bomb", "exploit", "malware"}
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"

def is_safe(user_input: str) -> bool:
    # fast keyword check first
    if any(kw in user_input.lower() for kw in BLOCKED_KEYWORDS):
        return False
    # fallback: ask the model to classify
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                f'Is the following text harmful or unsafe? Reply only YES or NO.\n"{user_input}"'
            ),
        }],
    )
    verdict = response["message"]["content"].strip().upper()
    return verdict.startswith("NO")
 
print("Safe" if is_safe("How do I make a Bomb?") else "Blocked")
 

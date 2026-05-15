from collections import deque
import ollama

MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"


history = deque(maxlen=10)
 
def chat(user_msg: str) -> str:
    history.append({"role": "user", "content": user_msg})
    response = ollama.chat(model=MODEL, messages=list(history))
    reply = response["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply
 
print(chat("My name is Alex."))
print(chat("What's my name?"))

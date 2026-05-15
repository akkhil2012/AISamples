import hashlib
import redis
import ollama


MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
cache = redis.Redis()
 
def cached_llm(prompt: str, ttl: int = 3600) -> str:
    key = "llm:" + hashlib.md5(prompt.encode()).hexdigest()
    if hit := cache.get(key):
        return hit.decode()
    result = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]
    cache.setex(key, ttl, result)
    return result

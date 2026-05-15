import asyncio
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"

from ollama import AsyncClient
 
async_client = AsyncClient()
 
async def ask(question: str) -> str:
    r = await async_client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return r["message"]["content"]
 
async def main():
    questions = ["Capital of Japan?", "Capital of France?", "Capital of Brazil?"]
    answers = await asyncio.gather(*[ask(q) for q in questions])
    print(answers)
 
asyncio.run(main())

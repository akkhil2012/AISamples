from openai import OpenAI

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
oa_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
 
response = oa_client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is agentic AI?"}],
)
print(response.choices[0].message.content)

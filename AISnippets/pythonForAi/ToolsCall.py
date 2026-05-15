import json
 
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
 
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)
if response["message"].get("tool_calls"):
    call = response["message"]["tool_calls"][0]
    print(call["function"]["arguments"])  # {'city': 'Tokyo'}

import json
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"


def search_web(query: str) -> str: return f"Results for '{query}': ..."  # stub
def calculator(expr: str)  -> str: return str(eval(expr))
 
TOOLS = {"search_web": search_web, "calculator": calculator}
 
TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "search_web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
    }},
]
 
def run_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    while True:
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOL_DEFS)
        msg = resp["message"]
        if not msg.get("tool_calls"):
            return msg["content"]
        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn   = tc["function"]["name"]
            args = tc["function"]["arguments"]
            result = TOOLS[fn](**args)
            messages.append({"role": "tool", "content": result})
 

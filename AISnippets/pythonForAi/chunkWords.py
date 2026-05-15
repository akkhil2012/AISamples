# ...existing code...
import ollama


MODEL      = "llama3.2"   # swap to "mistral", "phi3", "gemma2", etc.
EMBED_MODEL = "nomic-embed-text"  
# ...existing code...
for chunk in ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
):
    text = chunk.get("message", {}).get("content", "")
    if not text:
        continue

    # accumulate partial text and emit complete words only
    try:
        buffer
    except NameError:
        buffer = ""
    buffer += text

    last_ws = max(buffer.rfind(" "), buffer.rfind("\n"), buffer.rfind("\t"))
    if last_ws != -1:
        emit_part = buffer[: last_ws + 1]
        buffer = buffer[last_ws + 1 :]
        for word in emit_part.split():
            print(word, end=" ", flush=True)

# flush any remaining partial word at the end
if 'buffer' in globals() and buffer:
    print(buffer, end="", flush=True)
# ...existing code...
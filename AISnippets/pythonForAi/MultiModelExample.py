import base64

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
def describe_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    response = ollama.chat(
        model="llava",   # ollama pull llava
        messages=[{
            "role": "user",
            "content": "Describe this image in one sentence.",
            "images": [b64],
        }],
    )
    return response["message"]["content"]
 
print(describe_image("/Users/akhil/Downloads/akkhil.png"))

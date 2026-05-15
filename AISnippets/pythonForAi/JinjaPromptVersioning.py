from jinja2 import Template

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
template = Template("""
You are a {{ role }}.
Answer the following question concisely:
{{ question }}
""")
 
prompt = template.render(role="senior data scientist", question="What is overfitting?")
response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
print(response["message"]["content"])

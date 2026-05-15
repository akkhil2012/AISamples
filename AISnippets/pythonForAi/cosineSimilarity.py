
# to do: Defin custom cosine score.

import numpy as np

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"

def embed(text: str) -> np.ndarray:
    return np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0])
 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
score = cosine_similarity(embed("cat"), embed("kitten"))
print(f"Similarity: {score:.3f}")

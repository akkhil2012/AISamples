import faiss
import numpy as np

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
def embed(text: str) -> np.ndarray:
    return np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0], dtype="float32")
 
docs = ["Paris is the capital of France.", "Python was created by Guido."]
vectors = np.stack([embed(d) for d in docs])
 
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
 
query = embed("Who made Python?")
_, idx = index.search(query.reshape(1, -1), k=1)
print(docs[idx[0][0]])

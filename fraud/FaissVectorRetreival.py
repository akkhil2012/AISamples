import faiss
import numpy as np
import ollama
 
EMBED_MODEL = "nomic-embed-text"
 
def embed(text: str) -> np.ndarray:
    return np.array(
        ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0],
        dtype="float32",
    )
 
docs = [
    "Card transaction declined due to exceeded credit limit.",
    "Fraud alert triggered for cross-border transaction.",
    "Account blocked after 3 failed PIN attempts.",
]
vectors = np.stack([embed(d) for d in docs])
index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine on normalized
faiss.normalize_L2(vectors)
index.add(vectors)
 
def retrieve(query: str, k: int = 2) -> list[str]:
    qv = embed(query).reshape(1, -1)
    faiss.normalize_L2(qv)
    _, idx = index.search(qv, k)
    return [docs[i] for i in idx[0]]
 
print(retrieve("why was my card blocked?"))

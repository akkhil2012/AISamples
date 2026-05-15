import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
class OllamaEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [
            ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]
            for text in input
        ]
 
chroma = chromadb.Client()
collection = chroma.create_collection("docs", embedding_function=OllamaEmbeddings())
 
collection.add(
    documents=["FastAPI is a modern Python web framework."],
    ids=["doc1"],
)
results = collection.query(query_texts=["best Python API framework"], n_results=1)
print(results["documents"][0][0])

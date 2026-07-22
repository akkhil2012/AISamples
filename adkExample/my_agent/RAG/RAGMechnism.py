import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-4.1-mini"
TOP_K = 3

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

embedder = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Knowledge Base
# -----------------------------

class KnowledgeBase:

    def __init__(self, filename):

        with open(filename, "r", encoding="utf-8") as f:
            self.documents = [
                x.strip()
                for x in f.read().split("\n\n")
                if x.strip()
            ]

        self.embeddings = embedder.encode(
            self.documents,
            normalize_embeddings=False
        )


# -----------------------------
# Similarity Functions
# -----------------------------

def cosine_similarity(query, docs):

    query = query / np.linalg.norm(query)

    docs = docs / np.linalg.norm(
        docs,
        axis=1,
        keepdims=True
    )

    return docs @ query


def dot_product(query, docs):

    return docs @ query


def euclidean_distance(query, docs):

    distance = np.linalg.norm(
        docs - query,
        axis=1
    )

    return -distance


# -----------------------------
# Retriever
# -----------------------------

class Retriever:

    def __init__(self, kb):

        self.kb = kb

    def retrieve(
        self,
        query,
        metric="cosine",
        top_k=3
    ):

        query_embedding = embedder.encode(query)

        if metric == "cosine":

            scores = cosine_similarity(
                query_embedding,
                self.kb.embeddings
            )

        elif metric == "dot":

            scores = dot_product(
                query_embedding,
                self.kb.embeddings
            )

        elif metric == "euclidean":

            scores = euclidean_distance(
                query_embedding,
                self.kb.embeddings
            )

        else:
            raise ValueError(
                "metric must be cosine, dot or euclidean"
            )

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            self.kb.documents[i]
            for i in indices
        ]


# -----------------------------
# OpenRouter LLM
# -----------------------------

def ask_llm(context, question):

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the supplied context.

Context:

{context}

Question:

{question}
"""

    response = client.chat.completions.create(

        model=LLM_MODEL,

        temperature=0.2,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Main
# -----------------------------

def main():

    kb = KnowledgeBase("knowledge.txt")

    retriever = Retriever(kb)

    print("=" * 60)
    print("RAG using OpenRouter")
    print("=" * 60)

    while True:

        question = input("\nQuestion (exit to quit): ")

        if question.lower() == "exit":
            break

        print("\nChoose Similarity Metric")

        print("1. Cosine")

        print("2. Dot Product")

        print("3. Euclidean")

        choice = input("> ")

        if choice == "1":
            metric = "cosine"

        elif choice == "2":
            metric = "dot"

        elif choice == "3":
            metric = "euclidean"

        else:
            metric = "cosine"

        docs = retriever.retrieve(
            question,
            metric=metric,
            top_k=TOP_K
        )

        context = "\n\n".join(docs)

        print("\nRetrieved Documents")
        print("-" * 50)

        for d in docs:
            print(d)
            print()

        answer = ask_llm(context, question)

        print("\nAnswer")
        print("-" * 50)
        print(answer)


if __name__ == "__main__":
    main()
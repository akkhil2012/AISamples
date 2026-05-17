# IBM / OpenText: LLM-based compliance and explainability

import ollama
from FaissVectorRetreival import retrieve
 
def rag_answer(question: str, context_docs: list[str]) -> str:
    context = "\n".join(f"- {d}" for d in context_docs)
    prompt = f"""You are a banking compliance assistant.
Use only the context below to answer the question.
 
Context:
{context}
 
Question: {question}
Answer:"""
    return ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]
 
docs = retrieve("why was my card blocked?")
print(rag_answer("Why might a card get blocked?", docs))

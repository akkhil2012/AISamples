from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
chain = (
    ChatPromptTemplate.from_template("Summarise this in one line: {text}")
    | ChatOllama(model=MODEL)
    | StrOutputParser()
)
print(chain.invoke({"text": "Large language models are trained on vast text corpora..."}))

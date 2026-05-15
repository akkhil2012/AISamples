import json
from pydantic import BaseModel
import ollama
MODEL      = "llama3.2"   # swap
EMBED_MODEL = "nomic-embed-text"
 
class Movie(BaseModel):
    title: str
    year: int
    genre: str
 
response = ollama.chat(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": "Suggest a sci-fi movie. Reply ONLY with valid JSON matching: {title, year, genre}",
    }],
    format="json",
)
movie = Movie(**json.loads(response["message"]["content"]))
print(movie.title, movie.year)
 

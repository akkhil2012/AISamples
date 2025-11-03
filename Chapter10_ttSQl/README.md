## Chapter 10 – Text-to-SQL (Gemini + SQLite + Streamlit)

Small demo that converts natural-language questions to SQL using Google Gemini, executes against a local SQLite DB, and shows results in a Streamlit UI.

### Files

- `sql.py` – Streamlit app:
  - Prompts Gemini to turn English questions into SQL for a `STUDENT` table
  - Executes the generated SQL on `student.db`
  - Renders result rows in the UI
- `sqlite.py` – One-time helper to create `student.db` with a `STUDENT(NAME, CLASS, SECTION, MARKS)` table and seed rows
- `student.db` – SQLite database (can be re-created with `sqlite.py`)
- `requirements.txt` – Python deps (Streamlit, google-generativeai, LangChain, FAISS, Chroma, etc.)

### Prerequisites

- Python 3.9+
- Google Gemini API key in env: `GOOGLE_API_KEY`

Set the key (macOS/zsh example):
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### Install

```bash
pip install -r requirements.txt
```

### Initialize the database (optional)

If `student.db` is missing or you want to reset it, run:
```bash
python sqlite.py
```

This creates the `STUDENT` table and inserts sample rows.

### Run the Streamlit app

```bash
streamlit run sql.py
```

Open the printed URL (usually `http://localhost:8501`).

### Usage

Type a natural-language question. Examples (aligned with the prompt in `sql.py`):
- "How many entries of records are present?"
- "Tell me all the students studying in Data Science class?"

The app will:
1) Ask Gemini to convert the question to SQL (no backticks/``` and no leading "sql")
2) Run that SQL against `student.db`
3) Display the rows

### Notes & Tips

- The prompt in `sql.py` is tuned for the `STUDENT` schema; adapt it if you change the schema.
- Untrusted SQL risk: since the LLM generates SQL, prefer read-only queries and keep a backup of your DB.
- You can expand the schema/seed data in `sqlite.py` to test more complex queries.



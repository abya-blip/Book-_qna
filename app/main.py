# app/main.py  – Selected‑book‑first, knowledge‑fallback

import os, re, pickle, sqlite3, faiss, uvicorn, httpx
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ───── Config ─────
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE, "..", "data", "books.db")
INDEX_FILE  = os.path.join(BASE, "..", "data", "faiss_index", "index.faiss")
META_FILE   = os.path.join(BASE, "..", "data", "faiss_index", "metadata.pkl")

MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 10
TOP_CTX     = 3
DIST_THRES  = 2.0          # relaxed threshold

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

PROMPT_STRICT = """
You are an assistant. Answer ONLY with information in the context.
If the context does not contain the answer, respond exactly:
"The answer is not available in the selected book."
"""

PROMPT_FALLBACK = """
You are an assistant. The context below did not help.
Answer from your own knowledge of the book, but start with:
"(General knowledge answer – not found in the provided book context.)"
Do not hallucinate unrelated details.
"""

# ───── FastAPI setup ─────
app = FastAPI(title="Book Q&A RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ───── Load FAISS / metadata ─────
faiss_index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    chunk_meta: List[dict] = pickle.load(f)

# ───── Embedding model ─────
embedder = SentenceTransformer(MODEL_NAME)

# ───── SQLite ─────
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

def get_title(book_id: int):
    row = conn.execute("SELECT title FROM books WHERE id=?", (book_id,)).fetchone()
    return row["title"] if row else "Unknown Title"

def clean(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

# ───── Pydantic ─────
class Ask(BaseModel):
    question: str
    book_id: Optional[int] = None

# ───── Core RAG search ─────
def search_book_ctx(question: str, book_id: int):
    vec = embedder.encode(question).reshape(1, -1)
    dist, idx = faiss_index.search(vec, TOP_K)
    chunks, src = [], []
    for d, i in zip(dist[0], idx[0]):
        if d > DIST_THRES:
            break
        meta = chunk_meta[i]
        if meta["book_id"] != book_id:
            continue
        text = clean(meta["text"])
        if not text:
            continue
        chunks.append(text)
        src.append(
            {"book_id": book_id, "title": get_title(book_id),
             "snippet": text[:200] + ("…" if len(text) > 200 else "")}
        )
        if len(chunks) >= TOP_CTX:
            break
    return chunks, src

def call_ollama(prompt: str):
    try:
        r = httpx.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print("LLM error:", e)
        return "Error generating answer from Mistral."

# ───── API routes ─────
@app.post("/query")
def query(req: Ask):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if req.book_id is None:
        raise HTTPException(400, "book_id is required in strict mode")

    ctx, sources = search_book_ctx(req.question, req.book_id)

    # Strict answer from context
    if ctx:
        prompt = f"""{PROMPT_STRICT}

Context:
\"\"\"
{'\n\n'.join(ctx)}
\"\"\"

Question: {req.question}
Answer:"""
        answer = call_ollama(prompt)
        return {"answer": answer, "sources": sources}

    # Fallback: let LLM answer but flag as general knowledge
    fallback_prompt = f"""{PROMPT_FALLBACK}

Book title: {get_title(req.book_id)}

Question: {req.question}
Answer:"""
    answer = call_ollama(fallback_prompt)
    return {"answer": answer, "sources": sources}

# ───── book list route (unchanged) ─────
@app.get("/books")
def list_books():
    rows = conn.execute("SELECT id, title, author FROM books").fetchall()
    return [dict(r) for r in rows]

# ───── Local run ─────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

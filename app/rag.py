# app/rag.py
import os, sqlite3, time, faiss, pickle, re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_PATH       = "data/books.db"
INDEX_DIR     = "data/faiss_index/"
INDEX_FILE    = os.path.join(INDEX_DIR, "index.faiss")
META_FILE     = os.path.join(INDEX_DIR, "metadata.pkl")
MODEL_NAME    = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
BATCH_SIZE    = 64

# ──────────────────────────────────────────────────────────────
def load_books():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, title, text FROM books").fetchall()
    conn.close()

    print(f"✅ Loaded {len(rows)} books from database.")
    return rows

# ──────────────────────────────────────────────────────────────
HEADER_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
CLEAN_LINES_RE = re.compile(
    r"^(?:release date:|produced by|digitized by|language:|character set encoding:)", re.I,
)

def strip_gutenberg(text: str) -> str:
    m = HEADER_RE.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER_RE.search(text)
    if m:
        text = text[: m.start()]
    return "\n".join(
        line for line in text.splitlines() if not CLEAN_LINES_RE.match(line)
    ).strip()

# ──────────────────────────────────────────────────────────────
def chunk_text(text, book_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return [{"book_id": book_id, "text": chunk} for chunk in splitter.split_text(text)]

def build_index(chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    print(f"🚀 Embedding on device: {model.device}")

    sample = model.encode(["sample"], convert_to_numpy=True)
    dim = sample.shape[1]
    index = faiss.IndexFlatL2(dim)

    meta_saved = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), unit="chunks", desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        vecs = model.encode([c["text"] for c in batch], convert_to_numpy=True)
        index.add(vecs)
        meta_saved.extend(batch)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta_saved, f)

    print(f"✅ FAISS index saved to {INDEX_FILE} ({len(meta_saved):,} chunks).")

# ──────────────────────────────────────────────────────────────
def main():
    books = load_books()
    if not books:
        print("⚠️ No books found in the database. Make sure db.py ran correctly.")
        return

    all_chunks = []
    print("📚 Processing books …")
    for row in books:
        clean_text = strip_gutenberg(row["text"])
        chunks = chunk_text(clean_text, row["id"])
        all_chunks.extend(chunks)
        print(f"  • {row['title']} → {len(chunks)} chunks")

    if not all_chunks:
        print("⚠️ No chunks were created. Check if the book texts are empty or improperly formatted.")
        return

    print(f"🔄 Total chunks: {len(all_chunks):,}")
    build_index(all_chunks)

if __name__ == "__main__":
    main()

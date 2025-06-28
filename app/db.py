# app/db.py
import sqlite3
import os

DB_PATH   = "data/books.db"
BOOKS_DIR = "books/"

# ──────────────────────────────────────────────────────────────
# 1.  Initialize / connect to SQLite
# ──────────────────────────────────────────────────────────────
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT,
            author    TEXT,
            language  TEXT,
            text      TEXT,
            filename  TEXT UNIQUE
        )
        """
    )
    conn.commit()
    conn.close()

# ──────────────────────────────────────────────────────────────
# 2.  Helpers
# ──────────────────────────────────────────────────────────────
def extract_metadata(text: str, fallback_title: str):
    """
    Look for 'Title:' and 'Author:' lines within the first 100 lines.
    Returns (title, author) with reasonable fallbacks.
    """
    title  = ""
    author = ""
    for line in text.splitlines()[:100]:
        low = line.lower()
        if low.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif low.startswith("author:"):
            author = line.split(":", 1)[1].strip()
        if title and author:
            break

    if not title:
        # Use cleaned-up filename if Title tag missing
        title = fallback_title.replace("_", " ").strip()

    return title, author

def insert_book(title, text, author, filename, language="en"):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO books (title, text, author, language, filename)
        VALUES (?, ?, ?, ?, ?)
        """,
        (title, text, author, language, filename),
    )
    conn.commit()
    conn.close()

# ──────────────────────────────────────────────────────────────
# 3.  Main import routine
# ──────────────────────────────────────────────────────────────
def import_txt_books():
    txt_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print("⚠️  No .txt files found in 'books/' directory.")
        return

    for fname in txt_files:
        path = os.path.join(BOOKS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            if len(text.strip()) < 500:
                print(f"⚠️  Skipping tiny file: {fname}")
                continue

            base_title = os.path.splitext(fname)[0]
            title, author = extract_metadata(text, base_title)
            insert_book(title, text, author, fname)
            print(f"✅ Imported: {title}  —  {author or 'Unknown Author'}")

        except Exception as e:
            print(f"❌ Error importing {fname}: {e}")

# ──────────────────────────────────────────────────────────────
# 4.  Run from CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    import_txt_books()
    print("🎉  Done. Books are now in data/books.db")

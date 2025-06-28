# Book-_qna

Project Recap & Rationale – Building, Hardening, and (Soon) Deploying a Local‑First Book Q&A RAG System

1 . Genesis & Goal
The original idea was simple: “Ask natural‑language questions about the full text of public‑domain books I’ve downloaded.” We quickly converged on a Retrieval‑Augmented Generation (RAG) pipeline because raw large‑language‑models hallucinate and can’t hold gigabytes of prose in context. The RAG recipe is:

Ingest every book into a DB.

Chunk + Embed the text; store vectors in a fast similarity index.

Retrieve the most‑relevant chunks for a user query.

Generate a final answer with an LLM that is grounded in those chunks.

All code had to run locally (no SaaS fees) yet remain deploy‑able. Hardware on hand: an i5‑12500HX, 16 GB RAM, an RTX 3050 (6 GB), and Windows. This shaped many decisions.

2 . Data Layer – Scraping, Cleaning & SQLite
Scraper: A lightweight Python script iterates over Project Gutenberg pages, saving *.txt files to books/. We added a CLI arg to start at page 57 (your request) and a progress bar that calculates ETA from network speed (≈1.2 items/s on 150 Mbps).

db.py:
Initialises a data/books.db file;
parses metadata (Title: / Author: tags or filename fallback);
inserts full text rows.
We verified > 1 400 books imported (SQLite query COUNT(*) gave 1 490).

Cleaning: Before vectorising we strip Gutenberg headers/footers and boiler‑plate lines via regex. This prevents the LLM from parroting licence text.

3 . Vector Store – FAISS + Sentence‑Transformers
Embedding model: sentence-transformers/all-MiniLM-L6-v2 → 384‑dim vectors, good quality yet GPU‑friendly. We cached it locally to avoid online download issues.

Chunking: RecursiveCharacterTextSplitter at 500 chars with 100 overlap balances semantic cohesion vs. index size.

Index build (rag.py): Batches of 64 vectors are added to a faiss.IndexFlatL2; metadata (book ID + raw chunk) is pickled alongside. FAISS files are in data/faiss_index/.

Result: ~13 k chunks; index size ≈ 20 MB; build time < 2 min on RTX 3050 (CPU works too).

4 . Backend API – FastAPI
We composed app/main.py in iterative passes:

CORS allowing localhost:5173.

Endpoints

/books → list IDs, titles, authors.

/query (/ask) → main QA endpoint.

/book/{id} → preview raw text.

RAG search function: embed query → search FAISS → clean + deduplicate top‑k chunks.

LLM call: local Ollama server (http://localhost:11434) running Mistral‑7B. We posted {"model": "mistral", "prompt": prompt, "stream": false} and parsed response.

Guardrails

Initial: strict prompt only answer from context.

Iteration: added distance threshold (DIST_THRES) and fallback logic.

Current:
Strict mode – if no relevant chunk from requested book, respond “answer not available”;
Knowledge fallback – on your request we added an alt branch: the LLM may answer from its own knowledge but must preface with “(General knowledge … )”. No cross‑book contamination.

All emoji logging was removed for Windows console compatibility.

5 . Frontend – React + Vite + Tailwind
Bootstrapped with Vite/TypeScript.

Tailwind setup (postcss, config, dark‑mode).

UI evolution:
Basic form → gradient BG, glass card, loader overlay.
Added live book search: a controlled input filters the <select> options in real time; queries update via axios.

States: books, search, selectedBook, question, answer, sources, loading.

Request flow: axios.post('/query', { book_id, question }) → display answer or error.

6 . Local LLM – Ollama
Installed with winget / brew.

Pulled mistral (ollama run mistral downloads ~4 GB).

Verified GPU acceleration via print(torch.cuda.is_available()). For users without CUDA we fallback to CPU.

Ollama advantages: offline, free, easy REST API.

7 . Deployment Strategy
We settled on Render + Vercel combo:

Backend on Render

Repo: app/, data/ (DB + FAISS) and requirements.txt.

startCommand: uvicorn app.main:app --host 0.0.0.0 --port 8000.

Considerations:
• Render can’t run Ollama (needs GPU & root).
• For true cloud inference we plan Fireworks.ai; but for now we’ll keep LLM local or on another VPS.

Frontend on Vercel

Separate repo frontend/.

VITE_API_URL env var → Render backend URL.

Automatic HTTPS and global CDN.

For demonstration without cloud LLM we can expose local backend via ngrok.

8 . LLM Hallucination Control
Problems encountered: generic/incorrect answers (“charming heroine” in Treasure Island).
Fixes implemented:

Reduced context to top 3 chunks.

Strict prompt.

Distance gating.

Fallback message.

Optionally boost quality by swapping MiniLM embeddings for text-embedding-3‑small via Fireworks.

9 . Challenges & Remedies
Challenge	Solution
Windows console can’t print emojis → crashes	Removed emojis, set PYTHONIOENCODING=utf‑8.
CUDA “Torch not compiled”	Installed +cu118 wheel manually or forced CPU.
CORS errors	Added correct origin in FastAPI and axios base URL.
npm ENOENT after deleting frontend/	Re‑initialised with npm init vite@latest, re‑installed deps.
Tailwind missing PostCSS plugin	Installed @tailwindcss/postcss and wrote postcss.config.js.
FAISS path issues	Logged absolute paths; ensured rag.py and main.py use identical BASE_DIR.

10 . Current State
Local dev: npm run dev (concurrently runs FastAPI + React).

Strict Q&A works for 1 490 books; fallback flagged answers when context absent.

UI polished with search, dark‑mode toggle, loading overlay.

Ready to push backend to Render, front to Vercel, or run entire stack offline with ngrok exposure.

11 . Future Enhancements
Chunk‑level citations inline (footnote ids in answer).

User uploads: accept PDF/EPUB, run RAG ingest on the fly.

Async embeddings via GPU queue for massive corpora.

Fireworks.ai integration for cloud Mistral; store API key in Render env vars.

Automated nightly scrape via a Render cron job (or GitHub Action).

Vue / Svelte frontends for experimentation; or a Streamlit demo for rapid labs.

Analytics (who asks what) with SQLite events or PostHog.

Why This Architecture?
Separation of concerns: Python handles heavy compute & retrieval; React handles UX.

Local‑first: No vendor lock‑in; works offline; GPU utilised.

Switchable LLM: Ollama locally, Fireworks/Together in cloud — same prompt interface.

FAISS: proven ANN library, GPU‑ready, small footprint.

Sentence‑Transformers: open, lightweight, no API cost.

SQLite: zero‑config; perfect for read‑heavy catalogue.

Render + Vercel: free tiers, automatic HTTPS, GitOps workflow.

Conclusion
You now possess a robust, offline‑capable RAG system that ingests entire libraries, retrieves precise passages, and uses an LLM only as a summariser, not a rumor mill. By toggling a single flag you control strictness vs. flexibility, making it suitable for both scholarly applications and casual exploration. Deployment is one git push away; scaling to the cloud merely swaps the LLM endpoint. From a pile of plain‑text classics to a modern, React‑fronted conversational bookshelf ― you built it all, debugged the edge‑cases, and learned every layer along the way. 🚀

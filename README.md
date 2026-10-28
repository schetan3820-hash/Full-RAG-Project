# 📚 Full RAG: Hybrid Search + Reranking + Citations + Evaluation

Ek complete Retrieval-Augmented Generation system. Isme PDF, DOCX, TXT, MD, CSV aur HTML files daalo, phir unse chat karo. Har jawab ke saath source aur page number milta hai.

```
                    ┌──────────── INDEXING (ek baar: python ingest.py) ────────────┐
 data/*.pdf ──> Load (page-wise) ──> Recursive Chunking ──> BGE Embeddings ──> FAISS + BM25
                    └────────────────────────────────────────────────────────────────┘

                    ┌──────────── QUERY (har sawal pe) ────────────────────────────────┐
 Question ─┬─> (chat history ho to) Query Rewrite (LLM)
           ├─> Vector search top-20 ─┐
           └─> BM25 search   top-20 ─┴─> RRF Fusion ─> Cross-Encoder Rerank ─> top-5
                                                                         │
                       Numbered context [1]..[5] + strict prompt ─> LLM ─> Answer + [n] citations
                    └────────────────────────────────────────────────────────────────┘
```

## Tutorial wale code se kya better hai

| Cheez | Tutorial (`src/`) | Ye project |
|---|---|---|
| Metadata (source, page) | ❌ lost | ✅ har chunk ke saath, citations milti hain |
| Search | Sirf vector | **Hybrid**: vector + BM25 + RRF |
| Reranking | ❌ | ✅ Cross-encoder |
| Embeddings | MiniLM, normalized nahi, L2 distance | BGE-small, normalized, cosine |
| Naya document | Poora index dobara banao | **Incremental**: sirf nayi/badli files process hoti hain |
| Follow-up questions | ❌ | ✅ Query rewriting + chat history |
| "Pata nahi" handling | ❌ | ✅ Irrelevant chunks filter hote hain, aur LLM ko "nahi mila" bolne ka rule diya gaya hai |
| LLM provider | Hardcoded Groq | Groq / OpenAI / Ollama, sirf `.env` badal ke |
| Interface | Script | CLI chat + REST API + Streamlit UI |
| Evaluation | Alag demo pe | **Apne hi RAG pe**: hit rate, MRR, 4 LLM-judge metrics |
| Tests | ❌ | ✅ 7 tests (bina internet/API ke chalte hain) |

## Setup

```bash
cd full_rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # phir .env mein apni Groq key daalo
```
Free Groq key: https://console.groq.com/keys

> Pehli baar chalane pe embedding model (~130MB) aur reranker (~90MB) HuggingFace se download honge. Uske baad sab local chalta hai.

## Use karna

```bash
# 1. Apni files data/ folder mein daalo (ek sample rag_notes.md pehle se hai)
python ingest.py                    # index banao (dobara chalao to sirf nayi/badli files process hongi)
python ingest.py --reset            # sab mita ke shuru se

# 2a. Terminal chat
python ask.py                       # interactive chat, memory ke saath
python ask.py "What is RRF?" --debug

# 2b. Web UI
streamlit run ui.py                 # http://localhost:8501

# 2c. REST API
uvicorn api:app --reload            # http://localhost:8000/docs
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "How does reranking work?"}'

# 3. Evaluation
python eval/evaluate.py                       # full (LLM judge)
python eval/evaluate.py --no-rerank --tag norerank   # A/B: reranker ka fayda naapo
python eval/evaluate.py --retrieval-only      # sirf hit-rate/MRR (free, koi LLM call nahi)

# Tests
python -m pytest tests -q
```

## Folder structure

```
full_rag/
├── rag/
│   ├── config.py        # saari settings (.env se)
│   ├── loaders.py       # STEP 1: PDF/DOCX/TXT/MD/CSV/HTML -> Document(text, {source, page})
│   ├── chunker.py       # STEP 2: recursive chunking + overlap
│   ├── embedder.py      # STEP 3: BGE embeddings (normalized)
│   ├── vector_store.py  # STEP 4: FAISS/numpy index, disk pe save, incremental add/delete
│   ├── bm25.py          # keyword search (khud implement kiya, koi library nahi)
│   ├── retriever.py     # STEP 5-6: hybrid search + RRF + rerank
│   ├── reranker.py      # cross-encoder
│   ├── llm.py           # STEP 7: OpenAI-compatible client (Groq/OpenAI/Ollama), streaming
│   ├── prompts.py       # saare prompts
│   └── pipeline.py      # sab ko jodne wali RAGPipeline class
├── ingest.py            # CLI: indexing
├── ask.py               # CLI: chat
├── api.py               # FastAPI server
├── ui.py                # Streamlit chat UI
├── eval/
│   ├── dataset.json     # test questions (apne documents ke hisaab se badlo)
│   └── evaluate.py
├── tests/test_rag.py
├── data/                # apni files yahan
└── storage/             # index (auto-create hota hai)
```

## Har design choice: kyun?

- **Chunk 800 chars, overlap 150:** Itna bada chunk ki ek idea poora aa jaaye, aur itna chhota ki retrieval precise rahe. Overlap ki wajah se boundary pe toota hua sentence kisi ek chunk mein poora mil jaata hai.
- **BGE-small:** Free hai, local chalta hai, 384-dim vectors deta hai, aur MiniLM se retrieval mein achha hai. Query ke aage ek instruction lagta hai (`config.py` mein hai).
- **Hybrid (BM25 + vector):** Vectors meaning pakadte hain, BM25 exact words ("GATE 2027", "LoRA", "Section 302"). Dono ka combo dono ki kamiyan cover karta hai.
- **RRF:** Dono searches ke scores alag scale pe hote hain. RRF sirf rank dekhta hai, isliye normalize karne ki zarurat nahi padti.
- **Cross-encoder rerank:** Query aur chunk ko saath padhke score deta hai, isliye zyada accurate hai. Slow hai, isliye sirf ~30 candidates pe chalta hai. Score `RAG_RERANK_MIN_SCORE` se kam ho to chunk drop ho jaata hai, aur sab drop ho jaayein to LLM call hi nahi hoti ("couldn't find").
- **Numbered context + citation rule:** LLM ko `[1]`, `[2]` cite karna padta hai, isliye user verify kar sakta hai aur hallucination kam hota hai.
- **Query rewriting:** "aur uski limitations?" jaisa follow-up search karne layak nahi hota. LLM history dekhke usse standalone query bana deta hai.
- **Incremental ingest:** Har file ka SHA-256 hash `manifest.json` mein save hota hai. Same file skip hoti hai, badli hui file replace hoti hai, aur hatayi gayi file index se bhi hat jaati hai.

## Apne data pe evaluation kaise karein

1. `eval/dataset.json` mein 20-50 questions likho, apne documents se:
   ```json
   {"question": "...", "reference_answer": "...", "expected_sources": ["notes.pdf"]}
   ```
2. `python eval/evaluate.py --tag baseline`
3. Ek cheez badlo (jaise `.env` mein `RAG_CHUNK_SIZE=500`), phir `python ingest.py --reset` aur `python eval/evaluate.py --tag chunk500`.
4. `eval/results/summary_*.json` compare karo. Interview mein yahi numbers bolo: *"reranker add karne se groundedness X% se Y% hua"*.

## Aage kya add kar sakte ho

- Scanned PDFs ke liye OCR (`pytesseract`)
- Metadata filters (date, subject), jaise GATE PYQs ke liye year/subject
- LangGraph se Corrective RAG: docs bekaar hon to query rewrite karke retry karo
- Docker + deploy

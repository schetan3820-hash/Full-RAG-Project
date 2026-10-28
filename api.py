"""
REST API (FastAPI).

  uvicorn api:app --reload --port 8000
  Docs (try-it-out UI):  http://localhost:8000/docs

Endpoints:
  GET  /health          -> status + kitne chunks
  GET  /sources         -> ingested files ki list
  POST /query           -> {"question": "...", "history": [...]}  -> answer + sources
  POST /query/stream    -> same, lekin Server-Sent Events (token-by-token)
  POST /upload          -> file upload karke turant ingest
  POST /ingest          -> data/ folder dobara scan karo
"""
import json
import shutil
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import RAGPipeline, settings
from rag.loaders import SUPPORTED_EXTENSIONS

app = FastAPI(title="RAG API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAGPipeline()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: List[Message] = []
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    sources: Optional[List[str]] = None  # sirf in files mein search karo


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag.store), "files": len(rag.store.manifest), "llm": settings.llm_model}


@app.get("/sources")
def sources():
    return rag.store.sources()


@app.post("/query")
def query(req: QueryRequest):
    if not len(rag.store):
        raise HTTPException(400, "Index khaali hai — pehle documents ingest karo.")
    try:
        res = rag.ask(req.question, [m.model_dump() for m in req.history], req.top_k, req.sources)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return res.to_dict()


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not len(rag.store):
        raise HTTPException(400, "Index khaali hai — pehle documents ingest karo.")

    def events():
        try:
            for ev in rag.ask_stream(req.question, [m.model_dump() for m in req.history], req.top_k, req.sources):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = f.filename.replace("/", "_").replace("\\", "_")
        if not any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            raise HTTPException(400, f"Unsupported file: {name}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")
        dest = settings.data_dir / name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    stats = {}
    for d in saved:
        s = rag.ingest(d, verbose=False)
        for k, v in s.items():
            if isinstance(v, list):
                stats.setdefault(k, []).extend(v)
            elif isinstance(v, dict):
                stats.setdefault(k, {}).update(v)
            elif k == "chunks_added":
                stats[k] = stats.get(k, 0) + v
            else:
                stats[k] = v
    return stats


@app.post("/ingest")
def ingest(force: bool = False):
    return rag.ingest(force=force, verbose=False)

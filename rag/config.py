"""
Saari settings ek jagah. Values .env file (ya environment variables) se aati hain,
warna yahan ke defaults use hote hain.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional hai
    pass


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass
class Settings:
    # ---------- Paths ----------
    data_dir: Path = field(default_factory=lambda: Path(_env("RAG_DATA_DIR", "data")))
    storage_dir: Path = field(default_factory=lambda: Path(_env("RAG_STORAGE_DIR", "storage")))

    # ---------- Chunking ----------
    chunk_size: int = field(default_factory=lambda: _env("RAG_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env("RAG_CHUNK_OVERLAP", 150))

    # ---------- Embeddings ----------
    embedding_model: str = field(default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    # BGE models query ke aage ye instruction chahte hain (documents ke aage nahi)
    # Khaali chhodna allowed hai (RAG_QUERY_INSTRUCTION=) — non-BGE models ke liye
    query_instruction: str = field(
        default_factory=lambda: os.getenv(
            "RAG_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
        )
    )

    # ---------- Retrieval ----------
    vector_top_k: int = field(default_factory=lambda: _env("RAG_VECTOR_TOP_K", 20))
    bm25_top_k: int = field(default_factory=lambda: _env("RAG_BM25_TOP_K", 20))
    rrf_k: int = field(default_factory=lambda: _env("RAG_RRF_K", 60))
    final_top_k: int = field(default_factory=lambda: _env("RAG_FINAL_TOP_K", 5))

    # ---------- Reranker ----------
    use_reranker: bool = field(default_factory=lambda: _env("RAG_USE_RERANKER", True))
    reranker_model: str = field(default_factory=lambda: _env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    # Cross-encoder logit isse kam ho to chunk ko irrelevant maan ke hata do
    rerank_min_score: float = field(default_factory=lambda: _env("RAG_RERANK_MIN_SCORE", -5.0))

    # ---------- LLM (koi bhi OpenAI-compatible API: Groq, OpenAI, Ollama...) ----------
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", _env("GROQ_API_KEY", "")))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile"))
    llm_temperature: float = field(default_factory=lambda: _env("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = field(default_factory=lambda: _env("LLM_MAX_TOKENS", 1024))

    # ---------- Chat memory ----------
    max_history_turns: int = field(default_factory=lambda: _env("RAG_MAX_HISTORY_TURNS", 6))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.storage_dir = Path(self.storage_dir)


settings = Settings()

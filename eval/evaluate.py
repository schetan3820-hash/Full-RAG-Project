"""
RAG Evaluation — apne RAG ko numbers se naapo.

  python eval/evaluate.py                          # eval/dataset.json pe chalao
  python eval/evaluate.py --no-rerank              # reranker band karke compare karo (A/B test)
  python eval/evaluate.py --top-k 3 --tag k3       # alag config, alag naam se save
  python eval/evaluate.py --retrieval-only         # sirf retrieval metrics (LLM call nahi, free)

Metrics
───────
Retrieval (bina LLM ke, sasta):
  hit_rate   : kya expected source file top-k mein aayi?          (1/0)
  mrr        : pehli sahi file kis rank pe aayi -> 1/rank          (1 = best)

LLM-as-a-judge (har ek True/False + explanation):
  correctness         : answer  vs reference answer   (reference chahiye)
  relevance           : answer  vs question           (jawab topic pe hai?)
  groundedness        : answer  vs retrieved docs     (hallucination to nahi?)
  retrieval_relevance : docs    vs question           (retriever ne sahi cheez uthayi?)

Diagnosis:
  retrieval_relevance ↓          -> retriever/chunking/embeddings theek karo
  retrieval ✓ lekin grounded ↓   -> prompt / LLM problem (docs ignore kar raha hai)
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RAGPipeline, Settings  # noqa: E402

HERE = Path(__file__).resolve().parent

JUDGE_SYSTEM = (
    "You are a strict, fair grader. Think step by step, then decide. "
    'Reply ONLY with JSON: {"explanation": "<reasoning>", "score": true or false}'
)

JUDGE_PROMPTS = {
    "correctness": """Grade the STUDENT ANSWER for factual accuracy against the GROUND TRUTH.
- score=true if it contains no statements that conflict with the ground truth and covers its key points.
- Extra correct information is OK.

QUESTION: {question}
GROUND TRUTH: {reference}
STUDENT ANSWER: {answer}""",
    "relevance": """Does the STUDENT ANSWER directly address and help answer the QUESTION, concisely?
score=true if yes.

QUESTION: {question}
STUDENT ANSWER: {answer}""",
    "groundedness": """Is every claim in the STUDENT ANSWER supported by the FACTS?
score=false if it contains any information not found in the FACTS (hallucination).
An answer that says the information was not found is grounded.

FACTS:
{context}

STUDENT ANSWER: {answer}""",
    "retrieval_relevance": """Are the FACTS relevant to the QUESTION?
score=true if they contain keywords or meaning that help answer the question (some unrelated text is OK).
score=false only if they are essentially unrelated.

QUESTION: {question}
FACTS:
{context}""",
}


def judge(llm, metric: str, **kw) -> tuple:
    prompt = JUDGE_PROMPTS[metric].format(**kw)
    for attempt in range(2):
        try:
            raw = llm.chat(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0,
                json_mode=True,
            )
            data = json.loads(raw)
            return bool(data.get("score")), str(data.get("explanation", ""))
        except Exception as e:  # bad JSON / rate limit -> ek retry
            err = e
            time.sleep(2)
    return None, f"judge error: {err}"


def retrieval_metrics(retrieved_sources, expected):
    if not expected:
        return None, None
    expected = set(expected)
    for rank, s in enumerate(retrieved_sources, start=1):
        if s in expected:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(HERE / "dataset.json"))
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    s = Settings()
    if args.no_rerank:
        s.use_reranker = False
    rag = RAGPipeline(settings=s, llm=False if args.retrieval_only else None)
    if not len(rag.store):
        sys.exit("Index khaali hai. Pehle `python ingest.py` chalao.")

    rows = []
    for i, ex in enumerate(dataset, start=1):
        q = ex["question"]
        print(f"[{i}/{len(dataset)}] {q}")
        if args.retrieval_only:
            _, hits, _ = rag.retrieve(q, top_k=args.top_k)
            answer, sources_text, srcs = "", "", [h.chunk.metadata.get("source") for h in hits]
            context = "\n\n".join(h.chunk.text for h in hits)
        else:
            res = rag.ask(q, top_k=args.top_k)
            answer = res.answer
            srcs = [x.source for x in res.sources]
            context = "\n\n".join(f"[{x.n}] {x.text}" for x in res.sources)

        hit, rr = retrieval_metrics(srcs, ex.get("expected_sources"))
        row = {"question": q, "answer": answer, "retrieved_sources": "; ".join(srcs), "hit_rate": hit, "mrr": rr}

        if not args.retrieval_only:
            kw = {"question": q, "answer": answer, "context": context or "(no documents retrieved)",
                  "reference": ex.get("reference_answer", "")}
            metrics = ["relevance", "groundedness", "retrieval_relevance"]
            if ex.get("reference_answer"):
                metrics.insert(0, "correctness")
            for m in metrics:
                score, why = judge(rag.llm, m, **kw)
                row[m] = score
                row[f"{m}_why"] = why
        rows.append(row)

    # ---------- summary ----------
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(float(v) for v in vals) / len(vals)) if vals else None

    keys = ["hit_rate", "mrr", "correctness", "relevance", "groundedness", "retrieval_relevance"]
    summary = {k: avg(k) for k in keys if avg(k) is not None}
    config = {"top_k": args.top_k or s.final_top_k, "reranker": s.use_reranker, "embedding": s.embedding_model,
              "llm": s.llm_model, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap}

    print("\n════════ RESULTS ════════")
    print("Config:", config)
    for k, v in summary.items():
        print(f"  {k:<20} {v:6.1%}" if k != "mrr" else f"  {k:<20} {v:6.3f}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    csv_path = out_dir / f"eval_{stamp}.csv"
    fieldnames = sorted({k for r in rows for k in r}, key=lambda k: (k != "question", k))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2))
    print(f"\nDetails: {csv_path}")


if __name__ == "__main__":
    main()

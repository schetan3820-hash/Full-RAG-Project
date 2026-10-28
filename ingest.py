"""
Documents ko index mein daalo.

  python ingest.py                 # data/ folder (incremental — sirf nayi/badli files)
  python ingest.py path/to/file.pdf
  python ingest.py --reset         # sab mita ke shuru se
  python ingest.py --force         # saari files dobara embed karo
"""
import argparse
import time

from rag import RAGPipeline, settings


def main():
    p = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    p.add_argument("path", nargs="?", default=None, help=f"file ya folder (default: {settings.data_dir})")
    p.add_argument("--reset", action="store_true", help="index mita ke shuru se banao")
    p.add_argument("--force", action="store_true", help="unchanged files ko bhi re-embed karo")
    args = p.parse_args()

    rag = RAGPipeline(llm=False)  # ingestion mein LLM ki zarurat nahi
    if args.reset:
        rag.reset()
        print("🗑️  Purana index delete ho gaya.")

    print(f"📂 Ingesting: {args.path or settings.data_dir}")
    t0 = time.perf_counter()
    stats = rag.ingest(args.path, force=args.force)
    dt = time.perf_counter() - t0

    print("\n──────── Summary ────────")
    print(f"Added   : {len(stats['added'])}   {stats['added'] or ''}")
    print(f"Updated : {len(stats['updated'])}   {stats['updated'] or ''}")
    print(f"Skipped : {len(stats['skipped'])}  (unchanged)")
    print(f"Removed : {len(stats['removed'])}   {stats['removed'] or ''}")
    if stats["failed"]:
        print(f"Failed  : {stats['failed']}")
    print(f"New chunks: {stats['chunks_added']}  |  Total chunks in index: {stats['total_chunks']}  |  {dt:.1f}s")


if __name__ == "__main__":
    main()

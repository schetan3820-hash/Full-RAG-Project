"""
Terminal mein chat karo apne documents se.

  python ask.py                       # interactive chat (memory ke saath)
  python ask.py "What is LoRA?"       # ek sawal
  python ask.py --debug               # retrieval scores bhi dikhao

Chat mein commands:  /reset  (history clear)   /exit
"""
import argparse
import sys

from rag import RAGPipeline


def print_sources(sources, debug=False):
    if not sources:
        return
    print("\n📚 Sources:")
    for s in sources:
        line = f"  [{s['n']}] {s['citation']}"
        if debug:
            line += f"   (score={s['score']})"
        print(line)
        if debug:
            print("      " + s["text"][:200].replace("\n", " ") + "...")


def answer_once(rag, question, history, debug):
    sources = []
    print("\n🤖 ", end="", flush=True)
    full = []
    for event in rag.ask_stream(question, history=history):
        if event["type"] == "sources":
            sources = event["sources"]
            if debug and event["search_query"] != question:
                print(f"(rewritten query: {event['search_query']})\n   ", end="")
        else:
            print(event["text"], end="", flush=True)
            full.append(event["text"])
    print()
    print_sources(sources, debug)
    return "".join(full)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    rag = RAGPipeline()
    if not len(rag.store):
        print("⚠️  Index khaali hai. Pehle `python ingest.py` chalao (documents data/ folder mein daal ke).")
        sys.exit(1)

    if args.question:
        answer_once(rag, " ".join(args.question), None, args.debug)
        return

    print(f"💬 RAG chat — {len(rag.store)} chunks loaded. '/reset' = history clear, '/exit' = band.\n")
    history = []
    while True:
        try:
            q = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/exit", "/quit"}:
            break
        if q == "/reset":
            history = []
            print("History clear.")
            continue
        ans = answer_once(rag, q, history, args.debug)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


if __name__ == "__main__":
    main()

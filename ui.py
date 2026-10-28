"""
Chat UI (Streamlit).

  streamlit run ui.py
  -> browser mein http://localhost:8501 khulega
"""
import streamlit as st

from rag import RAGPipeline, settings
from rag.loaders import SUPPORTED_EXTENSIONS

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Models load ho rahe hain...")
def get_rag():
    return RAGPipeline()


rag = get_rag()

# ---------------- Sidebar: documents ----------------
with st.sidebar:
    st.header("📂 Documents")
    uploads = st.file_uploader(
        "Files upload karo", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS], accept_multiple_files=True
    )
    if uploads and st.button("Ingest", type="primary"):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Chunking + embedding..."):
            for up in uploads:
                dest = settings.data_dir / up.name
                dest.write_bytes(up.getbuffer())
                stats = rag.ingest(dest, verbose=False)
                if stats["failed"]:
                    st.error(f"{up.name}: {stats['failed']}")
        st.success(f"Done. Total chunks: {len(rag.store)}")

    if st.button("🔄 Re-scan data/ folder"):
        with st.spinner("Scanning..."):
            s = rag.ingest(verbose=False)
        st.info(f"Added {len(s['added'])}, updated {len(s['updated'])}, removed {len(s['removed'])}")

    files = rag.store.sources()
    st.caption(f"{len(files)} files · {len(rag.store)} chunks · LLM: {settings.llm_model}")
    selected = st.multiselect("Sirf in files mein search karo (optional)", [f["source"] for f in files])
    top_k = st.slider("Top-k chunks", 1, 10, settings.final_top_k)
    show_debug = st.toggle("Retrieval details dikhao", value=False)
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
st.title("📚 Chat with your documents")
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources, search_query=None):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} sources"):
        if show_debug and search_query:
            st.caption(f"Search query: `{search_query}`")
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['citation']}**" + (f"  · score `{s['score']}`" if show_debug else ""))
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"), m.get("search_query"))

if question := st.chat_input("Apna sawal poochho..."):
    if not len(rag.store):
        st.warning("Pehle sidebar se documents upload + ingest karo.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box = st.empty()
        text, sources, sq = "", [], question
        try:
            for ev in rag.ask_stream(question, history=history, top_k=top_k, source_filter=selected or None):
                if ev["type"] == "sources":
                    sources, sq = ev["sources"], ev["search_query"]
                else:
                    text += ev["text"]
                    box.markdown(text + "▌")
            box.markdown(text)
        except Exception as e:
            text = f"⚠️ Error: {e}"
            box.error(text)
        render_sources(sources, sq)

    st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources, "search_query": sq})

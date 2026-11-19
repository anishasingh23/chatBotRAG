
import io
import os
import re
import math
import uuid
from datetime import datetime

import streamlit as st
import pypdf
import pandas as pd
import plotly.graph_objects as go
from docx import Document as DocxDocument
import chromadb
from chromadb.utils import embedding_functions
import groq
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 30
MIN_DOC_LEN = 100

RESUME_TERMS = ["project", "experience", "work", "portfolio", "skill"]
TECH_TERMS = [
    "python", "java", "javascript", "html", "css", "react", "node", "sql",
    "nosql", "aws", "azure", "docker", "kubernetes", "machine learning",
    "ai", "data analysis", "web development", "cloud computing",
]

# ============================================================
# TEXT UTILS
# ============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def is_resume_query(q: str) -> bool:
    q = q.lower()
    return any(t in q for t in RESUME_TERMS)

# ============================================================
# EXTRACTORS
# ============================================================
def _extract_pdf(file) -> str:
    reader = pypdf.PdfReader(file)
    parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t:
            parts.append(f"--- PAGE {i + 1} ---\n{t}")
    return "\n\n".join(parts)


def _extract_docx(file) -> str:
    doc = DocxDocument(io.BytesIO(file.getvalue()))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_txt(file) -> str:
    return file.getvalue().decode("utf-8", errors="ignore")


def _extract_excel(file) -> str:
    return pd.read_excel(file).to_string()


def _extract_csv(file) -> str:
    for enc in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc).to_string()
        except UnicodeDecodeError:
            continue
    file.seek(0)
    return file.getvalue().decode("utf-8", errors="ignore")


_HANDLERS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".doc": _extract_docx,
    ".txt": _extract_txt,
    ".xlsx": _extract_excel,
    ".xls": _extract_excel,
    ".csv": _extract_csv,
}


def extract_text(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    handler = _HANDLERS.get(ext)
    if not handler:
        return ""
    try:
        return clean_text(handler(uploaded_file))
    except Exception:
        return ""

# ============================================================
# CHUNKING
# ============================================================
_SECTION_PATTERNS = [
    r"(?i)(?:projects?|work experience|experience|portfolio)[:\s]*\n",
    r"(?i)\b(?:project|experience)\b.*\n",
    r"(?i)^[A-Z][A-Za-z\s]+:.*$",
]


def _split_by_sentences(text: str, size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) > size and cur:
            chunks.append(" ".join(cur))
            keep = max(0, len(cur) - 3)
            cur = cur[keep:]
            cur_len = sum(len(x) for x in cur) + len(cur) - 1
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_text(text: str) -> list[str]:
    if not text:
        return []
    for pat in _SECTION_PATTERNS:
        parts = re.split(pat, text)
        if len(parts) > 1:
            out = []
            for i, part in enumerate(parts):
                if i == 0 or not part.strip():
                    continue
                if len(part) > CHUNK_SIZE * 2:
                    out.extend(_split_by_sentences(part, CHUNK_SIZE, CHUNK_OVERLAP))
                else:
                    out.append(part.strip())
            if out:
                return out
    return _split_by_sentences(text, CHUNK_SIZE, CHUNK_OVERLAP)


def process_documents(texts: list[str], metadata: list[dict]) -> list[dict]:
    chunks = []
    for i, text in enumerate(texts):
        if not text or len(text) < MIN_DOC_LEN:
            continue
        src = metadata[i]["source"].lower()
        is_resume = "resume" in src or "cv" in src
        for j, chunk in enumerate(split_text(text)):
            if len(chunk) <= MIN_CHUNK_LEN:
                continue
            chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata[i],
                    "chunk_id": j,
                    "chunk_length": len(chunk),
                    "word_count": len(chunk.split()),
                    "is_resume": is_resume,
                },
            })
    return chunks

# ============================================================
# VECTOR STORE
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    # Ephemeral: safe on read-only clouds (HF Spaces / Streamlit Cloud).
    # Swap to chromadb.PersistentClient(path="./chroma_db") for local persistence.
    return chromadb.EphemeralClient()


def fresh_collection(client, emb_fn):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)


def add_chunks(chunks: list[dict], collection) -> bool:
    if not chunks:
        return False
    try:
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[str(uuid.uuid4()) for _ in chunks],
        )
        return True
    except Exception:
        return False


def query_collection(collection, query_text: str, k: int = 6, extra_terms: str | None = None):
    try:
        res = collection.query(
            query_texts=[query_text], n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return None

    if extra_terms:
        try:
            boost = collection.query(
                query_texts=[extra_terms], n_results=min(k, 4),
                include=["documents", "metadatas", "distances"],
            )
            docs = list(res["documents"][0])
            metas = list(res["metadatas"][0])
            for d, m in zip(boost["documents"][0], boost["metadatas"][0]):
                if d not in docs:
                    docs.append(d)
                    metas.append(m)
            res["documents"] = [docs]
            res["metadatas"] = [metas]
        except Exception:
            pass
    return res

# ============================================================
# LLM (Groq, free tier)
# ============================================================
_SYS_PROMPT = (
    "You are a helpful assistant that answers strictly from provided context. "
    "When asked about projects or experience, list ALL items found with full details."
)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    return groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _chat(prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
    client = get_groq_client()
    if client is None:
        return "Error: GROQ_API_KEY missing."
    try:
        r = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL, temperature=temperature, max_tokens=max_tokens,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def llm_answer(query: str, context: str, is_resume: bool = False, research: bool = False) -> str:
    if is_resume:
        prompt = f"""Extract and describe ALL projects/experiences from the context.

CONTEXT:
{context}

QUERY:
{query}

List every item with technologies, durations, and achievements. If not present, say so."""
    elif research:
        prompt = f"""Provide a comprehensive analysis with evidence, based ONLY on the context.

CONTEXT:
{context}

QUESTION:
{query}"""
    else:
        prompt = f"""Answer concisely using ONLY the context. If absent, say so.

CONTEXT:
{context}

QUESTION:
{query}"""
    return _chat(prompt)


def llm_summarize(text: str, filename: str) -> str:
    if len(text) > 12000:
        text = text[:12000] + "... [truncated]"
    prompt = f"""Summarize the document "{filename}" concisely — key points and main topics.

CONTENT:
{text}

SUMMARY:"""
    return _chat(prompt, max_tokens=512)

# ============================================================
# ENTITY GRAPH (regex-based; honest label — not full NER)
# ============================================================
_STOP = {"the", "and", "for", "with", "this", "that"}


def extract_entities(text: str) -> list[str]:
    ents = set()
    for w in re.findall(r"\b[A-Z][a-z]+\b", text):
        if len(w) > 2 and w.lower() not in _STOP:
            ents.add(w)
    for w in re.findall(r"\b[A-Z][A-Za-z&]+\b", text):
        if len(w) > 3:
            ents.add(w)
    low = text.lower()
    for t in TECH_TERMS:
        if t in low:
            ents.add(t.title())
    return list(ents)[:15]


def build_graph(chunks: list[dict]) -> dict:
    nodes, edges, seen = [], [], set()

    def add_node(nid, label, ntype, size):
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "type": ntype, "size": size})

    for c in chunks:
        src = c["metadata"]["source"]
        doc_id = f"doc_{src}"
        add_node(doc_id, src, "document", 20)
        for ent in extract_entities(c["text"]):
            eid = f"ent_{ent}"
            add_node(eid, ent, "entity", 15)
            edges.append({"source": doc_id, "target": eid, "value": 5})
    return {"nodes": nodes, "edges": edges}


def add_query_to_graph(graph: dict, query: str, response: str) -> None:
    qid = f"q_{hash(query)}"
    if not any(n["id"] == qid for n in graph["nodes"]):
        graph["nodes"].append({"id": qid, "label": f"Query: {query[:30]}...",
                               "type": "query", "size": 15})
    for ent in set(extract_entities(query + " " + response)):
        eid = f"ent_{ent}"
        if not any(n["id"] == eid for n in graph["nodes"]):
            graph["nodes"].append({"id": eid, "label": ent, "type": "entity", "size": 10})
        exists = any(
            (e["source"] == qid and e["target"] == eid)
            or (e["source"] == eid and e["target"] == qid)
            for e in graph["edges"]
        )
        if not exists:
            graph["edges"].append({"source": qid, "target": eid, "value": 3})


def render_graph(graph: dict):
    if not graph["nodes"]:
        st.info("No entities to display.")
        return
    n = len(graph["nodes"])
    pos = {}
    for i, node in enumerate(graph["nodes"]):
        ang = 2 * math.pi * i / n
        pos[node["id"]] = (150 * math.cos(ang), 150 * math.sin(ang))

    ex, ey = [], []
    for e in graph["edges"]:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(width=1, color="#888"), hoverinfo="none")

    nx_, ny_, txt, col, sz = [], [], [], [], []
    color_map = {"document": "#1f77b4", "query": "#d62728"}
    for node in graph["nodes"]:
        x, y = pos[node["id"]]
        nx_.append(x); ny_.append(y)
        txt.append(node["label"]); sz.append(node.get("size", 10))
        col.append(color_map.get(node["type"], "#ff7f0e"))

    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers+text", text=txt, textposition="top center",
        hoverinfo="text",
        marker=dict(color=col, size=sz, line=dict(width=2, color="darkgray")),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Entity Graph", showlegend=False, hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "collection": None,
        "chat_sessions": {},
        "current_session_id": None,
        "processed_docs": False,
        "current_files": [],
        "file_previews": {},
        "document_summaries": {},
        "knowledge_graph": {"nodes": [], "edges": []},
        "research_mode": False,
        "show_graph": False,
        "show_upload": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def new_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.chat_sessions[sid] = []
    st.session_state.current_session_id = sid
    return sid


def ensure_session() -> str:
    if not st.session_state.get("current_session_id"):
        return new_session()
    st.session_state.chat_sessions.setdefault(st.session_state.current_session_id, [])
    return st.session_state.current_session_id


def current_history() -> list:
    return st.session_state.chat_sessions[ensure_session()]


def append_history(q: str, a: str, ctx: str) -> None:
    st.session_state.chat_sessions[ensure_session()].append((q, a, ctx))

# ============================================================
# STYLES
# ============================================================
CSS = """
<style>
.main-header { font-size: 2.5rem !important; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
.sub-header { font-size: 1.2rem !important; color: #666; text-align: center; margin-bottom: 2rem; }
.upload-container { background:#f8f9fa; padding:2rem; border-radius:10px; border:2px dashed #1f77b4; margin-bottom:2rem; }
.stButton button { background:#1f77b4; color:#fff; border:none; padding:.5rem 1rem; border-radius:5px; font-weight:bold; }
.stButton button:hover { background:#135e96; }
.chat-message-user { background:#e6f7ff; padding:1rem; border-radius:10px; margin-bottom:1rem; border-left:4px solid #1f77b4; color:#333 !important; }
.chat-message-assistant { background:#f0f2f6; padding:1rem; border-radius:10px; margin-bottom:1rem; border-left:4px solid #28a745; color:#333 !important; }
.status-box { background:#f8f9fa; padding:1rem; border-radius:5px; border-left:4px solid #6c757d; margin-bottom:1rem; color:#333 !important; }
.preview-container { background:#f8f9fa; padding:1rem; border-radius:5px; margin-bottom:1rem; border:1px solid #ddd; }
.summary-box { background:#e8f5e8; padding:1rem; border-radius:5px; margin-bottom:1rem; border-left:4px solid #28a745; }
.stMarkdown, .stText, .stCode { color:#333 !important; }
</style>
"""

# ============================================================
# UI — UPLOAD
# ============================================================
def reset_all():
    try:
        get_chroma_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    st.session_state.collection = None
    st.session_state.processed_docs = False
    st.session_state.current_files = []
    st.session_state.file_previews = {}
    st.session_state.document_summaries = {}
    st.session_state.knowledge_graph = {"nodes": [], "edges": []}
    st.session_state.show_upload = True
    new_session()


def process_files(files) -> bool:
    client = get_chroma_client()
    emb = get_embedding_fn()
    st.session_state.collection = fresh_collection(client, emb)

    texts, metas = [], []
    for f in files:
        text = extract_text(f)
        if text and len(text) >= MIN_DOC_LEN:
            texts.append(text)
            metas.append({
                "source": f.name,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(f.getvalue()),
            })
            if f.name not in st.session_state.document_summaries:
                with st.spinner(f"Summarizing {f.name}..."):
                    st.session_state.document_summaries[f.name] = llm_summarize(text, f.name)

    if not texts:
        st.error("No document yielded sufficient text.")
        return False

    chunks = process_documents(texts, metas)
    if not chunks or not add_chunks(chunks, st.session_state.collection):
        st.error("Failed to index chunks.")
        return False

    st.session_state.knowledge_graph = build_graph(chunks)
    st.session_state.processed_docs = True
    st.session_state.current_files = [f.name for f in files]
    st.session_state.show_upload = False
    new_session()
    return True


def render_upload():
    if not (st.session_state.show_upload or not st.session_state.processed_docs):
        return

    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Documents")
    files = st.file_uploader(
        "Add files", type=["pdf", "docx", "txt", "xlsx", "csv"],
        accept_multiple_files=True, key="file_uploader",
        help="PDF, Word, Text, Excel, CSV",
    )
    if files:
        for f in files:
            if f.name not in st.session_state.file_previews:
                txt = extract_text(f)[:500]
                st.session_state.file_previews[f.name] = (txt + "...") if txt else "No preview."

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if st.button("📤 Process Documents", type="primary", use_container_width=True):
            if not files:
                st.warning("Upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    if process_files(files):
                        st.success(f"Indexed {len(files)} file(s).")
                        st.rerun()
    with c2:
        if st.session_state.processed_docs and st.button("🔄 Add Files", use_container_width=True):
            st.session_state.show_upload = True
            st.rerun()
    with c3:
        if st.session_state.processed_docs and st.button("🗑️ Clear", use_container_width=True):
            reset_all()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# UI — CHAT
# ============================================================
def render_previews():
    if not st.session_state.file_previews:
        return
    st.markdown("### 📄 Document Previews")
    for name, preview in st.session_state.file_previews.items():
        with st.expander(f"📋 {name}"):
            st.markdown(f'<div class="preview-container"><pre>{preview}</pre></div>',
                        unsafe_allow_html=True)
            if name in st.session_state.document_summaries:
                st.markdown(
                    f'<div class="summary-box"><strong>📝 Summary:</strong><br>'
                    f'{st.session_state.document_summaries[name]}</div>',
                    unsafe_allow_html=True,
                )


def render_messages():
    for q, a, _ in current_history():
        st.markdown(f'<div class="chat-message-user"><strong>👤 You:</strong> {q}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message-assistant"><strong>🤖 Assistant:</strong> {a}</div>',
                    unsafe_allow_html=True)


def handle_question(q: str, research: bool) -> bool:
    if st.session_state.collection is None or st.session_state.collection.count() == 0:
        st.error("Process documents first.")
        return False

    resume_q = is_resume_query(q)
    k = 10 if resume_q else (8 if research else 6)
    extra = "projects experience work portfolio skills" if resume_q else None

    res = query_collection(st.session_state.collection, q, k=k, extra_terms=extra)
    if not res or not res.get("documents") or not res["documents"][0]:
        st.error("No relevant context found.")
        return False

    context = "\n\n".join(
        f"[From {m.get('source', 'Unknown')}]:\n{d}"
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    )

    with st.spinner("Analyzing..." if resume_q else "Thinking..."):
        answer = llm_answer(q, context, is_resume=resume_q, research=research)

    append_history(q, answer, context)
    if research:
        add_query_to_graph(st.session_state.knowledge_graph, q, answer)
    return True


def render_chat():
    if not st.session_state.processed_docs:
        return

    count = st.session_state.collection.count() if st.session_state.collection else 0
    st.markdown('<div class="status-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Documents:** {len(st.session_state.current_files)}")
    c2.write(f"**Chunks:** {count}")
    c3.write(f"**Mode:** {'🔍 Research' if st.session_state.research_mode else '💬 Standard'}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.research_mode = st.toggle("Deep Research Mode")
    st.session_state.show_graph = st.toggle("Show Entity Graph")

    if st.session_state.show_graph and st.session_state.knowledge_graph["nodes"]:
        render_graph(st.session_state.knowledge_graph)

    render_previews()
    render_messages()

    q = st.chat_input("Ask a question about your documents...")
    if q and handle_question(q, st.session_state.research_mode):
        st.rerun()

# ============================================================
# UI — SIDEBAR
# ============================================================
def render_sidebar():
    st.markdown("### 💬 Chat Sessions")
    if st.button("➕ New Chat", use_container_width=True):
        new_session()
        st.rerun()

    current = ensure_session()
    for sid in list(st.session_state.chat_sessions.keys()):
        hist = st.session_state.chat_sessions[sid]
        label = f"Chat {sid[:8]}"
        if hist:
            label += f": {hist[0][0][:20]}..."
        if sid == current:
            st.markdown(f"**▶ {label}**")
        else:
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()

    st.markdown("---")
    if st.button("🧹 Clear Current Chat", use_container_width=True):
        st.session_state.chat_sessions[ensure_session()] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info(
        "- Ask about **projects**, **experience**, **skills**\n"
        "- Enable **Deep Research** for analysis\n"
        "- Enable **Entity Graph** to explore concepts\n"
        "- Formats: PDF, DOCX, TXT, XLSX, CSV"
    )

# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="DocuChat", page_icon="🤖", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    st.markdown('<h1 class="main-header">DocuChat 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your documents</p>', unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing. Create a .env file with `GROQ_API_KEY=your_key`.")
        st.stop()

    render_upload()

    if st.session_state.processed_docs:
        col1, col2 = st.columns([3, 1])
        with col1:
            render_chat()
        with col2:
            render_sidebar()


if __name__ == "__main__":
    main()

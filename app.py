"""
╔══════════════════════════════════════════════════════════════╗
║         CONTRACT REVIEW ASSISTANT  — S.NO 365               ║
║   NVIDIA DLI Advanced DL / GenAI with LLMs                  ║
║                                                              ║
║   Works on:  PDF · DOCX · TXT datasets                      ║
║   Stack:     Streamlit · LangChain · FAISS · OpenAI          ║
╚══════════════════════════════════════════════════════════════╝

GITHUB SETUP:
  1. Create repo, add this file as app.py
  2. Add requirements.txt (see bottom of this file)
  3. Push to GitHub
  4. Go to share.streamlit.io → New app → pick your repo
  5. Set secret:  OPENAI_API_KEY = "sk-..."
  6. Deploy!

LOCAL SETUP:
  pip install -r requirements.txt
  streamlit run app.py
"""

# ─── Standard library ────────────────────────────────────────
import os, json, time, hashlib, logging, datetime, tempfile, io
from pathlib import Path
from typing import List, Optional, Dict, Any

# ─── Third-party ─────────────────────────────────────────────
import streamlit as st

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Retry
from tenacity import retry, stop_after_attempt, wait_exponential

# ──────────────────────────────────────────────────────────────
#  GLOBAL CONFIG
# ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5
DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL   = "text-embedding-3-small"

RISKY_KEYWORDS = [
    "indemnif","liabilit","penalty","terminat","arbitrat",
    "force majeure","warrant","confidential","exclusiv",
    "non-compet","liquidat","automatic renewal","unilateral",
    "waiver","lien","intellectual property","ownership","assign",
    "govern","unlimited","irrevocable","perpetual",
]

REQUIRED_OBLIGATIONS = [
    "payment terms","delivery schedule","governing law",
    "dispute resolution","termination clause","confidentiality",
    "intellectual property rights","liability cap",
    "warranty","indemnification","notice period",
]

INJECTION_PATTERNS = [
    "ignore previous","disregard your","you are now",
    "forget everything","system prompt","jailbreak",
    "ignore all instructions","override instructions",
]

PRESET_QUERIES = {
    "🔍 Full Risk Review": (
        "Review this entire contract. List ALL risky clauses with detailed explanations "
        "and flag every missing standard obligation. Provide a final RISK LEVEL: Low / Medium / High."
    ),
    "⚖️ Liability & Indemnification": (
        "What does the contract say about liability caps and indemnification? "
        "Is there unlimited or one-sided liability? Quote relevant clauses."
    ),
    "🚪 Termination Rights": (
        "Summarise termination rights for both parties. "
        "Are there one-sided, automatic, or penalty-heavy termination clauses?"
    ),
    "💰 Payment & Penalties": (
        "What are the exact payment terms, deadlines, and penalty or late-fee clauses?"
    ),
    "🧠 IP Ownership": (
        "Who owns intellectual property created under this contract? "
        "Are there any IP assignment or work-for-hire clauses?"
    ),
    "🌍 Governing Law & Disputes": (
        "What governing law applies? How are disputes resolved — arbitration, litigation, mediation?"
    ),
    "🔒 Confidentiality": (
        "Summarise all confidentiality and non-disclosure obligations. "
        "What is the duration and scope?"
    ),
    "📋 Missing Obligations Checklist": (
        "Check whether the contract contains all of these standard obligations: "
        "payment terms, delivery schedule, governing law, dispute resolution, "
        "termination clause, confidentiality, IP rights, liability cap, warranty, indemnification. "
        "List what is present and what is missing."
    ),
}

# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = []

def audit(event: str, payload: dict):
    entry = {
        "ts": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **payload,
    }
    st.session_state["audit_log"].append(entry)
    logger.info("[AUDIT] %s | %s", event, payload)

# ──────────────────────────────────────────────────────────────
#  DOCUMENT LOADING
# ──────────────────────────────────────────────────────────────
def load_uploaded_file(uploaded_file) -> List[Document]:
    """Save to temp file and load with the right LangChain loader."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(tmp_path)
    elif suffix == ".csv":
        loader = CSVLoader(tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    # Tag each doc with source filename
    for d in docs:
        d.metadata["source_file"] = uploaded_file.name
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def file_fingerprint(uploaded_file) -> str:
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(data).hexdigest()[:10]

# ──────────────────────────────────────────────────────────────
#  VECTOR STORE
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vector_store(chunks_json: str, api_key: str) -> FAISS:
    """Build FAISS index. Cached so it only rebuilds when chunks change."""
    chunks = [Document(page_content=c["text"], metadata=c["meta"])
              for c in json.loads(chunks_json)]
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
    return FAISS.from_documents(chunks, embeddings)


# ──────────────────────────────────────────────────────────────
#  PROMPT
# ──────────────────────────────────────────────────────────────
CONTRACT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior legal AI assistant specialised in contract review.
Analyse the provided contract excerpts and answer precisely.

RULES:
• Ground every finding in the provided context only.
• Flag risky clauses: unlimited liability, one-sided termination, auto-renewal,
  broad indemnification, IP assignment, penalty-heavy terms.
• Identify missing standard obligations if not found in context.
• Cite the clause or section number whenever visible.
• Never fabricate clauses. If uncertain, state your confidence.
• End with: RISK LEVEL: Low / Medium / High — one-line rationale.

CONTRACT CONTEXT:
{context}

TASK:
{question}

ANALYSIS:"""
)

# ──────────────────────────────────────────────────────────────
#  RAG CHAIN
# ──────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def run_query(chain: RetrievalQA, query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = chain.invoke({"query": query})
    return {
        "answer": result["result"],
        "latency": round(time.perf_counter() - t0, 2),
    }


def build_chain(store: FAISS, api_key: str, model: str, top_k: int) -> RetrievalQA:
    llm = ChatOpenAI(model_name=model, temperature=0,
                     max_tokens=1500, openai_api_key=api_key)
    retriever = store.as_retriever(search_type="similarity",
                                   search_kwargs={"k": top_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CONTRACT_PROMPT},
    )

# ──────────────────────────────────────────────────────────────
#  GUARDRAILS
# ──────────────────────────────────────────────────────────────
def is_injection(text: str) -> bool:
    lo = text.lower()
    return any(p in lo for p in INJECTION_PATTERNS)

def rule_scan_risky(text: str) -> List[str]:
    lo = text.lower()
    return [kw for kw in RISKY_KEYWORDS if kw in lo]

def scan_missing(text: str) -> List[str]:
    lo = text.lower()
    return [ob for ob in REQUIRED_OBLIGATIONS if ob not in lo]

def risk_color(answer: str) -> str:
    up = answer.upper()
    if "RISK LEVEL: HIGH" in up or "HIGH RISK" in up:
        return "high"
    if "RISK LEVEL: MEDIUM" in up or "MEDIUM RISK" in up:
        return "medium"
    return "low"

# ──────────────────────────────────────────────────────────────
#  CSS  (professional dark-navy + gold legal theme)
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1a1f29 100%);
    border-bottom: 1px solid #21262d;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 0;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0c040;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7d8590;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.badge {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.03em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #8b949e !important;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stSidebar"] .stButton button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 6px;
    font-size: 0.82rem;
    width: 100%;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #30363d;
    border-color: #f0c040;
    color: #f0c040;
}
.sidebar-section {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 14px;
}
.sidebar-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #f0c040;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 8px;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #f0c040;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.7rem;
}

/* ── Risk badges ── */
.risk-high {
    background: #2d1117;
    border-left: 4px solid #f85149;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.risk-medium {
    background: #271d0e;
    border-left: 4px solid #e3b341;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.risk-low {
    background: #0d1f17;
    border-left: 4px solid #3fb950;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.risk-label-high   { color:#f85149; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; }
.risk-label-medium { color:#e3b341; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; }
.risk-label-low    { color:#3fb950; font-weight:700; font-size:0.8rem; letter-spacing:0.05em; }

/* ── Tag pills ── */
.tag-pill {
    display:inline-block;
    background:#21262d;
    border:1px solid #30363d;
    color:#8b949e;
    font-size:0.7rem;
    padding:2px 9px;
    border-radius:20px;
    margin:2px 3px 2px 0;
}
.tag-pill-red    { background:#2d1117; border-color:#f85149; color:#f85149; }
.tag-pill-yellow { background:#271d0e; border-color:#e3b341; color:#e3b341; }
.tag-pill-green  { background:#0d1f17; border-color:#3fb950; color:#3fb950; }

/* ── KPI row ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.kpi-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.kpi-val  { font-size:1.6rem; font-weight:700; color:#f0c040; line-height:1; }
.kpi-lbl  { font-size:0.68rem; color:#7d8590; margin-top:4px; text-transform:uppercase; letter-spacing:0.07em; }

/* ── Answer box ── */
.answer-body {
    color: #c9d1d9;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ── Section divider ── */
.gold-divider {
    border: none;
    border-top: 1px solid #f0c04022;
    margin: 1.5rem 0;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: #f0c040 !important;
    box-shadow: 0 0 0 2px #f0c04033 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f0c040, #e3a820) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }
.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}
.stFileUploader {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
[data-testid="stMetricValue"]  { color: #f0c040 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: #7d8590 !important; font-size: 0.75rem !important; }
div[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}
.stAlert { border-radius: 8px !important; }

/* ── Scrollable audit log ── */
.audit-scroll {
    max-height: 220px;
    overflow-y: auto;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: monospace;
    font-size: 0.72rem;
    color: #8b949e;
}
.footer {
    text-align:center;
    color:#30363d;
    font-size:0.72rem;
    margin-top:2rem;
    padding:1rem 0;
    border-top:1px solid #21262d;
}
</style>
"""

# ══════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Contract Review Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state defaults ──────────────────────────────────
for k, v in {
    "store": None, "chain": None, "full_text": "",
    "file_name": "", "history": [],
    "kpi": {"queries": 0, "total_lat": 0.0, "blocked": 0},
    "index_key": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">⚖️ Contract Review Assistant</div>
  <div class="hero-sub">AI-powered clause analysis · Risky term detection · Missing obligation audit</div>
  <div class="hero-badges">
    <span class="badge">RAG Pipeline</span>
    <span class="badge">FAISS Vector Search</span>
    <span class="badge">GPT-4o-mini</span>
    <span class="badge">Prompt-injection Guard</span>
    <span class="badge">Audit Log</span>
    <span class="badge">PDF · DOCX · TXT · CSV</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-label">🔑 API Credentials</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="Get yours at platform.openai.com",
        label_visibility="collapsed",
        placeholder="sk-..."
    )

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">⚙️ Model Settings</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini = fastest & cheapest. gpt-4o = highest accuracy."
    )
    top_k = st.slider("Retrieval Top-K", 3, 10, TOP_K,
                      help="How many contract chunks to retrieve per query")
    chunk_size  = st.slider("Chunk Size (tokens)", 400, 1200, CHUNK_SIZE, step=100)
    chunk_over  = st.slider("Chunk Overlap", 50, 300, CHUNK_OVERLAP, step=25)

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📋 Audit Log</div>', unsafe_allow_html=True)

    logs = st.session_state["audit_log"]
    if logs:
        log_html = "".join(
            f"<div style='margin-bottom:4px'><span style='color:#f0c040'>{e['ts']}</span>"
            f" <span style='color:#58a6ff'>[{e['event']}]</span></div>"
            for e in logs[-15:]
        )
        st.markdown(f'<div class="audit-scroll">{log_html}</div>', unsafe_allow_html=True)
        buf = io.BytesIO(
            "\n".join(json.dumps(e) for e in logs).encode()
        )
        st.download_button("⬇ Download Full Log", buf,
                           file_name="audit_log.jsonl", mime="application/jsonl")
    else:
        st.caption("No events yet.")

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">🔄 Recovery</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Index & Reset"):
        for k in ["store","chain","full_text","file_name","index_key"]:
            st.session_state[k] = None if k in ("store","chain") else ""
        st.session_state["history"] = []
        st.session_state["kpi"] = {"queries":0,"total_lat":0.0,"blocked":0}
        build_vector_store.clear()
        audit("index_cleared", {})
        st.success("Index cleared. Upload a new contract.")

# ══════════════════════════════════════════════════════════════
#  MAIN — TWO COLUMNS
# ══════════════════════════════════════════════════════════════
left, right = st.columns([1, 1.15], gap="large")

# ────────────────────────────────────────────────────────────
#  LEFT — Upload & pre-scan
# ────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card-title">📂 Upload Contract Dataset</div>',
                unsafe_allow_html=True)
    st.caption("Supports: PDF · DOCX · TXT · CSV")

    uploaded = st.file_uploader(
        "Drop file here",
        type=["pdf","docx","doc","txt","csv"],
        label_visibility="collapsed",
    )

    if uploaded and not api_key:
        st.error("⚠️ Enter your OpenAI API key in the sidebar first.")
    elif uploaded and api_key:
        fp = file_fingerprint(uploaded)
        new_key = f"{uploaded.name}_{fp}"

        if new_key != st.session_state["index_key"]:
            with st.spinner(f"📖 Loading **{uploaded.name}** …"):
                try:
                    docs = load_uploaded_file(uploaded)
                    audit("document_uploaded", {
                        "filename": uploaded.name, "pages": len(docs), "hash": fp
                    })
                except Exception as e:
                    st.error(f"Failed to load file: {e}")
                    st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_over,
                separators=["\n\n", "\n", ". ", " "],
            )
            chunks = splitter.split_documents(docs)
            full_text = " ".join(d.page_content for d in docs)

            # Serialise chunks for cache key
            chunks_json = json.dumps([
                {"text": c.page_content, "meta": c.metadata} for c in chunks
            ])

            with st.spinner("🔢 Building vector index …"):
                store = build_vector_store(chunks_json, api_key)

            chain = build_chain(store, api_key, model_choice, top_k)

            st.session_state.update({
                "store": store, "chain": chain,
                "full_text": full_text, "file_name": uploaded.name,
                "index_key": new_key,
            })
            audit("index_built", {"chunks": len(chunks)})

        # ── File info card ──
        st.markdown(f"""
        <div class="card" style="margin-top:1rem">
          <div class="card-title">📄 Loaded File</div>
          <div style="font-size:0.88rem;color:#c9d1d9;font-weight:600">{st.session_state['file_name']}</div>
          <div style="font-size:0.72rem;color:#7d8590;margin-top:4px">Fingerprint: <code style="color:#f0c040">{fp}</code></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Rule-based pre-scan ──
        full_text = st.session_state["full_text"]
        risky_kws = rule_scan_risky(full_text)
        missing   = scan_missing(full_text)

        if risky_kws:
            pills = "".join(f'<span class="tag-pill tag-pill-red">{kw}</span>' for kw in risky_kws[:12])
            st.markdown(f"""
            <div class="risk-medium" style="margin-top:0.5rem">
              <div class="risk-label-medium">⚠️ RISKY KEYWORDS DETECTED</div>
              <div style="margin-top:6px">{pills}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low"><span class="risk-label-low">✅ No common risky keywords found in rule scan</span></div>',
                        unsafe_allow_html=True)

        if missing:
            pills = "".join(f'<span class="tag-pill tag-pill-yellow">{ob}</span>' for ob in missing)
            st.markdown(f"""
            <div class="risk-high" style="margin-top:0.5rem">
              <div class="risk-label-high">🚨 POTENTIALLY MISSING OBLIGATIONS</div>
              <div style="margin-top:6px">{pills}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low"><span class="risk-label-low">✅ All standard obligations found</span></div>',
                        unsafe_allow_html=True)

        # ── KPI row ──
        kpi = st.session_state["kpi"]
        avg_lat = (kpi["total_lat"] / max(kpi["queries"], 1))
        st.markdown(f"""
        <div class="kpi-grid" style="margin-top:1rem">
          <div class="kpi-box"><div class="kpi-val">{kpi['queries']}</div><div class="kpi-lbl">Queries</div></div>
          <div class="kpi-box"><div class="kpi-val">{avg_lat:.1f}s</div><div class="kpi-lbl">Avg Latency</div></div>
          <div class="kpi-box"><div class="kpi-val">{len(risky_kws)}</div><div class="kpi-lbl">Risky Terms</div></div>
          <div class="kpi-box"><div class="kpi-val">{kpi['blocked']}</div><div class="kpi-lbl">Blocked</div></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem 1rem">
          <div style="font-size:2.5rem;margin-bottom:0.7rem">📑</div>
          <div style="color:#7d8590;font-size:0.88rem">Upload a contract to begin analysis.<br>
          PDF · DOCX · TXT · CSV datasets supported.</div>
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
#  RIGHT — Query & Results
# ────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="card-title">🔍 Ask the Assistant</div>', unsafe_allow_html=True)

    preset_label = st.selectbox(
        "Quick presets",
        ["✏️ Custom query…"] + list(PRESET_QUERIES.keys()),
        label_visibility="collapsed",
    )
    default_q = PRESET_QUERIES.get(preset_label, "")
    user_query = st.text_area(
        "Your question",
        value=default_q,
        height=110,
        label_visibility="collapsed",
        placeholder="e.g. What are the termination conditions for the vendor?",
    )

    col_btn, col_clr = st.columns([3, 1])
    with col_btn:
        analyse_btn = st.button(
            "⚡ Analyse Contract",
            type="primary",
            disabled=(st.session_state["chain"] is None),
            use_container_width=True,
        )
    with col_clr:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state["history"] = []

    if analyse_btn and user_query.strip():
        if st.session_state["chain"] is None:
            st.error("Upload and index a contract first.")
        elif is_injection(user_query):
            st.error("🛡️ Prompt-injection attempt detected. Query blocked.")
            st.session_state["kpi"]["blocked"] += 1
            audit("injection_blocked", {"query": user_query[:80]})
        else:
            with st.spinner("🤔 Analysing …"):
                try:
                    res     = run_query(st.session_state["chain"], user_query)
                    answer  = res["answer"]
                    latency = res["latency"]

                    kpi = st.session_state["kpi"]
                    kpi["queries"]   += 1
                    kpi["total_lat"] += latency

                    rc = risk_color(answer)
                    audit("query_answered", {
                        "query": user_query[:100],
                        "latency_s": latency,
                        "risk": rc,
                    })

                    st.session_state["history"].insert(0, {
                        "q": user_query, "a": answer,
                        "lat": latency, "risk": rc,
                        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
                    })
                except Exception as e:
                    st.error(f"❌ API error: {e}")
                    audit("query_error", {"error": str(e)})

    # ── History / results ──
    if st.session_state["history"]:
        for idx, item in enumerate(st.session_state["history"]):
            rc   = item["risk"]
            cls  = f"risk-{rc}"
            lbl_cls = f"risk-label-{rc}"
            icon = {"high":"🔴","medium":"🟡","low":"🟢"}.get(rc,"⚪")

            with st.expander(
                f"{icon} {item['q'][:70]}{'…' if len(item['q'])>70 else ''}  "
                f"· {item['ts']} · {item['lat']}s",
                expanded=(idx == 0),
            ):
                st.markdown(
                    f'<div class="{cls}">'
                    f'<div class="{lbl_cls}">RISK LEVEL: {rc.upper()}</div>'
                    f'<div class="answer-body" style="margin-top:0.7rem">{item["a"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Copy button workaround
                st.code(item["a"], language="")
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem 1rem;margin-top:0.5rem">
          <div style="font-size:1.8rem;margin-bottom:0.5rem">💬</div>
          <div style="color:#7d8590;font-size:0.88rem">Select a preset or type a question,<br>then click <b>Analyse Contract</b>.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  S.NO 365 · NVIDIA DLI Advanced DL / GenAI with LLMs ·
  RAG · FAISS · LangChain · OpenAI · Streamlit
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  requirements.txt  (create this file in the same repo)
# ══════════════════════════════════════════════════════════════
#
# streamlit>=1.35.0
# langchain>=0.2.0
# langchain-openai>=0.1.0
# langchain-community>=0.2.0
# faiss-cpu>=1.8.0
# pypdf>=4.0.0
# docx2txt>=0.8
# openai>=1.30.0
# tiktoken>=0.7.0
# tenacity>=8.3.0
#

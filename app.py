import os, json, time, hashlib, logging, datetime, tempfile, io
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

from tenacity import retry, stop_after_attempt, wait_exponential


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBED_MODEL = "text-embedding-3-small"

RISKY_KEYWORDS = [
    "indemnif", "liabilit", "penalty", "terminat", "arbitrat",
    "force majeure", "warrant", "confidential", "exclusiv",
    "non-compet", "liquidat", "automatic renewal", "unilateral",
    "waiver", "lien", "intellectual property", "ownership", "assign",
    "unlimited", "irrevocable", "perpetual"
]

REQUIRED_OBLIGATIONS = [
    "payment terms", "delivery schedule", "governing law",
    "dispute resolution", "termination clause", "confidentiality",
    "intellectual property rights", "liability cap",
    "warranty", "indemnification", "notice period"
]

INJECTION_PATTERNS = [
    "ignore previous", "disregard your", "you are now",
    "forget everything", "system prompt", "jailbreak",
    "ignore all instructions", "override instructions"
]

PRESET_QUERIES = {
    "🔍 Full Risk Review": "Review this entire contract. List all risky clauses and missing obligations. End with RISK LEVEL: Low / Medium / High.",
    "⚖️ Liability & Indemnification": "What does the contract say about liability caps and indemnification?",
    "🚪 Termination Rights": "Summarise termination rights for both parties.",
    "💰 Payment & Penalties": "What are the payment terms, deadlines, and penalties?",
    "🧠 IP Ownership": "Who owns intellectual property created under this contract?",
    "🌍 Governing Law & Disputes": "What governing law applies and how are disputes resolved?",
    "🔒 Confidentiality": "Summarise confidentiality and non-disclosure obligations.",
    "📋 Missing Obligations Checklist": "Check payment terms, delivery schedule, governing law, dispute resolution, termination, confidentiality, IP rights, liability cap, warranty, indemnification."
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Contract Review Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}
.hero {
    background: linear-gradient(135deg,#0d1117,#161b22,#1a1f29);
    border: 1px solid #21262d;
    border-radius: 18px;
    padding: 35px 20px;
    text-align: center;
    margin-bottom: 25px;
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #f0c040;
}
.hero-sub {
    color: #8b949e;
    margin-top: 8px;
}
.badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 6px 3px;
    font-size: 12px;
}
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}
.card-title {
    font-size: 14px;
    font-weight: 800;
    color: #f0c040;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.risk-high {
    background: #2d1117;
    border-left: 5px solid #f85149;
    padding: 15px;
    border-radius: 8px;
}
.risk-medium {
    background: #271d0e;
    border-left: 5px solid #e3b341;
    padding: 15px;
    border-radius: 8px;
}
.risk-low {
    background: #0d1f17;
    border-left: 5px solid #3fb950;
    padding: 15px;
    border-radius: 8px;
}
.tag {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    padding: 4px 10px;
    border-radius: 20px;
    margin: 3px;
    font-size: 12px;
}
.red {
    color: #f85149;
    border-color: #f85149;
}
.yellow {
    color: #e3b341;
    border-color: #e3b341;
}
.answer {
    white-space: pre-wrap;
    line-height: 1.7;
    color: #c9d1d9;
}
.footer {
    text-align: center;
    color: #7d8590;
    font-size: 12px;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #21262d;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "store" not in st.session_state:
    st.session_state.store = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "index_key" not in st.session_state:
    st.session_state.index_key = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "kpi" not in st.session_state:
    st.session_state.kpi = {
        "queries": 0,
        "total_lat": 0.0,
        "blocked": 0
    }


def audit(event: str, payload: dict):
    entry = {
        "ts": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **payload
    }
    st.session_state.audit_log.append(entry)
    logger.info("%s %s", event, payload)


def file_fingerprint(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()[:10]
    uploaded_file.seek(0)
    return h


def load_uploaded_file(uploaded_file) -> List[Document]:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in [".docx", ".doc"]:
        loader = Docx2txtLoader(tmp_path)
    elif suffix == ".csv":
        loader = CSVLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = uploaded_file.name

    return docs


@st.cache_resource(show_spinner=False)
def build_vector_store(chunks_json: str, api_key: str):
    chunks = [
        Document(page_content=c["text"], metadata=c["meta"])
        for c in json.loads(chunks_json)
    ]

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=api_key
    )

    return FAISS.from_documents(chunks, embeddings)


CONTRACT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a senior legal AI assistant specialised in contract review.

Rules:
- Use only the provided contract context.
- Do not invent clauses.
- Flag risky clauses clearly.
- Identify missing obligations.
- Mention clause or section numbers if visible.
- End with: RISK LEVEL: Low / Medium / High — short reason.

Contract Context:
{context}

Question:
{question}

Answer:
"""
)


def build_chain(store, api_key: str, model: str, top_k: int):
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=1500,
        api_key=api_key
    )

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CONTRACT_PROMPT}
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def run_query(chain, query: str):
    t0 = time.perf_counter()
    result = chain.invoke({"query": query})
    return {
        "answer": result["result"],
        "latency": round(time.perf_counter() - t0, 2)
    }


def is_injection(text: str):
    lo = text.lower()
    return any(p in lo for p in INJECTION_PATTERNS)


def rule_scan_risky(text: str):
    lo = text.lower()
    return [kw for kw in RISKY_KEYWORDS if kw in lo]


def scan_missing(text: str):
    lo = text.lower()
    return [ob for ob in REQUIRED_OBLIGATIONS if ob not in lo]


def risk_color(answer: str):
    up = answer.upper()
    if "RISK LEVEL: HIGH" in up:
        return "high"
    if "RISK LEVEL: MEDIUM" in up:
        return "medium"
    return "low"


st.markdown("""
<div class="hero">
    <div class="hero-title">⚖️ Contract Review Assistant</div>
    <div class="hero-sub">AI-powered risky clause detection and missing obligation audit</div>
    <div>
        <span class="badge">RAG</span>
        <span class="badge">FAISS</span>
        <span class="badge">LangChain</span>
        <span class="badge">OpenAI</span>
        <span class="badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 🔑 API Key")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        placeholder="sk-..."
    )

    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )

    top_k = st.slider("Retrieval Top-K", 3, 10, TOP_K)
    chunk_size = st.slider("Chunk Size", 400, 1200, CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, CHUNK_OVERLAP, step=25)

    st.markdown("### 📋 Audit Log")

    if st.session_state.audit_log:
        for e in st.session_state.audit_log[-10:]:
            st.caption(f"{e['ts']} — {e['event']}")

        log_data = "\n".join(json.dumps(e) for e in st.session_state.audit_log)
        st.download_button(
            "Download Audit Log",
            data=log_data,
            file_name="audit.jsonl",
            mime="application/jsonl"
        )
    else:
        st.caption("No logs yet.")

    if st.button("Reset App"):
        st.session_state.store = None
        st.session_state.chain = None
        st.session_state.full_text = ""
        st.session_state.file_name = ""
        st.session_state.index_key = ""
        st.session_state.history = []
        st.session_state.kpi = {
            "queries": 0,
            "total_lat": 0.0,
            "blocked": 0
        }
        build_vector_store.clear()
        audit("reset", {})
        st.success("Reset done.")


left, right = st.columns([1, 1.15], gap="large")


with left:
    st.markdown('<div class="card-title">📂 Upload Contract</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or CSV",
        type=["pdf", "docx", "doc", "txt", "csv"]
    )

    if uploaded and not api_key:
        st.error("Add your OpenAI API key first.")

    elif uploaded and api_key:
        fp = file_fingerprint(uploaded)
        new_key = f"{uploaded.name}_{fp}"

        if new_key != st.session_state.index_key:
            with st.spinner("Loading document..."):
                try:
                    docs = load_uploaded_file(uploaded)
                    audit("document_uploaded", {
                        "filename": uploaded.name,
                        "pages": len(docs),
                        "hash": fp
                    })
                except Exception as e:
                    st.error(f"File loading error: {e}")
                    st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " "]
            )

            chunks = splitter.split_documents(docs)
            full_text = " ".join(d.page_content for d in docs)

            chunks_json = json.dumps([
                {"text": c.page_content, "meta": c.metadata}
                for c in chunks
            ])

            with st.spinner("Building vector index..."):
                try:
                    store = build_vector_store(chunks_json, api_key)
                    chain = build_chain(store, api_key, model_choice, top_k)
                except Exception as e:
                    st.error(f"Index/API error: {e}")
                    st.stop()

            st.session_state.store = store
            st.session_state.chain = chain
            st.session_state.full_text = full_text
            st.session_state.file_name = uploaded.name
            st.session_state.index_key = new_key

            audit("index_built", {"chunks": len(chunks)})

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Loaded File</div>
            <b>{st.session_state.file_name}</b><br>
            <small>Fingerprint: {fp}</small>
        </div>
        """, unsafe_allow_html=True)

        risky = rule_scan_risky(st.session_state.full_text)
        missing = scan_missing(st.session_state.full_text)

        if risky:
            tags = "".join([f'<span class="tag red">{x}</span>' for x in risky[:15]])
            st.markdown(f"""
            <div class="risk-medium">
                <b>⚠️ Risky Keywords Found</b><br><br>
                {tags}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                <b>✅ No risky keywords found</b>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        if missing:
            tags = "".join([f'<span class="tag yellow">{x}</span>' for x in missing])
            st.markdown(f"""
            <div class="risk-high">
                <b>🚨 Missing Obligations</b><br><br>
                {tags}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                <b>✅ Standard obligations appear present</b>
            </div>
            """, unsafe_allow_html=True)

        kpi = st.session_state.kpi
        avg_latency = kpi["total_lat"] / max(kpi["queries"], 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", kpi["queries"])
        c2.metric("Avg Latency", f"{avg_latency:.1f}s")
        c3.metric("Risk Terms", len(risky))
        c4.metric("Blocked", kpi["blocked"])

    else:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <h2>📑</h2>
            Upload a contract to begin.
        </div>
        """, unsafe_allow_html=True)


with right:
    st.markdown('<div class="card-title">🔍 Ask Assistant</div>', unsafe_allow_html=True)

    preset_label = st.selectbox(
        "Preset",
        ["✏️ Custom query..."] + list(PRESET_QUERIES.keys())
    )

    default_q = PRESET_QUERIES.get(preset_label, "")

    user_query = st.text_area(
        "Question",
        value=default_q,
        height=120,
        placeholder="Example: What are the termination conditions?"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        go = st.button(
            "⚡ Analyse Contract",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.chain is None
        )

    with col2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.history = []

    if go and user_query.strip():
        if st.session_state.chain is None:
            st.error("Upload a contract first.")

        elif is_injection(user_query):
            st.error("Prompt injection detected. Query blocked.")
            st.session_state.kpi["blocked"] += 1
            audit("injection_blocked", {"query": user_query[:80]})

        else:
            with st.spinner("Analysing contract..."):
                try:
                    res = run_query(st.session_state.chain, user_query)
                    answer = res["answer"]
                    latency = res["latency"]
                    rc = risk_color(answer)

                    st.session_state.kpi["queries"] += 1
                    st.session_state.kpi["total_lat"] += latency

                    audit("query_answered", {
                        "query": user_query[:100],
                        "latency": latency,
                        "risk": rc
                    })

                    st.session_state.history.insert(0, {
                        "q": user_query,
                        "a": answer,
                        "lat": latency,
                        "risk": rc,
                        "ts": datetime.datetime.now().strftime("%H:%M:%S")
                    })

                except Exception as e:
                    st.error(f"API error: {e}")
                    audit("query_error", {"error": str(e)})

    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            icon = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(item["risk"], "⚪")

            with st.expander(
                f"{icon} {item['q'][:70]} · {item['ts']} · {item['lat']}s",
                expanded=idx == 0
            ):
                st.markdown(f"""
                <div class="risk-{item['risk']}">
                    <b>RISK LEVEL: {item['risk'].upper()}</b>
                    <div class="answer">{item['a']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.code(item["a"], language="")

    else:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <h2>💬</h2>
            Select a preset or type your own question.
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    S.NO 365 · Contract Review Assistant · RAG · FAISS · LangChain · OpenAI · Streamlit
</div>
""", unsafe_allow_html=True)

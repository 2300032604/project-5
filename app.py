import streamlit as st
import PyPDF2
import docx
import pandas as pd

st.set_page_config(page_title="Contract Review Assistant", page_icon="⚖️", layout="wide")

RISKY_KEYWORDS = [
    "unlimited liability", "no liability cap", "indemnify", "indemnification",
    "automatic renewal", "penalty", "late fee", "termination",
    "non-compete", "exclusive", "irrevocable", "perpetual",
    "intellectual property", "assignment", "arbitration"
]

REQUIRED_OBLIGATIONS = [
    "payment", "delivery", "governing law", "dispute resolution",
    "termination", "confidentiality", "intellectual property",
    "liability", "warranty", "indemnification", "notice"
]

def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    document = docx.Document(file)
    return "\n".join([p.text for p in document.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        return read_pdf(file)
    elif name.endswith(".docx"):
        return read_docx(file)
    elif name.endswith(".txt"):
        return read_txt(file)
    elif name.endswith(".csv"):
        return read_csv(file)
    else:
        return ""

def find_risks(text):
    found = []
    low = text.lower()

    for word in RISKY_KEYWORDS:
        if word in low:
            found.append(word)

    return found

def find_missing(text):
    missing = []
    low = text.lower()

    for item in REQUIRED_OBLIGATIONS:
        if item not in low:
            missing.append(item)

    return missing

def simple_answer(text, question):
    q = question.lower()
    low = text.lower()

    if "payment" in q:
        keyword = "payment"
    elif "termination" in q:
        keyword = "termination"
    elif "liability" in q:
        keyword = "liability"
    elif "indemnification" in q or "indemnify" in q:
        keyword = "indemn"
    elif "confidential" in q:
        keyword = "confidential"
    elif "intellectual" in q or "ip" in q:
        keyword = "intellectual"
    elif "governing" in q or "law" in q:
        keyword = "governing"
    elif "dispute" in q or "arbitration" in q:
        keyword = "dispute"
    elif "warranty" in q:
        keyword = "warranty"
    else:
        keyword = ""

    lines = text.splitlines()
    matched = []

    for line in lines:
        if keyword and keyword in line.lower():
            matched.append(line.strip())

    if matched:
        return "\n".join(matched[:8])
    else:
        return "No exact matching clause found. Try asking about payment, termination, liability, indemnification, confidentiality, IP, governing law, or dispute resolution."

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: white;
}
.main-title {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
}
.box {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1e293b;
}
.risk {
    background: #3f1d1d;
    padding: 12px;
    border-left: 5px solid #ef4444;
    border-radius: 8px;
}
.safe {
    background: #123524;
    padding: 12px;
    border-left: 5px solid #22c55e;
    border-radius: 8px;
}
.warn {
    background: #3b2f12;
    padding: 12px;
    border-left: 5px solid #eab308;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ Contract Review Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">No API Key Needed · Rule-Based Contract Risk Checker</div>', unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader(
    "Upload contract file",
    type=["pdf", "docx", "txt", "csv"]
)

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("Could not read text from this file.")
        st.stop()

    st.success("File uploaded and scanned successfully.")

    risky = find_risks(text)
    missing = find_missing(text)

    col1, col2, col3 = st.columns(3)

    col1.metric("Risky Terms Found", len(risky))
    col2.metric("Missing Obligations", len(missing))
    col3.metric("Characters Scanned", len(text))

    st.write("---")

    left, right = st.columns(2)

    with left:
        st.subheader("🚨 Risky Clauses / Keywords")

        if risky:
            for r in risky:
                st.markdown(f'<div class="risk">⚠️ {r}</div><br>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe">✅ No risky keywords found.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("📋 Missing Obligations")

        if missing:
            for m in missing:
                st.markdown(f'<div class="warn">⚠️ Missing: {m}</div><br>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe">✅ All standard obligations found.</div>', unsafe_allow_html=True)

    st.write("---")

    st.subheader("🔍 Ask Simple Questions")

    question = st.text_input(
        "Ask about payment, termination, liability, indemnification, confidentiality, IP, governing law, dispute resolution, warranty"
    )

    if st.button("Analyse"):
        if question.strip():
            answer = simple_answer(text, question)
            st.markdown("### Answer")
            st.info(answer)
        else:
            st.warning("Please type a question.")

    with st.expander("📄 View Extracted Contract Text"):
        st.text_area("Contract Text", text, height=300)

else:
    st.info("Upload a PDF, DOCX, TXT, or CSV contract to begin.")

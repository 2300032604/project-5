import streamlit as st
import PyPDF2
import docx
import pandas as pd

st.set_page_config(page_title="Contract Review Assistant", page_icon="⚖️", layout="wide")

RISKY_KEYWORDS = [
    "unlimited liability", "no liability cap", "indemnify", "indemnification",
    "automatic renewal", "penalty", "late fee", "termination",
    "non-compete", "exclusive", "irrevocable", "perpetual",
    "intellectual property", "assignment", "arbitration",
    "unilateral", "liquidated damages", "waiver"
]

HIGH_RISK = [
    "unlimited liability", "no liability cap", "unlimited indemnification",
    "irrevocable", "perpetual", "non-compete"
]

MEDIUM_RISK = [
    "automatic renewal", "penalty", "late fee", "unilateral",
    "termination", "liquidated damages", "assignment"
]

LOW_RISK = [
    "confidentiality", "governing law", "dispute resolution",
    "warranty", "notice", "force majeure"
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
    return ""

def find_items(text, items):
    low = text.lower()
    return [item for item in items if item in low]

def find_missing(text):
    low = text.lower()
    return [item for item in REQUIRED_OBLIGATIONS if item not in low]

def get_risk_level(high_count, medium_count, missing_count):
    if high_count >= 2 or missing_count >= 5:
        return "HIGH", "🔴"
    elif high_count == 1 or medium_count >= 2 or missing_count >= 3:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"

def simple_answer(text, question):
    q = question.lower()

    keywords = {
        "payment": "payment",
        "termination": "termination",
        "liability": "liability",
        "indemnification": "indemn",
        "confidentiality": "confidential",
        "ip": "intellectual",
        "intellectual": "intellectual",
        "governing": "governing",
        "law": "governing",
        "dispute": "dispute",
        "arbitration": "arbitration",
        "warranty": "warranty",
        "notice": "notice"
    }

    keyword = ""
    for k, v in keywords.items():
        if k in q:
            keyword = v
            break

    matched = []
    for line in text.splitlines():
        if keyword and keyword in line.lower():
            matched.append(line.strip())

    if matched:
        return "\n".join(matched[:10])

    return "No exact matching clause found. Try asking about payment, termination, liability, indemnification, confidentiality, IP, governing law, dispute resolution, warranty, or notice."

st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: white;
}
.title {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-bottom: 15px;
}
.risk {
    background: #3f1d1d;
    padding: 12px;
    border-left: 5px solid #ef4444;
    border-radius: 8px;
    margin-bottom: 8px;
}
.warn {
    background: #3b2f12;
    padding: 12px;
    border-left: 5px solid #eab308;
    border-radius: 8px;
    margin-bottom: 8px;
}
.safe {
    background: #123524;
    padding: 12px;
    border-left: 5px solid #22c55e;
    border-radius: 8px;
    margin-bottom: 8px;
}
.big-risk {
    text-align: center;
    font-size: 35px;
    font-weight: 800;
    padding: 25px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚖️ Contract Review Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">No API Key · Rule-Based Risk Detection · Risk Level Graph</div>', unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("Upload contract file", type=["pdf", "docx", "txt", "csv"])

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("Could not read text from this file.")
        st.stop()

    st.success("File uploaded and scanned successfully.")

    risky_found = find_items(text, RISKY_KEYWORDS)
    high_found = find_items(text, HIGH_RISK)
    medium_found = find_items(text, MEDIUM_RISK)
    low_found = find_items(text, LOW_RISK)
    missing = find_missing(text)

    risk_level, risk_icon = get_risk_level(len(high_found), len(medium_found), len(missing))

    st.markdown(f"""
    <div class="big-risk">
        Overall Risk Level: {risk_icon} {risk_level}
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Risk", len(high_found))
    c2.metric("Medium Risk", len(medium_found))
    c3.metric("Low / Safe Clauses", len(low_found))
    c4.metric("Missing Obligations", len(missing))

    st.write("---")

    graph_data = pd.DataFrame({
        "Risk Type": ["High Risk", "Medium Risk", "Low / Safe", "Missing Obligations"],
        "Count": [len(high_found), len(medium_found), len(low_found), len(missing)]
    })

    st.subheader("📊 Risk Level Graph")
    st.bar_chart(graph_data.set_index("Risk Type"))

    st.write("---")

    left, right = st.columns(2)

    with left:
        st.subheader("🚨 Risky Clauses Found")

        if risky_found:
            for item in risky_found:
                st.markdown(f'<div class="risk">⚠️ {item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe">✅ No risky clauses found.</div>', unsafe_allow_html=True)

    with right:
        st.subheader("📋 Missing Obligations")

        if missing:
            for item in missing:
                st.markdown(f'<div class="warn">⚠️ Missing: {item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe">✅ All standard obligations found.</div>', unsafe_allow_html=True)

    st.write("---")

    st.subheader("🔍 Ask Simple Questions")

    question = st.text_input("Ask about payment, termination, liability, indemnification, confidentiality, IP, governing law, dispute resolution, warranty")

    if st.button("Analyse"):
        if question.strip():
            answer = simple_answer(text, question)
            st.info(answer)
        else:
            st.warning("Please type a question.")

    with st.expander("📄 View Extracted Contract Text"):
        st.text_area("Extracted Text", text, height=300)

else:
    st.info("Upload a PDF, DOCX, TXT, or CSV contract to begin.")

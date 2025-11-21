

import streamlit as st
import pandas as pd
import io, re, string, os
import PyPDF2
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Page config & small CSS
# ------------------------------
st.set_page_config(page_title="RecruitLense — Resume Matcher", layout="wide", page_icon="🔎")

# Simple CSS to improve look
st.markdown(
    """
    <style>
    /* Base typography */
    .title { font-size:32px !important; font-weight:700 !important; }
    .subtitle { font-size:14px !important; margin-bottom:8px; }

    /* Card look used in both themes */
    .card {
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(2,6,23,0.06);
        margin-bottom: 12px;
    }

    /* Inputs / uploader rounded */
    .stFileUploader, .stTextInput, .stMultiSelect, .stSelectbox {
        border-radius: 8px !important;
    }

    /* Light theme defaults */
    @media (prefers-color-scheme: light) {
        .title { color: #0f172a !important; }
        .subtitle { color: #334155 !important; }
        .card { background: #ffffff; color: #0f172a; }
        .stAlert, .stInfo { color: #0f172a; }
        .stDownloadButton>button { background-color: #0f172a !important; color: #fff !important; }
        .stMetric>div>div { color: #0f172a !important; }
        .stDataFrame, .css-1d391kg { color: #0f172a !important; }
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        /* page background note: streamlit sets its own; we only alter components */
        .title { color: #e6f0ff !important; }
        .subtitle { color: #cbd5e1 !important; }
        .card {
            background: #0b1220 !important;
            color: #e6eef8 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.45);
        }
        .stInfo, .stAlert {
            background: rgba(255,255,255,0.03) !important;
            color: #dbeafe !important;
            border-left: 3px solid rgba(99,102,241,0.9) !important;
        }
        .stDownloadButton>button {
            background-color: #1f2937 !important;
            color: #e6eef8 !important;
        }
        /* Uploader box contrast */
        .css-1dp5vir { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.04) !important; }
        /* Dataframe header */
        .stDataFrame table thead th {
            background: rgba(255,255,255,0.03) !important;
            color: #e6eef8 !important;
        }
        /* Metric color */
        .stMetric>div>div { color: #e6eef8 !important; }
        /* Sidebar clearer */
        .css-1d391kg { color: #e6eef8 !important; }
    }

    /* Small responsive tweaks */
    @media (max-width: 900px) {
        .title { font-size: 24px !important; }
    }

    div[data-testid="column"] > div:first-child {
    background: #ffffff !important;
    border-radius: 12px;
    padding: 10px;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Text cleaning
# ------------------------------
def clean_text(t):
    if not t:
        return ""
    t = str(t).replace("\n", " ").replace("\r", " ").lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ------------------------------
# Extractors (OCR-enabled)
# ------------------------------
def extract_pdf(file_bytes, ocr=False):
    text = ""
    # Try PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            p = page.extract_text()
            if p: text += p + "\n"
    except: pass
    if text.strip(): return text

    # Try pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                p = page.extract_text()
                if p: text += p + "\n"
    except: pass
    if text.strip(): return text

    # OCR fallback
    if ocr:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                ocr_text = ""
                for page in pdf.pages:
                    img = page.to_image(resolution=200).original
                    ocr_text += pytesseract.image_to_string(img) + "\n"
                return ocr_text
        except: return ""
    return ""

def extract_docx(file_bytes):
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except: return ""

def extract_txt(file_bytes):
    try: return file_bytes.decode("utf-8", errors="ignore")
    except: return ""

def extract_rtf(file_bytes):
    try:
        raw = file_bytes.decode("utf-8", errors="ignore")
        cleaned = re.sub(r"\\[a-zA-Z]+\d* ?", " ", raw)
        cleaned = re.sub(r"[{}]", " ", cleaned)
        return cleaned
    except: return ""

def extract_image(file_bytes, ocr=False):
    if not ocr: return ""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except: return ""

def extract_any(file, use_ocr):
    name = file.name.lower()
    fb = file.read()
    if name.endswith(".pdf"): return extract_pdf(fb, ocr=use_ocr)
    if name.endswith(".docx"): return extract_docx(fb)
    if name.endswith(".txt"): return extract_txt(fb)
    if name.endswith(".rtf"): return extract_rtf(fb)
    if name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")): return extract_image(fb, ocr=use_ocr)
    return ""

# ------------------------------
# Column Auto-Detection
# ------------------------------
def find_jd_col(df):
    # Priority keys for Job Description
    keys = ["description", "jd", "responsibil", "require", "profile", "job_desc", "job description"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    # Fallback: longest text column
    txt_cols = [c for c in df.columns if df[c].dtype == object]
    if txt_cols:
        return max(txt_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return None

def find_title_col(df):
    # Priority keys for Job Title / Designation
    keys = ["designation", "title", "role", "position", "job name", "post"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None

# ------------------------------
# Top area: title, subtitle, features, sidebar steps
# ------------------------------
st.markdown('<div class="title">🔎 RecruitLense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Resume ↔ Job Description matching • OCR-enabled • TF-IDF ranking • Docker-ready demo</div>', unsafe_allow_html=True)
st.markdown("")  # spacer

# Features card and non-technical explanation
with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**What this app does:**")
        st.info("RecruitLense reads a resume and a job description, identifies important keywords and phrases, compares them, and returns a relevance score that shows how well the resume fits the role. It supports scanned resumes via OCR.")
        st.markdown("**Quick tech summary:** TF-IDF vectorization, cosine similarity for scoring, Tesseract OCR for images/PDFs, and a Streamlit UI with Docker-ready backend.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Core features**")
        st.write("- OCR for scanned resumes")
        st.write("- Multiple file formats: PDF, DOCX, TXT, RTF, PNG, JPG, TIFF")
        st.write("- Auto Job Description column detection for Excel/CSV JDs")
        st.write("- TF-IDF + cosine similarity scoring")
        st.markdown("</div>", unsafe_allow_html=True)

# Sidebar with step-by-step checklist
with st.sidebar:
    st.header("How to use RecruitLense")
    st.markdown("**Step 1:** Upload a resume (PDF/DOCX/TXT/Image).")
    st.markdown("**Step 2:** Upload a JD file (Excel/CSV/PDF/Doc/Image).")
    st.markdown("**Step 3:** Toggle OCR if the document is scanned or an image (recommended for images).")
    st.markdown("**Step 4:** View extracted text, match scores, and download results.")
    st.divider()
    st.markdown("**Tips:**")
    st.markdown("- For Excel/CSV JD files, have a `Description` column for best results.")
    st.markdown("- Use sample data to test performance on many rows.")
    st.markdown("- This is a baseline model — consider contextual embeddings (BERT) for better semantic matching.")
    st.divider()
    st.markdown("Made by Arghyadeep ❤️ ")

# ------------------------------
# Main Uploader area
# ------------------------------
st.write("")  # spacer
col1, col2 = st.columns(2)
with col1:
    resume = st.file_uploader("Upload Resume", type=["pdf","docx","txt","rtf","png","jpg","jpeg","tiff","bmp"])
with col2:
    jd_file = st.file_uploader("Upload Job Description File (Excel/CSV/PDF/Doc/Image)", type=["xlsx","csv","pdf","docx","txt","png","jpg","jpeg","tiff","bmp"])

use_ocr = st.checkbox("Enable OCR (slower — use for scanned or image files)", value=True)

# ------------------------------
# Matching logic (unchanged core)
# ------------------------------
if resume and jd_file:
    # 1. Process JD File
    try:
        if jd_file.name.lower().endswith(".xlsx"):
            jd_df = pd.read_excel(jd_file)
        elif jd_file.name.lower().endswith(".csv"):
            jd_df = pd.read_csv(jd_file)
        else:
            # Non-tabular JD (PDF/Image/Doc) - treat as single entry
            jd_text = extract_any(jd_file, use_ocr)
            jd_df = pd.DataFrame({"Job Description": [jd_text], "Designation": ["Uploaded File"]})
    except Exception as e:
        st.error(f"Could not read JD file: {e}")
        st.stop()

    # 2. Identify Columns
    jd_col = find_jd_col(jd_df)
    title_col = find_title_col(jd_df)

    if not jd_col:
        st.error("Could not find a 'Description' column in the Job Description file. Please upload a file with a Description column or upload a single JD document (PDF/DOCX/TXT).")
        st.stop()

    # 3. Process Resume
    with st.spinner("Extracting resume text..."):
        resume_text = extract_any(resume, use_ocr)

    if not resume_text.strip():
        st.error("Resume text is empty. Try enabling OCR or upload a different file.")
        st.stop()

    # Show extracted text preview and allow toggling full view
    with st.expander("View Extracted Resume Text (preview)"):
        st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))

    # 4. Match
    cleaned_resume = clean_text(resume_text)
    jd_df["cleaned_jd"] = jd_df[jd_col].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    # Combine resume + all JDs to build corpus
    corpus = [cleaned_resume] + jd_df["cleaned_jd"].tolist()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity (Resume is index 0, JDs are 1..N)
    doc_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    jd_df["Match Score (%)"] = (doc_sim * 100).round(2)

    # 5. Sort and Display
    out = jd_df.sort_values("Match Score (%)", ascending=False)

    # Dynamic Column Display
    cols_to_show = ["Match Score (%)"]
    if title_col:
        cols_to_show.append(title_col)
    cols_to_show.append(jd_col)

    st.success(f"Matched against {len(jd_df)} job descriptions.")
    # Show top matches in a left column, details on the right
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.subheader("Top Matches")
        st.dataframe(out[cols_to_show].head(50).reset_index(drop=True), use_container_width=True)
    with right_col:
        st.subheader("Result Summary")
        st.metric("Top score (%)", float(out["Match Score (%)"].iloc[0]) if not out.empty else 0.0)
        st.markdown("**What this score means:** higher = more words/phrases overlap in important sections; baseline TF-IDF model may miss deep semantic matches that BERT could capture.")

    # Download
    csv_data = out.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        csv_data,
        "resume_matches.csv",
        "text/csv"
    )

else:
    st.info("Upload a Resume and a Job Description file to get started. Use the sidebar for tips and steps.")


# app.py — Resume ↔ JD Matcher with Full OCR (Docker-ready)
# Features:
# - OCR on images & scanned PDFs
# - PDF/DOCX/TXT/RTF/PNG/JPG/TIFF support for both resume and JD
# - Auto JD column detection (Description & Designation)
# - TF-IDF + cosine similarity (no transformers)
# - Preview extracted text
# - Clean & simple UI

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

st.set_page_config(page_title="Resume Matcher (OCR Enabled)", layout="wide")

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
    keys = ["description", "jd", "responsibil", "require", "profile", "job_desc"]
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
# UI
# ------------------------------
st.title("🔎 Resume Matcher (Full OCR)")
st.write("Upload your Resume and a Job Description file (Excel/CSV/PDF/Image).")

col1, col2 = st.columns(2)
with col1:
    resume = st.file_uploader("Upload Resume", type=["pdf","docx","txt","rtf","png","jpg"])
with col2:
    jd_file = st.file_uploader("Upload JD File", type=["xlsx","csv","pdf","docx","txt","png","jpg"])

use_ocr = st.checkbox("Enable OCR (Slows down processing but reads images)", value=True)

if resume and jd_file:
    # 1. Process JD File
    if jd_file.name.endswith(".xlsx"):
        jd_df = pd.read_excel(jd_file)
    elif jd_file.name.endswith(".csv"):
        jd_df = pd.read_csv(jd_file)
    else:
        # Non-tabular JD (PDF/Image/Doc) - treat as single entry
        jd_text = extract_any(jd_file, use_ocr)
        jd_df = pd.DataFrame({"Job Description": [jd_text], "Designation": ["Uploaded File"]})

    # 2. Identify Columns
    jd_col = find_jd_col(jd_df)
    title_col = find_title_col(jd_df)

    if not jd_col:
        st.error("Could not find a 'Description' column in the JD file.")
        st.stop()

    # 3. Process Resume
    with st.spinner("Extracting resume text..."):
        resume_text = extract_any(resume, use_ocr)
    
    if not resume_text.strip():
        st.error("Resume text is empty. Try enabling OCR.")
        st.stop()

    with st.expander("View Extracted Resume Text"):
        st.text(resume_text[:1000] + "...")

    # 4. Match
    cleaned_resume = clean_text(resume_text)
    jd_df["cleaned_jd"] = jd_df[jd_col].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    # Combine resume + all JDs to build corpus
    corpus = [cleaned_resume] + jd_df["cleaned_jd"].tolist()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity (Resume is index 0, JDs are 1..N)
    # We compare index 0 against 1..N
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
    st.dataframe(out[cols_to_show].head(50), use_container_width=True)

    st.download_button(
        "Download Results CSV",
        out.to_csv(index=False),
        "resume_matches.csv",
        "text/csv"
    )
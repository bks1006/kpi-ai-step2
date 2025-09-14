from __future__ import annotations

import io
import os
import re
import json
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- Optional OCR fallback ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- LLM toggle & setup ----------
USE_OPENAI = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if USE_OPENAI and not OPENAI_API_KEY:
    st.info("OPENAI_API_KEY not found. Running in heuristic mode.")
    USE_OPENAI = False

if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_MODEL = "gpt-4o-mini"
    except Exception:
        st.info("OpenAI SDK unavailable. Running in heuristic mode.")
        USE_OPENAI = False

# ---------- Demo credentials ----------
VALID_USERS = {
    "admin@company.com": "password123",
    "user@company.com": "welcome123"
}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System — LLM", layout="wide")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    :root { --brand:#b91c1c; --green:#16a34a; --red:#b91c1c; }

    /* Give breathing room so title is not clipped */
    [data-testid="stAppViewContainer"] .main .block-container {
      padding-top: 2.75rem !important;
    }

    /* App title */
    .app-title {
      color: #b91c1c;
      text-align: center;
      font-weight: 800;
      line-height: 1.15;
      margin: 0.75rem 0 1rem 0;
      font-size: 2.25rem;
    }

    /* Sticky top bar */
    .topbar { position: sticky; top: 0; z-index: 5; background: white;
              padding: 6px 0 8px; margin-bottom: 4px; border-bottom: 1px solid #eee; }
    .topbar-inner { display:flex; justify-content:space-between; align-items:center; }
    .who { color:#6b7280; font-size:14px; }

    /* Section tables */
    .th-row { background:#f3f4f6; border:1px solid #e5e7eb; border-bottom:0;
              padding:10px 12px; border-radius:10px 10px 0 0; font-weight:700; display:grid; }
    .tb { border:1px solid #e5e7eb; border-top:0; border-radius:0 0 10px 10px; }
    .cell { padding:10px 12px; border-top:1px solid #e5e7eb; }

    /* Status chips */
    .chip { display:inline-block; padding:4px 10px; border-radius:999px; color:#fff; font-size:12px;}
    .chip-pending{ background:#9ca3af;}
    .chip-ok{ background:#16a34a;}
    .chip-bad{ background:#b91c1c;}

    /* ============== LOGIN (scoped) ============== */
    .login-card .stTextInput > div > div,
    .login-card .stPasswordInput > div > div {
      border: 1px solid #d1d5db !important;
      border-radius: 8px !important;
      box-shadow: none !important;
      padding: 0 !important;
      background: #fff !important;
    }
    .login-card .stTextInput input,
    .login-card .stPasswordInput input {
      border: none !important;
      outline: none !important;
      box-shadow: none !important;
      padding: 10px 12px !important;
      width: 100% !important;
    }
    /* Remove highlighting entirely (no focus styles) */
    .login-card .stTextInput > div > div:focus-within,
    .login-card .stPasswordInput > div > div:focus-within {
      border: 1px solid #d1d5db !important;
      box-shadow: none !important;
    }
    .login-card .stButton > button {
      background: var(--brand) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.6rem 1rem !important;
      font-weight: 700 !important;
      width: 100%;
    }
    .login-card .stButton > button:hover { filter: brightness(0.95); }

    /* Validate/Reject */
    .btn-wrap button { background:#f9fafb !important; color:#111827 !important;
                       border:1px solid #e5e7eb !important; border-radius:6px !important;
                       padding:0.4rem 0.8rem !important; font-weight:600 !important; }
    .btn-wrap.on-validate button { background:var(--green)!important; color:#fff!important; }
    .btn-wrap.on-reject button { background:var(--red)!important; color:#fff!important; }

    /* Review & Accept */
    .accept-btn .stButton>button {
      background-color: #b91c1c !important; color: white !important; border: none !important;
      border-radius: 6px !important; padding: 0.6rem 1.2rem !important;
      font-weight: 600 !important; box-shadow: none !important;
    }
    .accept-btn .stButton>button:hover { filter: brightness(0.9); }

    .centered { display:flex; justify-content:center; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Session defaults ----------
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "projects" not in st.session_state:
    st.session_state["projects"] = {}
if "final_kpis" not in st.session_state:
    st.session_state["final_kpis"] = {}

# ---------- Auth ----------
def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

# ---------- Chips ----------
def _chip(status: str) -> str:
    cls = "chip-pending"
    if status == "Validated": cls = "chip-ok"
    elif status == "Rejected": cls = "chip-bad"
    return f"<span class='chip {cls}'>{status}</span>"

# ---------- File reading ----------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if txt.strip():
                return txt
        except Exception:
            pass
        if OCR_AVAILABLE:
            try:
                imgs = convert_from_bytes(data)
                parts = [pytesseract.image_to_string(im, lang="eng") for im in imgs]
                return "\n".join(p for p in parts if p.strip())
            except Exception:
                return ""
        return ""
    if lname.endswith(".docx"):
        try:
            doc = DocxDocument(bio)
        except Exception:
            return ""
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(parts)
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------- Simple fallback KPI extractor ----------
def extract_kpis(text: str) -> pd.DataFrame:
    rows = []
    low = text.lower()
    if "attrition" in low:
        rows.append({"KPI Name":"Attrition Rate","Description":"Track attrition rate.","Target Value":"", "Status":"Pending"})
    if "retention" in low:
        rows.append({"KPI Name":"Retention Rate","Description":"Track retention rate.","Target Value":"", "Status":"Pending"})
    if not rows:
        rows.append({"KPI Name":"Generic KPI","Description":"Placeholder KPI.","Target Value":"", "Status":"Pending"})
    return pd.DataFrame(rows)

# ---------- Login ----------
def login_page():
    st.markdown("<h1 class='app-title'>AI KPI System</h1>", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.write("Sign in to continue")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if _check_credentials(email, password):
                st.session_state["auth"] = True
                st.session_state["user"] = email.strip().lower()
                st.rerun()
            else:
                st.error("Invalid email or password")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================
#        MAIN
# ======================
if not st.session_state["auth"]:
    login_page()
    st.stop()

st.markdown("<div class='topbar'><div class='topbar-inner'>"
            f"<div class='who'>Signed in as <b>{st.session_state.get('user','')}</b></div>"
            "</div></div>", unsafe_allow_html=True)

st.title("AI KPI Extraction Demo")

uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)

if uploads:
    for f in uploads:
        text = read_uploaded(f)
        df = extract_kpis(text)
        st.subheader(f"📄 {f.name}")
        st.dataframe(df)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown("<div class='centered accept-btn'>", unsafe_allow_html=True)
            if st.button("Review & Accept", key=f"accept_{f.name}"):
                st.success("✅ Finalized KPIs have been accepted successfully!")
            st.markdown("</div>", unsafe_allow_html=True)

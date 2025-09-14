import io, re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- OCR fallback ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Demo credentials ----------
VALID_USERS = {
    "admin@company.com": "password123",
    "user@company.com": "welcome123"
}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    :root { --brand:#b91c1c; --green:#16a34a; --red:#b91c1c; }
    .block-container { padding-top: 1.0rem; }

    /* Sticky top bar */
    .topbar {
      position: sticky; top: 0; z-index: 5;
      background: white; padding: 6px 0 8px 0; margin-bottom: 4px;
      border-bottom: 1px solid #eee;
    }
    .topbar-inner { display:flex; justify-content:space-between; align-items:center; }
    .who { color:#6b7280; font-size:14px; }

    /* Section tables */
    .th-row {
      background:#f3f4f6; border:1px solid #e5e7eb; border-bottom:0;
      padding:10px 12px; border-radius:10px 10px 0 0; font-weight:700;
      display:grid;
    }
    .tb { border:1px solid #e5e7eb; border-top:0; border-radius:0 0 10px 10px; }
    .cell { padding:10px 12px; border-top:1px solid #e5e7eb; }

    /* Status chips */
    .chip { display:inline-block; padding:4px 10px; border-radius:999px; color:#fff; font-size:12px;}
    .chip-pending{ backgr

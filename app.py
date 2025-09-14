from __future__ import annotations
import io, os, re, json
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- Optional OCR ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- LLM Setup ----------
USE_OPENAI = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if USE_OPENAI and not OPENAI_API_KEY:
    USE_OPENAI = False
if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_MODEL = "gpt-4o-mini"  # swap to gpt-4.1 if available
    except Exception:
        USE_OPENAI = False

# ---------- Demo credentials ----------
VALID_USERS = {"admin@company.com": "password123", "user@company.com": "welcome123"}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

# ---------- Session defaults ----------
if "auth" not in st.session_state: st.session_state["auth"] = False
if "user" not in st.session_state: st.session_state["user"] = None
if "projects" not in st.session_state: st.session_state["projects"] = {}
if "final_kpis" not in st.session_state: st.session_state["final_kpis"] = {}

# ---------- Utils ----------
def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

# ---------- File reading ----------
def read_text_from_bytes(data: bytes, name: str) -> str:
    bio = io.BytesIO(data); lname = name.lower()
    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if txt.strip(): return txt
        except Exception: pass
        if OCR_AVAILABLE:
            try:
                imgs = convert_from_bytes(data)
                parts = [pytesseract.image_to_string(im, lang="eng") for im in imgs]
                return "\n".join(p for p in parts if p.strip())
            except Exception: return ""
        return ""
    if lname.endswith(".docx"):
        try:
            doc = DocxDocument(bio)
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for tbl in doc.tables:
                for row in tbl.rows:
                    for cell in row.cells:
                        if cell.text.strip(): parts.append(cell.text.strip())
            return "\n".join(parts)
        except Exception: return ""
    try: return data.decode(errors="ignore")
    except Exception: return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------- Heuristic fallback ----------
def detect_hr_subdomain_heuristic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["attrition","retention","predictive model","risk score"]): return "hr_attrition_model"
    if any(k in low for k in ["job description"," jd ","inclusive","bias","role profile"]): return "hr_jd_system"
    if any(k in low for k in ["ats","requisition","candidate","application","workflow"]): return "hr_ats"
    return "hr_attrition_model"

def extract_kpis_heuristic(text: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"KPI Name":"Generic KPI","Description":"Fallback KPI example.","Target Value":"","Status":"Pending"}
    ])

def recommend_heuristic(existing: list[str], raw_text: str) -> list[dict]:
    pool = [
        {"KPI Name":"Candidate Satisfaction","Description":"Measure candidate feedback.","Owner/ SME":"","Target Value":"","Status":"Pending"},
        {"KPI Name":"Recruiter Productivity","Description":"Measure recruiter output.","Owner/ SME":"","Target Value":"","Status":"Pending"},
    ]
    return [p for p in pool if p["KPI Name"] not in existing]

# ---------- LLM helpers ----------
def _json_from_llm(system: str, user: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?","",text).strip()
        text = re.sub(r"```$","",text).strip()
        return json.loads(text)
    except Exception:
        return None

CLASSIFY_SYS = "Classify BRD into hr_attrition_model, hr_jd_system, or hr_ats. Return ONLY JSON: {\"subdomain\": \"...\"}."
def classify_subdomain_llm(text: str) -> str:
    out = _json_from_llm(CLASSIFY_SYS, f"BRD:\n{text[:12000]}")
    if out and out.get("subdomain") in {"hr_attrition_model","hr_jd_system","hr_ats"}:
        return out["subdomain"]
    return detect_hr_subdomain_heuristic(text)

EXTRACT_SYS = (
    "Extract 3-6 KPIs from the BRD. Return JSON {\"kpis\":[{\"KPI Name\":...,\"Description\":...,\"Target Value\":...}]}"
)
def extract_kpis_llm(text: str) -> pd.DataFrame:
    out = _json_from_llm(EXTRACT_SYS, f"BRD:\n{text[:16000]}")
    rows = []
    if out and isinstance(out.get("kpis"), list):
        for it in out["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name: continue
            rows.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                         "Target Value":str(it.get("Target Value","")).strip(),"Status":"Pending"})
    if not rows: return extract_kpis_heuristic(text)
    return pd.DataFrame(rows).drop_duplicates(subset=["KPI Name"])

RECOMMEND_SYS = "Suggest 3-6 additional KPIs. Return JSON {\"kpis\":[{\"KPI Name\":...,\"Description\":...}]}"
def recommend_llm(existing: list[str], subdomain: str, text: str) -> list[dict]:
    existing_str = ", ".join(existing) or "(none)"
    out = _json_from_llm(RECOMMEND_SYS, f"Subdomain:{subdomain}\nExisting:{existing_str}\n\nBRD:\n{text[:16000]}")
    recs = []
    if out and isinstance(out.get("kpis"), list):
        for it in out["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name or name in existing: continue
            recs.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                         "Owner/ SME":"","Target Value":"","Status":"Pending"})
    if not recs: return recommend_heuristic(existing, text)
    return recs

# ---------- Login ----------
def login_page():
    st.markdown("### <div style='text-align:center;color:#b91c1c;'>AI KPI System</div>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if _check_credentials(email,password):
            st.session_state["auth"]=True; st.session_state["user"]=email.strip().lower(); st.rerun()
        else:
            st.error("Invalid email or password")

# ---------- MAIN ----------
if not st.session_state["auth"]:
    login_page(); st.stop()

st.title("AI KPI Extraction (LLM-powered)")

uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)
if st.button("Process BRDs"):
    if not uploads: st.warning("Please upload at least one file")
    else:
        for f in uploads:
            text = read_uploaded(f)
            if USE_OPENAI:
                sub = classify_subdomain_llm(text)
                extracted = extract_kpis_llm(text)
                recs = recommend_llm(extracted["KPI Name"].tolist(), sub, text)
            else:
                sub = detect_hr_subdomain_heuristic(text)
                extracted = extract_kpis_heuristic(text)
                recs = recommend_heuristic(extracted["KPI Name"].tolist(), text)
            recommended = pd.DataFrame(recs)
            st.session_state.projects[f.name] = {"extracted":extracted,"recommended":recommended,"domain":sub}
            st.session_state["final_kpis"].setdefault(
                f.name,pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
            )
        st.success(f"Processed {len(uploads)} BRDs")

for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname} — Subdomain: {proj['domain']}")
    st.subheader("Extracted KPIs")
    st.dataframe(proj["extracted"], use_container_width=True)
    st.subheader("Recommended KPIs")
    st.dataframe(proj["recommended"], use_container_width=True)

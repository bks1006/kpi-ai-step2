import io, re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

# -------- OCR fallback (auto-ignored if binaries missing) --------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -------- Page --------
st.set_page_config(page_title="KPI Recommender", layout="wide")
st.markdown("<h1 style='margin-top:0'>KPI Recommender</h1>", unsafe_allow_html=True)

# -------- Theme (red + white) --------
st.markdown("""
<style>
:root { --brand:#b91c1c; }
.stTextInput>div>div>input, .stSelectbox>div>div>select {
  border:1.5px solid var(--brand); border-radius:6px; padding:6px 8px; background:#fff;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
  border:2px solid var(--brand) !important; outline:none !important; box-shadow:0 0 5px var(--brand);
}
.kpi-header { background:#b91c1c; color:#fff; padding:10px 12px; border-radius:8px 8px 0 0; font-weight:700; }
.badge { display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; color:#fff; }
.badge-extracted { background:#6366f1; }      /* indigo */
.badge-pending { background:#9ca3af; }        /* gray */
.badge-validated { background:#16a34a; }      /* green */
.badge-rejected { background:#b91c1c; }       /* red */
.btn { border:0; border-radius:8px; padding:6px 12px; font-weight:600; cursor:pointer; }
.btn-ghost { background:#f3f4f6; color:#111827; }
.btn-validate { background:#2563eb; color:#fff; }   /* blue */
.btn-reject { background:#ef4444; color:#fff; }
.table-head { background:#f8fafc; border:1px solid #f1f5f9; border-bottom:0; border-radius:8px 8px 0 0; padding:8px 12px; font-weight:700; }
.table-body { border:1px solid #f1f5f9; border-top:0; border-radius:0 0 8px 8px; }
.cell { padding:10px 12px; border-top:1px solid #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# -------- Session --------
if "projects" not in st.session_state:
    st.session_state["projects"] = {}   # file -> {extracted, recommended, topic}
if "validated" not in st.session_state:
    st.session_state["validated"] = pd.DataFrame(
        columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"]
    )

# -------- Helpers --------
def status_badge(status: str) -> str:
    cls = "badge-pending"
    if status == "Extracted": cls = "badge-extracted"
    elif status == "Validated": cls = "badge-validated"
    elif status == "Rejected": cls = "badge-rejected"
    return f"<span class='badge {cls}'>{status}</span>"

def render_head(cols, headers):
    st.markdown(
        "<div class='table-head' style='display:grid;grid-template-columns:" +
        " ".join(cols) + ";'>" +
        "".join([f"<div>{h}</div>" for h in headers]) +
        "</div>", unsafe_allow_html=True
    )
    st.markdown("<div class='table-body'>", unsafe_allow_html=True)

def render_tail():
    st.markdown("</div>", unsafe_allow_html=True)

def upsert_validated(row: dict):
    """Insert/update in final validated table on (BRD, KPI Name)."""
    df = st.session_state["validated"]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.drop_duplicates(subset=["BRD","KPI Name"], keep="last", inplace=True)
    st.session_state["validated"] = df

# -------- File reading --------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if txt and txt.strip(): return txt
        except Exception:
            pass
        if OCR_AVAILABLE:
            try:
                imgs = convert_from_bytes(data)
                parts = []
                for im in imgs:
                    t = pytesseract.image_to_string(im, lang="eng")
                    if t and t.strip(): parts.append(t)
                return "\n".join(parts)
            except Exception:
                return ""
        return ""

    if lname.endswith(".docx"):
        try:
            doc = DocxDocument(bio)
        except Exception:
            return ""
        parts = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t: parts.append(t)
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    t = (cell.text or "").strip()
                    if t: parts.append(t)
        return "\n".join(parts)

    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# -------- Lightweight extractor (JD/ATS + general HR) --------
TARGET_PERCENT = r"\b(?:<|>|≤|≥)?\s*\d{1,3}(?:\.\d+)?\s*%\b"
TARGET_TIME    = r"\b(?:in|within|by)\s+\d+\s*(?:days?|weeks?|months?|quarters?|years?)\b"
TARGET_SECS    = r"\b(?:under|within|less than|<|≤)?\s*\d+(?:\.\d+)?\s*(?:ms|milliseconds|s|sec|secs|seconds)\b"
TARGET_USERS   = r"\b\d{1,5}(?:\.\d+)?\s*(?:concurrent\s+users|users)\b"
def find_targets(s: str) -> str:
    hits = []
    for pat in (TARGET_PERCENT, TARGET_TIME, TARGET_SECS, TARGET_USERS):
        for m in re.finditer(pat, s, flags=re.I):
            hits.append(m.group(0).strip())
    return " | ".join(dict.fromkeys(hits))

KPI_CANON = {
    "JD Generation Time": [r"\bgenerate\b.{0,40}\bjd\b.*(sec|seconds|ms|milliseconds|time|latency)"],
    "JD Parsing Accuracy": [r"\bjd parsing\b", r"\bjob description parsing\b"],
    "Resume Matching Accuracy": [r"\bresume matching\b", r"\bcv matching\b"],
    "Skills Extraction Precision": [r"\bskills extraction\b"],
    "Average Screening Time": [r"\bscreening time\b"],
    "Response Latency": [r"\blatency\b", r"\bresponse time\b"],
    "Concurrent Users Supported": [r"\bconcurrent users\b"],
    "System Uptime": [r"\buptime\b", r"\bavailability\b"],
    "Bias Flag Rate": [r"\bbias (flag|detection)\b", r"\bnon[- ]inclusive\b"],
    "JD Approval Rate": [r"\bapproval rate\b", r"\bapproved without major edits\b"],
    "Repository Compliance": [r"\bversion control\b", r"\bapproval logs\b", r"\brepository\b"],
    "Employee Turnover Rate": [r"\bturnover rate\b"],
    "Employee Retention Rate (1 YR)": [r"\bretention rate\b"],
    "Employee Satisfaction Score": [r"\bsatisfaction score\b", r"\benps\b"],
    "Time to Fill": [r"\btime to fill\b"],
}

def canonical_from_text(s: str) -> str | None:
    low = s.lower()
    for canon, pats in KPI_CANON.items():
        for pat in pats:
            if re.search(pat, low):
                return canon
    return None

def extract_kpis(text: str) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        name = canonical_from_text(ln)
        if name:
            rows.append({
                "KPI Name": name,
                "Description": ln if len(ln) < 220 else ln[:217] + "…",
                "Target Value": find_targets(ln),
                "Status": "Extracted"
            })
    df = pd.DataFrame(rows, columns=["KPI Name","Description","Target Value","Status"]).drop_duplicates()
    return df if not df.empty else pd.DataFrame(columns=["KPI Name","Description","Target Value","Status"])

# -------- Recommendations --------
TOPIC_KPIS = {
    "jd_ats": [
        "JD Parsing Accuracy","Skills Extraction Precision","Resume Matching Accuracy",
        "JD-Resume Match Score","Average Screening Time","Response Latency",
        "Automation Rate","SLA Compliance","Recruiter Productivity"
    ],
    "attrition_retention": [
        "Employee Turnover Rate","Employee Retention Rate (1 YR)","Absenteeism Rate","First Year Attrition Rate"
    ],
}
def infer_topic(text: str) -> str:
    low = text.lower()
    score_jd = sum(low.count(k) for k in ["job description","jd","resume","parsing","matching","skills"])
    score_att = sum(low.count(k) for k in ["attrition","retention","turnover","absenteeism"])
    return "jd_ats" if score_jd >= score_att else "attrition_retention"

def recommend_kpis(text: str, existing_names: list[str]) -> pd.DataFrame:
    topic = infer_topic(text)
    pool = TOPIC_KPIS.get(topic, [])
    existing = {e.lower() for e in existing_names}
    rows = []
    for k in pool:
        if k.lower() in existing: continue
        rows.append({"KPI Name": k, "Owner/ SME": "", "Target Value": "", "Status": "Pending"})
    return pd.DataFrame(rows, columns=["KPI Name","Owner/ SME","Target Value","Status"])

# -------- UI: Recommended table (buttons) --------
def render_recommended_table(brd: str, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        st.caption("No recommendations.")
        return df
    render_head(["2fr","1fr","1fr","0.7fr","1.4fr"], ["KPI Name","Owner/ SME","Target Value","Status","Actions"])
    updated = []
    for i, r in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2,1,1,0.7,1.4])
        with c1: st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
        with c2: owner_val = st.text_input("", value=r["Owner/ SME"], key=f"{key_prefix}_owner_{i}")
        with c3: target_val = st.text_input("", value=r["Target Value"], key=f"{key_prefix}_target_{i}")
        with c4: st.markdown(f"<div class='cell'>{status_badge(r['Status'])}</div>", unsafe_allow_html=True)
        with c5:
            colA, colB, colC = st.columns([1,1,1])
            with colA: st.markdown("<div class='cell'><button disabled class='btn btn-ghost'>Review Details</button></div>", unsafe_allow_html=True)
            with colB:
                if st.button("Validate", key=f"{key_prefix}_ok_{i}", type="secondary"):
                    r["Status"] = "Validated"
                    upsert_validated({
                        "BRD": brd, "KPI Name": r["KPI Name"], "Source":"Recommended",
                        "Description":"", "Owner/ SME": owner_val, "Target Value": target_val
                    })
            with colC:
                if st.button("Reject", key=f"{key_prefix}_rej_{i}", type="secondary"):
                    r["Status"] = "Rejected"
        updated.append({"KPI Name": r["KPI Name"], "Owner/ SME": owner_val, "Target Value": target_val, "Status": r["Status"]})
    render_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

# -------- UI: Extracted table (editable + buttons) --------
def render_extracted_table(brd: str, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        st.caption("No extracted KPIs.")
        return df
    render_head(["2fr","3fr","1fr","0.8fr","1.4fr"], ["KPI Name","Description","Target Value","Status","Actions"])
    updated = []
    for i, r in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2,3,1,0.8,1.4])
        with c1: st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)
        with c3: target_val = st.text_input("", value=r["Target Value"], key=f"{key_prefix}_target_{i}")
        with c4: st.markdown(f"<div class='cell'>{status_badge(r['Status'])}</div>", unsafe_allow_html=True)
        with c5:
            colA, colB, colC = st.columns([1,1,1])
            with colA: st.markdown("<div class='cell'><button disabled class='btn btn-ghost'>Review Details</button></div>", unsafe_allow_html=True)
            with colB:
                if st.button("Validate", key=f"{key_prefix}_ok_{i}", type="secondary"):
                    r["Status"] = "Validated"
                    upsert_validated({
                        "BRD": brd, "KPI Name": r["KPI Name"], "Source":"Extracted",
                        "Description": r["Description"], "Owner/ SME":"", "Target Value": target_val
                    })
            with colC:
                if st.button("Reject", key=f"{key_prefix}_rej_{i}", type="secondary"):
                    r["Status"] = "Rejected"
        updated.append({"KPI Name": r["KPI Name"], "Description": r["Description"], "Target Value": target_val, "Status": r["Status"]})
    render_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

# -------- Pipeline for one file --------
def process_file(file):
    text = read_uploaded(file)
    extracted = extract_kpis(text)
    existing = extracted["KPI Name"].tolist() if not extracted.empty else []
    recommended = recommend_kpis(text, existing)
    st.session_state["projects"][file.name] = {
        "extracted": extracted,
        "recommended": recommended,
        "topic": infer_topic(text)
    }

# -------- Upload + Process --------
left, right = st.columns([2,1])
with left:
    uploads = st.file_uploader("Upload Business Requirements Document (BRD)", type=["pdf","docx","txt"], accept_multiple_files=True)
with right:
    st.info("Supported formats: PDF, DOCX, TXT\n\nDrag & drop or browse.")
process = st.button("Process Uploaded File", use_container_width=True)
if process:
    if not uploads:
        st.warning("Please upload at least one file.")
    else:
        for f in uploads: process_file(f)
        st.success("Processed uploaded BRDs.")

# -------- Per-BRD sections --------
for fname, proj in st.session_state["projects"].items():
    st.markdown(f"### 📄 {fname}")
    st.markdown("<div class='kpi-header'>Preview Extracted Goals & KPIs</div>", unsafe_allow_html=True)
    proj["extracted"] = render_extracted_table(fname, proj["extracted"], key_prefix=f"ext_{fname}")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-header'>Extracted & Recommended KPIs</div>", unsafe_allow_html=True)
    st.caption("Review and manage extracted and recommended KPIs suggested by the AI to ensure accuracy and relevance.")
    proj["recommended"] = render_recommended_table(fname, proj["recommended"], key_prefix=f"rec_{fname}")

# -------- Final Validated KPIs --------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("<div class='kpi-header'>Final Validated KPIs</div>", unsafe_allow_html=True)

final_df = st.session_state["validated"]
if final_df.empty:
    st.caption("No validated KPIs yet. Validate items above to build this list.")
else:
    show = final_df[["BRD","KPI Name","Source","Owner/ SME","Target Value","Description"]].sort_values(["BRD","Source","KPI Name"])
    st.dataframe(show, use_container_width=True, hide_index=True)

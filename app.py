import os, re, io
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

st.set_page_config(page_title="AI KPI System", layout="wide")
st.title("AI KPI Extraction & Recommendations (Per BRD)")

# ---------------- Init ----------------
if "projects" not in st.session_state:
    st.session_state["projects"] = {}   # {file_name: {domain, extracted_df, recommended_df}}

# ---------------- Styling ----------------
STATUS_COLORS = {
    "Extracted": "#b91c1c",   # red
    "Recommended": "#9ca3af", # gray
    "Validated": "#16a34a",   # green
    "Rejected": "#7f1d1d",    # dark red
}

def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{text}</span>'

def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

# ---------------- File Reading ----------------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)
    if lname.endswith(".pdf"):
        reader = PdfReader(bio)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if lname.endswith(".docx"):
        doc = DocxDocument(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------------- Domain Detection ----------------
DOMAIN_HINTS = {
    "hr": ["employee","attrition","turnover","recruitment","hiring","retention"],
    "sales": ["pipeline","deal","quota","win rate"],
    "marketing": ["campaign","cpl","cac","ctr","impressions"],
    "finance": ["revenue","margin","cash","roi"],
}

def infer_domain(text: str) -> str:
    low = text.lower()
    scores = {d: sum(low.count(tok) for tok in toks) for d, toks in DOMAIN_HINTS.items()}
    d = max(scores, key=scores.get)
    return d if scores[d] > 0 else "hr"

# ---------------- KPI Extraction ----------------
TARGET_PATTERNS = [r"\b\d+\.?\d*\s*%(\b|$)", r"\b\d+\s*(?:days?|weeks?|months?)\b"]

KPI_SEEDS = [
    "Employee Turnover Rate","Employee Satisfaction Score","Employee Retention Rate",
    "Time to Fill","Net Promoter Score","Revenue Growth","Gross Margin",
    "Deployment Frequency","Change Failure Rate","On-time Delivery"
]

def kpiish(s: str) -> bool:
    return re.search(r"(kpi|metric|rate|score|ratio|time|margin|revenue|conversion|retention)", s, re.I) is not None

def find_target(s: str) -> str:
    for pat in TARGET_PATTERNS:
        m = re.search(pat, s, re.I)
        if m: return m.group(0)
    return ""

def extract_kpis(text: str) -> pd.DataFrame:
    lines = [re.sub(r"\s+"," ", x).strip(" -•") for x in text.splitlines() if x.strip()]
    cands = [ln for ln in lines if kpiish(ln)]
    rows = []
    for ln in cands:
        target = find_target(ln)
        name_guess = re.split(r"[:\-–]| \(", ln, maxsplit=1)[0].strip()
        best = process.extractOne(name_guess, KPI_SEEDS, scorer=fuzz.WRatio)
        name = best[0] if best and best[1] >= 85 else name_guess.title()[:80]
        desc = ln if len(ln) <= 200 else ln[:197] + "..."
        rows.append({"KPI Name": name, "Description": desc, "Target Value": target, "Status": "Extracted"})
    return pd.DataFrame(rows)

# ---------------- Recommendations (Fallback Only) ----------------
FALLBACK_KPIS = {
    "hr": ["Offer Acceptance Rate","Absenteeism Rate","Training Completion Rate","Employee NPS"],
    "sales": ["Win Rate","Lead Conversion","Quota Attainment","Average Deal Size"],
    "marketing": ["CTR","CAC","CPL","Brand Awareness Index"],
    "finance": ["Operating Margin","EBITDA Margin","Cash Burn","Runway Months"],
}

def recommend(domain: str, existing: list) -> list:
    fallback = FALLBACK_KPIS.get(domain, FALLBACK_KPIS["hr"])
    return [k for k in fallback if k.lower() not in {e.lower() for e in existing}]

# ---------------- UI Helpers ----------------
def render_editable_table(df: pd.DataFrame, editable_cols: list, key_prefix: str):
    if df.empty:
        st.caption("No data available.")
        return df

    # Red header
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:{'1fr ' * (len(df.columns))};
        background:#b91c1c;color:white;padding:8px 12px;font-weight:600;border-radius:6px 6px 0 0;">
        {''.join(f"<div>{col}</div>" for col in df.columns)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    updated_rows = []
    for i, row in df.iterrows():
        cols = st.columns([1 for _ in df.columns])
        row_data = {}
        for j, col in enumerate(df.columns):
            val = row[col]
            if col in editable_cols:
                row_data[col] = cols[j].text_input("", value=val, key=f"{key_prefix}_{i}_{col}")
            elif col == "Status":
                # Action buttons
                if cols[j].button("Validate", key=f"{key_prefix}_val_{i}"):
                    row_data[col] = "Validated"
                elif cols[j].button("Reject", key=f"{key_prefix}_rej_{i}"):
                    row_data[col] = "Rejected"
                else:
                    row_data[col] = val
                cols[j].markdown(status_chip(row_data[col]), unsafe_allow_html=True)
            else:
                cols[j].write(val if val else "—")
                row_data[col] = val
        updated_rows.append(row_data)

    return pd.DataFrame(updated_rows)

def process_file(file):
    text = read_uploaded(file)
    domain = infer_domain(text)
    extracted = extract_kpis(text)
    recs = recommend(domain, extracted["KPI Name"].tolist())
    recommended = pd.DataFrame([{"KPI Name": r, "Owner/ SME": "", "Target Value": "", "Status": "Recommended"} for r in recs])
    st.session_state.projects[file.name] = {"domain": domain, "extracted": extracted, "recommended": recommended}

# ---------------- Main UI ----------------
uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)
if st.button("Process BRDs"):
    if not uploads:
        st.warning("Please upload at least one file")
    for file in uploads:
        process_file(file)
    st.success("Processed uploaded BRDs")

# Show projects
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname} — Domain: **{proj['domain'].upper()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_editable_table(proj["extracted"], editable_cols=["Target Value"], key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_editable_table(proj["recommended"], editable_cols=["Owner/ SME","Target Value"], key_prefix=f"rec_{fname}")

import os, re, io
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

st.set_page_config(page_title="AI KPI System", layout="wide")
st.title("AI KPI Extraction & Recommendations")

# ---------------- Custom CSS (red + white) ----------------
st.markdown(
    """
    <style>
    /* Inputs */
    .stTextInput>div>div>input {
        border: 1.5px solid #b91c1c;
        border-radius: 6px;
        padding: 6px 8px;
        background: #fff;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #b91c1c !important;
        outline: none !important;
        box-shadow: 0 0 5px #b91c1c;
    }
    /* Dropdowns */
    .stSelectbox>div>div>select {
        border: 1.5px solid #b91c1c;
        border-radius: 6px;
        padding: 6px 8px;
        background: #fff;
    }
    .stSelectbox>div>div>select:focus {
        border: 2px solid #b91c1c !important;
        outline: none !important;
        box-shadow: 0 0 5px #b91c1c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session init ----------------
if "projects" not in st.session_state:
    st.session_state["projects"] = {}   # {file_name: {domain, extracted_df, recommended_df}}

# ---------------- Status chip colors ----------------
STATUS_COLORS = {
    "Validated": "#16a34a",  # green
    "Rejected":  "#7f1d1d",  # dark red
}

def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{text}</span>'

def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

# ---------------- File reading ----------------
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

# ---------------- Domain inference (simple) ----------------
DOMAIN_HINTS = {
    "hr": ["employee","attrition","turnover","recruitment","hiring","retention","satisfaction","absenteeism"],
    "sales": ["pipeline","deal","quota","win rate","opportunity","lead"],
    "marketing": ["campaign","cpl","cac","ctr","impressions","engagement"],
    "finance": ["revenue","margin","cash","roi","ebitda"],
}

def infer_domain(text: str) -> str:
    low = text.lower()
    scores = {d: sum(low.count(tok) for tok in toks) for d, toks in DOMAIN_HINTS.items()}
    d = max(scores, key=scores.get)
    return d if scores[d] > 0 else "hr"

# ---------------- KPI extraction (stricter) ----------------
TARGET_PATTERNS = [
    r"\b\d+\.?\d*\s*%(\b|$)",
    r"\b\d+\s*(?:days?|weeks?|months?)\b",
    r"\b\d+\.?\d*\s*/\s*10\b",
]

KPI_SEEDS = [
    "Employee Turnover Rate","Employee Satisfaction Score","Employee Retention Rate",
    "Time to Fill","Net Promoter Score","Revenue Growth","Gross Margin",
    "Deployment Frequency","Change Failure Rate","On-time Delivery","Customer Churn Rate",
]

# Lines that should never be treated as KPIs
EXCLUDE_PHRASES = [
    "business requirements document","brd","document","project","model","introduction","purpose","scope",
    "assumptions","out of scope","table of contents","revision history","version","author","date","appendix"
]

# Real KPI-like patterns
STRICT_KPI_PATTERNS = [
    r"\b(rate|ratio|score|index)\b",
    r"\b(time to|average .*time|resolution time|lead time|cycle time)\b",
    r"\b(net promoter score|nps|mttr|sla|on-?time delivery)\b",
    r"\b(retention|attrition|absenteeism)\s+rate\b",
    r"\b(conversion|win)\s+rate\b",
    r"\b(gross margin|operating margin)\b",
    r"\bcustomer churn rate\b",
]

def find_target(s: str) -> str:
    for pat in TARGET_PATTERNS:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(0)
    return ""

def kpiish(s: str) -> bool:
    low = s.lower()

    # exclude obvious headings/titles
    if any(ph in low for ph in EXCLUDE_PHRASES):
        return False

    # if a numeric target exists, ensure it's tied to a metric-y word
    if find_target(s):
        if re.search(r"\b(rate|ratio|score|index|time|margin|nps|mttr|sla|churn|retention|attrition|absenteeism|conversion|win)\b", low):
            return True

    # otherwise require strict KPI phrasing
    for pat in STRICT_KPI_PATTERNS:
        if re.search(pat, low):
            return True

    return False

def extract_kpis(text: str) -> pd.DataFrame:
    lines = [re.sub(r"\s+"," ", x).strip(" -•:") for x in text.splitlines() if x.strip()]
    cands = [ln for ln in lines if kpiish(ln)]
    rows = []
    for ln in cands:
        target = find_target(ln)
        name_guess = re.split(r"[:\-–]| \(", ln, maxsplit=1)[0].strip()
        best = process.extractOne(name_guess, KPI_SEEDS, scorer=fuzz.WRatio)
        name = best[0] if best and best[1] >= 85 else name_guess.title()[:80]
        desc = ln if len(ln) <= 200 else ln[:197] + "..."
        rows.append({
            "KPI Name": name,
            "Description": desc,
            "Target Value": target,
            "Status": "Validated"   # default decision; you can change to Rejected
        })
    return pd.DataFrame(rows)

# ---------------- Recommendations (fallback only) ----------------
FALLBACK_KPIS = {
    "hr": ["Offer Acceptance Rate","Absenteeism Rate","Training Completion Rate","Employee NPS","Internal Mobility Rate"],
    "sales": ["Win Rate","Lead Conversion","Quota Attainment","Average Deal Size","Sales Cycle Length"],
    "marketing": ["CTR","CAC","CPL","Brand Awareness Index","Email Open Rate"],
    "finance": ["Operating Margin","EBITDA Margin","Cash Burn","Runway Months","DSO"],
}

def recommend(domain: str, existing: list) -> list:
    fallback = FALLBACK_KPIS.get(domain, FALLBACK_KPIS["hr"])
    existing_l = {e.lower() for e in existing}
    return [k for k in fallback if k.lower() not in existing_l]

# ---------------- UI helpers ----------------
def render_editable_table(df: pd.DataFrame, editable_cols: list, key_prefix: str):
    if df.empty:
        st.caption("No data available.")
        return df

    # Red header bar
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
                # Only two review outcomes
                options = ["Validated", "Rejected"]
                default = val if val in options else "Validated"
                row_data[col] = cols[j].selectbox("", options, index=options.index(default), key=f"{key_prefix}_{i}_{col}_status")
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
    recommended = pd.DataFrame(
        [{"KPI Name": r, "Owner/ SME": "", "Target Value": "", "Status": "Validated"} for r in recs]
    )
    st.session_state.projects[file.name] = {
        "domain": domain,
        "extracted": extracted,
        "recommended": recommended
    }

# ---------------- Main UI ----------------
uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)

col_left, col_right = st.columns([1,1])
with col_left:
    if st.button("Process BRDs"):
        if not uploads:
            st.warning("Please upload at least one file")
        else:
            for file in uploads:
                process_file(file)
            st.success(f"✅ Processed {len(uploads)} BRD{'s' if len(uploads) > 1 else ''} successfully")

with col_right:
    if st.button("Clear results"):
        st.session_state["projects"] = {}
        st.info("Cleared.")

# Render each BRD section
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname} — Domain: **{proj['domain'].upper()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_editable_table(
        proj["extracted"],
        editable_cols=["Target Value"],
        key_prefix=f"ext_{fname}"
    )

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_editable_table(
        proj["recommended"],
        editable_cols=["Owner/ SME","Target Value"],
        key_prefix=f"rec_{fname}"
    )

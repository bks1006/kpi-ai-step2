import re, io
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

# ---------------- Page setup ----------------
st.set_page_config(page_title="AI KPI System", layout="wide")
st.title("AI KPI Extraction & Recommendations")

# ---------------- Theme CSS (red + white) ----------------
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

# ---------------- Chips ----------------
STATUS_COLORS = {
    "Validated": "#16a34a",   # green
    "Rejected":  "#7f1d1d",   # dark red
    "Pending":   "#9ca3af"    # gray
}
def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{text}</span>'
def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

# ---------------- File reading (PDF/DOCX incl. tables) ----------------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        reader = PdfReader(bio)
        return "\n".join((p.extract_text() or "") for p in reader.pages)

    if lname.endswith(".docx"):
        doc = DocxDocument(bio)
        blocks = []

        # paragraphs
        for p in doc.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                blocks.append(txt)

        # tables (cells often contain KPIs)
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    txt = (cell.text or "").strip()
                    if txt:
                        blocks.append(txt)

        return "\n".join(blocks)

    # txt / fallback
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------------- Domain inference (very light) ----------------
DOMAIN_HINTS = {
    "hr":        ["employee","attrition","turnover","recruitment","hiring","retention","satisfaction","absenteeism","time to fill"],
    "sales":     ["pipeline","deal","quota","win rate","opportunity","lead"],
    "marketing": ["campaign","cpl","cac","ctr","impressions","engagement"],
    "finance":   ["revenue","margin","cash","roi","ebitda"],
}
def infer_domain(text: str) -> str:
    low = text.lower()
    scores = {d: sum(low.count(tok) for tok in toks) for d, toks in DOMAIN_HINTS.items()}
    d = max(scores, key=scores.get)
    return d if scores[d] > 0 else "hr"

# ---------------- Preprocess text (join wraps, split bullets) ----------------
SPLIT_TOKENS = r"[•\u2022\-\–\—\·]|(?:^\s*\d+[\.\)])"  # bullets/dashes/1), 2., etc.
def preprocess_text(raw: str) -> str:
    # Join soft-wrapped lines
    lines = [l.strip() for l in raw.splitlines()]
    fused, buf = [], ""
    for l in lines:
        if not l:
            if buf:
                fused.append(buf.strip())
                buf = ""
            continue
        if buf:
            if not re.search(r"[\.!\?;:]$", buf) and not re.match(r"^\s*-\s*", l):
                buf = f"{buf} {l}"
            else:
                fused.append(buf.strip()); buf = l
        else:
            buf = l
    if buf: fused.append(buf.strip())

    # Explode bullets / numbered items
    parts = []
    for chunk in fused:
        for s in re.split(SPLIT_TOKENS, chunk):
            s = re.sub(r"\s+", " ", s).strip(" -•:\t")
            if s:
                parts.append(s)
    return "\n".join(parts)

# ---------------- KPI extraction (hybrid detector) ----------------
# Targets
TARGET_PERCENT = r"\b(?:<|>|≤|≥)?\s*\d{1,3}(?:\.\d+)?\s*%\b"
TARGET_RATIO   = r"\b\d+(?:\.\d+)?\s*/\s*\d+\b"
TARGET_TIME    = r"\b(?:in|within|by)\s+\d+\s*(?:days?|weeks?|months?|quarters?|years?)\b"
TARGET_GENERIC = r"(?:<|>|≤|≥)\s*\d+(?:\.\d+)?"
def find_targets(s: str) -> str:
    hits = []
    for pat in (TARGET_PERCENT, TARGET_RATIO, TARGET_TIME, TARGET_GENERIC):
        for m in re.finditer(pat, s, flags=re.I):
            hits.append(m.group(0).strip())
    # de-dup while preserving order
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
    return " | ".join(out)

# Seeds (for fuzzy name normalization)
KPI_SEEDS = [
    "Employee Turnover Rate","Employee Satisfaction Score","Employee Retention Rate",
    "Time to Fill","Offer Acceptance Rate","Absenteeism Rate","Employee NPS",
    "Internal Mobility Rate","Training Completion Rate","Quality of Hire","Cost per Hire",
    "Diversity Ratio","First Year Attrition Rate","Average Tenure","Engagement Score",
    "Vacancy Rate","Headcount Growth","Revenue Growth","Gross Margin","Customer Churn Rate"
]

# Exclusions and strict patterns
EXCLUDE_PHRASES = [
    "business requirements document","brd","document","project","model","introduction","purpose","scope",
    "assumptions","out of scope","table of contents","revision history","version","author","date","appendix"
]
STRICT_KPI_PATTERNS = [
    r"\b(rate|ratio|score|index)\b",
    r"\b(time to|average .*time|resolution time|lead time|cycle time)\b",
    r"\b(net promoter score|nps|mttr|sla|on-?time delivery)\b",
    r"\b(retention|attrition|absenteeism)\s+rate\b",
    r"\b(conversion|win)\s+rate\b",
    r"\b(gross margin|operating margin)\b",
    r"\bcustomer churn rate\b",
    r"\btime to fill\b",
    r"\boffer acceptance rate\b",
    r"\bemployee satisfaction\b",
    r"\bemployee nps\b",
    r"\babsenteeism rate\b",
]

GOAL_VERBS = r"(reduce|increase|decrease|improve|raise|lower|achieve|maintain)"

def kpiish(s: str) -> bool:
    low = s.lower()
    if any(ph in low for ph in EXCLUDE_PHRASES):
        return False

    # goal-like sentences referencing metrics
    if re.search(GOAL_VERBS, low) and re.search(r"(attrition|retention|turnover|satisfaction|nps|time to fill|offer acceptance|absenteeism|churn|margin|conversion|win rate|lead time|resolution)", low):
        return True

    # target values tied to metric words
    if find_targets(s) and re.search(r"\b(rate|ratio|score|index|time|margin|nps|mttr|sla|churn|retention|attrition|absenteeism|conversion|win|fill|acceptance)\b", low):
        return True

    # strict KPI phrasing
    for pat in STRICT_KPI_PATTERNS:
        if re.search(pat, low):
            return True
    return False

# HR KPI lexicon sweep (fallback detector)
HR_KPI_LEXICON = [
    "Employee Turnover Rate",
    "Voluntary Attrition Rate",
    "Involuntary Attrition Rate",
    "Employee Retention Rate",
    "Time to Fill",
    "Time to Hire",
    "Offer Acceptance Rate",
    "Absenteeism Rate",
    "Employee Satisfaction Score",
    "Employee NPS",
    "Net Promoter Score",
    "Engagement Score",
    "Training Completion Rate",
    "Quality of Hire",
    "Cost per Hire",
    "Internal Mobility Rate",
    "Diversity Ratio",
    "First Year Attrition Rate",
    "Average Tenure",
    "Vacancy Rate",
    "Headcount Growth"
]

def lexicon_hits(text: str):
    """Return rows from lexicon matches with nearby context and targets."""
    low = text.lower()
    rows = []
    for kpi in HR_KPI_LEXICON:
        pattern = re.escape(kpi.lower())
        m = re.search(pattern, low)
        if not m:
            continue
        # grab context window around the match
        start = max(0, m.start() - 180)
        end   = min(len(text), m.end() + 180)
        context = re.sub(r"\s+", " ", text[start:end]).strip()
        targets = find_targets(context)
        rows.append({
            "KPI Name": kpi,
            "Description": context[:240] + ("…" if len(context) > 240 else ""),
            "Target Value": targets,
            "Status": "Pending"
        })
    return rows

def extract_kpis(text: str) -> pd.DataFrame:
    """Hybrid extractor: sentence/bullet detector + lexicon sweep."""
    cols = ["KPI Name", "Description", "Target Value", "Status"]

    pre = preprocess_text(text)
    candidates = [c.strip() for c in pre.split("\n") if c.strip()]

    rows = []
    # 1) sentence/bullet detection
    for ln in candidates:
        if not kpiish(ln):
            continue
        targets = find_targets(ln)  # e.g., "10% | within 12 months"
        # normalize name with fuzzy match
        name_guess = re.split(r"[:\-–]| \(", ln, maxsplit=1)[0].strip()
        best = process.extractOne(name_guess, KPI_SEEDS, scorer=fuzz.WRatio)
        name = best[0] if best and best[1] >= 85 else name_guess[:120].strip().rstrip(" .")
        desc = ln if len(ln) <= 240 else ln[:237] + "…"
        rows.append({
            "KPI Name": name.title(),
            "Description": desc,
            "Target Value": targets,
            "Status": "Pending"
        })

    # 2) lexicon sweep over full raw text (to catch missed KPIs)
    rows += lexicon_hits(text)

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)

    # de-dup similar names (favor rows that have a Target Value)
    def _norm(k): return re.sub(r"[^a-z0-9]+", " ", str(k).lower()).strip()
    best_by_name = {}
    for _, r in df.iterrows():
        key = _norm(r["KPI Name"])
        current = best_by_name.get(key)
        # prefer row with targets; otherwise first seen
        if current is None or (not current["Target Value"] and r["Target Value"]):
            best_by_name[key] = r
    df = pd.DataFrame(best_by_name.values(), columns=cols).reset_index(drop=True)
    return df

# ---------------- Recommendations (fallback only; filtered vs extracted) ----------------
FALLBACK_KPIS = {
    "hr": ["Offer Acceptance Rate","Absenteeism Rate","Training Completion Rate","Employee NPS","Internal Mobility Rate",
           "Quality of Hire","Cost per Hire","Diversity Ratio","Vacancy Rate","First Year Attrition Rate"],
    "sales": ["Win Rate","Lead Conversion","Quota Attainment","Average Deal Size","Sales Cycle Length"],
    "marketing": ["CTR","CAC","CPL","Brand Awareness Index","Email Open Rate"],
    "finance": ["Operating Margin","EBITDA Margin","Cash Burn","Runway Months","DSO"],
}
def recommend(domain: str, existing: list) -> list:
    fallback = FALLBACK_KPIS.get(domain, FALLBACK_KPIS["hr"])
    existing_l = {e.lower() for e in existing}
    return [k for k in fallback if k.lower() not in existing_l]

# ---------------- Table UI (editable) ----------------
def render_editable_table(df: pd.DataFrame, editable_cols: list, key_prefix: str):
    if df.empty:
        st.caption("No data available.")
        return df

    # red header row
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
                row_data[col] = cols[j].text_input("", value=str(val) if pd.notna(val) else "", key=f"{key_prefix}_{i}_{col}")
            elif col == "Status":
                # Only decision: Validated or Rejected (with explicit choice)
                options = ["Choose…","Validated","Rejected"]
                default = "Choose…" if val not in ("Validated","Rejected") else val
                choice = cols[j].selectbox("", options, index=options.index(default), key=f"{key_prefix}_{i}_{col}_status")
                row_data[col] = "Pending" if choice == "Choose…" else choice
                cols[j].markdown(status_chip(row_data[col]), unsafe_allow_html=True)
            else:
                cols[j].write(val if (isinstance(val, str) and val.strip()) else "—")
                row_data[col] = val
        updated_rows.append(row_data)

    return pd.DataFrame(updated_rows, columns=list(df.columns))

# ---------------- Pipeline per file ----------------
def process_file(file):
    text = read_uploaded(file)
    domain = infer_domain(text)

    extracted = extract_kpis(text)

    # Build existing list safely
    if not extracted.empty and "KPI Name" in extracted.columns:
        existing = extracted["KPI Name"].astype(str).tolist()
    else:
        existing = []

    recs = recommend(domain, existing)
    recommended = pd.DataFrame(
        [{"KPI Name": r, "Owner/ SME": "", "Target Value": "", "Status": "Pending"} for r in recs],
        columns=["KPI Name", "Owner/ SME", "Target Value", "Status"]
    )

    st.session_state.projects[file.name] = {
        "domain": domain,
        "extracted": extracted,          # safe even if empty
        "recommended": recommended       # correct columns even if empty
    }

# ---------------- Main UI ----------------
uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)

if st.button("Process BRDs"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for file in uploads:
            process_file(file)
        st.success(f"✅ Processed {len(uploads)} BRD{'s' if len(uploads) > 1 else ''} successfully")

# Render each BRD section
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname} — Domain: **{proj['domain'].upper()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_editable_table(
        proj["extracted"], editable_cols=["Target Value"], key_prefix=f"ext_{fname}"
    )

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_editable_table(
        proj["recommended"], editable_cols=["Owner/ SME","Target Value"], key_prefix=f"rec_{fname}"
    )

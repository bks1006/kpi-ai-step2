import io, re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

# OCR / image
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------- Page setup ----------------
st.set_page_config(page_title="AI KPI System", layout="wide")
st.title("AI KPI Extraction & Recommendations (Per BRD)")

# ---------------- Theme CSS (red + white) ----------------
st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        border: 1.5px solid #b91c1c; border-radius: 6px; padding: 6px 8px; background:#fff;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #b91c1c !important; outline:none !important; box-shadow:0 0 5px #b91c1c;
    }
    .stSelectbox>div>div>select {
        border: 1.5px solid #b91c1c; border-radius: 6px; padding: 6px 8px; background:#fff;
    }
    .stSelectbox>div>div>select:focus {
        border: 2px solid #b91c1c !important; outline:none !important; box-shadow:0 0 5px #b91c1c;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session init ----------------
if "projects" not in st.session_state:
    st.session_state["projects"] = {}   # {file_name: {domain, topic, extracted_df, recommended_df}}

# ---------------- Chips ----------------
STATUS_COLORS = {"Validated": "#16a34a", "Rejected": "#7f1d1d", "Pending": "#9ca3af"}
def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{text}</span>'
def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

# ---------------- File reading (PDF/DOCX incl. tables + OCR fallback) ----------------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        # Try native PDF extraction first
        try:
            reader = PdfReader(bio)
            native_text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if native_text and native_text.strip():
                return native_text
        except Exception:
            pass

        # OCR fallback if available
        if OCR_AVAILABLE:
            try:
                pages = convert_from_bytes(data)  # needs poppler-utils
                ocr_text = []
                for img in pages:
                    txt = pytesseract.image_to_string(img, lang="eng")  # needs tesseract-ocr
                    if txt and txt.strip():
                        ocr_text.append(txt)
                return "\n".join(ocr_text)
            except Exception:
                return ""
        return ""

    if lname.endswith(".docx"):
        try:
            doc = DocxDocument(bio)
        except Exception:
            return ""
        blocks = []
        for p in doc.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                blocks.append(txt)
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

# ---------------- Domain inference ----------------
DOMAIN_HINTS = {
    "hr": [
        "employee","attrition","turnover","recruitment","hiring","retention",
        "satisfaction","absenteeism","time to fill","job description","resume","cv","parsing","ats","matching"
    ],
    "sales":     ["pipeline","deal","quota","win rate","opportunity","lead"],
    "marketing": ["campaign","cpl","cac","ctr","impressions","engagement"],
    "finance":   ["revenue","margin","cash","roi","ebitda"],
}
def infer_domain(text: str) -> str:
    low = text.lower()
    scores = {d: sum(low.count(tok) for tok in toks) for d, toks in DOMAIN_HINTS.items()}
    d = max(scores, key=scores.get)
    return d if scores[d] > 0 else "hr"

# ---------------- HR sub-topic detection & catalogs ----------------
HR_TOPIC_KEYWORDS = {
    "attrition_retention": ["attrition","turnover","retention","churn","stay interview","exit interview"],
    "jd_ats": ["job description","jd","resume","cv","parsing","ats","matching","skills extraction","screening"],
    "workforce_planning": ["headcount","workforce planning","manpower","capacity","utilization","vacancy","forecast"],
    "recruiting": ["recruiting","sourcing","hiring","offer","candidate","interview","requisition"],
    "learning_development": ["training","learning","l&d","course","skill","upskilling","certification","completion"],
    "engagement_culture": ["engagement","enps","nps","pulse","survey","satisfaction","morale","culture","recognition"],
}
TOPIC_KPIS = {
    "attrition_retention": [
        "Voluntary Attrition Rate","Involuntary Attrition Rate","Employee Retention Rate",
        "First Year Attrition Rate","Average Tenure","Internal Mobility Rate","Stay Interview Coverage",
        "Exit Interview Completion Rate","Regretted Attrition Rate"
    ],
    "jd_ats": [
        "JD Parsing Accuracy","Skills Extraction Precision","Resume Matching Accuracy",
        "JD-Resume Match Score","Automation Rate","Throughput Per Hour",
        "Average Screening Time","Recruiter Productivity","False Positive Match Rate","SLA Compliance"
    ],
    "workforce_planning": [
        "Headcount Growth","Vacancy Rate","Time to Backfill","Capacity Utilization",
        "Forecast Accuracy","Bench Strength Index","Span of Control"
    ],
    "recruiting": [
        "Time to Fill","Time to Hire","Offer Acceptance Rate","Quality of Hire",
        "Cost per Hire","Candidate Drop-off Rate","Interview to Offer Ratio"
    ],
    "learning_development": [
        "Training Completion Rate","Training Effectiveness Score","Certification Pass Rate",
        "Learning Hours per Employee","Time to Competency","Skills Coverage Index"
    ],
    "engagement_culture": [
        "Engagement Score","Employee NPS","Participation Rate (Surveys)",
        "Recognition Frequency","Manager Feedback Response Time","eNPS Promoter Ratio"
    ],
}
FALLBACK_KPIS = {
    "hr": [
        "Offer Acceptance Rate","Absenteeism Rate","Training Completion Rate","Employee NPS",
        "Internal Mobility Rate","Quality of Hire","Cost per Hire","Diversity Ratio",
        "Vacancy Rate","First Year Attrition Rate"
    ],
    "sales": ["Win Rate","Lead Conversion","Quota Attainment","Average Deal Size","Sales Cycle Length"],
    "marketing": ["CTR","CAC","CPL","Brand Awareness Index","Email Open Rate"],
    "finance": ["Operating Margin","EBITDA Margin","Cash Burn","Runway Months","DSO"],
}
def infer_hr_topic(text: str) -> str:
    low = text.lower()
    best_topic, best_score = None, 0
    for topic, kws in HR_TOPIC_KEYWORDS.items():
        score = sum(low.count(k) for k in kws)
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic or "recruiting"

# ---------------- Preprocess text (join wraps, split bullets) ----------------
SPLIT_TOKENS = r"[•\u2022\-\–\—\·]|(?:^\s*\d+[\.\)])"
def preprocess_text(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines()]
    fused, buf = [], ""
    for l in lines:
        if not l:
            if buf: fused.append(buf.strip()); buf = ""
            continue
        if buf and not re.search(r"[\.!\?;:]$", buf) and not re.match(r"^\s*-\s*", l):
            buf = f"{buf} {l}"
        else:
            if buf: fused.append(buf.strip())
            buf = l
    if buf: fused.append(buf.strip())

    parts = []
    for chunk in fused:
        for s in re.split(SPLIT_TOKENS, chunk):
            s = re.sub(r"\s+", " ", s).strip(" -•:\t")
            if s: parts.append(s)
    return "\n".join(parts)

# ---------------- KPI extraction (hybrid detector) ----------------
TARGET_PERCENT = r"\b(?:<|>|≤|≥)?\s*\d{1,3}(?:\.\d+)?\s*%\b"
TARGET_RATIO   = r"\b\d+(?:\.\d+)?\s*/\s*\d+\b"
TARGET_TIME    = r"\b(?:in|within|by)\s+\d+\s*(?:days?|weeks?|months?|quarters?|years?)\b"
TARGET_GENERIC = r"(?:<|>|≤|≥)\s*\d+(?:\.\d+)?"
def find_targets(s: str) -> str:
    hits = []
    for pat in (TARGET_PERCENT, TARGET_RATIO, TARGET_TIME, TARGET_GENERIC):
        for m in re.finditer(pat, s, flags=re.I):
            hits.append(m.group(0).strip())
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
    return " | ".join(out)

KPI_SEEDS = [
    "Employee Turnover Rate","Employee Satisfaction Score","Employee Retention Rate",
    "Time to Fill","Offer Acceptance Rate","Absenteeism Rate","Employee NPS",
    "Internal Mobility Rate","Training Completion Rate","Quality of Hire","Cost per Hire",
    "Diversity Ratio","First Year Attrition Rate","Average Tenure","Engagement Score",
    "Vacancy Rate","Headcount Growth","Customer Churn Rate",
    # JD/ATS style
    "JD Parsing Accuracy","Skills Extraction Precision","Resume Matching Accuracy",
    "JD-Resume Match Score","Throughput Per Hour","Automation Rate","Recruiter Productivity",
    "Average Screening Time","Response Latency","SLA Compliance"
]

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
    r"\btime to fill\b", r"\boffer acceptance rate\b",
    r"\bemployee satisfaction\b", r"\bemployee nps\b", r"\babsenteeism rate\b",
    # JD system phrasing
    r"\bjd parsing accuracy\b", r"\bresume matching accuracy\b",
    r"\bskills extraction (precision|recall|f1)\b", r"\bmatch score\b",
    r"\bthroughput\b", r"\bautomation rate\b", r"\brecruiter productivity\b",
    r"\bsla compliance\b", r"\blatency\b",
]
GOAL_VERBS = r"(reduce|increase|decrease|improve|raise|lower|achieve|maintain)"

def kpiish(s: str) -> bool:
    low = s.lower()
    if any(ph in low for ph in EXCLUDE_PHRASES):
        return False
    if re.search(GOAL_VERBS, low) and re.search(
        r"(attrition|retention|turnover|satisfaction|nps|time to fill|offer acceptance|absenteeism|churn|margin|conversion|win rate|lead time|resolution|parsing|matching|skills|throughput|automation|latency|sla)",
        low):
        return True
    if find_targets(s) and re.search(
        r"\b(rate|ratio|score|index|time|margin|nps|mttr|sla|churn|retention|attrition|absenteeism|conversion|win|fill|acceptance|parsing|matching|precision|recall|throughput|latency)\b",
        low):
        return True
    for pat in STRICT_KPI_PATTERNS:
        if re.search(pat, low):
            return True
    return False

HR_KPI_LEXICON = [
    # Core HR
    "Employee Turnover Rate","Voluntary Attrition Rate","Involuntary Attrition Rate",
    "Employee Retention Rate","Time to Fill","Time to Hire","Offer Acceptance Rate",
    "Absenteeism Rate","Employee Satisfaction Score","Employee NPS","Engagement Score",
    "Training Completion Rate","Quality of Hire","Cost per Hire","Internal Mobility Rate",
    "Diversity Ratio","First Year Attrition Rate","Average Tenure","Vacancy Rate","Headcount Growth",
    # JD/ATS
    "JD Parsing Accuracy","Skills Extraction Precision","Resume Matching Accuracy",
    "JD-Resume Match Score","Throughput Per Hour","Automation Rate","Recruiter Productivity",
    "Average Screening Time","Response Latency","SLA Compliance","False Positive Match Rate"
]
def lexicon_hits(text: str):
    low = text.lower()
    rows = []
    for kpi in HR_KPI_LEXICON:
        m = re.search(re.escape(kpi.lower()), low)
        if not m: 
            continue
        start = max(0, m.start() - 180); end = min(len(text), m.end() + 180)
        ctx = re.sub(r"\s+", " ", text[start:end]).strip()
        targets = find_targets(ctx)
        rows.append({
            "KPI Name": kpi, "Description": ctx[:240] + ("…" if len(ctx) > 240 else ""),
            "Target Value": targets, "Status": "Pending"
        })
    return rows

def extract_kpis(text: str) -> pd.DataFrame:
    cols = ["KPI Name", "Description", "Target Value", "Status"]
    pre = preprocess_text(text)
    candidates = [c.strip() for c in pre.split("\n") if c.strip()]

    rows = []
    # (1) sentence/bullet detection
    for ln in candidates:
        if not kpiish(ln):
            continue
        targets = find_targets(ln)
        name_guess = re.split(r"[:\-–]| \(", ln, maxsplit=1)[0].strip()
        best = process.extractOne(name_guess, KPI_SEEDS, scorer=fuzz.WRatio)
        name = best[0] if best and best[1] >= 85 else name_guess[:120].strip().rstrip(" .")
        desc = ln if len(ln) <= 240 else ln[:237] + "…"
        rows.append({"KPI Name": name.title(), "Description": desc, "Target Value": targets, "Status": "Pending"})

    # (2) lexicon sweep
    rows += lexicon_hits(text)

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=cols)

    # de-dup similar names; prefer entries with targets
    def _norm(k): return re.sub(r"[^a-z0-9]+", " ", str(k).lower()).strip()
    best_by = {}
    for _, r in df.iterrows():
        key = _norm(r["KPI Name"])
        cur = best_by.get(key)
        if cur is None or (not cur["Target Value"] and r["Target Value"]):
            best_by[key] = r
    df = pd.DataFrame(best_by.values(), columns=cols).reset_index(drop=True)
    return df

# ---------------- Topic-aware Recommendations ----------------
def recommend(domain: str, existing: list, topic: str = None, raw_text: str = "") -> list:
    existing_l = {e.lower() for e in existing}
    if domain == "hr":
        topic = topic or infer_hr_topic(raw_text)
        pool = TOPIC_KPIS.get(topic, FALLBACK_KPIS["hr"])
    else:
        pool = FALLBACK_KPIS.get(domain, [])
    recs = [k for k in pool if k.lower() not in existing_l]
    seen, out = set(), []
    for k in recs:
        key = k.lower()
        if key not in seen:
            out.append(k); seen.add(key)
        if len(out) >= 10:
            break
    return out

# ---------------- Table UI (editable) ----------------
def render_editable_table(df: pd.DataFrame, editable_cols: list, key_prefix: str):
    if df.empty:
        st.caption("No data available.")
        return df

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
    topic = infer_hr_topic(text) if domain == "hr" else None

    extracted = extract_kpis(text)
    existing = extracted["KPI Name"].astype(str).tolist() if not extracted.empty and "KPI Name" in extracted.columns else []

    recs = recommend(domain, existing, topic=topic, raw_text=text)
    recommended = pd.DataFrame(
        [{"KPI Name": r, "Owner/ SME": "", "Target Value": "", "Status": "Pending"} for r in recs],
        columns=["KPI Name", "Owner/ SME", "Target Value", "Status"]
    )

    st.session_state.projects[file.name] = {
        "domain": domain, "topic": topic,
        "extracted": extracted, "recommended": recommended
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
    topic_text = f" — Topic: **{proj.get('topic','').replace('_',' ').title()}**" if proj.get("topic") else ""
    st.markdown(f"## 📄 {fname} — Domain: **{proj['domain'].upper()}**{topic_text}")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_editable_table(
        proj["extracted"], editable_cols=["Target Value"], key_prefix=f"ext_{fname}"
    )

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_editable_table(
        proj["recommended"], editable_cols=["Owner/ SME","Target Value"], key_prefix=f"rec_{fname}"
    )

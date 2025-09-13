import io, re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

# OCR / image fallback (optional but recommended)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------- Page ----------------
st.set_page_config(page_title="AI KPI System", layout="wide")
st.title("AI KPI Extraction & Recommendations (Per BRD)")

# ---------------- Theme (red + white) ----------------
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

# ---------------- Session ----------------
if "projects" not in st.session_state:
    st.session_state["projects"] = {}

# ---------------- Chips ----------------
STATUS_COLORS = {"Validated": "#16a34a", "Rejected": "#7f1d1d", "Pending": "#9ca3af"}
def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{text}</span>'
def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

# ---------------- Read files (PDF with OCR, DOCX incl. tables) ----------------
def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            native_text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if native_text and native_text.strip():
                return native_text
        except Exception:
            pass
        if OCR_AVAILABLE:
            try:
                pages = convert_from_bytes(data)
                ocr_text = []
                for img in pages:
                    txt = pytesseract.image_to_string(img, lang="eng")
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
            t = (p.text or "").strip()
            if t:
                blocks.append(t)
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    t = (cell.text or "").strip()
                    if t:
                        blocks.append(t)
        return "\n".join(blocks)

    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------------- Domain & Topic ----------------
DOMAIN_HINTS = {
    "hr": [
        "employee","attrition","turnover","recruitment","hiring","retention",
        "satisfaction","absenteeism","time to fill","job description","resume",
        "cv","parsing","ats","matching","screening"
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

HR_TOPIC_KEYWORDS = {
    "attrition_retention": ["attrition","turnover","retention","churn","stay interview","exit interview"],
    "jd_ats": ["job description","jd","resume","cv","parsing","ats","matching","skills extraction","screening"],
    "workforce_planning": ["headcount","workforce planning","manpower","capacity","utilization","vacancy","forecast"],
    "recruiting": ["recruiting","sourcing","hiring","offer","candidate","interview","requisition"],
    "learning_development": ["training","learning","l&d","course","skill","upskilling","certification","completion"],
    "engagement_culture": ["engagement","enps","nps","pulse","survey","satisfaction","morale","culture","recognition"],
}
def infer_hr_topic(text: str) -> str:
    low = text.lower()
    best, score = None, 0
    for topic, kws in HR_TOPIC_KEYWORDS.items():
        s = sum(low.count(k) for k in kws)
        if s > score:
            best, score = topic, s
    return best or "recruiting"

# ---------------- Preprocess (join wraps, split bullets) ----------------
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

# ---------------- KPI rules ----------------
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

# Canonical KPI names and aliases (used to normalize names)
KPI_CANON = {
    # Attrition / retention
    "Voluntary Attrition Rate": [r"\bvoluntary attrition\b", r"\bvoluntary turnover\b"],
    "Involuntary Attrition Rate": [r"\binvoluntary attrition\b", r"\binvoluntary turnover\b"],
    "Employee Retention Rate": [r"\bretention rate\b", r"\bemployee retention\b"],
    "First Year Attrition Rate": [r"\bfirst[- ]year attrition\b"],
    "Average Tenure": [r"\baverage tenure\b", r"\bavg tenure\b"],
    "Internal Mobility Rate": [r"\binternal mobility\b"],
    "Stay Interview Coverage": [r"\bstay interview\b"],
    "Exit Interview Completion Rate": [r"\bexit interview\b"],
    "Regretted Attrition Rate": [r"\bregretted attrition\b"],

    # Recruiting
    "Time to Fill": [r"\btime to fill\b"],
    "Time to Hire": [r"\btime to hire\b"],
    "Offer Acceptance Rate": [r"\boffer acceptance\b"],
    "Quality of Hire": [r"\bquality of hire\b"],
    "Cost per Hire": [r"\bcost per hire\b"],
    "Candidate Drop-off Rate": [r"\bdrop[- ]off\b"],
    "Interview to Offer Ratio": [r"\binterview to offer\b"],

    # Engagement / satisfaction
    "Employee Satisfaction Score": [r"\bemployee satisfaction\b", r"\bsatisfaction score\b"],
    "Employee NPS": [r"\bemployee nps\b", r"\benps\b", r"\bnet promoter score\b"],
    "Engagement Score": [r"\bengagement score\b"],

    # Attendance / diversity
    "Absenteeism Rate": [r"\babsenteeism\b"],
    "Diversity Ratio": [r"\bdiversity ratio\b"],

    # Workforce
    "Headcount Growth": [r"\bheadcount growth\b"],
    "Vacancy Rate": [r"\bvacancy rate\b"],
    "Time to Backfill": [r"\btime to backfill\b"],
    "Capacity Utilization": [r"\bcapacity utilization\b"],
    "Forecast Accuracy": [r"\bforecast accuracy\b"],

    # JD/ATS
    "JD Parsing Accuracy": [r"\bjd parsing\b", r"\bjob description parsing\b"],
    "Skills Extraction Precision": [r"\bskills extraction\b", r"\bskill extraction\b"],
    "Resume Matching Accuracy": [r"\bresume matching\b", r"\bcv matching\b"],
    "JD-Resume Match Score": [r"\bmatch score\b"],
    "Automation Rate": [r"\bautomation rate\b"],
    "Throughput Per Hour": [r"\bthroughput\b"],
    "Average Screening Time": [r"\bscreening time\b"],
    "Recruiter Productivity": [r"\brecruiter productivity\b"],
    "Response Latency": [r"\blatency\b", r"\bresponse time\b"],
    "SLA Compliance": [r"\bsla compliance\b"],
    "False Positive Match Rate": [r"\bfalse positive\b"],
}

# handy set of metric keywords to filter out generic sentences
METRIC_WORDS = re.compile(
    r"\b(rate|ratio|score|index|time|latency|throughput|tenure|utilization|accuracy|precision|recall|f1|cost)\b",
    re.I,
)

EXCLUDE_PHRASES = [
    "business requirements document","brd","document","project","model","introduction","purpose","scope",
    "assumptions","out of scope","table of contents","revision history","version","author","date","appendix"
]

def canonical_from_text(s: str) -> str | None:
    """Return canonical KPI name if any alias matches the text."""
    low = s.lower()
    for canon, pats in KPI_CANON.items():
        for pat in pats:
            if re.search(pat, low):
                return canon
    return None

# fuzzy seeds (fallback normalization)
KPI_SEEDS = list(KPI_CANON.keys()) + [
    "Employee Turnover Rate","Customer Churn Rate","Training Completion Rate","Internal Mobility Rate"
]

def is_probably_kpi(line: str) -> bool:
    low = line.lower()
    if any(ph in low for ph in EXCLUDE_PHRASES):
        return False
    if canonical_from_text(low):
        return True
    if METRIC_WORDS.search(low) and (find_targets(low) or re.search(r"\b(kpi|metric)\b", low)):
        return True
    return False

# ---------------- Extraction ----------------
def preprocess_for_candidates(text: str) -> list[str]:
    pre = preprocess_text(text)
    cands = [c.strip() for c in pre.split("\n") if c.strip()]
    # limit overlong generic paragraphs
    cands = [c for c in cands if len(c) <= 400]
    return cands

def normalize_name(raw_line: str) -> str:
    canon = canonical_from_text(raw_line)
    if canon:
        return canon
    guess = re.split(r"[:\-–]| \(", raw_line, maxsplit=1)[0].strip()
    # avoid keeping whole sentences as name
    if len(guess.split()) > 6 and not METRIC_WORDS.search(guess):
        guess = " ".join(guess.split()[:6])
    match = process.extractOne(guess, KPI_SEEDS, scorer=fuzz.WRatio)
    if match and match[1] >= 85:
        return match[0]
    # title-case a short guess otherwise
    return guess.title()

def dedup_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def _norm(k): return re.sub(r"[^a-z0-9]+", " ", str(k).lower()).strip()
    best = {}
    for _, r in df.iterrows():
        key = _norm(r["KPI Name"])
        cur = best.get(key)
        # prefer rows with a target value
        if cur is None or (not cur["Target Value"] and r["Target Value"]):
            best[key] = r
    return pd.DataFrame(best.values()).reset_index(drop=True)

def extract_kpis(text: str) -> pd.DataFrame:
    cols = ["KPI Name", "Description", "Target Value", "Status"]
    rows = []

    # 1) sentence/bullet based
    for ln in preprocess_for_candidates(text):
        if not is_probably_kpi(ln):
            continue
        name = normalize_name(ln)
        targets = find_targets(ln)
        desc = ln if len(ln) <= 240 else ln[:237] + "…"
        rows.append({"KPI Name": name, "Description": desc, "Target Value": targets, "Status": "Pending"})

    # 2) lexicon sweep in raw text (for missed items)
    low = text.lower()
    for canon in KPI_CANON.keys():
        if re.search(re.escape(canon.lower()), low) and not any(canon == r["KPI Name"] for r in rows):
            # capture local context to find targets
            m = re.search(re.escape(canon.lower()), low)
            start = max(0, m.start() - 180); end = min(len(text), m.end() + 180)
            ctx = re.sub(r"\s+", " ", text[start:end]).strip()
            rows.append({
                "KPI Name": canon,
                "Description": ctx[:240] + ("…" if len(ctx) > 240 else ""),
                "Target Value": find_targets(ctx),
                "Status": "Pending"
            })

    df = pd.DataFrame(rows, columns=cols)
    df = dedup_rows(df)
    # always return with columns
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols]

# ---------------- Recommendations (topic-aware) ----------------
TOPIC_KPIS = {
    "attrition_retention": [
        "Voluntary Attrition Rate","Involuntary Attrition Rate","Employee Retention Rate",
        "First Year Attrition Rate","Average Tenure","Internal Mobility Rate",
        "Stay Interview Coverage","Exit Interview Completion Rate","Regretted Attrition Rate"
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

def recommend(domain: str, existing: list, topic: str = None, raw_text: str = "") -> list:
    existing_l = {e.lower() for e in existing}
    if domain == "hr":
        topic = topic or infer_hr_topic(raw_text)
        pool = TOPIC_KPIS.get(topic, FALLBACK_KPIS["hr"])
    else:
        pool = FALLBACK_KPIS.get(domain, [])
    out, seen = [], set()
    for k in pool:
        if k.lower() in existing_l or k.lower() in seen:
            continue
        out.append(k); seen.add(k.lower())
        if len(out) >= 12:
            break
    return out

# ---------------- UI table ----------------
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
        """, unsafe_allow_html=True
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
                opts = ["Choose…","Validated","Rejected"]
                default = "Choose…" if val not in ("Validated","Rejected") else val
                choice = cols[j].selectbox("", opts, index=opts.index(default), key=f"{key_prefix}_{i}_{col}_status")
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
    existing = extracted["KPI Name"].astype(str).tolist() if not extracted.empty else []

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
        for f in uploads:
            process_file(f)
        st.success(f"✅ Processed {len(uploads)} BRD{'s' if len(uploads) > 1 else ''} successfully")

for fname, proj in st.session_state.projects.items():
    topic_text = f" — Topic: **{proj.get('topic','').replace('_',' ').title()}**" if proj.get("topic") else ""
    st.markdown(f"## 📄 {fname} — Domain: **{proj['domain'].upper()}**{topic_text}")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_editable_table(proj["extracted"], editable_cols=["Target Value"], key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_editable_table(
        proj["recommended"], editable_cols=["Owner/ SME","Target Value"], key_prefix=f"rec_{fname}"
    )

import os, re, time, io
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
import requests

st.set_page_config(page_title="AI KPI System — Step 2", layout="wide")
st.title("AI System — KPI Extraction & Recommendations")

# ---------------- Session init ----------------
def ss_init():
    st.session_state.setdefault("domain", None)
    st.session_state.setdefault("extracted_df", pd.DataFrame(columns=["KPI Name","Description","Target Value","Status"]))
    st.session_state.setdefault("recommended_df", pd.DataFrame(columns=["KPI Name","Owner/ SME","Target Value","Status"]))
    st.session_state.setdefault("autoloaded", False)  # run ./brds autoload once

ss_init()

# ---------------- Helpers ----------------
STATUS_COLORS = {
    "Extracted": "#4f46e5",
    "Recommended": "#0891b2",
    "Validated": "#059669",
    "Rejected": "#dc2626",
}

def chip(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:12px">{text}</span>'

def status_chip(s: str) -> str:
    return chip(s, STATUS_COLORS.get(s, "#6b7280"))

def read_text_from_bytes(data: bytes, name: str) -> str:
    lname = name.lower()
    bio = io.BytesIO(data)
    if lname.endswith(".pdf"):
        reader = PdfReader(bio)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if lname.endswith(".docx"):
        doc = DocxDocument(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    # txt/fallback
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_text_uploaded(file) -> str:
    data = file.read()
    return read_text_from_bytes(data, file.name)

# ---------------- Domain inference ----------------
DOMAIN_HINTS = {
    "hr": ["employee","attrition","turnover","recruitment","hiring","satisfaction","retention","headcount","absenteeism","time to fill","job description","ats"],
    "sales": ["pipeline","win rate","quota","deal","conversion","lead","mql","sql","revenue","arpu","churn"],
    "marketing": ["campaign","ctr","impressions","cac","cpl","cpm","engagement","brand","reach"],
    "finance": ["ebitda","gross margin","opex","cash flow","roi","npv","payback","runway"],
    "product": ["feature adoption","activation","maus","daus","onboarding","nps"],
    "engineering": ["mttr","deployment frequency","change failure rate","lead time","slo","incident","sla"],
    "support": ["csat","first response time","fcr","resolution","ticket","backlog"],
    "ops": ["on-time","cycle time","throughput","utilization","yield","defect rate"],
}

def infer_domain(text: str) -> str:
    low = text.lower()
    scores = {d: sum(low.count(tok) for tok in toks) for d, toks in DOMAIN_HINTS.items()}
    d = max(scores, key=scores.get)
    return d if scores[d] > 0 else "hr"

# ---------------- KPI extraction ----------------
TARGET_PATTERNS = [
    r"(?:>=|=>|≥)\s*\d+\.?\d*\s*%?",
    r"(?:<=|=<|≤)\s*\d+\.?\d*\s*%?",
    r"[<>]=?\s*\d+\.?\d*\s*%?",
    r"\b\d+\.?\d*\s*/\s*10\b",
    r"\b\d+\.?\d*\s*%(\b|$)",
    r"\b\d+\s*(?:days?|weeks?|months?)\b",
]

KPI_SEEDS = [
    "Employee Turnover Rate","Employee Satisfaction Score","Employee Retention Rate (1 YR)",
    "Involuntary Attrition","Absenteeism Rate","Time to Fill","Net Promoter Score",
    "Average Resolution Time","MTTR","Revenue Growth","Gross Margin","Operating Margin",
    "Deployment Frequency","Change Failure Rate","On-time Delivery","Lead Conversion Rate","Customer Churn Rate",
]

def clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" -•\t")

def kpiish(s: str) -> bool:
    if len(s) < 5: return False
    if re.search(r"\b(kpi|goal|objective|metric|target|okr|measure)\b", s, re.I): return True
    if re.search(r"\b(rate|score|ratio|retention|turnover|satisfaction|time|churn|margin|revenue|conversion)\b", s, re.I): return True
    return False

def find_target(s: str) -> str:
    for pat in TARGET_PATTERNS:
        m = re.search(pat, s, flags=re.I)
        if m: return m.group(0).strip()
    return ""

def extract_kpis(text: str) -> pd.DataFrame:
    lines = [clean_line(x) for x in text.splitlines() if clean_line(x)]
    cands = [ln for ln in lines if kpiish(ln)]
    rows = []
    for ln in cands:
        target = find_target(ln)
        name_guess = re.split(r"[:\-–]| \(", ln, maxsplit=1)[0].strip()
        best = process.extractOne(name_guess, KPI_SEEDS, scorer=fuzz.WRatio)
        name = best[0] if best and best[1] >= 85 else name_guess.title()[:80]
        desc = ln if len(ln) <= 220 else ln[:217] + "..."
        rows.append({"KPI Name": name, "Description": desc, "Target Value": target, "Status": "Extracted"})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.sort_values(by=["Target Value"], ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["KPI Name"], keep="first").reset_index(drop=True)
    return df

# ---------------- Recommendations ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CX = os.getenv("GOOGLE_CX", "").strip()

DOMAIN_QUERY_MAP = {
    "hr": ["most common HR KPIs", "people analytics KPIs", "talent acquisition KPIs"],
    "sales": ["top sales KPIs", "B2B sales metrics KPIs"],
    "marketing": ["top marketing KPIs", "digital marketing KPIs list"],
    "finance": ["finance KPIs list", "financial performance metrics"],
    "product": ["product management KPIs", "SaaS product KPIs"],
    "engineering": ["software engineering KPIs", "devops DORA metrics"],
    "support": ["customer support KPIs", "service desk KPIs"],
    "ops": ["operations KPIs list", "supply chain KPIs"],
}

FALLBACK_KPIS = {
    "hr": ["Employee Turnover Rate","Employee Retention Rate (1 YR)","Employee Satisfaction Score",
           "Time to Fill","Offer Acceptance Rate","Absenteeism Rate","Training Completion Rate",
           "Internal Mobility Rate","Diversity Ratio","Involuntary Attrition"],
    "sales": ["Win Rate","Lead to SQL Conversion","Average Deal Size","Sales Cycle Length",
              "Quota Attainment","Pipeline Coverage","Customer Churn Rate","ARPU"],
    "marketing": ["Website Conversion Rate","CTR","CPL","CAC","MQL to SQL Conversion",
                  "Organic Traffic Growth","Email Open Rate","Brand Awareness Index"],
    "finance": ["Revenue Growth","Gross Margin","Operating Margin","EBITDA Margin","Cash Burn",
                "Runway Months","DSO","ROIC"],
    "product": ["DAU","MAU","Activation Rate","Feature Adoption Rate","Retention (D30)","NPS","Time to Value","Churn Rate"],
    "engineering": ["Deployment Frequency","Lead Time for Changes","Change Failure Rate","MTTR",
                    "Defect Escape Rate","SLO Compliance","Incident Rate"],
    "support": ["First Response Time","Average Resolution Time","FCR","CSAT","Backlog Size","SLA Breach Rate"],
    "ops": ["On-time Delivery","Cycle Time","Throughput","Capacity Utilization","Yield","Defect Rate","Inventory Turnover"],
}

def google_search(query: str, n: int=6) -> List[Dict]:
    if not (GOOGLE_API_KEY and GOOGLE_CX):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    try:
        r = requests.get(url, params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query}, timeout=15)
        if r.status_code == 200:
            return r.json().get("items", [])[:n]
    except Exception:
        pass
    return []

def mine_kpi_phrases(text: str) -> List[str]:
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    toks = re.findall(r"[A-Z][A-Za-z ]{2,}(?:Rate|Score|Ratio|Time|Cost|Revenue|Margin|Churn|Retention|Conversion)", text)
    toks += [x.strip() for x in re.split(r"[•\-\u2022,\|/]", text)
             if re.search(r"\b(rate|score|ratio|time|margin|churn|retention|conversion|nps|mttr|lead|win|defect|yield|utilization)\b", x, re.I)]
    out = []
    for t in toks:
        t = re.sub(r"\s+"," ", t).strip(" -•")
        if 3 < len(t) <= 60 and t not in out:
            out.append(t)
    return out

def recommend(domain: str, existing: List[str]) -> List[str]:
    lower_exist = {e.lower() for e in existing}
    fetched = []
    for q in DOMAIN_QUERY_MAP.get(domain, DOMAIN_QUERY_MAP["hr"]):
        items = google_search(q, n=6)
        for it in items:
            fetched += mine_kpi_phrases((it.get("title","") + ". " + it.get("snippet","")))
        time.sleep(0.15)
    if not fetched:
        fetched = FALLBACK_KPIS.get(domain, FALLBACK_KPIS["hr"])
    seeds = KPI_SEEDS + FALLBACK_KPIS.get(domain, [])
    normalized = []
    for cand in fetched:
        best = process.extractOne(cand, seeds, scorer=fuzz.WRatio)
        normalized.append(best[0] if best and best[1] >= 85 else cand.title())
    out = []
    for k in normalized:
        if k.lower() not in lower_exist and k not in out:
            out.append(k)
    return out[:10]

# ---------------- UI: Auto-load ./brds once ----------------
def autoload_brds_once():
    if st.session_state.autoloaded:
        return
    folder = os.path.join(os.getcwd(), "brds")
    if not os.path.isdir(folder):
        st.session_state.autoloaded = True
        return
    texts = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".pdf",".docx",".txt")):
            continue
        path = os.path.join(folder, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
            txt = read_text_from_bytes(data, fname)
            if txt:
                texts.append(txt)
        except Exception:
            pass
    if texts:
        full = "\n".join(texts)
        st.session_state.domain = infer_domain(full)
        ext = extract_kpis(full)
        st.session_state.extracted_df = ext
        existing = ext["KPI Name"].tolist() if not ext.empty else []
        recs = recommend(st.session_state.domain or "hr", existing)
        st.session_state.recommended_df = pd.DataFrame(
            [{"KPI Name": k, "Owner/ SME": "", "Target Value": "", "Status": "Recommended"} for k in recs]
            + [{"KPI Name": r["KPI Name"], "Owner/ SME": "", "Target Value": r["Target Value"], "Status": "Extracted"}
               for _, r in ext.iterrows()]
        )
        st.info(f"Loaded {len(texts)} BRD(s) from ./brds. Domain: {(st.session_state.domain or 'hr').upper()}")
    st.session_state.autoloaded = True

autoload_brds_once()

# ---------------- UI: Upload + Process ----------------
st.subheader("Upload BRDs")
uploads = st.file_uploader("PDF / DOCX / TXT", type=["pdf","docx","txt"], accept_multiple_files=True)

if st.button("Process Uploaded BRDs"):
    if not uploads:
        st.warning("Please upload at least one BRD.")
    else:
        texts = [read_text_uploaded(f) for f in uploads]
        full = "\n".join(texts)
        st.session_state.domain = infer_domain(full)
        ext = extract_kpis(full)
        st.session_state.extracted_df = ext

        existing = ext["KPI Name"].tolist() if not ext.empty else []
        recs = recommend(st.session_state.domain, existing)
        st.session_state.recommended_df = pd.DataFrame(
            [{"KPI Name": k, "Owner/ SME": "", "Target Value": "", "Status": "Recommended"} for k in recs]
            + [{"KPI Name": r["KPI Name"], "Owner/ SME": "", "Target Value": r["Target Value"], "Status": "Extracted"}
               for _, r in ext.iterrows()]
        )
        st.success(f"Processed {len(uploads)} file(s). Domain guessed: {st.session_state.domain.upper()}")

st.markdown("---")

# ---------------- UI: Preview Extracted ----------------
st.subheader("Preview Extracted KPIs")
if st.session_state.extracted_df.empty:
    st.caption("No extracted KPIs yet.")
else:
    st.markdown(
        """<div style="display:grid;grid-template-columns:1.5fr 2.4fr 0.9fr 0.7fr 0.7fr;
        padding:8px 12px;background:#f5f6f7;border:1px solid #e5e7eb;font-weight:600">
        <div>KPI Name</div><div>Description</div><div>Target Value</div><div>Status</div><div>Actions</div></div>""",
        unsafe_allow_html=True,
    )
    df = st.session_state.extracted_df
    for i, row in df.iterrows():
        c1,c2,c3,c4,c5 = st.columns([1.5,2.4,0.9,0.7,0.7])
        c1.write(row["KPI Name"])
        c2.write(row["Description"])
        c3.write(row["Target Value"] or "—")
        c4.markdown(status_chip(row["Status"]), unsafe_allow_html=True)
        if c5.button("Review", key=f"ext_rev_{i}"):
            st.session_state["ext_idx"] = i
            st.session_state["ext_open"] = True

    if st.session_state.get("ext_open"):
        i = st.session_state.get("ext_idx", 0)
        with st.expander(f"Review: {df.loc[i,'KPI Name']}", expanded=True):
            n = st.text_input("KPI Name", value=df.loc[i,"KPI Name"])
            d = st.text_area("Description", value=df.loc[i,"Description"])
            t = st.text_input("Target Value", value=df.loc[i,"Target Value"])
            if st.button("Apply changes", key="apply_ext"):
                df.loc[i,"KPI Name"] = n.strip()
                df.loc[i,"Description"] = d.strip()
                df.loc[i,"Target Value"] = t.strip()
                st.toast("Updated")
st.markdown("---")

# ---------------- UI: Recommendations ----------------
st.subheader("Recommended & Extracted (for review)")
low_df = st.session_state.recommended_df
if low_df.empty:
    st.caption("No recommendations yet.")
else:
    st.markdown(
        """<div style="display:grid;grid-template-columns:1.6fr 1fr 0.9fr 0.7fr 1fr 0.7fr;
        padding:8px 12px;background:#f5f6f7;border:1px solid #e5e7eb;font-weight:600">
        <div>KPI Name</div><div>Owner/ SME</div><div>Target Value</div><div>Status</div><div>Actions</div><div></div></div>""",
        unsafe_allow_html=True,
    )
    for i, row in low_df.iterrows():
        c1,c2,c3,c4,c5,c6 = st.columns([1.6,1.0,0.9,0.7,1.0,0.7])
        c1.write(row["KPI Name"])
        owner = c2.text_input("Owner", value=row["Owner/ SME"], key=f"own_{i}")
        target = c3.text_input("Target", value=row["Target Value"], key=f"targ_{i}")
        c4.markdown(status_chip(row["Status"]), unsafe_allow_html=True)
        review = c5.button("Review Details", key=f"rec_rev_{i}")
        validate = c6.button("Validate", key=f"val_{i}")
        reject = c6.button("Reject", key=f"rej_{i}")

        # persist edits
        low_df.loc[i,"Owner/ SME"] = owner
        low_df.loc[i,"Target Value"] = target

        if validate:
            low_df.loc[i,"Status"] = "Validated"; st.toast("Validated")
        if reject:
            low_df.loc[i,"Status"] = "Rejected"; st.toast("Rejected")
        if review:
            st.session_state["rec_idx"] = i
            st.session_state["rec_open"] = True

    if st.session_state.get("rec_open"):
        i = st.session_state.get("rec_idx", 0)
        with st.expander(f"Review Details — {low_df.loc[i,'KPI Name']}", expanded=True):
            st.write(low_df.loc[i].to_frame().rename(columns={i:"Value"}))

    _, rcol = st.columns([5,1.3])
    with rcol:
        if st.button("Validate All"):
            low_df["Status"] = "Validated"
            st.toast("All KPIs validated.")

st.caption("Domain guess drives the recommendation query set. Provide GOOGLE_API_KEY/GOOGLE_CX for web-powered suggestions; otherwise, curated fallbacks are used.")

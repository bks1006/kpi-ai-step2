from __future__ import annotations
import io, os, re, json, hashlib
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
        OPENAI_MODEL = "gpt-4o-mini"  # or gpt-4.1 if you have it
    except Exception:
        USE_OPENAI = False

# ---------- Demo credentials ----------
VALID_USERS = {"admin@company.com": "password123", "user@company.com": "welcome123"}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

# ---------- Session ----------
if "auth" not in st.session_state: st.session_state["auth"] = False
if "user" not in st.session_state: st.session_state["user"] = None
if "projects" not in st.session_state: st.session_state["projects"] = {}
if "final_kpis" not in st.session_state: st.session_state["final_kpis"] = {}
if "llm_cache" not in st.session_state: st.session_state["llm_cache"] = {}  # simple memo to avoid repeated calls

# ---------- Utility ----------
def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

def _chip(status: str) -> str:
    cls = "chip-pending"
    if status == "Validated": cls = "chip-ok"
    elif status == "Rejected": cls = "chip-bad"
    return f"<span class='chip {cls}'>{status}</span>"

def _upsert_final(brd, row):
    df = st.session_state["final_kpis"].get(
        brd, pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
    )
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.drop_duplicates(subset=["KPI Name"], keep="last", inplace=True)
    st.session_state["final_kpis"][brd] = df

def _remove_from_final(brd, name):
    df = st.session_state["final_kpis"].get(
        brd, pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
    )
    if not df.empty:
        df = df[df["KPI Name"] != name].reset_index(drop=True)
    st.session_state["final_kpis"][brd] = df

# ---------- File reading ----------
def read_text_from_bytes(data: bytes, name: str) -> str:
    bio = io.BytesIO(data); lname = name.lower()
    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if txt.strip(): return txt
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
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for tbl in doc.tables:
                for row in tbl.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            parts.append(cell.text.strip())
            return "\n".join(parts)
        except Exception:
            return ""
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------- Domain detection (content + filename) ----------
def detect_hr_subdomain_heuristic(text: str, filename: str = "") -> str:
    low = (text + " " + filename).lower()
    if any(k in low for k in ["attrition","retention","churn","predictive model","risk score"]): return "hr_attrition_model"
    if any(k in low for k in ["job description"," jd ","inclusive","bias","role profile"]): return "hr_jd_system"
    if any(k in low for k in ["ats","applicant tracking","requisition","candidate","application workflow"]): return "hr_ats"
    return "hr_attrition_model"

# ---------- Detailed heuristic KPIs per domain (fallback) ----------
def extract_kpis_heuristic(text: str, filename: str) -> pd.DataFrame:
    sub = detect_hr_subdomain_heuristic(text, filename)
    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name":"Model Discrimination (AUC)",
             "Description":"ROC-AUC on a held-out test set never seen in training. Report AUC overall and stratified by job family, location, tenure to catch cohort drift. Use time-split validation to mimic real deployment.",
             "Target Value":"≥ 0.80","Status":"Pending"},
            {"KPI Name":"Voluntary Attrition Reduction",
             "Description":"Percent reduction in voluntary exits compared to a 12-month baseline, normalized for headcount/seasonality. Attribute impact to model-triggered actions (manager outreach, compensation review, internal mobility).",
             "Target Value":"≥ 10% within 12 months","Status":"Pending"},
            {"KPI Name":"Actionable Insight Coverage",
             "Description":"Share of employees flagged high-risk for whom the system surfaces top drivers and at least one actionable lever within the same cycle (e.g., coaching plan, comp check, role change).",
             "Target Value":"≥ 80%","Status":"Pending"},
        ]
    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name":"JD Generation Latency (p50/p90)",
             "Description":"Median and 90th percentile time from request to a publishable JD draft including bias audit and grading. Segment by job family and seniority to ensure consistent throughput.",
             "Target Value":"p50 < 10s; p90 < 20s","Status":"Pending"},
            {"KPI Name":"Inclusive Language Improvement",
             "Description":"Decrease in flagged gendered/ableist/age-coded terms per 1,000 words after rewrite. Detection combines curated lexicons with embedding similarity for near-matches and phrases.",
             "Target Value":"≥ 60% reduction vs baseline","Status":"Pending"},
            {"KPI Name":"Approval Cycle Time",
             "Description":"Elapsed time from draft to final HR approval across manager/HR review steps. Highlights workflow friction and rework hotspots. Track by department to focus enablement.",
             "Target Value":"≥ 30% faster vs baseline","Status":"Pending"},
        ]
    else:  # hr_ats
        rows = [
            {"KPI Name":"Application Step Drop-off",
             "Description":"Fraction of candidates who abandon at each step (profile, questions, assessments, scheduling), segmented by device and geography. Identifies friction and optimizes the funnel.",
             "Target Value":"Reduce p95 step drop-off by 20%","Status":"Pending"},
            {"KPI Name":"Time-to-Fill (p50/p90)",
             "Description":"Days from requisition open to accepted offer, excluding candidate notice period. Report median and p90 by role family/location to surface bottlenecks.",
             "Target Value":"≥ 25% reduction vs baseline","Status":"Pending"},
            {"KPI Name":"Automation Rate",
             "Description":"Share of recruiter tasks executed automatically (screening, scheduling, nudges, feedback reminders). Measured as automated events / total workflow events.",
             "Target Value":"≥ 40% automated within 6 months","Status":"Pending"},
        ]
    return pd.DataFrame(rows)

HR_KPI_LIB = {
    "hr_attrition_model": [
        ("High-Risk Follow-through",
         "Share of high-risk employees who receive at least one retention action within 14 days of being flagged (contact, comp review, mobility plan)."),
        ("Dashboard Adoption",
         "Monthly active HR users / licensed users on the attrition dashboard; indicates whether insights drive operational decisions."),
    ],
    "hr_jd_system": [
        ("Template Utilization",
         "Share of roles that start from approved JD templates. Drives consistency and reduces legal / compliance risk."),
        ("Hiring Manager Adoption",
         "Percent of requisitions where the hiring manager edited the AI draft rather than uploading a manual JD. Measures trust and usability."),
    ],
    "hr_ats": [
        ("Candidate Satisfaction (CSAT)",
         "Average post-application/interview rating on clarity, fairness, and communication. Break out by role and country."),
        ("Recruiter Productivity",
         "Requisitions or candidates handled per recruiter per month while meeting SLAs (time to response, feedback latency)."),
    ],
}

def recommend_heuristic(existing: list[str], text: str, filename: str) -> list[dict]:
    sub = detect_hr_subdomain_heuristic(text, filename)
    out = []
    for name, desc in HR_KPI_LIB[sub]:
        if name not in existing:
            out.append({"KPI Name":name,"Description":desc,"Owner/ SME":"","Target Value":"","Status":"Pending"})
    return out

# ---------- LLM helpers ----------
def _cache_key(prefix: str, payload: str) -> str:
    return prefix + ":" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _json_from_llm(system: str, user: str) -> dict | None:
    if not USE_OPENAI:
        return None
    cache_id = _cache_key("llm", system + "\n---\n" + user)
    if cache_id in st.session_state["llm_cache"]:
        return st.session_state["llm_cache"][cache_id]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        data = json.loads(text)
        st.session_state["llm_cache"][cache_id] = data
        return data
    except Exception:
        return None

CLASSIFY_SYS = (
    "Classify the HR BRD into exactly one of:\n"
    "- hr_attrition_model\n- hr_jd_system\n- hr_ats\n"
    'Return ONLY JSON: {"subdomain":"<one>"}'
)

EXTRACT_SYS = (
    "Extract 4–7 KPIs from the BRD. Each KPI must have:\n"
    "- KPI Name (short and specific)\n"
    "- Description (2–3 sentences, operational, how to measure, segmentation)\n"
    "- Target Value (string; empty if not stated)\n"
    'Return ONLY JSON: {"kpis":[{"KPI Name":str,"Description":str,"Target Value":str}]}\n'
    "Avoid duplicates, vanity metrics, and generic 'System Uptime'."
)

RECOMMEND_SYS = (
    "Suggest 3–6 additional KPIs tailored to the BRD subdomain. Avoid any already in the existing list.\n"
    "Each must have a 2–3 sentence actionable description (what, how to measure, segmentation).\n"
    'Return ONLY JSON: {"kpis":[{"KPI Name":str,"Description":str}]}.'
)

def classify_subdomain_llm(text: str, filename: str) -> str:
    out = _json_from_llm(CLASSIFY_SYS, f"Filename: {filename}\n\nBRD:\n{text[:12000]}")
    if out and out.get("subdomain") in {"hr_attrition_model","hr_jd_system","hr_ats"}:
        return out["subdomain"]
    return detect_hr_subdomain_heuristic(text, filename)

def extract_kpis_llm(text: str, filename: str) -> pd.DataFrame:
    out = _json_from_llm(EXTRACT_SYS, f"Filename: {filename}\n\nBRD:\n{text[:16000]}")
    rows = []
    if out and isinstance(out.get("kpis"), list):
        for it in out["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name: continue
            rows.append({
                "KPI Name": name,
                "Description": str(it.get("Description","")).strip(),
                "Target Value": str(it.get("Target Value","")).strip(),
                "Status": "Pending",
            })
    if not rows:
        return extract_kpis_heuristic(text, filename)
    df = pd.DataFrame(rows).drop_duplicates(subset=["KPI Name"])
    return df

def recommend_llm(existing: list[str], subdomain: str, text: str, filename: str) -> list[dict]:
    existing_str = ", ".join(sorted(existing)) or "(none)"
    out = _json_from_llm(RECOMMEND_SYS,
                         f"Subdomain: {subdomain}\nExisting: {existing_str}\nFilename: {filename}\n\nBRD:\n{text[:16000]}")
    recs = []
    if out and isinstance(out.get("kpis"), list):
        for it in out["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name or name in existing: continue
            recs.append({
                "KPI Name": name,
                "Description": str(it.get("Description","")).strip(),
                "Owner/ SME": "",
                "Target Value": "",
                "Status": "Pending",
            })
    if not recs:
        return recommend_heuristic(existing, text, filename)
    return recs[:6]

# ---------- Table UI (keep buttons on the same row) ----------
def render_table(brd, df, source, key_prefix):
    if df.empty:
        st.caption(f"No {source} KPIs.")
        return df

    st.markdown(
        "<style>"
        ".chip{display:inline-block;padding:4px 10px;border-radius:999px;color:#fff;font-size:12px}"
        ".chip-pending{background:#9ca3af}.chip-ok{background:#16a34a}.chip-bad{background:#b91c1c}"
        ".btn-wrap button{background:#f9fafb!important;color:#111827!important;border:1px solid #e5e7eb!important;"
        "border-radius:8px!important;padding:.4rem .8rem!important;font-weight:600!important}"
        ".btn-wrap.on-validate button{background:#16a34a!important;color:#fff!important}"
        ".btn-wrap.on-reject button{background:#b91c1c!important;color:#fff!important}"
        ".cell{padding:8px 10px;border-top:1px solid #e5e7eb}"
        ".rowcard{border:1px solid #e5e7eb;border-radius:8px;margin:6px 0}"
        ".thin-input>div>div{border:1.4px solid #e5e7eb!important;border-radius:8px!important;background:#fff!important}"
        "</style>",
        unsafe_allow_html=True
    )

    updated = []
    for i, r in df.iterrows():
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2.1, 3.3, 1.2, 0.9, 1.6], gap="small")
            with c1:
                st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)
            with c3:
                target_val = st.text_input(
                    label="",
                    value=r.get("Target Value",""),
                    key=f"{key_prefix}_t_{i}",
                    label_visibility="collapsed",
                    placeholder="Target value",
                )
            with c4:
                st.markdown(f"<div class='cell'>{_chip(r['Status'])}</div>", unsafe_allow_html=True)
            with c5:
                st.markdown("<div class='cell'>", unsafe_allow_html=True)
                col_v, col_r = st.columns(2, gap="small")
                with col_v:
                    st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
                    if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                        _upsert_final(
                            brd,
                            {"BRD": brd,
                             "KPI Name": r["KPI Name"],
                             "Source": source,
                             "Description": r["Description"],
                             "Owner/ SME": r.get("Owner/ SME",""),
                             "Target Value": target_val}
                        )
                        df.at[i,"Status"] = "Validated"
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_r:
                    st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
                    if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                        _remove_from_final(brd, r["KPI Name"])
                        df.at[i,"Status"] = "Rejected"
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        updated.append({
            "KPI Name": r["KPI Name"],
            "Description": r["Description"],
            "Target Value": target_val,
            "Status": df.at[i, "Status"],
            "Owner/ SME": r.get("Owner/ SME","")
        })
    return pd.DataFrame(updated)

# ---------- Login ----------
def login_page():
    st.markdown("""
    <style>
      :root { --brand:#b91c1c; }
      [data-testid="stAppViewContainer"] .main .block-container { padding-top: 12vh; }
      .login-btn > button { background:var(--brand)!important;color:#fff!important;border:none!important;border-radius:10px!important;
        font-weight:700!important;padding:.65rem 1.2rem!important;width:100%!important; }
      .stTextInput > div > div { border:1.6px solid var(--brand)!important;border-radius:10px!important;background:#fff!important; }
    </style>
    """, unsafe_allow_html=True)

    left, mid, right = st.columns([1,2,1])
    with mid:
        card = st.container(border=True)
        with card:
            st.markdown("### <div style='text-align:center;color:#b91c1c;'>AI KPI System</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;color:#6b7280;'>Sign in to continue</div>", unsafe_allow_html=True)
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                if st.button("Sign in", key="signin_btn"):
                    if _check_credentials(email, password):
                        st.session_state["auth"] = True
                        st.session_state["user"] = email.strip().lower()
                        st.rerun()
                    else:
                        st.error("Invalid email or password")

# ===================== MAIN =====================
if not st.session_state["auth"]:
    login_page()
    st.stop()

# Light app CSS
st.markdown("""
<style>
:root { --brand:#b91c1c; --green:#16a34a; --red:#b91c1c; }
.topbar { position:sticky; top:0; z-index:5; background:#fff; padding:6px 0 8px; border-bottom:1px solid #eee; margin-bottom:8px; }
.topbar-inner { display:flex; justify-content:space-between; align-items:center; }
.who { color:#6b7280; font-size:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='topbar'><div class='topbar-inner'>"
    f"<div class='who'>Signed in as <b>{st.session_state.get('user','')}</b></div>"
    "</div></div>", unsafe_allow_html=True
)

st.title("AI KPI Extraction & Recommendations (LLM-first)")

# Single uploader
uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True, key="brd_uploader")

def process_file(f):
    text = read_uploaded(f)
    fname = f.name
    if USE_OPENAI:
        sub = classify_subdomain_llm(text, fname)
        extracted = extract_kpis_llm(text, fname)
        recs = recommend_llm(extracted["KPI Name"].tolist(), sub, text, fname)
    else:
        sub = detect_hr_subdomain_heuristic(text, fname)
        extracted = extract_kpis_heuristic(text, fname)
        recs = recommend_heuristic(extracted["KPI Name"].tolist(), text, fname)

    recommended = pd.DataFrame(recs)
    st.session_state.projects[fname] = {"extracted": extracted, "recommended": recommended, "domain": sub}
    st.session_state["final_kpis"].setdefault(
        fname, pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
    )

if st.button("Process BRDs", key="process_btn"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for f in uploads:
            process_file(f)
        st.success(f"✅ Processed {len(uploads)} BRD(s) successfully")

# Render per BRD
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname}")
    st.caption(f"Detected subdomain: **{proj.get('domain','hr_attrition_model').replace('_',' ').title()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_table(fname, proj["extracted"], "Extracted", key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_table(fname, proj["recommended"], "Recommended", key_prefix=f"rec_{fname}")

    st.markdown("#### Finalized KPIs")
    final_df = st.session_state["final_kpis"].get(fname, pd.DataFrame())
    if final_df.empty:
        st.caption("No validated KPIs yet for this BRD.")
    else:
        show = final_df[["KPI Name","Source","Owner/ SME","Target Value","Description"]].sort_values("KPI Name")
        st.dataframe(show, use_container_width=True, hide_index=True)

        _, c2, _ = st.columns([1,2,1])
        with c2:
            if st.button("Review & Accept", key=f"accept_{fname}"):
                st.success("✅ Finalized KPIs have been accepted successfully!")

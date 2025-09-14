from __future__ import annotations
import io, os, re, json, hashlib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from io import BytesIO

# =========================
# Optional OCR (for image-only PDFs)
# =========================
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# =========================
# LLM setup (optional)
# =========================
USE_OPENAI = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if USE_OPENAI and not OPENAI_API_KEY:
    USE_OPENAI = False

if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_MODEL = "gpt-4o-mini"
    except Exception:
        USE_OPENAI = False

# =========================
# Demo credentials
# =========================
VALID_USERS = {"admin@company.com": "password123", "user@company.com": "welcome123"}

# =========================
# Page config
# =========================
st.set_page_config(page_title="AI KPI System", layout="wide")

# =========================
# Session state
# =========================
if "auth" not in st.session_state: st.session_state["auth"] = False
if "user" not in st.session_state: st.session_state["user"] = None
if "projects" not in st.session_state: st.session_state["projects"] = {}
if "final_kpis" not in st.session_state: st.session_state["final_kpis"] = {}
if "llm_cache" not in st.session_state: st.session_state["llm_cache"] = {}

# =========================
# Helpers / constants
# =========================
FINAL_COLS = ["BRD","KPI Name","Source","Description","Owner/ SME","Target Value","Status"]

def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

def _chip(status: str) -> str:
    cls = "chip-pending"
    if status == "Validated": cls = "chip-ok"
    elif status == "Rejected": cls = "chip-bad"
    elif status == "Accepted": cls = "chip-accepted"
    return f"<span class='chip {cls}'>{status}</span>"

def _ensure_final_df(brd: str) -> pd.DataFrame:
    df = st.session_state["final_kpis"].get(brd)
    if df is None or df.empty:
        df = pd.DataFrame(columns=FINAL_COLS)
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[FINAL_COLS].copy()
    st.session_state["final_kpis"][brd] = df
    return df

def _upsert_final(brd: str, row: dict):
    df = _ensure_final_df(brd).copy()
    for c in FINAL_COLS:
        row.setdefault(c, "")
    mask = (df["BRD"] == row["BRD"]) & (df["KPI Name"] == row["KPI Name"])
    df = df.loc[~mask].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    st.session_state["final_kpis"][brd] = df

def _remove_from_final(brd: str, name: str):
    df = _ensure_final_df(brd).copy()
    df = df.loc[df["KPI Name"] != name].reset_index(drop=True)
    st.session_state["final_kpis"][brd] = df

# =========================
# File reading
# =========================
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

# =========================
# Domain detection
# =========================
def detect_hr_subdomain_heuristic(text: str, filename: str = "") -> str:
    low = (text + " " + filename).lower()
    if any(k in low for k in ["attrition","retention","churn","risk score","predictive model"]):
        return "hr_attrition_model"
    if any(k in low for k in ["job description"," jd ","inclusive","bias","role profile"]):
        return "hr_jd_system"
    if any(k in low for k in ["ats","applicant tracking","requisition","candidate","workflow"]):
        return "hr_ats"
    return "hr_attrition_model"

# =========================
# Heuristic KPIs
# =========================
def extract_kpis_heuristic(text: str, filename: str) -> pd.DataFrame:
    sub = detect_hr_subdomain_heuristic(text, filename)
    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name":"Voluntary Attrition Reduction",
             "Description":"% reduction in voluntary exits vs baseline, normalized for headcount and seasonality.",
             "Target Value":"≥ 10%","Status":"Pending"},
            {"KPI Name":"AUC Accuracy",
             "Description":"ROC-AUC for predicting attrition risk; also report by job family and tenure to detect cohort drift.",
             "Target Value":"≥ 0.80","Status":"Pending"},
            {"KPI Name":"High-Risk Follow-through",
             "Description":"Share of high-risk employees receiving a retention action within 14 days (contact, comp review, mobility plan).",
             "Target Value":"≥ 80%","Status":"Pending"},
        ]
    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name":"JD Latency (p50/p90)",
             "Description":"Time from request to publishable draft including bias audit; include median and 90th percentile.",
             "Target Value":"p50 < 10s; p90 < 20s","Status":"Pending"},
            {"KPI Name":"Inclusive Language Improvement",
             "Description":"Reduction in biased terms per 1,000 words after rewrite; uses curated lexicons + semantic matches.",
             "Target Value":"≥ 60%","Status":"Pending"},
            {"KPI Name":"Approval Cycle Time",
             "Description":"Elapsed time from draft to HR signoff across review steps; highlights rework hotspots.",
             "Target Value":"≥ 30% faster","Status":"Pending"},
        ]
    else:
        rows = [
            {"KPI Name":"Time-to-Fill (p50/p90)",
             "Description":"Days from requisition open to accepted offer; report median and p90 by role/location.",
             "Target Value":"≥ 25% faster","Status":"Pending"},
            {"KPI Name":"Application Step Drop-off",
             "Description":"Abandonment rate at each funnel step by device and geography; identifies friction.",
             "Target Value":"Reduce p95 step drop-off by 20%","Status":"Pending"},
            {"KPI Name":"Automation Rate",
             "Description":"% recruiter tasks executed automatically (screening, scheduling, nudges, reminders).",
             "Target Value":"≥ 40% in 6 months","Status":"Pending"},
        ]
    return pd.DataFrame(rows)

HR_KPI_LIB = {
    "hr_attrition_model": [
        ("Dashboard Adoption","Monthly active HR users / licensed users on the attrition dashboard; measures operationalization."),
        ("Retention Program Uptake","% of flagged employees enrolled in retention actions (coaching, mobility, comp review)."),
    ],
    "hr_jd_system": [
        ("Template Utilization","Share of roles starting from approved JD templates; improves consistency and compliance."),
        ("Hiring Manager Adoption","% requisitions where managers edit AI drafts rather than uploading manual JDs."),
    ],
    "hr_ats": [
        ("Candidate Satisfaction (CSAT)","Post-application/interview rating of clarity, fairness, and communication."),
        ("Recruiter Productivity","Candidates/requisitions handled per recruiter while meeting SLAs."),
    ],
}
def recommend_heuristic(existing: list[str], text: str, filename: str) -> list[dict]:
    sub = detect_hr_subdomain_heuristic(text, filename)
    out = []
    for name, desc in HR_KPI_LIB[sub]:
        if name not in existing:
            out.append({"KPI Name":name,"Description":desc,"Owner/ SME":"","Target Value":"","Status":"Pending"})
    return out

# =========================
# LLM helpers (optional)
# =========================
def _cache_key(prefix: str, payload: str) -> str:
    return prefix + ":" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _json_from_llm(system: str, user: str) -> dict | None:
    if not USE_OPENAI: return None
    cache_id = _cache_key("llm", system + "\n---\n" + user)
    if cache_id in st.session_state["llm_cache"]:
        return st.session_state["llm_cache"][cache_id]
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?","",text).strip()
        text = re.sub(r"```$","",text).strip()
        data = json.loads(text)
        st.session_state["llm_cache"][cache_id] = data
        return data
    except Exception:
        return None

CLASSIFY_SYS = "Classify the HR BRD into hr_attrition_model, hr_jd_system, or hr_ats. Return ONLY JSON {\"subdomain\":\"...\"}."
EXTRACT_SYS = (
    "Extract 4–7 KPIs with 2–3 sentence descriptions and targets if stated. "
    "Return ONLY JSON {\"kpis\":[{\"KPI Name\":str,\"Description\":str,\"Target Value\":str}]}. "
    "Avoid generic 'System Uptime'."
)
RECOMMEND_SYS = (
    "Suggest 3–6 additional KPIs tailored to the BRD. Avoid duplicates from the existing list. "
    "Each needs a 2–3 sentence actionable description. "
    "Return ONLY JSON {\"kpis\":[{\"KPI Name\":str,\"Description\":str}]}."
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
            rows.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                         "Target Value":str(it.get("Target Value","")).strip(),"Status":"Pending"})
    if not rows: return extract_kpis_heuristic(text, filename)
    return pd.DataFrame(rows).drop_duplicates(subset=["KPI Name"])

def recommend_llm(existing: list[str], subdomain: str, text: str, filename: str) -> list[dict]:
    existing_str = ", ".join(sorted(existing)) or "(none)"
    out = _json_from_llm(RECOMMEND_SYS,
                         f"Subdomain: {subdomain}\nExisting: {existing_str}\nFilename: {filename}\n\nBRD:\n{text[:16000]}")
    recs = []
    if out and isinstance(out.get("kpis"), list):
        for it in out["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name or name in existing: continue
            recs.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                         "Owner/ SME":"","Target Value":"","Status":"Pending"})
    if not recs: return recommend_heuristic(existing, text, filename)
    return recs[:6]

# =========================
# Row CSS and no-wrap buttons
# =========================
st.markdown("""
<style>
.chip{display:inline-block;padding:4px 10px;border-radius:999px;color:#fff;font-size:12px}
.chip-pending{background:#9ca3af}.chip-ok{background:#16a34a}.chip-bad{background:#b91c1c}.chip-accepted{background:#059669}
.cell{padding:8px 10px;border-top:1px solid #e5e7eb}
.stButton>button{white-space:nowrap;}
</style>
""", unsafe_allow_html=True)

# =========================
# Table renderers
# =========================
def render_table(brd, df, source, key_prefix):
    if df.empty:
        st.caption(f"No {source} KPIs.")
        return df

    updated_rows = []
    for i, r in df.iterrows():
        if source == "Recommended":
            # widen action column to avoid wrapping
            c1, c2, c3, c4, c5, c6 = st.columns([2.1, 3.3, 1.6, 1.2, 0.9, 2.4], gap="small")
            with c1: st.markdown(f"**{r['KPI Name']}**")
            with c2: st.markdown(r["Description"])
            with c3:
                owner_val = st.text_input("", value=r.get("Owner/ SME",""),
                                          key=f"{key_prefix}_owner_{i}",
                                          label_visibility="collapsed",
                                          placeholder="Owner / SME")
            with c4:
                target_val = st.text_input("", value=r.get("Target Value",""),
                                           key=f"{key_prefix}_target_{i}",
                                           label_visibility="collapsed",
                                           placeholder="Target")
            with c5:
                st.markdown(_chip(r.get("Status","Pending")), unsafe_allow_html=True)
            with c6:
                b1, b2 = st.columns(2, gap="small")
                with b1:
                    if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                        _upsert_final(
                            brd,
                            {"BRD": brd, "KPI Name": r["KPI Name"], "Source": source,
                             "Description": r["Description"], "Owner/ SME": owner_val.strip(),
                             "Target Value": target_val.strip(), "Status": "Validated"}
                        )
                        df.at[i, "Status"] = "Validated"; st.rerun()
                with b2:
                    if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                        _remove_from_final(brd, r["KPI Name"])
                        df.at[i, "Status"] = "Rejected"; st.rerun()
            updated_rows.append({
                "KPI Name": r["KPI Name"], "Description": r["Description"],
                "Owner/ SME": owner_val, "Target Value": target_val,
                "Status": df.at[i, "Status"]
            })

        else:  # Extracted
            c1, c2, c3, c4, c5 = st.columns([2.1, 3.3, 1.2, 0.9, 2.4], gap="small")
            with c1: st.markdown(f"**{r['KPI Name']}**")
            with c2: st.markdown(r["Description"])
            with c3:
                target_val = st.text_input("", value=r.get("Target Value",""),
                                           key=f"{key_prefix}_target_{i}",
                                           label_visibility="collapsed",
                                           placeholder="Target")
            with c4:
                st.markdown(_chip(r.get("Status","Pending")), unsafe_allow_html=True)
            with c5:
                b1, b2 = st.columns(2, gap="small")
                with b1:
                    if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                        _upsert_final(
                            brd,
                            {"BRD": brd, "KPI Name": r["KPI Name"], "Source": source,
                             "Description": r["Description"], "Owner/ SME": "",
                             "Target Value": target_val.strip(), "Status": "Validated"}
                        )
                        df.at[i, "Status"] = "Validated"; st.rerun()
                with b2:
                    if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                        _remove_from_final(brd, r["KPI Name"])
                        df.at[i, "Status"] = "Rejected"; st.rerun()
            updated_rows.append({
                "KPI Name": r["KPI Name"], "Description": r["Description"],
                "Owner/ SME": "", "Target Value": target_val,
                "Status": df.at[i, "Status"]
            })

    return pd.DataFrame(updated_rows)

def render_finalized_table(brd: str):
    """
    Finalized KPIs table (same 6-column layout as Recommended):
    Name | Description | Owner/SME (editable) | Target (editable) | Status chip | spacer
    Footer buttons:
      - Review & Accept (sets all rows to Accepted)
      - Download Excel
    """
    df = _ensure_final_df(brd).copy()
    if df.empty:
        st.caption("No validated KPIs yet.")
        return

    for i, r in df.iterrows():
        c1, c2, c3, c4, c5, c6 = st.columns([2.1, 3.3, 1.6, 1.2, 0.9, 2.4], gap="small")
        with c1: st.markdown(f"**{r['KPI Name']}**")
        with c2: st.markdown(r["Description"])
        with c3:
            owner_val = st.text_input(
                "", value=r.get("Owner/ SME",""),
                key=f"final_owner_{brd}_{i}", label_visibility="collapsed",
                placeholder="Owner / SME"
            )
        with c4:
            target_val = st.text_input(
                "", value=r.get("Target Value",""),
                key=f"final_target_{brd}_{i}", label_visibility="collapsed",
                placeholder="Target"
            )
        with c5:
            st.markdown(_chip(r.get("Status","Validated")), unsafe_allow_html=True)
        with c6:
            st.write("")  # spacer for symmetry

        # Persist edits immediately
        df.at[i, "Owner/ SME"] = owner_val.strip()
        df.at[i, "Target Value"] = target_val.strip()

    st.session_state["final_kpis"][brd] = df

    st.markdown("")
    b1, b2 = st.columns([1, 1], gap="large")

    with b1:
        if st.button(f"Review & Accept", key=f"accept_all_{brd}"):
            # Accept all non-empty KPI rows
            mask = df["KPI Name"].astype(str).str.strip() != ""
            df.loc[mask, "Status"] = "Accepted"
            st.session_state["final_kpis"][brd] = df
            st.success("✅ All finalized KPIs have been accepted.")

    with b2:
        export_cols = ["KPI Name","Source","Owner/ SME","Target Value","Description","Status"]
        export_df = df.copy()[["BRD"] + export_cols]
        export_df = export_df.sort_values(["BRD","KPI Name"]).reset_index(drop=True)

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Finalized KPIs")
            ws = writer.sheets["Finalized KPIs"]
            ws.set_column(0, 0, 28)  # BRD
            ws.set_column(1, 1, 30)  # KPI Name
            ws.set_column(2, 2, 12)  # Source
            ws.set_column(3, 3, 18)  # Owner/ SME
            ws.set_column(4, 4, 14)  # Target Value
            ws.set_column(5, 5, 60)  # Description
            ws.set_column(6, 6, 12)  # Status
        buf.seek(0)

        st.download_button(
            "Download Excel",
            data=buf,
            file_name=f"{brd}_finalized_kpis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{brd}",
        )

# =========================
# Login UI
# =========================
def login_page():
    st.markdown("### <div style='text-align:center;color:#b91c1c;'>AI KPI System</div>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if _check_credentials(email, password):
            st.session_state["auth"]=True; st.session_state["user"]=email.strip().lower(); st.rerun()
        else:
            st.error("Invalid email or password")

# =========================
# MAIN
# =========================
if not st.session_state["auth"]:
    login_page(); st.stop()

st.title("AI KPI Extraction & Recommendations (LLM-first)")

uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"], accept_multiple_files=True)

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

    st.session_state.projects[fname] = {
        "extracted": extracted,
        "recommended": pd.DataFrame(recs),
        "domain": sub
    }
    _ensure_final_df(fname)

if st.button("Process BRDs"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for f in uploads:
            process_file(f)
        st.success(f"Processed {len(uploads)} BRDs")

# Per-BRD pages
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname}")
    st.caption(f"Detected subdomain: **{proj.get('domain','hr_attrition_model').replace('_',' ').title()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_table(fname, proj["extracted"], "Extracted", key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_table(fname, proj["recommended"], "Recommended", key_prefix=f"rec_{fname}")

    st.subheader("Finalized KPIs")
    render_finalized_table(fname)

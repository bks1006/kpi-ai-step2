from __future__ import annotations
import io, os, re, json, hashlib
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

# ============ Optional OCR ============
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ============ LLM Setup ============
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

# ============ Demo credentials ============
VALID_USERS = {"admin@company.com": "password123", "user@company.com": "welcome123"}

# ============ Page ============
st.set_page_config(page_title="AI KPI System", layout="wide")

# ============ Session ============
if "auth" not in st.session_state: st.session_state["auth"] = False
if "user" not in st.session_state: st.session_state["user"] = None
if "projects" not in st.session_state: st.session_state["projects"] = {}
if "final_kpis" not in st.session_state: st.session_state["final_kpis"] = {}
if "llm_cache" not in st.session_state: st.session_state["llm_cache"] = {}

# ============ Utilities ============
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
    # make sure all columns exist (fixes KeyError)
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[FINAL_COLS]
    st.session_state["final_kpis"][brd] = df
    return df

def _upsert_final(brd, row):
    df = _ensure_final_df(brd)
    for c in FINAL_COLS:
        if c not in row:  # fill missing fields
            row[c] = ""
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.drop_duplicates(subset=["KPI Name"], keep="last", inplace=True)
    st.session_state["final_kpis"][brd] = df

def _remove_from_final(brd, name):
    df = _ensure_final_df(brd)
    if not df.empty:
        df = df[df["KPI Name"] != name].reset_index(drop=True)
    st.session_state["final_kpis"][brd] = df

# ============ File reading ============
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

# ============ Domain detection ============
def detect_hr_subdomain_heuristic(text: str, filename: str = "") -> str:
    low = (text + " " + filename).lower()
    if any(k in low for k in ["attrition","retention","churn"]): return "hr_attrition_model"
    if any(k in low for k in ["job description"," jd ","inclusive","bias"]): return "hr_jd_system"
    if any(k in low for k in ["ats","applicant tracking","requisition","candidate"]): return "hr_ats"
    return "hr_attrition_model"

# ============ Heuristic KPIs ============
def extract_kpis_heuristic(text: str, filename: str) -> pd.DataFrame:
    sub = detect_hr_subdomain_heuristic(text, filename)
    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name":"Voluntary Attrition Reduction",
             "Description":"% reduction in voluntary exits vs baseline, normalized for headcount and seasonality.",
             "Target Value":"≥ 10%","Status":"Pending"},
            {"KPI Name":"AUC Accuracy",
             "Description":"ROC-AUC for predicting attrition risk; also report by job family and tenure to catch cohort drift.",
             "Target Value":"≥ 0.80","Status":"Pending"},
        ]
    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name":"JD Latency",
             "Description":"Median time from request to publishable draft including bias audit; measure p50/p90.",
             "Target Value":"< 20s","Status":"Pending"},
            {"KPI Name":"Inclusive Language Improvement",
             "Description":"Reduction in flagged gendered/ableist terms per 1,000 words in generated JDs.",
             "Target Value":"≥ 60%","Status":"Pending"},
        ]
    else:
        rows = [
            {"KPI Name":"Time-to-Fill",
             "Description":"Days from requisition open to accepted offer; show p50/p90 by role/location.",
             "Target Value":"< 30 days","Status":"Pending"},
            {"KPI Name":"Automation Rate",
             "Description":"% recruiter tasks executed automatically (screening, scheduling, nudges).",
             "Target Value":"≥ 40%","Status":"Pending"},
        ]
    return pd.DataFrame(rows)

HR_KPI_LIB = {
    "hr_attrition_model": [("Dashboard Adoption","Monthly active users / licensed users.")],
    "hr_jd_system": [("Template Utilization","Share of roles starting from approved JD templates.")],
    "hr_ats": [("Candidate Satisfaction","Average candidate satisfaction rating.")]
}
def recommend_heuristic(existing: list[str], text: str, filename: str) -> list[dict]:
    sub = detect_hr_subdomain_heuristic(text, filename)
    out = []
    for name, desc in HR_KPI_LIB[sub]:
        if name not in existing:
            out.append({"KPI Name":name,"Description":desc,"Owner/ SME":"","Target Value":"","Status":"Pending"})
    return out

# ============ LLM helpers ============
def _cache_key(prefix: str, payload: str) -> str:
    return prefix + ":" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _json_from_llm(system: str, user: str) -> dict | None:
    if not USE_OPENAI: return None
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
        text = re.sub(r"^```(?:json)?","",text).strip()
        text = re.sub(r"```$","",text).strip()
        data = json.loads(text)
        st.session_state["llm_cache"][cache_id] = data
        return data
    except Exception:
        return None

CLASSIFY_SYS = "Classify BRD into hr_attrition_model, hr_jd_system, or hr_ats. Return JSON {\"subdomain\":\"...\"}."
EXTRACT_SYS = "Extract 4–7 KPIs with descriptions and targets. Return JSON {\"kpis\":[{...}]}."
RECOMMEND_SYS = "Suggest 3–6 additional KPIs with detailed descriptions. Return JSON {\"kpis\":[{...}]}."

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
    return pd.DataFrame(rows)

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

# ============ Row table rendering ============
ROW_CSS = """
<style>
.chip{display:inline-block;padding:4px 10px;border-radius:999px;color:#fff;font-size:12px}
.chip-pending{background:#9ca3af}.chip-ok{background:#16a34a}.chip-bad{background:#b91c1c}.chip-accepted{background:#059669}
.inline-btn > div > button{width:100%}
.cell{padding:8px 10px;border-top:1px solid #e5e7eb}
</style>
"""
st.markdown(ROW_CSS, unsafe_allow_html=True)

def render_table(brd, df, source, key_prefix):
    if df.empty: return df
    updated = []
    for i, r in df.iterrows():
        if source == "Recommended":
            c1, c2, c3, c4, c5, c6 = st.columns([2.1, 3.3, 1.5, 1.2, 0.9, 1.6], gap="small")
            with c1: st.markdown(f"**{r['KPI Name']}**")
            with c2: st.markdown(r['Description'])
            with c3:
                owner_val = st.text_input("", value=r.get("Owner/ SME",""),
                                          key=f"{key_prefix}_o_{i}", label_visibility="collapsed",
                                          placeholder="Owner / SME")
            with c4:
                target_val = st.text_input("", value=r.get("Target Value",""),
                                           key=f"{key_prefix}_t_{i}", label_visibility="collapsed",
                                           placeholder="Target")
            with c5: st.markdown(_chip(r["Status"]), unsafe_allow_html=True)
            with c6:
                b1, b2 = st.columns([1,1], gap="small")   # keeps buttons on the SAME ROW
                with b1:
                    if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                        _upsert_final(
                            brd,
                            {"BRD": brd, "KPI Name": r["KPI Name"], "Source": source,
                             "Description": r["Description"], "Owner/ SME": owner_val,
                             "Target Value": target_val, "Status": "Validated"}
                        )
                        df.at[i,"Status"] = "Validated"; st.rerun()
                with b2:
                    if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                        _remove_from_final(brd, r["KPI Name"])
                        df.at[i,"Status"] = "Rejected"; st.rerun()
            updated.append({"KPI Name":r["KPI Name"],"Description":r["Description"],
                            "Owner/ SME":owner_val,"Target Value":target_val,"Status":df.at[i,"Status"]})
        else:
            c1, c2, c3, c4, c5 = st.columns([2.1,3.3,1.2,0.9,1.6], gap="small")
            with c1: st.markdown(f"**{r['KPI Name']}**")
            with c2: st.markdown(r['Description'])
            with c3:
                target_val = st.text_input("", value=r.get("Target Value",""),
                                           key=f"{key_prefix}_t_{i}", label_visibility="collapsed",
                                           placeholder="Target")
            with c4: st.markdown(_chip(r["Status"]), unsafe_allow_html=True)
            with c5:
                b1, b2 = st.columns([1,1], gap="small")   # SAME ROW
                with b1:
                    if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                        _upsert_final(
                            brd,
                            {"BRD": brd, "KPI Name": r["KPI Name"], "Source": source,
                             "Description": r["Description"], "Owner/ SME": "",
                             "Target Value": target_val, "Status": "Validated"}
                        )
                        df.at[i,"Status"] = "Validated"; st.rerun()
                with b2:
                    if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                        _remove_from_final(brd, r["KPI Name"])
                        df.at[i,"Status"] = "Rejected"; st.rerun()
            updated.append({"KPI Name":r["KPI Name"],"Description":r["Description"],
                            "Owner/ SME":"", "Target Value":target_val,"Status":df.at[i,"Status"]})
    return pd.DataFrame(updated)

# ============ Login ============
def login_page():
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if _check_credentials(email, password):
            st.session_state["auth"]=True; st.session_state["user"]=email.strip().lower(); st.rerun()
        else: st.error("Invalid email or password")

# ============ MAIN ============
if not st.session_state["auth"]:
    login_page(); st.stop()

st.title("AI KPI Extraction & Recommendations")

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
    recommended = pd.DataFrame(recs)
    st.session_state.projects[fname] = {"extracted":extracted,"recommended":recommended,"domain":sub}
    _ensure_final_df(fname)

if st.button("Process BRDs"):
    if not uploads: st.warning("Please upload at least one file")
    else:
        for f in uploads: process_file(f)
        st.success(f"Processed {len(uploads)} BRDs")

# ============ Per BRD ============
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname} — Domain: {proj.get('domain','')}")
    st.subheader("Extracted KPIs")
    proj["extracted"] = render_table(fname, proj["extracted"], "Extracted", key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_table(fname, proj["recommended"], "Recommended", key_prefix=f"rec_{fname}")

    st.subheader("Finalized KPIs")
    final_df = _ensure_final_df(fname)
    if final_df.empty:
        st.caption("No validated KPIs yet.")
    else:
        # show each with chip and accept per-row
        for i, row in final_df.iterrows():
            owner = row.get("Owner/ SME","")
            target = row.get("Target Value","")
            status = row.get("Status","Validated") or "Validated"

            st.markdown(f"**{row['KPI Name']}** — {row['Description']}")
            st.markdown(f"Owner/ SME: {owner} | Target: {target} | {_chip(status)}",
                        unsafe_allow_html=True)

            # Per-row Accept button
            if status != "Accepted":
                if st.button("Review & Accept", key=f"accept_{fname}_{i}"):
                    final_df.at[i,"Status"] = "Accepted"
                    st.session_state["final_kpis"][fname] = final_df
                    st.success(f"✅ {row['KPI Name']} accepted.")
            st.divider()

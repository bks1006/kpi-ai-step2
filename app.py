from __future__ import annotations

import io
import os
import json
import re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # pulls the secret from Streamlit Cloud

if not OPENAI_API_KEY:
    # You can also hardcode a fallback here for local testing (not recommended in production)
    # OPENAI_API_KEY = "sk-..."
    pass

client = OpenAI(api_key=OPENAI_API_KEY)



# ---------- OCR fallback ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- LLM toggle & setup ----------
USE_OPENAI = True  # set False to force heuristic fallback without calling an API

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if USE_OPENAI and not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set. Falling back to heuristics.")
    USE_OPENAI = False

if USE_OPENAI:
    try:
        # OpenAI python SDK >= 1.0
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_MODEL = "gpt-4o-mini"  # fast & capable; you can switch to "gpt-4.1" for higher quality
    except Exception:
        st.warning("OpenAI SDK not available. Falling back to heuristics.")
        USE_OPENAI = False

# ---------- Demo credentials ----------
VALID_USERS = {
    "admin@company.com": "password123",
    "user@company.com": "welcome123"
}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    :root { --brand:#b91c1c; --green:#16a34a; --red:#b91c1c; }
    .block-container { padding-top: 1.0rem; }

    /* Sticky top bar */
    .topbar {
      position: sticky; top: 0; z-index: 5;
      background: white; padding: 6px 0 8px 0; margin-bottom: 4px;
      border-bottom: 1px solid #eee;
    }
    .topbar-inner { display:flex; justify-content:space-between; align-items:center; }
    .who { color:#6b7280; font-size:14px; }

    /* Section tables */
    .th-row {
      background:#f3f4f6; border:1px solid #e5e7eb; border-bottom:0;
      padding:10px 12px; border-radius:10px 10px 0 0; font-weight:700;
      display:grid;
    }
    .tb { border:1px solid #e5e7eb; border-top:0; border-radius:0 0 10px 10px; }
    .cell { padding:10px 12px; border-top:1px solid #e5e7eb; }

    /* Status chips */
    .chip { display:inline-block; padding:4px 10px; border-radius:999px; color:#fff; font-size:12px;}
    .chip-pending{ background:#9ca3af;}
    .chip-ok{ background:#16a34a;}
    .chip-bad{ background:#b91c1c;}

    /* Global brand inputs */
    .stTextInput > div > div > input,
    .stTextArea  > div > div > textarea,
    .stSelectbox > div > div > select {
      border:1.6px solid var(--brand) !important; border-radius:8px !important; background:#fff !important;
      padding:6px 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea  > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
      border:2px solid var(--brand) !important; box-shadow:0 0 6px var(--brand) !important; outline:none !important;
    }

    /* LOGIN: neutral border by default, red on focus */
    .login-card .stTextInput > div > div > input,
    .login-card .stPasswordInput > div > div > input {
      border:1px solid #d1d5db !important;
      box-shadow:none !important;
      outline:none !important;
      border-radius:8px !important;
      background:#fff !important;
    }
    .login-card .stTextInput > div > div > input:focus,
    .login-card .stPasswordInput > div > div > input:focus {
      border:2px solid var(--brand) !important;
      box-shadow:0 0 6px var(--brand) !important;
      outline:none !important;
    }

    /* Validate/Reject buttons: plain by default; color AFTER action */
    .btn-wrap button {
      background:#f9fafb !important;
      color:#111827 !important;
      border:1px solid #e5e7eb !important;
      border-radius:6px !important;
      padding:0.4rem 0.8rem !important;
      font-weight:600 !important;
      box-shadow:none !important;
    }
    .btn-wrap.on-validate button { background:var(--green)!important; color:#fff!important; }
    .btn-wrap.on-reject   button { background:var(--red)!important;   color:#fff!important; }
    .btn-wrap button:hover { filter:brightness(0.96); }

    /* FORCE "Review & Accept" button styling (robust to Streamlit wrappers) */
    .accept-btn button,
    .accept-btn .stButton button,
    div.accept-btn > div > div > button {
        background-color: #b91c1c !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .accept-btn button:hover { filter: brightness(0.9); }

    .centered { display:flex; justify-content:center; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Session defaults ----------
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "projects" not in st.session_state:
    st.session_state["projects"] = {}
if "final_kpis" not in st.session_state:
    st.session_state["final_kpis"] = {}

# ---------- Auth utils ----------
def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

# ---------- UI helpers ----------
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
    lname = name.lower()
    bio = io.BytesIO(data)

    if lname.endswith(".pdf"):
        try:
            reader = PdfReader(bio)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            if txt.strip():
                return txt
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
        except Exception:
            return ""
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts)

    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def read_uploaded(file) -> str:
    return read_text_from_bytes(file.read(), file.name)

# ---------- Heuristic fallback (used if LLM is off/unavailable) ----------
HR_KPI_LIB = {
    "hr_attrition_model": [
        ("Model Accuracy", "Classification accuracy of the attrition prediction model."),
        ("Voluntary Attrition Reduction", "Percent reduction in voluntary attrition vs baseline over 12 months."),
        ("High-Risk Coverage", "Percent of high-risk employees flagged with actionable insights."),
        ("Insight Coverage", "Percent of high-risk cases with identified drivers/insights."),
        ("Dashboard Adoption", "Share of HR users who actively use the risk dashboard each month.")
    ],
    "hr_jd_system": [
        ("JD Generation Latency", "Median time to generate or redesign a job description."),
        ("Bias Term Reduction", "Percent reduction of gendered or non-inclusive terms in JDs."),
        ("Approval Cycle Time", "Median time from JD draft to final HR approval."),
        ("JD Repository Utilization", "Percent of roles using repository templates or redesigned JDs."),
        ("Hiring Manager Adoption", "Share of hiring managers who use the AI JD tool monthly.")
    ],
    "hr_ats": [
        ("Application Drop-off Rate", "Percent of candidates abandoning during the application flow."),
        ("Time-to-Fill", "Median days from requisition open to offer acceptance."),
        ("Automation Rate", "Share of recruiter tasks handled by automated workflows."),
        ("Recruiter Productivity", "Requisitions or candidates handled per recruiter per month."),
        ("Candidate Satisfaction (CSAT)", "Post-application or post-interview satisfaction score.")
    ],
}

def detect_hr_subdomain_heuristic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["attrition", "retention", "predictive model", "risk score"]):
        return "hr_attrition_model"
    if any(k in low for k in ["job description", " jd ", "bias detection", "inclusive language", "jd tool"]):
        return "hr_jd_system"
    if any(k in low for k in ["ats", "requisition", "candidate experience", "application", "workflow"]):
        return "hr_ats"
    return "hr_attrition_model"

def extract_kpis_heuristic(text: str) -> pd.DataFrame:
    sub = detect_hr_subdomain_heuristic(text)
    rows = []
    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name": "Model Accuracy", "Description": "Classification accuracy of the attrition model.", "Target Value": "≥ 85%", "Status": "Pending"},
            {"KPI Name": "Voluntary Attrition Reduction", "Description": "Reduction in voluntary attrition vs baseline over 12 months.", "Target Value": "10% in 12 months", "Status": "Pending"},
            {"KPI Name": "Insight Coverage", "Description": "Percent of high-risk cases with identified drivers/insights.", "Target Value": "≥ 80%", "Status": "Pending"},
        ]
        if "dashboard" in text.lower():
            rows.append({"KPI Name": "Dashboard Adoption", "Description": "Active HR users of the risk dashboard per month.", "Target Value": "", "Status": "Pending"})
    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name": "JD Generation Latency", "Description": "Median time to generate/redesign a JD.", "Target Value": "< 10 seconds", "Status": "Pending"},
            {"KPI Name": "Bias Term Reduction", "Description": "Reduction of gendered/non-inclusive terms in JDs.", "Target Value": "Increase vs baseline", "Status": "Pending"},
            {"KPI Name": "Approval Cycle Time", "Description": "Draft → manager review → HR approval time.", "Target Value": "Decrease vs baseline", "Status": "Pending"},
        ]
    elif sub == "hr_ats":
        rows = [
            {"KPI Name": "Application Drop-off Rate", "Description": "Percent abandoning during application stages.", "Target Value": "Decrease vs baseline", "Status": "Pending"},
            {"KPI Name": "Time-to-Fill", "Description": "Median days from requisition to offer acceptance.", "Target Value": "Decrease vs baseline", "Status": "Pending"},
            {"KPI Name": "Automation Rate", "Description": "Share of workflow steps automated end-to-end.", "Target Value": "Increase vs baseline", "Status": "Pending"},
        ]
    return pd.DataFrame(rows)

def recommend_heuristic(existing: list, raw_text: str = "") -> list[dict]:
    sub = detect_hr_subdomain_heuristic(raw_text)
    pool = HR_KPI_LIB[sub]
    out = []
    for name, desc in pool:
        if name not in existing:
            out.append({"KPI Name": name, "Description": desc, "Owner/ SME": "", "Target Value": "", "Status": "Pending"})
    return out[:5]

# ---------- LLM prompts ----------
CLASSIFY_SYS_PROMPT = (
    "You are a precise classifier for HR BRD documents. "
    "Classify the document into one of: "
    "[hr_attrition_model, hr_jd_system, hr_ats]. "
    "Return ONLY a JSON object: {\"subdomain\": \"<one_of_three>\"}."
)

EXTRACT_SYS_PROMPT = (
    "You are an information extraction system. Given a BRD, extract KPIs explicitly required "
    "or clearly implied. Return a JSON object with a 'kpis' array. Each item has: "
    "{\"KPI Name\": str, \"Description\": str, \"Target Value\": str}. "
    "If a target isn't specified, leave it empty. Keep 3-6 concise KPIs, no duplicates."
)

RECOMMEND_SYS_PROMPT = (
    "You are a KPI recommender. Based on the BRD and the subdomain, suggest 3-6 additional KPIs "
    "that are relevant and not in the provided list. Return a JSON object with 'kpis' array "
    "of items {\"KPI Name\": str, \"Description\": str}. Keep names short and standard."
)

def openai_json_chat(system_prompt: str, user_prompt: str) -> dict | None:
    """Call OpenAI chat and parse a top-level JSON object safely."""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = resp.choices[0].message.content.strip()
        # Try direct JSON; if it contains code fences, strip them.
        content = re.sub(r"^```(json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
        return json.loads(content)
    except Exception as e:
        # You can print e for debugging locally; Streamlit Cloud hides details.
        return None

def detect_hr_subdomain_llm(text: str) -> str:
    payload = openai_json_chat(CLASSIFY_SYS_PROMPT, f"BRD Text:\n{text[:12000]}")
    if payload and isinstance(payload, dict) and "subdomain" in payload:
        v = payload["subdomain"]
        if v in {"hr_attrition_model", "hr_jd_system", "hr_ats"}:
            return v
    # fallback
    return detect_hr_subdomain_heuristic(text)

def extract_kpis_llm(text: str) -> pd.DataFrame:
    payload = openai_json_chat(EXTRACT_SYS_PROMPT, f"BRD Text:\n{text[:16000]}")
    rows = []
    if payload and isinstance(payload, dict) and isinstance(payload.get("kpis"), list):
        for item in payload["kpis"]:
            name = str(item.get("KPI Name", "")).strip()
            if not name:
                continue
            rows.append({
                "KPI Name": name,
                "Description": str(item.get("Description", "")).strip(),
                "Target Value": str(item.get("Target Value", "")).strip(),
                "Status": "Pending",
            })
    if not rows:
        # final fallback
        return extract_kpis_heuristic(text)
    # dedupe by name
    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["KPI Name"], keep="first", inplace=True)
    return df

def recommend_llm(existing: list[str], subdomain: str, text: str) -> list[dict]:
    existing_str = ", ".join(sorted(existing))
    user_prompt = (
        f"Subdomain: {subdomain}\n"
        f"Existing KPIs: {existing_str if existing_str else '(none)'}\n\n"
        f"BRD Text:\n{text[:16000]}"
    )
    payload = openai_json_chat(RECOMMEND_SYS_PROMPT, user_prompt)
    out = []
    if payload and isinstance(payload, dict) and isinstance(payload.get("kpis"), list):
        for item in payload["kpis"]:
            name = str(item.get("KPI Name", "")).strip()
            if not name or name in existing:
                continue
            out.append({
                "KPI Name": name,
                "Description": str(item.get("Description", "")).strip(),
                "Owner/ SME": "",
                "Target Value": "",
                "Status": "Pending",
            })
    if not out:
        return recommend_heuristic(existing, raw_text=text)
    # keep it tidy
    return out[:6]

# ---------- Table helpers ----------
def _table_head(col_template: str, headers: list[str]):
    st.markdown(
        f"<div class='th-row' style='grid-template-columns:{col_template};'>" +
        "".join([f"<div>{h}</div>" for h in headers]) + "</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='tb'>", unsafe_allow_html=True)

def _table_tail():
    st.markdown("</div>", unsafe_allow_html=True)

def render_extracted_table(brd, df, key_prefix):
    if df.empty:
        st.caption("No extracted KPIs.")
        return df
    cols = "2fr 3fr 1.2fr 0.9fr 1.6fr"
    _table_head(cols, ["KPI Name","Description","Target Value","Status","Actions"])

    updated = []
    for i, r in df.iterrows():
        status = r["Status"]
        c1, c2, c3, c4, c5 = st.columns([2,3,1.2,0.9,1.6])
        with c1: st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)
        with c3: target_val = st.text_input("", value=r.get("Target Value",""), key=f"{key_prefix}_t_{i}")
        with c4: st.markdown(f"<div class='cell'>{_chip(status)}</div>", unsafe_allow_html=True)
        with c5:
            st.markdown("<div class='cell'>", unsafe_allow_html=True)
            v_on  = "on-validate" if status == "Validated" else ""
            rej_on= "on-reject"   if status == "Rejected"  else ""
            col_v, col_r = st.columns([1,1])
            with col_v:
                st.markdown(f"<div class='btn-wrap {v_on}'>", unsafe_allow_html=True)
                if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                    status = "Validated"
                    _upsert_final(brd, {
                        "BRD": brd, "KPI Name": r["KPI Name"], "Source": "Extracted",
                        "Description": r["Description"], "Owner/ SME": "", "Target Value": target_val
                    })
                    r["Status"] = status
                    st.session_state["projects"][brd]["extracted"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div class='btn-wrap {rej_on}'>", unsafe_allow_html=True)
                if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                    status = "Rejected"
                    _remove_from_final(brd, r["KPI Name"])
                    r["Status"] = status
                    st.session_state["projects"][brd]["extracted"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        updated.append({"KPI Name":r["KPI Name"],"Description":r["Description"],"Target Value":target_val,"Status":status})
    _table_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

def render_recommended_table(brd, df, key_prefix):
    if df.empty:
        st.caption("No recommended KPIs.")
        return df
    cols = "2fr 2.5fr 1fr 1fr 0.9fr 1.6fr"
    _table_head(cols, ["KPI Name","Description","Owner/ SME","Target Value","Status","Actions"])

    updated = []
    for i, r in df.iterrows():
        status = r["Status"]
        c1,c2,c3,c4,c5,c6 = st.columns([2,2.5,1,1,0.9,1.6])
        with c1: st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)
        with c3: owner_val  = st.text_input("", value=r.get("Owner/ SME",""), key=f"{key_prefix}_o_{i}")
        with c4: target_val = st.text_input("", value=r.get("Target Value",""), key=f"{key_prefix}_t_{i}")
        with c5: st.markdown(f"<div class='cell'>{_chip(status)}</div>", unsafe_allow_html=True)
        with c6:
            st.markdown("<div class='cell'>", unsafe_allow_html=True)
            v_on  = "on-validate" if status == "Validated" else ""
            rej_on= "on-reject"   if status == "Rejected"  else ""
            col_v, col_r = st.columns([1,1])
            with col_v:
                st.markdown(f"<div class='btn-wrap {v_on}'>", unsafe_allow_html=True)
                if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                    status = "Validated"
                    _upsert_final(brd, {
                        "BRD": brd, "KPI Name": r["KPI Name"], "Source": "Recommended",
                        "Description": r["Description"], "Owner/ SME": owner_val, "Target Value": target_val
                    })
                    r["Status"] = status
                    st.session_state["projects"][brd]["recommended"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div class='btn-wrap {rej_on}'>", unsafe_allow_html=True)
                if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                    status = "Rejected"
                    _remove_from_final(brd, r["KPI Name"])
                    r["Status"] = status
                    st.session_state["projects"][brd]["recommended"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        updated.append({
            "KPI Name":r["KPI Name"], "Description":r["Description"],
            "Owner/ SME":owner_val, "Target Value":target_val, "Status":status
        })
    _table_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

def manual_kpi_adder(brd):
    st.markdown("#### Add KPI manually")
    with st.form(key=f"manual_add_{brd}", clear_on_submit=True):
        kpi_name = st.text_input("KPI Name *", value="")
        c3, c4 = st.columns([1,1])
        owner    = c3.text_input("Owner/ SME", value="")
        target   = c4.text_input("Target Value", value="")
        add = st.form_submit_button("Add KPI")
    if add:
        if not kpi_name.strip():
            st.warning("Please enter a KPI Name.")
            return
        rec_df = st.session_state["projects"][brd]["recommended"]
        ext_df = st.session_state["projects"][brd]["extracted"]
        all_names = set(n.lower() for n in pd.concat([rec_df["KPI Name"], ext_df["KPI Name"]], ignore_index=True).astype(str))
        if kpi_name.strip().lower() in all_names:
            st.warning("KPI already exists in this BRD.")
            return
        new_row = {
            "KPI Name": kpi_name.strip(),
            "Description": f"Auto-generated description for {kpi_name.strip()}",
            "Owner/ SME": owner.strip(),
            "Target Value": target.strip(),
            "Status": "Pending",
        }
        rec_df = pd.concat([rec_df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["projects"][brd]["recommended"] = rec_df
        st.success("KPI added to Recommended.")
        st.rerun()

# ---------- Pipeline ----------
def process_file(file):
    text = read_uploaded(file)

    if USE_OPENAI:
        subdomain = detect_hr_subdomain_llm(text)
        extracted = extract_kpis_llm(text)
        recs = recommend_llm(extracted["KPI Name"].tolist(), subdomain, text)
    else:
        subdomain = detect_hr_subdomain_heuristic(text)
        extracted = extract_kpis_heuristic(text)
        recs = recommend_heuristic(extracted["KPI Name"].tolist(), raw_text=text)

    recommended = pd.DataFrame(recs)
    st.session_state.projects[file.name] = {
        "extracted": extracted, "recommended": recommended, "domain": subdomain
    }
    st.session_state["final_kpis"].setdefault(
        file.name, pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
    )

# ---------- Login ----------
def login_page():
    st.markdown("<h2 style='color:#b91c1c;text-align:center'>AI KPI System</h2>", unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.write("Sign in to continue")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", use_container_width=True):
            if _check_credentials(email, password):
                st.session_state["auth"] = True
                st.session_state["user"] = email.strip().lower()
                st.rerun()
            else:
                st.error("Invalid email or password")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================
#        MAIN
# ======================
if not st.session_state["auth"]:
    login_page()
    st.stop()

# --- Top bar ---
st.markdown("<div class='topbar'><div class='topbar-inner'>"
            f"<div class='who'>Signed in as <b>{st.session_state.get('user','')}</b></div>"
            "</div></div>",
            unsafe_allow_html=True)

left, spacer, right = st.columns([9,1,1])
with right:
    if st.button("Log out"):
        for k in ["auth", "user", "projects", "final_kpis"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

st.title("AI KPI Extraction & Recommendations (Per BRD) — LLM Edition")

uploads = st.file_uploader("Upload BRDs", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Process BRDs"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for f in uploads:
            process_file(f)
        count = len(uploads)
        st.success(f"✅ Processed {count} BRD{'s' if count != 1 else ''} successfully")

# Show per BRD
for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname}")
    st.caption(f"Detected subdomain: **{proj.get('domain','hr_attrition_model').replace('_',' ').title()}**")

    st.subheader("Extracted KPIs")
    proj["extracted"] = render_extracted_table(fname, proj["extracted"], key_prefix=f"ext_{fname}")

    st.subheader("Recommended KPIs")
    manual_kpi_adder(fname)
    proj["recommended"] = render_recommended_table(fname, proj["recommended"], key_prefix=f"rec_{fname}")

    st.markdown("#### Finalized KPIs")
    final_df = st.session_state["final_kpis"].get(fname, pd.DataFrame())
    if final_df.empty:
        st.caption("No validated KPIs yet for this BRD.")
    else:
        show = final_df[["KPI Name","Source","Owner/ SME","Target Value","Description"]].sort_values("KPI Name")
        st.dataframe(show, use_container_width=True, hide_index=True)

        # Centered red "Review & Accept" button
        csp1, csp2, csp3 = st.columns([1,2,1])
        with csp2:
            st.markdown("<div class='centered accept-btn'>", unsafe_allow_html=True)
            if st.button("Review & Accept", key=f"accept_{fname}"):
                st.success("✅ Finalized KPIs have been accepted successfully!")
            st.markdown("</div>", unsafe_allow_html=True)




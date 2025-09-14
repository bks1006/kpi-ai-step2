from __future__ import annotations

import io, os, re, json
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- Optional OCR (safe fallback) ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- LLM setup (falls back to heuristics if no key) ----------
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

# ---------- Demo credentials ----------
VALID_USERS = {"admin@company.com": "password123",
               "user@company.com": "welcome123"}

# ---------- Page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

# ---------- Session ----------
if "auth" not in st.session_state: st.session_state["auth"] = False
if "user" not in st.session_state: st.session_state["user"] = None
if "projects" not in st.session_state: st.session_state["projects"] = {}
if "final_kpis" not in st.session_state: st.session_state["final_kpis"] = {}

# ---------- Utils ----------
def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password

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

# ---------- Heuristic (fallback) ----------
def detect_hr_subdomain_heuristic(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["attrition","retention","predictive model","risk score"]): return "hr_attrition_model"
    if any(k in low for k in ["job description"," jd ","inclusive","bias","role profile"]): return "hr_jd_system"
    if any(k in low for k in ["ats","requisition","candidate","application","workflow"]): return "hr_ats"
    return "hr_attrition_model"

def extract_kpis_heuristic(text: str) -> pd.DataFrame:
    sub = detect_hr_subdomain_heuristic(text)
    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name":"Model Accuracy","Description":"Classification accuracy of attrition model.","Target Value":"≥ 85%","Status":"Pending"},
            {"KPI Name":"Voluntary Attrition Reduction","Description":"Reduction vs baseline over 12 months.","Target Value":"10% in 12 months","Status":"Pending"},
            {"KPI Name":"Insight Coverage","Description":"Share of high-risk cases with identified drivers.","Target Value":"≥ 80%","Status":"Pending"},
        ]
    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name":"JD Generation Latency","Description":"Median time to generate/redesign a JD.","Target Value":"< 10 sec","Status":"Pending"},
            {"KPI Name":"Bias Term Reduction","Description":"Reduction of gendered/non-inclusive terms in JDs.","Target Value":"Increase vs baseline","Status":"Pending"},
            {"KPI Name":"Approval Cycle Time","Description":"Time from draft to final approval.","Target Value":"Decrease vs baseline","Status":"Pending"},
        ]
    else:
        rows = [
            {"KPI Name":"Application Drop-off Rate","Description":"Share abandoning during application flow.","Target Value":"Decrease vs baseline","Status":"Pending"},
            {"KPI Name":"Time-to-Fill","Description":"Days from requisition open to offer accept.","Target Value":"Decrease vs baseline","Status":"Pending"},
            {"KPI Name":"Automation Rate","Description":"Share of recruiter steps automated.","Target Value":"Increase vs baseline","Status":"Pending"},
        ]
    return pd.DataFrame(rows)

HR_KPI_LIB = {
    "hr_attrition_model":[("Dashboard Adoption","Share of HR users active monthly."),
                          ("High-Risk Coverage","Percent of high-risk employees flagged.")],
    "hr_jd_system":[("JD Repository Utilization","Share of roles using templates."),
                    ("Hiring Manager Adoption","Share of managers using the tool.")],
    "hr_ats":[("Candidate Satisfaction (CSAT)","Post-application/interview score."),
              ("Recruiter Productivity","Requisitions handled per recruiter.")],
}

def recommend_heuristic(existing: list[str], raw_text: str) -> list[dict]:
    sub = detect_hr_subdomain_heuristic(raw_text)
    out = []
    for name, desc in HR_KPI_LIB[sub]:
        if name not in existing:
            out.append({"KPI Name":name,"Description":desc,"Owner/ SME":"","Target Value":"","Status":"Pending"})
    return out

# ---------- LLM helpers ----------
CLASSIFY_SYS_PROMPT = "Classify the HR BRD into one of: hr_attrition_model, hr_jd_system, hr_ats. Return ONLY JSON: {\"subdomain\": \"<one>\"}."
EXTRACT_SYS_PROMPT = ("Extract 3-6 KPIs from the BRD. Return JSON: {\"kpis\":[{\"KPI Name\":str,\"Description\":str,\"Target Value\":str}]}."
                      "Leave Target Value empty if not specified. Avoid duplicates. Keep concise.")
RECOMMEND_SYS_PROMPT = ("Suggest 3-6 additional KPIs for the given subdomain that are NOT in the existing list. "
                        "Return JSON: {\"kpis\":[{\"KPI Name\":str,\"Description\":str}]}.")

def openai_json_chat(system_prompt: str, user_prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}]
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?","",content).strip()
        content = re.sub(r"```$","",content).strip()
        return json.loads(content)
    except Exception:
        return None

def detect_hr_subdomain_llm(text: str) -> str:
    payload = openai_json_chat(CLASSIFY_SYS_PROMPT, f"BRD Text:\n{text[:12000]}")
    if payload and payload.get("subdomain") in {"hr_attrition_model","hr_jd_system","hr_ats"}:
        return payload["subdomain"]
    return detect_hr_subdomain_heuristic(text)

def extract_kpis_llm(text: str) -> pd.DataFrame:
    payload = openai_json_chat(EXTRACT_SYS_PROMPT, f"BRD Text:\n{text[:16000]}")
    rows = []
    if payload and isinstance(payload.get("kpis"), list):
        for it in payload["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name: continue
            rows.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                         "Target Value":str(it.get("Target Value","")).strip(),"Status":"Pending"})
    if not rows: return extract_kpis_heuristic(text)
    df = pd.DataFrame(rows); df.drop_duplicates(subset=["KPI Name"], inplace=True)
    return df

def recommend_llm(existing: list[str], subdomain: str, text: str) -> list[dict]:
    existing_str = ", ".join(sorted(existing))
    payload = openai_json_chat(RECOMMEND_SYS_PROMPT,
                               f"Subdomain:{subdomain}\nExisting KPIs:{existing_str or '(none)'}\n\nBRD Text:\n{text[:16000]}")
    out = []
    if payload and isinstance(payload.get("kpis"), list):
        for it in payload["kpis"]:
            name = str(it.get("KPI Name","")).strip()
            if not name or name in existing: continue
            out.append({"KPI Name":name,"Description":str(it.get("Description","")).strip(),
                        "Owner/ SME":"","Target Value":"","Status":"Pending"})
    if not out: return recommend_heuristic(existing, raw_text=text)
    return out[:6]

# ---------- Tables / chips ----------
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

def _table_head(cols, headers):
    st.markdown(
        f"<div class='th-row' style='grid-template-columns:{cols};'>" +
        "".join(f"<div>{h}</div>" for h in headers) + "</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='tb'>", unsafe_allow_html=True)

def _table_tail(): st.markdown("</div>", unsafe_allow_html=True)

def render_extracted_table(brd, df, key_prefix):
    if df.empty: st.caption("No extracted KPIs."); return df
    _table_head("2fr 3fr 1.2fr 0.9fr 1.6fr", ["KPI Name","Description","Target Value","Status","Actions"])
    updated = []
    for i, r in df.iterrows():
        status = r["Status"]
        c1,c2,c3,c4,c5 = st.columns([2,3,1.2,0.9,1.6])
        with c1: st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)
        with c3: target_val = st.text_input("", value=r.get("Target Value",""), key=f"{key_prefix}_t_{i}")
        with c4: st.markdown(f"<div class='cell'>{_chip(status)}</div>", unsafe_allow_html=True)
        with c5:
            st.markdown("<div class='cell'>", unsafe_allow_html=True)
            v_on = "on-validate" if status == "Validated" else ""
            rej_on = "on-reject" if status == "Rejected" else ""
            col_v, col_r = st.columns([1,1])
            with col_v:
                st.markdown(f"<div class='btn-wrap {v_on}'>", unsafe_allow_html=True)
                if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                    status = "Validated"
                    _upsert_final(brd, {"BRD":brd,"KPI Name":r["KPI Name"],"Source":"Extracted",
                                        "Description":r["Description"],"Owner/ SME":"","Target Value":target_val})
                    r["Status"] = status
                    st.session_state["projects"][brd]["extracted"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div class='btn-wrap {rej_on}'>", unsafe_allow_html=True)
                if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                    status = "Rejected"; _remove_from_final(brd, r["KPI Name"])
                    r["Status"] = status
                    st.session_state["projects"][brd]["extracted"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        updated.append({"KPI Name":r["KPI Name"],"Description":r["Description"],"Target Value":target_val,"Status":status})
    _table_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

def render_recommended_table(brd, df, key_prefix):
    if df.empty: st.caption("No recommended KPIs."); return df
    _table_head("2fr 2.5fr 1fr 1fr 0.9fr 1.6fr", ["KPI Name","Description","Owner/ SME","Target Value","Status","Actions"])
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
            v_on = "on-validate" if status == "Validated" else ""
            rej_on = "on-reject" if status == "Rejected" else ""
            col_v, col_r = st.columns([1,1])
            with col_v:
                st.markdown(f"<div class='btn-wrap {v_on}'>", unsafe_allow_html=True)
                if st.button("Validate", key=f"{key_prefix}_ok_{i}"):
                    status = "Validated"
                    _upsert_final(brd, {"BRD":brd,"KPI Name":r["KPI Name"],"Source":"Recommended",
                                        "Description":r["Description"],"Owner/ SME":owner_val,"Target Value":target_val})
                    r["Status"] = status
                    st.session_state["projects"][brd]["recommended"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div class='btn-wrap {rej_on}'>", unsafe_allow_html=True)
                if st.button("Reject", key=f"{key_prefix}_rej_{i}"):
                    status = "Rejected"; _remove_from_final(brd, r["KPI Name"])
                    r["Status"] = status
                    st.session_state["projects"][brd]["recommended"].iloc[i]["Status"] = status
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        updated.append({"KPI Name":r["KPI Name"],"Description":r["Description"],
                        "Owner/ SME":owner_val,"Target Value":target_val,"Status":status})
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
        if not kpi_name.strip(): st.warning("Please enter a KPI Name."); return
        rec_df = st.session_state["projects"][brd]["recommended"]
        ext_df = st.session_state["projects"][brd]["extracted"]
        all_names = set(n.lower() for n in pd.concat([rec_df["KPI Name"], ext_df["KPI Name"]], ignore_index=True).astype(str))
        if kpi_name.strip().lower() in all_names:
            st.warning("KPI already exists in this BRD."); return
        new_row = {"KPI Name":kpi_name.strip(),"Description":f"Auto-generated description for {kpi_name.strip()}",
                   "Owner/ SME":owner.strip(),"Target Value":target.strip(),"Status":"Pending"}
        rec_df = pd.concat([rec_df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["projects"][brd]["recommended"] = rec_df
        st.success("KPI added to Recommended."); st.rerun()

# ---------- Login (columns/containers only) ----------
def login_page():
    # Scoped login CSS
    st.markdown("""
    <style>
      :root { --brand:#b91c1c; }
      body, .stApp { background:#ffffff !important; }
      [data-testid="stAppViewContainer"] .main .block-container { padding-top: 12vh; }
      /* Style the first two inputs (email/password) which are the only ones on the page */
      .stTextInput:nth-of-type(1) > div > div,
      .stTextInput:nth-of-type(2) > div > div {
        border:1.6px solid var(--brand) !important;
        border-radius:10px !important;
        background:#fff !important;
        box-shadow:0 0 8px rgba(185,28,28,.15) !important;
      }
      .login-btn > button {
        background: var(--brand) !important; color:#fff !important;
        border:none !important; border-radius:10px !important;
        font-weight:700 !important; padding:.65rem 1.2rem !important; width:100% !important;
      }
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
                btn = st.container()
                with btn:
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

# After login styles
st.markdown("""
<style>
:root { --brand:#b91c1c; --green:#16a34a; --red:#b91c1c; }
.topbar { position:sticky; top:0; z-index:5; background:#fff; padding:6px 0 8px; border-bottom:1px solid #eee; margin-bottom:8px; }
.topbar-inner { display:flex; justify-content:space-between; align-items:center; }
.who { color:#6b7280; font-size:14px; }
.th-row { background:#f3f4f6; border:1px solid #e5e7eb; border-bottom:0; padding:10px 12px; border-radius:10px 10px 0 0; font-weight:700; display:grid; }
.tb { border:1px solid #e5e7eb; border-top:0; border-radius:0 0 10px 10px; }
.cell { padding:10px 12px; border-top:1px solid #e5e7eb; }
.chip { display:inline-block; padding:4px 10px; border-radius:999px; color:#fff; font-size:12px; }
.chip-pending { background:#9ca3af; } .chip-ok { background:#16a34a; } .chip-bad { background:#b91c1c; }
.btn-wrap button { background:#f9fafb !important; color:#111827 !important; border:1px solid #e5e7eb !important; border-radius:8px !important; padding:.45rem .9rem !important; font-weight:600 !important; }
.btn-wrap.on-validate button { background:var(--green)!important; color:#fff!important; }
.btn-wrap.on-reject button { background:var(--red)!important; color:#fff!important; }
.accept-btn .stButton>button { background:#b91c1c !important; color:#fff !important; border:none !important; border-radius:10px !important; padding:.7rem 1.3rem !important; font-weight:700 !important; box-shadow:none !important; }
.accept-btn .stButton>button:hover { filter:brightness(.92); }
.centered { display:flex; justify-content:center; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='topbar'><div class='topbar-inner'>"
    f"<div class='who'>Signed in as <b>{st.session_state.get('user','')}</b></div>"
    "</div></div>", unsafe_allow_html=True
)

st.title("AI KPI Extraction & Recommendations (Per BRD)")

# -------- Single uploader with unique key --------
uploads = st.file_uploader("Upload BRDs", type=["pdf","docx","txt"],
                           accept_multiple_files=True, key="brd_uploader")
if st.button("Process BRDs", key="process_btn"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for f in uploads:
            text = read_uploaded(f)
            if USE_OPENAI:
                sub = detect_hr_subdomain_llm(text)
                extracted = extract_kpis_llm(text)
                recs = recommend_llm(extracted["KPI Name"].tolist(), sub, text)
            else:
                sub = detect_hr_subdomain_heuristic(text)
                extracted = extract_kpis_heuristic(text)
                recs = recommend_heuristic(extracted["KPI Name"].tolist(), raw_text=text)

            recommended = pd.DataFrame(recs)
            st.session_state.projects[f.name] = {
                "extracted": extracted, "recommended": recommended, "domain": sub
            }
            st.session_state["final_kpis"].setdefault(
                f.name, pd.DataFrame(columns=["BRD","KPI Name","Source","Description","Owner/ SME","Target Value"])
            )
        st.success(f"✅ Processed {len(uploads)} BRD(s) successfully")

# -------- Render per BRD --------
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

        _, c2, _ = st.columns([1,2,1])
        with c2:
            st.markdown("<div class='centered accept-btn'>", unsafe_allow_html=True)
            if st.button("Review & Accept", key=f"accept_{fname}"):
                st.success("✅ Finalized KPIs have been accepted successfully!")
            st.markdown("</div>", unsafe_allow_html=True)

import io
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument

# ---------- OCR fallback ----------
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

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

    /* FORCE "Review & Accept" button styling */
    .accept-btn .stButton>button {
        background-color: red !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .accept-btn .stButton>button:hover { filter: brightness(0.9); }

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

# ---------- Utils ----------
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

# ---------- Minimal extraction / recommendation ----------
def extract_kpis(text: str) -> pd.DataFrame:
    rows = []
    low = text.lower()
    if "attrition" in low or "retention" in low:
        rows.append({
            "KPI Name":"Voluntary Attrition Rate",
            "Description":"Track voluntary attrition and target a 10% reduction in 12 months.",
            "Target Value":"10% reduction in 12 months",
            "Status":"Pending"
        })
    rows.append({
        "KPI Name":"System Uptime",
        "Description":"Ensure service availability and scalability to minimize downtime.",
        "Target Value":"99.9%",
        "Status":"Pending"
    })
    return pd.DataFrame(rows)

def recommend(domain: str, existing: list, topic: str = None, raw_text: str = "") -> list:
    pool = [
        ("Involuntary Attrition Rate",
         "Measure attrition due to terminations or layoffs; track trend vs. prior quarter."),
        ("Employee Retention Rate",
         "Percentage of employees retained over a period; segment by department and tenure."),
        ("First Year Attrition Rate",
         "Attrition within the first 12 months; highlight onboarding or hiring quality issues.")
    ]
    return [
        {"KPI Name": k, "Description": d, "Owner/ SME": "", "Target Value": "", "Status": "Pending"}
        for k, d in pool if k not in existing
    ]

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
        with c3: target_val = st.text_input("", value=r["Target Value"], key=f"{key_prefix}_t_{i}")
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
    extracted = extract_kpis(text)
    recs = recommend("hr", extracted["KPI Name"].tolist(), raw_text=text)
    recommended = pd.DataFrame(recs)
    st.session_state.projects[file.name] = {
        "extracted": extracted, "recommended": recommended
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

st.title("AI KPI Extraction & Recommendations (Per BRD)")

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

import io, re
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from docx import Document as DocxDocument
from rapidfuzz import fuzz, process

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

# ---------- Streamlit page setup ----------
st.set_page_config(page_title="AI KPI System", layout="wide")

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
STATUS_COLORS = {"Validated": "#16a34a", "Rejected": "#dc2626", "Pending": "#9ca3af"}


def status_chip(s: str) -> str:
    color = STATUS_COLORS.get(s, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px">{s}</span>'


def _check_credentials(email: str, password: str) -> bool:
    return email.strip().lower() in VALID_USERS and VALID_USERS[email.strip().lower()] == password


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


# ---------- KPI extraction stubs ----------
def extract_kpis(text: str) -> pd.DataFrame:
    # Simple demo extraction
    if "attrition" in text.lower():
        rows = [
            {"KPI Name": "Voluntary Attrition Rate", "Description": "10% reduction in voluntary attrition in 12 months.", "Target Value": "in 12 months", "Status": "Pending"},
            {"KPI Name": "System Uptime", "Description": "Ensure system availability and scalability.", "Target Value": "", "Status": "Pending"},
        ]
    else:
        rows = [
            {"KPI Name": "Employee Retention Rate", "Description": "Percentage of employees remaining after 12 months.", "Target Value": ">85%", "Status": "Pending"}
        ]
    return pd.DataFrame(rows)


def recommend(domain: str, existing: list, topic: str = None, raw_text: str = "") -> list:
    pool = ["Involuntary Attrition Rate", "Employee Retention Rate", "First Year Attrition Rate"]
    return [k for k in pool if k not in existing]


# ---------- UI helpers ----------
def render_kpi_table(brd: str, df: pd.DataFrame, key_prefix: str, is_recommended=False):
    if df.empty:
        st.caption("No data available.")
        return df

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:{'1fr ' * (len(df.columns)+1)};
        background:#f3f4f6;padding:8px 12px;font-weight:600;border-radius:6px 6px 0 0;">
        {''.join(f"<div>{col}</div>" for col in df.columns)}<div>Actions</div>
        </div>
        """, unsafe_allow_html=True
    )

    updated = []
    for i, row in df.iterrows():
        cols = st.columns([1 for _ in range(len(df.columns)+1)])
        row_data = {}
        for j, col in enumerate(df.columns):
            val = row[col]
            if col in ["Owner/ SME", "Target Value"]:
                row_data[col] = cols[j].text_input("", value=str(val) if pd.notna(val) else "",
                                                   key=f"{key_prefix}_{i}_{col}")
            elif col == "Status":
                cols[j].markdown(status_chip(val), unsafe_allow_html=True)
                row_data[col] = val
            else:
                cols[j].write(val if val else "—")
                row_data[col] = val

        # Actions: Validate / Reject buttons
        validate = cols[-1].button("Validate", key=f"{key_prefix}_{i}_val")
        reject = cols[-1].button("Reject", key=f"{key_prefix}_{i}_rej")

        if validate:
            row_data["Status"] = "Validated"
            st.session_state["final_kpis"].setdefault(brd, []).append(row_data)
        elif reject:
            row_data["Status"] = "Rejected"
            if brd in st.session_state["final_kpis"]:
                st.session_state["final_kpis"][brd] = [r for r in st.session_state["final_kpis"][brd]
                                                       if r["KPI Name"] != row_data["KPI Name"]]

        updated.append(row_data)

    return pd.DataFrame(updated)


# ---------- Pipeline ----------
def process_file(file):
    text = read_uploaded(file)
    extracted = extract_kpis(text)
    recs = recommend("hr", extracted["KPI Name"].tolist(), raw_text=text)
    recommended = pd.DataFrame(
        [{"KPI Name": r, "Description": f"Auto-generated description for {r}", "Owner/ SME": "", "Target Value": "", "Status": "Pending"} for r in recs]
    )
    st.session_state.projects[file.name] = {
        "extracted": extracted, "recommended": recommended
    }


# ---------- Login ----------
def render_login():
    st.markdown("<h2 style='color:#b91c1c'>AI KPI System</h2>", unsafe_allow_html=True)
    st.subheader("Sign in to continue")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        if _check_credentials(email, password):
            st.session_state["auth"] = True
            st.session_state["user"] = email.strip().lower()
            st.success("Signed in successfully")
            st.rerun()
        else:
            st.error("Invalid email or password")


# ---------- Main App ----------
if not st.session_state["auth"]:
    render_login()
    st.stop()

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.get('user','')}**")
    if st.button("Log out"):
        for k in ["auth", "user", "projects", "final_kpis"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

uploads = st.file_uploader("Upload BRDs", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Process BRDs"):
    if not uploads:
        st.warning("Please upload at least one file")
    else:
        for f in uploads:
            process_file(f)
        count = len(uploads)
        st.success("✅ Processed {} BRD{} successfully".format(count, "" if count == 1 else "s"))

for fname, proj in st.session_state.projects.items():
    st.markdown(f"## 📄 {fname}")
    st.subheader("Extracted KPIs")
    proj["extracted"] = render_kpi_table(fname, proj["extracted"], f"ext_{fname}")

    st.subheader("Recommended KPIs")
    proj["recommended"] = render_kpi_table(fname, proj["recommended"], f"rec_{fname}", is_recommended=True)

    # Manual add
    with st.expander("➕ Add KPI manually"):
        new_kpi = st.text_input(f"New KPI Name ({fname})", key=f"newkpi_{fname}")
        if st.button(f"Add KPI to {fname}", key=f"btn_{fname}"):
            if new_kpi:
                new_row = {"KPI Name": new_kpi, "Description": f"Auto-generated description for {new_kpi}",
                           "Owner/ SME": "", "Target Value": "", "Status": "Pending"}
                proj["recommended"] = pd.concat([proj["recommended"], pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"Added {new_kpi} to recommended KPIs")
                st.rerun()

    # Show final KPIs
    if fname in st.session_state["final_kpis"] and st.session_state["final_kpis"][fname]:
        st.subheader("✅ Finalized KPIs")
        st.table(pd.DataFrame(st.session_state["final_kpis"][fname]))

# ---------- Subdomain-aware KPI library (no System Uptime) ----------
HR_KPI_LIB = {
    # Attrition model BRD
    "hr_attrition_model": [
        ("Model Accuracy", "Classification accuracy of the attrition prediction model (goal ≥85%)."),
        ("Voluntary Attrition Reduction", "Percent reduction in voluntary attrition vs. baseline over 12 months."),
        ("High-Risk Coverage", "Percent of high-risk employees correctly flagged with actionable insights."),
        ("Insight Coverage", "Percent of high-risk cases with identified drivers/insights."),
        ("Dashboard Adoption", "Share of HR users who actively use the risk dashboard each month.")
    ],
    # AI Job Description system BRD
    "hr_jd_system": [
        ("JD Generation Latency", "Median time to generate or redesign a JD."),
        ("Bias Term Reduction", "Percent reduction of gendered or non-inclusive terms in JDs."),
        ("Approval Cycle Time", "Median time from JD draft to final HR approval."),
        ("JD Reuse/Repository Utilization", "Percent of roles using repository templates or redesigned JDs."),
        ("Hiring Manager Adoption", "Share of active hiring managers who use the AI JD tool monthly.")
    ],
    # ATS implementation BRD
    "hr_ats": [
        ("Application Drop-off Rate", "Percent of candidates abandoning during application flow."),
        ("Time-to-Fill", "Median days from requisition open to offer acceptance."),
        ("Automation Rate", "Share of recruiter tasks handled by automated workflows."),
        ("Recruiter Productivity", "Requisitions or candidates handled per recruiter per month."),
        ("Candidate Satisfaction (CSAT)", "Post-application or post-interview satisfaction score.")
    ],
}

# ---------- Subdomain detection ----------
def detect_hr_subdomain(text: str) -> str:
    low = text.lower()
    # explicit signals
    if any(k in low for k in ["attrition", "retention", "predictive model", "risk score"]):
        return "hr_attrition_model"
    if any(k in low for k in ["job description", "jd ", "jd-", "bias detection", "inclusive language"]):
        return "hr_jd_system"
    if any(k in low for k in ["ats", "irecruit", "phenom", "requisition", "candidate experience"]):
        return "hr_ats"
    # fallback to the safest HR subdomain
    return "hr_attrition_model"

# ---------- Extraction tailored to each BRD ----------
def extract_kpis(text: str) -> pd.DataFrame:
    sub = detect_hr_subdomain(text)
    rows = []

    if sub == "hr_attrition_model":
        rows = [
            {"KPI Name": "Model Accuracy", "Description": "Classification accuracy of the attrition model.", "Target Value": "≥ 85%", "Status": "Pending"},
            {"KPI Name": "Voluntary Attrition Reduction", "Description": "Reduction in voluntary attrition vs baseline over 12 months.", "Target Value": "10% in 12 months", "Status": "Pending"},
            {"KPI Name": "Insight Coverage", "Description": "Percent of high-risk cases with identified drivers/insights.", "Target Value": "≥ 80%", "Status": "Pending"},
        ]
        # optionally add dashboard usage if text mentions dashboard
        if "dashboard" in text.lower():
            rows.append({"KPI Name": "Dashboard Adoption", "Description": "Active HR users of the risk dashboard per month.", "Target Value": "", "Status": "Pending"})

    elif sub == "hr_jd_system":
        rows = [
            {"KPI Name": "JD Generation Latency", "Description": "Median time to generate/redesign a JD.", "Target Value": "< 10 seconds", "Status": "Pending"},
            {"KPI Name": "Bias Term Reduction", "Description": "Reduction of gendered/non-inclusive terms in JDs.", "Target Value": "", "Status": "Pending"},
            {"KPI Name": "Approval Cycle Time", "Description": "Draft → manager review → HR approval time.", "Target Value": "", "Status": "Pending"},
        ]

    elif sub == "hr_ats":
        rows = [
            {"KPI Name": "Application Drop-off Rate", "Description": "Percent abandoning during application stages.", "Target Value": "Decrease vs baseline", "Status": "Pending"},
            {"KPI Name": "Time-to-Fill", "Description": "Median days from requisition to offer acceptance.", "Target Value": "Decrease vs baseline", "Status": "Pending"},
            {"KPI Name": "Automation Rate", "Description": "Share of workflow steps automated end-to-end.", "Target Value": "Increase vs baseline", "Status": "Pending"},
        ]

    return pd.DataFrame(rows)

# ---------- Recommendations that complement the extracted set ----------
def recommend(domain: str, existing: list, topic: str = None, raw_text: str = "") -> list:
    sub = detect_hr_subdomain(raw_text)
    pool = HR_KPI_LIB[sub]
    recs = []
    for name, desc in pool:
        if name not in existing:
            recs.append({
                "KPI Name": name,
                "Description": desc,
                "Owner/ SME": "",
                "Target Value": "",
                "Status": "Pending",
            })
    # be concise
    return recs[:5]

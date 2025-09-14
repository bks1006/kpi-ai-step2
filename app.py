def render_extracted_table(brd: str, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        st.caption("No extracted KPIs.")
        return df

    # widen the Actions column so labels don't wrap
    _table_head(
        ["2fr", "3fr", "1fr", "0.9fr", "2.2fr"],
        ["KPI Name", "Description", "Target Value", "Status", "Actions"]
    )

    updated = []
    for i, r in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2, 3, 1, 0.9, 2.2])

        with c1:
            st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)

        with c2:
            st.markdown(f"<div class='cell'>{r['Description']}</div>", unsafe_allow_html=True)

        with c3:
            target_val = st.text_input("", value=r["Target Value"], key=f"{key_prefix}_target_{i}")

        with c4:
            # show the green/red/gray chip
            st.markdown(f"<div class='cell'>{_status_badge(r['Status'])}</div>", unsafe_allow_html=True)

        with c5:
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                st.markdown("<div class='cell'><button disabled class='btn btn-ghost'>Review Details</button></div>", unsafe_allow_html=True)
            with colB:
                # disable Validate if already validated/rejected
                disabled = r["Status"] in ("Validated", "Rejected")
                if st.button("Validate", key=f"{key_prefix}_ok_{i}", disabled=disabled):
                    r["Status"] = "Validated"
                    _upsert_final(brd, {
                        "BRD": brd,
                        "KPI Name": r["KPI Name"],
                        "Source": "Extracted",
                        "Description": r["Description"],
                        "Owner/ SME": "",
                        "Target Value": target_val
                    })
            with colC:
                disabled = r["Status"] in ("Validated", "Rejected")
                if st.button("Reject", key=f"{key_prefix}_rej_{i}", disabled=disabled):
                    r["Status"] = "Rejected"

        updated.append({
            "KPI Name": r["KPI Name"],
            "Description": r["Description"],
            "Target Value": target_val,
            "Status": r["Status"]
        })

    _table_tail()
    return pd.DataFrame(updated, columns=list(df.columns))


def render_recommended_table(brd: str, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        st.caption("No recommendations.")
        return df

    # widen the Actions column so labels don't wrap
    _table_head(
        ["2fr", "1fr", "1fr", "0.8fr", "2.2fr"],
        ["KPI Name", "Owner/ SME", "Target Value", "Status", "Actions"]
    )

    updated = []
    for i, r in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 0.8, 2.2])

        with c1:
            st.markdown(f"<div class='cell'><b>{r['KPI Name']}</b></div>", unsafe_allow_html=True)

        with c2:
            owner_val = st.text_input("", value=r["Owner/ SME"], key=f"{key_prefix}_owner_{i}")

        with c3:
            target_val = st.text_input("", value=r["Target Value"], key=f"{key_prefix}_target_{i}")

        with c4:
            st.markdown(f"<div class='cell'>{_status_badge(r['Status'])}</div>", unsafe_allow_html=True)

        with c5:
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                st.markdown("<div class='cell'><button disabled class='btn btn-ghost'>Review Details</button></div>", unsafe_allow_html=True)
            with colB:
                disabled = r["Status"] in ("Validated", "Rejected")
                if st.button("Validate", key=f"{key_prefix}_ok_{i}", disabled=disabled):
                    r["Status"] = "Validated"
                    _upsert_final(brd, {
                        "BRD": brd,
                        "KPI Name": r["KPI Name"],
                        "Source": "Recommended",
                        "Description": "",
                        "Owner/ SME": owner_val,
                        "Target Value": target_val
                    })
            with colC:
                disabled = r["Status"] in ("Validated", "Rejected")
                if st.button("Reject", key=f"{key_prefix}_rej_{i}", disabled=disabled):
                    r["Status"] = "Rejected"

        updated.append({
            "KPI Name": r["KPI Name"],
            "Owner/ SME": owner_val,
            "Target Value": target_val,
            "Status": r["Status"]
        })

    _table_tail()
    return pd.DataFrame(updated, columns=list(df.columns))

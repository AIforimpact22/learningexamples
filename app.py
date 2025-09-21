import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
import hashlib, hmac
from datetime import date
from typing import Optional

st.set_page_config(page_title="Neon + Streamlit • Employees & Attendance", page_icon="🗄️", layout="wide")

# =========================
# Simple Access Gate
# =========================
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _get_query_param(name: str) -> Optional[str]:
    try:
        qp = st.query_params  # newer Streamlit
        val = qp.get(name, None)
        if isinstance(val, list):
            return val[0] if val else None
        return val
    except Exception:
        val = st.experimental_get_query_params().get(name, [None])
        return val[0] if isinstance(val, list) else val

def require_login() -> bool:
    if st.session_state.get("_authed"):
        with st.sidebar:
            st.success(f"Signed in ({st.session_state.get('_method')})")
            if st.button("Sign out", use_container_width=True):
                st.session_state.clear()
                try: st.rerun()
                except Exception: st.experimental_rerun()
        return True

    secrets_auth = st.secrets.get("auth", {})
    # Token in URL
    token = _get_query_param("token")
    allowed_tokens = set()
    if "tokens" in secrets_auth:
        allowed_tokens = {str(x) for x in secrets_auth["tokens"]}
    elif "token" in secrets_auth:
        allowed_tokens = {str(secrets_auth["token"])}

    if token and token in allowed_tokens:
        st.session_state["_authed"] = True
        st.session_state["_method"] = "token"
        try: st.rerun()
        except Exception: st.experimental_rerun()

    # Password in sidebar
    with st.sidebar:
        st.markdown("### Sign in")
        pw = st.text_input("Access password", type="password")
        if st.button("Sign in", use_container_width=True):
            ok = False
            if "password_sha256" in secrets_auth:
                ok = hmac.compare_digest(_sha256(pw), secrets_auth["password_sha256"])
            elif "password" in secrets_auth:
                ok = hmac.compare_digest(pw, secrets_auth["password"])
            if ok:
                st.session_state["_authed"] = True
                st.session_state["_method"] = "password"
                st.toast("Signed in")
                try: st.rerun()
                except Exception: st.experimental_rerun()
            else:
                st.error("Invalid password")
    st.info("Enter the access password to continue.")
    return False

if not require_login():
    st.stop()

# =========================
# Connections (pooled + optional admin)
# =========================
# Use pool_pre_ping to avoid stale pooled connections
conn = st.connection("neon", type="sql", engine_kwargs={"pool_pre_ping": True})

def _has_attendance_table() -> bool:
    try:
        df = conn.query(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='app' AND table_name='attendance_log' LIMIT 1",
            ttl=0
        )
        return not df.empty
    except OperationalError as e:
        st.error(f"Metadata check failed: {e}")
        return False
    except Exception as e:
        st.error(f"Metadata check error: {e}")
        return False

def _bootstrap_via_admin():
    # Try to open a direct (non-pooled) admin connection if provided
    try:
        conn_admin = st.connection("neon_admin", type="sql", engine_kwargs={"pool_pre_ping": True})
    except Exception:
        conn_admin = None

    if conn_admin is None:
        st.warning(
            "Table app.attendance_log is missing and no direct admin connection is configured.\n"
            "Open Neon SQL Editor and run the provided DDL to create it."
        )
        return False

    try:
        with conn_admin.session as s:
            s.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
            s.execute(text(\"\"\"
                CREATE TABLE IF NOT EXISTS app.attendance_log (
                    log_id BIGSERIAL PRIMARY KEY,
                    employee_id TEXT NOT NULL REFERENCES app.employees(employee_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
                    date DATE NOT NULL,
                    check_in_time TIMESTAMPTZ,
                    check_out_time TIMESTAMPTZ,
                    status TEXT,
                    notes TEXT,
                    CONSTRAINT attendance_unique_per_day UNIQUE (employee_id, date)
                )
            \"\"\"))
            s.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_attendance_employee_date "
                "ON app.attendance_log (employee_id, date)"
            ))
            s.commit()
        st.success("Initialized app.attendance_log via admin connection.")
        return True
    except OperationalError as e:
        st.error(f"Admin DDL failed (connection issue): {e}")
        return False
    except Exception as e:
        st.error(f"Admin DDL error: {e}")
        return False

# Ensure table exists WITHOUT running DDL over pooled connection
if not _has_attendance_table():
    if not _bootstrap_via_admin():
        st.stop()

st.title("Employees & Attendance")
st.caption("Neon (Postgres) backend • Streamlit frontend")

# =========================
# Helpers
# =========================
def _list_employees(q: str = ""):
    try:
        if q:
            return conn.query(\"\"\"
                SELECT employee_id, first_name, last_name, email, department, job_title, status, hire_date, created_at
                FROM app.employees
                WHERE employee_id ILIKE :q
                   OR first_name ILIKE :q
                   OR last_name ILIKE :q
                   OR email ILIKE :q
                ORDER BY created_at DESC
            \"\"\", params={"q": f"%{q}%"}, ttl="20s")
        return conn.query(\"\"\"
            SELECT employee_id, first_name, last_name, email, department, job_title, status, hire_date, created_at
            FROM app.employees
            ORDER BY created_at DESC
        \"\"\", ttl="20s")
    except OperationalError as e:
        st.error(f"Employee list failed: {e}")
        return conn.query("SELECT NULL WHERE FALSE", ttl=0)  # empty DF

def _get_employee(eid: str):
    try:
        df = conn.query(\"\"\"
            SELECT employee_id, first_name, last_name, email, department, job_title, status, hire_date
            FROM app.employees WHERE employee_id = :eid
        \"\"\", params={"eid": eid}, ttl=0)
        return None if df.empty else df.iloc[0].to_dict()
    except OperationalError as e:
        st.error(f"Load employee failed: {e}")
        return None

def _employee_has_attendance(eid: str) -> int:
    try:
        df = conn.query(
            "SELECT COUNT(*) AS c FROM app.attendance_log WHERE employee_id = :eid",
            params={"eid": eid}, ttl=0
        )
        return int(df.iloc[0]["c"])
    except Exception:
        return 0

def _list_attendance(eid: Optional[str], d_from: Optional[date], d_to: Optional[date], status: str):
    clauses, p = [], {}
    if eid:
        clauses.append("al.employee_id = :eid"); p["eid"] = eid
    if d_from:
        clauses.append("al.date >= :dfrom"); p["dfrom"] = d_from.isoformat()
    if d_to:
        clauses.append("al.date <= :dto"); p["dto"] = d_to.isoformat()
    if status and status != "Any":
        clauses.append("COALESCE(al.status,'') = :st"); p["st"] = status
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f\"\"\"
      SELECT
        al.log_id, al.date, al.employee_id,
        e.first_name, e.last_name,
        al.check_in_time, al.check_out_time, al.status, al.notes
      FROM app.attendance_log al
      JOIN app.employees e ON e.employee_id = al.employee_id
      {where}
      ORDER BY al.date DESC, e.first_name, e.last_name
      LIMIT 500
    \"\"\"
    try:
        return conn.query(sql, params=p, ttl="15s")
    except OperationalError as e:
        st.error(f"Attendance query failed: {e}")
        return conn.query("SELECT NULL WHERE FALSE", ttl=0)

# =========================
# UI Tabs
# =========================
tab_emp, tab_att, tab_health = st.tabs(["👤 Employees", "🕒 Attendance", "🩺 Health"])

# =========================
# EMPLOYEES (Full CRUD)
# =========================
with tab_emp:
    st.subheader("Create employee")
    with st.form("emp_create"):
        c1, c2, c3 = st.columns(3)
        employee_id = c1.text_input("Employee ID *", placeholder="E1003")
        first_name = c2.text_input("First name *")
        last_name  = c3.text_input("Last name *")
        email      = c1.text_input("Email")
        department = c2.text_input("Department")
        job_title  = c3.text_input("Job title")
        status     = c1.selectbox("Status *", ["active", "inactive"], index=0)
        hire_date_ = c2.date_input("Hire date *", value=date.today(), format="YYYY-MM-DD")
        submitted = st.form_submit_button("Create employee", use_container_width=True)
        if submitted:
            if not (employee_id and first_name and last_name and hire_date_ and status):
                st.error("Employee ID, First name, Last name, Hire date and Status are required.")
            else:
                try:
                    with conn.session as s:
                        s.execute(text(\"\"\"
                            INSERT INTO app.employees
                              (employee_id, first_name, last_name, email, department, job_title, status, hire_date)
                            VALUES
                              (:eid, :fn, :ln, :em, :dp, :jt, :st, :hd)
                        \"\"\"), dict(eid=employee_id, fn=first_name, ln=last_name,
                                     em=email or None, dp=department or None,
                                     jt=job_title or None, st=status, hd=hire_date_.isoformat()))
                        s.commit()
                    st.success(f"Employee {employee_id} created.")
                except OperationalError as e:
                    st.error(f"Create failed (connection): {e}")
                except Exception as e:
                    st.error(f"Create failed: {e}")

    st.divider()

    st.subheader("Read / Search")
    q = st.text_input("Search ID / name / email", placeholder="e.g., E1001 or mary or [email protected]")
    emp_list = _list_employees(q.strip())
    st.dataframe(emp_list, use_container_width=True, height=320)

    st.divider()

    st.subheader("Update / Delete")
    mini = conn.query("SELECT employee_id, first_name, last_name FROM app.employees ORDER BY first_name, last_name", ttl="30s")
    if mini.empty:
        st.info("No employees yet.")
    else:
        options = mini.apply(lambda r: f"{r['first_name']} {r['last_name']} ({r['employee_id']})", axis=1).tolist()
        choice = st.selectbox("Pick an employee to edit", options)
        eid = mini.iloc[options.index(choice)]["employee_id"]
        row = _get_employee(eid)
        if not row:
            st.warning("Could not load employee details.")
        else:
            with st.form("emp_update_delete", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                c1.text_input("Employee ID", value=row["employee_id"], disabled=True)
                fn = c2.text_input("First name *", value=row["first_name"])
                ln = c3.text_input("Last name *", value=row["last_name"])
                em = c1.text_input("Email", value=row.get("email") or "")
                dp = c2.text_input("Department", value=row.get("department") or "")
                jt = c3.text_input("Job title", value=row.get("job_title") or "")
                stt= c1.selectbox("Status *", ["active", "inactive"], index=0 if row["status"]=="active" else 1)
                hd = c2.date_input("Hire date *", value=row.get("hire_date"), format="YYYY-MM-DD")

                ucol1, ucol2, ucol3 = st.columns([1,1,1])
                do_update = ucol1.form_submit_button("Save changes", use_container_width=True)
                with_att = ucol2.checkbox("Also delete attendance logs", value=False, help="Needed if the employee has attendance rows.")
                do_delete = ucol3.form_submit_button("Delete employee", use_container_width=True)

                if do_update:
                    if not (fn and ln and stt and hd):
                        st.error("First name, Last name, Hire date, and Status are required.")
                    else:
                        try:
                            with conn.session as s:
                                s.execute(text(\"\"\"
                                    UPDATE app.employees
                                       SET first_name=:fn, last_name=:ln, email=:em,
                                           department=:dp, job_title=:jt, status=:st, hire_date=:hd
                                     WHERE employee_id=:eid
                                \"\"\"), dict(fn=fn, ln=ln, em=em or None, dp=dp or None, jt=jt or None,
                                             st=stt, hd=hd.isoformat(), eid=eid))
                                s.commit()
                            st.success("Employee updated.")
                        except OperationalError as e:
                            st.error(f"Update failed (connection): {e}")
                        except Exception as e:
                            st.error(f"Update failed: {e}")

                if do_delete:
                    try:
                        cnt = _employee_has_attendance(eid)
                        with conn.session as s:
                            if cnt and not with_att:
                                st.error(f"Employee has {cnt} attendance row(s). Tick 'Also delete attendance logs' to proceed.")
                            else:
                                if cnt and with_att:
                                    s.execute(text("DELETE FROM app.attendance_log WHERE employee_id=:eid"), {"eid": eid})
                                s.execute(text("DELETE FROM app.employees WHERE employee_id=:eid"), {"eid": eid})
                                s.commit()
                        if (cnt == 0) or with_att:
                            st.success("Employee deleted.")
                            try: st.rerun()
                            except Exception: st.experimental_rerun()
                    except OperationalError as e:
                        st.error(f"Delete failed (connection): {e}")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

# =========================
# ATTENDANCE (Full CRUD)
# =========================
with tab_att:
    st.subheader("Quick actions (today)")
    mini = conn.query("SELECT employee_id, first_name, last_name FROM app.employees ORDER BY first_name, last_name", ttl="30s")
    if mini.empty:
        st.info("No employees found. Create one in the Employees tab.")
    else:
        options = mini.apply(lambda r: f"{r['first_name']} {r['last_name']} ({r['employee_id']})", axis=1).tolist()
        choice_q = st.selectbox("Select employee", options, key="att_emp_sel")
        eid_q = mini.iloc[options.index(choice_q)]["employee_id"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Check In (now)"):
                try:
                    with conn.session as s:
                        s.execute(text(\"\"\"
                            INSERT INTO app.attendance_log (employee_id, date, check_in_time, status)
                            VALUES (:eid, CURRENT_DATE, now(), 'Present')
                            ON CONFLICT (employee_id, date)
                            DO UPDATE SET check_in_time = COALESCE(app.attendance_log.check_in_time, EXCLUDED.check_in_time)
                        \"\"\"), {"eid": eid_q})
                        s.commit()
                    st.success("Checked in.")
                except OperationalError as e:
                    st.error(f"Check-in failed (connection): {e}")
                except Exception as e:
                    st.error(f"Check-in failed: {e}")
        with c2:
            if st.button("Check Out (now)"):
                try:
                    with conn.session as s:
                        res = s.execute(text(\"\"\"
                            UPDATE app.attendance_log
                               SET check_out_time = now()
                             WHERE employee_id=:eid AND date=CURRENT_DATE AND check_out_time IS NULL
                        \"\"\"), {"eid": eid_q})
                        s.commit()
                    st.success("Checked out.") if res.rowcount else st.warning("No open check-in for today.")
                except OperationalError as e:
                    st.error(f"Check-out failed (connection): {e}")
                except Exception as e:
                    st.error(f"Check-out failed: {e}")
        with c3:
            if st.button("Clear Check-out (today)"):
                try:
                    with conn.session as s:
                        res = s.execute(text(
                            "UPDATE app.attendance_log SET check_out_time = NULL "
                            "WHERE employee_id=:eid AND date=CURRENT_DATE"
                        ), {"eid": eid_q})
                        s.commit()
                    st.success("Cleared.") if res.rowcount else st.info("No row to clear.")
                except OperationalError as e:
                    st.error(f"Clear failed (connection): {e}")
                except Exception as e:
                    st.error(f"Clear failed: {e}")
        with c4:
            new_status = st.text_input("Status/Notes for today", key="today_status_note")
            if st.button("Set Status/Notes (today)"):
                try:
                    with conn.session as s:
                        s.execute(text(
                            "INSERT INTO app.attendance_log (employee_id, date) "
                            "VALUES (:eid, CURRENT_DATE) "
                            "ON CONFLICT (employee_id, date) DO NOTHING"
                        ), {"eid": eid_q})
                        s.execute(text(
                            "UPDATE app.attendance_log SET status=:st, notes=:nt "
                            "WHERE employee_id=:eid AND date=CURRENT_DATE"
                        ), {"st": new_status or None, "nt": new_status or None, "eid": eid_q})
                        s.commit()
                    st.success("Status/notes updated.")
                except OperationalError as e:
                    st.error(f"Update failed (connection): {e}")
                except Exception as e:
                    st.error(f"Update failed: {e}")

    st.divider()
    st.subheader("Create attendance (any date)")

    with st.form("att_create"):
        col1, col2, col3 = st.columns(3)
        if mini.empty:
            eid_new = col1.text_input("Employee ID *", placeholder="E1001")
        else:
            choice_c = col1.selectbox("Employee *", options if not mini.empty else [], index=0 if not mini.empty else None)
            eid_new = mini.iloc[options.index(choice_c)]["employee_id"] if not mini.empty else None

        date_new = col2.date_input("Date *", value=date.today(), format="YYYY-MM-DD")
        status_new = col3.text_input("Status (optional)", placeholder="Present / Sick / Leave ...")
        notes_new = st.text_input("Notes (optional)")
        submitted_att = st.form_submit_button("Create/Upsert attendance")

        if submitted_att:
            try:
                with conn.session as s:
                    s.execute(text(\"\"\"
                        INSERT INTO app.attendance_log (employee_id, date, status, notes)
                        VALUES (:eid, :dt, :st, :nt)
                        ON CONFLICT (employee_id, date)
                        DO UPDATE SET status = EXCLUDED.status, notes = EXCLUDED.notes
                    \"\"\"), {"eid": eid_new, "dt": date_new.isoformat(),
                             "st": status_new or None, "nt": notes_new or None})
                    s.commit()
                st.success("Attendance created/updated.")
            except OperationalError as e:
                st.error(f"Create failed (connection): {e}")
            except Exception as e:
                st.error(f"Create failed: {e}")

    st.divider()
    st.subheader("Read / Update / Delete")

    colf1, colf2, colf3, colf4 = st.columns([1,1,1,1])
    eid_filter = None
    if not mini.empty:
        ch = colf1.selectbox("Employee filter", ["Any"] + options, index=0)
        eid_filter = None if ch == "Any" else mini.iloc[options.index(ch)]["employee_id"]
    dfrom = colf2.date_input("From", value=None, format="YYYY-MM-DD")
    dto   = colf3.date_input("To", value=None, format="YYYY-MM-DD")
    status_filter = colf4.selectbox("Status", ["Any","Present","Sick","Leave","Other"], index=0)

    att_df = _list_attendance(eid_filter, dfrom if dfrom != date.min else None, dto if dto != date.min else None, status_filter)
    st.dataframe(att_df, use_container_width=True, height=360)

    if not att_df.empty:
        with st.expander("Edit or delete a specific row"):
            labels = [f"#{r.log_id} • {r.date} • {r.first_name} {r.last_name} ({r.employee_id})" for r in att_df.itertuples()]
            sel = st.selectbox("Pick a row", labels)
            idx = labels.index(sel)
            row = att_df.iloc[idx]
            st.write(f"Selected: log_id={row['log_id']}")

            cc1, cc2, cc3, cc4 = st.columns(4)
            if cc1.button("Set check-in = now()", key="btn_ci"):
                try:
                    with conn.session as s:
                        s.execute(text("UPDATE app.attendance_log SET check_in_time = now() WHERE log_id = :id"),
                                  {"id": int(row["log_id"])})
                        s.commit()
                    st.success("Updated check-in.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                except OperationalError as e:
                    st.error(f"Update failed (connection): {e}")
                except Exception as e:
                    st.error(f"Update failed: {e}")

            if cc2.button("Set check-out = now()", key="btn_co"):
                try:
                    with conn.session as s:
                        s.execute(text("UPDATE app.attendance_log SET check_out_time = now() WHERE log_id = :id"),
                                  {"id": int(row["log_id"])})
                        s.commit()
                    st.success("Updated check-out.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                except OperationalError as e:
                    st.error(f"Update failed (connection): {e}")
                except Exception as e:
                    st.error(f"Update failed: {e}")

            new_status = cc3.text_input("Status", value=row.get("status") or "", key="row_status")
            new_notes  = cc4.text_input("Notes",  value=row.get("notes") or "", key="row_notes")
            uu1, uu2 = st.columns([1,1])
            if uu1.button("Save status/notes"):
                try:
                    with conn.session as s:
                        s.execute(text(
                            "UPDATE app.attendance_log SET status=:st, notes=:nt WHERE log_id=:id"
                        ), {"st": new_status or None, "nt": new_notes or None, "id": int(row["log_id"])})
                        s.commit()
                    st.success("Row updated.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                except OperationalError as e:
                    st.error(f"Update failed (connection): {e}")
                except Exception as e:
                    st.error(f"Update failed: {e}")

            if uu2.button("Delete this row", type="primary"):
                try:
                    with conn.session as s:
                        s.execute(text("DELETE FROM app.attendance_log WHERE log_id=:id"), {"id": int(row["log_id"])})
                        s.commit()
                    st.success("Row deleted.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                except OperationalError as e:
                    st.error(f"Delete failed (connection): {e}")
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# =========================
# Health
# =========================
with tab_health:
    st.subheader("Health check")
    try:
        df = conn.query("SELECT now() AS server_time", ttl=0)
        st.write(df.iloc[0]["server_time"])
    except OperationalError as e:
        st.error(f"DB error (connection): {e}")
    except Exception as e:
        st.error(f"DB error: {e}")

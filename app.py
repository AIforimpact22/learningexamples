import streamlit as st
from sqlalchemy import text
import hashlib, hmac
from datetime import date

# -------------------- App & Security --------------------
st.set_page_config(page_title="Neon + Streamlit • Employees & Attendance", page_icon="🗄️", layout="centered")

def _sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _get_query_param(name: str):
    try:
        qp = st.query_params
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
            st.success(f"Signed in ({st.session_state.get('_method','')})")
            if st.button("Sign out", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        return True

    secrets_auth = st.secrets.get("auth", {})
    # URL token (magic link): ?token=XXXX
    token = _get_query_param("token")
    allowed_tokens = set()
    if "tokens" in secrets_auth:
        allowed_tokens = {str(x) for x in secrets_auth["tokens"]}
    elif "token" in secrets_auth:
        allowed_tokens = {str(secrets_auth["token"])}

    if token and token in allowed_tokens:
        st.session_state["_authed"] = True
        st.session_state["_method"] = "token"
        st.rerun()

    with st.sidebar:
        st.markdown("### Sign in")
        pw_input = st.text_input("Access password", type="password")
        if st.button("Sign in", use_container_width=True):
            ok = False
            if "password_sha256" in secrets_auth:
                ok = hmac.compare_digest(_sha256(pw_input), secrets_auth["password_sha256"])
            elif "password" in secrets_auth:
                ok = hmac.compare_digest(pw_input, secrets_auth["password"])
            if ok:
                st.session_state["_authed"] = True
                st.session_state["_method"] = "password"
                st.toast("Signed in")
                st.rerun()
            else:
                st.error("Invalid password")
    st.info("Enter the access password to continue.")
    return False

if not require_login():
    st.stop()

st.title("Employees + Attendance")
st.caption("Neon (Postgres) backend • Streamlit frontend")

# -------------------- DB Connection --------------------
# Requires: .streamlit/secrets.toml with [connections.neon].url
conn = st.connection("neon", type="sql")

# Ensure ONLY the attendance table exists (we assume app.employees already exists)
with conn.session as s:
    s.execute(text("""
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
    """))
    s.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_attendance_employee_date
        ON app.attendance_log (employee_id, date)
    """))
    s.commit()

# -------------------- Page Navigation --------------------
with st.sidebar:
    page = st.radio("Pages", ["Attendance", "Employees (CRUD)"], index=0)

# -------------------- Employees: helpers --------------------
def list_employees(search: str | None):
    q = """
        SELECT employee_id, first_name, last_name, email, department, job_title,
               status, hire_date, created_at
        FROM app.employees
        {where}
        ORDER BY created_at DESC
        LIMIT 500
    """
    if search:
        like = f"%{search.strip()}%"
        return conn.query(
            q.format(where="WHERE employee_id ILIKE :q OR first_name ILIKE :q OR last_name ILIKE :q OR email ILIKE :q"),
            params={"q": like}, ttl="10s"
        )
    else:
        return conn.query(q.format(where=""), ttl="10s")

def get_employee(eid: str):
    df = conn.query("""
        SELECT employee_id, first_name, last_name, email, department, job_title,
               status, hire_date
        FROM app.employees
        WHERE employee_id = :eid
        LIMIT 1
    """, params={"eid": eid}, ttl=0)
    return None if df.empty else df.iloc[0].to_dict()

def insert_employee(payload: dict):
    with conn.session as s:
        s.execute(text("""
            INSERT INTO app.employees (employee_id, first_name, last_name, email,
                                       department, job_title, status, hire_date)
            VALUES (:employee_id, :first_name, :last_name, :email,
                    :department, :job_title, :status, :hire_date)
        """), payload)
        s.commit()

def update_employee(eid: str, payload: dict):
    with conn.session as s:
        s.execute(text("""
            UPDATE app.employees
               SET first_name = :first_name,
                   last_name  = :last_name,
                   email      = :email,
                   department = :department,
                   job_title  = :job_title,
                   status     = :status,
                   hire_date  = :hire_date
             WHERE employee_id = :employee_id
        """), {**payload, "employee_id": eid})
        s.commit()

def delete_employee(eid: str):
    with conn.session as s:
        s.execute(text("DELETE FROM app.employees WHERE employee_id = :eid"), {"eid": eid})
        s.commit()

# -------------------- Attendance Page --------------------
if page == "Attendance":
    # Load employees to drive attendance UI
    employees_df = conn.query("""
        SELECT employee_id, first_name, last_name
        FROM app.employees
        ORDER BY first_name, last_name
    """, ttl="2m")

    st.subheader("Attendance (Today)")
    if employees_df.empty:
        st.info("No employees found. Add employees first, then return to this page.")
    else:
        options = employees_df.apply(
            lambda r: f"{r['first_name']} {r['last_name']} ({r['employee_id']})", axis=1
        ).tolist()
        choice = st.selectbox("Select employee", options)
        eid = employees_df.iloc[options.index(choice)]["employee_id"]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("Check In", use_container_width=True):
                try:
                    with conn.session as s:
                        s.execute(text("""
                            INSERT INTO app.attendance_log (employee_id, date, check_in_time, status)
                            VALUES (:eid, CURRENT_DATE, now(), 'Present')
                            ON CONFLICT (employee_id, date)
                            DO UPDATE SET check_in_time = COALESCE(app.attendance_log.check_in_time, EXCLUDED.check_in_time)
                        """), {"eid": eid})
                        s.commit()
                    st.success(f"{eid} checked in.")
                except Exception as e:
                    st.error(f"Check-in failed: {e}")

        with c2:
            if st.button("Check Out", use_container_width=True):
                try:
                    with conn.session as s:
                        res = s.execute(text("""
                            UPDATE app.attendance_log
                               SET check_out_time = now()
                             WHERE employee_id = :eid
                               AND date = CURRENT_DATE
                               AND check_out_time IS NULL
                        """), {"eid": eid})
                        s.commit()
                    st.success(f"{eid} checked out.") if res.rowcount else st.warning("No open check-in for today.")
                except Exception as e:
                    st.error(f"Check-out failed: {e}")

        with c3:
            status_note = st.text_input("Optional status/notes (e.g., Sick, Leave)")
            if st.button("Set Status/Note", use_container_width=True):
                try:
                    with conn.session as s:
                        s.execute(text("""
                            INSERT INTO app.attendance_log (employee_id, date)
                            VALUES (:eid, CURRENT_DATE)
                            ON CONFLICT (employee_id, date) DO NOTHING
                        """), {"eid": eid})
                        s.execute(text("""
                            UPDATE app.attendance_log
                               SET status = :st, notes = :nt
                             WHERE employee_id = :eid AND date = CURRENT_DATE
                        """), {"st": status_note or None, "nt": status_note or None, "eid": eid})
                        s.commit()
                    st.success("Status/notes updated.")
                except Exception as e:
                    st.error(f"Update failed: {e}")

        with c4:
            if st.button("Delete today’s row", use_container_width=True):
                try:
                    with conn.session as s:
                        res = s.execute(text("""
                            DELETE FROM app.attendance_log
                             WHERE employee_id = :eid AND date = CURRENT_DATE
                        """), {"eid": eid})
                        s.commit()
                    st.success("Deleted.") if res.rowcount else st.info("No row to delete.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    st.subheader("Today’s attendance")
    today_df = conn.query("""
        SELECT
          al.date,
          e.employee_id,
          e.first_name,
          e.last_name,
          al.check_in_time,
          al.check_out_time,
          al.status,
          al.notes
        FROM app.attendance_log al
        JOIN app.employees e ON e.employee_id = al.employee_id
        WHERE al.date = CURRENT_DATE
        ORDER BY e.first_name, e.last_name
    """, ttl="30s")
    st.dataframe(today_df, use_container_width=True)

    st.subheader("Recent attendance (last 30 days)")
    recent_df = conn.query("""
        SELECT
          al.date,
          e.employee_id,
          e.first_name,
          e.last_name,
          al.check_in_time,
          al.check_out_time,
          al.status
        FROM app.attendance_log al
        JOIN app.employees e ON e.employee_id = al.employee_id
        WHERE al.date >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY al.date DESC, e.first_name, e.last_name
    """, ttl="2m")
    st.dataframe(recent_df, use_container_width=True)

    st.caption("DB server time:")
    st.write(conn.query("SELECT now() AS server_time;").iloc[0]["server_time"])

# -------------------- Employees (CRUD) Page --------------------
else:
    st.subheader("Employees (CRUD)")

    # -------- Create --------
    with st.expander("➕ Create new employee", expanded=False):
        with st.form("create_emp"):
            col1, col2 = st.columns(2)
            employee_id = col1.text_input("Employee ID", placeholder="E1003")
            first_name  = col1.text_input("First name")
            last_name   = col2.text_input("Last name")
            email       = col2.text_input("Email (optional)")
            department  = col1.text_input("Department (optional)")
            job_title   = col2.text_input("Job title (optional)")
            status      = st.selectbox("Status", ["active", "inactive"], index=0)
            hire_dt     = st.date_input("Hire date", value=date.today())
            submit_new  = st.form_submit_button("Create employee", use_container_width=True)

            if submit_new:
                if not (employee_id and first_name and last_name):
                    st.error("Employee ID, First name, and Last name are required.")
                else:
                    try:
                        payload = dict(
                            employee_id=employee_id,
                            first_name=first_name,
                            last_name=last_name,
                            email=email or None,
                            department=department or None,
                            job_title=job_title or None,
                            status=status,
                            hire_date=hire_dt.isoformat()
                        )
                        insert_employee(payload)
                        st.success(f"Created {employee_id}")
                    except Exception as e:
                        st.error(f"Create failed: {e}")

    # -------- Read (search & list) --------
    st.markdown("#### Search employees")
    q = st.text_input("Search by ID, name, or email", placeholder="e.g., E1001 or Mary or [email protected]")
    emp_list_df = list_employees(q)
    st.dataframe(emp_list_df, use_container_width=True, height=300)

    # -------- Update/Delete --------
    st.markdown("#### Edit / Delete")
    if emp_list_df.empty:
        st.info("No employees to edit. Adjust search or create a new employee above.")
    else:
        # Build options from the current list (id → display)
        opts = [f"{r.employee_id} — {r.first_name} {r.last_name}" for r in emp_list_df.itertuples(index=False)]
        select = st.selectbox("Select employee to edit", opts)
        selected_id = select.split(" — ", 1)[0]

        # Load current record
        rec = get_employee(selected_id)
        if not rec:
            st.warning("Selected employee not found. Refresh list.")
        else:
            with st.form("edit_emp"):
                c1, c2 = st.columns(2)
                first_name = c1.text_input("First name", value=rec["first_name"])
                last_name  = c2.text_input("Last name",  value=rec["last_name"])
                email      = c2.text_input("Email (optional)", value=rec.get("email") or "")
                department = c1.text_input("Department (optional)", value=rec.get("department") or "")
                job_title  = c2.text_input("Job title (optional)", value=rec.get("job_title") or "")
                status     = st.selectbox("Status", ["active", "inactive"], index=0 if rec["status"]=="active" else 1)
                hire_dt    = st.date_input("Hire date", value=rec.get("hire_date") or date.today())
                colu, cold = st.columns(2)
                update_btn = colu.form_submit_button("💾 Update", use_container_width=True)
                delete_btn = cold.form_submit_button("🗑️ Delete", use_container_width=True)

                if update_btn:
                    try:
                        payload = dict(
                            first_name=first_name,
                            last_name=last_name,
                            email=email or None,
                            department=department or None,
                            job_title=job_title or None,
                            status=status,
                            hire_date=hire_dt.isoformat()
                        )
                        update_employee(selected_id, payload)
                        st.success("Updated successfully.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")

                if delete_btn:
                    # simple confirmation guard
                    st.warning("You are about to delete this employee. This will fail if attendance rows exist.", icon="⚠️")
                    if st.checkbox("Yes, delete this employee permanently"):
                        try:
                            delete_employee(selected_id)
                            st.success("Deleted.")
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

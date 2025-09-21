import streamlit as st
from sqlalchemy import text

st.set_page_config(page_title="Neon + Streamlit • Attendance", page_icon="🗄️", layout="centered")
st.title("Employees + Attendance")
st.caption("Neon (Postgres) backend • Streamlit frontend")

# 1) Connect via secrets (you already set [connections.neon].url in secrets.toml)
conn = st.connection("neon", type="sql")

# 2) Bootstrap ONLY the new table (safe to run; does not touch app.employees)
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

# 3) Employees list (for UI selection) — reads existing table, no DDL
employees_df = conn.query("""
    SELECT employee_id, first_name, last_name
    FROM app.employees
    ORDER BY first_name, last_name
""", ttl="2m")

st.subheader("Attendance (Today)")
if employees_df.empty:
    st.info("No employees found. Add employees first, then return to this page.")
else:
    # Choose an employee
    options = employees_df.apply(
        lambda r: f"{r['first_name']} {r['last_name']} ({r['employee_id']})", axis=1
    ).tolist()
    choice = st.selectbox("Select employee", options)
    eid = employees_df.iloc[options.index(choice)]["employee_id"]

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Check In"):
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
        if st.button("Check Out"):
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
        if st.button("Set Status/Note"):
            try:
                with conn.session as s:
                    # Ensure row exists, then update status/notes
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

# 4) Views
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

# 5) Health check
st.caption("DB server time:")
st.write(conn.query("SELECT now() AS server_time;").iloc[0]["server_time"])
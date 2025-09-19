# app_weather_dashboard.py
# Streamlit weather dashboard with an Upload page (CSV-driven, app-like UI)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Page setup & style
# =========================
st.set_page_config(page_title="Weather Dashboard", page_icon="🌤️", layout="wide")
st.markdown("""
    <style>
      .app-header { display:flex; align-items:center; gap:12px; }
      .app-pill   { background:#EEF2FF; color:#3730A3; padding:2px 10px; border-radius:999px; font-size:12px; }
      .app-footer { color:#8f8f8f; text-align:center; margin-top:32px; font-size:12px; }
      .metric-note { color:#6b7280; font-size:12px; }
      .ok { color:#059669; }
      .warn { color:#b45309; }
      .err { color:#b91c1c; }
    </style>
""", unsafe_allow_html=True)

DEFAULT_PATH = "/workspaces/learningexamples/weather_data.csv"

# =========================
# Helpers
# =========================
def find_col(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_csv(path: str, file_bytes=None) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(file_bytes)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    return pd.read_csv(p)

def normalize_df(df_in: pd.DataFrame):
    """Return (df, city_col, date_col) with standardized columns:
       TemperatureC, HumidityPct, DateParsed (optional)"""
    df = df_in.copy()
    cols = df.columns.tolist()

    temp_c_col = find_col(["Temperature (°C)", "Temp (°C)", "temperature_c", "temp_c"], cols)
    temp_f_col = find_col(["Temperature (°F)", "Temp (°F)", "Temperature", "temperature_f", "temp_f", "temp"], cols)
    hum_col    = find_col(["Humidity (%)", "Humidity", "humidity", "hum"], cols)
    city_col   = find_col(["City", "city", "Location", "location"], cols)
    date_col   = find_col(["Date", "date", "Day", "day", "Timestamp", "timestamp"], cols)

    # TemperatureC
    if temp_c_col:
        df["TemperatureC"] = pd.to_numeric(df[temp_c_col], errors="coerce")
    elif temp_f_col:
        df["TemperatureC"] = (pd.to_numeric(df[temp_f_col], errors="coerce") - 32) * 5.0/9.0
    else:
        df["TemperatureC"] = np.nan

    # HumidityPct
    if hum_col:
        df["HumidityPct"] = pd.to_numeric(df[hum_col], errors="coerce")
    else:
        df["HumidityPct"] = np.nan

    # DateParsed
    if date_col:
        try:
            df["DateParsed"] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            df["DateParsed"] = pd.NaT
    else:
        df["DateParsed"] = pd.NaT

    return df, city_col, date_col

def save_csv(df: pd.DataFrame, path: str) -> tuple[bool, str]:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True, f"Saved to {path}"
    except Exception as e:
        return False, str(e)

# =========================
# Sidebar (global controls)
# =========================
st.sidebar.title("⚙️ Controls")

# A quick path picker for the main dataset (still available)
csv_path = st.sidebar.text_input("Active CSV path", value=DEFAULT_PATH)
uploaded_quick = st.sidebar.file_uploader("Quick load CSV (optional)", type=["csv"], key="quick_load")

smooth = st.sidebar.checkbox("Apply rolling smoothing", value=True)
window = st.sidebar.slider("Rolling window (days)", min_value=2, max_value=14, value=3, step=1)
show_points = st.sidebar.checkbox("Show points on charts", value=False)
show_table_index = st.sidebar.checkbox("Show Day index", value=True)

# =========================
# Data source (active)
# =========================
try:
    if uploaded_quick is not None:
        raw_df = load_csv("", uploaded_quick)
    else:
        raw_df = load_csv(csv_path)
except Exception as e:
    st.error(f"Could not load active CSV: {e}")
    st.stop()

df, city_col, date_col = normalize_df(raw_df)

# A display copy with optional 1-based Day index
df_display = df.copy()
if show_table_index:
    df_display.index = df_display.index + 1
    df_display.index_name = "Day"

# =========================
# Header & Metrics
# =========================
st.markdown('<div class="app-header">'
            '<span class="app-pill">Weather</span>'
            '<h1 style="margin:0">Weather Dashboard</h1>'
            '</div>', unsafe_allow_html=True)
st.caption("CSV-driven, interactive dashboard with charts, insights, and an Upload page to manage data files.")

# City filter (if available)
if city_col:
    cities = ["(All)"] + sorted([c for c in df[city_col].dropna().astype(str).unique()])
    selected_city = st.selectbox("City", options=cities, index=0)
    if selected_city != "(All)":
        df = df[df[city_col].astype(str) == selected_city].copy()
else:
    selected_city = None

# Smoothing for plotting
plot_df = df[["TemperatureC", "HumidityPct", "DateParsed"]].copy()
if smooth and not plot_df.empty:
    plot_df["TemperatureC_s"] = plot_df["TemperatureC"].rolling(window=window, min_periods=1, center=True).mean()
    plot_df["HumidityPct_s"] = plot_df["HumidityPct"].rolling(window=window, min_periods=1, center=True).mean()
else:
    plot_df["TemperatureC_s"] = plot_df["TemperatureC"]
    plot_df["HumidityPct_s"] = plot_df["HumidityPct"]

# Top metrics
left, mid, right = st.columns(3)
avg_temp = plot_df["TemperatureC"].mean() if not plot_df["TemperatureC"].isna().all() else np.nan
max_temp = plot_df["TemperatureC"].max() if not plot_df["TemperatureC"].isna().all() else np.nan
avg_hum  = plot_df["HumidityPct"].mean() if not plot_df["HumidityPct"].isna().all() else np.nan
left.metric("Average Temp (°C)", f"{avg_temp:.1f}" if pd.notna(avg_temp) else "—")
mid.metric("Max Temp (°C)", f"{max_temp:.1f}" if pd.notna(max_temp) else "—")
right.metric("Average Humidity (%)", f"{avg_hum:.0f}" if pd.notna(avg_hum) else "—")
st.markdown('<div class="metric-note">Metrics computed from the current filter.</div>', unsafe_allow_html=True)

# =========================
# Pages (tabs) — now includes Upload
# =========================
tab_overview, tab_data, tab_charts, tab_insights, tab_upload = st.tabs(
    ["Overview", "Data", "Charts", "Insights", "Upload"]
)

# -------- Overview --------
with tab_overview:
    st.subheader("Quick Overview")
    st.write("A high-level look at your dataset. Use the sidebar to adjust smoothing and file source, and the city selector to filter.")
    if not plot_df[["TemperatureC_s", "HumidityPct_s"]].dropna(how="all").empty:
        st.line_chart(plot_df[["TemperatureC_s", "HumidityPct_s"]], use_container_width=True)
    else:
        st.info("No plottable data found yet.")

# -------- Data --------
with tab_data:
    st.subheader("Data Table")
    st.caption("Edit cells to correct values or explore; changes affect charts and insights during the session.")
    st.data_editor(
        df_display,
        use_container_width=True,
        num_rows="dynamic",
        key="data_editor_main",
    )
    st.download_button(
        "Download current view as CSV",
        data=df_display.to_csv(index=show_table_index).encode("utf-8"),
        file_name="weather_current_view.csv",
        mime="text/csv",
    )

# -------- Charts --------
with tab_charts:
    st.subheader("Charts")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temperature (°C)**")
        fig1, ax1 = plt.subplots()
        y = plot_df["TemperatureC_s"]
        x = plot_df["DateParsed"] if plot_df["DateParsed"].notna().any() else np.arange(len(y)) + 1
        ax1.plot(x, y, marker="o" if show_points else None)
        ax1.set_xlabel("Date" if plot_df["DateParsed"].notna().any() else "Day")
        ax1.set_ylabel("°C")
        ax1.set_title("Temperature Trend")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with c2:
        st.markdown("**Humidity (%)**")
        fig2, ax2 = plt.subplots()
        y2 = plot_df["HumidityPct_s"]
        x2 = plot_df["DateParsed"] if plot_df["DateParsed"].notna().any() else np.arange(len(y2)) + 1
        ax2.plot(x2, y2, marker="o" if show_points else None)
        ax2.set_xlabel("Date" if plot_df["DateParsed"].notna().any() else "Day")
        ax2.set_ylabel("%")
        ax2.set_title("Humidity Trend")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("**Scatter: Temperature vs Humidity**")
    fig3, ax3 = plt.subplots()
    ax3.scatter(plot_df["TemperatureC"], plot_df["HumidityPct"])
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Humidity (%)")
    ax3.set_title("Temp vs Humidity")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

# -------- Insights --------
with tab_insights:
    st.subheader("Insights")
    bullets = []
    if pd.notna(avg_temp):
        if avg_temp >= 30:
            bullets.append("High average temperature — consider heat alerts.")
        elif avg_temp < 10:
            bullets.append("Low average temperature — cold conditions likely.")
        else:
            bullets.append("Moderate average temperature overall.")
    if pd.notna(avg_hum):
        if avg_hum >= 70:
            bullets.append("Humidity is typically high — potential for discomfort or storms.")
        elif avg_hum <= 35:
            bullets.append("Humidity is relatively low — dry conditions.")
        else:
            bullets.append("Humidity is in a comfortable mid-range.")
    if not bullets:
        st.info("Not enough data for insights yet.")
    else:
        for b in bullets:
            st.write(f"- {b}")

# -------- Upload (NEW PAGE) --------
with tab_upload:
    st.subheader("Upload / Manage Data")
    st.write("Upload a CSV, preview it, map columns, and optionally **save** it as the active dataset.")

    # 1) Upload
    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        up_file = st.file_uploader("Choose a CSV to upload", type=["csv"], key="upload_page_uploader")
    with up_col2:
        save_target = st.text_input("Save as (path)", value=DEFAULT_PATH, help="Where to save the uploaded CSV")

    if up_file is not None:
        try:
            up_raw = pd.read_csv(up_file)
            st.success(f"Loaded file with {up_raw.shape[0]} rows × {up_raw.shape[1]} columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            up_raw = None

        if up_raw is not None:
            st.markdown("**Preview (first 10 rows):**")
            st.dataframe(up_raw.head(10), use_container_width=True)

            st.markdown("### Column Mapping")
            cols = up_raw.columns.tolist()

            # Pickers for columns
            temp_col = st.selectbox("Temperature column", options=["(none)"] + cols, index=0)
            temp_unit = st.radio("Temperature unit", ["°C", "°F"], index=0, horizontal=True)
            hum_col  = st.selectbox("Humidity column", options=["(none)"] + cols, index=0)
            date_col_u = st.selectbox("Date/Day column (optional)", options=["(none)"] + cols, index=0)
            city_col_u = st.selectbox("City column (optional)", options=["(none)"] + cols, index=0)

            # Build normalized frame
            norm = pd.DataFrame()
            if temp_col != "(none)":
                tser = pd.to_numeric(up_raw[temp_col], errors="coerce")
                if temp_unit == "°F":
                    tser = (tser - 32) * 5.0/9.0
                norm["TemperatureC"] = tser
            if hum_col != "(none)":
                norm["HumidityPct"] = pd.to_numeric(up_raw[hum_col], errors="coerce")
            if date_col_u != "(none)":
                norm["DateParsed"] = pd.to_datetime(up_raw[date_col_u], errors="coerce")
            if city_col_u != "(none)":
                norm["City"] = up_raw[city_col_u].astype(str)

            # Show normalized result
            st.markdown("**Normalized preview (first 10):**")
            st.dataframe(norm.head(10), use_container_width=True)

            # Validation
            problems = []
            if "TemperatureC" not in norm.columns:
                problems.append("Temperature column is required.")
            if "HumidityPct" not in norm.columns:
                problems.append("Humidity column is recommended.")
            if norm.get("TemperatureC", pd.Series(dtype=float)).dropna().empty:
                problems.append("Temperature has no numeric values after conversion.")

            if problems:
                st.markdown("#### Validation")
                for p in problems:
                    st.markdown(f"- <span class='warn'>{p}</span>", unsafe_allow_html=True)

            # 2) Save
            st.markdown("### Save")
            save_btn = st.button("Save normalized CSV to path", disabled=bool(problems))
            if save_btn:
                ok, msg = save_csv(norm, save_target)
                if ok:
                    st.success(msg)
                    st.info("To use this saved file as the active dataset, set the **Active CSV path** in the sidebar.")
                else:
                    st.error(f"Save failed: {msg}")

    st.divider()
    st.caption("Tip: Use this page to onboard new files in varied formats. Map columns, convert °F→°C, and save a clean CSV for analysis.")

# =========================
# Footer
# =========================
st.markdown('<div class="app-footer">Built with Streamlit • CSV-driven • Upload page with mapping & auto °F→°C</div>', unsafe_allow_html=True)

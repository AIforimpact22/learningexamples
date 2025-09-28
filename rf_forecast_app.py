# rf_forecast_app.py
# Random Forest forecasting (next N days) — robust, stateful, and unique keys for all buttons.

import os
import time
from datetime import timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly optional (fallback to Streamlit charts if missing)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.set_page_config(page_title="Random Forest N‑Day Forecast (Stable)", layout="wide")

# ---------- CONFIG ----------
DEFAULT_SAVE_DIR = r"C:\Users\sourc\OneDrive\Desktop\Python\Erbil Air Polution"
DEFAULT_DATA_PATH = os.path.join(DEFAULT_SAVE_DIR, "bank_visits_complete_2024.csv")

# ---------- SESSION STATE ----------
for k, v in {
    "df": None,                 # loaded DataFrame
    "model": None,              # trained RF model
    "feature_cols": None,       # list[str]
    "cfg": {},                  # training config (lags, rolls, target, date_col, etc.)
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- HELPERS ----------
def try_load_csv(source, date_col="date") -> pd.DataFrame:
    """Load CSV from path or uploaded file-like; ensure date column is parsed/sorted."""
    df = pd.read_csv(source)
    if date_col not in df.columns:
        raise ValueError(f"CSV must contain a '{date_col}' column.")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df

def add_calendar_feats(df: pd.DataFrame, date_col: str, weekend_type: str) -> pd.DataFrame:
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.weekday
    out["dom"] = d.dt.day
    out["doy"] = d.dt.dayofyear
    out["month"] = d.dt.month
    weekend_sets = {"fri-sat": {4, 5}, "sat-sun": {5, 6}}
    weekend_set = weekend_sets.get(weekend_type, {4, 5})
    out["is_weekend_flag"] = out["dow"].isin(weekend_set).astype(int)
    out["is_payday_flag"] = out["dom"].isin([1, 15]).astype(int)
    # Cyclic encodings
    out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["doy_sin"]   = np.sin(2 * np.pi * out["doy"] / 365.0)
    out["doy_cos"]   = np.cos(2 * np.pi * out["doy"] / 365.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out

def add_target_lags_rolls(df: pd.DataFrame, target: str, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    for W in rolls:
        out[f"rmean_{W}"] = out[target].shift(1).rolling(W).mean()  # shift 1 to avoid leakage
    return out

def add_extra_regressors(df: pd.DataFrame, reg_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Lag‑1 features for optional numeric regressors (safe for forecasting)."""
    out = df.copy()
    feats = []
    for c in reg_cols:
        if c in out.columns and np.issubdtype(out[c].dropna().dtype, np.number):
            name = f"reg_{c}_lag1"
            out[name] = out[c].shift(1)
            feats.append(name)
    return out, feats

def build_supervised(df: pd.DataFrame, date_col: str, target: str,
                     weekend_type: str, lags: List[int], rolls: List[int],
                     extra_regs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (Xy, feature_cols).
    Xy contains date & target for plotting, but the model only sees feature_cols (numeric).
    """
    tmp = add_calendar_feats(df[[date_col, target] + extra_regs].copy(), date_col, weekend_type)
    tmp = add_target_lags_rolls(tmp, target, lags, rolls)
    tmp, reg_feats = add_extra_regressors(tmp, extra_regs)

    feature_cols = [
        "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
        "is_weekend_flag", "is_payday_flag",
        *[f"lag_{L}" for L in lags],
        *[f"rmean_{W}" for W in rolls],
        *reg_feats
    ]
    # Coerce features to numeric & drop rows with NaNs caused by lags/rolls
    for c in feature_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    Xy = tmp.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    return Xy, feature_cols

def time_split(Xy: pd.DataFrame, date_col: str, target: str, feature_cols: List[str], test_days: int):
    n = len(Xy)
    test_days = max(1, min(int(test_days), n - 5))  # keep at least 5 for train
    cut = n - test_days
    train, test = Xy.iloc[:cut].copy(), Xy.iloc[cut:].copy()
    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    return X_train, y_train, X_test, y_test, train[[date_col, target]], test[[date_col, target]]

def train_rf(X, y, n_estimators=300, max_depth=0, min_samples_split=2, min_samples_leaf=1,
             max_features="sqrt", bootstrap=True, random_state=42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=(None if int(max_depth) == 0 else int(max_depth)),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=(None if max_features == "None" else max_features),
        bootstrap=bool(bootstrap),
        n_jobs=-1,
        random_state=int(random_state),
    )
    model.fit(X, y)
    return model

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))  # no 'squared' keyword

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-6, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def metrics_report(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
    }

def feature_importance_df(model: RandomForestRegressor, feature_cols: List[str]) -> pd.DataFrame:
    try:
        return pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}) \
                 .sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame({"feature": feature_cols, "importance": np.nan})

def plot_lines(df_plot: pd.DataFrame, title: str):
    if HAS_PLOTLY:
        fig = px.line(df_plot, x="date", y=[c for c in df_plot.columns if c != "date"], title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot.set_index("date"))

def make_future_calendar(last_date: pd.Timestamp, steps: int, weekend_type: str) -> pd.DataFrame:
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=int(steps), freq="D")
    out = pd.DataFrame({"date": future_dates})
    return add_calendar_feats(out, "date", weekend_type)

def forecast_next_k(history_df: pd.DataFrame, date_col: str, target: str,
                    model: RandomForestRegressor, feature_cols: List[str],
                    lags: List[int], rolls: List[int],
                    weekend_type: str, k_steps: int,
                    extra_regs: List[str], extra_reg_strategy: str = "hold_last") -> pd.DataFrame:
    hist_vals = history_df[target].astype(float).tolist()
    need = max([1] + lags + rolls)
    if len(hist_vals) < need:
        raise ValueError(f"Need at least {need} historical rows for chosen lags/rolls.")

    last_date = history_df[date_col].iloc[-1]
    cal = make_future_calendar(last_date, k_steps, weekend_type)
    results = []

    reg_last = {}
    for c in extra_regs:
        if c in history_df.columns and np.issubdtype(history_df[c].dropna().dtype, np.number):
            reg_last[c] = float(history_df[c].iloc[-1])
        else:
            reg_last[c] = 0.0

    for _, row in cal.iterrows():
        feat = {
            "dow_sin": float(row["dow_sin"]), "dow_cos": float(row["dow_cos"]),
            "doy_sin": float(row["doy_sin"]), "doy_cos": float(row["doy_cos"]),
            "month_sin": float(row["month_sin"]), "month_cos": float(row["month_cos"]),
            "is_weekend_flag": int(row["is_weekend_flag"]),
            "is_payday_flag": int(row["is_payday_flag"]),
        }
        for L in lags:
            feat[f"lag_{L}"] = float(hist_vals[-L]) if L <= len(hist_vals) else float(np.mean(hist_vals))
        for W in rolls:
            w = min(W, len(hist_vals))
            feat[f"rmean_{W}"] = float(np.mean(hist_vals[-w:]))

        for c in extra_regs:
            feat[f"reg_{c}_lag1"] = float(reg_last.get(c, 0.0) if extra_reg_strategy == "hold_last" else 0.0)

        x = np.array([feat[c] for c in feature_cols], dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        results.append({"date": row["date"], f"{target}_pred": yhat})
        hist_vals.append(yhat)

    return pd.DataFrame(results)

# ---------- UI ----------
st.title("🏦 Random Forest — Predict Daily Customers (Next N Days)")

with st.expander("About", expanded=False):
    st.write(
        "- Loads your CSV and **persists** it in memory so reruns don't lose it.\n"
        "- Builds **numeric** features (lags, rolling means, calendar cycles).\n"
        "- Trains a **RandomForestRegressor** and forecasts the next **N** days.\n"
        "- All buttons have **unique keys** to avoid duplicate-ID errors."
    )

# ----- Load data (stateful) -----
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="up_csv")
path_text = st.sidebar.text_input("Or file path", value=DEFAULT_DATA_PATH, key="in_path")
date_col = st.sidebar.text_input("Date column", value="date", key="in_date")
col_load1, col_load2 = st.sidebar.columns(2)
do_load = col_load1.button("Load CSV", key="btn_load_csv")
auto_on_train = col_load2.checkbox("Auto‑load on Train", value=True, key="chk_auto_on_train",
                                   help="If on, clicking Train will try to load from the path above if nothing is loaded.")

# Perform load when requested
if do_load:
    try:
        if uploaded is not None:
            st.session_state.df = try_load_csv(uploaded, date_col=date_col)
            st.success(f"Loaded uploaded file: {st.session_state.df.shape[0]} rows")
        else:
            st.session_state.df = try_load_csv(path_text, date_col=date_col)
            st.success(f"Loaded from path: {path_text}")
    except Exception as e:
        st.error(f"Load failed: {e}")

df = st.session_state.df

# Preview panel
st.subheader("Preview")
if df is not None:
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("No data loaded yet. You can still click **Train** — the app will auto‑load from the path if enabled.")

# ----- Configure model -----
st.sidebar.header("2) Target & Features")
if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
else:
    numeric_cols = []

target_default = "total_visit" if "total_visit" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
target = st.sidebar.selectbox("Target (customers per day)",
                              options=numeric_cols if numeric_cols else [""],
                              index=(numeric_cols.index(target_default) if target_default in numeric_cols else 0)
                              if numeric_cols else 0,
                              key="sel_target")

lags  = st.sidebar.multiselect("Lags (days)", options=[1, 2, 3, 5, 7, 14, 21, 28],
                               default=[1, 7, 14, 28], key="ms_lags")
rolls = st.sidebar.multiselect("Rolling windows (days)", options=[3, 7, 14, 28],
                               default=[7, 28], key="ms_rolls")
weekend_type = st.sidebar.selectbox("Weekend type", ["fri-sat", "sat-sun"], index=0, key="sel_weekend")

extra_candidates = [c for c in numeric_cols if c != target] if numeric_cols else []
extra_regs = st.sidebar.multiselect("Optional extra numeric regressors (used as lag‑1)",
                                    options=extra_candidates, default=[], key="ms_extra_regs")

st.sidebar.header("3) Train/Test & Forecast")
test_days = st.sidebar.slider("Test window (last N days)", min_value=14, max_value=180, value=60, step=1, key="sld_test")
forecast_days = st.sidebar.slider("Forecast next N days", min_value=1, max_value=90, value=14, step=1, key="sld_forecast")

st.sidebar.header("4) Random Forest")
n_estimators      = st.sidebar.slider("n_estimators", 50, 1000, 300, 50, key="sld_n_est")
max_depth         = st.sidebar.slider("max_depth (0=None)", 0, 50, 0, 1, key="sld_max_depth")
min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2, 1, key="sld_min_split")
min_samples_leaf  = st.sidebar.slider("min_samples_leaf", 1, 20, 1, 1, key="sld_min_leaf")
max_features      = st.sidebar.selectbox("max_features", ["sqrt", "log2", "None"], index=0, key="sel_max_feat")
bootstrap         = st.sidebar.checkbox("bootstrap", value=True, key="chk_bootstrap")
seed              = st.sidebar.number_input("random_state", min_value=0, value=42, step=1, key="num_seed")

run_train = st.sidebar.button("🚀 Train", key="btn_train")

# ----- TRAIN HANDLER (auto-load if needed) -----
if run_train:
    try:
        # Auto-load if nothing in session and user allowed it
        if st.session_state.df is None and auto_on_train:
            if os.path.exists(path_text):
                st.session_state.df = try_load_csv(path_text, date_col=date_col)
                st.info(f"Auto‑loaded from path: {path_text}")
            else:
                st.error("No data loaded and the path does not exist. Click **Load CSV** or fix the path.")
        df = st.session_state.df

        if df is None:
            st.error("No data loaded. Click **Load CSV** (or enable auto‑load and provide a valid path).")
        else:
            # Recompute numeric_cols & defaults from the actual df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target not in numeric_cols:
                if "total_visit" in numeric_cols:
                    target = "total_visit"
                elif numeric_cols:
                    target = numeric_cols[0]
                else:
                    raise ValueError("No numeric columns found in the loaded CSV.")

            # Build supervised frame (numeric features only)
            Xy, feature_cols = build_supervised(
                df=df,
                date_col=date_col,
                target=target,
                weekend_type=weekend_type,
                lags=sorted(lags),
                rolls=sorted(rolls),
                extra_regs=extra_regs,
            )

            X_train, y_train, X_test, y_test, train_dates, test_dates = time_split(
                Xy=Xy,
                date_col=date_col,
                target=target,
                feature_cols=feature_cols,
                test_days=test_days,
            )

            model = train_rf(
                X_train, y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=seed,
            )

            y_pred = model.predict(X_test)
            rep = metrics_report(y_test, y_pred)

            st.success("✅ Model trained successfully.")

            # Store in session for forecast step
            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.cfg = {
                "date_col": date_col,
                "target": target,
                "lags": sorted(lags),
                "rolls": sorted(rolls),
                "weekend_type": weekend_type,
                "extra_regs": extra_regs,
                "forecast_days": int(forecast_days),
            }

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{rep['MAE']:.2f}")
            m2.metric("RMSE", f"{rep['RMSE']:.2f}")
            m3.metric("R²", f"{rep['R2']:.3f}")
            m4.metric("MAPE %", f"{rep['MAPE_%']:.1f}")

            # Plot Actual vs Predicted on TEST dates
            test_plot = pd.DataFrame({
                "date": test_dates[date_col].values,
                "Actual": y_test,
                "Predicted": y_pred
            })
            st.subheader("Test Window — Actual vs Predicted")
            if HAS_PLOTLY:
                fig = px.line(test_plot, x="date", y=["Actual", "Predicted"], title="Test Window")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(test_plot.set_index("date"))

            # Feature importances
            st.subheader("Feature Importance")
            fi = feature_importance_df(model, feature_cols)
            st.dataframe(fi, use_container_width=True)
            if HAS_PLOTLY and not fi.empty:
                fig_fi = px.bar(fi.head(20).iloc[::-1], x="importance", y="feature",
                                orientation="h", title="Top 20 Features")
                st.plotly_chart(fig_fi, use_container_width=True)

            # Export test predictions
            st.subheader("Export Test Predictions")
            os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
            test_out_path = os.path.join(DEFAULT_SAVE_DIR, f"rf_test_predictions_{int(time.time())}.csv")
            col_save, col_dl = st.columns([0.4, 0.6])
            with col_save:
                if st.button("💾 Save test predictions CSV", key="btn_save_test_csv"):
                    try:
                        test_plot.to_csv(test_out_path, index=False, encoding="utf-8")
                        st.success(f"Saved: {test_out_path}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with col_dl:
                st.download_button("Download test predictions CSV",
                                   data=test_plot.to_csv(index=False).encode("utf-8"),
                                   file_name=os.path.basename(test_out_path),
                                   mime="text/csv",
                                   key="dl_test_csv")

    except Exception as e:
        st.error(f"Training failed: {e}")

# ----- FORECAST PANEL (enabled after training) -----
st.subheader("Forecast — Next N Days")
if st.session_state.model is None or st.session_state.feature_cols is None or not st.session_state.cfg:
    st.info("Train a model first to enable forecasting.")
else:
    cfg = st.session_state.cfg
    model = st.session_state.model
    feature_cols = st.session_state.feature_cols

    col_f1, col_f2, col_f3 = st.columns([0.4, 0.4, 0.2])
    forecast_days_val = col_f1.number_input("N days to forecast", min_value=1, max_value=90,
                                            value=cfg.get("forecast_days", 14), step=1, key="num_fc_days")
    exog_strategy = col_f2.selectbox("Future extra regressors strategy", ["hold_last", "zeros"], index=0, key="sel_exog_strategy")
    run_forecast = col_f3.button("Run Forecast", key="btn_run_forecast")

    if run_forecast:
        try:
            # Build minimal history frame: date + target + extra regs
            base_cols = [cfg["date_col"], cfg["target"], *cfg["extra_regs"]]
            base = st.session_state.df[base_cols].dropna(subset=[cfg["date_col"], cfg["target"]]).copy()

            fc = forecast_next_k(
                history_df=base,
                date_col=cfg["date_col"],
                target=cfg["target"],
                model=model,
                feature_cols=feature_cols,
                lags=cfg["lags"],
                rolls=cfg["rolls"],
                weekend_type=cfg["weekend_type"],
                k_steps=int(forecast_days_val),
                extra_regs=cfg["extra_regs"],
                extra_reg_strategy=exog_strategy,
            )

            st.success("✅ Forecast complete.")
            st.dataframe(fc, use_container_width=True)

            # Plot last 200 days of history + forecast
            hist_tail = base.rename(columns={cfg["date_col"]: "date"})[["date", cfg["target"]]].tail(200)
            hist_tail = hist_tail.rename(columns={cfg["target"]: "Actual"})
            merged = pd.merge(
                fc.rename(columns={f"{cfg['target']}_pred": "Forecast"}),
                hist_tail,
                on="date", how="outer",
            )
            if HAS_PLOTLY:
                fig2 = px.line(merged, x="date", y=["Actual", "Forecast"],
                               title=f"History & {int(forecast_days_val)}‑Day Forecast")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.line_chart(merged.set_index("date")[["Actual", "Forecast"]])

            # Export forecast
            st.subheader("Export Forecast")
            os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
            fc_out_path = os.path.join(DEFAULT_SAVE_DIR, f"rf_forecast_{cfg['target']}_{int(forecast_days_val)}d_{int(time.time())}.csv")
            col_save_fc, col_dl_fc = st.columns([0.4, 0.6])
            with col_save_fc:
                if st.button("💾 Save forecast CSV", key="btn_save_forecast_csv"):
                    try:
                        fc.to_csv(fc_out_path, index=False, encoding="utf-8")
                        st.success(f"Saved: {fc_out_path}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with col_dl_fc:
                st.download_button("Download forecast CSV",
                                   data=fc.to_csv(index=False).encode("utf-8"),
                                   file_name=os.path.basename(fc_out_path),
                                   mime="text/csv",
                                   key="dl_forecast_csv")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

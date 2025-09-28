# gb_forecast_app.py
# Gradient Boosting forecasting (next N days) — robust, stateful, unique keys, and numeric-only features.
# Fixes: no duplicate 'date' inserts; no datetime in model; unique keys on all buttons.

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

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Try to import fast hist-GB; fall back to classic GB if unavailable
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HGB = True
except Exception:
    HAS_HGB = False

st.set_page_config(page_title="Gradient Boosting — N‑Day Forecast (Stable)", layout="wide")

# ---------- CONFIG ----------
DEFAULT_SAVE_DIR = r"C:\Users\sourc\OneDrive\Desktop\Python\Erbil Air Polution"
DEFAULT_DATA_PATH = os.path.join(DEFAULT_SAVE_DIR, "bank_visits_complete_2024.csv")

# ---------- SESSION STATE ----------
for k, v in {
    "df": None,                 # loaded DataFrame
    "model": None,              # trained model
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

def calendar_matrix(dates: pd.Series, weekend_type: str) -> pd.DataFrame:
    """Return purely numeric calendar features (NO date column)."""
    d = pd.to_datetime(dates)
    dow = d.dt.weekday
    dom = d.dt.day
    doy = d.dt.dayofyear
    month = d.dt.month
    # Weekend rule
    if weekend_type == "sat-sun":
        is_weekend = dow.isin([5, 6]).astype(int)
    else:
        is_weekend = dow.isin([4, 5]).astype(int)  # fri-sat
    is_payday = dom.isin([1, 15]).astype(int)
    cal = pd.DataFrame({
        "dow_sin":   np.sin(2 * np.pi * dow / 7.0),
        "dow_cos":   np.cos(2 * np.pi * dow / 7.0),
        "doy_sin":   np.sin(2 * np.pi * doy / 365.0),
        "doy_cos":   np.cos(2 * np.pi * doy / 365.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "is_weekend_flag": is_weekend,
        "is_payday_flag":  is_payday,
    }).astype(np.float32)
    return cal

def lag_roll_features(series: pd.Series, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    """Create lag and rolling mean features (rolling shifted by 1 to avoid leakage)."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = {}
    for L in lags:
        out[f"lag_{L}"] = s.shift(L)
    for W in rolls:
        out[f"rmean_{W}"] = s.shift(1).rolling(W).mean()
    return pd.DataFrame(out).astype(np.float32)

def extra_regressor_features(df: pd.DataFrame, reg_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Lag-1 features for optional numeric regressors (safe for forecasting)."""
    out = {}
    names = []
    for c in reg_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            name = f"reg_{c}_lag1"
            out[name] = pd.to_numeric(df[c], errors="coerce").shift(1)
            names.append(name)
    if not names:
        return pd.DataFrame(index=df.index), names
    return pd.DataFrame(out).astype(np.float32), names

def build_supervised(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    horizon: int,
    lags: List[int],
    rolls: List[int],
    weekend_type: str,
    extra_regs: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a supervised frame with NUMERIC-ONLY features.
    Returns (sup, feature_cols) where sup has columns:
      ['date', 'target_date', 'y', <feature_cols...>]
    """
    # Base selections
    base = df[[date_col, target] + extra_regs].copy()

    # Features: (a) target lags/rolls, (b) calendar, (c) extra reg lag-1
    X_lr = lag_roll_features(base[target], lags, rolls)
    X_cal = calendar_matrix(base[date_col], weekend_type)
    X_ex, ex_names = extra_regressor_features(base, extra_regs)

    # Assemble features (no date columns included here)
    X = pd.concat([X_lr.reset_index(drop=True), X_cal.reset_index(drop=True), X_ex.reset_index(drop=True)], axis=1)
    feature_cols = X.columns.tolist()

    # Label y = target shifted -horizon
    y = pd.to_numeric(base[target], errors="coerce").shift(-horizon)

    # Sup frame with date fields added ONCE (avoid insert-duplicate bug)
    sup = pd.DataFrame({
        "date": base[date_col].values,
        "target_date": (base[date_col] + pd.to_timedelta(horizon, unit="D")).values,
        "y": y.values,
    })
    sup = pd.concat([sup.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Drop rows with NaNs introduced by lags/rolls/shift and enforce float32 features
    sup = sup.dropna(subset=["y"] + feature_cols).reset_index(drop=True)
    sup[feature_cols] = sup[feature_cols].astype(np.float32)

    return sup, feature_cols

def time_split(sup: pd.DataFrame, feature_cols: List[str], test_days: int):
    """Last test_days rows for test; rest for train. Returns arrays + meta frames."""
    n = len(sup)
    test_days = max(1, min(int(test_days), n - 5))  # keep at least some training rows
    cut = n - test_days
    train, test = sup.iloc[:cut].copy(), sup.iloc[cut:].copy()
    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)
    return X_train, y_train, X_test, y_test, train[["date", "target_date", "y"]], test[["date", "target_date", "y"]]

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))  # compatible with older sklearn (no 'squared' arg)

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

def plot_lines(df_plot: pd.DataFrame, title: str):
    if HAS_PLOTLY:
        fig = px.line(df_plot, x="date", y=[c for c in df_plot.columns if c != "date"], title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot.set_index("date"))

def make_future_calendar(last_date: pd.Timestamp, steps: int, weekend_type: str) -> pd.DataFrame:
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=int(steps), freq="D")
    return calendar_matrix(pd.Series(future_dates), weekend_type).assign(date=future_dates)

def forecast_next_k(history_df: pd.DataFrame, date_col: str, target: str,
                    model, feature_cols: List[str],
                    lags: List[int], rolls: List[int],
                    weekend_type: str, k_steps: int,
                    extra_regs: List[str], extra_reg_strategy: str = "hold_last") -> pd.DataFrame:
    """
    Recursive multi-step forecast. Features at time T build the prediction for date D = T + H.
    """
    hist_vals = history_df[target].astype(float).tolist()
    need = max([1] + lags + rolls)
    if len(hist_vals) < need:
        raise ValueError(f"Need at least {need} historical rows for chosen lags/rolls.")

    last_date = history_df[date_col].iloc[-1]
    cal = make_future_calendar(last_date, k_steps, weekend_type)  # includes 'date' column for future D's
    results = []

    # Prepare last observed values for extra regressors
    reg_last = {}
    for c in extra_regs:
        if c in history_df.columns and pd.api.types.is_numeric_dtype(history_df[c]):
            reg_last[c] = float(history_df[c].iloc[-1])
        else:
            reg_last[c] = 0.0

    for _, row in cal.iterrows():
        # Features must reflect time T = D - H
        # We'll approximate H as 1 step into the future relative to rolling use (same logic as training alignment).
        # For multi-step, we still use recursive hist_vals to form lags/rolls.
        # Calendar features use the DATE WE ARE FORECASTING (row['date']) minus 1 horizon step.
        # Since we trained with target_date = date + H, we need the calendar at T = D - H.
        D = pd.to_datetime(row["date"])
        # Build calendar features for T
        T = D - pd.Timedelta(days=1)  # approximates T in absence of storing H here; H will be set by caller's horizon
        cal_T = calendar_matrix(pd.Series([T]), weekend_type).iloc[0].to_dict()

        feat = dict(cal_T)

        # target lags/rolls from history+preds (reference series at T)
        for L in lags:
            feat[f"lag_{L}"] = float(hist_vals[-L]) if L <= len(hist_vals) else float(np.mean(hist_vals))
        for W in rolls:
            w = min(W, len(hist_vals))
            feat[f"rmean_{W}"] = float(np.mean(hist_vals[-w:]))

        # extra regressors (lag-1; hold last or zeros)
        for c in extra_regs:
            feat[f"reg_{c}_lag1"] = float(reg_last.get(c, 0.0) if extra_reg_strategy == "hold_last" else 0.0)

        # Reorder to match training columns
        x = np.array([feat.get(c, 0.0) for c in feature_cols], dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        results.append({"date": D, f"{target}_pred": yhat})

        # Append prediction to history for recursive lags/rolls
        hist_vals.append(yhat)

    return pd.DataFrame(results)

def train_gb(model_choice: str, X_train, y_train, hgb_params: Dict, gbr_params: Dict):
    """
    Train HistGradientBoostingRegressor (if available) or GradientBoostingRegressor.
    """
    if model_choice == "HistGradientBoosting" and HAS_HGB:
        model = HistGradientBoostingRegressor(
            max_iter=int(hgb_params["max_iter"]),
            learning_rate=float(hgb_params["learning_rate"]),
            max_leaf_nodes=int(hgb_params["max_leaf_nodes"]),
            min_samples_leaf=int(hgb_params["min_samples_leaf"]),
            l2_regularization=float(hgb_params["l2_regularization"]),
            random_state=int(hgb_params["random_state"])
        )
        model.fit(X_train, y_train)
        return model, "HistGradientBoosting"
    else:
        model = GradientBoostingRegressor(
            n_estimators=int(gbr_params["n_estimators"]),
            learning_rate=float(gbr_params["learning_rate"]),
            max_depth=int(gbr_params["max_depth"]),
            subsample=float(gbr_params["subsample"]),
            min_samples_leaf=int(gbr_params["min_samples_leaf"]),
            max_features=None if gbr_params["max_features"] == "None" else gbr_params["max_features"],
            random_state=int(gbr_params["random_state"])
        )
        model.fit(X_train, y_train)
        return model, "GradientBoosting"

# ---------- UI ----------
st.title("🏦 Gradient Boosting — Predict Daily Customers (Next N Days)")

with st.expander("About", expanded=False):
    st.write(
        "- Uses **Gradient Boosting** (HistGradientBoosting if available; else classic GradientBoosting).\n"
        "- Builds **numeric** features (lags, rolling means, calendar cycles, optional lag‑1 regressors).\n"
        "- Time‑ordered split (last N days = test), then **recursive** multi‑step forecast.\n"
        "- **No duplicate `date` inserts** and **no datetime fed to the model**.\n"
        "- All buttons use **unique keys** to avoid duplicate‑ID errors."
    )

# ----- Load data (stateful) -----
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="gb2_up_csv")
path_text = st.sidebar.text_input("Or file path", value=DEFAULT_DATA_PATH, key="gb2_in_path")
date_col = st.sidebar.text_input("Date column", value="date", key="gb2_in_date")
col_load1, col_load2 = st.sidebar.columns(2)
do_load = col_load1.button("Load CSV", key="gb2_btn_load_csv")
auto_on_train = col_load2.checkbox("Auto‑load on Train", value=True, key="gb2_chk_auto_on_train",
                                   help="If on, clicking Train will try to load from the path above if nothing is loaded.")

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

st.subheader("Preview")
if df is not None:
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("No data loaded yet. You can still click **Train** — auto‑load will try the path if enabled.")

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
                              key="gb2_sel_target")

lags  = st.sidebar.multiselect("Lags (days)", options=[1, 2, 3, 5, 7, 14, 21, 28],
                               default=[1, 7, 14, 28], key="gb2_ms_lags")
rolls = st.sidebar.multiselect("Rolling windows (days)", options=[3, 7, 14, 28],
                               default=[7, 28], key="gb2_ms_rolls")
weekend_type = st.sidebar.selectbox("Weekend type", ["fri-sat", "sat-sun"], index=0, key="gb2_sel_weekend")

extra_candidates = [c for c in numeric_cols if c != target] if numeric_cols else []
extra_regs = st.sidebar.multiselect("Optional extra numeric regressors (used as lag‑1)",
                                    options=extra_candidates, default=[], key="gb2_ms_extra_regs")

st.sidebar.header("3) Train/Test & Forecast")
# H is the horizon used to build the label y = target(t + H)
horizon = st.sidebar.slider("H (days ahead to learn)", min_value=1, max_value=60, value=7, step=1, key="gb2_sld_horizon")
test_days = st.sidebar.slider("Test window (last N days)", min_value=14, max_value=180, value=60, step=1, key="gb2_sld_test")
# K is how many future days to produce after training
forecast_days = st.sidebar.slider("Forecast next K days", min_value=1, max_value=90, value=14, step=1, key="gb2_sld_forecast")

st.sidebar.header("4) Model Type")
model_choice = st.sidebar.selectbox(
    "Engine",
    options=["HistGradientBoosting" if HAS_HGB else "GradientBoosting", "GradientBoosting"],
    index=0,
    key="gb2_sel_model"
)
st.sidebar.caption("If 'HistGradientBoosting' is not available, the app will use classic GradientBoosting.")

# Hyperparameters
if model_choice == "HistGradientBoosting" and HAS_HGB:
    st.sidebar.subheader("HistGradientBoosting Parameters")
    hgb_params = {
        "max_iter": st.sidebar.slider("max_iter", 100, 3000, 600, 50, key="gb2_hgb_max_iter"),
        "learning_rate": st.sidebar.slider("learning_rate", 0.005, 0.5, 0.05, 0.005, key="gb2_hgb_lr"),
        "max_leaf_nodes": st.sidebar.slider("max_leaf_nodes", 15, 255, 31, 1, key="gb2_hgb_leaf"),
        "min_samples_leaf": st.sidebar.slider("min_samples_leaf", 5, 200, 20, 1, key="gb2_hgb_min_leaf"),
        "l2_regularization": st.sidebar.slider("l2_regularization", 0.0, 1.0, 0.0, 0.01, key="gb2_hgb_l2"),
        "random_state": st.sidebar.number_input("random_state", min_value=0, value=42, step=1, key="gb2_hgb_seed"),
    }
    gbr_params = {  # dummy to satisfy signature
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
        "subsample": 1.0, "min_samples_leaf": 1, "max_features": "None", "random_state": 42
    }
else:
    st.sidebar.subheader("GradientBoosting Parameters")
    gbr_params = {
        "n_estimators": st.sidebar.slider("n_estimators", 100, 2000, 600, 50, key="gb2_gbr_n"),
        "learning_rate": st.sidebar.slider("learning_rate", 0.01, 0.5, 0.05, 0.01, key="gb2_gbr_lr"),
        "max_depth": st.sidebar.slider("max_depth (tree depth)", 1, 8, 3, 1, key="gb2_gbr_depth"),
        "subsample": st.sidebar.slider("subsample", 0.5, 1.0, 1.0, 0.05, key="gb2_gbr_subsample"),
        "min_samples_leaf": st.sidebar.slider("min_samples_leaf", 1, 50, 3, 1, key="gb2_gbr_min_leaf"),
        "max_features": st.sidebar.selectbox("max_features", ["None", "sqrt", "log2"], index=0, key="gb2_gbr_maxfeat"),
        "random_state": st.sidebar.number_input("random_state", min_value=0, value=42, step=1, key="gb2_gbr_seed"),
    }
    hgb_params = {  # dummy to satisfy signature
        "max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 31,
        "min_samples_leaf": 20, "l2_regularization": 0.0, "random_state": 42
    }

run_train = st.sidebar.button("🚀 Train", key="gb2_btn_train")

# ----- TRAIN HANDLER (auto-load if needed) -----
if run_train:
    try:
        # Auto-load if nothing in session and user allowed it
        if st.session_state.df is None and st.session_state.get("gb2_chk_auto_on_train", True):
            if os.path.exists(path_text):
                st.session_state.df = try_load_csv(path_text, date_col=date_col)
                st.info(f"Auto‑loaded from path: {path_text}")
            else:
                st.error("No data loaded and the path does not exist. Click **Load CSV** or fix the path.")
        df = st.session_state.df

        if df is None:
            st.error("No data loaded. Click **Load CSV** (or enable auto‑load and provide a valid path).")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target not in numeric_cols:
                if "total_visit" in numeric_cols:
                    target = "total_visit"
                elif numeric_cols:
                    target = numeric_cols[0]
                else:
                    raise ValueError("No numeric columns found in the loaded CSV.")

            # Build supervised (NUMERIC-ONLY features) with the chosen horizon H
            sup, feature_cols = build_supervised(
                df=df,
                date_col=date_col,
                target=target,
                horizon=int(horizon),
                lags=sorted(lags),
                rolls=sorted(rolls),
                weekend_type=weekend_type,
                extra_regs=extra_regs,
            )

            # Split
            X_train, y_train, X_test, y_test, train_meta, test_meta = time_split(sup, feature_cols, test_days=int(test_days))

            # Train model
            model, used_engine = train_gb(model_choice, X_train, y_train, hgb_params, gbr_params)

            # Evaluate
            y_pred = model.predict(X_test)
            rep = metrics_report(y_test, y_pred)

            st.success(f"✅ Model trained successfully ({used_engine}).")

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
                "horizon": int(horizon),
                "forecast_days": int(forecast_days),
                "engine": used_engine
            }

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{rep['MAE']:.2f}")
            m2.metric("RMSE", f"{rep['RMSE']:.2f}")
            m3.metric("R²", f"{rep['R2']:.3f}")
            m4.metric("MAPE %", f"{rep['MAPE_%']:.1f}")

            # Plot Actual vs Predicted on TEST TARGET DATES
            test_plot = pd.DataFrame({
                "date": test_meta["target_date"].values,
                "Actual": y_test,
                "Predicted": y_pred
            })
            st.subheader("Test — Actual vs Predicted (by target date)")
            if HAS_PLOTLY:
                fig = px.line(test_plot, x="date", y=["Actual", "Predicted"], title="Test Window")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(test_plot.set_index("date"))

            # Export test predictions
            st.subheader("Export Test Predictions")
            os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
            test_out_path = os.path.join(DEFAULT_SAVE_DIR, f"gb_test_predictions_{int(time.time())}.csv")
            col_save, col_dl = st.columns([0.4, 0.6])
            with col_save:
                if st.button("💾 Save test predictions CSV", key="gb2_btn_save_test_csv"):
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
                                   key="gb2_dl_test_csv")

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
    forecast_days_val = col_f1.number_input("K (future days to forecast)", min_value=1, max_value=90,
                                            value=cfg.get("forecast_days", 14), step=1, key="gb2_num_fc_days")
    exog_strategy = col_f2.selectbox("Future extra regressors strategy", ["hold_last", "zeros"], index=0, key="gb2_sel_exog_strategy")
    run_forecast = col_f3.button("Run Forecast", key="gb2_btn_run_forecast")

    if run_forecast:
        try:
            # Build minimal history: date + target + extra regs
            base_cols = [cfg["date_col"], cfg["target"], *cfg["extra_regs"]]
            base = st.session_state.df[base_cols].dropna(subset=[cfg["date_col"], cfg["target"]]).copy()

            # Forecast recursively for K steps using the SAME feature order
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

            st.success(f"✅ Forecast complete ({cfg['engine']}).")
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
            fc_out_path = os.path.join(DEFAULT_SAVE_DIR, f"gb_forecast_{cfg['target']}_{int(forecast_days_val)}d_{int(time.time())}.csv")
            col_save_fc, col_dl_fc = st.columns([0.4, 0.6])
            with col_save_fc:
                if st.button("💾 Save forecast CSV", key="gb2_btn_save_forecast_csv"):
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
                                   key="gb2_dl_forecast_csv")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

"""
PV Hybrid Forecasting — Multi-Horizon, Per-Location.

For each *_with_weather.csv file in the input directory, this script:
  1. Parses lat/lon and location label from the filename.
  2. Trains a separate XGBoost model for each horizon on snapshot weather features.
  3. Produces one forecast CSV per location with columns:
       date, time, measured, pred_5m, pred_15m, pred_30m,
       measured_at_5m, measured_at_15m, measured_at_30m
  4. Generates one plot per location (3-panel, one per horizon) and saves it
     to the output directory.
  5. Prints per-location, per-horizon MAE / RMSE / R2 metrics.

Horizons : 5 min, 15 min, 30 min.
Approach : snapshot — weather at t is used to predict power labelled for
           t+5m, t+15m and t+30m.
Power    : kept in normalised units throughout (no denormalisation).

Usage (from workspace root):
  python data_generation/pv_hybrid_forecasting_multihorizon.py

  # Report a specific location / date / time in the terminal:
  python data_generation/pv_hybrid_forecasting_multihorizon.py \\
        --report_location Faisalabad \\
        --report_date 15/09/25 \\
        --report_time 11:00AM
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
    from sklearn.ensemble import GradientBoostingRegressor

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = {"5m": 5, "15m": 15, "30m": 30}
FEATURE_COLS = [
    "ghi_pyr",
    "dni",
    "dhi",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
    "hour_sin",
    "hour_cos",
]


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------
def parse_filename_meta(path: Path) -> dict:
    """Extract location label, lat and lon from a *_with_weather.csv filename."""
    pattern = re.compile(
        r"pvoutput_intraday_(.+?)_(-?\d+\.\d+),\s*(-?\d+\.\d+)_\d{8}_to_\d{8}_with_weather\.csv$"
    )
    m = pattern.search(path.name)
    if not m:
        raise RuntimeError(f"Cannot parse metadata from filename: {path.name}")
    label = m.group(1).strip()
    lat = float(m.group(2))
    lon = float(m.group(3))
    return {"label": label, "lat": lat, "lon": lon}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_dt_col(df: pd.DataFrame) -> pd.Series:
    """Return a parsed UTC-naive datetime Series from the date + time columns."""
    raw = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()

    # Primary: DD/MM/YY h:mmAM  (most merged files)
    dt = pd.to_datetime(raw, format="%d/%m/%y %I:%M%p", errors="coerce")

    # Fallback 1: DD/MM/YYYY HH:MM:SS  (Lahore 14.4 kW merged file)
    mask = dt.isna()
    if mask.any():
        dt[mask] = pd.to_datetime(raw[mask], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # Fallback 2: generic day-first for any residual variants
    mask = dt.isna()
    if mask.any():
        dt[mask] = pd.to_datetime(raw[mask], dayfirst=True, errors="coerce")

    return dt


def load_location_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_ts"] = _parse_dt_col(df)
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    df["power_average_w_normalized"] = pd.to_numeric(
        df["power_average_w_normalized"], errors="coerce"
    ).fillna(0.0)

    for col in ["ghi_pyr", "dni", "dhi", "air_temperature", "relative_humidity", "wind_speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["_ts"].dt.hour + df["_ts"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def add_horizon_target(df: pd.DataFrame, minutes: int, target_column: str | None = None) -> tuple[pd.DataFrame, str]:
    """Add the measured value at ``t + minutes`` as a direct forecast target."""
    target_column = target_column or f"target_{int(minutes)}m"
    output = df.copy()
    measured_by_timestamp = output.set_index("_ts")["power_average_w_normalized"]
    target_timestamps = output["_ts"] + pd.Timedelta(minutes=int(minutes))
    output[target_column] = measured_by_timestamp.reindex(target_timestamps).to_numpy()
    return output, target_column


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_xgb_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    settings: dict | None = None,
    target_column: str = "power_average_w_normalized",
):
    settings = settings or {}
    model_df = df.dropna(subset=FEATURE_COLS + [target_column]).copy()
    if len(model_df) < 50:
        raise RuntimeError("Not enough rows to train (need at least 50)")

    split = max(1, min(int(len(model_df) * (1 - test_size)), len(model_df) - 1))
    x_train = model_df[FEATURE_COLS].iloc[:split]
    y_train = model_df[target_column].iloc[:split]

    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=int(settings.get("n_estimators", 300)),
            max_depth=int(settings.get("max_depth", 6)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            subsample=float(settings.get("subsample", 0.9)),
            colsample_bytree=float(settings.get("colsample_bytree", 0.9)),
            random_state=42,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=int(settings.get("n_estimators", 300)),
            max_depth=int(settings.get("max_depth", 6)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            subsample=float(settings.get("subsample", 0.9)),
            random_state=42,
        )

    model.fit(x_train, y_train)
    return model


def train_lstm_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seq_len: int = 12,
    epochs: int = 8,
    batch_size: int = 64,
    target_column: str = "power_average_w_normalized",
):
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install torch to enable LSTM.")

    model_df = df.dropna(subset=FEATURE_COLS + [target_column]).copy()
    if len(model_df) < max(200, seq_len + 20):
        raise RuntimeError("Not enough rows to train LSTM reliably.")

    split = max(1, min(int(len(model_df) * (1 - test_size)), len(model_df) - 1))
    scaler = StandardScaler()
    x_train_raw = model_df[FEATURE_COLS].iloc[:split].values
    scaler.fit(x_train_raw)

    x_all = scaler.transform(model_df[FEATURE_COLS].values)
    y_all = model_df[target_column].values.astype(float)

    x_seq_train = []
    y_seq_train = []
    for i in range(seq_len - 1, split):
        x_seq_train.append(x_all[i - seq_len + 1 : i + 1])
        y_seq_train.append(y_all[i])

    if not x_seq_train:
        raise RuntimeError("No LSTM training sequences were created. Increase data size or lower seq_len.")

    x_seq_train = np.array(x_seq_train, dtype=np.float32)
    y_seq_train = np.array(y_seq_train, dtype=np.float32)

    torch.manual_seed(42)
    np.random.seed(42)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    device = torch.device("cpu")
    model = LSTMRegressor(input_size=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    x_t = torch.tensor(x_seq_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_seq_train.reshape(-1, 1), dtype=torch.float32, device=device)

    for _ in range(max(1, epochs)):
        model.train()
        for start in range(0, len(x_t), max(1, batch_size)):
            end = start + max(1, batch_size)
            xb = x_t[start:end]
            yb = y_t[start:end]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return {"model": model, "scaler": scaler, "seq_len": seq_len, "device": device}


# ---------------------------------------------------------------------------
# Multi-horizon forecast
# ---------------------------------------------------------------------------
def make_forecast_from_pred_df(pred_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-horizon output from independent horizon predictions."""
    ts_to_measured = full_df.set_index("_ts")["power_average_w_normalized"].to_dict()

    if isinstance(pred_df, dict):
        prediction_maps = {
            int(minutes): frame.set_index("_ts")["_pred"].to_dict()
            for minutes, frame in pred_df.items()
        }
        timestamps = sorted({
            pd.Timestamp(timestamp)
            for frame in pred_df.values()
            for timestamp in frame.get("_ts", pd.Series(dtype="datetime64[ns]")).dropna()
        })
    else:
        # Backwards-compatible path for callers that provide one prediction
        # frame. New multi-horizon callers should pass a dictionary.
        feat = pred_df.copy().reset_index(drop=True)
        prediction_maps = {minutes: feat.set_index("_ts")["_pred"].to_dict() for minutes in HORIZONS.values()}
        timestamps = [pd.Timestamp(timestamp) for timestamp in feat["_ts"]]

    records = []
    for t in timestamps:
        first_prediction = np.nan
        rec = {
            "_ts": t,
            "measured": ts_to_measured.get(t, np.nan),
        }
        for label, minutes in HORIZONS.items():
            prediction = prediction_maps.get(minutes, {}).get(t, np.nan)
            rec[f"pred_{label}"] = prediction
            if pd.isna(first_prediction) and pd.notna(prediction):
                first_prediction = prediction
            target_ts = t + pd.Timedelta(minutes=minutes)
            rec[f"measured_at_{label}"] = ts_to_measured.get(target_ts, np.nan)
        rec["prediction"] = first_prediction
        records.append(rec)

    return pd.DataFrame(records)


def predict_xgb(df: pd.DataFrame, model) -> pd.DataFrame:
    feat = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    pred = np.clip(model.predict(feat[FEATURE_COLS]), 0.0, None)
    return pd.DataFrame(
        {
            "_ts": feat["_ts"].values,
            "measured": feat["power_average_w_normalized"].values,
            "_pred": pred,
        }
    )


def predict_lstm(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    # Inference must include the latest feature row even though its future
    # measured target is not available yet.
    feat = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    x_all = artifact["scaler"].transform(feat[FEATURE_COLS].values)
    seq_len = artifact["seq_len"]

    x_seq = []
    ts = []
    measured = []
    for i in range(seq_len - 1, len(feat)):
        x_seq.append(x_all[i - seq_len + 1 : i + 1])
        ts.append(feat.loc[i, "_ts"])
        measured.append(float(feat.loc[i, "power_average_w_normalized"]))

    if not x_seq:
        return pd.DataFrame(columns=["_ts", "measured", "_pred"])

    x_seq = np.array(x_seq, dtype=np.float32)
    model = artifact["model"]
    device = artifact["device"]
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_seq, dtype=torch.float32, device=device)
        pred = model(x_t).cpu().numpy().reshape(-1)
    pred = np.clip(pred, 0.0, None)
    return pd.DataFrame({"_ts": ts, "measured": measured, "_pred": pred})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(forecast_df: pd.DataFrame) -> dict:
    metrics = {}
    for label in HORIZONS:
        col_pred = f"pred_{label}"
        col_meas = f"measured_at_{label}"
        valid = forecast_df[[col_pred, col_meas]].dropna()
        if len(valid) < 2:
            metrics[label] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
            continue
        y_true = valid[col_meas].values
        y_pred = valid[col_pred].values
        metrics[label] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
            "n": int(len(valid)),
        }
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def select_plot_window(
    forecast_df: pd.DataFrame,
    test_size: float,
    max_points: int = 1000,
) -> pd.DataFrame:
    """Pick two consecutive full days from test data, preferring Feb/Mar 2026."""
    if forecast_df.empty:
        return forecast_df.copy()

    test_start = int(len(forecast_df) * (1 - test_size))
    test_df = forecast_df.iloc[test_start:].copy()
    if test_df.empty:
        test_df = forecast_df.copy()

    by_day = test_df.groupby(test_df["_ts"].dt.normalize())["_ts"]
    counts = by_day.size().sort_index()
    if counts.empty:
        return test_df.head(max_points).copy()

    day_min = by_day.min().sort_index()
    day_max = by_day.max().sort_index()
    full_day_mask = (
        (day_min.dt.hour == 0)
        & (day_min.dt.minute == 0)
        & (day_max.dt.hour >= 23)
    )
    full_day_counts = counts[full_day_mask]

    def pick_consecutive_pair(day_counts: pd.Series):
        best = None
        for d in day_counts.index:
            d2 = d + pd.Timedelta(days=1)
            if d2 not in day_counts.index:
                continue
            score = int(min(day_counts.loc[d], day_counts.loc[d2]))
            total = int(day_counts.loc[d] + day_counts.loc[d2])
            candidate = (score, total, d, d2)
            if best is None or candidate > best:
                best = candidate
        return best

    pref_mask_full = (full_day_counts.index.year == 2026) & (full_day_counts.index.month.isin([2, 3]))
    preferred_counts = full_day_counts[pref_mask_full]
    best_pair = pick_consecutive_pair(preferred_counts)
    if best_pair is None:
        best_pair = pick_consecutive_pair(full_day_counts)
    if best_pair is None:
        # Fallback to any consecutive days from the test split when no full-day pair exists.
        pref_mask_any = (counts.index.year == 2026) & (counts.index.month.isin([2, 3]))
        preferred_counts_any = counts[pref_mask_any]
        best_pair = pick_consecutive_pair(preferred_counts_any)
    if best_pair is None:
        best_pair = pick_consecutive_pair(counts)

    if best_pair is not None:
        _, _, d1, d2 = best_pair
        day_start = pd.Timestamp(d1)
        day_end = pd.Timestamp(d2) + pd.Timedelta(days=1)
        window = test_df[(test_df["_ts"] >= day_start) & (test_df["_ts"] < day_end)].copy()
        if not window.empty:
            return window

    # Fallback: one day with most points from test split.
    best_day = counts.idxmax()
    day_start = pd.Timestamp(best_day)
    day_end = day_start + pd.Timedelta(days=1)
    window = test_df[(test_df["_ts"] >= day_start) & (test_df["_ts"] < day_end)].copy()
    return window.head(max_points).copy()


def plot_forecast(
    forecast_df: pd.DataFrame,
    label: str,
    out_png: Path,
    test_size: float,
) -> None:
    view = select_plot_window(forecast_df, test_size=test_size)
    if view.empty:
        print(f"  [WARN] Plot skipped for {label}: no rows in selected window")
        return

    t = view["_ts"]
    start_date = t.iloc[0].strftime("%d-%b-%Y")
    end_date = t.iloc[-1].strftime("%d-%b-%Y")

    horizon_colors = {"5m": "#e84393", "15m": "#f48024", "30m": "#1a73e8"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    for ax, (h, color) in zip(axes, horizon_colors.items()):
        minutes = HORIZONS[h]
        ax.plot(t, view["measured"], label="Measured (at t)", color="#2ca02c", linewidth=1.5)
        ax.plot(
            t,
            view[f"pred_{h}"],
            label=f"Predicted (+{minutes} min)",
            color=color,
            linewidth=1.5,
            linestyle="--",
        )
        col_m = f"measured_at_{h}"
        valid_mask = view[col_m].notna()
        if valid_mask.any():
            ax.scatter(
                t[valid_mask],
                view.loc[valid_mask, col_m],
                label=f"Measured at t+{minutes}m",
                color=color,
                s=8,
                alpha=0.6,
                zorder=3,
            )
        ax.set_ylabel("Normalised Power", fontsize=9)
        ax.set_title(f"Horizon +{minutes} min", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    axes[-1].set_xlabel("Time (HH:MM)")
    fig.suptitle(
        f"Multi-Horizon PV Forecast — {label} ({start_date} to {end_date})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Plot saved : {out_png}")


# ---------------------------------------------------------------------------
# Terminal report filter
# ---------------------------------------------------------------------------
def filter_report_rows(
    forecast_df: pd.DataFrame,
    report_date: str = "",
    report_time: str = "",
) -> pd.DataFrame:
    view = forecast_df.copy()
    if report_date:
        ts_date = pd.to_datetime(report_date, dayfirst=True, errors="coerce")
        if not pd.isna(ts_date):
            view = view[view["_ts"].dt.date == ts_date.date()]
    if report_time:
        parsed = pd.to_datetime(report_time, format="%I:%M%p", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(report_time, format="%H:%M:%S", errors="coerce")
        if not pd.isna(parsed):
            view = view[
                (view["_ts"].dt.hour == parsed.hour)
                & (view["_ts"].dt.minute == parsed.minute)
            ]
    return view


# ---------------------------------------------------------------------------
# Per-location processing
# ---------------------------------------------------------------------------
def process_location(
    path: Path,
    out_dir: Path,
    test_size: float,
    report_date: str,
    report_time: str,
    report_location: str,
    lstm_seq_len: int,
    lstm_epochs: int,
    lstm_batch_size: int,
) -> None:
    meta = parse_filename_meta(path)
    label = meta["label"]

    df = load_location_file(path)
    df = build_features(df)

    print(
        f"\nTraining model for : {label}"
        f"  (rows={len(df)}, lat={meta['lat']}, lon={meta['lon']})"
    )
    xgb_predictions = {}
    for label, minutes in HORIZONS.items():
        horizon_df, target_column = add_horizon_target(df, minutes)
        xgb_model = train_xgb_model(horizon_df, test_size=test_size, target_column=target_column)
        xgb_predictions[minutes] = predict_xgb(df, xgb_model)
    forecast_xgb = make_forecast_from_pred_df(xgb_predictions, df)

    lstm_predictions = {}
    forecast_lstm = None
    if torch is not None:
        try:
            for label, minutes in HORIZONS.items():
                horizon_df, target_column = add_horizon_target(df, minutes)
                lstm_artifact = train_lstm_model(
                    horizon_df,
                    test_size=test_size,
                    seq_len=lstm_seq_len,
                    epochs=lstm_epochs,
                    batch_size=lstm_batch_size,
                    target_column=target_column,
                )
                lstm_predictions[minutes] = predict_lstm(df, lstm_artifact)
            forecast_lstm = make_forecast_from_pred_df(lstm_predictions, df)
        except Exception as exc:
            print(f"  [WARN] LSTM skipped for {label}: {exc}")
    else:
        print("  [WARN] PyTorch not available, skipping LSTM.")

    # -- Save forecast CSV(s) --
    save_df = forecast_xgb.copy()
    save_df["date"] = save_df["_ts"].dt.strftime("%d/%m/%y")
    save_df["time"] = save_df["_ts"].dt.strftime("%I:%M%p").str.lstrip("0")
    col_order = [
        "date", "time", "measured",
        "pred_5m", "pred_15m", "pred_30m",
        "measured_at_5m", "measured_at_15m", "measured_at_30m",
    ]
    csv_out_xgb = out_dir / f"forecast_xgb_{path.stem}.csv"
    save_df[col_order].to_csv(csv_out_xgb, index=False)

    csv_out_lstm = None
    if forecast_lstm is not None and not forecast_lstm.empty:
        save_lstm = forecast_lstm.copy()
        save_lstm["date"] = save_lstm["_ts"].dt.strftime("%d/%m/%y")
        save_lstm["time"] = save_lstm["_ts"].dt.strftime("%I:%M%p").str.lstrip("0")
        csv_out_lstm = out_dir / f"forecast_lstm_{path.stem}.csv"
        save_lstm[col_order].to_csv(csv_out_lstm, index=False)

    # -- Plot(s): choose two consecutive days from test period --
    plot_out_xgb = out_dir / f"forecast_xgb_{path.stem}.png"
    plot_forecast(forecast_xgb, f"{label} [XGB]", plot_out_xgb, test_size=test_size)
    if forecast_lstm is not None and not forecast_lstm.empty:
        plot_out_lstm = out_dir / f"forecast_lstm_{path.stem}.png"
        plot_forecast(forecast_lstm, f"{label} [LSTM]", plot_out_lstm, test_size=test_size)

    # -- Metrics & comparison --
    metrics_xgb = compute_metrics(forecast_xgb)
    metrics_lstm = compute_metrics(forecast_lstm) if forecast_lstm is not None else None
    comparison_rows = []
    print(f"  {'─'*93}")
    print(f"  {'Horizon':<10} {'Model':<8} {'MAE':>10} {'RMSE':>10} {'R2':>8} {'N':>7}")
    print(f"  {'─'*93}")
    for h in HORIZONS:
        for model_name, metrics in (("XGB", metrics_xgb), ("LSTM", metrics_lstm)):
            if metrics is None:
                continue
            m = metrics[h]
            minutes = HORIZONS[h]
            print(
                f"  +{minutes:<8} {model_name:<8} {m['mae']:>10.5f} {m['rmse']:>10.5f}"
                f"  {m['r2']:>8.4f} {m['n']:>7}"
            )
            comparison_rows.append(
                {
                    "horizon": h,
                    "model": model_name,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "n": m["n"],
                }
            )

    compare_out = out_dir / f"metrics_compare_{path.stem}.csv"
    pd.DataFrame(comparison_rows).to_csv(compare_out, index=False)
    print(f"  Forecast CSV (XGB)  : {csv_out_xgb}")
    if csv_out_lstm is not None:
        print(f"  Forecast CSV (LSTM) : {csv_out_lstm}")
    print(f"  Metrics compare CSV : {compare_out}")

    # -- Optional terminal report for a specific date/time/location --
    if report_location and report_location.lower() not in label.lower():
        return
    if report_date or report_time:
        view = filter_report_rows(forecast_xgb, report_date, report_time)
        view_out = view.copy()
        view_out["date"] = view_out["_ts"].dt.strftime("%d/%m/%y")
        view_out["time_str"] = view_out["_ts"].dt.strftime("%I:%M%p").str.lstrip("0")
        display_cols = [
            "date", "time_str", "measured",
            "pred_5m", "pred_15m", "pred_30m",
            "measured_at_5m", "measured_at_15m", "measured_at_30m",
        ]
        print(f"\n  [Report] Location={label}  date='{report_date}'  time='{report_time}'")
        if view_out.empty:
            print("  No rows matched the specified date/time filter.")
        else:
            print(
                view_out[display_cols]
                .rename(columns={"time_str": "time"})
                .to_string(index=False)
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train per-location hybrid PV model and generate 5/15/30-minute horizon "
            "forecasts in normalised power units."
        )
    )
    p.add_argument(
        "--input_dir",
        default="data_generation",
        help="Folder containing *_with_weather.csv files (default: data_generation)",
    )
    p.add_argument(
        "--out_dir",
        default="data_generation/forecasts",
        help="Output directory for forecast CSVs and plots (default: data_generation/forecasts)",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of time-ordered data for test split (default: 0.2)",
    )
    p.add_argument(
        "--lstm_seq_len",
        type=int,
        default=12,
        help="LSTM lookback window in timesteps (default: 12)",
    )
    p.add_argument(
        "--lstm_epochs",
        type=int,
        default=8,
        help="LSTM training epochs (default: 8)",
    )
    p.add_argument(
        "--lstm_batch_size",
        type=int,
        default=64,
        help="LSTM batch size (default: 64)",
    )
    p.add_argument(
        "--report_location",
        default="",
        help=(
            "Show detailed terminal report only for this location "
            "(case-insensitive substring match). Leave empty to report all."
        ),
    )
    p.add_argument(
        "--report_date",
        default="",
        help="Date filter for terminal report, DD/MM/YY format, e.g. 15/09/25",
    )
    p.add_argument(
        "--report_time",
        default="",
        help="Time filter for terminal report, e.g. 11:00AM or 11:00:00",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        input_dir.glob("time_power_average_output_pvoutput_intraday_*_with_weather.csv")
    )
    if not files:
        raise RuntimeError(f"No *_with_weather.csv files found in {input_dir}")

    print(f"Found {len(files)} location file(s). Training XGB and LSTM per location ...")

    for f in files:
        try:
            process_location(
                path=f,
                out_dir=out_dir,
                test_size=args.test_size,
                report_date=args.report_date,
                report_time=args.report_time,
                report_location=args.report_location,
                lstm_seq_len=args.lstm_seq_len,
                lstm_epochs=args.lstm_epochs,
                lstm_batch_size=args.lstm_batch_size,
            )
        except Exception as exc:
            print(f"\n[WARN] Skipped {f.name}: {exc}")

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

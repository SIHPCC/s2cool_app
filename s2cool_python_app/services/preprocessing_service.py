"""Preprocessing and feature engineering service for the S2Cool research dashboard.

This module provides pure functions used by the Data Preprocessing and Feature
Engineering section under Data Analysis. Functions are deliberately free of
Dash imports so they can be unit-tested and reused outside the UI.

Pipeline order (deterministic):
    cleaning -> time features -> solar features -> weather features
    -> feature selection summary -> forecast target -> temporal split
    -> scaling -> quality report -> export payloads
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from services.config_service import REPO_ROOT
from services.dataset_service import (
    PV_DATA_DIR,
    _try_parse_ts,
    parse_system_dataset_filename,
)

PREPROCESSING_OUTPUT_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "preprocessing"

SOLAR_CONSTANT = 1367.0  # W/m2

DEFAULT_LAG_COLUMNS = ["power_average_w_normalized", "ghi_pyr", "air_temperature"]

DEFAULT_CONFIG = {
    "cleaning": {
        "missing_strategy": "interpolate",
        "outlier_method": "noop",
        "outlier_threshold": 3.0,
        "outlier_policy": "clip",
        "outlier_columns": [],
        "remove_duplicates": True,
        "filter_night": True,
        "night_ghi_threshold": 10.0,
    },
    "time_features": {
        "cyclic": True,
        "lag_shifts": [],
        "lag_columns": [],
        "rolling_windows": [],
        "rolling_columns": [],
        "rolling_stats": [],
        "diff_periods": [],
        "diff_columns": [],
    },
    "solar_features": {
        "solar_position": True,
        "clear_sky_ghi": True,
        "clearness_index": True,
        "irradiance_ratios": True,
        "performance_ratio": True,
    },
    "weather_features": {
        "interactions": True,
        "ema_spans": [],
        "ema_columns": [],
        "ramp_periods": [],
        "ramp_columns": [],
    },
    "feature_selection": {
        "target_column": "power_average_w_normalized",
        "strategy": "all",
        "top_k": 10,
        "min_abs_correlation": 0.2,
        "manual_features": [],
    },
    "dataset_builder": {
        "horizon_steps": [1, 6, 12],
        "train_fraction": 0.7,
        "val_fraction": 0.15,
        "scaling_method": "minmax",
    },
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _normalize_config(config: dict | None) -> dict:
    """Merge a partial config over the default config (nested)."""
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(config, dict):
        return merged
    for section, values in config.items():
        if isinstance(values, dict) and section in merged:
            merged[section].update(values)
        elif section in merged:
            merged[section] = values
    return merged


def hash_config(config: dict | None) -> str:
    """Return a stable short hash for a config snapshot."""
    canonical = json.dumps(
        _normalize_config(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Source dataset loader (mirrors data_analysis page conventions)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _cached_load_source(file_name: str) -> pd.DataFrame | None:
    """Load and prep the source dataset using repository conventions."""
    path = PV_DATA_DIR / file_name
    if not path.exists():
        return None

    df = pd.read_csv(path)
    ts = _try_parse_ts(df)
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    parsed = parse_system_dataset_filename(path)
    if parsed and "power_average_w_normalized" in df.columns:
        capacity_kw = parsed.get("capacity_kw")
        if capacity_kw:
            df["pv_power_actual_kw"] = (
                pd.to_numeric(df["power_average_w_normalized"], errors="coerce")
                * float(capacity_kw)
            )
    return df


def load_source_dataset(file_name: str) -> pd.DataFrame | None:
    """Return a copy of the prepped source dataset (or None when missing)."""
    df = _cached_load_source(file_name)
    return df.copy() if df is not None else None


# ---------------------------------------------------------------------------
# Cleaning primitives
# ---------------------------------------------------------------------------


def _build_missing_strategy_map(df: pd.DataFrame, cleaning: dict) -> dict:
    """Map every numeric column to the selected missing-value strategy."""
    strategy = cleaning.get("missing_strategy", "interpolate")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return {column: strategy for column in numeric_columns}


def clean_missing_values(df: pd.DataFrame, strategy_map: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Fill or drop missing values per column using the provided strategy map."""
    out = df.copy()
    applied: list[dict] = []
    for column, strategy in strategy_map.items():
        if column not in out.columns:
            continue
        if strategy == "drop":
            out = out.dropna(subset=[column])
        else:
            numeric = pd.to_numeric(out[column], errors="coerce")
            if strategy == "interpolate":
                out[column] = numeric.interpolate(method="linear", limit_direction="both")
            elif strategy == "ffill":
                out[column] = numeric.ffill()
            elif strategy == "zero":
                out[column] = numeric.fillna(0.0)
            elif strategy == "median":
                out[column] = numeric.fillna(numeric.median())
            else:
                continue
        applied.append({"column": column, "strategy": strategy})
    return out.reset_index(drop=True), applied


def remove_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows and return the number removed."""
    before = len(df)
    out = df.drop_duplicates(subset=subset).reset_index(drop=True)
    return out, before - len(out)


def filter_night_rows(df: pd.DataFrame, ghi_column: str = "ghi_pyr", threshold: float = 10.0) -> tuple[pd.DataFrame, int]:
    """Drop rows where GHI is below the night threshold."""
    if ghi_column not in df.columns:
        return df, 0
    mask = pd.to_numeric(df[ghi_column], errors="coerce") < float(threshold)
    mask = mask.fillna(False)
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def detect_outliers(df: pd.DataFrame, columns: list[str], method: str = "zscore", threshold: float = 3.0) -> tuple[pd.Series, list[dict]]:
    """Return a boolean mask marking outlier rows plus a summary per column.

    The mask is the union across all requested columns. No data is mutated.
    """
    mask = pd.Series(False, index=df.index)
    summary: list[dict] = []
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() < 2:
            continue
        if method == "zscore":
            mean = series.mean()
            std = series.std(ddof=0)
            if pd.isna(std) or std == 0:
                row_mask = pd.Series(False, index=df.index)
            else:
                row_mask = (series - mean).abs() > (float(threshold) * std)
        elif method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - float(threshold) * iqr
            upper = q3 + float(threshold) * iqr
            row_mask = (series < lower) | (series > upper)
        else:
            continue
        row_mask = row_mask.fillna(False)
        mask = mask | row_mask
        summary.append(
            {
                "column": column,
                "method": method,
                "threshold": float(threshold),
                "flagged": int(row_mask.sum()),
            }
        )
    return mask, summary


def apply_outlier_policy(df: pd.DataFrame, mask: pd.Series, policy: str = "clip") -> tuple[pd.DataFrame, int]:
    """Apply the selected outlier policy. 'drop' removes flagged rows; 'clip' winsorizes."""
    out = df.copy()
    if policy == "drop":
        removed = int(mask.sum())
        return out.loc[~mask].reset_index(drop=True), removed
    if policy == "clip":
        # Winsorize every numeric column to the 1st/99th percentile (covers flagged rows).
        for column in out.select_dtypes(include=[np.number]).columns:
            series = pd.to_numeric(out[column], errors="coerce")
            if series.notna().sum() < 2:
                continue
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            if pd.isna(lower) or pd.isna(upper):
                continue
            out[column] = series.clip(lower=lower, upper=upper)
        return out, 0
    return out, 0


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame, ts_column: str = "_ts") -> pd.DataFrame:
    """Add cyclic calendar encodings derived from the timestamp column."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_column], errors="coerce")
    hour = ts.dt.hour
    dayofweek = ts.dt.dayofweek
    month = ts.dt.month
    dayofyear = ts.dt.dayofyear

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dayofweek_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    out["dayofweek_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    out["dayofyear_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.0)
    out["dayofyear_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.0)
    return out


def add_lag_features(df: pd.DataFrame, columns: list[str], shifts: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Add lagged copies of selected columns."""
    out = df.copy()
    added: list[str] = []
    for column in columns:
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        for shift in shifts:
            name = f"{column}_lag{int(shift)}"
            out[name] = series.shift(int(shift))
            added.append(name)
    return out, added


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int], stats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling-window statistics for selected columns."""
    out = df.copy()
    added: list[str] = []
    for column in columns:
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        for window in windows:
            for stat in stats:
                name = f"{column}_rolling{int(window)}_{stat}"
                if stat == "mean":
                    out[name] = series.rolling(window=int(window), min_periods=1).mean()
                elif stat == "std":
                    out[name] = series.rolling(window=int(window), min_periods=1).std()
                elif stat == "min":
                    out[name] = series.rolling(window=int(window), min_periods=1).min()
                elif stat == "max":
                    out[name] = series.rolling(window=int(window), min_periods=1).max()
                else:
                    continue
                added.append(name)
    return out, added


def add_difference_features(df: pd.DataFrame, columns: list[str], periods: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Add first/period differences for selected columns."""
    out = df.copy()
    added: list[str] = []
    for column in columns:
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        for period in periods:
            name = f"{column}_diff{int(period)}"
            out[name] = series.diff(periods=int(period))
            added.append(name)
    return out, added


# ---------------------------------------------------------------------------
# Solar features (numpy-only geometry, no pvlib dependency)
# ---------------------------------------------------------------------------


def _solar_declination_rad(day_of_year: np.ndarray) -> np.ndarray:
    """Solar declination in radians (Cooper approximation)."""
    return np.radians(23.44) * np.sin(np.radians((360.0 / 365.0) * (day_of_year - 81)))


def _equation_of_time_minutes(day_of_year: np.ndarray) -> np.ndarray:
    """Equation of time in minutes."""
    b = np.radians((360.0 / 365.0) * (day_of_year - 81))
    return 9.87 * np.sin(2.0 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


def compute_solar_position(ts, lat: float, lon: float) -> pd.DataFrame:
    """Return solar elevation, zenith and azimuth (degrees) aligned to `ts.index`.

    Uses a NOAA-style approximation intended for feature engineering and
    clearness-ratio watermarking — not for astronomy-grade precision.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour_decimal = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0

    lat_rad = np.radians(float(lat))
    decl_rad = _solar_declination_rad(day_of_year)
    eot_min = _equation_of_time_minutes(day_of_year)

    # Hour angle in degrees (local solar time offset via equation of time).
    hour_angle = 15.0 * (hour_decimal - 12.0) + eot_min / 4.0
    hra_rad = np.radians(hour_angle)

    sin_elev = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hra_rad)
    )
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    elevation = np.degrees(np.arcsin(sin_elev))
    zenith = 90.0 - elevation

    # Azimuth (NOAA formulation).
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = np.sin(decl_rad) * np.cos(lat_rad) - np.cos(decl_rad) * np.sin(lat_rad) * np.cos(hra_rad)
        denominator = np.cos(np.radians(elevation)) * np.cos(lat_rad)
        ratio = numerator / denominator
        ratio = np.clip(ratio, -1.0, 1.0)
        cos_az = np.where(np.cos(np.radians(zenith)) > 0.01, ratio, 1.0)
    azimuth = np.degrees(np.arccos(cos_az))
    azimuth = np.where(hra_rad > 0, 360.0 - azimuth, azimuth)

    return pd.DataFrame(
        {
            "solar_elevation_deg": elevation,
            "solar_zenith_deg": zenith,
            "solar_azimuth_deg": azimuth,
        },
        index=ts.index,
    )


def compute_clear_sky_ghi(ts, lat: float, lon: float) -> pd.Series:
    """Estimate clear-sky GHI (W/m2) using the Haurwitz model."""
    ts = pd.to_datetime(ts, errors="coerce")
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour_decimal = ts.dt.hour.to_numpy(dtype=float) + ts.dt.minute.to_numpy(dtype=float) / 60.0

    lat_rad = np.radians(float(lat))
    decl_rad = _solar_declination_rad(day_of_year)
    eot_min = _equation_of_time_minutes(day_of_year)
    hour_angle = 15.0 * (hour_decimal - 12.0) + eot_min / 4.0
    hra_rad = np.radians(hour_angle)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hra_rad)
    )
    cos_zenith = np.clip(cos_zenith, 0.0, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ghi_clear = 1098.0 * cos_zenith * np.exp(-0.059 / np.where(cos_zenith > 0.001, cos_zenith, np.nan))
    ghi_clear = np.where(cos_zenith > 0.001, ghi_clear, 0.0)

    return pd.Series(ghi_clear, index=ts.index, name="clear_sky_ghi_wm2")


def add_solar_blocks(
    df: pd.DataFrame,
    config: dict,
    lat: float,
    lon: float,
    capacity_kw: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add solar-derived feature columns according to the config block."""
    out = df.copy()
    added: list[str] = []
    ts = pd.to_datetime(out["_ts"], errors="coerce")

    if config.get("solar_position", False):
        geometry = compute_solar_position(ts, lat, lon)
        for column in geometry.columns:
            out[column] = geometry[column].to_numpy()
            added.append(column)

    if config.get("clear_sky_ghi", False):
        out["clear_sky_ghi_wm2"] = compute_clear_sky_ghi(ts, lat, lon).to_numpy()
        added.append("clear_sky_ghi_wm2")
        out["clear_sky_pv_power_normalized"] = out["clear_sky_ghi_wm2"] / 1000.0
        added.append("clear_sky_pv_power_normalized")
        if capacity_kw is not None:
            out["clear_sky_pv_power_actual_kw"] = out["clear_sky_pv_power_normalized"] * float(capacity_kw)
            added.append("clear_sky_pv_power_actual_kw")

    if config.get("clearness_index", False) and "ghi_pyr" in out.columns and "clear_sky_ghi_wm2" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["clearness_index"] = (out["ghi_pyr"] / out["clear_sky_ghi_wm2"]).replace([np.inf, -np.inf], np.nan)
        added.append("clearness_index")

    if config.get("irradiance_ratios", False):
        if "dni" in out.columns and "ghi_pyr" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["dni_ghi_ratio"] = (out["dni"] / out["ghi_pyr"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            added.append("dni_ghi_ratio")
        if "dhi" in out.columns and "ghi_pyr" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["dhi_ghi_ratio"] = (out["dhi"] / out["ghi_pyr"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            added.append("dhi_ghi_ratio")

    if config.get("performance_ratio", False):
        if "power_average_w_normalized" in out.columns and "ghi_pyr" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                ghi_kw = out["ghi_pyr"] / 1000.0
                out["performance_ratio"] = (out["power_average_w_normalized"] / ghi_kw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            added.append("performance_ratio")

    return out, added


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------


def add_weather_blocks(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add weather interactions, EMA smooths and ramp rates."""
    out = df.copy()
    added: list[str] = []
    ghi_col = "ghi_pyr"
    temp_col = "air_temperature"
    rh_col = "relative_humidity"

    if config.get("interactions", False):
        if temp_col in out.columns and ghi_col in out.columns:
            out["temp_ghi_interaction"] = pd.to_numeric(out[temp_col], errors="coerce") * pd.to_numeric(out[ghi_col], errors="coerce")
            added.append("temp_ghi_interaction")
        if rh_col in out.columns and ghi_col in out.columns:
            out["rh_ghi_interaction"] = pd.to_numeric(out[rh_col], errors="coerce") * pd.to_numeric(out[ghi_col], errors="coerce")
            added.append("rh_ghi_interaction")
        if temp_col in out.columns and rh_col in out.columns:
            out["temp_rh_interaction"] = pd.to_numeric(out[temp_col], errors="coerce") * pd.to_numeric(out[rh_col], errors="coerce")
            added.append("temp_rh_interaction")

    for column in (config.get("ema_columns") or []):
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        for span in (config.get("ema_spans") or []):
            name = f"{column}_ema{int(span)}"
            out[name] = series.ewm(span=int(span), adjust=False).mean()
            added.append(name)

    for column in (config.get("ramp_columns") or []):
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        for period in (config.get("ramp_periods") or []):
            name = f"{column}_ramp{int(period)}"
            out[name] = series.diff(periods=int(period))
            added.append(name)

    return out, added


# ---------------------------------------------------------------------------
# Feature selection summaries
# ---------------------------------------------------------------------------


def build_feature_summary(df: pd.DataFrame, target_column: str) -> list[dict]:
    """Return per-numeric-column summary sorted by |correlation| desc."""
    summary: list[dict] = []
    if target_column not in df.columns:
        return summary
    numeric = df.select_dtypes(include=[np.number])
    target_series = pd.to_numeric(df[target_column], errors="coerce")
    for column in numeric.columns:
        if column == target_column:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        corr = series.corr(target_series)
        variance = float(series.var()) if series.notna().sum() > 1 else 0.0
        summary.append(
            {
                "column": column,
                "correlation": round(float(corr), 4) if pd.notna(corr) else None,
                "variance": round(variance, 6),
                "missing_pct": round(float(series.isna().mean() * 100.0), 2),
                "cardinality": int(series.nunique()),
            }
        )
    summary.sort(
        key=lambda row: (row["correlation"] is None, -abs(row["correlation"] or 0.0))
    )
    return summary


def try_mutual_info_topk(df: pd.DataFrame, target_column: str, top_k: int = 10) -> tuple[bool, list[dict] | None]:
    """Return (available, rows). Never raises when sklearn is absent.

    When sklearn is not installed or unavailable, returns ``(False, None)``
    so the UI can show a muted notice instead of blocking the page.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        return False, None

    if target_column not in df.columns:
        return True, []
    numeric = df.select_dtypes(include=[np.number]).dropna()
    features = [column for column in numeric.columns if column != target_column]
    if numeric.empty or not features:
        return True, []
    X = numeric[features].fillna(0.0)
    target = pd.to_numeric(df.loc[numeric.index, target_column], errors="coerce").fillna(0.0)
    if X.shape[0] < 5 or X.shape[1] < 1:
        return True, []
    try:
        mi = mutual_info_regression(X, target, random_state=42)
    except Exception:
        return False, None
    rows = sorted(zip(features, mi), key=lambda item: item[1], reverse=True)
    return True, [
        {"column": column, "mi": round(float(value), 6)}
        for column, value in rows[: max(int(top_k), 0)]
    ]


# ---------------------------------------------------------------------------
# Dataset builder primitives
# ---------------------------------------------------------------------------


def create_forecast_target(df: pd.DataFrame, power_column: str, horizon_steps: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Create shifted target columns ``target_{h}`` for each forecast horizon step."""
    out = df.copy()
    added: list[str] = []
    if power_column not in out.columns:
        return out, added
    series = pd.to_numeric(out[power_column], errors="coerce")
    for step in horizon_steps:
        name = f"target_{int(step)}"
        out[name] = series.shift(-int(step))
        added.append(name)
    return out, added


def split_temporal(row_count: int, train_fraction: float = 0.7, val_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train, val, test) index arrays from a chronological split."""
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")
    if not (0.0 <= float(val_fraction) < 1.0 - float(train_fraction)):
        raise ValueError("val_fraction must be non-negative and less than 1 - train_fraction.")
    n = int(row_count)
    train_end = int(round(n * float(train_fraction)))
    val_end = int(round(n * (float(train_fraction) + float(val_fraction))))
    return (
        np.arange(0, train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n),
    )


def scale_features(df: pd.DataFrame, feature_columns: list[str], method: str = "minmax") -> tuple[pd.DataFrame, list[dict]]:
    """Scale selected feature columns into ``{column}_scaled`` columns.

    Hand-rolled numpy implementations of minmax/standard scaling so the app
    has no hard dependency on scikit-learn.
    """
    out = df.copy()
    scaler_params: list[dict] = []
    if method == "none":
        return out, scaler_params
    for column in feature_columns:
        if column not in out.columns:
            continue
        series = pd.to_numeric(out[column], errors="coerce")
        raw = series.to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        if finite.size < 2:
            continue
        if method == "minmax":
            lo, hi = float(finite.min()), float(finite.max())
            scaled = (raw - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(raw)
            params = {"method": "minmax", "min": lo, "max": hi}
        elif method == "standard":
            mean, std = float(finite.mean()), float(finite.std(ddof=0))
            scaled = (raw - mean) / std if std > 1e-12 else np.zeros_like(raw)
            params = {"method": "standard", "mean": mean, "std": std}
        else:
            continue
        scaled = np.where(np.isnan(raw), np.nan, scaled)
        out[f"{column}_scaled"] = scaled
        params["source_column"] = column
        params["output_column"] = f"{column}_scaled"
        scaler_params.append(params)
    return out, scaler_params


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------


def build_quality_report(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    log: list[dict],
    split_info: dict | None = None,
    scaler_params: list[dict] | None = None,
) -> dict:
    """Build a JSON-serializable data quality report."""
    original_rows = len(original_df)
    final_rows = len(result_df)
    original_columns = list(original_df.columns)
    final_columns = list(result_df.columns)
    removed_columns = [column for column in original_columns if column not in final_columns]
    added_columns = [column for column in final_columns if column not in original_columns]

    if "_ts" in result_df.columns:
        ts = pd.to_datetime(result_df["_ts"], errors="coerce")
    else:
        ts = pd.Series(dtype="datetime64[ns]")
    ts_clean = ts.dropna()

    interval_min = None
    if len(ts_clean) > 1:
        delta = ts_clean.diff().dropna().dt.total_seconds() / 60.0
        if not delta.empty:
            interval_min = float(delta.median())

    gap_count = 0
    if interval_min and len(ts_clean) > 1:
        delta = ts_clean.diff().dropna().dt.total_seconds() / 60.0
        gap_count = int((delta > interval_min * 1.5).sum())

    missingness: dict[str, float] = {}
    for column in final_columns:
        if column == "_ts":
            missingness[column] = round(float(ts.isna().mean() * 100.0), 2)
        else:
            missingness[column] = round(float(pd.to_numeric(result_df[column], errors="coerce").isna().mean() * 100.0), 2)

    monotonic = bool(ts_clean.is_monotonic_increasing) if len(ts_clean) > 1 else True

    return {
        "original_rows": int(original_rows),
        "final_rows": int(final_rows),
        "original_columns": int(len(original_columns)),
        "final_columns": int(len(final_columns)),
        "removed_columns": removed_columns,
        "added_columns": added_columns,
        "row_delta": int(final_rows - original_rows),
        "column_delta": int(len(final_columns) - len(original_columns)),
        "timestamp": {
            "start": ts_clean.min().strftime("%Y-%m-%d %H:%M:%S") if not ts_clean.empty else None,
            "end": ts_clean.max().strftime("%Y-%m-%d %H:%M:%S") if not ts_clean.empty else None,
            "rows": int(len(ts_clean)),
            "monotonic": monotonic,
            "median_interval_minutes": interval_min,
            "gap_count_gt_1_5x": gap_count,
        },
        "missingness_percent": missingness,
        "pipeline_log": log,
        "splits": split_info or {},
        "scaler_summary": scaler_params or [],
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _build_metadata(file_name: str, source_df: pd.DataFrame, config: dict, quality_report: dict) -> dict:
    """Build JSON-serializable export metadata."""
    parsed = parse_system_dataset_filename(PV_DATA_DIR / file_name)
    stem = parsed["file_name"].replace(".csv", "") if parsed else Path(file_name).stem
    normalized_config = _normalize_config(config)
    horizon_steps = [int(v) for v in normalized_config["dataset_builder"].get("horizon_steps", [])]
    interval_minutes = quality_report.get("timestamp", {}).get("median_interval_minutes")
    horizons_minutes = [
        round(step * float(interval_minutes), 4)
        for step in horizon_steps
    ] if interval_minutes else horizon_steps
    selected_features = list(quality_report.get("selected_features", []))
    return {
        "source_file": file_name,
        "stem": stem,
        "system_id": parsed["system_id"] if parsed else None,
        "city": parsed["city"] if parsed else None,
        "capacity_kw": parsed["capacity_kw"] if parsed else None,
        "lat": parsed["lat"] if parsed else None,
        "lon": parsed["lon"] if parsed else None,
        "config": normalized_config,
        "config_hash": hash_config(normalized_config),
        "artifact_version": 1,
        "target_column": normalized_config["feature_selection"].get(
            "target_column", "power_average_w_normalized"
        ),
        "feature_columns": selected_features,
        "horizon_steps": horizon_steps,
        "horizons_minutes": horizons_minutes,
        "sampling_interval_minutes": interval_minutes,
        "split_boundaries": quality_report.get("splits", {}),
        "quality_summary": {
            "rows_before": quality_report["original_rows"],
            "rows_after": quality_report["final_rows"],
            "columns_before": quality_report["original_columns"],
            "columns_after": quality_report["final_columns"],
            "start_time": quality_report["timestamp"]["start"],
            "end_time": quality_report["timestamp"]["end"],
        },
    }


def build_export_payload(df: pd.DataFrame, metadata: dict, output_dir: Path | str | None = None) -> dict:
    """Write processed CSV and quality JSON; return artifact information."""
    output_dir = Path(output_dir) if output_dir else PREPROCESSING_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = metadata.get("stem", "processed")
    csv_path = output_dir / f"{stem}_processed_{stamp}.csv"
    json_path = output_dir / f"{stem}_quality_{stamp}.json"

    export_df = df.copy()
    if "_ts" in export_df.columns:
        export_df["_ts"] = pd.to_datetime(export_df["_ts"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    export_df.to_csv(csv_path, index=False)

    payload = dict(metadata)
    payload["exported_at"] = stamp
    payload["file"] = csv_path.name
    payload["artifact_id"] = f"{payload.get('stem', 'processed')}-{stamp}-{payload.get('config_hash', 'unknown')}"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "csv_name": csv_path.name,
        "json_name": json_path.name,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Immutable-style result container returned by the pipeline."""

    dataframe: pd.DataFrame = field(repr=False)
    config: dict
    log: list[dict]
    quality_report: dict
    feature_summary: list[dict]
    mutual_info: dict
    selected_feature_columns: list[str]
    split_indices: dict
    scaler_params: list[dict]
    metadata: dict


def _default_lag_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in DEFAULT_LAG_COLUMNS if column in df.columns]
    if columns:
        return columns
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _feature_columns_for_scaling(df: pd.DataFrame, target_column: str, horizon_steps: list[int]) -> list[str]:
    """Numeric modeling features excluding timestamps, targets and scaled outputs."""
    excluded = {"_ts", "date", "time", target_column}
    excluded.update(f"target_{int(h)}" for h in (horizon_steps or []))
    columns = []
    for column in df.select_dtypes(include=[np.number]).columns:
        if column in excluded or column.endswith("_scaled"):
            continue
        columns.append(column)
    return columns


def _select_feature_columns(
    candidate_columns: list[str],
    feature_summary: list[dict],
    selection_cfg: dict,
) -> list[str]:
    """Return the modeling feature columns selected by the active strategy."""
    strategy = selection_cfg.get("strategy", "all") or "all"
    top_k = max(int(selection_cfg.get("top_k", 10) or 0), 0)
    min_abs_corr = float(selection_cfg.get("min_abs_correlation", 0.2) or 0.0)
    manual_features = selection_cfg.get("manual_features") or []

    candidate_set = set(candidate_columns)
    ranked_by_corr = [
        row["column"]
        for row in feature_summary
        if row["column"] in candidate_set
    ]

    if strategy == "top_k_corr":
        chosen = set(ranked_by_corr[:top_k])
    elif strategy == "corr_threshold":
        chosen = {
            row["column"]
            for row in feature_summary
            if row["column"] in candidate_set
            and row["correlation"] is not None
            and abs(row["correlation"]) >= min_abs_corr
        }
    elif strategy == "manual":
        chosen = {column for column in manual_features if column in candidate_set}
    else:
        chosen = candidate_set

    return [column for column in candidate_columns if column in chosen]


def _retain_selected_features(
    df: pd.DataFrame,
    target_column: str,
    horizon_steps: list[int],
    selected_features: list[str],
) -> pd.DataFrame:
    """Reduce the modeling dataframe to timestamps, target, forecast targets and chosen features."""
    keep = {"_ts", "date", "time", target_column}
    keep.update(f"target_{int(h)}" for h in (horizon_steps or []))
    keep.update(selected_features)
    columns = [column for column in df.columns if column in keep]
    return df.loc[:, columns].copy()


def run_preprocessing_pipeline(file_name: str, config: dict | None) -> PipelineResult:
    """Run the full deterministic preprocessing pipeline on a source dataset."""
    config = _normalize_config(config)
    source = load_source_dataset(file_name)
    if source is None:
        raise ValueError(f"Dataset not available: {file_name}")

    log: list[dict] = []
    df = source.copy()

    # --- 1) Cleaning ---
    cleaning = config["cleaning"]
    strategy_map = _build_missing_strategy_map(df, cleaning)
    df, applied = clean_missing_values(df, strategy_map)
    log.append({"step": "clean_missing", "strategies": applied})

    if cleaning.get("remove_duplicates", False):
        df, dropped = remove_duplicate_rows(df)
        log.append({"step": "remove_duplicates", "dropped": dropped})

    if cleaning.get("filter_night", False):
        night_threshold = float(cleaning.get("night_ghi_threshold", 10.0))
        df, dropped = filter_night_rows(df, "ghi_pyr", night_threshold)
        log.append({"step": "filter_night", "dropped": dropped, "threshold": night_threshold})

    outlier_columns = cleaning.get("outlier_columns") or []
    outlier_method = cleaning.get("outlier_method", "noop")
    if outlier_method != "noop" and outlier_columns:
        outlier_threshold = float(cleaning.get("outlier_threshold", 3.0))
        mask, outlier_summary = detect_outliers(
            df, outlier_columns, method=outlier_method, threshold=outlier_threshold
        )
        log.append(
            {
                "step": "detect_outliers",
                "summary": outlier_summary,
                "flagged_total": int(mask.sum()),
            }
        )
        if int(mask.sum()) > 0:
            policy = cleaning.get("outlier_policy", "clip")
            df, affected = apply_outlier_policy(df, mask, policy=policy)
            log.append({"step": "apply_outlier_policy", "policy": policy, "affected_rows": affected})

    # --- 2) Time features ---
    time_cfg = config["time_features"]
    if time_cfg.get("cyclic", False):
        df = add_time_features(df)
        log.append({"step": "time_cyclic", "columns": 8})

    if time_cfg.get("lag_shifts"):
        columns = time_cfg.get("lag_columns") or _default_lag_columns(df)
        df, added = add_lag_features(df, columns, time_cfg["lag_shifts"])
        log.append({"step": "time_lag", "columns": added})

    if time_cfg.get("rolling_windows") and time_cfg.get("rolling_stats"):
        columns = time_cfg.get("rolling_columns") or _default_lag_columns(df)
        df, added = add_rolling_features(
            df, columns, time_cfg["rolling_windows"], time_cfg["rolling_stats"]
        )
        log.append({"step": "time_rolling", "columns": added})

    if time_cfg.get("diff_periods"):
        columns = time_cfg.get("diff_columns") or _default_lag_columns(df)
        df, added = add_difference_features(df, columns, time_cfg["diff_periods"])
        log.append({"step": "time_diff", "columns": added})

    # --- 3) Solar features ---
    solar_cfg = config["solar_features"]
    parsed = parse_system_dataset_filename(PV_DATA_DIR / file_name)
    lat = float(parsed["lat"]) if parsed else 0.0
    lon = float(parsed["lon"]) if parsed else 0.0
    capacity_kw = float(parsed["capacity_kw"]) if parsed and parsed.get("capacity_kw") is not None else None
    if any(solar_cfg.values()):
        df, added = add_solar_blocks(df, solar_cfg, lat, lon, capacity_kw)
        log.append({"step": "solar_features", "columns": added, "lat": lat, "lon": lon})

    # --- 4) Weather features ---
    weather_cfg = config["weather_features"]
    if any(weather_cfg.values()):
        df, added = add_weather_blocks(df, weather_cfg)
        log.append({"step": "weather_features", "columns": added})

    # --- 5) Feature selection summary and active feature set ---
    target_column = config["feature_selection"].get("target_column", "power_average_w_normalized")
    feature_summary = build_feature_summary(df, target_column)
    mi_available, mi_rows = try_mutual_info_topk(
        df, target_column, config["feature_selection"].get("top_k", 10)
    )
    candidate_features = _feature_columns_for_scaling(df, target_column, [])
    selected_feature_columns = _select_feature_columns(
        candidate_features,
        feature_summary,
        config["feature_selection"],
    )
    log.append(
        {
            "step": "feature_selection",
            "strategy": config["feature_selection"].get("strategy", "all"),
            "candidate_count": len(candidate_features),
            "selected_count": len(selected_feature_columns),
            "selected_columns": selected_feature_columns,
        }
    )

    # --- 6) Forecast target ---
    builder_cfg = config["dataset_builder"]
    horizon_steps = builder_cfg.get("horizon_steps") or []
    if horizon_steps:
        df, added = create_forecast_target(df, target_column, horizon_steps)
        log.append(
            {"step": "forecast_target", "columns": added, "horizon_steps": horizon_steps}
        )

    df = _retain_selected_features(df, target_column, horizon_steps, selected_feature_columns)

    # --- 7) Temporal split ---
    train_fraction = float(builder_cfg.get("train_fraction", 0.7))
    val_fraction = float(builder_cfg.get("val_fraction", 0.15))
    train_idx, val_idx, test_idx = split_temporal(len(df), train_fraction, val_fraction)
    split_info = {
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": round(1.0 - train_fraction - val_fraction, 4),
    }
    log.append({"step": "temporal_split", **split_info})

    # --- 8) Scaling ---
    scaling_method = builder_cfg.get("scaling_method", "minmax")
    feature_columns = _feature_columns_for_scaling(df, target_column, horizon_steps)
    df, scaler_params = scale_features(df, feature_columns, method=scaling_method)
    log.append({"step": "scaling", "method": scaling_method, "scaled": len(scaler_params)})

    # --- 9) Quality report ---
    quality_report = build_quality_report(
        source, df, log, split_info=split_info, scaler_params=scaler_params
    )
    quality_report["selected_features"] = list(selected_feature_columns)
    metadata = _build_metadata(file_name, source, config, quality_report)

    return PipelineResult(
        dataframe=df,
        config=config,
        log=log,
        quality_report=quality_report,
        feature_summary=feature_summary,
        mutual_info={"available": mi_available, "rows": mi_rows},
        selected_feature_columns=selected_feature_columns,
        split_indices={
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
        scaler_params=scaler_params,
        metadata=metadata,
    )


@lru_cache(maxsize=16)
def _cached_pipeline(file_name: str, canonical_config: str) -> PipelineResult:
    """Cached pipeline entry; keyed by file name + canonical config JSON."""
    return run_preprocessing_pipeline(file_name, json.loads(canonical_config))


def get_pipeline_result(file_name: str, config: dict | None) -> PipelineResult:
    """Cached public entry point used by callbacks."""
    normalized = _normalize_config(config)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return _cached_pipeline(file_name, canonical)


def export_preprocessing_artifacts(file_name: str, config: dict | None) -> tuple[dict, PipelineResult]:
    """Run the pipeline and write CSV + quality JSON export artifacts."""
    result = get_pipeline_result(file_name, config)
    export_info = build_export_payload(result.dataframe, result.metadata)
    return export_info, result

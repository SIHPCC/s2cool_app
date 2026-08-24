from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from models.domain import DatasetSnapshot
from services.config_service import REPO_ROOT

PV_DATA_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "data"
PV_FORECAST_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "forecast"
COOLING_DIR = REPO_ROOT / "cooling_demand"

REQUIRED_ANALYSIS_COLUMNS = [
    "date",
    "time",
    "power_average_w_normalized",
    "ghi_pyr",
    "dni",
    "dhi",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
]

SYSTEM_FILE_PATTERN = re.compile(
    r"^system(?P<system_id>\d+)_"
    r"(?P<capacity_kw>[\d.]+)kW_"
    r"(?P<city>[^_]+)_"
    r"lat(?P<lat>-?[\d.]+)_"
    r"lon(?P<lon>-?[\d.]+)_"
    r"(?P<name_start>\d{8})_(?P<name_end>\d{8})_weather\.csv$"
)


def _try_parse_ts(df: pd.DataFrame) -> pd.Series:
    if "_ts" in df.columns:
        return pd.to_datetime(df["_ts"], errors="coerce")
    if {"date", "time"}.issubset(df.columns):
        raw = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(raw, format="%d/%m/%y %I:%M%p", errors="coerce")
        mask = parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        mask = parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(raw.loc[mask], dayfirst=True, errors="coerce")
        return parsed
    return pd.Series(dtype="datetime64[ns]")


def list_system_dataset_paths() -> list[Path]:
    if not PV_DATA_DIR.exists():
        return []
    return sorted(PV_DATA_DIR.glob("system*_weather.csv"))


def _system_number_from_name(path: Path) -> int | None:
    match = re.match(r"system(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def _parse_filename_date(date_token: str) -> str:
    return f"{date_token[0:4]}-{date_token[4:6]}-{date_token[6:8]}"


def parse_system_dataset_filename(path: Path) -> dict | None:
    match = SYSTEM_FILE_PATTERN.match(path.name)
    if not match:
        return None
    tokens = match.groupdict()
    return {
        "file_name": path.name,
        "system_id": int(tokens["system_id"]),
        "capacity_kw": float(tokens["capacity_kw"]),
        "city": tokens["city"],
        "lat": float(tokens["lat"]),
        "lon": float(tokens["lon"]),
        "name_start": _parse_filename_date(tokens["name_start"]),
        "name_end": _parse_filename_date(tokens["name_end"]),
    }


def list_available_system_files() -> list[dict]:
    systems: list[dict] = []
    for path in list_system_dataset_paths():
        parsed = parse_system_dataset_filename(path)
        if not parsed:
            continue
        systems.append(parsed)
    return sorted(systems, key=lambda item: item["system_id"])


def build_system_file_options() -> list[dict]:
    options = []
    for item in list_available_system_files():
        label = f"System {item['system_id']:02d} | {item['city']} | {item['capacity_kw']:.3g} kW"
        options.append({"label": label, "value": item["file_name"]})
    return options


@lru_cache(maxsize=128)
def build_system_dataset_profile(file_name: str) -> dict | None:
    path = PV_DATA_DIR / file_name
    if not path.exists():
        return None

    parsed = parse_system_dataset_filename(path)
    if not parsed:
        return None

    df = pd.read_csv(path)
    ts = _try_parse_ts(df).dropna().sort_values().reset_index(drop=True)

    interval_minutes = None
    if len(ts) > 1:
        delta = ts.diff().dropna().dt.total_seconds() / 60.0
        if not delta.empty:
            interval_minutes = float(delta.median())

    observed_start = ts.iloc[0].strftime("%Y-%m-%d %H:%M") if not ts.empty else parsed["name_start"]
    observed_end = ts.iloc[-1].strftime("%Y-%m-%d %H:%M") if not ts.empty else parsed["name_end"]

    return {
        "file_name": parsed["file_name"],
        "system_id": parsed["system_id"],
        "city": parsed["city"],
        "capacity_kw": parsed["capacity_kw"],
        "lat": parsed["lat"],
        "lon": parsed["lon"],
        "data_points": int(df.shape[0]),
        "time_interval_min": interval_minutes,
        "start_time": observed_start,
        "end_time": observed_end,
        "declared_start": parsed["name_start"],
        "declared_end": parsed["name_end"],
    }


def latest_system_dataset_path(system_number: int) -> Path | None:
    candidates = [path for path in list_system_dataset_paths() if _system_number_from_name(path) == system_number]
    return candidates[-1] if candidates else None


def build_dataset_snapshot(system_number: int | None) -> DatasetSnapshot:
    if system_number is None:
        return DatasetSnapshot("not_selected", None, 0, 0, "-", "-", None, 0, 0.0)

    path = latest_system_dataset_path(system_number)
    if path is None:
        return DatasetSnapshot("missing", None, 0, 0, "-", "-", None, 0, 0.0)

    df = pd.read_csv(path)
    ts = _try_parse_ts(df)
    ts = ts.dropna().sort_values()

    median_interval_min = None
    if len(ts) > 1:
        delta = ts.diff().dropna().dt.total_seconds() / 60.0
        if not delta.empty:
            median_interval_min = float(delta.median())

    total_cells = max(df.shape[0] * max(df.shape[1], 1), 1)
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100.0, 2)

    start_ts = ts.iloc[0].strftime("%Y-%m-%d %H:%M") if not ts.empty else "-"
    end_ts = ts.iloc[-1].strftime("%Y-%m-%d %H:%M") if not ts.empty else "-"

    return DatasetSnapshot(
        status="ready",
        path=path,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        start_ts=start_ts,
        end_ts=end_ts,
        median_interval_min=median_interval_min,
        missing_cells=missing_cells,
        missing_pct=missing_pct,
    )


def load_preview_frame(system_number: int | None, max_rows: int = 720) -> pd.DataFrame:
    path = latest_system_dataset_path(system_number) if system_number is not None else None
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["_ts"] = _try_parse_ts(df)
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    return df


def build_missingness_summary(df: pd.DataFrame) -> list[tuple[str, float]]:
    if df.empty:
        return []
    summary = []
    for column in REQUIRED_ANALYSIS_COLUMNS:
        if column not in df.columns:
            summary.append((column, math.nan))
            continue
        summary.append((column, round(float(df[column].isna().mean() * 100.0), 2)))
    return summary


def build_preprocessing_actions(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["Load or register a system dataset before preprocessing."]

    actions: list[str] = []
    if "_ts" not in df.columns:
        actions.append("Create a canonical timestamp column from date and time fields.")
    if df["_ts"].diff().dropna().dt.total_seconds().div(60).median() not in {5.0, 10.0, 15.0}:
        actions.append("Regularize the sampling interval before training or backtesting.")
    if df.isna().sum().sum() > 0:
        actions.append("Define gap handling per variable family: irradiance, weather, and power.")
    missing_columns = [column for column in REQUIRED_ANALYSIS_COLUMNS if column not in df.columns]
    if missing_columns:
        actions.append("Map missing source columns into the standard schema before forecasting.")
    if "power_average_w_normalized" in df.columns:
        actions.append("Preserve normalized power for modeling and scale to kW only for display/export.")
    return actions or ["Dataset already matches the expected baseline schema."]


def count_existing_forecast_outputs() -> int:
    if not PV_FORECAST_DIR.exists():
        return 0
    return len(list(PV_FORECAST_DIR.glob("*.csv")))


def count_existing_cooling_outputs() -> int:
    if not COOLING_DIR.exists():
        return 0
    patterns = ["cooling_hybrid_forecast_*.csv", "cooling_backtest_metrics_*.csv", "cooling_run_summary_*.json"]
    count = 0
    for pattern in patterns:
        count += len(list(COOLING_DIR.glob(pattern)))
    return count

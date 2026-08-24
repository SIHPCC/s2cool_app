"""Service for creating new system datasets from uploaded PV power data.

This service:
1) Validates the uploaded PV CSV (must contain date, time, and PV power columns).
2) Fetches hourly weather from Open-Meteo archive for the PV date range.
3) Aligns hourly weather to PV timestamps via time interpolation.
4) Normalizes PV power to `power_average_w_normalized` using system capacity.
5) Writes output in the same schema used by existing system weather files.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from services.config_service import REPO_ROOT
from services.dataset_service import PV_DATA_DIR

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Asia/Karachi"

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]

# Common PV power column names that may appear in uploaded files
PV_POWER_COLUMN_ALIASES = [
    "power_w",
    "power",
    "pv_power",
    "pv_power_w",
    "power_kw",
    "pv_power_kw",
    "pac",
    "active_power",
    "power_output",
    "power_output_w",
    "power_average_w_normalized",
    "power_average_w",
    "power_normalized",
]

# Common timestamp column names
DATE_COLUMN_ALIASES = ["date", "Date", "DATE", "timestamp", "Timestamp", "datetime", "DateTime"]
TIME_COLUMN_ALIASES = ["time", "Time", "TIME"]


class NewSystemError(RuntimeError):
    """Raised when a new system dataset cannot be created."""


def _resolve_pv_power_column(df: pd.DataFrame) -> str | None:
    """Find the PV power column in the uploaded dataframe."""
    for alias in PV_POWER_COLUMN_ALIASES:
        if alias in df.columns:
            return alias
    # Fallback: look for any column containing 'power' or 'w' in the name
    for column in df.columns:
        lowered = str(column).lower()
        if "power" in lowered or lowered in {"w", "watts", "pv"}:
            return column
    return None


def _resolve_date_column(df: pd.DataFrame) -> str | None:
    """Find the date column in the uploaded dataframe."""
    for alias in DATE_COLUMN_ALIASES:
        if alias in df.columns:
            return alias
    return None


def _resolve_time_column(df: pd.DataFrame) -> str | None:
    """Find the time column in the uploaded dataframe."""
    for alias in TIME_COLUMN_ALIASES:
        if alias in df.columns:
            return alias
    return None


def _parse_uploaded_timestamps(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    """Parse uploaded date/time columns into a datetime series.

    Some uploads include a full datetime string in the `time` column itself.
    In that case, parse from `time` directly instead of concatenating `date` + `time`.
    """
    date_text = df[date_col].astype(str).str.strip()
    time_text = df[time_col].astype(str).str.strip()

    # If the time field already contains a full date-time token, parse it directly.
    has_embedded_date = time_text.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True, na=False)
    parsed = pd.to_datetime(pd.Series(pd.NaT, index=df.index), errors="coerce")
    if has_embedded_date.any():
        parsed.loc[has_embedded_date] = pd.to_datetime(
            time_text.loc[has_embedded_date],
            errors="coerce",
            dayfirst=False,
        )

    raw = date_text + " " + time_text
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%d/%m/%y %I:%M%p", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%m/%d/%y %I:%M%p", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%d/%m/%y %H:%M", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%m/%d/%y %H:%M", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format="%d/%m/%Y %H:%M", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], dayfirst=True, errors="coerce")
    return parsed


def _to_local_naive(series: pd.Series, timezone: str) -> pd.Series:
    """Normalize timestamps to timezone-naive local clock values."""
    ts = pd.to_datetime(series, errors="coerce")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        return ts.dt.tz_convert(timezone).dt.tz_localize(None)
    return ts


MAX_WEATHER_RETRIES = 4
WEATHER_RETRY_BASE_SECONDS = 2


def _fetch_openmeteo_weather(
    lat: float,
    lon: float,
    timezone: str,
    start_day: date,
    end_day: date,
) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo archive with retry logic."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_day.strftime("%Y-%m-%d"),
        "end_date": end_day.strftime("%Y-%m-%d"),
        "timezone": timezone,
        "hourly": ",".join(WEATHER_COLUMNS),
    }

    payload = None
    last_error: Exception | None = None
    for attempt in range(MAX_WEATHER_RETRIES + 1):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= MAX_WEATHER_RETRIES:
                break

            import time

            wait_seconds = WEATHER_RETRY_BASE_SECONDS * (2**attempt)
            time.sleep(wait_seconds)

    if payload is None:
        raise NewSystemError(
            "Open-Meteo weather fetch failed after retries. "
            f"Last error: {last_error}"
        )

    hourly = payload.get("hourly") or {}
    required = ["time", *WEATHER_COLUMNS]
    missing = [col for col in required if col not in hourly]
    if missing:
        raise NewSystemError(f"Open-Meteo response missing keys: {missing}")

    weather_times = pd.to_datetime(hourly["time"], errors="coerce")
    if isinstance(weather_times.dtype, pd.DatetimeTZDtype):
        weather_times = weather_times.dt.tz_convert(timezone).dt.tz_localize(None)

    weather = pd.DataFrame(
        {
            "time": weather_times,
            "air_temperature": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "relative_humidity": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
            "wind_speed": pd.to_numeric(hourly["wind_speed_10m"], errors="coerce"),
            "ghi_pyr": pd.to_numeric(hourly["shortwave_radiation"], errors="coerce"),
            "dni": pd.to_numeric(hourly["direct_normal_irradiance"], errors="coerce"),
            "dhi": pd.to_numeric(hourly["diffuse_radiation"], errors="coerce"),
        }
    )

    weather = weather.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if weather.empty:
        raise NewSystemError("Open-Meteo returned no usable rows.")
    return weather


def _align_weather_to_pv(pv_times: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    """Align hourly weather data to PV timestamps via time interpolation.

    Uses a DatetimeIndex (matching the reference in update_systems_01to05_to_today.py)
    to guarantee the reindex matches on timestamp values rather than the Series index.
    """
    target_index = pd.DatetimeIndex(pd.to_datetime(pv_times, errors="coerce").to_numpy(), name="time")
    weather_indexed = weather.set_index("time").sort_index()

    # Interpolate on a union index to support sub-hourly timestamps with no exact hourly matches.
    union_index = weather_indexed.index.union(target_index.unique()).sort_values()
    interpolated = weather_indexed.reindex(union_index).interpolate(method="time").ffill().bfill()

    aligned = interpolated.reindex(target_index).reset_index().rename(columns={"index": "time"})
    return aligned[["time", "ghi_pyr", "dni", "dhi", "air_temperature", "relative_humidity", "wind_speed"]]


def _to_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the merged dataframe to the standard output schema."""
    out = df.copy()
    out["date"] = out["time"].dt.strftime("%d/%m/%y")
    out["time"] = out["time"].dt.strftime("%I:%M%p").str.lstrip("0")
    return out[
        [
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
    ]


def _build_output_name(
    system_id: int,
    capacity_kw: float,
    city: str,
    lat: float,
    lon: float,
    start_day: date,
    end_day: date,
) -> str:
    """Build the output filename following the existing convention."""
    safe_city = city.replace(" ", "_")
    return (
        f"system{int(system_id):02d}_{capacity_kw:.1f}kW_{safe_city}_"
        f"lat{lat}_lon{lon}_{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}_weather.csv"
    )


def create_new_system_dataset(
    *,
    system_id: int,
    capacity_kw: float,
    city: str,
    lat: float,
    lon: float,
    uploaded_file_path: str | Path,
) -> dict:
    """Create a new system dataset from an uploaded PV power CSV.

    Args:
        system_id: System identifier number.
        capacity_kw: System capacity in kW.
        city: City name for the system.
        lat: Latitude for Open-Meteo weather fetch.
        lon: Longitude for Open-Meteo weather fetch.
        uploaded_file_path: Path to the uploaded PV power CSV file.

    Returns:
        A dict with keys: file_name, output_path, row_count, start_time, end_time.

    Raises:
        NewSystemError: If the uploaded file is invalid or processing fails.
    """
    # --- Validate inputs ---
    if not system_id or system_id <= 0:
        raise NewSystemError("System ID must be a positive integer.")
    if not capacity_kw or capacity_kw <= 0:
        raise NewSystemError("Capacity must be a positive number (kW).")
    if not city or not str(city).strip():
        raise NewSystemError("City name is required.")
    if lat is None or lon is None:
        raise NewSystemError("Latitude and longitude are required.")

    # --- Load uploaded file ---
    upload_path = Path(uploaded_file_path)
    if not upload_path.exists():
        raise NewSystemError(f"Uploaded file not found: {upload_path}")

    try:
        df = pd.read_csv(upload_path)
    except Exception as exc:
        raise NewSystemError(f"Could not read uploaded CSV: {exc}") from exc

    if df.empty:
        raise NewSystemError("Uploaded CSV is empty.")

    # --- Validate required columns ---
    date_col = _resolve_date_column(df)
    time_col = _resolve_time_column(df)
    power_col = _resolve_pv_power_column(df)

    if date_col is None and time_col is None:
        # Try to find a single datetime column
        datetime_col = None
        for alias in ["timestamp", "Timestamp", "datetime", "DateTime", "date_time", "Date_Time"]:
            if alias in df.columns:
                datetime_col = alias
                break
        if datetime_col is None:
            raise NewSystemError(
                "Uploaded file must contain a 'date' and 'time' column (or a single 'timestamp' column)."
            )
        ts = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        if date_col is None:
            raise NewSystemError("Uploaded file must contain a 'date' column.")
        if time_col is None:
            raise NewSystemError("Uploaded file must contain a 'time' column.")
        ts = _parse_uploaded_timestamps(df, date_col, time_col)

    if power_col is None:
        raise NewSystemError(
            "Uploaded file must contain a PV power column (e.g. 'power_w', 'power', 'pv_power')."
        )

    # --- Parse timestamps ---
    ts = _to_local_naive(pd.to_datetime(ts, errors="coerce"), TIMEZONE)
    valid_mask = ts.notna()
    if not valid_mask.any():
        raise NewSystemError("No valid timestamps found in the uploaded file.")

    raw_power_values = pd.to_numeric(df.loc[valid_mask, power_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    pv = pd.DataFrame(
        {
            "time": ts[valid_mask].to_numpy(),
            "power_w": raw_power_values.to_numpy(),
        }
    )
    pv = pv.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if pv.empty:
        raise NewSystemError("No valid rows with timestamps found in the uploaded file.")

    # --- Determine date range from data ---
    data_start = pv["time"].min()
    data_end = pv["time"].max()
    start_day = data_start.date()
    end_day = data_end.date()

    # --- Normalize PV power ---
    capacity_w = max(float(capacity_kw) * 1000.0, 1.0)
    if power_col in {"power_average_w_normalized", "power_normalized"}:
        # Data is already normalized (0-1 range); use it directly.
        pv["power_average_w_normalized"] = pv["power_w"].clip(lower=0.0, upper=1.0)
    else:
        # Raw power (watts, kW, or other) → normalize by system capacity.
        pv["power_average_w_normalized"] = (pv["power_w"] / capacity_w).clip(lower=0.0)

    # --- Fetch weather from Open-Meteo ---
    try:
        weather = _fetch_openmeteo_weather(
            lat=float(lat),
            lon=float(lon),
            timezone=TIMEZONE,
            start_day=start_day,
            end_day=end_day,
        )
    except requests.RequestException as exc:
        raise NewSystemError(f"Failed to fetch weather from Open-Meteo: {exc}") from exc

    # --- Align weather to PV timestamps ---
    aligned_weather = _align_weather_to_pv(pv["time"], weather)
    merged = pv.merge(aligned_weather, on="time", how="left")

    # --- Build output ---
    final_out = _to_output_schema(merged)
    output_name = _build_output_name(
        system_id=system_id,
        capacity_kw=float(capacity_kw),
        city=str(city).strip(),
        lat=float(lat),
        lon=float(lon),
        start_day=start_day,
        end_day=end_day,
    )

    output_path = PV_DATA_DIR / output_name
    PV_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # If file already exists, add a timestamp suffix
    if output_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PV_DATA_DIR / output_name.replace(".csv", f"_created_{stamp}.csv")

    final_out.to_csv(output_path, index=False)

    return {
        "file_name": output_path.name,
        "output_path": str(output_path),
        "row_count": int(len(final_out)),
        "start_time": data_start.strftime("%Y-%m-%d %H:%M"),
        "end_time": data_end.strftime("%Y-%m-%d %H:%M"),
    }
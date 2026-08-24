from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemRecord:
    system_number: int
    pvoutput_id: str
    pvoutput_sid: str
    name: str
    location: str
    capacity_kw: float
    lat: float
    lon: float
    tilt_deg: float
    azimuth_deg: float
    timezone: str
    status: str


@dataclass(frozen=True)
class CoolingSiteRecord:
    site_id: str
    name: str
    city: str
    lat: float
    lon: float
    surface_tilt: float
    surface_azimuth: float
    design_cooling_load_kw: float
    status: str


@dataclass(frozen=True)
class DatasetSnapshot:
    status: str
    path: Path | None
    row_count: int
    column_count: int
    start_ts: str
    end_ts: str
    median_interval_min: float | None
    missing_cells: int
    missing_pct: float

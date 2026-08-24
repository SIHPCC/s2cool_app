"""Application adapter for the M3 hybrid cooling-demand forecaster.

The Dash page deliberately talks to this module instead of the M3 command
line script.  The adapter keeps large datasets out of layout construction,
normalises M3 output into a JSON-safe result contract, and centralises input
validation and reproducible exports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import base64
import io
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.config_service import (
    CONFIG_DIR,
    REPO_ROOT,
    _load_json,
    load_cooling_site_payloads,
    load_model_defaults,
    load_systems,
)

M3_DIR = REPO_ROOT / "M3_CoolingLoad_prediction_module"
M3_DATA_DIR = M3_DIR / "data"
COOLING_OUTPUT_DIR = M3_DIR / "forecast"
DATA_ANALYSIS_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "preprocessing"
SUPPORTED_HORIZONS = (5, 10, 15, 20, 30, 60, 120)
TREE_MODEL_NAMES = ("xgboost", "extra_trees", "random_forest", "lightgbm", "catboost")
FORECAST_MODEL_NAMES = TREE_MODEL_NAMES + ("lstm",)
FEATURE_CANDIDATES = (
    "Gpoa", "ghi_pyr", "ghi", "dni", "dhi", "air_temperature",
    "relative_humidity", "wind_speed", "Q_phys_kW", "Q_solar_kW",
    "Q_env_kW", "Q_wall_kW", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
)

MODEL_SPECS = {
    "xgboost": {"label": "XGBoost", "package": "xgboost"},
    "extra_trees": {"label": "Extra Trees", "package": "scikit-learn"},
    "random_forest": {"label": "Random Forest", "package": "scikit-learn"},
    "lightgbm": {"label": "LightGBM", "package": "lightgbm"},
    "catboost": {"label": "CatBoost", "package": "catboost"},
    "lstm": {"label": "LSTM", "package": "torch"},
    # Kept for API compatibility with existing service callers.
    "hybrid_xgboost": {"label": "Hybrid XGBoost", "package": "xgboost"},
    "hybrid_gradient_boosting": {"label": "Hybrid Gradient Boosting", "package": "scikit-learn"},
    "physics_only": {"label": "Physics-only baseline", "package": "built-in"},
    "persistence": {"label": "Persistence baseline", "package": "built-in"},
}

DEFAULT_MODEL_SETTINGS = {
    "xgboost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
        "max_training_rows": 0,
    },
    "extra_trees": {"n_estimators": 250, "max_depth": 0, "min_samples_leaf": 2, "max_features": 1.0, "max_training_rows": 0},
    "random_forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 2, "max_features": 1.0, "max_training_rows": 0},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "max_training_rows": 0},
    "catboost": {"iterations": 300, "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "max_training_rows": 0},
    "lstm": {"sequence_length": 12, "epochs": 8, "batch_size": 64, "max_training_rows": 0},
    "hybrid_xgboost": {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
        "max_training_rows": 0,
    },
    "hybrid_gradient_boosting": {
        "n_estimators": 250, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.9, "max_training_rows": 0,
    },
}


class CoolingDemandError(RuntimeError):
    """A validation or model error safe to show in the Dash page."""


@dataclass(frozen=True)
class CoolingSource:
    source_id: str
    source_type: str
    label: str
    path: Path
    system_id: int | None = None
    site_id: str | None = None
    metadata: dict[str, Any] | None = None


def _load_m3_module():
    path = M3_DIR / "cooling_hybrid_forecasting_multihorizon.py"
    spec = importlib.util.spec_from_file_location("s2cool_m3_cooling_forecasting", path)
    if spec is None or spec.loader is None:
        raise CoolingDemandError(f"Unable to load M3 module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CoolingDemandError(f"M3 dependencies are unavailable: {exc}") from exc
    return module


def model_availability() -> dict[str, bool]:
    availability: dict[str, bool] = {
        # Keep XGBoost selectable like PV Forecasting; the estimator path
        # falls back to sklearn when the optional wheel is unavailable.
        "xgboost": True,
        "extra_trees": True,
        "random_forest": True,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "catboost": importlib.util.find_spec("catboost") is not None,
        "lstm": importlib.util.find_spec("torch") is not None,
        "hybrid_gradient_boosting": True,
        "physics_only": True,
        "persistence": True,
    }
    availability["hybrid_xgboost"] = availability["xgboost"]
    return availability


def model_options(include_baselines: bool = True) -> list[dict[str, Any]]:
    availability = model_availability()
    options = []
    names = list(MODEL_SPECS) if include_baselines else list(FORECAST_MODEL_NAMES)
    for name in names:
        spec = MODEL_SPECS[name]
        ready = availability[name]
        label = spec["label"] if ready else f"{spec['label']} (install {spec['package']})"
        options.append({"label": label, "value": name, "disabled": not ready})
    return options


def _system_by_id(system_id: int | None):
    return next((item for item in load_systems() if item.system_number == int(system_id)), None) if system_id is not None else None


def _parse_source_metadata(path: Path, system_id: int | None = None) -> dict[str, Any]:
    stem = path.stem
    match = re.search(r"system(\d+)_([^_]+)kW_([^_]+)", stem, re.IGNORECASE)
    inferred_id = int(match.group(1)) if match else system_id
    capacity = None
    city = None
    if match:
        try:
            capacity = float(match.group(2))
        except ValueError:
            capacity = None
        city = match.group(3).replace("_", " ").title()
    system = _system_by_id(inferred_id)
    if system:
        capacity = system.capacity_kw
        city = system.location
    return {
        "system_id": inferred_id,
        "capacity_kw": capacity,
        "city": city,
        "file_name": path.name,
        "path": str(path),
    }


def _find_system_dataset(system_id: int) -> Path | None:
    matches = sorted(M3_DATA_DIR.glob(f"q_measured_system{int(system_id):02d}_*.csv"))
    return matches[0] if matches else None


def _find_data_analysis_artifact(system_id: int | None) -> tuple[Path | None, dict[str, Any]]:
    if system_id is None or not DATA_ANALYSIS_DIR.exists():
        return None, {}
    for metadata_path in sorted(DATA_ANALYSIS_DIR.glob("*_quality_*.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(metadata.get("system_id", -1)) != int(system_id):
            continue
        file_name = metadata.get("file")
        csv_path = DATA_ANALYSIS_DIR / str(file_name) if file_name else None
        if csv_path and csv_path.exists():
            return csv_path, metadata
    return None, {}


def _profile_by_id(profile_id: str | None) -> dict[str, Any] | None:
    profiles = load_cooling_site_payloads()
    if profile_id:
        return next((p for p in profiles if p.get("site_id") == profile_id), None)
    return profiles[0] if profiles else None


def list_cooling_profiles() -> list[dict[str, Any]]:
    return [
        {
            "site_id": str(profile.get("site_id")),
            "label": f"{profile.get('name', profile.get('site_id'))} | {profile.get('city', '-')}",
            "profile": profile,
        }
        for profile in load_cooling_site_payloads()
    ]


def _profile_source(profile: dict[str, Any]) -> Path | None:
    explicit = profile.get("data_source")
    if explicit:
        path = REPO_ROOT / str(explicit)
        if path.exists():
            return path
    if profile.get("data_source_system_id"):
        return _find_system_dataset(int(profile["data_source_system_id"]))
    city = str(profile.get("city", "")).lower()
    candidates = []
    for path in sorted(M3_DATA_DIR.glob("q_measured_system*.csv")):
        metadata = _parse_source_metadata(path)
        if city and str(metadata.get("city", "")).lower() == city:
            candidates.append(path)
    return candidates[0] if candidates else None


def list_cooling_sources(system_id: int | None = None) -> list[CoolingSource]:
    sources: list[CoolingSource] = []
    for path in sorted(M3_DATA_DIR.glob("q_measured_system*.csv")):
        metadata = _parse_source_metadata(path)
        source_system_id = metadata.get("system_id")
        if system_id is not None and source_system_id != int(system_id):
            continue
        source_system = _system_by_id(source_system_id)
        label = f"System {int(source_system_id):02d} | {metadata.get('city') or '-'} | {metadata.get('capacity_kw') or '-'} kW"
        sources.append(CoolingSource(
            source_id=f"system-{int(source_system_id):02d}",
            source_type="system",
            label=label,
            path=path,
            system_id=int(source_system_id),
            metadata={**metadata, "status": "available"},
        ))
    for item in list_cooling_profiles():
        profile = item["profile"]
        path = _profile_source(profile)
        if path is None:
            continue
        sources.append(CoolingSource(
            source_id=f"profile-{profile['site_id']}",
            source_type="profile",
            label=f"Profile | {profile.get('name', profile['site_id'])}",
            path=path,
            system_id=profile.get("data_source_system_id"),
            site_id=str(profile["site_id"]),
            metadata={"profile": profile, "file_name": path.name, "path": str(path)},
        ))
    return sources


def _source(source_id: str | None, source_type: str | None = None, system_id: int | None = None) -> CoolingSource:
    sources = list_cooling_sources(system_id if source_type == "system" else None)
    match = next((item for item in sources if item.source_id == source_id and (source_type is None or item.source_type == source_type)), None)
    if match is None:
        raise CoolingDemandError("Select a valid cooling dataset or site profile.")
    return match


def get_cooling_source_info(source_id: str | None, source_type: str | None = None, system_id: int | None = None) -> dict[str, Any]:
    source = _source(source_id, source_type, system_id)
    header = pd.read_csv(source.path, nrows=0)
    metadata = dict(source.metadata or {})
    metadata.update({
        "source_id": source.source_id,
        "source_type": source.source_type,
        "label": source.label,
        "columns": list(header.columns),
        "file_size_mb": round(source.path.stat().st_size / (1024 * 1024), 2),
        "row_count": None,
        "status": "ready",
    })
    analysis_path, analysis_metadata = _find_data_analysis_artifact(source.system_id or system_id)
    metadata["data_analysis_artifact"] = str(analysis_path) if analysis_path else None
    metadata["data_analysis_artifact_name"] = analysis_path.name if analysis_path else None
    metadata["data_analysis_artifact_id"] = analysis_metadata.get("artifact_id")
    metadata["data_analysis_lat"] = analysis_metadata.get("lat")
    metadata["data_analysis_lon"] = analysis_metadata.get("lon")
    metadata["data_analysis_city"] = analysis_metadata.get("city")
    metadata["data_analysis_capacity_kw"] = analysis_metadata.get("capacity_kw")
    quality = analysis_metadata.get("quality_summary") or {}
    metadata["data_analysis_rows"] = quality.get("rows_after")
    metadata["data_analysis_start"] = quality.get("start_time")
    metadata["data_analysis_end"] = quality.get("end_time")
    metadata["data_analysis_interval_minutes"] = analysis_metadata.get("sampling_interval_minutes")
    metadata["data_analysis_feature_count"] = len(analysis_metadata.get("feature_columns") or [])
    if source.site_id:
        metadata["profile"] = _profile_by_id(source.site_id)
    return metadata


def _validate_number(name: str, value: Any, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CoolingDemandError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        limit = f" and <= {maximum}" if maximum is not None else ""
        raise CoolingDemandError(f"{name} must be >= {minimum}{limit}.")
    return number


def _profile_from_request(request: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads(json.dumps(_profile_by_id(request.get("profile_id")) or {}))
    hall = profile.setdefault("hall", {})
    physics = profile.setdefault("physics_initial", {})
    hall_fields = (
        "room_length_m", "room_width_m", "room_height_m", "window_count",
        "window_area_each_m2", "people_count", "sensible_w_per_person",
        "latent_w_per_person", "lighting_w_per_m2", "non_it_misc_kw", "it_load_kw",
    )
    physics_fields = ("shgc", "r_env_kw_per_k", "r_int_kw_per_k", "c_air_kj_per_k")
    schedule_fields = (
        "day_start_hour", "day_end_hour", "it_day_multiplier", "it_night_multiplier",
        "people_day_multiplier", "people_night_multiplier", "lighting_day_multiplier",
        "lighting_night_multiplier", "misc_day_multiplier", "misc_night_multiplier",
        "weekend_multiplier",
    )
    for field in hall_fields:
        if field in request and request[field] not in (None, ""):
            hall[field] = request[field]
    for field in physics_fields:
        if field in request and request[field] not in (None, ""):
            physics[field] = request[field]
    schedule = profile.setdefault("internal_load_schedule", {})
    for field in schedule_fields:
        if field in request and request[field] not in (None, ""):
            schedule[field] = request[field]
    return profile


def _validate_profile(profile: dict[str, Any]) -> None:
    if not profile.get("site_id"):
        raise CoolingDemandError("A cooling profile requires a site ID.")
    for field in ("lat", "lon"):
        _validate_number(field, profile.get(field), -90 if field == "lat" else -180, 90 if field == "lat" else 180)
    hall = profile.get("hall", {})
    for field in ("room_length_m", "room_width_m", "room_height_m", "window_area_each_m2", "sensible_w_per_person", "latent_w_per_person", "lighting_w_per_m2", "non_it_misc_kw", "it_load_kw"):
        _validate_number(field, hall.get(field))
    for field in ("window_count", "people_count"):
        _validate_number(field, hall.get(field), 0)
    physics = profile.get("physics_initial", {})
    _validate_number("shgc", physics.get("shgc"), 0, 1)
    for field in ("r_env_kw_per_k", "r_int_kw_per_k", "c_air_kj_per_k"):
        _validate_number(field, physics.get(field), 0.000001)


def validate_cooling_input(request: dict[str, Any]) -> dict[str, Any]:
    source = _source(request.get("source_id"), request.get("source_type"), request.get("system_id"))
    horizons = sorted({int(value) for value in (request.get("horizons") or SUPPORTED_HORIZONS)})
    invalid = [value for value in horizons if value not in SUPPORTED_HORIZONS]
    if invalid or not horizons:
        raise CoolingDemandError("Choose one or more horizons from 5, 10, 15, 20, 30, 60, and 120 minutes.")
    requested_models = request.get("models") or request.get("model") or "xgboost"
    models = [str(value) for value in requested_models] if isinstance(requested_models, (list, tuple, set)) else [str(requested_models)]
    invalid_models = [name for name in models if name not in FORECAST_MODEL_NAMES and name not in {"hybrid_xgboost", "hybrid_gradient_boosting", "physics_only", "persistence"}]
    if invalid_models:
        raise CoolingDemandError(f"Unknown cooling model: {invalid_models[0]}.")
    unavailable = [name for name in models if not model_availability().get(name, False)]
    if unavailable:
        spec = MODEL_SPECS[unavailable[0]]
        raise CoolingDemandError(f"{spec['label']} is unavailable. Install {spec['package']} or choose another model.")
    model = models[0]
    profile = _profile_from_request(request)
    _validate_profile(profile)
    measurement_mode = str(request.get("measurement_mode") or "synthetic")
    if measurement_mode not in {"synthetic", "experimental"}:
        raise CoolingDemandError("Choose synthetic or experimental Q measured data.")
    if measurement_mode == "experimental" and not request.get("measurement_upload_contents"):
        raise CoolingDemandError("Upload an experimental CSV for the selected Q measured mode.")
    alpha = _validate_number("interval alpha", request.get("interval_alpha", 0.1), 0.001, 0.99)
    iterations = int(_validate_number("calibration iterations", request.get("calibration_iterations", 1000), 0, 100000))
    return {"source": source, "horizons": horizons, "model": model, "models": models, "profile": profile, "interval_alpha": alpha, "calibration_iterations": iterations, "measurement_mode": measurement_mode}


def _parse_window(request: dict[str, Any], df: pd.DataFrame, m3) -> tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    start_value = request.get("start")
    end_value = request.get("end")
    ts = pd.to_datetime(df["_ts"], errors="coerce")
    if start_value or end_value:
        start = pd.to_datetime(start_value, errors="coerce")
        end = pd.to_datetime(end_value, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end <= start:
            raise CoolingDemandError("Choose a valid backtest start and end date/time.")
        train = ts < start
        test = (ts >= start) & (ts <= end)
        if int(test.sum()) < 1:
            raise CoolingDemandError("The selected backtest window has no dataset rows.")
        if int(train.sum()) < 50:
            raise CoolingDemandError("At least 50 rows are required before the backtest window.")
        return train, test, pd.Timestamp(start), pd.Timestamp(end)
    test_days = int(request.get("test_days", 2))
    train_month = str(request.get("train_month") or "")
    work = df
    if train_month:
        try:
            work = m3.filter_month(df, train_month)
        except Exception:
            # Current files may not contain the historical configuration month.
            # Auto-selecting the latest month is safer than returning an empty UI.
            latest_month = pd.to_datetime(df["_ts"]).dt.strftime("%Y-%m").max()
            work = m3.filter_month(df, latest_month)
    train, test, start, end = m3.select_consecutive_test_window(work, test_days, str(request.get("test_start_date") or "2026-05-01"))
    full_ts = pd.to_datetime(df["_ts"])
    return full_ts < start, (full_ts >= start) & (full_ts < end), start, end


def _make_config(profile: dict[str, Any], system: Any, hall: dict[str, Any], m3, test_size: float = 0.2):
    physics = profile.get("physics_initial", {})
    lat = float(getattr(system, "lat", profile.get("lat"))) if system else float(profile["lat"])
    lon = float(getattr(system, "lon", profile.get("lon"))) if system else float(profile["lon"])
    tilt = float(getattr(system, "tilt_deg", profile.get("surface_tilt", 30.0))) if system else float(profile.get("surface_tilt", 30.0))
    azimuth = float(getattr(system, "azimuth_deg", profile.get("surface_azimuth", 180.0))) if system else float(profile.get("surface_azimuth", 180.0))
    return m3.CoolingConfig(
        lat=lat, lon=lon, surface_tilt=tilt, surface_azimuth=azimuth,
        shgc=float(physics.get("shgc", 0.35)),
        window_area_m2=float(hall["window_area_m2"]),
        r_env_kw_per_k=float(physics.get("r_env_kw_per_k", 7.75)),
        r_int_kw_per_k=float(physics.get("r_int_kw_per_k", 1.5)),
        c_air_kj_per_k=float(physics.get("c_air_kj_per_k", 18000.0)),
        q_internal_kw=float(hall["q_internal_kw"]), test_size=test_size,
    )


def _fit_residual_model(df: pd.DataFrame, feature_cols: list[str], train_mask: pd.Series, model_name: str, settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_df = df.dropna(subset=feature_cols + ["Q_measured_kW", "Q_phys_kW"]).copy()
    if len(model_df) < 200:
        raise CoolingDemandError("At least 200 valid cooling rows are required for residual forecasting.")
    model_df["residual"] = model_df["Q_measured_kW"] - model_df["Q_phys_kW"]
    train = train_mask.reindex(model_df.index).fillna(False)
    train_df = model_df.loc[train]
    if len(train_df) < 100:
        raise CoolingDemandError("At least 100 training rows are required after filtering.")
    max_rows = int(settings.get("max_training_rows", 0))
    if max_rows > 0 and len(train_df) > max_rows:
        train_df = train_df.tail(max_rows)
    x_train = train_df[feature_cols]
    y_train = train_df["residual"]
    if model_name in {"xgboost", "hybrid_xgboost"}:
        try:
            from xgboost import XGBRegressor
            estimator = XGBRegressor(
                n_estimators=int(settings.get("n_estimators", 500)),
                max_depth=int(settings.get("max_depth", 6)),
                learning_rate=float(settings.get("learning_rate", 0.03)),
                subsample=float(settings.get("subsample", 0.9)),
                colsample_bytree=float(settings.get("colsample_bytree", 0.9)),
                reg_lambda=float(settings.get("reg_lambda", 1.0)),
                random_state=42, objective="reg:squarederror", verbosity=0,
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            estimator = GradientBoostingRegressor(
                n_estimators=int(settings.get("n_estimators", 300)), max_depth=int(settings.get("max_depth", 6)),
                learning_rate=float(settings.get("learning_rate", 0.05)), subsample=float(settings.get("subsample", 0.9)), random_state=42,
            )
    elif model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        estimator = ExtraTreesRegressor(
            n_estimators=int(settings.get("n_estimators", 250)),
            max_depth=None if int(settings.get("max_depth", 0)) <= 0 else int(settings["max_depth"]),
            min_samples_leaf=int(settings.get("min_samples_leaf", 2)),
            max_features=float(settings.get("max_features", 1.0)), random_state=42, n_jobs=-1,
        )
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(
            n_estimators=int(settings.get("n_estimators", 200)),
            max_depth=None if int(settings.get("max_depth", 0)) <= 0 else int(settings["max_depth"]),
            min_samples_leaf=int(settings.get("min_samples_leaf", 2)),
            max_features=float(settings.get("max_features", 1.0)), random_state=42, n_jobs=-1,
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(
            n_estimators=int(settings.get("n_estimators", 300)), learning_rate=float(settings.get("learning_rate", 0.05)),
            num_leaves=int(settings.get("num_leaves", 31)), max_depth=int(settings.get("max_depth", -1)),
            random_state=42, verbosity=-1,
        )
    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(
            iterations=int(settings.get("iterations", 300)), depth=int(settings.get("depth", 7)),
            learning_rate=float(settings.get("learning_rate", 0.05)), l2_leaf_reg=float(settings.get("l2_leaf_reg", 3.0)),
            loss_function="RMSE", verbose=False, random_seed=42,
        )
    elif model_name == "lstm":
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise CoolingDemandError(f"LSTM requires PyTorch and scikit-learn: {exc}") from exc
        sequence_length = max(2, int(settings.get("sequence_length", 12)))
        ordered = model_df.sort_index().copy()
        scaler = StandardScaler()
        features = scaler.fit_transform(ordered[feature_cols].to_numpy(dtype=np.float32))
        targets = ordered["residual"].to_numpy(dtype=np.float32)
        train_flags = train.reindex(ordered.index).fillna(False).to_numpy(dtype=bool)
        sequences, sequence_targets, sequence_positions = [], [], []
        for position in range(sequence_length - 1, len(ordered)):
            if train_flags[position]:
                sequences.append(features[position - sequence_length + 1:position + 1])
                sequence_targets.append(targets[position])
                sequence_positions.append(position)
        if len(sequences) < 100:
            raise CoolingDemandError("LSTM requires at least 100 valid training sequences.")
        torch.manual_seed(42)
        device = torch.device("cpu")
        network = torch.nn.Sequential()  # placeholder keeps the model CPU-only and reproducible
        class CoolingLSTM(torch.nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, 32, batch_first=True)
                self.output = torch.nn.Linear(32, 1)

            def forward(self, batch):
                values, _ = self.lstm(batch)
                return self.output(values[:, -1, :])

        network = CoolingLSTM(len(feature_cols)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss()
        x_tensor = torch.tensor(np.asarray(sequences), dtype=torch.float32, device=device)
        y_tensor = torch.tensor(np.asarray(sequence_targets)[:, None], dtype=torch.float32, device=device)
        batch_size = max(1, int(settings.get("batch_size", 64)))
        network.train()
        for _ in range(max(1, int(settings.get("epochs", 8)))):
            for start in range(0, len(x_tensor), batch_size):
                prediction = network(x_tensor[start:start + batch_size])
                loss = loss_function(prediction, y_tensor[start:start + batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        all_sequences = [features[position - sequence_length + 1:position + 1] for position in range(sequence_length - 1, len(ordered))]
        network.eval()
        with torch.no_grad():
            all_predictions = network(torch.tensor(np.asarray(all_sequences), dtype=torch.float32, device=device)).cpu().numpy().ravel()
        residual_prediction = pd.Series(np.nan, index=ordered.index, dtype=float)
        residual_prediction.iloc[sequence_length - 1:] = all_predictions
        model_df["residual_pred"] = residual_prediction.reindex(model_df.index).to_numpy()
        model_df["Q_hybrid_kW"] = (model_df["Q_phys_kW"] + model_df["residual_pred"].fillna(0.0)).clip(lower=0)
        model_df["is_train"] = train.values
        model_df["is_test"] = train_mask.reindex(model_df.index).notna() & ~train
        return model_df, {"estimator": "CoolingLSTM", "settings": settings, "training_rows": int(len(sequence_targets))}
    elif model_name == "hybrid_gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        estimator = GradientBoostingRegressor(
            n_estimators=int(settings.get("n_estimators", 250)), max_depth=int(settings.get("max_depth", 3)),
            learning_rate=float(settings.get("learning_rate", 0.05)), subsample=float(settings.get("subsample", 0.9)), random_state=42,
        )
    else:
        raise CoolingDemandError(f"Unsupported model: {model_name}")
    estimator.fit(x_train, y_train)
    model_df["residual_pred"] = estimator.predict(model_df[feature_cols])
    model_df["Q_hybrid_kW"] = (model_df["Q_phys_kW"] + model_df["residual_pred"]).clip(lower=0)
    model_df["is_train"] = train.values
    model_df["is_test"] = train_mask.reindex(model_df.index).notna() & ~train
    return model_df, {"estimator": type(estimator).__name__, "settings": settings, "training_rows": int(len(train_df))}


def _build_model_frame(df: pd.DataFrame, feature_cols: list[str], train_mask: pd.Series, test_mask: pd.Series, model: str, settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model in {"hybrid_xgboost", "hybrid_gradient_boosting"}:
        frame, details = _fit_residual_model(df, feature_cols, train_mask, model, settings)
    else:
        frame = df.dropna(subset=["Q_measured_kW", "Q_phys_kW"]).copy()
        if len(frame) < 50:
            raise CoolingDemandError("At least 50 valid rows are required for the selected baseline.")
        frame["residual_pred"] = 0.0
        frame["Q_hybrid_kW"] = frame["Q_phys_kW"] if model == "physics_only" else frame["Q_measured_kW"]
        frame["is_train"] = train_mask.reindex(frame.index).fillna(False).values
        frame["is_test"] = test_mask.reindex(frame.index).fillna(False).values
        details = {"estimator": "none", "settings": settings, "training_rows": int(frame["is_train"].sum())}
    return frame, details


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    if "_ts" in output:
        output["_ts"] = pd.to_datetime(output["_ts"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    output = output.replace({np.nan: None})
    return output.to_dict("records")


def _build_cooling_horizon_frame(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Create persistence-style cooling horizon columns for any PV-compatible horizon."""
    output = frame.copy().reset_index(drop=True)
    measured = output.set_index("_ts")["Q_measured_kW"].to_dict()
    for horizon in horizons:
        output[f"pred_{horizon}m"] = output["Q_hybrid_kW"]
        output[f"measured_at_{horizon}m"] = [
            measured.get(timestamp + pd.Timedelta(minutes=horizon), np.nan)
            for timestamp in output["_ts"]
        ]
        if "Q_hybrid_lower_kW" in output:
            output[f"pred_{horizon}m_lower"] = output["Q_hybrid_lower_kW"]
        if "Q_hybrid_upper_kW" in output:
            output[f"pred_{horizon}m_upper"] = output["Q_hybrid_upper_kW"]
    return output


def _metrics(frame: pd.DataFrame, horizons: list[int], eval_mask: pd.Series | None, capacity_kw: float | None = None) -> list[dict[str, Any]]:
    rows = []
    for horizon in horizons:
        actual = f"measured_at_{horizon}m"
        prediction = f"pred_{horizon}m"
        if actual not in frame or prediction not in frame:
            rows.append({"horizon_minutes": horizon, "mae": None, "rmse": None, "r2": None, "bias": None, "n": 0})
            continue
        work = frame[[actual, prediction]].copy()
        if eval_mask is not None:
            work = work.loc[eval_mask.reindex(frame.index).fillna(False).values]
        work = work.dropna()
        if len(work) == 0:
            rows.append({"horizon_minutes": horizon, "mae": None, "rmse": None, "r2": None, "bias": None, "n": 0})
            continue
        y_true = work[actual].to_numpy(dtype=float)
        y_pred = work[prediction].to_numpy(dtype=float)
        error = y_pred - y_true
        ss_total = float(((y_true - y_true.mean()) ** 2).sum())
        mae = float(np.abs(error).mean())
        rows.append({
            "horizon_minutes": horizon, "mae": mae,
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1.0 - np.square(error).sum() / ss_total) if ss_total > 1e-12 else None,
            "bias": float(error.mean()), "n": int(len(work)),
            "mae_kw": mae, "normalized_mae": mae / max(abs(float(y_true.mean())), 1e-12),
        })
    return rows


def _quality_report(raw: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    numeric = raw.select_dtypes(include=[np.number])
    missing = int(raw.isna().sum().sum())
    return {
        "source_rows": int(len(raw)),
        "processed_rows": int(len(frame)),
        "source_columns": list(raw.columns),
        "missing_cells": missing,
        "missing_pct": round(100.0 * missing / max(raw.shape[0] * raw.shape[1], 1), 3),
        "duplicate_timestamps": int(raw["_ts"].duplicated().sum()) if "_ts" in raw else 0,
        "numeric_columns": list(numeric.columns),
        "warnings": [],
    }


def _merge_data_analysis_weather(raw: pd.DataFrame, system_id: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the latest Data Analysis artifact as the weather/solar source."""
    artifact_path, metadata = _find_data_analysis_artifact(system_id)
    if artifact_path is None:
        return raw, {"status": "unavailable", "warnings": ["No saved Data Analysis artifact was found; M3 source weather/solar columns were used."]}
    try:
        artifact = pd.read_csv(artifact_path)
        if "_ts" not in artifact.columns:
            if {"date", "time"}.issubset(artifact.columns):
                artifact["_ts"] = pd.to_datetime(artifact["date"].astype(str) + " " + artifact["time"].astype(str), errors="coerce", dayfirst=True)
            else:
                raise ValueError("artifact has no timestamp column")
        artifact["_ts"] = pd.to_datetime(artifact["_ts"], errors="coerce")
        artifact = artifact.dropna(subset=["_ts"]).drop_duplicates("_ts", keep="last").set_index("_ts")
        output = raw.copy().set_index("_ts")
        columns = [column for column in ("ghi_pyr", "ghi", "dni", "dhi", "air_temperature", "relative_humidity", "wind_speed") if column in artifact.columns]
        for column in columns:
            values = pd.to_numeric(artifact[column], errors="coerce").reindex(output.index)
            output[column] = values.combine_first(pd.to_numeric(output.get(column), errors="coerce"))
        return output.reset_index(), {"status": "ready", "path": str(artifact_path), "name": artifact_path.name, "artifact_id": metadata.get("artifact_id"), "columns": columns}
    except Exception as exc:
        return raw, {"status": "error", "path": str(artifact_path), "warnings": [f"Data Analysis artifact could not be merged: {exc}"]}


def _request_config_hash(request: dict[str, Any], metadata: dict[str, Any]) -> str:
    payload = json.dumps({"request": request, "metadata": metadata}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _apply_load_schedule(df: pd.DataFrame, profile: dict[str, Any], hall: dict[str, Any]) -> pd.DataFrame:
    """Create profile-driven internal gains when the selected profile is used."""
    schedule = profile.get("internal_load_schedule", {})
    output = df.copy()
    timestamps = pd.to_datetime(output["_ts"], errors="coerce")
    hours = timestamps.dt.hour
    day_start = float(schedule.get("day_start_hour", 8))
    day_end = float(schedule.get("day_end_hour", 20))
    is_day = (hours >= day_start) & (hours < day_end)
    weekend = timestamps.dt.dayofweek >= 5
    it = float(profile.get("hall", {}).get("it_load_kw", 0))
    people = float(hall.get("people_kw", 0))
    lighting = float(hall.get("lighting_kw", 0))
    misc = float(profile.get("hall", {}).get("non_it_misc_kw", 0))
    it_multiplier = is_day.map(lambda value: float(schedule.get("it_day_multiplier" if value else "it_night_multiplier", 1.0)))
    people_multiplier = is_day.map(lambda value: float(schedule.get("people_day_multiplier" if value else "people_night_multiplier", 1.0)))
    lighting_multiplier = is_day.map(lambda value: float(schedule.get("lighting_day_multiplier" if value else "lighting_night_multiplier", 1.0)))
    misc_multiplier = is_day.map(lambda value: float(schedule.get("misc_day_multiplier" if value else "misc_night_multiplier", 1.0)))
    weekend_multiplier = weekend.map(lambda value: float(schedule.get("weekend_multiplier", 1.0)) if value else 1.0)
    output["q_internal_kw"] = (it * it_multiplier + people * people_multiplier + lighting * lighting_multiplier + misc * misc_multiplier) * weekend_multiplier
    return output


def _synthetic_q_measured(df: pd.DataFrame, profile: dict[str, Any], request: dict[str, Any], m3) -> pd.DataFrame:
    """Generate a reproducible air-side Q measured signal from UI parameters."""
    settings = profile.get("synthetic_measurement", {})
    setpoint = float(request.get("synthetic_indoor_setpoint_c", settings.get("indoor_temp_setpoint_c", 24.0)))
    supply_temp = float(request.get("synthetic_supply_temp_c", settings.get("supply_temp_c", 17.5)))
    supply_rh = float(request.get("synthetic_supply_rh_pct", settings.get("supply_rh_pct", 90.0)))
    return_rise = float(request.get("synthetic_return_rise_c", settings.get("return_rise_c", settings.get("return_temp_rise_c", 6.5))))
    airflow = max(0.0, float(request.get("synthetic_airflow_m3_s", settings.get("airflow_m3_s", 0.70))))
    noise = max(0.0, float(request.get("synthetic_noise_std_kw", settings.get("measurement_noise_std_kw", 0.05))))
    indoor = pd.to_numeric(df.get("indoor_temperature", pd.Series(setpoint, index=df.index)), errors="coerce").fillna(setpoint)
    return_temp = indoor + return_rise
    supply_temp_series = pd.Series(supply_temp, index=df.index)
    supply_rh_series = pd.Series(supply_rh, index=df.index)
    return_rh_series = pd.Series(float(request.get("synthetic_return_rh_pct", 50.0)), index=df.index)
    w_supply = m3.humidity_ratio_kgkg(supply_temp_series, supply_rh_series)
    w_return = m3.humidity_ratio_kgkg(return_temp, return_rh_series)
    h_supply = m3.enthalpy_kjkg(supply_temp_series, w_supply)
    h_return = m3.enthalpy_kjkg(return_temp, w_return)
    rng = np.random.default_rng(int(request.get("synthetic_seed", 42)))
    measured = airflow * m3.AIR_DENSITY * (h_return - h_supply)
    if noise:
        measured = measured + rng.normal(0.0, noise, len(df))
    output = df.copy()
    output["Q_measured_kW"] = pd.Series(measured, index=df.index).clip(lower=0.0)
    output["synthetic_return_temperature"] = return_temp
    output["synthetic_supply_temperature"] = supply_temp
    return output


def _experimental_q_measured(contents: str, filename: str | None, m3) -> pd.DataFrame:
    try:
        encoded = contents.split(",", 1)[1] if "," in contents else contents
        uploaded = pd.read_csv(io.BytesIO(base64.b64decode(encoded)))
    except Exception as exc:
        raise CoolingDemandError(f"Unable to read experimental CSV{f' {filename}' if filename else ''}: {exc}") from exc
    if "_ts" in uploaded.columns:
        uploaded["_ts"] = pd.to_datetime(uploaded["_ts"], errors="coerce")
    elif {"date", "time"}.issubset(uploaded.columns):
        uploaded["_ts"] = m3.parse_datetime(uploaded)
    else:
        timestamp = next((column for column in ("timestamp", "datetime", "date_time") if column in uploaded.columns), None)
        if not timestamp:
            raise CoolingDemandError("Experimental CSV must contain _ts, date/time, or timestamp columns.")
        uploaded["_ts"] = pd.to_datetime(uploaded[timestamp], errors="coerce")
    q_column = next((column for column in ("Q_measured_kW", "Q_measured_W", "Q_measured", "cooling_kw", "cooling_load_kw") if column in uploaded.columns), None)
    if not q_column:
        raise CoolingDemandError("Experimental CSV must contain Q_measured_kW, Q_measured_W, or cooling_kw.")
    uploaded["Q_measured_kW"] = pd.to_numeric(uploaded[q_column], errors="coerce")
    if str(q_column).lower().endswith("_w"):
        uploaded["Q_measured_kW"] = uploaded["Q_measured_kW"] / 1000.0
    uploaded = uploaded.dropna(subset=["_ts", "Q_measured_kW"])
    if len(uploaded) < 50:
        raise CoolingDemandError("Experimental CSV must contain at least 50 valid timestamped measurements.")
    return uploaded[["_ts", "Q_measured_kW"]].drop_duplicates("_ts", keep="last")


def _run(request: dict[str, Any], latest_only: bool = False) -> dict[str, Any]:
    checked = validate_cooling_input(request)
    source: CoolingSource = checked["source"]
    profile = checked["profile"]
    m3 = _load_m3_module()
    system = _system_by_id(source.system_id or request.get("system_id"))
    raw = m3.load_dataset(source.path)
    raw["_ts"] = pd.to_datetime(raw["_ts"], errors="coerce")
    raw = raw.dropna(subset=["_ts"]).sort_values("_ts").drop_duplicates("_ts").reset_index(drop=True)
    if raw.empty:
        raise CoolingDemandError("The selected cooling dataset contains no valid timestamped rows.")
    raw_before = raw.copy()
    raw = m3.regularize_to_5min(raw, enable=bool(request.get("regularize_5min", True)))
    raw, data_analysis_source = _merge_data_analysis_weather(raw, source.system_id or request.get("system_id"))
    hall_spec = m3.DatacenterHallSpec(**{key: float(profile.get("hall", {}).get(key, 0)) if key not in {"window_count", "people_count"} else int(profile.get("hall", {}).get(key, 0)) for key in (
        "room_length_m", "room_width_m", "room_height_m", "window_count", "window_area_each_m2", "people_count", "sensible_w_per_person", "latent_w_per_person", "lighting_w_per_m2", "non_it_misc_kw", "it_load_kw")})
    hall = m3.derive_hall_parameters(hall_spec)
    config = _make_config(profile, system, hall, m3, float(request.get("test_size", 0.2)))
    train_mask, test_mask, test_start, test_end = _parse_window(request, raw, m3)
    m3.numeric_fill(raw, ["ghi_pyr", "ghi", "dni", "dhi", "air_temperature", "relative_humidity", "wind_speed", "indoor_temperature", "wall_temperature", "q_internal_kw", "Q_measured_kW", "Q_measured_W", "supply_air_temperature", "supply_air_rh", "return_air_temperature", "return_air_rh", "airflow_m3_s"])
    raw = _apply_load_schedule(raw, profile, hall)
    raw = m3.compute_gpoa(raw, config)
    if checked["measurement_mode"] == "synthetic":
        raw = _synthetic_q_measured(raw, profile, request, m3)
    else:
        uploaded = _experimental_q_measured(request["measurement_upload_contents"], request.get("measurement_upload_filename"), m3)
        raw = raw.drop(columns=["Q_measured_kW"], errors="ignore").merge(uploaded, on="_ts", how="left")
        if int(raw["Q_measured_kW"].notna().sum()) < 50:
            raise CoolingDemandError("Experimental measurements do not overlap the selected system dataset timestamps.")
        if int((test_mask & raw["Q_measured_kW"].notna()).sum()) < 1:
            raise CoolingDemandError("Experimental measurements do not overlap the backtest window. Choose dates covered by the uploaded CSV.")
    calibration_summary: dict[str, Any] = {"enabled": False}
    calibration_history = pd.DataFrame()
    if bool(request.get("auto_calibrate", False)):
        config, calibration_history, calibration_summary = m3.auto_calibrate_thermal_parameters(
            raw, config, train_mask=train_mask, val_mask=test_mask,
            n_iter=checked["calibration_iterations"], seed=int(request.get("calibration_seed", 42)),
        )
        calibration_summary["enabled"] = True
    raw = m3.compute_q_physics(raw, config)
    if bool(request.get("steady_state_only", False)):
        threshold = float(request.get("steady_state_threshold_k_per_min", 0.02)) / 60.0
        steady = raw["d_tin_dt_k_per_s"].abs() <= threshold
        train_mask = train_mask & steady
        test_mask = test_mask & steady
        if int(train_mask.sum()) < 100 and checked["model"] in set(TREE_MODEL_NAMES) | {"hybrid_xgboost", "hybrid_gradient_boosting"}:
            raise CoolingDemandError("Too few steady-state training rows. Relax the threshold or disable steady-state evaluation.")
    raw = m3.add_time_features(raw)
    feature_cols = [column for column in FEATURE_CANDIDATES if column in raw.columns]
    if len(feature_cols) < 4:
        raise CoolingDemandError("The dataset does not contain enough weather, thermal, or time features.")
    effective_train = raw.index.to_series().map(train_mask).fillna(False)
    effective_test = raw.index.to_series().map(test_mask).fillna(False)
    if latest_only:
        effective_train = raw["Q_measured_kW"].notna()
    settings = dict(DEFAULT_MODEL_SETTINGS.get(checked["model"], {}))
    requested_settings = request.get("model_settings") or {}
    if isinstance(requested_settings, dict) and isinstance(requested_settings.get(checked["model"]), dict):
        settings.update(requested_settings[checked["model"]])
    elif isinstance(requested_settings, dict):
        settings.update(requested_settings)
    model_frame, model_details = _build_model_frame(raw, feature_cols, effective_train, effective_test, checked["model"], settings)
    model_frame = m3.add_prediction_intervals(model_frame, alpha=checked["interval_alpha"])
    forecast = _build_cooling_horizon_frame(model_frame, checked["horizons"])
    horizons = checked["horizons"]
    eval_mask = forecast["is_test"] if "is_test" in forecast.columns and not latest_only else None
    metrics = _metrics(forecast, horizons, eval_mask, getattr(system, "capacity_kw", None))
    display = forecast.copy()
    if latest_only:
        display = display.tail(288)
    else:
        start = pd.to_datetime(request.get("start"), errors="coerce") if request.get("start") else None
        end = pd.to_datetime(request.get("end"), errors="coerce") if request.get("end") else None
        if start is not None and not pd.isna(start):
            display = display[display["_ts"] >= start]
        if end is not None and not pd.isna(end):
            display = display[display["_ts"] <= end]
        if (start is None or pd.isna(start)) and (end is None or pd.isna(end)) and "is_test" in display.columns:
            display = display[display["is_test"] == True]
    metadata = {
        "source_id": source.source_id, "source_type": source.source_type,
        "system_id": source.system_id, "site_id": source.site_id,
        "source_file": str(source.path), "source_file_name": source.path.name,
        "profile": profile, "hall_parameters": hall,
        "feature_columns": feature_cols, "horizons_minutes": horizons,
        "horizon_strategy": "M3 persistence-style horizon output",
        "model": checked["model"], "model_details": model_details,
        "measurement_mode": checked["measurement_mode"],
        "measurement_upload_filename": request.get("measurement_upload_filename"),
        "data_analysis_source": data_analysis_source,
        "train_start": str(raw.loc[train_mask, "_ts"].min()) if train_mask.any() else None,
        "test_start": str(test_start), "test_end": str(test_end),
        "calibration": calibration_summary,
        "steady_state_only": bool(request.get("steady_state_only", False)),
        "interval_alpha": checked["interval_alpha"],
    }
    quality = _quality_report(raw_before, forecast)
    quality["data_analysis_source"] = data_analysis_source
    quality["warnings"].extend(data_analysis_source.get("warnings") or [])
    if not any(column in raw.columns for column in ("ghi_pyr", "ghi", "dni", "dhi")):
        quality["warnings"].append("No measured irradiance channels were found; M3 clear-sky fallback was used.")
    metadata["quality"] = quality
    metadata["config_hash"] = _request_config_hash(request, metadata)
    result = {
        "run_id": f"cooling-{metadata['config_hash']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_only": latest_only, "source_rows": int(len(raw)),
        "source_start": str(raw["_ts"].min()), "source_end": str(raw["_ts"].max()),
        "latest_timestamp": str(forecast["_ts"].max()), "horizons": horizons,
        "model": checked["model"], "metadata": metadata, "quality": quality,
        "records": _json_records(display), "latest_record": _json_records(forecast.tail(1))[0] if not forecast.empty else None,
        "metrics": metrics, "calibration_summary": calibration_summary,
        "calibration_history": _json_records(calibration_history) if not calibration_history.empty else [],
    }
    return result


def _run_for_models(request: dict[str, Any], latest_only: bool) -> dict[str, Any]:
    requested = request.get("models") or request.get("model") or "xgboost"
    models = [str(value) for value in requested] if isinstance(requested, (list, tuple, set)) else [str(requested)]
    if len(models) == 1:
        single = dict(request)
        single["model"] = models[0]
        single["models"] = models
        return _run(single, latest_only=latest_only)
    results = {}
    for model_name in models:
        single = dict(request)
        single["model"] = model_name
        single["models"] = [model_name]
        results[model_name] = _run(single, latest_only=latest_only)
    first = results[models[0]]
    merged = dict(first)
    merged["model"] = models[0]
    merged["models"] = models
    merged["results"] = results
    merged["metrics"] = build_cooling_metrics(merged)
    return merged


def run_cooling_backtest(request: dict[str, Any]) -> dict[str, Any]:
    return _run_for_models(request, latest_only=False)


def generate_cooling_forecast(request: dict[str, Any]) -> dict[str, Any]:
    return _run_for_models(request, latest_only=True)


def build_cooling_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("results"):
        rows = []
        for model_name, payload in result["results"].items():
            for row in payload.get("metrics") or []:
                rows.append({"model": model_name, **row})
        return rows
    return list(result.get("metrics") or [])


def export_cooling_result(result: dict[str, Any]) -> dict[str, str]:
    COOLING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"cooling_{result.get('metadata', {}).get('system_id') or result.get('metadata', {}).get('site_id') or 'run'}_{stamp}"
    forecast_path = COOLING_OUTPUT_DIR / f"{stem}_forecast.csv"
    pd.DataFrame(result.get("records") or []).to_csv(forecast_path, index=False)
    metrics_path = COOLING_OUTPUT_DIR / f"{stem}_metrics.csv"
    pd.DataFrame(build_cooling_metrics(result)).to_csv(metrics_path, index=False)
    calibration_path = COOLING_OUTPUT_DIR / f"{stem}_calibration_history.csv"
    pd.DataFrame(result.get("calibration_history") or []).to_csv(calibration_path, index=False)
    calibration_summary_path = COOLING_OUTPUT_DIR / f"{stem}_calibration_summary.json"
    calibration_summary_path.write_text(json.dumps(result.get("calibration_summary") or {}, indent=2, default=str), encoding="utf-8")
    profile_path = COOLING_OUTPUT_DIR / f"{stem}_profile.json"
    profile_path.write_text(json.dumps(result.get("metadata", {}).get("profile") or {}, indent=2, default=str), encoding="utf-8")
    summary = dict(result)
    summary.update({
        "forecast_csv": str(forecast_path), "metrics_csv": str(metrics_path),
        "calibration_history_csv": str(calibration_path),
        "calibration_summary_json": str(calibration_summary_path), "profile_json": str(profile_path),
    })
    summary_path = COOLING_OUTPUT_DIR / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"forecast_csv": str(forecast_path), "metrics_csv": str(metrics_path), "calibration_csv": str(calibration_path), "calibration_summary": str(calibration_summary_path), "profile_json": str(profile_path), "summary_json": str(summary_path)}


def save_cooling_profile(profile: dict[str, Any]) -> dict[str, Any]:
    _validate_profile(profile)
    path = CONFIG_DIR / "cooling_sites.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sites = payload.setdefault("sites", [])
    existing = next((index for index, item in enumerate(sites) if item.get("site_id") == profile.get("site_id")), None)
    if existing is None:
        sites.append(profile)
    else:
        sites[existing] = profile
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _load_json.cache_clear()
    load_cooling_site_payloads.cache_clear()
    return profile

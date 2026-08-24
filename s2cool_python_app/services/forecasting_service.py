"""Application adapter for the M2 PV hybrid forecasting models.

The Dash page talks to this module instead of importing the research script
directly.  Preprocessing exports are the input contract, which keeps the
forecast reproducible and prevents the forecast page from silently applying a
second, different preprocessing pipeline.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from services.config_service import REPO_ROOT

PREPROCESSING_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "preprocessing"
FORECAST_OUTPUT_DIR = REPO_ROOT / "M2_PVnowcasting_module" / "forecast"
SUPPORTED_HORIZONS = (5, 10, 15, 20, 30, 60, 120)
M2_FEATURES = (
    "ghi_pyr", "dni", "dhi", "air_temperature", "relative_humidity",
    "wind_speed", "hour_sin", "hour_cos",
)

MODEL_SPECS = {
    "xgboost": {
        "label": "XGBoost",
        "description": "Gradient-boosted model; uses sklearn fallback when xgboost is unavailable.",
        "package": "xgboost",
    },
    "extra_trees": {
        "label": "Extra Trees",
        "description": "Fast randomized tree ensemble available through scikit-learn.",
        "package": "scikit-learn",
    },
    "random_forest": {
        "label": "Random Forest",
        "description": "Bagged tree ensemble available through scikit-learn.",
        "package": "scikit-learn",
    },
    "lightgbm": {
        "label": "LightGBM",
        "description": "High-performance gradient boosting for tabular forecasting.",
        "package": "lightgbm",
    },
    "catboost": {
        "label": "CatBoost",
        "description": "Ordered boosting model with strong tabular performance.",
        "package": "catboost",
    },
    "lstm": {
        "label": "LSTM",
        "description": "Recurrent sequence model for temporal patterns.",
        "package": "torch",
    },
}

DEFAULT_MODEL_SETTINGS = {
    "xgboost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    "extra_trees": {"n_estimators": 250, "max_depth": 0, "min_samples_leaf": 2, "max_features": 1.0},
    "random_forest": {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 2, "max_features": 1.0},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1},
    "catboost": {"iterations": 300, "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    "lstm": {"sequence_length": 12, "epochs": 8, "batch_size": 64},
}


class ForecastingError(RuntimeError):
    """A user-facing validation or model execution error."""


_TRAINED_MODELS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ForecastArtifact:
    artifact_id: str
    csv_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def system_id(self) -> int | None:
        value = self.metadata.get("system_id")
        return int(value) if value is not None else None
def _load_m2_module():
    path = REPO_ROOT / "M2_PVnowcasting_module" / "pv_hybrid_forecasting_multihorizon.py"
    spec = importlib.util.spec_from_file_location("s2cool_m2_pv_forecasting", path)
    if spec is None or spec.loader is None:
        raise ForecastingError(f"Unable to load M2 model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_availability() -> dict[str, bool]:
    """Return whether each optional model dependency is available."""
    availability = {}
    for name, spec in MODEL_SPECS.items():
        if name == "xgboost":
            # M2 has a GradientBoostingRegressor fallback, so this option is
            # always runnable even when the optional xgboost wheel is absent.
            availability[name] = True
        elif spec["package"] == "scikit-learn":
            availability[name] = True
        else:
            try:
                importlib.import_module(spec["package"])
                availability[name] = True
            except Exception:
                # A package can be discoverable while failing to load because
                # of missing native libraries or an incompatible environment.
                availability[name] = False
    return availability


def model_options() -> list[dict[str, Any]]:
    """Build checklist options with dependency-aware availability labels."""
    available = model_availability()
    options = []
    for name, spec in MODEL_SPECS.items():
        ready = available[name]
        label = spec["label"] if ready else f"{spec['label']} (install {spec['package']})"
        options.append({"label": label, "value": name, "disabled": not ready})
    return options


def list_forecast_artifacts(system_id: int | None = None) -> list[ForecastArtifact]:
    """Return exported Data Analysis artifacts compatible with a system."""
    if not PREPROCESSING_DIR.exists():
        return []
    artifacts: list[ForecastArtifact] = []
    for metadata_path in sorted(PREPROCESSING_DIR.glob("*_quality_*.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if system_id is not None and metadata.get("system_id") != system_id:
            continue
        csv_name = metadata.get("file")
        if not csv_name:
            continue
        csv_path = PREPROCESSING_DIR / str(csv_name)
        if not csv_path.exists():
            continue
        artifact_id = str(metadata.get("artifact_id") or metadata_path.stem)
        artifacts.append(ForecastArtifact(artifact_id, csv_path, metadata_path, metadata))
    return artifacts


def get_artifact(artifact_id: str | None, system_id: int | None = None) -> ForecastArtifact:
    for artifact in list_forecast_artifacts(system_id):
        if artifact.artifact_id == artifact_id:
            return artifact
    raise ForecastingError("Select a valid exported Data Analysis preprocessing artifact.")


def _parse_horizons(horizons: list[int] | tuple[int, ...] | None) -> list[int]:
    values = sorted({int(value) for value in (horizons or [5, 15, 30])})
    invalid = [value for value in values if value not in SUPPORTED_HORIZONS]
    if invalid:
        raise ForecastingError("Unsupported horizon(s): " f"{invalid}. Choose 5, 10, 15, 20, 30, 60, or 120 minutes.")
    if not values:
        raise ForecastingError("Choose at least one forecast horizon.")
    return values


def horizon_minutes_to_steps(minutes: int, interval_minutes: float) -> int:
    """Convert a requested horizon into a dataset step count safely."""
    if minutes not in SUPPORTED_HORIZONS:
        raise ForecastingError(f"Unsupported horizon: {minutes} minutes.")
    if not interval_minutes or interval_minutes <= 0:
        raise ForecastingError("The artifact has no valid sampling interval.")
    steps = int(round(float(minutes) / float(interval_minutes)))
    if steps < 1 or not math.isclose(steps * float(interval_minutes), float(minutes), rel_tol=0.0, abs_tol=0.01):
        raise ForecastingError(f"{minutes} minutes is not aligned with the artifact interval of {interval_minutes:g} minutes.")
    return steps


def _normalize_models(models: Any = None, model: str | None = None) -> list[str]:
    selected = models if models is not None else model
    if isinstance(selected, str):
        selected = ["xgboost", "lstm"] if selected == "comparison" else [selected]
    values = list(dict.fromkeys(str(value) for value in (selected or ["xgboost"])))
    invalid = [value for value in values if value not in MODEL_SPECS]
    if invalid:
        raise ForecastingError(f"Unknown model(s): {', '.join(invalid)}")
    return values


def validate_forecast_input(artifact: ForecastArtifact, horizons: list[int], model: str | None = None, models: Any = None) -> dict[str, Any]:
    horizons = _parse_horizons(horizons)
    selected_models = _normalize_models(models, model)
    metadata = artifact.metadata
    target = metadata.get("target_column", "power_average_w_normalized")
    features = list(metadata.get("feature_columns") or [])
    if not features:
        raise ForecastingError("The selected artifact has no feature metadata. Re-export it from Data Analysis.")
    if target != "power_average_w_normalized":
        raise ForecastingError("PV forecasting currently requires power_average_w_normalized as the target.")
    interval = metadata.get("sampling_interval_minutes")
    if interval:
        for horizon in horizons:
            horizon_minutes_to_steps(horizon, float(interval))
    return {"horizons": horizons, "target": target, "features": features, "models": selected_models}


def _load_artifact_frame(artifact: ForecastArtifact) -> pd.DataFrame:
    df = pd.read_csv(artifact.csv_path)
    if "_ts" not in df.columns:
        raise ForecastingError("The processed artifact does not contain a timestamp column.")
    df["_ts"] = pd.to_datetime(df["_ts"], errors="coerce")
    df = df.dropna(subset=["_ts"]).sort_values("_ts").drop_duplicates("_ts").reset_index(drop=True)
    if len(df) < 100:
        raise ForecastingError("At least 100 timestamped rows are required for a reliable forecast.")
    if "power_average_w_normalized" not in df.columns:
        raise ForecastingError("The processed artifact is missing the normalized PV power target.")
    for column in M2_FEATURES + ("power_average_w_normalized",):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    m2 = _load_m2_module()
    df = m2.build_features(df)
    missing = [column for column in M2_FEATURES if column not in df.columns]
    if missing:
        raise ForecastingError(f"The artifact is missing M2 model features: {', '.join(missing)}")
    return df


def _add_horizon_target(df: pd.DataFrame, horizon_minutes: int) -> tuple[pd.DataFrame, str]:
    """Attach the measured PV target at t + horizon to each feature row.

    Timestamp lookup is used instead of a positional shift so missing rows do
    not silently turn a 15-minute target into a different time offset.
    """
    target_column = f"_forecast_target_{int(horizon_minutes)}m"
    output = df.copy()
    measured_by_timestamp = output.set_index("_ts")["power_average_w_normalized"]
    target_timestamps = output["_ts"] + pd.Timedelta(minutes=int(horizon_minutes))
    output[target_column] = measured_by_timestamp.reindex(target_timestamps).to_numpy()
    return output, target_column


def _train_and_predict(
    df: pd.DataFrame,
    model_name: str,
    test_size: float,
    model_settings: dict | None,
    horizon_minutes: int,
) -> pd.DataFrame:
    m2 = _load_m2_module()
    settings = dict(DEFAULT_MODEL_SETTINGS.get(model_name, {}))
    settings.update(model_settings or {})
    horizon_df, target_column = _add_horizon_target(df, horizon_minutes)
    train_rows = max(60, min(len(df) - 1, int(len(df) * (1.0 - test_size))))
    train_df = horizon_df.iloc[:train_rows].copy()
    # The research script uses XGBoost when installed and sklearn's
    # GradientBoostingRegressor otherwise.  Keep the same M2 estimator path,
    # but bound the fallback training window so a Dash request remains usable
    # on the long historical files in this repository.
    max_training_rows = int(settings.get("max_training_rows", 12000))
    if len(train_df) > max_training_rows:
        train_df = train_df.tail(max_training_rows).reset_index(drop=True)
    if model_name == "xgboost":
        model = m2.train_xgb_model(
            train_df,
            test_size=max(1.0 / len(train_df), 0.001),
            settings=settings,
            target_column=target_column,
        )
        return m2.predict_xgb(df, model)
    if model_name == "lstm":
        artifact = m2.train_lstm_model(
            train_df,
            test_size=max(1.0 / len(train_df), 0.001),
            seq_len=int(settings.get("sequence_length", 12)),
            epochs=int(settings.get("epochs", 8)),
            batch_size=int(settings.get("batch_size", 64)),
            target_column=target_column,
        )
        return m2.predict_lstm(df, artifact)

    model_df = train_df.dropna(subset=list(M2_FEATURES) + [target_column]).copy()
    if len(model_df) < 50:
        raise ForecastingError(f"Not enough rows to train {MODEL_SPECS[model_name]['label']}.")
    x_train = model_df[list(M2_FEATURES)]
    y_train = model_df[target_column]
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=int(settings["n_estimators"]), max_depth=None if int(settings["max_depth"]) <= 0 else int(settings["max_depth"]), min_samples_leaf=int(settings["min_samples_leaf"]), max_features=float(settings["max_features"]), random_state=42, n_jobs=-1)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=int(settings["n_estimators"]), max_depth=None if int(settings["max_depth"]) <= 0 else int(settings["max_depth"]), min_samples_leaf=int(settings["min_samples_leaf"]), max_features=float(settings["max_features"]), random_state=42, n_jobs=-1)
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=int(settings["n_estimators"]), learning_rate=float(settings["learning_rate"]), num_leaves=int(settings["num_leaves"]), max_depth=int(settings["max_depth"]), random_state=42, verbosity=-1)
    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=int(settings["iterations"]), depth=int(settings["depth"]), learning_rate=float(settings["learning_rate"]), l2_leaf_reg=float(settings["l2_leaf_reg"]), loss_function="RMSE", verbose=False, random_seed=42)
    else:
        raise ForecastingError(f"Unsupported model: {model_name}")
    model.fit(x_train, y_train)
    prediction_df = df.dropna(subset=list(M2_FEATURES)).copy().reset_index(drop=True)
    predictions = np.clip(model.predict(prediction_df[list(M2_FEATURES)]), 0.0, None)
    return pd.DataFrame({"_ts": prediction_df["_ts"].values, "measured": prediction_df["power_average_w_normalized"].values, "_pred": predictions})


def _fit_model(df: pd.DataFrame, model_name: str, test_size: float, model_settings: dict | None, horizon_minutes: int):
    """Fit one estimator for one horizon and retain it for later inference."""
    m2 = _load_m2_module()
    settings = dict(DEFAULT_MODEL_SETTINGS.get(model_name, {}))
    settings.update(model_settings or {})
    horizon_df, target_column = _add_horizon_target(df, horizon_minutes)
    train_rows = max(60, min(len(df) - 1, int(len(df) * (1.0 - test_size))))
    train_df = horizon_df.iloc[:train_rows].copy()
    max_training_rows = int(settings.get("max_training_rows", 12000))
    if len(train_df) > max_training_rows > 0:
        train_df = train_df.tail(max_training_rows).reset_index(drop=True)
    if model_name == "xgboost":
        return m2.train_xgb_model(train_df, test_size=max(1.0 / len(train_df), 0.001), settings=settings, target_column=target_column)
    if model_name == "lstm":
        try:
            return m2.train_lstm_model(train_df, test_size=max(1.0 / len(train_df), 0.001), seq_len=int(settings.get("sequence_length", 12)), epochs=int(settings.get("epochs", 8)), batch_size=int(settings.get("batch_size", 64)), target_column=target_column)
        except Exception as exc:
            raise ForecastingError(f"LSTM training failed: {exc}") from exc
    model_df = train_df.dropna(subset=list(M2_FEATURES) + [target_column]).copy()
    if len(model_df) < 50:
        raise ForecastingError(f"Not enough rows to train {MODEL_SPECS[model_name]['label']}.")
    x_train = model_df[list(M2_FEATURES)]
    y_train = model_df[target_column]
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=int(settings["n_estimators"]), max_depth=None if int(settings["max_depth"]) <= 0 else int(settings["max_depth"]), min_samples_leaf=int(settings["min_samples_leaf"]), max_features=float(settings["max_features"]), random_state=42, n_jobs=-1)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=int(settings["n_estimators"]), max_depth=None if int(settings["max_depth"]) <= 0 else int(settings["max_depth"]), min_samples_leaf=int(settings["min_samples_leaf"]), max_features=float(settings["max_features"]), random_state=42, n_jobs=-1)
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=int(settings["n_estimators"]), learning_rate=float(settings["learning_rate"]), num_leaves=int(settings["num_leaves"]), max_depth=int(settings["max_depth"]), random_state=42, verbosity=-1)
    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=int(settings["iterations"]), depth=int(settings["depth"]), learning_rate=float(settings["learning_rate"]), l2_leaf_reg=float(settings["l2_leaf_reg"]), loss_function="RMSE", verbose=False, random_seed=42)
    else:
        raise ForecastingError(f"Unsupported model: {model_name}")
    model.fit(x_train, y_train)
    return model


def train_model(request: dict[str, Any]) -> dict[str, Any]:
    artifact = get_artifact(request.get("artifact_id"))
    horizons = _parse_horizons(request.get("horizons"))
    models = _normalize_models(request.get("models"), request.get("model"))
    validate_forecast_input(artifact, horizons, models=models)
    df = _load_artifact_frame(artifact)
    test_size = float(request.get("test_size", artifact.metadata.get("config", {}).get("test_size", 0.2)))
    configured = request.get("model_settings") or {}
    effective_settings = {
        model_name: {**DEFAULT_MODEL_SETTINGS.get(model_name, {}), **(configured.get(model_name) or {})}
        for model_name in models
    }
    bundle = {"artifact_id": artifact.artifact_id, "metadata": artifact.metadata, "horizons": horizons, "models": models, "model_settings": effective_settings, "capacity_kw": float(request.get("capacity_kw") or artifact.metadata.get("capacity_kw") or 1.0), "estimators": {}}
    for model_name in models:
        bundle["estimators"][model_name] = {
            horizon: _fit_model(df, model_name, test_size, configured.get(model_name), horizon)
            for horizon in horizons
        }
    token = uuid.uuid4().hex
    _TRAINED_MODELS[token] = bundle
    return {"model_token": token, "artifact_id": artifact.artifact_id, "horizons": horizons, "models": models, "model_settings": effective_settings, "metadata": artifact.metadata, "trained_at": datetime.now().isoformat(timespec="seconds")}


def _bundle_payload(bundle: dict[str, Any]) -> bytes:
    stream = io.BytesIO()
    joblib.dump(bundle, stream)
    return stream.getvalue()


def serialize_trained_model(token: str) -> bytes:
    bundle = _TRAINED_MODELS.get(token)
    if not bundle:
        raise ForecastingError("Train or upload a model before saving it.")
    return _bundle_payload(bundle)


def load_trained_model(contents: bytes) -> dict[str, Any]:
    try:
        bundle = joblib.load(io.BytesIO(contents))
    except Exception as exc:
        raise ForecastingError(f"Could not load the trained model file: {exc}") from exc
    required = {"artifact_id", "horizons", "models", "estimators"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ForecastingError("The uploaded file is not a compatible PV forecasting model bundle.")
    token = uuid.uuid4().hex
    _TRAINED_MODELS[token] = bundle
    return {"model_token": token, "artifact_id": bundle["artifact_id"], "horizons": bundle["horizons"], "models": bundle["models"], "model_settings": bundle.get("model_settings", {}), "metadata": bundle.get("metadata", {}), "trained_at": "Uploaded model"}


def predict_trained_model(token: str, request: dict[str, Any]) -> dict[str, Any]:
    context = request.get("_context")
    bundle = context["bundle"] if context else _TRAINED_MODELS.get(token)
    if not bundle:
        raise ForecastingError("Train or upload a trained model before predicting.")
    artifact = context["artifact"] if context else get_artifact(bundle["artifact_id"])
    horizons = _parse_horizons(request.get("horizons") or bundle["horizons"])
    models = list(bundle["models"])
    df = context["frame"] if context else _load_artifact_frame(artifact)
    timestamp = pd.to_datetime(request.get("timestamp"), errors="coerce")
    if pd.isna(timestamp):
        raise ForecastingError("Enter a valid forecast date and time.")
    # The year is a user-selected prediction year, not a training constraint.
    # Reuse the trained calendar month/day and nearest available sample time
    # so a model trained on one year can predict the same seasonal date later.
    same_calendar_day = df[(df["_ts"].dt.month == timestamp.month) & (df["_ts"].dt.day == timestamp.day)]
    if same_calendar_day.empty:
        raise ForecastingError("The trained model has no data for the selected month and day.")
    time_delta = (same_calendar_day["_ts"].dt.hour * 3600 + same_calendar_day["_ts"].dt.minute * 60 + same_calendar_day["_ts"].dt.second) - (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second)
    source = same_calendar_day.loc[[time_delta.abs().idxmin()]]
    source_timestamp = source.iloc[0]["_ts"]
    if source.empty:
        raise ForecastingError("The selected date and time is not available in the preprocessing artifact.")
    results = {}
    for model_name in models:
        records = {"_ts": timestamp.isoformat(), "source_timestamp": source_timestamp.isoformat(), "measured": float(source.iloc[0]["power_average_w_normalized"]), "measured_kw": float(source.iloc[0]["power_average_w_normalized"]) * bundle.get("capacity_kw", 1.0)}
        for horizon in horizons:
            estimator = bundle["estimators"].get(model_name, {}).get(horizon)
            if estimator is None:
                raise ForecastingError(f"The uploaded model does not contain a +{horizon} minute estimator.")
            if model_name == "lstm":
                prediction = _load_m2_module().predict_lstm(df, estimator)
                row = prediction[prediction["_ts"] == source_timestamp]
                value = float(row.iloc[0]["_pred"]) if not row.empty else None
            else:
                value = float(np.clip(estimator.predict(source[list(M2_FEATURES)])[0], 0.0, None))
            records[f"pred_{horizon}m"] = value
            records[f"pred_{horizon}m_kw"] = None if value is None else value * bundle.get("capacity_kw", 1.0)
        results[model_name] = {"records": [records]}
    return {"artifact_id": artifact.artifact_id, "horizons": horizons, "models": models, "model": models[0] if len(models) == 1 else "comparison", "metadata": artifact.metadata, "capacity_kw": bundle.get("capacity_kw", 1.0), "latest_timestamp": timestamp.isoformat(), "generated_at": datetime.now().isoformat(timespec="seconds"), "results": results}


def predict_trained_model_range(token: str, request: dict[str, Any]) -> dict[str, Any]:
    """Generate timestamp predictions across a user-selected time range."""
    start = pd.to_datetime(request.get("start"), errors="coerce")
    end = pd.to_datetime(request.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ForecastingError("Enter a valid forecast start and end date and time.")
    if end < start:
        raise ForecastingError("Forecast end must be after the forecast start.")

    bundle = _TRAINED_MODELS.get(token)
    if not bundle:
        raise ForecastingError("Train or upload a trained model before predicting.")
    artifact = get_artifact(bundle["artifact_id"])
    frame = _load_artifact_frame(artifact).copy()
    interval = float(artifact.metadata.get("sampling_interval_minutes") or 5)
    timestamps = pd.date_range(start=start, end=end, freq=pd.Timedelta(minutes=interval))
    if len(timestamps) > 5000:
        raise ForecastingError("Select a shorter forecast range (maximum 5,000 points).")

    requested_horizons = _parse_horizons(request.get("horizons") or bundle["horizons"])
    horizons = [h for h in requested_horizons if h in bundle["horizons"]]
    if not horizons:
        raise ForecastingError("None of the selected horizons are trained in this model.")

    models = list(bundle["models"])

    # Pre-compute predictions for all models and horizons on the entire frame
    # to avoid running slow models (especially PyTorch LSTM) in a timestamp loop.
    pred_columns = {}
    for model_name in models:
        pred_columns[model_name] = {}
        for horizon in horizons:
            estimator = bundle["estimators"].get(model_name, {}).get(horizon)
            if estimator is None:
                continue
            if model_name == "lstm":
                pred_df = _load_m2_module().predict_lstm(frame, estimator)
                pred_series = pred_df.set_index("_ts")["_pred"]
                pred_columns[model_name][horizon] = frame["_ts"].map(pred_series).to_numpy()
            else:
                pred_raw = estimator.predict(frame[list(M2_FEATURES)])
                pred_columns[model_name][horizon] = np.clip(pred_raw, 0.0, None)

    results_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    skipped_points = 0

    # Build calendar lookup mapping (month, day) -> group of rows in frame
    frame["_month"] = frame["_ts"].dt.month
    frame["_day"] = frame["_ts"].dt.day
    frame["_day_seconds"] = frame["_ts"].dt.hour * 3600 + frame["_ts"].dt.minute * 60 + frame["_ts"].dt.second

    lookup_groups = {name: group for name, group in frame.groupby(["_month", "_day"])}

    for timestamp in timestamps:
        month_day = (timestamp.month, timestamp.day)
        group = lookup_groups.get(month_day)
        if group is None or group.empty:
            skipped_points += 1
            continue

        t_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        idx = (group["_day_seconds"] - t_seconds).abs().idxmin()
        source_row = frame.loc[idx]
        source_timestamp = source_row["_ts"]
        measured_val = float(source_row["power_average_w_normalized"])
        measured_kw = measured_val * bundle.get("capacity_kw", 1.0)

        for model_name in models:
            records = {
                "_ts": timestamp.isoformat(),
                "source_timestamp": source_timestamp.isoformat(),
                "measured": measured_val,
                "measured_kw": measured_kw,
            }
            for horizon in horizons:
                val = pred_columns[model_name][horizon][idx]
                records[f"pred_{horizon}m"] = float(val) if pd.notna(val) else None
                records[f"pred_{horizon}m_kw"] = None if pd.isna(val) else float(val) * bundle.get("capacity_kw", 1.0)
                target_timestamp = source_timestamp + pd.Timedelta(minutes=horizon)
                target_group = lookup_groups.get((target_timestamp.month, target_timestamp.day))
                if target_group is not None and not target_group.empty:
                    target_seconds = target_timestamp.hour * 3600 + target_timestamp.minute * 60 + target_timestamp.second
                    target_idx = (target_group["_day_seconds"] - target_seconds).abs().idxmin()
                    records[f"measured_at_{horizon}m"] = float(frame.loc[target_idx, "power_average_w_normalized"])
                else:
                    records[f"measured_at_{horizon}m"] = None
            results_by_model[model_name].append(records)

    if not any(results_by_model.values()):
        raise ForecastingError("No forecast points could be generated for the selected range.")

    return {
        "artifact_id": artifact.artifact_id,
        "horizons": horizons,
        "models": models,
        "model": models[0] if len(models) == 1 else "comparison",
        "metadata": artifact.metadata,
        "capacity_kw": bundle.get("capacity_kw", 1.0),
        "latest_timestamp": end.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_start": start.isoformat(),
        "source_end": end.isoformat(),
        "skipped_points": skipped_points,
        "results": {
            model: {
                "records": records,
                "metrics": _metrics(pd.DataFrame(records), horizons, bundle.get("capacity_kw", 1.0), False),
            }
            for model, records in results_by_model.items()
        },
    }


def _build_horizon_frame(
    predictions_by_horizon: dict[int, pd.DataFrame],
    source: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Combine independently generated horizon predictions into one frame."""
    source_index = source.set_index("_ts")
    measured = source_index["power_average_w_normalized"].to_dict()
    ghi_values = source_index["ghi_pyr"].to_dict() if "ghi_pyr" in source_index.columns else {}
    prediction_maps = {
        int(minutes): frame.set_index("_ts")["_pred"].to_dict()
        for minutes, frame in predictions_by_horizon.items()
    }
    timestamps = sorted({
        pd.Timestamp(timestamp)
        for frame in predictions_by_horizon.values()
        for timestamp in frame.get("_ts", pd.Series(dtype="datetime64[ns]")).dropna()
    })
    rows = []
    for timestamp in timestamps:
        row: dict[str, Any] = {
            "_ts": timestamp,
            "measured": measured.get(timestamp, np.nan),
        }
        first_prediction = np.nan
        for minutes in horizons:
            value = prediction_maps.get(int(minutes), {}).get(timestamp, np.nan)
            value = float(max(0.0, value)) if pd.notna(value) else np.nan
            row[f"pred_{minutes}m"] = value
            row[f"measured_at_{minutes}m"] = measured.get(
                timestamp + pd.Timedelta(minutes=minutes), np.nan
            )
            if pd.isna(first_prediction) and pd.notna(value):
                first_prediction = value
        # Keep the legacy summary field for existing exported artifacts while
        # the horizon-specific columns carry the actual predictions.
        row["prediction"] = first_prediction
        if ghi_values:
            row["ghi_pyr"] = ghi_values.get(timestamp, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("_ts").reset_index(drop=True)


def _metrics(frame: pd.DataFrame, horizons: list[int], capacity_kw: float, daytime_only: bool) -> list[dict[str, Any]]:
    rows = []
    for minutes in horizons:
        actual_col = f"measured_at_{minutes}m"
        pred_col = f"pred_{minutes}m"
        valid = frame[[actual_col, pred_col]].dropna()
        if daytime_only and "ghi_pyr" in frame.columns:
            valid = frame.loc[valid.index].loc[frame.loc[valid.index, "ghi_pyr"] > 10, [actual_col, pred_col]].dropna()
        n = len(valid)
        if n == 0:
            rows.append({"horizon_minutes": minutes, "mae": None, "rmse": None, "r2": None, "bias": None, "n": 0, "nmae": None})
            continue
        y_true = valid[actual_col].to_numpy(dtype=float)
        y_pred = valid[pred_col].to_numpy(dtype=float)
        errors = y_pred - y_true
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        mae = float(np.abs(errors).mean())
        rows.append({
            "horizon_minutes": minutes,
            "mae": mae,
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "r2": float(1.0 - (errors ** 2).sum() / ss_tot) if ss_tot > 1e-12 else None,
            "bias": float(errors.mean()),
            "n": int(n),
            "nmae": float(mae / max(abs(float(y_true.mean())), 1e-12)),
            "mae_kw": float(mae * capacity_kw),
        })
    return rows


def _json_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    output["_ts"] = output["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    output = output.replace({np.nan: None})
    return output.to_dict("records")


def _run(request: dict[str, Any], latest_only: bool = False) -> dict[str, Any]:
    system_id = request.get("system_id")
    # The artifact selector intentionally spans all systems.  The selected
    # artifact is the source of truth for system metadata and capacity.
    artifact = get_artifact(request.get("artifact_id"))
    horizons = _parse_horizons(request.get("horizons"))
    selected_models = _normalize_models(request.get("models"), request.get("model"))
    validate_forecast_input(artifact, horizons, models=selected_models)
    df = _load_artifact_frame(artifact)
    metadata = artifact.metadata
    capacity_kw = float(request.get("capacity_kw") or metadata.get("capacity_kw") or 1.0)
    configured_settings = request.get("model_settings") or {}
    test_size = float(request.get("test_size", metadata.get("config", {}).get("test_size", 0.2)))
    model_names = selected_models
    results: dict[str, Any] = {}
    for model_name in model_names:
        try:
            predictions_by_horizon = {
                horizon: _train_and_predict(
                    df,
                    model_name,
                    test_size,
                    configured_settings.get(model_name),
                    horizon,
                )
                for horizon in horizons
            }
        except Exception as exc:
            results[model_name] = {"error": f"{MODEL_SPECS[model_name]['label']} failed: {exc}"}
            continue
        frame = _build_horizon_frame(predictions_by_horizon, df, horizons)
        if not latest_only:
            start = pd.to_datetime(request.get("start"), errors="coerce") if request.get("start") else None
            end = pd.to_datetime(request.get("end"), errors="coerce") if request.get("end") else None
            if start is not None and not pd.isna(start):
                frame = frame[frame["_ts"] >= start]
            if end is not None and not pd.isna(end):
                frame = frame[frame["_ts"] <= end]
        daylight_record = None
        if latest_only and "ghi_pyr" in frame.columns:
            daylight = frame[pd.to_numeric(frame["ghi_pyr"], errors="coerce") > 10].tail(1)
            if not daylight.empty:
                daylight_record = _json_frame(daylight)[0]
        if latest_only:
            frame = frame.tail(1).copy()
        frame["measured_kw"] = frame["measured"] * capacity_kw
        for minutes in horizons:
            frame[f"pred_{minutes}m_kw"] = frame[f"pred_{minutes}m"] * capacity_kw
        metrics_frame = frame if latest_only is False else _build_horizon_frame(predictions_by_horizon, df, horizons)
        results[model_name] = {
            "records": _json_frame(frame),
            "daylight_record": daylight_record,
            "metrics": _metrics(metrics_frame, horizons, capacity_kw, bool(request.get("daytime_only"))),
        }
    latest_timestamp = df["_ts"].max()
    return {
        "artifact_id": artifact.artifact_id,
        "metadata": metadata,
        "horizons": horizons,
        "model": model_names[0] if len(model_names) == 1 else "comparison",
        "models": model_names,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_timestamp": latest_timestamp.isoformat(),
        "capacity_kw": capacity_kw,
        "source_rows": int(len(df)),
        "source_start": df["_ts"].min().isoformat(),
        "source_end": latest_timestamp.isoformat(),
        "results": results,
    }


def run_backtest(request: dict[str, Any]) -> dict[str, Any]:
    return _run(request, latest_only=False)


def generate_forecast(request: dict[str, Any]) -> dict[str, Any]:
    return _run(request, latest_only=True)


def build_forecast_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, payload in result.get("results", {}).items():
        for row in payload.get("metrics", []):
            rows.append({"model": model, **row})
    return rows


def export_forecast_result(result: dict[str, Any]) -> dict[str, str]:
    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"pv_forecast_{result.get('metadata', {}).get('system_id', 'system')}_{stamp}"
    csv_paths = []
    for model, payload in result.get("results", {}).items():
        if payload.get("records"):
            path = FORECAST_OUTPUT_DIR / f"{stem}_{model}.csv"
            pd.DataFrame(payload["records"]).to_csv(path, index=False)
            csv_paths.append(str(path))
    metrics_path = FORECAST_OUTPUT_DIR / f"{stem}_metrics.csv"
    pd.DataFrame(build_forecast_metrics(result)).to_csv(metrics_path, index=False)
    summary_path = FORECAST_OUTPUT_DIR / f"{stem}_summary.json"
    summary = dict(result)
    summary["forecast_csv_paths"] = csv_paths
    summary["metrics_csv_path"] = str(metrics_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {"forecast_csv": ", ".join(csv_paths), "metrics_csv": str(metrics_path), "summary_json": str(summary_path)}

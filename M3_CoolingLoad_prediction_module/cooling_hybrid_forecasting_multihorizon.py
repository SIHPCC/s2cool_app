"""
Hybrid cooling-load requirement model for medium-size datacenter halls.

Methodology implemented:
1) Weather + solar geometry -> GPOA (same pvlib approach used in pv_hybrid_forecasting.py)
2) Physics cooling model -> Q_phys
3) Measurement model from air-side enthalpy difference -> Q_measured
4) Residual learning -> r = Q_measured - Q_phys using XGBoost/GBR
5) Hybrid forecast -> Q_hybrid = Q_phys + f_ML(weather, state)
6) Optional PV coupling -> P_grid = Q_hybrid - P_pv

This script supports multi-horizon forecasting (default: 10, 20, 30 min).

Example:
    python cooling_demand/cooling_hybrid_forecasting_multihorizon.py --input_csv "data_generation/example_with_weather_and_hvac.csv" --lat 31.581052 --lon 74.359936 --supply_temp_col supply_air_temperature --supply_rh_col supply_air_rh --return_temp_col return_air_temperature --return_rh_col return_air_rh --airflow_col airflow_m3_s
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import pvlib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pvlib is required. Install with: pip install pvlib"
    ) from exc

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
    from sklearn.ensemble import GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = {"10m": 10, "20m": 20, "30m": 30}
AIR_DENSITY = 1.2  # kg/m3
CP_AIR_KJ_KGK = 1.005


@dataclass
class CoolingConfig:
    lat: float
    lon: float
    surface_tilt: float
    surface_azimuth: float
    shgc: float
    window_area_m2: float
    r_env_kw_per_k: float
    r_int_kw_per_k: float
    c_air_kj_per_k: float
    q_internal_kw: float
    test_size: float

@dataclass
class DatacenterHallSpec:
    room_length_m: float
    room_width_m: float
    room_height_m: float
    window_count: int
    window_area_each_m2: float
    people_count: int
    sensible_w_per_person: float
    latent_w_per_person: float
    lighting_w_per_m2: float
    non_it_misc_kw: float
    it_load_kw: float


# ---------------------------------------------------------------------------
# Time + psychrometrics
# ---------------------------------------------------------------------------
def parse_datetime(df: pd.DataFrame, date_col: str = "date", time_col: str = "time") -> pd.Series:
    raw = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()

    dt = pd.to_datetime(raw, format="%d/%m/%y %I:%M%p", errors="coerce")

    mask = dt.isna()
    if mask.any():
        dt[mask] = pd.to_datetime(raw[mask], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    mask = dt.isna()
    if mask.any():
        dt[mask] = pd.to_datetime(raw[mask], dayfirst=True, errors="coerce")

    return dt


def saturation_pressure_kpa(t_c: pd.Series) -> pd.Series:
    return 0.61078 * np.exp((17.2694 * t_c) / (t_c + 237.3))


def humidity_ratio_kgkg(t_c: pd.Series, rh_pct: pd.Series, p_kpa: float = 101.325) -> pd.Series:
    pv = (rh_pct.clip(lower=0, upper=100) / 100.0) * saturation_pressure_kpa(t_c)
    return 0.622 * pv / (p_kpa - pv).clip(lower=1e-6)


def enthalpy_kjkg(t_c: pd.Series, w: pd.Series) -> pd.Series:
    return CP_AIR_KJ_KGK * t_c + w * (2501.0 + 1.86 * t_c)


# ---------------------------------------------------------------------------
# Data loading + standardization
# ---------------------------------------------------------------------------
def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        c_lower = str(c).lower()
        if c_lower in lower_map:
            return str(lower_map[c_lower])
    return ""


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "_ts" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["_ts"] = parse_datetime(df)
        elif "time" in df.columns:
            df["_ts"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            raise RuntimeError("No usable timestamp columns found. Need either _ts or date/time.")

    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    return df


def numeric_fill(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def regularize_to_5min(df: pd.DataFrame, enable: bool) -> pd.DataFrame:
    if not enable:
        return df.copy().sort_values("_ts").reset_index(drop=True)

    out = df.copy().sort_values("_ts").reset_index(drop=True)
    out = out.set_index("_ts")
    out = out[~out.index.duplicated(keep="last")]
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="5min")
    out = out.reindex(full_index)
    out.index.name = "_ts"

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = [c for c in out.columns if c not in numeric_cols]

    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].interpolate(method="time").ffill().bfill()
    if non_num_cols:
        out[non_num_cols] = out[non_num_cols].ffill().bfill()

    return out.reset_index().rename(columns={"index": "_ts"})


def filter_month(df: pd.DataFrame, month_yyyy_mm: str) -> pd.DataFrame:
    dt = pd.to_datetime(df["_ts"], errors="coerce")
    mask = dt.dt.strftime("%Y-%m") == month_yyyy_mm
    out = df.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No rows found for requested train month: {month_yyyy_mm}")
    return out


def select_consecutive_test_window(
    df: pd.DataFrame,
    test_days: int,
    test_start_date: str,
) -> tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    if test_days < 1:
        raise RuntimeError("test_days must be >= 1")

    start_ts: Optional[pd.Timestamp] = None
    if test_start_date:
        parsed = pd.to_datetime(test_start_date, errors="coerce")
        if pd.isna(parsed):
            raise RuntimeError("Invalid --test_start_date. Use YYYY-MM-DD")
        start_ts = pd.Timestamp(parsed).normalize()
    else:
        # Default: latest consecutive block in the selected month.
        start_ts = pd.Timestamp(df["_ts"].max()).normalize() - pd.Timedelta(days=test_days - 1)

    end_ts = start_ts + pd.Timedelta(days=test_days)
    ts = pd.to_datetime(df["_ts"], errors="coerce")
    test_mask = (ts >= start_ts) & (ts < end_ts)
    train_mask = ts < start_ts

    if int(test_mask.sum()) < 1:
        raise RuntimeError("Selected test window has no rows. Choose a different --test_start_date.")
    if int(train_mask.sum()) < 50:
        raise RuntimeError("Not enough training rows before selected test window.")

    return train_mask, test_mask, start_ts, end_ts


def derive_hall_parameters(spec: DatacenterHallSpec) -> dict:
    floor_area_m2 = spec.room_length_m * spec.room_width_m
    volume_m3 = floor_area_m2 * spec.room_height_m
    window_area_m2 = max(0.0, spec.window_count * spec.window_area_each_m2)
    people_kw = spec.people_count * (spec.sensible_w_per_person + spec.latent_w_per_person) / 1000.0
    lighting_kw = floor_area_m2 * spec.lighting_w_per_m2 / 1000.0
    q_internal_kw = people_kw + lighting_kw + spec.non_it_misc_kw + spec.it_load_kw
    return {
        "floor_area_m2": floor_area_m2,
        "volume_m3": volume_m3,
        "window_area_m2": window_area_m2,
        "people_kw": people_kw,
        "lighting_kw": lighting_kw,
        "q_internal_kw": q_internal_kw,
    }


# ---------------------------------------------------------------------------
# Physics model blocks
# ---------------------------------------------------------------------------
def compute_gpoa(df: pd.DataFrame, cfg: CoolingConfig) -> pd.DataFrame:
    dni_col = find_first_existing(df, ["dni"])
    dhi_col = find_first_existing(df, ["dhi"])
    ghi_col = find_first_existing(df, ["ghi_pyr", "ghi"])

    loc = pvlib.location.Location(cfg.lat, cfg.lon)
    ts_index = pd.DatetimeIndex(df["_ts"])
    solpos = loc.get_solarposition(ts_index)

    out = df.copy()
    out["solar_zenith"] = solpos["zenith"].values
    out["solar_azimuth"] = solpos["azimuth"].values

    if dni_col and dhi_col and ghi_col:
        dni = pd.to_numeric(out[dni_col], errors="coerce")
        dhi = pd.to_numeric(out[dhi_col], errors="coerce")
        ghi = pd.to_numeric(out[ghi_col], errors="coerce")
    else:
        # Fallback when measured irradiance channels are unavailable.
        clearsky = loc.get_clearsky(ts_index)
        dni = pd.to_numeric(clearsky["dni"], errors="coerce")
        dhi = pd.to_numeric(clearsky["dhi"], errors="coerce")
        ghi = pd.to_numeric(clearsky["ghi"], errors="coerce")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=cfg.surface_tilt,
        surface_azimuth=cfg.surface_azimuth,
        dni=dni.to_numpy(),
        ghi=ghi.to_numpy(),
        dhi=dhi.to_numpy(),
        solar_zenith=out["solar_zenith"].to_numpy(),
        solar_azimuth=out["solar_azimuth"].to_numpy(),
    )
    poa_global = pd.Series(poa["poa_global"], index=out.index)
    out["Gpoa"] = pd.to_numeric(poa_global, errors="coerce").fillna(0.0).clip(lower=0)
    return out


def compute_q_measured(
    df: pd.DataFrame,
    supply_temp_col: str,
    supply_rh_col: str,
    return_temp_col: str,
    return_rh_col: str,
    airflow_col: str,
    airflow_unit: str,
    measured_q_col: str,
) -> pd.DataFrame:
    out = df.copy()

    q_col = measured_q_col if measured_q_col in out.columns else ""
    if not q_col:
        q_col = find_first_existing(
            out,
            ["Q_measured_kW", "Q_measured_W", "Q_measured", "q_measured_kw", "q_measured_w"],
        )

    if q_col:
        q_values = pd.to_numeric(out[q_col], errors="coerce")
        if str(q_col).lower().endswith("_w"):
            q_values = q_values / 1000.0
        out["Q_measured_kW"] = q_values
        return out

    needed = [supply_temp_col, supply_rh_col, return_temp_col, return_rh_col, airflow_col]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise RuntimeError(
            "Missing columns for Q_measured from enthalpy model: "
            f"{missing}. Provide --measured_q_col or all psychrometric columns."
        )

    numeric_fill(out, needed)

    w_sup = humidity_ratio_kgkg(out[supply_temp_col], out[supply_rh_col])
    w_ret = humidity_ratio_kgkg(out[return_temp_col], out[return_rh_col])
    h_sup = enthalpy_kjkg(out[supply_temp_col], w_sup)
    h_ret = enthalpy_kjkg(out[return_temp_col], w_ret)

    if airflow_unit.lower() == "kg_s":
        m_dot = out[airflow_col].clip(lower=0)
    elif airflow_unit.lower() == "cfm":
        m_dot = out[airflow_col].clip(lower=0) * 0.00047194745 * AIR_DENSITY
    else:
        # Default m3/s
        m_dot = out[airflow_col].clip(lower=0) * AIR_DENSITY

    # kW = kg/s * (kJ/kg)
    out["Q_measured_kW"] = m_dot * (h_ret - h_sup)
    out["Q_measured_kW"] = out["Q_measured_kW"].clip(lower=0)
    return out


def compute_q_physics(df: pd.DataFrame, cfg: CoolingConfig) -> pd.DataFrame:
    out = df.copy()

    t_out_col = find_first_existing(out, ["air_temperature", "ambient_temperature", "t_out", "T_out"])
    t_in_col = find_first_existing(out, ["indoor_temperature", "zone_temperature", "t_in", "T_in"])
    t_wall_col = find_first_existing(out, ["wall_temperature", "t_wall"])

    if not t_out_col:
        raise RuntimeError("Need outdoor temperature column: air_temperature / ambient_temperature / t_out")
    if not t_in_col:
        raise RuntimeError("Need indoor temperature column: indoor_temperature / zone_temperature / t_in")
    if not t_wall_col:
        # fallback: if wall is unavailable, use indoor for zero wall transfer delta
        out["wall_temperature"] = pd.to_numeric(out[t_in_col], errors="coerce")
        t_wall_col = "wall_temperature"

    q_internal_col = find_first_existing(out, ["q_internal_kw", "it_load_kw", "internal_gains_kw"])

    numeric_fill(out, [t_out_col, t_in_col, t_wall_col])
    if q_internal_col:
        out[q_internal_col] = pd.to_numeric(out[q_internal_col], errors="coerce")

    dt_seconds = out["_ts"].diff().dt.total_seconds().replace(0, np.nan)
    dt_seconds = dt_seconds.fillna(dt_seconds.median() if dt_seconds.notna().any() else 300.0)

    d_tin_dt = out[t_in_col].diff() / dt_seconds
    d_tin_dt = d_tin_dt.fillna(0.0)
    out["d_tin_dt_k_per_s"] = d_tin_dt

    out["Q_solar_kW"] = cfg.shgc * cfg.window_area_m2 * out["Gpoa"] / 1000.0

    q_internal = out[q_internal_col].fillna(cfg.q_internal_kw) if q_internal_col else cfg.q_internal_kw

    # R terms are configured as K/kW for direct kW balance.
    out["Q_env_kW"] = (out[t_out_col] - out[t_in_col]) / cfg.r_env_kw_per_k
    out["Q_wall_kW"] = (out[t_wall_col] - out[t_in_col]) / cfg.r_int_kw_per_k

    # Dynamic term (kW): C_air(kJ/K) * dT/dt(K/s) = kJ/s = kW
    out["Q_dynamic_kW"] = cfg.c_air_kj_per_k * d_tin_dt

    out["Q_phys_kW"] = out["Q_env_kW"] + out["Q_wall_kW"] + q_internal + out["Q_solar_kW"] - out["Q_dynamic_kW"]
    out["Q_phys_kW"] = out["Q_phys_kW"].clip(lower=0)

    return out


def _calibration_objective(
    df: pd.DataFrame,
    r_env_kw_per_k: float,
    r_int_kw_per_k: float,
    c_air_kj_per_k: float,
    shgc: float,
    window_area_m2: float,
) -> pd.Series:
    q_solar = shgc * window_area_m2 * df["Gpoa"] / 1000.0
    q_env = (df["_t_out"] - df["_t_in"]) / r_env_kw_per_k
    q_wall = (df["_t_wall"] - df["_t_in"]) / r_int_kw_per_k
    q_dyn = c_air_kj_per_k * df["_d_tin_dt"]
    q_phys = q_env + q_wall + df["_q_internal"] + q_solar - q_dyn
    return q_phys.clip(lower=0)


def auto_calibrate_thermal_parameters(
    df: pd.DataFrame,
    cfg: CoolingConfig,
    train_mask: pd.Series,
    val_mask: pd.Series,
    n_iter: int = 600,
    seed: int = 42,
):
    required = ["Q_measured_kW", "Gpoa"]
    if any(c not in df.columns for c in required):
        raise RuntimeError("Auto-calibration requires Q_measured_kW and Gpoa columns.")

    work = df.copy()
    t_out_col = find_first_existing(work, ["air_temperature", "ambient_temperature", "t_out", "T_out"])
    t_in_col = find_first_existing(work, ["indoor_temperature", "zone_temperature", "t_in", "T_in"])
    t_wall_col = find_first_existing(work, ["wall_temperature", "t_wall"])
    q_internal_col = find_first_existing(work, ["q_internal_kw", "it_load_kw", "internal_gains_kw"])

    if not t_out_col or not t_in_col:
        raise RuntimeError("Auto-calibration needs outdoor and indoor temperature columns.")
    if not t_wall_col:
        work["_t_wall"] = pd.to_numeric(work[t_in_col], errors="coerce")
    else:
        work["_t_wall"] = pd.to_numeric(work[t_wall_col], errors="coerce")

    work["_t_out"] = pd.to_numeric(work[t_out_col], errors="coerce")
    work["_t_in"] = pd.to_numeric(work[t_in_col], errors="coerce")

    if q_internal_col:
        work["_q_internal"] = pd.to_numeric(work[q_internal_col], errors="coerce").fillna(cfg.q_internal_kw)
    else:
        work["_q_internal"] = cfg.q_internal_kw

    dt_seconds = work["_ts"].diff().dt.total_seconds().replace(0, np.nan)
    dt_seconds = dt_seconds.fillna(dt_seconds.median() if dt_seconds.notna().any() else 300.0)
    work["_d_tin_dt"] = (work["_t_in"].diff() / dt_seconds).fillna(0.0)

    valid = work[["Q_measured_kW", "Gpoa", "_t_out", "_t_in", "_t_wall", "_q_internal", "_d_tin_dt"]].notna().all(axis=1)
    if int(valid.sum()) < 150:
        raise RuntimeError("Not enough valid rows for calibration. Need at least 150 rows.")

    train_mask = valid & train_mask
    val_mask = valid & val_mask

    if int(train_mask.sum()) < 100:
        raise RuntimeError("Not enough calibration training rows after split. Reduce test_size or add data.")

    # Reasonable bounds for medium-size datacenter halls.
    bounds = {
        "r_env": (0.03, 1.0),
        "r_int": (0.12, 1.5),
        "c_air": (5000.0, 120000.0),
        "shgc": (0.05, 0.9),
    }

    rng = np.random.default_rng(seed)

    def sample_params() -> tuple[float, float, float, float]:
        r_env = float(np.exp(rng.uniform(np.log(bounds["r_env"][0]), np.log(bounds["r_env"][1]))))
        r_int = float(np.exp(rng.uniform(np.log(bounds["r_int"][0]), np.log(bounds["r_int"][1]))))
        c_air = float(np.exp(rng.uniform(np.log(bounds["c_air"][0]), np.log(bounds["c_air"][1]))))
        shgc = float(rng.uniform(bounds["shgc"][0], bounds["shgc"][1]))
        return r_env, r_int, c_air, shgc

    def score(params: tuple[float, float, float, float]) -> tuple[float, float]:
        r_env, r_int, c_air, shgc = params
        q_phys = _calibration_objective(work, r_env, r_int, c_air, shgc, cfg.window_area_m2)
        e = work.loc[train_mask, "Q_measured_kW"] - q_phys.loc[train_mask]
        rmse = float(np.sqrt(np.mean(np.square(e))))
        mae = float(np.mean(np.abs(e)))
        return rmse, mae

    init = (cfg.r_env_kw_per_k, cfg.r_int_kw_per_k, cfg.c_air_kj_per_k, cfg.shgc)
    best = init
    best_rmse, best_mae = score(best)

    history = [{
        "iter": 0,
        "r_env_kw_per_k": best[0],
        "r_int_kw_per_k": best[1],
        "c_air_kj_per_k": best[2],
        "shgc": best[3],
        "rmse_train": best_rmse,
        "mae_train": best_mae,
    }]

    for i in range(1, max(2, n_iter) + 1):
        p = sample_params()
        rmse, mae = score(p)
        history.append(
            {
                "iter": i,
                "r_env_kw_per_k": p[0],
                "r_int_kw_per_k": p[1],
                "c_air_kj_per_k": p[2],
                "shgc": p[3],
                "rmse_train": rmse,
                "mae_train": mae,
            }
        )
        if rmse < best_rmse:
            best = p
            best_rmse = rmse
            best_mae = mae

    q_best = _calibration_objective(work, best[0], best[1], best[2], best[3], cfg.window_area_m2)

    def _metrics(mask: pd.Series) -> Dict[str, float]:
        if int(mask.sum()) < 2:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": int(mask.sum())}
        y_true = work.loc[mask, "Q_measured_kW"].values
        y_pred = q_best.loc[mask].values
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "n": int(mask.sum()),
        }

    metrics_train = _metrics(train_mask)
    metrics_val = _metrics(val_mask)

    calibrated_cfg = CoolingConfig(
        lat=cfg.lat,
        lon=cfg.lon,
        surface_tilt=cfg.surface_tilt,
        surface_azimuth=cfg.surface_azimuth,
        shgc=best[3],
        window_area_m2=cfg.window_area_m2,
        r_env_kw_per_k=best[0],
        r_int_kw_per_k=best[1],
        c_air_kj_per_k=best[2],
        q_internal_kw=cfg.q_internal_kw,
        test_size=cfg.test_size,
    )

    summary = {
        "best_params": {
            "r_env_kw_per_k": best[0],
            "r_int_kw_per_k": best[1],
            "c_air_kj_per_k": best[2],
            "shgc": best[3],
        },
        "train_metrics": metrics_train,
        "val_metrics": metrics_val,
        "n_iter": int(max(2, n_iter)),
        "seed": int(seed),
    }

    history_df = pd.DataFrame(history).sort_values("rmse_train", ascending=True).reset_index(drop=True)
    return calibrated_cfg, history_df, summary


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["_ts"].dt.hour + out["_ts"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["_ts"].dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["_ts"].dt.dayofweek / 7.0)
    return out


# ---------------------------------------------------------------------------
# Residual ML
# ---------------------------------------------------------------------------
def train_residual_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
):
    model_df = df.dropna(subset=feature_cols + ["Q_measured_kW", "Q_phys_kW"]).copy()
    if len(model_df) < 200:
        raise RuntimeError("Not enough valid rows to train robustly. Need at least 200 samples.")

    model_df["residual"] = model_df["Q_measured_kW"] - model_df["Q_phys_kW"]

    tr_mask = train_mask.reindex(model_df.index).fillna(False)
    te_mask = test_mask.reindex(model_df.index).fillna(False)
    train_df = model_df.loc[tr_mask]

    if len(train_df) < 100:
        raise RuntimeError("Not enough steady-state training rows after filtering.")

    x_train = train_df[feature_cols]
    y_train = train_df["residual"]

    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(x_train, y_train)

    model_df["residual_pred"] = model.predict(model_df[feature_cols])
    model_df["Q_hybrid_kW"] = (model_df["Q_phys_kW"] + model_df["residual_pred"]).clip(lower=0)
    model_df["is_train"] = tr_mask.values
    model_df["is_test"] = te_mask.values

    return model_df, model


def add_prediction_intervals(
    model_df: pd.DataFrame,
    alpha: float = 0.1,
) -> pd.DataFrame:
    out = model_df.copy()
    train_mask = out["is_train"] == True
    residual = (out.loc[train_mask, "Q_measured_kW"] - out.loc[train_mask, "Q_hybrid_kW"]).dropna()
    if len(residual) < 30:
        out["Q_hybrid_lower_kW"] = out["Q_hybrid_kW"]
        out["Q_hybrid_upper_kW"] = out["Q_hybrid_kW"]
        return out

    q_low = float(np.quantile(residual, alpha / 2.0))
    q_high = float(np.quantile(residual, 1.0 - alpha / 2.0))
    out["Q_hybrid_lower_kW"] = (out["Q_hybrid_kW"] + q_low).clip(lower=0)
    out["Q_hybrid_upper_kW"] = (out["Q_hybrid_kW"] + q_high).clip(lower=0)
    return out


def build_multi_horizon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    ts_to_q = out.set_index("_ts")["Q_measured_kW"].to_dict()

    for h in HORIZONS:
        out[f"pred_{h}"] = out["Q_hybrid_kW"]

    if "Q_hybrid_lower_kW" in out.columns and "Q_hybrid_upper_kW" in out.columns:
        for h in HORIZONS:
            out[f"pred_{h}_lower"] = out["Q_hybrid_lower_kW"]
            out[f"pred_{h}_upper"] = out["Q_hybrid_upper_kW"]

    for h, minutes in HORIZONS.items():
        out[f"measured_at_{h}"] = out["_ts"].map(lambda t: ts_to_q.get(t + pd.Timedelta(minutes=minutes), np.nan))

    return out


def _bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        return {
            "mae": (float("nan"), float("nan")),
            "rmse": (float("nan"), float("nan")),
            "r2": (float("nan"), float("nan")),
        }

    mae_v = []
    rmse_v = []
    r2_v = []
    for _ in range(max(100, n_boot)):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        mae_v.append(float(mean_absolute_error(yt, yp)))
        rmse_v.append(float(np.sqrt(mean_squared_error(yt, yp))))
        try:
            r2_v.append(float(r2_score(yt, yp)))
        except Exception:
            r2_v.append(float("nan"))

    def _ci(v: List[float]) -> tuple[float, float]:
        arr = np.array(v, dtype=float)
        return float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))

    return {"mae": _ci(mae_v), "rmse": _ci(rmse_v), "r2": _ci(r2_v)}


def compute_metrics(
    forecast_df: pd.DataFrame,
    eval_mask: Optional[pd.Series] = None,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    if eval_mask is None:
        mask = pd.Series(True, index=forecast_df.index)
    else:
        mask = eval_mask.reindex(forecast_df.index).fillna(False)
    for h in HORIZONS:
        y_col = f"measured_at_{h}"
        p_col = f"pred_{h}"
        valid = forecast_df.loc[mask, [y_col, p_col]].dropna()
        if len(valid) < 2:
            metrics[h] = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
            continue
        y_true = valid[y_col].values
        y_pred = valid[p_col].values
        ci = _bootstrap_metric_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
        metrics[h] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mae_ci_low": ci["mae"][0],
            "mae_ci_high": ci["mae"][1],
            "rmse_ci_low": ci["rmse"][0],
            "rmse_ci_high": ci["rmse"][1],
            "r2_ci_low": ci["r2"][0],
            "r2_ci_high": ci["r2"][1],
            "n": int(len(valid)),
        }
    return metrics


# ---------------------------------------------------------------------------
# Optional PV coupling
# ---------------------------------------------------------------------------
def couple_with_pv(
    forecast_df: pd.DataFrame,
    pv_forecast_csv: str,
    pv_time_col: str,
    pv_power_col: str,
    pv_power_unit: str,
    pv_rated_kw: float,
) -> pd.DataFrame:
    if not pv_forecast_csv:
        return forecast_df

    pv_df = pd.read_csv(pv_forecast_csv)
    if pv_time_col not in pv_df.columns or pv_power_col not in pv_df.columns:
        raise RuntimeError(
            f"PV forecast file must include '{pv_time_col}' and '{pv_power_col}'."
        )

    pv_df = pv_df.copy()
    if pv_time_col == "_ts":
        pv_df["_ts"] = pd.to_datetime(pv_df[pv_time_col], errors="coerce")
    else:
        if "date" in pv_df.columns and "time" in pv_df.columns and pv_time_col == "date_time":
            pv_df["_ts"] = parse_datetime(pv_df, "date", "time")
        else:
            pv_df["_ts"] = pd.to_datetime(pv_df[pv_time_col], errors="coerce")

    pv_df = pv_df.dropna(subset=["_ts"]).copy()
    pv_df["P_pv"] = pd.to_numeric(pv_df[pv_power_col], errors="coerce").fillna(0.0)

    if pv_power_unit.lower() == "w":
        pv_df["P_pv_kW"] = pv_df["P_pv"] / 1000.0
    elif pv_power_unit.lower() == "normalized":
        pv_df["P_pv_kW"] = pv_df["P_pv"].clip(lower=0) * max(pv_rated_kw, 0.0)
    else:
        pv_df["P_pv_kW"] = pv_df["P_pv"]

    keep = pv_df[["_ts", "P_pv_kW"]].drop_duplicates(subset=["_ts"], keep="last")

    out = forecast_df.merge(keep, on="_ts", how="left")
    out["P_pv_kW"] = out["P_pv_kW"].fillna(0.0)
    out["P_grid_kW"] = out["Q_hybrid_kW"] - out["P_pv_kW"]
    out["P_grid_kW"] = out["P_grid_kW"].clip(lower=0)
    if "Q_hybrid_lower_kW" in out.columns and "Q_hybrid_upper_kW" in out.columns:
        out["P_grid_lower_kW"] = (out["Q_hybrid_lower_kW"] - out["P_pv_kW"]).clip(lower=0)
        out["P_grid_upper_kW"] = (out["Q_hybrid_upper_kW"] - out["P_pv_kW"]).clip(lower=0)
    return out


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hybrid cooling-load requirement model: Q = Q_phys + f_ML(residual)."
    )
    p.add_argument("--input_csv", required=True, help="Merged weather + hall data CSV")
    p.add_argument("--out_csv", default="cooling_demand/cooling_hybrid_forecast.csv")

    # Geometry and irradiance
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--surface_tilt", type=float, default=30.0)
    p.add_argument("--surface_azimuth", type=float, default=180.0)

    # Thermal model parameters (tunable per hall)
    p.add_argument("--shgc", type=float, default=0.35, help="Effective SHGC")
    p.add_argument("--window_area_m2", type=float, default=-1.0, help="Equivalent glazed area; <=0 uses window_count*window_area_each_m2")
    p.add_argument("--r_env_kw_per_k", type=float, default=0.12, help="Envelope thermal resistance in K/kW")
    p.add_argument("--r_int_kw_per_k", type=float, default=0.2, help="Internal mass thermal resistance in K/kW")
    p.add_argument("--c_air_kj_per_k", type=float, default=18000.0, help="Effective hall thermal capacitance (kJ/K)")
    p.add_argument("--q_internal_kw", type=float, default=-1.0, help="Fallback internal gains (kW); <=0 uses hall-based estimate")

    # Q measured from enthalpy
    p.add_argument("--measured_q_col", default="", help="Direct measured cooling load column (kW).")
    p.add_argument("--supply_temp_col", default="supply_air_temperature")
    p.add_argument("--supply_rh_col", default="supply_air_rh")
    p.add_argument("--return_temp_col", default="return_air_temperature")
    p.add_argument("--return_rh_col", default="return_air_rh")
    p.add_argument("--airflow_col", default="airflow_m3_s")
    p.add_argument(
        "--airflow_unit",
        default="m3_s",
        choices=["m3_s", "kg_s", "cfm"],
        help="Airflow unit for airflow_col",
    )

    # ML
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--train_month", default="2026-03", help="Training/evaluation month in YYYY-MM")
    p.add_argument("--resample_5min", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--test_days", type=int, default=2, help="Consecutive days held out as test")
    p.add_argument("--test_start_date", default="", help="Optional test start date YYYY-MM-DD")
    p.add_argument(
        "--steady_state_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use steady-state rows only for calibration and model training",
    )
    p.add_argument(
        "--steady_state_dtin_threshold_k_per_min",
        type=float,
        default=0.02,
        help="Steady-state threshold for |dTin/dt| in K/min",
    )

    # Datacenter hall parameters for reproducible simulation setup
    p.add_argument("--room_length_m", type=float, default=30.0)
    p.add_argument("--room_width_m", type=float, default=18.0)
    p.add_argument("--room_height_m", type=float, default=4.0)
    p.add_argument("--window_count", type=int, default=4)
    p.add_argument("--window_area_each_m2", type=float, default=2.0)
    p.add_argument("--people_count", type=int, default=6)
    p.add_argument("--sensible_w_per_person", type=float, default=120.0)
    p.add_argument("--latent_w_per_person", type=float, default=60.0)
    p.add_argument("--lighting_w_per_m2", type=float, default=8.0)
    p.add_argument("--non_it_misc_kw", type=float, default=12.0)
    p.add_argument("--it_load_kw", type=float, default=220.0)
    p.add_argument(
        "--auto_calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically calibrate R_env, R_int, C_air, SHGC on training split",
    )
    p.add_argument("--calibration_iters", type=int, default=600)
    p.add_argument("--calibration_seed", type=int, default=42)
    p.add_argument("--calibration_report_csv", default="cooling_demand/cooling_calibration_report.csv")
    p.add_argument("--calibration_summary_json", default="cooling_demand/cooling_calibration_summary.json")

    p.add_argument("--interval_alpha", type=float, default=0.1, help="Prediction interval alpha (0.1 => 90%% band)")
    p.add_argument("--metric_bootstrap", type=int, default=500, help="Bootstrap rounds for metric confidence intervals")
    p.add_argument("--target_r2_min", type=float, default=0.9, help="Publication quality gate for minimum test R2")
    p.add_argument("--target_rmse_max", type=float, default=9999.0, help="Optional quality gate for max test RMSE (kW)")

    # Optional PV coupling
    p.add_argument("--pv_forecast_csv", default="", help="Optional PV forecast CSV to compute grid power")
    p.add_argument("--pv_time_col", default="date_time", help="PV timestamp column, or use date_time for date+time")
    p.add_argument("--pv_power_col", default="pred_10m", help="PV forecast power column")
    p.add_argument("--pv_power_unit", default="normalized", choices=["normalized", "kW", "W"])
    p.add_argument("--pv_rated_kw", type=float, default=14.4, help="Used when pv_power_unit=normalized")

    return p


def main() -> None:
    args = build_parser().parse_args()

    hall_spec = DatacenterHallSpec(
        room_length_m=args.room_length_m,
        room_width_m=args.room_width_m,
        room_height_m=args.room_height_m,
        window_count=args.window_count,
        window_area_each_m2=args.window_area_each_m2,
        people_count=args.people_count,
        sensible_w_per_person=args.sensible_w_per_person,
        latent_w_per_person=args.latent_w_per_person,
        lighting_w_per_m2=args.lighting_w_per_m2,
        non_it_misc_kw=args.non_it_misc_kw,
        it_load_kw=args.it_load_kw,
    )
    hall = derive_hall_parameters(hall_spec)

    window_area_m2 = args.window_area_m2 if args.window_area_m2 > 0 else hall["window_area_m2"]
    q_internal_kw = args.q_internal_kw if args.q_internal_kw > 0 else hall["q_internal_kw"]

    cfg = CoolingConfig(
        lat=args.lat,
        lon=args.lon,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        shgc=args.shgc,
        window_area_m2=window_area_m2,
        r_env_kw_per_k=args.r_env_kw_per_k,
        r_int_kw_per_k=args.r_int_kw_per_k,
        c_air_kj_per_k=args.c_air_kj_per_k,
        q_internal_kw=q_internal_kw,
        test_size=args.test_size,
    )

    df = load_dataset(Path(args.input_csv))
    df = filter_month(df, args.train_month)
    df = regularize_to_5min(df, enable=args.resample_5min)

    train_mask, test_mask, test_start, test_end = select_consecutive_test_window(
        df,
        test_days=args.test_days,
        test_start_date=args.test_start_date,
    )

    numeric_fill(
        df,
        [
            "ghi_pyr",
            "ghi",
            "dni",
            "dhi",
            "air_temperature",
            "relative_humidity",
            "wind_speed",
            args.supply_temp_col,
            args.supply_rh_col,
            args.return_temp_col,
            args.return_rh_col,
            args.airflow_col,
            args.measured_q_col,
            "indoor_temperature",
            "zone_temperature",
            "wall_temperature",
            "q_internal_kw",
            "it_load_kw",
        ],
    )

    df = compute_gpoa(df, cfg)
    df = compute_q_measured(
        df,
        supply_temp_col=args.supply_temp_col,
        supply_rh_col=args.supply_rh_col,
        return_temp_col=args.return_temp_col,
        return_rh_col=args.return_rh_col,
        airflow_col=args.airflow_col,
        airflow_unit=args.airflow_unit,
        measured_q_col=args.measured_q_col,
    )
    if args.auto_calibrate:
        cfg, calib_df, calib_summary = auto_calibrate_thermal_parameters(
            df,
            cfg,
            train_mask=train_mask,
            val_mask=test_mask,
            n_iter=args.calibration_iters,
            seed=args.calibration_seed,
        )
        calib_summary["experiment"] = {
            "train_month": args.train_month,
            "test_days": args.test_days,
            "test_start": str(test_start),
            "test_end": str(test_end),
            "steady_state_only": bool(args.steady_state_only),
            "steady_state_dtin_threshold_k_per_min": float(args.steady_state_dtin_threshold_k_per_min),
        }
        calib_summary["hall_spec"] = {
            "room_length_m": hall_spec.room_length_m,
            "room_width_m": hall_spec.room_width_m,
            "room_height_m": hall_spec.room_height_m,
            "window_count": hall_spec.window_count,
            "window_area_each_m2": hall_spec.window_area_each_m2,
            "window_area_m2_effective": window_area_m2,
            "people_count": hall_spec.people_count,
            "sensible_w_per_person": hall_spec.sensible_w_per_person,
            "latent_w_per_person": hall_spec.latent_w_per_person,
            "lighting_w_per_m2": hall_spec.lighting_w_per_m2,
            "non_it_misc_kw": hall_spec.non_it_misc_kw,
            "it_load_kw": hall_spec.it_load_kw,
            "q_internal_kw_effective": q_internal_kw,
            "floor_area_m2": hall["floor_area_m2"],
            "volume_m3": hall["volume_m3"],
        }
        calib_out = Path(args.calibration_report_csv)
        calib_out.parent.mkdir(parents=True, exist_ok=True)
        calib_df.to_csv(calib_out, index=False)

        summary_out = Path(args.calibration_summary_json)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with summary_out.open("w", encoding="utf-8") as f:
            json.dump(calib_summary, f, indent=2)

    df = compute_q_physics(df, cfg)

    if args.steady_state_only:
        threshold_k_per_s = args.steady_state_dtin_threshold_k_per_min / 60.0
        ss_mask = df["d_tin_dt_k_per_s"].abs() <= threshold_k_per_s
        train_mask = train_mask & ss_mask
        test_mask = test_mask & ss_mask
        if int(train_mask.sum()) < 100:
            raise RuntimeError("Too few steady-state training rows. Relax steady-state threshold.")

    df = add_time_features(df)

    feature_candidates = [
        "Gpoa",
        "ghi_pyr",
        "ghi",
        "dni",
        "dhi",
        "air_temperature",
        "relative_humidity",
        "wind_speed",
        "Q_phys_kW",
        "Q_solar_kW",
        "Q_env_kW",
        "Q_wall_kW",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    feature_cols = [c for c in feature_candidates if c in df.columns]

    model_df, _ = train_residual_model(
        df,
        feature_cols,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    model_df = add_prediction_intervals(model_df, alpha=args.interval_alpha)
    forecast_df = build_multi_horizon(model_df)

    forecast_df = couple_with_pv(
        forecast_df,
        pv_forecast_csv=args.pv_forecast_csv,
        pv_time_col=args.pv_time_col,
        pv_power_col=args.pv_power_col,
        pv_power_unit=args.pv_power_unit,
        pv_rated_kw=args.pv_rated_kw,
    )

    metrics = compute_metrics(
        forecast_df,
        eval_mask=forecast_df["is_test"] if "is_test" in forecast_df.columns else None,
        n_boot=args.metric_bootstrap,
        seed=args.calibration_seed,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_cols = [
        "_ts",
        "Q_measured_kW",
        "Q_phys_kW",
        "Q_hybrid_kW",
        "Q_hybrid_lower_kW",
        "Q_hybrid_upper_kW",
        "is_train",
        "is_test",
        "pred_5m",
        "pred_15m",
        "pred_30m",
        "pred_5m_lower",
        "pred_15m_lower",
        "pred_30m_lower",
        "pred_5m_upper",
        "pred_15m_upper",
        "pred_30m_upper",
        "measured_at_5m",
        "measured_at_15m",
        "measured_at_30m",
        "Gpoa",
    ]
    if "P_pv_kW" in forecast_df.columns:
        save_cols.extend(["P_pv_kW", "P_grid_kW", "P_grid_lower_kW", "P_grid_upper_kW"])

    save_cols = [c for c in save_cols if c in forecast_df.columns]
    forecast_df[save_cols].to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Datacenter hall assumptions used:")
    print(
        f"  Room: {hall_spec.room_length_m:.1f} m x {hall_spec.room_width_m:.1f} m x {hall_spec.room_height_m:.1f} m "
        f"(floor={hall['floor_area_m2']:.1f} m2, volume={hall['volume_m3']:.1f} m3)"
    )
    print(
        f"  Windows: {hall_spec.window_count} x {hall_spec.window_area_each_m2:.2f} m2 "
        f"(effective area={window_area_m2:.2f} m2)"
    )
    print(
        f"  Occupancy: {hall_spec.people_count} people, lighting={hall_spec.lighting_w_per_m2:.1f} W/m2, "
        f"IT={hall_spec.it_load_kw:.1f} kW, non-IT misc={hall_spec.non_it_misc_kw:.1f} kW"
    )
    print(
        f"Protocol: month={args.train_month}, 5-min grid={args.resample_5min}, "
        f"test window=[{test_start} to {test_end}), steady-state only={args.steady_state_only}"
    )
    print(f"Rows used: train={int(train_mask.sum())}, test={int(test_mask.sum())}")
    print(
        "Calibrated thermal parameters: "
        f"R_env={cfg.r_env_kw_per_k:.4f} K/kW, "
        f"R_int={cfg.r_int_kw_per_k:.4f} K/kW, "
        f"C_air={cfg.c_air_kj_per_k:.1f} kJ/K, "
        f"SHGC={cfg.shgc:.4f}"
    )
    print("Cooling forecast metrics (hybrid):")
    for h, m in metrics.items():
        print(
            f"  +{HORIZONS[h]:<2} min | MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
            f"R2={m['r2']:.4f} N={m['n']}"
        )
        print(
            f"            95% CI: MAE[{m['mae_ci_low']:.4f}, {m['mae_ci_high']:.4f}] "
            f"RMSE[{m['rmse_ci_low']:.4f}, {m['rmse_ci_high']:.4f}] "
            f"R2[{m['r2_ci_low']:.4f}, {m['r2_ci_high']:.4f}]"
        )

    quality_failures = []
    for h, m in metrics.items():
        if np.isfinite(m.get("r2", np.nan)) and m["r2"] < args.target_r2_min:
            quality_failures.append(
                f"+{HORIZONS[h]}min R2 {m['r2']:.4f} < target {args.target_r2_min:.4f}"
            )
        if np.isfinite(m.get("rmse", np.nan)) and m["rmse"] > args.target_rmse_max:
            quality_failures.append(
                f"+{HORIZONS[h]}min RMSE {m['rmse']:.4f} > target {args.target_rmse_max:.4f}"
            )

    if quality_failures:
        print("Quality gate: NOT MET")
        for msg in quality_failures:
            print(f"  - {msg}")
    else:
        print("Quality gate: MET")


if __name__ == "__main__":
    main()

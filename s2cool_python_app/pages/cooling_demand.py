from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.ui import info_rows, kpi_card, panel, pill, simple_table
from models.domain import SystemRecord
from services.config_service import get_system, load_model_defaults
from services.cooling_demand_service import (
    CoolingDemandError,
    DEFAULT_MODEL_SETTINGS,
    FORECAST_MODEL_NAMES,
    MODEL_SPECS,
    build_cooling_metrics,
    export_cooling_result,
    generate_cooling_forecast,
    get_cooling_source_info,
    list_cooling_sources,
    list_cooling_profiles,
    model_options,
    run_cooling_backtest,
    save_cooling_profile,
)

COOLING_HORIZONS = [5, 10, 15, 20, 30, 60, 120]
DEFAULT_COOLING_HORIZONS = [5, 15, 30]
HORIZONS = COOLING_HORIZONS
TIME_OPTIONS = [{"label": f"{h:02d}:{m:02d}", "value": f"{h:02d}:{m:02d}"} for h in range(24) for m in range(0, 60, 5)]


def _empty(message: str, height: int = 420) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(template="plotly_white", height=height, margin={"l": 55, "r": 24, "t": 44, "b": 45}, annotations=[{"text": message, "xref": "paper", "yref": "paper", "x": .5, "y": .5, "showarrow": False, "font": {"size": 14, "color": "#7b88a5"}}], xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def _styled(figure: go.Figure, title: str, height: int = 430) -> go.Figure:
    figure.update_layout(template="plotly_white", height=height, title={"text": title, "x": .02, "font": {"size": 16, "color": "#31456d"}}, margin={"l": 58, "r": 24, "t": 58, "b": 52}, legend={"orientation": "h", "y": 1.1, "x": 0, "font": {"size": 11}}, hovermode="x unified")
    figure.update_xaxes(showgrid=True, gridcolor="#edf1f8", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#edf1f8", zeroline=False)
    return figure


def _frame(result: dict[str, Any] | None, model: str | None = None) -> pd.DataFrame:
    payload = result or {}
    if payload.get("results"):
        model = model or (payload.get("models") or list(payload["results"]))[0]
        payload = payload.get("results", {}).get(model, {})
    frame = pd.DataFrame(payload.get("records") or [])
    if not frame.empty and "_ts" in frame:
        frame["_ts"] = pd.to_datetime(frame["_ts"], errors="coerce")
    return frame


def _forecast_figure(result: dict[str, Any] | None) -> go.Figure:
    payloads = (result or {}).get("results") or {((result or {}).get("model") or "xgboost"): result or {}}
    frames = {name: _frame(payload, name) if payload is result else _frame(payload) for name, payload in payloads.items()}
    frames = {name: frame for name, frame in frames.items() if not frame.empty}
    if not frames:
        return _empty("Run a backtest or generate the latest cooling forecast.")
    figure = go.Figure()
    colors = ["#6080d1", "#ff8b1f", "#9b51e0", "#20a39e", "#c44536", "#52718b"]
    for model_index, (model_name, frame) in enumerate(frames.items()):
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        if model_index == 0 and "Q_measured_kW" in frame:
            figure.add_trace(go.Scatter(x=frame["_ts"], y=frame["Q_measured_kW"], name="Measured", mode="lines", line={"color": "#31456d", "width": 2.2}))
        horizons = (result or {}).get("horizons") or HORIZONS
        for index, horizon in enumerate(horizons):
            column = f"pred_{horizon}m"
            if column in frame:
                target_time = frame["_ts"] + pd.to_timedelta(horizon, unit="m")
                figure.add_trace(go.Scatter(x=target_time, y=frame[column], name=f"{label} +{horizon} min", mode="lines", line={"color": colors[model_index % len(colors)], "dash": ["solid", "dash", "dot", "dashdot"][index % 4], "width": 1.5}))
        if model_index == 0 and horizons:
            interval_horizon = horizons[0]
            target_time = frame["_ts"] + pd.to_timedelta(interval_horizon, unit="m")
            lower_column = f"pred_{interval_horizon}m_lower"
            upper_column = f"pred_{interval_horizon}m_upper"
            if lower_column in frame and upper_column in frame:
                figure.add_trace(go.Scatter(x=target_time, y=frame[lower_column], mode="lines", line={"width": 0}, showlegend=False))
                figure.add_trace(go.Scatter(x=target_time, y=frame[upper_column], name="90% interval", mode="lines", fill="tonexty", fillcolor="rgba(96,128,209,.14)", line={"width": 0}))
    figure.update_yaxes(title="Cooling demand (kW)")
    return _styled(figure, "Cooling demand forecast")


def _components_figure(result: dict[str, Any] | None) -> go.Figure:
    frame = _frame(result)
    if frame.empty:
        return _empty("Component breakdown will appear after a successful run.")
    figure = go.Figure()
    for column, name, color in [("Q_solar_kW", "Solar", "#f2c94c"), ("Q_env_kW", "Envelope", "#56ccf2"), ("Q_wall_kW", "Wall", "#bb6bd9"), ("Q_dynamic_kW", "Dynamic", "#f2994a"), ("Q_phys_kW", "Physics total", "#20b26b"), ("Q_hybrid_kW", "Hybrid", "#6080d1")]:
        if column in frame:
            figure.add_trace(go.Scatter(x=frame["_ts"], y=frame[column], name=name, mode="lines", line={"color": color, "width": 1.7}))
    figure.update_yaxes(title="Cooling load (kW)")
    return _styled(figure, "Physics component breakdown")


def _residual_figure(result: dict[str, Any] | None) -> go.Figure:
    payloads = (result or {}).get("results") or {((result or {}).get("model") or "xgboost"): result or {}}
    figure = go.Figure()
    has_trace = False
    for model_name, payload in payloads.items():
        frame = _frame(payload)
        if frame.empty:
            continue
        model_label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        for horizon in (result or {}).get("horizons") or HORIZONS:
            actual, pred = f"measured_at_{horizon}m", f"pred_{horizon}m"
            if actual in frame and pred in frame:
                figure.add_trace(go.Scatter(x=frame["_ts"], y=frame[pred] - frame[actual], name=f"{model_label} +{horizon} min", mode="lines"))
                has_trace = True
    if not has_trace:
        return _empty("Residuals will appear after a backtest.", 390)
    figure.add_hline(y=0, line_color="#31456d")
    figure.update_yaxes(title="Prediction error (kW)")
    return _styled(figure, "Residual time series")


def _scatter_figure(result: dict[str, Any] | None) -> go.Figure:
    frame = _frame(result)
    if frame.empty:
        return _empty("Predicted-versus-measured diagnostics require a backtest.", 390)
    figure, values = go.Figure(), []
    for horizon in result.get("horizons") or HORIZONS:
        actual, pred = f"measured_at_{horizon}m", f"pred_{horizon}m"
        if actual not in frame or pred not in frame:
            continue
        valid = frame[[actual, pred]].dropna()
        if valid.empty:
            continue
        values.extend(valid[actual].tolist() + valid[pred].tolist())
        metric = next((row for row in build_cooling_metrics(result) if row.get("horizon_minutes") == horizon), {})
        r2 = metric.get("r2")
        label = f"+{horizon} min | R² {r2:.3f} | N={len(valid)}" if isinstance(r2, (int, float)) else f"+{horizon} min | N={len(valid)}"
        figure.add_trace(go.Scatter(x=valid[actual], y=valid[pred], name=label, mode="markers", marker={"size": 5, "opacity": .58}))
    if not values:
        return _empty("No valid measured/predicted pairs are available.", 390)
    low, high = min(values), max(values)
    figure.add_trace(go.Scatter(x=[low, high], y=[low, high], name="Ideal 1:1", mode="lines", line={"color": "#9aa8c5", "dash": "dash"}))
    figure.update_xaxes(title="Measured cooling load (kW)", range=[low, high])
    figure.update_yaxes(title="Predicted cooling load (kW)", range=[low, high], scaleanchor="x", scaleratio=1)
    return _styled(figure, "Predicted versus measured cooling demand")


def _metric_figure(result: dict[str, Any] | None, key: str, title: str, label: str) -> go.Figure:
    rows = [row for row in build_cooling_metrics(result or {}) if row.get(key) is not None]
    if not rows:
        return _empty(f"{label} diagnostics will appear after a backtest.", 360)
    figure = go.Figure()
    model_names = list(dict.fromkeys(row.get("model", (result or {}).get("model", "xgboost")) for row in rows))
    colors = ["#6080d1", "#ff8b1f", "#9b51e0", "#20a39e", "#c44536", "#52718b"]
    for index, model_name in enumerate(model_names):
        model_rows = [row for row in rows if row.get("model", model_name) == model_name]
        figure.add_trace(go.Bar(x=[f"+{row['horizon_minutes']} min" for row in model_rows], y=[row[key] for row in model_rows], name=MODEL_SPECS.get(model_name, {}).get("label", model_name), marker_color=colors[index % len(colors)], text=[f"{row[key]:.3f}" for row in model_rows], textposition="outside"))
    figure.update_layout(barmode="group")
    figure.update_yaxes(title=label)
    return _styled(figure, title, 360)


def _field(label: str, component_id: str, value: Any, step: str = "any") -> html.Div:
    visible_label = "Training rows (0 = all)" if component_id == "cooling-max-training-rows" else label
    return html.Div([html.Label(visible_label, className="muted"), dcc.Input(id=component_id, type="number", value=value, step=step, className="cooling-input")], className="cooling-field")


def _cooling_tuning_input(control_id: str, value: Any, step: float = 1, minimum: float | None = None) -> dcc.Input:
    props = {"id": control_id, "type": "number", "value": value, "step": step, "className": "cooling-tuning-input"}
    if minimum is not None:
        props["min"] = minimum
    return dcc.Input(**props)


def _cooling_tuning_card(model_name: str, title: str, fields: list[tuple[str, str, Any, float, float | None]], visible: bool = False) -> html.Div:
    return html.Div(
        id=f"cooling-settings-{model_name}",
        className="cooling-model-settings-card",
        style={"display": "block" if visible else "none"},
        children=[
            html.H4(title, className="cooling-model-settings-title"),
            html.Div(className="cooling-tuning-grid", children=[
                html.Div([html.Label(label, className="cooling-tuning-label"), _cooling_tuning_input(control_id, value, step, minimum)], className="cooling-tuning-field")
                for label, control_id, value, step, minimum in fields
            ]),
        ],
    )


def _cooling_model_tuning_layout() -> html.Div:
    defaults = DEFAULT_MODEL_SETTINGS
    return html.Div(className="cooling-model-settings-grid", children=[
        _cooling_tuning_card("xgboost", "XGBoost settings", [
            ("Estimators", "cooling-xgb-n-estimators", defaults["xgboost"]["n_estimators"], 10, 10),
            ("Max depth", "cooling-xgb-max-depth", defaults["xgboost"]["max_depth"], 1, 1),
            ("Learning rate", "cooling-xgb-learning-rate", defaults["xgboost"]["learning_rate"], 0.01, 0.001),
            ("Subsample", "cooling-xgb-subsample", defaults["xgboost"]["subsample"], 0.05, 0.1),
            ("Column sample", "cooling-xgb-colsample", defaults["xgboost"]["colsample_bytree"], 0.05, 0.1),
        ], visible=True),
        _cooling_tuning_card("extra_trees", "Extra Trees settings", [
            ("Estimators", "cooling-et-n-estimators", defaults["extra_trees"]["n_estimators"], 10, 10),
            ("Max depth (0 = auto)", "cooling-et-max-depth", defaults["extra_trees"]["max_depth"], 1, 0),
            ("Min samples leaf", "cooling-et-min-leaf", defaults["extra_trees"]["min_samples_leaf"], 1, 1),
            ("Max features", "cooling-et-max-features", defaults["extra_trees"]["max_features"], 0.05, 0.1),
        ]),
        _cooling_tuning_card("random_forest", "Random Forest settings", [
            ("Estimators", "cooling-rf-n-estimators", defaults["random_forest"]["n_estimators"], 10, 10),
            ("Max depth (0 = auto)", "cooling-rf-max-depth", defaults["random_forest"]["max_depth"], 1, 0),
            ("Min samples leaf", "cooling-rf-min-leaf", defaults["random_forest"]["min_samples_leaf"], 1, 1),
            ("Max features", "cooling-rf-max-features", defaults["random_forest"]["max_features"], 0.05, 0.1),
        ]),
        _cooling_tuning_card("lightgbm", "LightGBM settings", [
            ("Estimators", "cooling-lgb-n-estimators", defaults["lightgbm"]["n_estimators"], 10, 10),
            ("Learning rate", "cooling-lgb-learning-rate", defaults["lightgbm"]["learning_rate"], 0.01, 0.001),
            ("Number of leaves", "cooling-lgb-num-leaves", defaults["lightgbm"]["num_leaves"], 1, 2),
            ("Max depth (-1 = auto)", "cooling-lgb-max-depth", defaults["lightgbm"]["max_depth"], 1, -1),
        ]),
        _cooling_tuning_card("catboost", "CatBoost settings", [
            ("Iterations", "cooling-cat-iterations", defaults["catboost"]["iterations"], 10, 10),
            ("Depth", "cooling-cat-depth", defaults["catboost"]["depth"], 1, 1),
            ("Learning rate", "cooling-cat-learning-rate", defaults["catboost"]["learning_rate"], 0.01, 0.001),
            ("L2 regularization", "cooling-cat-l2", defaults["catboost"]["l2_leaf_reg"], 0.5, 0),
        ]),
        _cooling_tuning_card("lstm", "LSTM settings", [
            ("Sequence length", "cooling-lstm-sequence-length", defaults["lstm"]["sequence_length"], 1, 2),
            ("Epochs", "cooling-lstm-epochs", defaults["lstm"]["epochs"], 1, 1),
            ("Batch size", "cooling-lstm-batch-size", defaults["lstm"]["batch_size"], 1, 1),
        ]),
    ])


def _profile_values(profile: dict[str, Any] | None) -> list[Any]:
    profile = profile or {}
    hall, schedule = profile.get("hall", {}), profile.get("internal_load_schedule", {})
    physics, synthetic = profile.get("physics_initial", {}), profile.get("synthetic_measurement", {})
    return [hall.get("room_length_m"), hall.get("room_width_m"), hall.get("room_height_m"), hall.get("window_count"), hall.get("window_area_each_m2"), hall.get("people_count"), hall.get("sensible_w_per_person"), hall.get("latent_w_per_person"), hall.get("lighting_w_per_m2"), hall.get("non_it_misc_kw"), hall.get("it_load_kw"), schedule.get("day_start_hour"), schedule.get("day_end_hour"), schedule.get("it_day_multiplier"), schedule.get("it_night_multiplier"), schedule.get("people_day_multiplier"), schedule.get("people_night_multiplier"), schedule.get("lighting_day_multiplier"), schedule.get("lighting_night_multiplier"), schedule.get("misc_day_multiplier"), schedule.get("misc_night_multiplier"), schedule.get("weekend_multiplier"), physics.get("shgc"), physics.get("r_env_kw_per_k"), physics.get("r_int_kw_per_k"), physics.get("c_air_kj_per_k"), synthetic.get("indoor_temp_setpoint_c", 24.0), synthetic.get("supply_temp_c", 17.5), synthetic.get("supply_rh_pct", 90.0), synthetic.get("return_rise_c", synthetic.get("return_temp_rise_c", 6.5)), synthetic.get("airflow_m3_s", 0.7), synthetic.get("measurement_noise_std_kw", 0.05)]


def _available_system_options(system: SystemRecord | None) -> tuple[list[dict[str, Any]], int | None]:
    options = []
    seen: set[int] = set()
    for source in list_cooling_sources():
        if source.source_type != "system" or source.system_id is None or source.system_id in seen:
            continue
        seen.add(int(source.system_id))
        options.append({"label": source.label, "value": int(source.system_id)})
    options.sort(key=lambda item: item["value"])
    preferred = int(system.system_number) if system and int(system.system_number) in seen else None
    return options, preferred if preferred is not None else (options[0]["value"] if options else None)


def _system_blocks(system_id: int | None) -> tuple[Any, Any, Any]:
    if not system_id:
        return pill("No system selected", "orange"), html.P("Choose an available system to continue.", className="section-subtitle"), html.P("System weather and solar data are unavailable.", className="section-subtitle")
    system = get_system(int(system_id))
    artifact_ready = False
    try:
        metadata = get_cooling_source_info(f"system-{int(system_id):02d}", "system", int(system_id))
        artifact_ready = bool(metadata.get("data_analysis_artifact"))
        status = pill("System data ready" if artifact_ready else "Cooling data ready", "green")
    except Exception as exc:
        metadata, status, source = {}, pill("Cooling dataset unavailable", "red"), html.P(str(exc), className="preprocessing-export-error")
    name = system.name if system else f"System {int(system_id)}"
    location = system.location if system else (metadata.get("data_analysis_city") or metadata.get("city") or "-")
    lat = system.lat if system else (metadata.get("data_analysis_lat") or "-")
    lon = system.lon if system else (metadata.get("data_analysis_lon") or "-")
    capacity = system.capacity_kw if system else (metadata.get("data_analysis_capacity_kw") or metadata.get("capacity_kw"))
    capacity_text = f"{float(capacity):.3g} kW" if capacity is not None else "-"
    start = str(metadata.get("data_analysis_start") or "-")[:16]
    end = str(metadata.get("data_analysis_end") or "-")[:16]
    interval = metadata.get("data_analysis_interval_minutes")
    interval_text = f"{float(interval):.0f} minutes" if interval is not None else "-"
    rows = metadata.get("data_analysis_rows")
    rows_text = f"{int(rows):,}" if rows is not None else "-"
    config_id = str(metadata.get("data_analysis_artifact_id") or "")
    config_text = f"Ready · {config_id[-8:]}" if config_id else ("Not exported" if not artifact_ready else "Ready")
    source = html.Div(className="cooling-source-summary", children=[
        info_rows([
            ("Weather & solar", "Data Analysis artifact" if artifact_ready else "Not exported"),
            ("Cooling load", "M3 measured dataset"),
            ("Preprocessing", config_text),
            ("Run behavior", "Loaded when backtest or forecast starts"),
        ])
    ])
    parameters = html.Div(className="cooling-system-parameters-table", children=[simple_table(["Parameter", "Value"], [
        ["System ID", name],
        ["City", location],
        ["Capacity", capacity_text],
        ["Coordinates", f"Lat {float(lat):.6f}, Lon {float(lon):.6f}" if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) else f"{lat}, {lon}"],
        ["Dataset period", f"{start} to {end}" if start != "-" and end != "-" else "Available on run"],
        ["Sampling interval", interval_text],
        ["Processed rows", rows_text],
        ["Features", str(metadata.get("data_analysis_feature_count") or "Available on run")],
    ])])
    return status, source, parameters


def _summary_cards(result: dict[str, Any] | None) -> list[html.Div]:
    if not result:
        return [kpi_card("Latest cooling demand", "Not run", "Run a backtest or latest forecast"), kpi_card("Forecast status", "Ready", "Complete the four setup steps"), kpi_card("Data coverage", "—", "No result loaded"), kpi_card("Model", "—", "No model executed")]
    latest = result.get("latest_record") or {}
    fmt = lambda value: f"{float(value):.2f} kW" if isinstance(value, (int, float)) else "—"
    return [kpi_card("Latest cooling demand", fmt(latest.get("Q_measured_kW")), str(latest.get("_ts", "—"))[:19]), kpi_card("Hybrid forecast", fmt(latest.get("Q_hybrid_kW")), "Latest valid record"), kpi_card("Data coverage", f"{result.get('source_rows', 0):,} rows", f"{str(result.get('source_start', '—'))[:16]} to {str(result.get('source_end', '—'))[:16]}"), kpi_card("Model status", "Completed", str(result.get("model", "—")))]


SCRIPT_DEFAULT_COMPONENT_VALUES = {
    "cooling-start-date": {"date": "2026-05-01"},
    "cooling-end-date": {"date": "2026-05-02"},
    "cooling-evaluation-options": {"value": ["steady"]},
    "cooling-max-training-rows": {"value": 0},
}


def _apply_script_defaults(component: Any) -> None:
    """Apply the M3 March-run defaults to controls created in the layout."""
    if isinstance(component, (list, tuple)):
        for child in component:
            _apply_script_defaults(child)
        return
    component_id = getattr(component, "id", None)
    for property_name, value in SCRIPT_DEFAULT_COMPONENT_VALUES.get(component_id, {}).items():
        setattr(component, property_name, value)
    children = getattr(component, "children", None)
    if children is not None:
        _apply_script_defaults(children)


def _augment_forecast_controls(layout: html.Div) -> None:
    """Replace the legacy single-model controls with the PV-style controls."""
    model_options_ui = model_options(include_baselines=False)
    inserted_tuning = False

    def visit(component: Any) -> None:
        nonlocal inserted_tuning
        if isinstance(component, (list, tuple)):
            for child in component:
                visit(child)
            return
        children = getattr(component, "children", None)
        if not isinstance(children, list):
            if children is not None:
                visit(children)
            return
        for index, child in list(enumerate(children)):
            child_id = getattr(child, "id", None)
            child_class = getattr(child, "className", "") or ""
            if child_id == "cooling-horizon-select":
                child.options = [{"label": f" {value} min" if value < 60 else f" {value // 60} hr", "value": value} for value in COOLING_HORIZONS]
                child.value = DEFAULT_COOLING_HORIZONS
            if child_id == "cooling-model-select":
                child.style = {"display": "none"}
                if index > 0 and getattr(children[index - 1], "children", None) == "Forecast model":
                    children[index - 1].style = {"display": "none"}
                children[index:index + 1] = [
                    html.Div(className="cooling-model-selector", style={"gridColumn": "1 / -1"}, children=[
                        child,
                        html.Label("Forecast models", className="muted"),
                        dcc.Checklist(id="cooling-model-select-ui", options=model_options_ui, value=["xgboost"], className="cooling-model-checklist"),
                        html.Div("Select one or more models. Each selected model is trained and plotted separately.", className="cooling-model-help"),
                    ]),
                ]
            if child_class == "cooling-model-settings":
                child.style = {"display": "none"}
            if child_class == "cooling-forecast-grid" and not inserted_tuning:
                grid_children = getattr(child, "children", None)
                if isinstance(grid_children, list) and len(grid_children) > 1:
                    grid_children[1].style = {"gridColumn": "1 / -1"}
                children.insert(index + 1, html.Div(className="cooling-model-tuning-section", children=[
                    html.H3("Model tuning", className="cooling-subheading"),
                    html.P("Adjust the parameters for each selected model before running the forecast.", className="cooling-model-help"),
                    _cooling_model_tuning_layout(),
                ]))
                inserted_tuning = True
            visit(child)

    visit(layout)
    layout.children.insert(0, dcc.Store(id="cooling-model-settings", data=DEFAULT_MODEL_SETTINGS))


def build_layout(system: SystemRecord | None) -> html.Div:
    defaults = load_model_defaults().get("cooling_demand", {})
    profiles = list_cooling_profiles()
    profile = profiles[0].get("profile", {}) if profiles else {}
    values = _profile_values(profile)
    system_options, default_system_id = _available_system_options(system)
    status, source_info, parameters = _system_blocks(default_system_id)
    models = model_options()
    default_model = "hybrid_xgboost" if any(item["value"] == "hybrid_xgboost" and not item.get("disabled") for item in models) else "hybrid_gradient_boosting"
    layout = html.Div(className="cooling-demand-page", children=[
        dcc.Store(id="cooling-result", data=None),
        dcc.Download(id="cooling-download-forecast"), dcc.Download(id="cooling-download-metrics"), dcc.Download(id="cooling-download-summary"), dcc.Download(id="cooling-download-calibration"), dcc.Download(id="cooling-download-profile"),
        panel("Cooling Demand Workbench", "Sequential M3 workflow: system data → cooling site profile → demand model → forecast model.", [html.Div(className="cooling-step-heading", children=[html.Span("STEP 1", className="cooling-step-number"), html.H3("Specify system", className="section-title"), html.P("Select the available system whose Data Analysis weather and solar artifact will be used for this run.", className="section-subtitle")]), html.Div(className="two-col-grid cooling-setup-grid", children=[html.Div(className="panel cooling-inner-panel cooling-system-card", children=[html.H3("System data source", className="section-title"), dcc.Dropdown(id="cooling-system-select", options=system_options, value=default_system_id, clearable=False, className="cooling-select"), html.Div(id="cooling-system-status", children=status), html.Div(id="cooling-system-source-info", children=source_info)]), html.Div(className="panel cooling-inner-panel cooling-system-card", children=[html.H3("System Parameters", className="section-title"), html.Div(id="cooling-system-parameters", children=parameters)])])]),
        panel("STEP 2 · Cooling Site Profile settings", "Define the physical hall and internal load schedule used by the cooling model.", [html.Div(className="cooling-profile-toolbar", children=[dcc.Dropdown(id="cooling-profile-select", options=[{"label": item["label"], "value": item["site_id"]} for item in profiles], value=profiles[0]["site_id"] if profiles else None, clearable=False, className="cooling-select"), html.Button("Save Profile", id="cooling-save-profile", className="action-btn"), html.Div(id="cooling-profile-status")]), html.Div(className="two-col-grid cooling-profile-grid", children=[html.Div(className="cooling-subpanel", children=[html.H3("Hall settings", className="cooling-subheading"), html.Div(className="cooling-fields-grid", children=[_field("Room length (m)", "cooling-room-length", values[0]), _field("Room width (m)", "cooling-room-width", values[1]), _field("Room height (m)", "cooling-room-height", values[2]), _field("Window count", "cooling-window-count", values[3], "1"), _field("Window area each (m²)", "cooling-window-area", values[4]), _field("People count", "cooling-people-count", values[5], "1"), _field("Sensible W/person", "cooling-sensible", values[6]), _field("Latent W/person", "cooling-latent", values[7]), _field("Lighting W/m²", "cooling-lighting", values[8]), _field("Misc load (kW)", "cooling-misc", values[9]), _field("IT load (kW)", "cooling-it-load", values[10])])]), html.Div(className="cooling-subpanel", children=[html.H3("Load schedule", className="cooling-subheading"), html.Div(className="cooling-fields-grid", children=[_field("Day start hour", "cooling-day-start", values[11], "1"), _field("Day end hour", "cooling-day-end", values[12], "1"), _field("IT day multiplier", "cooling-it-day", values[13], "0.01"), _field("IT night multiplier", "cooling-it-night", values[14], "0.01"), _field("People day multiplier", "cooling-people-day", values[15], "0.01"), _field("People night multiplier", "cooling-people-night", values[16], "0.01"), _field("Lighting day multiplier", "cooling-lighting-day", values[17], "0.01"), _field("Lighting night multiplier", "cooling-lighting-night", values[18], "0.01"), _field("Misc day multiplier", "cooling-misc-day", values[19], "0.01"), _field("Misc night multiplier", "cooling-misc-night", values[20], "0.01"), _field("Weekend multiplier", "cooling-weekend", values[21], "0.01")])])])]),
        panel("STEP 3 · Cooling demand model settings", "Configure Q physics and choose how Q measured is supplied to the model.", [html.Div(className="two-col-grid cooling-model-grid", children=[html.Div(className="cooling-subpanel", children=[html.H3("Q physics parameters", className="cooling-subheading"), html.Div(className="cooling-fields-grid", children=[_field("SHGC", "cooling-shgc", values[22], "0.01"), _field("R envelope (K/kW)", "cooling-r-env", values[23]), _field("R internal (K/kW)", "cooling-r-int", values[24]), _field("Air capacitance (kJ/K)", "cooling-c-air", values[25])]), html.Div(className="cooling-physics-explanation", children=[html.H4("Physics model equation", className="cooling-explanation-heading"), html.Div("Q_phys = Q_env + Q_wall + Q_internal + Q_solar - Q_dynamic", className="cooling-equation"), html.P("Q_solar = SHGC × window area × Gpoa / 1000. SHGC is the fraction of incident solar radiation entering through the windows; Gpoa is plane-of-array irradiance in W/m².", className="cooling-explanation-text"), html.P("Q_env = (T_out - T_in) / R_env. R envelope is the effective envelope resistance in K/kW; a larger value means less heat transfer through the building envelope.", className="cooling-explanation-text"), html.P("Q_wall = (T_wall - T_in) / R_int. R internal is the effective internal thermal-mass resistance in K/kW; it controls heat transfer between the wall mass and indoor air.", className="cooling-explanation-text"), html.P("Q_dynamic = C_air × dT_in/dt. Air capacitance is the effective hall thermal capacitance in kJ/K; it represents stored indoor thermal energy and is subtracted when indoor temperature is rising.", className="cooling-explanation-text"), html.P("Q_internal is the scheduled IT, people, lighting, and miscellaneous internal heat gain in kW. All Q terms are in kW, temperatures are in °C, and the final physics load is clipped at zero.", className="cooling-explanation-text")])]), html.Div(className="cooling-subpanel", children=[html.H3("Q measured source", className="cooling-subheading"), dcc.RadioItems(id="cooling-measurement-mode", options=[{"label": "Synthetic Q measured", "value": "synthetic"}, {"label": "Experimental CSV", "value": "experimental"}], value="synthetic", className="cooling-radio"), html.Div(id="cooling-synthetic-panel", children=[html.Div(className="cooling-fields-grid", children=[_field("Indoor setpoint (°C)", "cooling-synthetic-setpoint", values[26]), _field("Supply temperature (°C)", "cooling-synthetic-supply-temp", values[27]), _field("Supply RH (%)", "cooling-synthetic-supply-rh", values[28]), _field("Return temperature rise (°C)", "cooling-synthetic-return-rise", values[29]), _field("Airflow (m³/s)", "cooling-synthetic-airflow", values[30]), _field("Noise σ (kW)", "cooling-synthetic-noise", values[31])])]), html.Div(id="cooling-upload-panel", style={"display": "none"}, children=[dcc.Upload(id="cooling-measured-upload", children=html.Div(["Drop experimental Q measured CSV here or ", html.A("browse")]), className="cooling-upload"), html.Div(id="cooling-upload-status", className="cooling-upload-status")]), html.P("Synthetic mode uses the selected psychrometric parameters. Experimental mode requires timestamp plus Q_measured_kW, Q_measured_W, or cooling_kw.", id="cooling-measurement-help", className="cooling-help")])])]),
        panel("STEP 4 · Forecast model settings", "Select horizons, machine-learning model, time window, evaluation, and calibration settings.", [html.Div(className="cooling-forecast-grid", children=[html.Div([html.Label("Forecast horizons", className="muted"), dcc.Checklist(id="cooling-horizon-select", options=[{"label": f"{value} min", "value": value} for value in HORIZONS], value=HORIZONS, className="cooling-checklist")]), html.Div([html.Label("Forecast model", className="muted"), dcc.Dropdown(id="cooling-model-select", options=models, value=default_model, clearable=False, className="cooling-select")])]), html.Div(className="cooling-model-settings", children=[html.H3("Model tuning", className="cooling-subheading"), html.Div(className="cooling-fields-grid", children=[_field("Estimators", "cooling-estimators", 500, "1"), _field("Max depth", "cooling-max-depth", 6, "1"), _field("Learning rate", "cooling-learning-rate", 0.03, "0.001"), _field("Subsample", "cooling-subsample", 0.9, "0.05"), _field("Column sample", "cooling-colsample", 0.9, "0.05"), _field("Training rows", "cooling-max-training-rows", 12000, "100")])]), html.Div(className="cooling-date-grid", children=[html.Div([html.Label("Backtest start", className="muted"), dcc.DatePickerSingle(id="cooling-start-date", display_format="YYYY-MM-DD", className="cooling-date-picker"), dcc.Dropdown(id="cooling-start-time", options=TIME_OPTIONS, value="00:00", clearable=False, className="cooling-time-select")]), html.Div([html.Label("Backtest end", className="muted"), dcc.DatePickerSingle(id="cooling-end-date", display_format="YYYY-MM-DD", className="cooling-date-picker"), dcc.Dropdown(id="cooling-end-time", options=TIME_OPTIONS, value="23:55", clearable=False, className="cooling-time-select")])]), html.Div(className="cooling-forecast-options", children=[dcc.Checklist(id="cooling-evaluation-options", options=[{"label": "Daytime / steady-state rows only", "value": "steady"}, {"label": "Run thermal calibration", "value": "calibrate"}], value=[], className="cooling-checklist"), _field("Calibration iterations", "cooling-calibration-iterations", defaults.get("calibration_iterations", 1000), "100"), _field("Interval alpha", "cooling-interval-alpha", defaults.get("interval_alpha", 0.1), "0.01")]), html.Div(className="action-button-row", children=[html.Button("Run Backtest", id="cooling-run-backtest", className="action-btn action-btn-primary"), html.Button("Generate Latest Forecast", id="cooling-run-forecast", className="action-btn")]), dcc.Loading(id="cooling-loading", type="circle", children=html.Div(id="cooling-run-status", className="cooling-run-status"))]),
        html.Div(className="kpi-grid cooling-kpi-grid", id="cooling-kpi-grid", children=_summary_cards(None)),
        panel("Forecast curve", "Measured, physics-only, hybrid, and persistence-style horizon forecasts.", [dcc.Graph(id="cooling-forecast-graph", figure=_empty("Run a forecast to view cooling demand."), className="trend-graph")]),
        panel("Physics components", "Inspect the physical terms contributing to total cooling demand.", [dcc.Graph(id="cooling-components-graph", figure=_empty("Run a forecast to view physics components."), className="trend-graph")]),
        panel("Diagnostics", "Publication-grade diagnostics for residuals, predicted-versus-measured behavior, and error metrics.", [dcc.Tabs(id="cooling-diagnostic-tabs", value="residuals", className="cooling-diagnostic-tabs", children=[dcc.Tab(label="Residuals", value="residuals", children=[dcc.Graph(id="cooling-residual-graph", figure=_empty("Backtest diagnostics will appear here.", 390))]), dcc.Tab(label="Predicted vs measured", value="scatter", children=[dcc.Graph(id="cooling-scatter-graph", figure=_empty("Backtest diagnostics will appear here.", 390))]), dcc.Tab(label="MAE", value="mae", children=[dcc.Graph(id="cooling-mae-graph", figure=_empty("Backtest diagnostics will appear here.", 360))]), dcc.Tab(label="RMSE", value="rmse", children=[dcc.Graph(id="cooling-rmse-graph", figure=_empty("Backtest diagnostics will appear here.", 360))]), dcc.Tab(label="R²", value="r2", children=[dcc.Graph(id="cooling-r2-graph", figure=_empty("Backtest diagnostics will appear here.", 360))])])]),
        html.Div(className="two-col-grid cooling-results-grid", children=[panel("Metrics table", "Metrics are calculated on selected evaluation rows for each horizon.", [html.Div(id="cooling-metrics-table", children=simple_table(["Horizon", "MAE (kW)", "RMSE (kW)", "R²", "Bias (kW)", "Samples"], []))]), panel("Calibration and data quality", "Review calibration state, data coverage, and warnings.", [html.Div(id="cooling-calibration-summary", children=html.P("No calibration has been run.", className="section-subtitle")), dcc.Graph(id="cooling-calibration-graph", figure=_empty("Calibration history will appear when calibration is enabled.", 300)), html.Div(id="cooling-data-quality", children=html.P("No quality report available.", className="section-subtitle"))])]),
        panel("Run artifacts", "Download reproducible cooling forecasts, metrics, calibration history, profile metadata, and the run summary.", [html.Div(className="action-button-row", children=[html.Button("Download Forecast CSV", id="cooling-download-forecast-btn", className="action-btn"), html.Button("Download Metrics CSV", id="cooling-download-metrics-btn", className="action-btn"), html.Button("Download Run Summary JSON", id="cooling-download-summary-btn", className="action-btn"), html.Button("Download Calibration CSV", id="cooling-download-calibration-btn", className="action-btn"), html.Button("Download Profile JSON", id="cooling-download-profile-btn", className="action-btn")]), html.Div(id="cooling-export-status", className="preprocessing-export-status")]),
    ])
    _apply_script_defaults(layout)
    _augment_forecast_controls(layout)
    return layout


@callback(Output("cooling-synthetic-panel", "style"), Output("cooling-upload-panel", "style"), Input("cooling-measurement-mode", "value"))
def toggle_measurement_source(mode: str):
    return ({"display": "none"}, {"display": "block"}) if mode == "experimental" else ({"display": "block"}, {"display": "none"})


@callback(
    Output("cooling-system-status", "children"),
    Output("cooling-system-source-info", "children"),
    Output("cooling-system-parameters", "children"),
    Input("cooling-system-select", "value"),
)
def render_cooling_system(system_id: int | None):
    return _system_blocks(int(system_id) if system_id else None)


@callback(
    Output("cooling-settings-xgboost", "style"),
    Output("cooling-settings-extra_trees", "style"),
    Output("cooling-settings-random_forest", "style"),
    Output("cooling-settings-lightgbm", "style"),
    Output("cooling-settings-catboost", "style"),
    Output("cooling-settings-lstm", "style"),
    Input("cooling-model-select-ui", "value"),
)
def toggle_cooling_model_settings(models: list[str] | None):
    selected = set(models or [])
    return tuple({"display": "block" if name in selected else "none"} for name in FORECAST_MODEL_NAMES)


@callback(
    Output("cooling-model-settings", "data"),
    Input("cooling-xgb-n-estimators", "value"), Input("cooling-xgb-max-depth", "value"), Input("cooling-xgb-learning-rate", "value"), Input("cooling-xgb-subsample", "value"), Input("cooling-xgb-colsample", "value"),
    Input("cooling-et-n-estimators", "value"), Input("cooling-et-max-depth", "value"), Input("cooling-et-min-leaf", "value"), Input("cooling-et-max-features", "value"),
    Input("cooling-rf-n-estimators", "value"), Input("cooling-rf-max-depth", "value"), Input("cooling-rf-min-leaf", "value"), Input("cooling-rf-max-features", "value"),
    Input("cooling-lgb-n-estimators", "value"), Input("cooling-lgb-learning-rate", "value"), Input("cooling-lgb-num-leaves", "value"), Input("cooling-lgb-max-depth", "value"),
    Input("cooling-cat-iterations", "value"), Input("cooling-cat-depth", "value"), Input("cooling-cat-learning-rate", "value"), Input("cooling-cat-l2", "value"),
    Input("cooling-lstm-sequence-length", "value"), Input("cooling-lstm-epochs", "value"), Input("cooling-lstm-batch-size", "value"),
)
def collect_cooling_model_settings(*values):
    names = [
        "xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate", "xgb_subsample", "xgb_colsample",
        "et_n_estimators", "et_max_depth", "et_min_leaf", "et_max_features",
        "rf_n_estimators", "rf_max_depth", "rf_min_leaf", "rf_max_features",
        "lgb_n_estimators", "lgb_learning_rate", "lgb_num_leaves", "lgb_max_depth",
        "cat_iterations", "cat_depth", "cat_learning_rate", "cat_l2",
        "lstm_sequence_length", "lstm_epochs", "lstm_batch_size",
    ]
    values = dict(zip(names, values))
    return {
        "xgboost": {"n_estimators": values["xgb_n_estimators"], "max_depth": values["xgb_max_depth"], "learning_rate": values["xgb_learning_rate"], "subsample": values["xgb_subsample"], "colsample_bytree": values["xgb_colsample"], "max_training_rows": 0},
        "extra_trees": {"n_estimators": values["et_n_estimators"], "max_depth": values["et_max_depth"], "min_samples_leaf": values["et_min_leaf"], "max_features": values["et_max_features"], "max_training_rows": 0},
        "random_forest": {"n_estimators": values["rf_n_estimators"], "max_depth": values["rf_max_depth"], "min_samples_leaf": values["rf_min_leaf"], "max_features": values["rf_max_features"], "max_training_rows": 0},
        "lightgbm": {"n_estimators": values["lgb_n_estimators"], "learning_rate": values["lgb_learning_rate"], "num_leaves": values["lgb_num_leaves"], "max_depth": values["lgb_max_depth"], "max_training_rows": 0},
        "catboost": {"iterations": values["cat_iterations"], "depth": values["cat_depth"], "learning_rate": values["cat_learning_rate"], "l2_leaf_reg": values["cat_l2"], "max_training_rows": 0},
        "lstm": {"sequence_length": values["lstm_sequence_length"], "epochs": values["lstm_epochs"], "batch_size": values["lstm_batch_size"], "max_training_rows": 0},
    }


@callback(Output("cooling-upload-status", "children"), Input("cooling-measured-upload", "filename"), Input("cooling-measured-upload", "contents"))
def show_upload_status(filename: str | None, contents: str | None):
    if not contents:
        return html.P("No experimental CSV selected.", className="cooling-help")
    return html.P(f"Ready: {filename or 'uploaded CSV'}", className="preprocessing-export-success")


@callback(
    Output("cooling-room-length", "value"), Output("cooling-room-width", "value"), Output("cooling-room-height", "value"), Output("cooling-window-count", "value"), Output("cooling-window-area", "value"), Output("cooling-people-count", "value"), Output("cooling-sensible", "value"), Output("cooling-latent", "value"), Output("cooling-lighting", "value"), Output("cooling-misc", "value"), Output("cooling-it-load", "value"),
    Output("cooling-day-start", "value"), Output("cooling-day-end", "value"), Output("cooling-it-day", "value"), Output("cooling-it-night", "value"), Output("cooling-people-day", "value"), Output("cooling-people-night", "value"), Output("cooling-lighting-day", "value"), Output("cooling-lighting-night", "value"), Output("cooling-misc-day", "value"), Output("cooling-misc-night", "value"), Output("cooling-weekend", "value"),
    Output("cooling-shgc", "value"), Output("cooling-r-env", "value"), Output("cooling-r-int", "value"), Output("cooling-c-air", "value"), Output("cooling-synthetic-setpoint", "value"), Output("cooling-synthetic-supply-temp", "value"), Output("cooling-synthetic-supply-rh", "value"), Output("cooling-synthetic-return-rise", "value"), Output("cooling-synthetic-airflow", "value"), Output("cooling-synthetic-noise", "value"), Input("cooling-profile-select", "value"),
)
def load_cooling_profile(profile_id: str | None):
    profile = next((item["profile"] for item in list_cooling_profiles() if item["site_id"] == profile_id), None)
    return _profile_values(profile)


def _combine_datetime(date_value: str | None, time_value: str | None) -> str | None:
    return f"{date_value} {time_value or '00:00'}" if date_value else None


def _empty_run_outputs(message: str):
    return (None, html.P(message, className="preprocessing-export-error"), _summary_cards(None), _empty(message), _empty(message), _empty(message, 390), _empty(message, 390), _empty(message, 360), _empty(message, 360), _empty(message, 360), _empty(message, 300), simple_table(["Horizon", "MAE (kW)", "RMSE (kW)", "R²", "Bias (kW)", "Samples"], []), html.P("No calibration result."), html.P("The run did not produce a quality report."))


@callback(
    Output("cooling-result", "data"), Output("cooling-run-status", "children"), Output("cooling-kpi-grid", "children"), Output("cooling-forecast-graph", "figure"), Output("cooling-components-graph", "figure"), Output("cooling-residual-graph", "figure"), Output("cooling-scatter-graph", "figure"), Output("cooling-mae-graph", "figure"), Output("cooling-rmse-graph", "figure"), Output("cooling-r2-graph", "figure"), Output("cooling-calibration-graph", "figure"), Output("cooling-metrics-table", "children"), Output("cooling-calibration-summary", "children"), Output("cooling-data-quality", "children"),
    Input("cooling-run-backtest", "n_clicks"), Input("cooling-run-forecast", "n_clicks"), State("cooling-measurement-mode", "value"), State("cooling-measured-upload", "contents"), State("cooling-measured-upload", "filename"), State("cooling-profile-select", "value"), State("cooling-horizon-select", "value"), State("cooling-model-select", "value"), State("cooling-start-date", "date"), State("cooling-start-time", "value"), State("cooling-end-date", "date"), State("cooling-end-time", "value"), State("cooling-evaluation-options", "value"), State("cooling-calibration-iterations", "value"), State("cooling-interval-alpha", "value"), State("cooling-estimators", "value"), State("cooling-max-depth", "value"), State("cooling-learning-rate", "value"), State("cooling-subsample", "value"), State("cooling-colsample", "value"), State("cooling-max-training-rows", "value"), State("cooling-room-length", "value"), State("cooling-room-width", "value"), State("cooling-room-height", "value"), State("cooling-window-count", "value"), State("cooling-window-area", "value"), State("cooling-people-count", "value"), State("cooling-sensible", "value"), State("cooling-latent", "value"), State("cooling-lighting", "value"), State("cooling-misc", "value"), State("cooling-it-load", "value"), State("cooling-day-start", "value"), State("cooling-day-end", "value"), State("cooling-it-day", "value"), State("cooling-it-night", "value"), State("cooling-people-day", "value"), State("cooling-people-night", "value"), State("cooling-lighting-day", "value"), State("cooling-lighting-night", "value"), State("cooling-misc-day", "value"), State("cooling-misc-night", "value"), State("cooling-weekend", "value"), State("cooling-shgc", "value"), State("cooling-r-env", "value"), State("cooling-r-int", "value"), State("cooling-c-air", "value"), State("cooling-synthetic-setpoint", "value"), State("cooling-synthetic-supply-temp", "value"), State("cooling-synthetic-supply-rh", "value"), State("cooling-synthetic-return-rise", "value"), State("cooling-synthetic-airflow", "value"), State("cooling-synthetic-noise", "value"), State("system-select", "value"), State("cooling-system-select", "value"),
    State("cooling-model-select-ui", "value"), State("cooling-model-settings", "data"),
    prevent_initial_call=True,
)
def run_cooling(_backtest_clicks, _forecast_clicks, measurement_mode, upload_contents, upload_filename, profile_id, horizons, model, start_date, start_time, end_date, end_time, evaluation, calibration_iterations, interval_alpha, estimators, max_depth, learning_rate, subsample, colsample, max_training_rows, room_length, room_width, room_height, window_count, window_area, people_count, sensible, latent, lighting, misc, it_load, day_start, day_end, it_day, it_night, people_day, people_night, lighting_day, lighting_night, misc_day, misc_night, weekend, shgc, r_env, r_int, c_air, synthetic_setpoint, synthetic_supply_temp, synthetic_supply_rh, synthetic_return_rise, synthetic_airflow, synthetic_noise, _global_system_id, system_id, selected_models, model_settings):
    if not system_id:
        return _empty_run_outputs("Select a system from the global Data Analysis/system selector first.")
    selected_models = selected_models or ["xgboost"]
    model_settings = model_settings or {"xgboost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "max_training_rows": 0}}
    request = {
        "source_id": f"system-{int(system_id):02d}", "source_type": "system", "system_id": int(system_id), "profile_id": profile_id, "horizons": horizons, "models": selected_models, "model": selected_models[0], "measurement_mode": measurement_mode, "measurement_upload_contents": upload_contents, "measurement_upload_filename": upload_filename,
        "start": _combine_datetime(start_date, start_time), "end": _combine_datetime(end_date, end_time), "steady_state_only": "steady" in (evaluation or []), "auto_calibrate": "calibrate" in (evaluation or []), "calibration_iterations": calibration_iterations, "interval_alpha": interval_alpha,
        "model_settings": model_settings,
        "room_length_m": room_length, "room_width_m": room_width, "room_height_m": room_height, "window_count": window_count, "window_area_each_m2": window_area, "people_count": people_count, "sensible_w_per_person": sensible, "latent_w_per_person": latent, "lighting_w_per_m2": lighting, "non_it_misc_kw": misc, "it_load_kw": it_load,
        "day_start_hour": day_start, "day_end_hour": day_end, "it_day_multiplier": it_day, "it_night_multiplier": it_night, "people_day_multiplier": people_day, "people_night_multiplier": people_night, "lighting_day_multiplier": lighting_day, "lighting_night_multiplier": lighting_night, "misc_day_multiplier": misc_day, "misc_night_multiplier": misc_night, "weekend_multiplier": weekend,
        "shgc": shgc, "r_env_kw_per_k": r_env, "r_int_kw_per_k": r_int, "c_air_kj_per_k": c_air, "synthetic_indoor_setpoint_c": synthetic_setpoint, "synthetic_supply_temp_c": synthetic_supply_temp, "synthetic_supply_rh_pct": synthetic_supply_rh, "synthetic_return_rise_c": synthetic_return_rise, "synthetic_airflow_m3_s": synthetic_airflow, "synthetic_noise_std_kw": synthetic_noise,
    }
    try:
        result = generate_cooling_forecast(request) if ctx.triggered_id == "cooling-run-forecast" else run_cooling_backtest(request)
        rows = [[f"+{row['horizon_minutes']} min", f"{row['mae']:.3f}" if row.get("mae") is not None else "—", f"{row['rmse']:.3f}" if row.get("rmse") is not None else "—", f"{row['r2']:.3f}" if row.get("r2") is not None else "—", f"{row['bias']:.3f}" if row.get("bias") is not None else "—", str(row.get("n", 0))] for row in build_cooling_metrics(result)]
        calibration = result.get("calibration_summary") or {}
        calibration_children = info_rows([("Status", "Enabled" if calibration.get("enabled") else "Not run"), ("Iterations", str(calibration.get("n_iter", "—"))), ("Train RMSE", str((calibration.get("train_metrics") or {}).get("rmse", "—"))), ("Validation RMSE", str((calibration.get("val_metrics") or {}).get("rmse", "—")))])
        quality = result.get("quality") or {}
        quality_children = info_rows([("Source rows", f"{quality.get('source_rows', 0):,}"), ("Processed rows", f"{quality.get('processed_rows', 0):,}"), ("Missing cells", str(quality.get("missing_cells", 0))), ("Warnings", "; ".join(quality.get("warnings") or []) or "None")])
        history = pd.DataFrame(result.get("calibration_history") or [])
        calibration_figure = _empty("Calibration was not enabled.", 300)
        if not history.empty and "iter" in history and "rmse_train" in history:
            calibration_figure = _styled(go.Figure(go.Scatter(x=history["iter"], y=history["rmse_train"], mode="lines", name="Training RMSE", line={"color": "#6080d1"})), "Calibration search history", 300)
        return result, html.P("Latest forecast generated successfully." if ctx.triggered_id == "cooling-run-forecast" else "Backtest completed successfully.", className="preprocessing-export-success"), _summary_cards(result), _forecast_figure(result), _components_figure(result), _residual_figure(result), _scatter_figure(result), _metric_figure(result, "mae", "MAE by horizon", "MAE (kW)"), _metric_figure(result, "rmse", "RMSE by horizon", "RMSE (kW)"), _metric_figure(result, "r2", "R² by horizon", "R²"), calibration_figure, simple_table(["Horizon", "MAE (kW)", "RMSE (kW)", "R²", "Bias (kW)", "Samples"], rows), calibration_children, quality_children
    except Exception as exc:
        return _empty_run_outputs(str(exc) if isinstance(exc, CoolingDemandError) else f"Cooling run failed: {exc}")


@callback(
    Output("cooling-profile-status", "children"), Output("cooling-profile-select", "options"), Output("cooling-profile-select", "value"), Input("cooling-save-profile", "n_clicks"), State("cooling-profile-select", "value"), State("cooling-room-length", "value"), State("cooling-room-width", "value"), State("cooling-room-height", "value"), State("cooling-window-count", "value"), State("cooling-window-area", "value"), State("cooling-people-count", "value"), State("cooling-sensible", "value"), State("cooling-latent", "value"), State("cooling-lighting", "value"), State("cooling-misc", "value"), State("cooling-it-load", "value"), State("cooling-day-start", "value"), State("cooling-day-end", "value"), State("cooling-it-day", "value"), State("cooling-it-night", "value"), State("cooling-people-day", "value"), State("cooling-people-night", "value"), State("cooling-lighting-day", "value"), State("cooling-lighting-night", "value"), State("cooling-misc-day", "value"), State("cooling-misc-night", "value"), State("cooling-weekend", "value"), State("cooling-shgc", "value"), State("cooling-r-env", "value"), State("cooling-r-int", "value"), State("cooling-c-air", "value"), State("cooling-synthetic-setpoint", "value"), State("cooling-synthetic-supply-temp", "value"), State("cooling-synthetic-supply-rh", "value"), State("cooling-synthetic-return-rise", "value"), State("cooling-synthetic-airflow", "value"), State("cooling-synthetic-noise", "value"), prevent_initial_call=True,
)
def save_profile(_clicks, profile_id, room_length, room_width, room_height, window_count, window_area, people_count, sensible, latent, lighting, misc, it_load, day_start, day_end, it_day, it_night, people_day, people_night, lighting_day, lighting_night, misc_day, misc_night, weekend, shgc, r_env, r_int, c_air, setpoint, supply_temp, supply_rh, return_rise, airflow, noise):
    profile = next((item["profile"] for item in list_cooling_profiles() if item["site_id"] == profile_id), None)
    if not profile:
        return html.P("Select a profile before saving.", className="preprocessing-export-error"), no_update, no_update
    profile = json.loads(json.dumps(profile))
    profile.setdefault("hall", {}).update({"room_length_m": room_length, "room_width_m": room_width, "room_height_m": room_height, "window_count": window_count, "window_area_each_m2": window_area, "people_count": people_count, "sensible_w_per_person": sensible, "latent_w_per_person": latent, "lighting_w_per_m2": lighting, "non_it_misc_kw": misc, "it_load_kw": it_load})
    profile.setdefault("internal_load_schedule", {}).update({"day_start_hour": day_start, "day_end_hour": day_end, "it_day_multiplier": it_day, "it_night_multiplier": it_night, "people_day_multiplier": people_day, "people_night_multiplier": people_night, "lighting_day_multiplier": lighting_day, "lighting_night_multiplier": lighting_night, "misc_day_multiplier": misc_day, "misc_night_multiplier": misc_night, "weekend_multiplier": weekend})
    profile.setdefault("physics_initial", {}).update({"shgc": shgc, "r_env_kw_per_k": r_env, "r_int_kw_per_k": r_int, "c_air_kj_per_k": c_air})
    profile.setdefault("synthetic_measurement", {}).update({"indoor_temp_setpoint_c": setpoint, "supply_temp_c": supply_temp, "supply_rh_pct": supply_rh, "return_rise_c": return_rise, "airflow_m3_s": airflow, "measurement_noise_std_kw": noise})
    try:
        save_cooling_profile(profile)
        return html.P("Profile saved.", className="preprocessing-export-success"), [{"label": item["label"], "value": item["site_id"]} for item in list_cooling_profiles()], profile_id
    except Exception as exc:
        return html.P(str(exc), className="preprocessing-export-error"), no_update, no_update


@callback(
    Output("cooling-export-status", "children"), Output("cooling-download-forecast", "data"), Output("cooling-download-metrics", "data"), Output("cooling-download-summary", "data"), Output("cooling-download-calibration", "data"), Output("cooling-download-profile", "data"), Input("cooling-download-forecast-btn", "n_clicks"), Input("cooling-download-metrics-btn", "n_clicks"), Input("cooling-download-summary-btn", "n_clicks"), Input("cooling-download-calibration-btn", "n_clicks"), Input("cooling-download-profile-btn", "n_clicks"), State("cooling-result", "data"), prevent_initial_call=True,
)
def export_cooling(_forecast_clicks, _metrics_clicks, _summary_clicks, _calibration_clicks, _profile_clicks, result):
    if not result:
        return html.P("Run a cooling forecast before exporting artifacts.", className="preprocessing-export-error"), no_update, no_update, no_update, no_update, no_update
    try:
        paths = export_cooling_result(result)
        records = pd.DataFrame(result.get("records") or [])
        metrics = pd.DataFrame(build_cooling_metrics(result))
        calibration = pd.DataFrame(result.get("calibration_history") or [])
        profile = result.get("metadata", {}).get("profile") or {}
        triggered = ctx.triggered_id
        return html.P(f"Artifacts saved: {paths['summary_json']}", className="preprocessing-export-success"), dcc.send_data_frame(records.to_csv, "cooling_forecast.csv", index=False) if triggered == "cooling-download-forecast-btn" else no_update, dcc.send_data_frame(metrics.to_csv, "cooling_metrics.csv", index=False) if triggered == "cooling-download-metrics-btn" else no_update, dcc.send_string(json.dumps(result, indent=2, default=str), "cooling_run_summary.json") if triggered == "cooling-download-summary-btn" else no_update, dcc.send_data_frame(calibration.to_csv, "cooling_calibration_history.csv", index=False) if triggered == "cooling-download-calibration-btn" else no_update, dcc.send_string(json.dumps(profile, indent=2, default=str), "cooling_profile.json") if triggered == "cooling-download-profile-btn" else no_update
    except Exception as exc:
        return html.P(str(exc), className="preprocessing-export-error"), no_update, no_update, no_update, no_update, no_update

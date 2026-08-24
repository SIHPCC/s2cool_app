from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.ui import info_rows, kpi_card, panel, pill, simple_table
from models.domain import SystemRecord
from services.config_service import load_model_defaults
from services.forecasting_service import (
    DEFAULT_MODEL_SETTINGS,
    MODEL_SPECS,
    ForecastingError,
    build_forecast_metrics,
    export_forecast_result,
    load_trained_model,
    list_forecast_artifacts,
    model_options,
    predict_trained_model,
    predict_trained_model_range,
    serialize_trained_model,
    train_model,
)


FORECAST_HORIZON_OPTIONS = [
    (5, "5 min"),
    (10, "10 min"),
    (15, "15 min"),
    (20, "20 min"),
    (30, "30 min"),
    (60, "1 hr"),
    (120, "2 hr"),
]


def _time_options() -> list[dict[str, str]]:
    return [
        {"label": f"{hour:02d}:{minute:02d}", "value": f"{hour:02d}:{minute:02d}"}
        for hour in range(24)
        for minute in range(0, 60, 5)
    ]


def _tuning_input(control_id: str, value: Any, step: float = 1, minimum: float | None = None) -> dcc.Input:
    props = {"id": control_id, "type": "number", "value": value, "step": step, "className": "pv-tuning-input"}
    if minimum is not None:
        props["min"] = minimum
    return dcc.Input(**props)


def _tuning_card(model_name: str, title: str, fields: list[tuple[str, str, Any, float, float | None]]) -> html.Div:
    return html.Div(
        id=f"pv-settings-{model_name}",
        className="pv-model-settings-card",
        children=[
            html.H4(title, className="pv-model-settings-title"),
            html.Div(
                className="pv-tuning-grid",
                children=[
                    html.Div([html.Label(label, className="pv-tuning-label"), _tuning_input(control_id, value, step, minimum)], className="pv-tuning-field")
                    for label, control_id, value, step, minimum in fields
                ],
            ),
        ],
    )


def _model_tuning_layout() -> html.Div:
    defaults = DEFAULT_MODEL_SETTINGS
    return html.Div(
        className="pv-model-settings-grid",
        children=[
            _tuning_card("xgboost", "XGBoost settings", [
                ("Estimators", "pv-xgb-n-estimators", defaults["xgboost"]["n_estimators"], 10, 10),
                ("Max depth", "pv-xgb-max-depth", defaults["xgboost"]["max_depth"], 1, 1),
                ("Learning rate", "pv-xgb-learning-rate", defaults["xgboost"]["learning_rate"], 0.01, 0.001),
                ("Subsample", "pv-xgb-subsample", defaults["xgboost"]["subsample"], 0.05, 0.1),
                ("Column sample", "pv-xgb-colsample", defaults["xgboost"]["colsample_bytree"], 0.05, 0.1),
            ]),
            _tuning_card("extra-trees", "Extra Trees settings", [
                ("Estimators", "pv-et-n-estimators", defaults["extra_trees"]["n_estimators"], 10, 10),
                ("Max depth (0 = auto)", "pv-et-max-depth", defaults["extra_trees"]["max_depth"], 1, 0),
                ("Min samples leaf", "pv-et-min-leaf", defaults["extra_trees"]["min_samples_leaf"], 1, 1),
                ("Max features", "pv-et-max-features", defaults["extra_trees"]["max_features"], 0.05, 0.1),
            ]),
            _tuning_card("random-forest", "Random Forest settings", [
                ("Estimators", "pv-rf-n-estimators", defaults["random_forest"]["n_estimators"], 10, 10),
                ("Max depth (0 = auto)", "pv-rf-max-depth", defaults["random_forest"]["max_depth"], 1, 0),
                ("Min samples leaf", "pv-rf-min-leaf", defaults["random_forest"]["min_samples_leaf"], 1, 1),
                ("Max features", "pv-rf-max-features", defaults["random_forest"]["max_features"], 0.05, 0.1),
            ]),
            _tuning_card("lightgbm", "LightGBM settings", [
                ("Estimators", "pv-lgb-n-estimators", defaults["lightgbm"]["n_estimators"], 10, 10),
                ("Learning rate", "pv-lgb-learning-rate", defaults["lightgbm"]["learning_rate"], 0.01, 0.001),
                ("Number of leaves", "pv-lgb-num-leaves", defaults["lightgbm"]["num_leaves"], 1, 2),
                ("Max depth (-1 = auto)", "pv-lgb-max-depth", defaults["lightgbm"]["max_depth"], 1, -1),
            ]),
            _tuning_card("catboost", "CatBoost settings", [
                ("Iterations", "pv-cat-iterations", defaults["catboost"]["iterations"], 10, 10),
                ("Depth", "pv-cat-depth", defaults["catboost"]["depth"], 1, 1),
                ("Learning rate", "pv-cat-learning-rate", defaults["catboost"]["learning_rate"], 0.01, 0.001),
                ("L2 regularization", "pv-cat-l2", defaults["catboost"]["l2_leaf_reg"], 0.5, 0),
            ]),
            _tuning_card("lstm", "LSTM settings", [
                ("Sequence length", "pv-lstm-sequence-length", defaults["lstm"]["sequence_length"], 1, 2),
                ("Epochs", "pv-lstm-epochs", defaults["lstm"]["epochs"], 1, 1),
                ("Batch size", "pv-lstm-batch-size", defaults["lstm"]["batch_size"], 1, 1),
            ]),
        ],
    )


def _empty_figure(message: str, height: int = 420) -> dict:
    return {
        "data": [],
        "layout": {
            "template": "plotly_white", "height": height,
            "margin": {"l": 52, "r": 24, "t": 34, "b": 48},
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "annotations": [{"text": message, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 14, "color": "#7b88a5"}}],
        },
    }


def _records(result: dict[str, Any], model: str | None = None) -> pd.DataFrame:
    selected = model or result.get("model", "xgboost")
    if selected == "comparison":
        selected = (result.get("models") or ["xgboost"])[0]
    payload = result.get("results", {}).get(selected, {})
    return pd.DataFrame(payload.get("records") or [])


def _multi_forecast_figure(result: dict[str, Any], show_weather: bool = False) -> dict:
    model_names = list(result.get("models") or ["xgboost"])
    frames = {name: _records(result, name) for name in model_names}
    frames = {name: frame for name, frame in frames.items() if not frame.empty}
    if not frames:
        return _empty_figure("Selected models did not produce forecast records.")
    first = next(iter(frames.values())).copy()
    first["_ts"] = pd.to_datetime(first["_ts"], errors="coerce")
    traces = [{"x": first["_ts"].tolist(), "y": first["measured_kw"].tolist(), "name": "Measured", "mode": "lines", "line": {"color": "#2ca02c", "width": 2}}]
    colors = ["#4f72cb", "#e67e22", "#8e44ad", "#16a085", "#d35400", "#c0392b"]
    dashes = ["solid", "dash", "dot", "dashdot"]
    for model_index, (model_name, frame) in enumerate(frames.items()):
        frame = frame.copy()
        frame["_ts"] = pd.to_datetime(frame["_ts"], errors="coerce")
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        for horizon_index, horizon in enumerate(result.get("horizons", [])):
            column = f"pred_{horizon}m_kw"
            if column in frame:
                traces.append({"x": frame["_ts"].tolist(), "y": frame[column].tolist(), "name": f"{label} +{horizon} min", "mode": "lines", "line": {"color": colors[model_index % len(colors)], "dash": dashes[horizon_index % len(dashes)], "width": 2}})
    if show_weather and "ghi_pyr" in first:
        traces.append({"x": first["_ts"].tolist(), "y": first["ghi_pyr"].tolist(), "name": "GHI (W/m²)", "mode": "lines", "yaxis": "y2", "line": {"color": "#f0a202", "width": 1}})
    layout = {"template": "plotly_white", "height": 450, "margin": {"l": 52, "r": 24, "t": 34, "b": 48}, "hovermode": "x unified", "xaxis": {"title": "Timestamp", "showgrid": False}, "yaxis": {"title": "PV Power (kW)", "gridcolor": "#edf1f8"}, "legend": {"orientation": "h", "y": 1.08, "x": 0}}
    if show_weather:
        layout["yaxis2"] = {"title": "GHI (W/m²)", "overlaying": "y", "side": "right", "showgrid": False}
    return {"data": traces, "layout": layout}


def _multi_residual_figure(result: dict[str, Any]) -> dict:
    traces = []
    for model_name in result.get("models") or ["xgboost"]:
        frame = _records(result, model_name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_ts"] = pd.to_datetime(frame["_ts"], errors="coerce")
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        for horizon in result.get("horizons", []):
            actual, pred = f"measured_at_{horizon}m", f"pred_{horizon}m"
            if actual in frame and pred in frame:
                residual = pd.to_numeric(frame[pred], errors="coerce") - pd.to_numeric(frame[actual], errors="coerce")
                traces.append({"x": frame["_ts"].tolist(), "y": residual.tolist(), "name": f"{label} +{horizon} min", "mode": "lines"})
    if not traces:
        return _empty_figure("Selected models did not produce residual records.", 360)
    return {"data": traces, "layout": {"template": "plotly_white", "height": 360, "margin": {"l": 52, "r": 24, "t": 34, "b": 48}, "hovermode": "x unified", "xaxis": {"title": "Timestamp"}, "yaxis": {"title": "Prediction error (p.u.)", "zeroline": True, "zerolinecolor": "#d8dfed"}}}


def _forecast_figure(result: dict[str, Any], model: str | None = None, show_weather: bool = False) -> dict:
    if model is None and len(result.get("models") or []) > 1:
        return _multi_forecast_figure(result, show_weather)
    frame = _records(result, model)
    if frame.empty:
        return _empty_figure("Run a backtest or generate a forecast to view the PV curve.")
    frame["_ts"] = pd.to_datetime(frame["_ts"], errors="coerce")
    traces = [{"x": frame["_ts"].tolist(), "y": frame["measured_kw"].tolist(), "name": "Measured", "mode": "lines", "line": {"color": "#2ca02c", "width": 2}}]
    for horizon in result.get("horizons", []):
        column = f"pred_{horizon}m_kw"
        if column in frame:
            traces.append({"x": frame["_ts"].tolist(), "y": frame[column].tolist(), "name": f"Forecast +{horizon} min", "mode": "lines", "line": {"dash": "dash"}})
    if show_weather and "ghi_pyr" in frame:
        traces.append({"x": frame["_ts"].tolist(), "y": frame["ghi_pyr"].tolist(), "name": "GHI (W/m²)", "mode": "lines", "yaxis": "y2", "line": {"color": "#f0a202", "width": 1}})
    layout = {"template": "plotly_white", "height": 450, "margin": {"l": 52, "r": 24, "t": 34, "b": 48}, "hovermode": "x unified", "xaxis": {"title": "Timestamp", "showgrid": False}, "yaxis": {"title": "PV Power (kW)", "gridcolor": "#edf1f8"}, "legend": {"orientation": "h", "y": 1.08, "x": 0}}
    if show_weather:
        layout["yaxis2"] = {"title": "GHI (W/m²)", "overlaying": "y", "side": "right", "showgrid": False}
    return {"data": traces, "layout": layout}


def _residual_figure(result: dict[str, Any], model: str | None = None) -> dict:
    if model is None and len(result.get("models") or []) > 1:
        return _multi_residual_figure(result)
    frame = _records(result, model)
    if frame.empty:
        return _empty_figure("Residual diagnostics will appear after a backtest.", 360)
    frame["_ts"] = pd.to_datetime(frame["_ts"], errors="coerce")
    traces = []
    for horizon in result.get("horizons", []):
        actual = f"measured_at_{horizon}m"
        pred = f"pred_{horizon}m"
        if actual in frame and pred in frame:
            residual = pd.to_numeric(frame[pred], errors="coerce") - pd.to_numeric(frame[actual], errors="coerce")
            traces.append({"x": frame["_ts"].tolist(), "y": residual.tolist(), "name": f"+{horizon} min", "mode": "lines"})
    return {"data": traces, "layout": {"template": "plotly_white", "height": 360, "margin": {"l": 52, "r": 24, "t": 34, "b": 48}, "hovermode": "x unified", "xaxis": {"title": "Timestamp"}, "yaxis": {"title": "Prediction error (p.u.)", "zeroline": True, "zerolinecolor": "#d8dfed"}}}


def _diagnostic_metric_figure(result: dict[str, Any], metric: str, title: str, y_title: str) -> dict:
    """Build a grouped model-by-horizon diagnostic chart."""
    metric_rows = [row for row in build_forecast_metrics(result) if row.get(metric) is not None]
    if not metric_rows:
        return _empty_figure(f"{title} will appear after a successful backtest.", 400)

    horizons = result.get("horizons") or []
    labels = [f"+{h} min" for h in horizons]
    colors = ["#4f72cb", "#e67e22", "#2e8b72", "#8e44ad", "#bd3f5c", "#55758c"]
    traces = []
    for model_index, model_name in enumerate(result.get("models") or [result.get("model", "xgboost")]):
        model_rows = {int(row["horizon_minutes"]): row for row in metric_rows if row.get("model") == model_name}
        values = [model_rows.get(h, {}).get(metric) for h in horizons]
        if not any(value is not None for value in values):
            continue
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        sample_counts = [model_rows.get(h, {}).get("n", 0) for h in horizons]
        traces.append({
            "type": "bar",
            "x": labels,
            "y": values,
            "name": label,
            "marker": {"color": colors[model_index % len(colors)]},
            "text": ["" if value is None else f"{float(value):.3f}" for value in values],
            "textposition": "outside",
            "cliponaxis": False,
            "customdata": sample_counts,
            "hovertemplate": f"%{{x}}<br>{label}: %{{y:.4f}}<br>Valid samples: %{{customdata:,}}<extra></extra>",
        })
    if not traces:
        return _empty_figure(f"No {title.lower()} values are available for this run.", 400)
    return {
        "data": traces,
        "layout": {
            "template": "plotly_white",
            "height": 400,
            "margin": {"l": 62, "r": 24, "t": 58, "b": 92},
            "barmode": "group",
            "hovermode": "x unified",
            "font": {"family": "Segoe UI, Tahoma, Arial, sans-serif", "size": 11, "color": "#31456d"},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "title": {"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 14, "color": "#213b6b"}},
            "xaxis": {"title": "Forecast horizon", "showgrid": False, "linecolor": "#cbd5e8", "mirror": True},
            "yaxis": {"title": y_title, "gridcolor": "#e5ebf5", "zeroline": True, "zerolinecolor": "#b9c5d8", "rangemode": "tozero", "linecolor": "#cbd5e8", "mirror": True},
            "legend": {"orientation": "h", "y": -0.28, "yanchor": "top", "x": 0, "xanchor": "left"},
            "uniformtext": {"minsize": 9, "mode": "hide"},
        },
    }


def _r2_scatter_figure(result: dict[str, Any]) -> dict:
    """Show measured-versus-predicted samples for the R² diagnostic."""
    traces = []
    horizon_colors = ["#4f72cb", "#e67e22", "#2e8b72", "#8e44ad", "#bd3f5c", "#55758c", "#c48a24"]
    model_symbols = ["circle", "diamond", "square", "triangle-up", "cross", "x"]
    all_values: list[float] = []

    for model_index, model_name in enumerate(result.get("models") or [result.get("model", "xgboost")]):
        frame = _records(result, model_name)
        if frame.empty:
            continue
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        metric_rows = {
            int(row["horizon_minutes"]): row
            for row in build_forecast_metrics(result)
            if row.get("model") == model_name
        }
        for horizon_index, horizon in enumerate(result.get("horizons") or []):
            measured_column = f"measured_at_{horizon}m"
            predicted_column = f"pred_{horizon}m"
            if measured_column not in frame or predicted_column not in frame:
                continue
            valid = frame[[measured_column, predicted_column]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) < 2:
                continue
            metric_row = metric_rows.get(horizon, {})
            r2 = metric_row.get("r2")
            sample_count = len(valid)
            # Keep browser payloads responsive for long backtests while
            # retaining a representative measured-versus-predicted cloud.
            plot_valid = valid
            if len(plot_valid) > 2500:
                stride = max(int(len(plot_valid) / 2500), 1)
                plot_valid = plot_valid.iloc[::stride]
            x_values = plot_valid[measured_column].tolist()
            y_values = plot_valid[predicted_column].tolist()
            all_values.extend(x_values)
            all_values.extend(y_values)
            r2_text = "-" if r2 is None else f"{float(r2):.3f}"
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": x_values,
                "y": y_values,
                "name": f"{label} +{horizon} min | R² {r2_text} | N={sample_count:,}",
                "marker": {
                    "color": horizon_colors[horizon_index % len(horizon_colors)],
                    "symbol": model_symbols[model_index % len(model_symbols)],
                    "size": 5,
                    "opacity": 0.58,
                    "line": {"width": 0.45, "color": "#ffffff"},
                },
                "hovertemplate": "Measured: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
            })

    if not traces or len(all_values) < 2:
        return _diagnostic_metric_figure(result, "r2", "R² by forecast horizon", "R²")

    low = min(all_values)
    high = max(all_values)
    padding = max((high - low) * 0.06, 0.01)
    axis_min = 0.0 if low >= 0 else low - padding
    axis_max = high + padding
    traces.insert(0, {
        "type": "scatter",
        "mode": "lines",
        "x": [axis_min, axis_max],
        "y": [axis_min, axis_max],
        "name": "Ideal 1:1",
        "line": {"color": "#9aa7bd", "dash": "dash", "width": 1.5},
        "hoverinfo": "skip",
    })
    return {
        "data": traces,
        "layout": {
            "template": "plotly_white",
            "height": 400,
            "margin": {"l": 62, "r": 24, "t": 60, "b": 108},
            "hovermode": "closest",
            "font": {"family": "Segoe UI, Tahoma, Arial, sans-serif", "size": 11, "color": "#31456d"},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "title": {"text": "Predicted versus measured power", "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top", "font": {"size": 14, "color": "#213b6b"}},
            "xaxis": {"title": "Measured power (p.u.)", "range": [axis_min, axis_max], "gridcolor": "#e5ebf5", "linecolor": "#cbd5e8", "mirror": True, "zeroline": True, "zerolinecolor": "#9aa7bd"},
            "yaxis": {"title": "Predicted power (p.u.)", "range": [axis_min, axis_max], "gridcolor": "#e5ebf5", "scaleanchor": "x", "scaleratio": 1, "linecolor": "#cbd5e8", "mirror": True, "zeroline": True, "zerolinecolor": "#9aa7bd"},
            "legend": {"orientation": "h", "y": -0.31, "yanchor": "top", "x": 0, "xanchor": "left"},
        },
    }


def _metric_table(result: dict[str, Any]) -> html.Table:
    rows = []
    for row in build_forecast_metrics(result):
        rows.append([
            row.get("model", "-").upper(), f"+{row.get('horizon_minutes', '-') } min",
            "-" if row.get("mae") is None else f"{row['mae']:.4f}",
            "-" if row.get("rmse") is None else f"{row['rmse']:.4f}",
            "-" if row.get("r2") is None else f"{row['r2']:.4f}",
            "-" if row.get("bias") is None else f"{row['bias']:.4f}", str(row.get("n", 0)),
        ])
    return simple_table(["Model", "Horizon", "MAE", "RMSE", "R²", "Bias", "N"], rows or [["-", "-", "-", "-", "-", "-", "0"]])


def _summary_cards(result: dict[str, Any] | None) -> list:
    if not result:
        return [kpi_card("Latest run", "Not run", "Choose an artifact and run a forecast")]
    models = result.get("models") or [result.get("model", "xgboost")]
    horizons = result.get("horizons") or [5]

    def summary_record(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        daylight = payload.get("daylight_record")
        if isinstance(daylight, dict) and daylight:
            return daylight, True
        records = payload.get("records") or []
        if not records:
            return {}, False
        daylight_records = [
            row for row in records
            if row.get("ghi_pyr") is not None and float(row.get("ghi_pyr") or 0) > 10
        ]
        return (daylight_records[-1], True) if daylight_records else (records[-1], False)

    first_payload = next(
        (result.get("results", {}).get(model_name, {}) for model_name in models
         if not result.get("results", {}).get(model_name, {}).get("error")),
        {},
    )
    reference_record, is_daylight = summary_record(first_payload)
    reference_timestamp = reference_record.get("_ts") or result.get("latest_timestamp", "-")
    cards = [
        kpi_card(
            "Forecast timestamp",
            str(reference_timestamp)[:19],
            "Latest daylight record" if is_daylight else f"{len(models)} model(s) selected | latest available record",
        )
    ]
    for model_name in models:
        payload = result.get("results", {}).get(model_name, {})
        label = MODEL_SPECS.get(model_name, {}).get("label", model_name)
        if payload.get("error"):
            cards.append(kpi_card(label, "Unavailable", "Check model installation"))
            continue
        record, model_is_daylight = summary_record(payload)
        values = []
        for horizon in horizons:
            value = record.get(f"pred_{horizon}m_kw")
            formatted = "-" if value is None else f"{float(value):.3f} kW"
            values.append(html.Span(f"+{horizon} min: {formatted}", className="pv-kpi-horizon-item"))
        cards.append(
            kpi_card(
                label,
                html.Div(values, className="pv-kpi-horizon-values"),
                "Latest daylight forecast" if model_is_daylight else "Latest available record",
            )
        )
    return cards


def _artifact_system_parameters(metadata: dict[str, Any]) -> html.Div:
    quality = metadata.get("quality_summary") or {}
    system_id = metadata.get("system_id")
    capacity = metadata.get("capacity_kw")
    interval = metadata.get("sampling_interval_minutes")
    rows = [
        ["Data Points", f"{int(quality.get('rows_after', 0)):,}" if quality.get("rows_after") is not None else "-"],
        ["Timestamp Interval", f"{float(interval):.2f} minutes" if interval is not None else "Unavailable"],
        ["Start Time", str(quality.get("start_time") or "-")],
        ["End Time", str(quality.get("end_time") or "-")],
        ["System ID", f"System {int(system_id):02d}" if system_id is not None else "-"],
        ["City", str(metadata.get("city") or "-")],
        ["Capacity", f"{float(capacity):.3g} kW" if capacity is not None else "-"],
        ["Location", f"Lat {float(metadata['lat']):.6f}, Lon {float(metadata['lon']):.6f}" if metadata.get("lat") is not None and metadata.get("lon") is not None else "-"],
    ]
    return html.Div(
        className="pv-system-parameters",
        children=[
            simple_table(["Parameter", "Value"], rows),
        ],
    )


def build_layout(system: SystemRecord | None) -> html.Div:
    defaults = load_model_defaults().get("pv_forecasting", {})
    artifacts = list_forecast_artifacts()
    options = [
        {
            "label": (
                f"System {a.metadata.get('system_id', '-')} | "
                f"{a.metadata.get('city', 'Unknown')} | "
                f"{float(a.metadata.get('capacity_kw')):.3g} kW"
            ),
            "value": a.artifact_id,
        }
        for a in artifacts
    ]
    default_artifact = options[0]["value"] if options else None
    return html.Div(className="pv-forecasting-page", children=[
        dcc.Store(id="pv-system-id", data=system.system_number if system else None),
        dcc.Store(id="pv-forecast-result", data=None),
        dcc.Store(id="pv-model-settings", data=DEFAULT_MODEL_SETTINGS),
        dcc.Store(id="pv-trained-model", data=None),
        dcc.Download(id="pv-download-model"), dcc.Download(id="pv-download-forecast"), dcc.Download(id="pv-download-metrics"), dcc.Download(id="pv-download-summary"),
        panel("PV Forecasting Workbench", "PV forecasting connected to the exported Data Analysis preprocessing profile.", [
            html.Div(className="two-col-grid", children=[
                html.Div(className="panel pv-panel pv-setup-panel", children=[html.H3("Forecast setup", className="section-title"),
                    html.Label("Data Analysis artifact (all systems)", className="muted"), dcc.Dropdown(id="pv-artifact-select", options=options, value=default_artifact, clearable=False, placeholder="Export a preprocessing artifact first"),
                    html.Div(id="pv-artifact-status", className="preprocessing-summary"),
                    html.Label("Forecast horizons", className="muted"), dcc.Checklist(id="pv-horizon-select", className="pv-checklist pv-horizon-checklist", options=[{"label": f" {label}", "value": value} for value, label in FORECAST_HORIZON_OPTIONS], value=list(defaults.get("horizons_minutes", [5, 15, 30])), inline=True),
                    html.Label("Forecast models", className="muted"), dcc.Checklist(id="pv-model-select", className="pv-checklist pv-model-checklist", options=model_options(), value=["xgboost"], inline=False),
                    html.Div("Select one or more models. Each selected model is trained and plotted separately.", className="pv-model-help"),
                    html.Div(className="pv-model-tuning-section", children=[
                        html.H3("Model tuning", className="pv-tuning-heading"),
                        html.P("Adjust the parameters for each selected model before running the forecast.", className="pv-model-help"),
                        _model_tuning_layout(),
                    ]),
                    dcc.Checklist(id="pv-weather-select", className="pv-checklist pv-weather-checklist", options=[{"label": " Show GHI trace", "value": "ghi"}], value=[], inline=True),
                    html.Div(className="action-button-row", children=[
                        html.Button("Train Model", id="pv-train-model", className="action-btn action-btn-primary"),
                        html.Button("Save Trained Model", id="pv-save-model-btn", className="action-btn", disabled=True),
                        dcc.Upload(id="pv-trained-model-upload", contents=None, filename=None, children=html.Button("Upload Saved Model", id="pv-upload-model-btn", className="action-btn"), className="pv-upload-model"),
                    ]),
                    html.Div(id="pv-upload-model-status", className="preprocessing-export-status"),
                    html.Div(id="pv-save-model-status", className="preprocessing-export-status"),
                    html.Div(id="pv-run-status", className="preprocessing-export-status"),
                ]),
                html.Div(className="panel pv-panel pv-source-panel", children=[
                    html.H3("Forecast prediction", className="section-title"),
                    html.Div(id="pv-artifact-system-info", children=html.P("Select a Data Analysis artifact to view system parameters.", className="section-subtitle")),
                    html.Div(id="pv-run-summary", className="preprocessing-summary"),
                ]),
                html.Div(className="panel pv-panel pv-prediction-panel", children=[
                    html.H3("Prediction controls", className="section-title"),
                    html.Div(className="pv-prediction-controls", children=[
                        html.Label("Prediction date", className="muted"),
                        dcc.DatePickerSingle(id="pv-predict-date", date=defaults.get("test_end", "2026-05-03"), display_format="YYYY-MM-DD", clearable=False, className="pv-date-picker"),
                        html.Label("Prediction time", className="muted"),
                        dcc.Dropdown(id="pv-predict-time", options=_time_options(), value="12:00", clearable=False, className="pv-time-select"),
                        html.Button("Predict Forecast", id="pv-predict-forecast", className="action-btn action-btn-primary", disabled=True),
                    ]),
                ])
            ])
        ]),
        html.Div(id="pv-kpi-grid", className="kpi-grid", children=_summary_cards(None)),
        panel("Forecast curve", "Generate a forecast curve for a selected date and time range.", [
            html.Div(className="pv-curve-controls", children=[
                html.Div(className="pv-curve-field", children=[
                    html.Label("Start date", className="muted"),
                    dcc.DatePickerSingle(id="pv-curve-start-date", date=defaults.get("test_start", defaults.get("test_end", "2026-05-03")), display_format="YYYY-MM-DD", clearable=False, className="pv-date-picker"),
                ]),
                html.Div(className="pv-curve-field", children=[
                    html.Label("Start time", className="muted"),
                    dcc.Dropdown(id="pv-curve-start-time", options=_time_options(), value="00:00", clearable=False, className="pv-time-select"),
                ]),
                html.Div(className="pv-curve-field", children=[
                    html.Label("End date", className="muted"),
                    dcc.DatePickerSingle(id="pv-curve-end-date", date=defaults.get("test_end", "2026-05-03"), display_format="YYYY-MM-DD", clearable=False, className="pv-date-picker"),
                ]),
                html.Div(className="pv-curve-field", children=[
                    html.Label("End time", className="muted"),
                    dcc.Dropdown(id="pv-curve-end-time", options=_time_options(), value="23:55", clearable=False, className="pv-time-select"),
                ]),
                html.Button("Generate Forecast Curve", id="pv-generate-forecast-curve", className="action-btn action-btn-primary", disabled=True),
            ]),
            dcc.Loading(
                id="pv-forecast-loading",
                type="circle",
                color="#4f72cb",
                children=dcc.Graph(
                    id="pv-forecast-graph",
                    figure=_empty_figure("Train or upload a model, then generate a forecast curve."),
                    className="trend-graph"
                )
            ),
        ]),
        html.Div(className="two-col-grid", children=[
            panel("Evaluation metrics", "MAE, RMSE, R², bias, and sample count for each horizon.", [html.Div(id="pv-metrics-table", children=_metric_table({}))]),
            panel("Forecast diagnostics", "Review residual behavior and error metrics by model and forecast horizon.", [
                dcc.Tabs(
                    id="pv-diagnostic-tabs",
                    value="residual",
                    className="pv-diagnostic-tabs",
                    children=[
                        dcc.Tab(label="Residuals", value="residual", className="pv-diagnostic-tab", selected_className="pv-diagnostic-tab-selected", children=dcc.Graph(id="pv-diagnostic-residual-graph", figure=_empty_figure("Residual diagnostics will appear after a backtest.", 400), className="trend-graph")),
                        dcc.Tab(label="R² / Predicted vs Measured", value="r2", className="pv-diagnostic-tab", selected_className="pv-diagnostic-tab-selected", children=dcc.Graph(id="pv-diagnostic-r2-graph", figure=_empty_figure("R² predicted-versus-measured diagnostics will appear after a backtest.", 400), className="trend-graph")),
                        dcc.Tab(label="MAE", value="mae", className="pv-diagnostic-tab", selected_className="pv-diagnostic-tab-selected", children=dcc.Graph(id="pv-diagnostic-mae-graph", figure=_empty_figure("MAE will appear after a backtest.", 400), className="trend-graph")),
                        dcc.Tab(label="RMSE", value="rmse", className="pv-diagnostic-tab", selected_className="pv-diagnostic-tab-selected", children=dcc.Graph(id="pv-diagnostic-rmse-graph", figure=_empty_figure("RMSE will appear after a backtest.", 400), className="trend-graph")),
                    ],
                ),
            ]),
        ]),
        panel("Run artifacts", "Download reproducible outputs after a successful run.", [html.Div(className="action-button-row", children=[html.Button("Download Forecast CSV", id="pv-download-forecast-btn", className="action-btn"), html.Button("Download Metrics CSV", id="pv-download-metrics-btn", className="action-btn"), html.Button("Download Run Summary JSON", id="pv-download-summary-btn", className="action-btn")]), html.Div(id="pv-export-status", className="preprocessing-export-status")]),
    ])


@callback(Output("pv-artifact-status", "children"), Output("pv-artifact-system-info", "children"), Input("pv-artifact-select", "value"))
def render_artifact_status(artifact_id: str | None):
    if not artifact_id:
        return pill("No exported artifact available", "amber"), html.P("Select a Data Analysis artifact to view system parameters.", className="section-subtitle")
    artifacts = list_forecast_artifacts()
    artifact = next((item for item in artifacts if item.artifact_id == artifact_id), None)
    if artifact is None:
        return pill("Artifact unavailable", "red"), html.P("The selected artifact is unavailable.", className="section-subtitle")
    meta = artifact.metadata
    return (
        html.Div([pill("Artifact ready", "green"), html.Span(f" {meta.get('quality_summary', {}).get('rows_after', '?'):,} rows | config {meta.get('config_hash', '-')}")], className="preprocessing-summary"),
        _artifact_system_parameters(meta),
    )


@callback(
    Output("pv-settings-xgboost", "style"),
    Output("pv-settings-extra-trees", "style"),
    Output("pv-settings-random-forest", "style"),
    Output("pv-settings-lightgbm", "style"),
    Output("pv-settings-catboost", "style"),
    Output("pv-settings-lstm", "style"),
    Input("pv-model-select", "value"),
)
def toggle_model_settings(models: list[str] | None):
    selected = set(models or [])
    return tuple({"display": "block" if name in selected else "none"} for name in ("xgboost", "extra_trees", "random_forest", "lightgbm", "catboost", "lstm"))


@callback(
    Output("pv-model-settings", "data"),
    Input("pv-xgb-n-estimators", "value"),
    Input("pv-xgb-max-depth", "value"),
    Input("pv-xgb-learning-rate", "value"),
    Input("pv-xgb-subsample", "value"),
    Input("pv-xgb-colsample", "value"),
    Input("pv-et-n-estimators", "value"),
    Input("pv-et-max-depth", "value"),
    Input("pv-et-min-leaf", "value"),
    Input("pv-et-max-features", "value"),
    Input("pv-rf-n-estimators", "value"),
    Input("pv-rf-max-depth", "value"),
    Input("pv-rf-min-leaf", "value"),
    Input("pv-rf-max-features", "value"),
    Input("pv-lgb-n-estimators", "value"),
    Input("pv-lgb-learning-rate", "value"),
    Input("pv-lgb-num-leaves", "value"),
    Input("pv-lgb-max-depth", "value"),
    Input("pv-cat-iterations", "value"),
    Input("pv-cat-depth", "value"),
    Input("pv-cat-learning-rate", "value"),
    Input("pv-cat-l2", "value"),
    Input("pv-lstm-sequence-length", "value"),
    Input("pv-lstm-epochs", "value"),
    Input("pv-lstm-batch-size", "value"),
)
def collect_model_settings(*values):
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
        "xgboost": {"n_estimators": values["xgb_n_estimators"], "max_depth": values["xgb_max_depth"], "learning_rate": values["xgb_learning_rate"], "subsample": values["xgb_subsample"], "colsample_bytree": values["xgb_colsample"]},
        "extra_trees": {"n_estimators": values["et_n_estimators"], "max_depth": values["et_max_depth"], "min_samples_leaf": values["et_min_leaf"], "max_features": values["et_max_features"]},
        "random_forest": {"n_estimators": values["rf_n_estimators"], "max_depth": values["rf_max_depth"], "min_samples_leaf": values["rf_min_leaf"], "max_features": values["rf_max_features"]},
        "lightgbm": {"n_estimators": values["lgb_n_estimators"], "learning_rate": values["lgb_learning_rate"], "num_leaves": values["lgb_num_leaves"], "max_depth": values["lgb_max_depth"]},
        "catboost": {"iterations": values["cat_iterations"], "depth": values["cat_depth"], "learning_rate": values["cat_learning_rate"], "l2_leaf_reg": values["cat_l2"]},
        "lstm": {"sequence_length": values["lstm_sequence_length"], "epochs": values["lstm_epochs"], "batch_size": values["lstm_batch_size"]},
    }


@callback(Output("pv-predict-forecast", "disabled"), Output("pv-generate-forecast-curve", "disabled"), Input("pv-trained-model", "data"))
def toggle_prediction_button(trained_model):
    model_token = (trained_model or {}).get("model_token")
    disabled = not isinstance(model_token, str) or not model_token.strip()
    return disabled, disabled


@callback(Output("pv-train-model", "disabled", allow_duplicate=True), Output("pv-train-model", "className", allow_duplicate=True), Input("pv-train-model", "n_clicks"), prevent_initial_call=True)
def start_training_state(_clicks):
    return True, "action-btn action-btn-primary pv-training-active"


@callback(Output("pv-train-model", "disabled", allow_duplicate=True), Output("pv-train-model", "className", allow_duplicate=True), Input("pv-run-status", "children"), prevent_initial_call=True)
def finish_training_state(_status):
    return False, "action-btn action-btn-primary"


@callback(Output("pv-trained-model", "data"), Output("pv-save-model-btn", "disabled"), Output("pv-upload-model-status", "children"), Output("pv-forecast-result", "data"), Output("pv-run-status", "children"), Output("pv-run-summary", "children"), Output("pv-kpi-grid", "children"), Output("pv-forecast-graph", "figure"), Output("pv-diagnostic-residual-graph", "figure"), Output("pv-diagnostic-r2-graph", "figure"), Output("pv-diagnostic-mae-graph", "figure"), Output("pv-diagnostic-rmse-graph", "figure"), Output("pv-metrics-table", "children"), Input("pv-train-model", "n_clicks"), Input("pv-trained-model-upload", "contents"), Input("pv-predict-forecast", "n_clicks"), Input("pv-generate-forecast-curve", "n_clicks"), State("pv-trained-model-upload", "filename"), State("pv-trained-model", "data"), State("pv-artifact-select", "value"), State("pv-horizon-select", "value"), State("pv-model-select", "value"), State("pv-model-settings", "data"), State("pv-predict-date", "date"), State("pv-predict-time", "value"), State("pv-curve-start-date", "date"), State("pv-curve-start-time", "value"), State("pv-curve-end-date", "date"), State("pv-curve-end-time", "value"), State("pv-weather-select", "value"), State("pv-system-id", "data"), prevent_initial_call=True)
def run_pv_forecast(_train_clicks, upload_contents, _predict_clicks, _curve_clicks, upload_filename, trained_model, artifact_id, horizons, models, model_settings, predict_date, predict_time, curve_start_date, curve_start_time, curve_end_date, curve_end_time, weather, system_id):
    empty = _empty_figure("Train or upload a model, then predict a timestamp.")
    empty_small = _empty_figure("Diagnostics are available after a prediction.", 360)
    triggered = ctx.triggered_id
    if triggered == "pv-trained-model-upload" and upload_contents:
        import base64
        try:
            encoded = upload_contents.split(",", 1)[1]
            info = load_trained_model(base64.b64decode(encoded))
            return info, False, html.P(f"Loaded {upload_filename or 'trained model'}.", className="preprocessing-export-success"), no_update, html.P("Trained model uploaded and ready for prediction.", className="preprocessing-export-success"), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        except Exception as exc:
            return no_update, True, html.P(str(exc), className="preprocessing-export-error"), no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if not artifact_id:
        message = html.P("Export a Data Analysis preprocessing artifact before forecasting.", className="preprocessing-export-error")
        return no_update, True, no_update, no_update, message, no_update, _summary_cards(None), empty, empty_small, empty_small, empty_small, empty_small, _metric_table({})
    if not models:
        message = html.P("Select at least one forecast model.", className="preprocessing-export-error")
        return no_update, True, no_update, no_update, message, no_update, _summary_cards(None), empty, empty_small, empty_small, empty_small, empty_small, _metric_table({})
    request = {"artifact_id": artifact_id, "horizons": horizons, "models": models, "model_settings": model_settings, "system_id": system_id}
    try:
        if triggered == "pv-train-model":
            info = train_model(request)
            summary = info_rows([("Artifact", info["artifact_id"]), ("Models", ", ".join(MODEL_SPECS.get(name, {}).get("label", name) for name in info["models"])), ("Horizons", ", ".join(f"+{item} min" for item in info["horizons"])), ("Trained", info["trained_at"])])
            return info, False, no_update, no_update, html.P("Model training completed. Save it or predict a forecast.", className="preprocessing-export-success"), summary, _summary_cards(None), empty, empty_small, empty_small, empty_small, empty_small, _metric_table({})
        if triggered not in {"pv-predict-forecast", "pv-generate-forecast-curve"}:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        token = (trained_model or {}).get("model_token", "")
        if not isinstance(token, str) or not token.strip():
            message = html.P("Train or upload a model before predicting.", className="preprocessing-export-error")
            return no_update, no_update, no_update, no_update, message, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        if triggered == "pv-generate-forecast-curve":
            start = f"{curve_start_date} {curve_start_time or '00:00'}" if curve_start_date else None
            end = f"{curve_end_date} {curve_end_time or '23:55'}" if curve_end_date else None
            result = predict_trained_model_range(token, {"start": start, "end": end, "horizons": horizons})
        else:
            timestamp = f"{predict_date} {predict_time or '00:00'}" if predict_date else None
            result = predict_trained_model(token, {"timestamp": timestamp, "horizons": horizons})
    except ForecastingError as exc:
        message = html.P(str(exc), className="preprocessing-export-error")
        return no_update, no_update, no_update, no_update, message, no_update, _summary_cards(None), _empty_figure(str(exc)), empty_small, empty_small, empty_small, empty_small, _metric_table({})
    except Exception as exc:
        message = html.P(f"Forecast failed: {exc}", className="preprocessing-export-error")
        return no_update, no_update, no_update, no_update, message, no_update, _summary_cards(None), _empty_figure(str(exc)), empty_small, empty_small, empty_small, empty_small, _metric_table({})
    failed = [f"{MODEL_SPECS.get(name, {}).get('label', name)}: {payload.get('error')}" for name, payload in result.get("results", {}).items() if payload.get("error")]
    summary = info_rows([("Artifact", str(result.get("artifact_id", "-"))), ("Models", ", ".join(MODEL_SPECS.get(name, {}).get("label", name) for name in result.get("models", []))), ("Source period", f"{result.get('source_start', '')[:16]} to {result.get('source_end', '')[:16]}"), ("Generated", str(result.get("generated_at", "-"))[:19])])
    status_text = "Forecast curve generated successfully." if triggered == "pv-generate-forecast-curve" else "Forecast predicted successfully."
    if result.get("skipped_points"):
        status_text += f" {result['skipped_points']} points were outside the available source calendar."
    status = html.Div([html.P(status_text, className="preprocessing-export-success")] + [html.P(error, className="preprocessing-export-error") for error in failed])
    return (
        no_update,
        no_update,
        no_update,
        result,
        status,
        summary,
        _summary_cards(result),
        _forecast_figure(result, None, "ghi" in (weather or [])),
        _residual_figure(result, None),
        _r2_scatter_figure(result),
        _diagnostic_metric_figure(result, "mae", "Mean absolute error by horizon", "MAE (p.u.)"),
        _diagnostic_metric_figure(result, "rmse", "Root mean squared error by horizon", "RMSE (p.u.)"),
        _metric_table(result),
    )


@callback(Output("pv-download-model", "data"), Output("pv-save-model-status", "children"), Input("pv-save-model-btn", "n_clicks"), State("pv-trained-model", "data"), prevent_initial_call=True)
def save_pv_model(_clicks, trained_model):
    token = (trained_model or {}).get("model_token")
    if not token:
        return no_update, html.P("Train or upload a model before saving.", className="preprocessing-export-error")
    models = "-".join((trained_model.get("models") or ["model"]))
    horizons = "-".join(str(value) for value in (trained_model.get("horizons") or []))
    settings = trained_model.get("model_settings") or {}
    setting_parts = []
    for model_name in trained_model.get("models") or []:
        values = settings.get(model_name) or {}
        key_aliases = {
            "n_estimators": "est", "max_depth": "depth", "learning_rate": "lr",
            "sequence_length": "seq", "epochs": "ep", "batch_size": "bs",
            "min_samples_leaf": "leaf", "max_features": "feat", "num_leaves": "leaves",
            "iterations": "iter", "depth": "depth", "l2_leaf_reg": "l2",
        }
        model_settings = [
            f"{key_aliases.get(key, key)}{value}"
            for key, value in values.items()
            if value is not None
        ]
        setting_parts.append(f"{model_name}-{'_'.join(model_settings)}")
    trained_at = str(trained_model.get("trained_at", "saved")).replace("-", "").replace(":", "").replace(" ", "_")
    filename = f"pv_model_{models}_h{horizons}_{'_'.join(setting_parts)}_{trained_at}.joblib"
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.asksaveasfilename(
            title="Save trained PV model",
            initialfile=filename,
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        root.destroy()
        if not selected_path:
            return no_update, html.P("Model save cancelled.", className="preprocessing-export-status")
        Path(selected_path).write_bytes(serialize_trained_model(token))
        return no_update, html.P(f"Model saved to {selected_path}", className="preprocessing-export-success")
    except Exception as exc:
        return no_update, html.P(f"Could not save model: {exc}", className="preprocessing-export-error")


@callback(Output("pv-export-status", "children"), Output("pv-download-forecast", "data"), Output("pv-download-metrics", "data"), Output("pv-download-summary", "data"), Input("pv-download-forecast-btn", "n_clicks"), Input("pv-download-metrics-btn", "n_clicks"), Input("pv-download-summary-btn", "n_clicks"), State("pv-forecast-result", "data"), prevent_initial_call=True)
def export_pv_run(_forecast_clicks, _metrics_clicks, _summary_clicks, result):
    if not result:
        return html.P("Run a forecast before exporting artifacts.", className="preprocessing-export-error"), no_update, no_update, no_update
    try:
        paths = export_forecast_result(result)
        model = result.get("model", "xgboost")
        if model == "comparison":
            model = "xgboost"
        payload = result.get("results", {}).get(model, {})
        forecast_df = pd.DataFrame(payload.get("records") or [])
        metrics_df = pd.DataFrame(build_forecast_metrics(result))
        summary_json = json.dumps(result, indent=2, default=str)
        forecast_download = dcc.send_data_frame(forecast_df.to_csv, f"pv_forecast_{model}.csv", index=False) if ctx.triggered_id == "pv-download-forecast-btn" else no_update
        metrics_download = dcc.send_data_frame(metrics_df.to_csv, "pv_forecast_metrics.csv", index=False) if ctx.triggered_id == "pv-download-metrics-btn" else no_update
        summary_download = dcc.send_string(summary_json, "pv_forecast_run_summary.json") if ctx.triggered_id == "pv-download-summary-btn" else no_update
        return html.P(f"Saved reproducible artifacts: {paths['summary_json']}", className="preprocessing-export-success"), forecast_download, metrics_download, summary_download
    except Exception as exc:
        return html.P(f"Export failed: {exc}", className="preprocessing-export-error"), no_update, no_update, no_update

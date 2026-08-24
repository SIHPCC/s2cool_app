from __future__ import annotations

import base64
import json
from functools import lru_cache

import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.ui import simple_table
from models.domain import SystemRecord
from services.dataset_service import (
    PV_DATA_DIR,
    build_system_dataset_profile,
    build_system_file_options,
    parse_system_dataset_filename,
)
from services.new_system_service import NewSystemError, create_new_system_dataset
from services.preprocessing_service import (
    DEFAULT_CONFIG,
    DEFAULT_LAG_COLUMNS,
    export_preprocessing_artifacts,
    get_pipeline_result,
)


ADD_SYSTEM_MESSAGE = (
    "Registering a new system is the intake path for fresh sites. "
    "Upload the weather dataset in the M2 data folder with this naming format: "
    "systemXX_capacitykW_City_latXX_lonYY_YYYYMMDD_YYYYMMDD_weather.csv"
)

VARIABLE_LABELS = {
    "pv_power_actual_kw": "PV Power (Actual)",
    "power_average_w_normalized": "PV Power (Normalized)",
    "clear_sky_pv_power_actual_kw": "Clear-sky PV Power (Actual)",
    "clear_sky_pv_power_normalized": "Clear-sky PV Power (Normalized)",
    "ghi_pyr": "GHI Pyr",
    "dni": "DNI",
    "dhi": "DHI",
    "air_temperature": "Air Temperature",
    "relative_humidity": "Relative Humidity",
    "wind_speed": "Wind Speed",
}

VARIABLE_UNITS = {
    "pv_power_actual_kw": "kW",
    "power_average_w_normalized": "p.u.",
    "clear_sky_pv_power_actual_kw": "kW",
    "clear_sky_pv_power_normalized": "p.u.",
    "ghi_pyr": "W/m2",
    "dni": "W/m2",
    "dhi": "W/m2",
    "air_temperature": "deg C",
    "relative_humidity": "%",
    "wind_speed": "m/s",
}


def _analysis_time_options() -> list[dict[str, str]]:
    return [
        {"label": f"{hour:02d}:{minute:02d}", "value": f"{hour:02d}:{minute:02d}"}
        for hour in range(24)
        for minute in range(0, 60, 5)
    ]

SELECTION_STRATEGY_OPTIONS = [
    {"label": "Use all candidate features", "value": "all"},
    {"label": "Keep top K by correlation", "value": "top_k_corr"},
    {"label": "Keep features above correlation threshold", "value": "corr_threshold"},
    {"label": "Manually choose features", "value": "manual"},
]

SELECTION_STRATEGY_LABELS = {
    "all": "All candidate features",
    "top_k_corr": "Top K by correlation",
    "corr_threshold": "Correlation threshold",
    "manual": "Manual selection",
}

PREFERRED_VARIABLE_ORDER = [
    "pv_power_actual_kw",
    "power_average_w_normalized",
    "clear_sky_pv_power_actual_kw",
    "clear_sky_pv_power_normalized",
    "dni",
    "ghi_pyr",
    "dhi",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
]


def _format_interval(interval_minutes: float | None) -> str:
    if interval_minutes is None:
        return "Unavailable"
    return f"{interval_minutes:.2f} minutes"


def _format_datetime_local(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None or pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M")


def _parse_analysis_timestamps(df: pd.DataFrame) -> pd.Series:
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


def _variable_display_name(column: str) -> str:
    label = VARIABLE_LABELS.get(column, column.replace("_", " ").title())
    unit = VARIABLE_UNITS.get(column)
    if unit:
        return f"{label} [{unit}]"
    return label


def _add_derived_power_column(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if "power_average_w_normalized" not in df.columns:
        return df

    parsed = parse_system_dataset_filename(PV_DATA_DIR / file_name)
    if not parsed:
        return df

    capacity_kw = parsed.get("capacity_kw")
    if capacity_kw is None:
        return df

    derived = df.copy()
    derived["pv_power_actual_kw"] = pd.to_numeric(derived["power_average_w_normalized"], errors="coerce") * float(capacity_kw)
    return derived


@lru_cache(maxsize=32)
def _load_analysis_dataset(file_name: str) -> pd.DataFrame | None:
    path = PV_DATA_DIR / file_name
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["_ts"] = _parse_analysis_timestamps(df)
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    df = _add_derived_power_column(df, file_name)
    return df


def _analysis_variable_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "time", "_ts"}
    numeric_columns = [column for column in df.columns if column not in excluded and pd.api.types.is_numeric_dtype(df[column])]
    return numeric_columns


def _analysis_variable_options(df: pd.DataFrame) -> list[dict]:
    options: list[dict] = []
    for column in _analysis_variable_columns(df):
        options.append({"label": _variable_display_name(column), "value": column})
    return options


def _default_variable_selection(available_columns: list[str]) -> list[str]:
    selected = [column for column in PREFERRED_VARIABLE_ORDER if column in available_columns]
    if selected:
        return selected[:1]
    return available_columns[:1]


def _parse_datetime_input(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _build_empty_figure(message: str) -> dict:
    return {
        "data": [],
        "layout": {
            "template": "plotly_white",
            "font": {"family": "Segoe UI, Tahoma, Arial, sans-serif", "color": "#31456d", "size": 12},
            "height": 420,
            "margin": {"l": 48, "r": 24, "t": 24, "b": 48},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": "#7b88a5"},
                }
            ],
        },
    }


def _build_trend_figure(
    df: pd.DataFrame,
    selected_variables: list[str],
    start_value: str | None,
    end_value: str | None,
    visible_range: tuple[str, str] | None = None,
) -> dict:
    if df.empty or not selected_variables:
        return _build_empty_figure("Select one or more variables to view the trend chart.")

    start_ts = _parse_datetime_input(start_value) or pd.Timestamp(df["_ts"].min())
    end_ts = _parse_datetime_input(end_value) or pd.Timestamp(df["_ts"].max())

    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts

    window = df[(df["_ts"] >= start_ts) & (df["_ts"] <= end_ts)].copy()
    if window.empty:
        return _build_empty_figure("No rows fall inside the selected time window.")

    # Keep browser payloads responsive for long operational datasets while
    # retaining the full-resolution window for summary calculations.
    plot_window = window
    if len(plot_window) > 6000:
        stride = max(int(len(plot_window) / 6000), 1)
        plot_window = plot_window.iloc[::stride].copy()

    traces = []
    timestamps = plot_window["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    palette = ["#1f5fae", "#e07a2d", "#2e8b72", "#7a55a8", "#bd3f5c", "#55758c"]
    for index, column in enumerate(selected_variables):
        if column not in window.columns:
            continue
        unit = VARIABLE_UNITS.get(column, "")
        hover_unit = f" {unit}" if unit else ""
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": _variable_display_name(column),
                "x": timestamps,
                "y": plot_window[column].tolist(),
                "line": {"width": 2, "color": palette[index % len(palette)]},
                "hovertemplate": f"%{{x|%Y-%m-%d %H:%M}}<br>%{{fullData.name}}: %{{y:.3f}}{hover_unit}<extra></extra>",
                "connectgaps": False,
            }
        )

    if not traces:
        return _build_empty_figure("No supported numeric variables are selected.")

    # Plotly does not automatically recompute the Y-axis when only the X-axis
    # is zoomed. Recalculate it from the visible time slice so the scale stays
    # meaningful for the operator's current view.
    visible_window = window
    visible_start = visible_end = None
    if visible_range:
        visible_start = _parse_datetime_input(visible_range[0])
        visible_end = _parse_datetime_input(visible_range[1])
        if visible_start is not None and visible_end is not None:
            if visible_start > visible_end:
                visible_start, visible_end = visible_end, visible_start
            visible_window = window[
                (window["_ts"] >= visible_start) & (window["_ts"] <= visible_end)
            ]

    y_values = []
    for column in selected_variables:
        if column in visible_window.columns:
            y_values.extend(pd.to_numeric(visible_window[column], errors="coerce").dropna().tolist())

    y_range = None
    if y_values:
        y_min, y_max = float(min(y_values)), float(max(y_values))
        padding = max((y_max - y_min) * 0.06, abs(y_max) * 0.01, 1e-9)
        y_range = [y_min - padding, y_max + padding]

    xaxis = {
        "title": "Timestamp",
        "showgrid": False,
        "rangeslider": {"visible": True, "thickness": 0.08, "bordercolor": "#dbe3f0"},
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikedash": "dash",
        "spikecolor": "#8ea1c7",
        "spikethickness": 1,
    }
    if visible_start is not None and visible_end is not None:
        xaxis["range"] = [visible_start.isoformat(), visible_end.isoformat()]
        xaxis["autorange"] = False

    yaxis = {
        "title": "Value (native variable units)",
        "gridcolor": "#edf1f8",
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikedash": "dash",
        "spikecolor": "#8ea1c7",
        "spikethickness": 1,
    }
    if y_range:
        yaxis["range"] = y_range
        yaxis["autorange"] = False

    return {
        "data": traces,
        "layout": {
            "template": "plotly_white",
            "font": {"family": "Segoe UI, Tahoma, Arial, sans-serif", "color": "#31456d", "size": 12},
            "height": 440,
            "margin": {"l": 52, "r": 24, "t": 24, "b": 48},
            "hovermode": "closest",
            "hoverdistance": -1,
            "spikedistance": -1,
            "uirevision": "data-analysis-trend",
            "legend": {"orientation": "h", "y": 1.04, "x": 0},
            "xaxis": xaxis,
            "yaxis": yaxis,
        },
    }


def _build_profile_layout(profile: dict) -> html.Div:
    lat = profile["lat"]
    lon = profile["lon"]
    system_label = f"System {profile['system_id']:02d}"
    city_label = profile["city"]

    map_srcdoc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      background: #0f172a;
      overflow: hidden;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    .leaflet-container {{
      font-family: 'Segoe UI', Tahoma, sans-serif;
      background: #0f172a;
    }}
    .leaflet-control-attribution {{
      font-size: 10px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map', {{ zoomControl: true }}).setView([{lat:.6f}, {lon:.6f}], 15);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19,
      attribution: 'Tiles &copy; Esri'
    }}).addTo(map);

    const marker = L.marker([{lat:.6f}, {lon:.6f}]).addTo(map);
    marker.bindPopup({json.dumps(system_label)} + '<br>' + {json.dumps(city_label)} + '<br>{lat:.6f}, {lon:.6f}').openPopup();
  </script>
</body>
</html>
"""

    return html.Div(
        className="analysis-profile-grid",
        children=[
            html.Div(
                className="panel nested-panel profile-metrics-panel",
                children=[
                    html.H3("System Parameters", className="section-title"),
                    simple_table(
                        ["Parameter", "Value"],
                        [
                            ["Data Points", f"{profile['data_points']:,}"],
                            ["Timestamp Interval", _format_interval(profile["time_interval_min"])],
                            ["Start Time", profile["start_time"]],
                            ["End Time", profile["end_time"]],
                            ["System ID", system_label],
                            ["City", city_label],
                            ["Capacity", f"{profile['capacity_kw']:.3g} kW"],
                            ["Location", f"Lat {lat:.6f}, Lon {lon:.6f}"],
                        ],
                    ),
                ],
            ),
            html.Div(
                className="panel nested-panel profile-map-panel",
                children=[
                    html.H3("System Location", className="section-title"),
                    html.P(
                        f"Satellite view centered at {lat:.6f}, {lon:.6f} for {city_label}.",
                        className="section-subtitle",
                    ),
                    html.Div(
                        className="location-map-shell",
                        children=[
                            html.Div(
                                className="location-map-meta",
                                children=[
                                    html.Span("Latitude", className="muted"),
                                    html.Strong(f"{lat:.6f}"),
                                    html.Span("Longitude", className="muted"),
                                    html.Strong(f"{lon:.6f}"),
                                ],
                            ),
                            html.Iframe(
                                srcDoc=map_srcdoc,
                                className="location-map-frame",
                                title=f"{system_label} location map",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_trend_panel() -> html.Div:
    return html.Div(
        id="data-analysis-trend-panel",
        className="panel trend-panel",
        children=[
            html.H2("Data Visualization", className="section-title"),
            html.P(
                "Choose one or more processed variables, narrow the time window, and inspect operational trends with hover tooltips.",
                className="section-subtitle",
            ),
            html.Div(
                className="trend-toolbar",
                children=[
                    html.Div(
                        className="trend-card trend-card-variables",
                        children=[
                            html.Div("Available Variables", className="trend-card-title"),
                            html.P(
                                "Select the numeric signals you want to compare on the same chart.",
                                className="trend-card-note",
                            ),
                            dcc.Checklist(
                                id="data-analysis-variable-select",
                                options=[],
                                value=[],
                                className="trend-variable-checklist",
                                inputClassName="trend-variable-input",
                                labelClassName="trend-variable-label",
                            ),
                            html.Button(
                                "Generate Plot",
                                id="data-analysis-generate-plot-btn",
                                className="action-btn action-btn-primary trend-generate-btn",
                            ),
                        ],
                    ),
                    html.Div(
                        className="trend-card trend-card-window",
                        children=[
                            html.Div("Time Window", className="trend-card-title"),
                            html.P(
                                "Choose a calendar date and time for the plot window.",
                                className="trend-card-note",
                            ),
                            html.Div(
                                className="trend-time-grid",
                                children=[
                                    html.Div(
                                        className="trend-time-field",
                                        children=[
                                            html.Span("Start Date and Time", className="muted"),
                                            html.Div(
                                                className="trend-date-time-row",
                                                children=[
                                                    dcc.DatePickerSingle(
                                                        id="data-analysis-trend-start-date",
                                                        display_format="YYYY-MM-DD",
                                                        clearable=False,
                                                        className="trend-date-picker",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="data-analysis-trend-start-time",
                                                        options=_analysis_time_options(),
                                                        value="00:00",
                                                        clearable=False,
                                                        className="trend-time-select",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="trend-time-field",
                                        children=[
                                            html.Span("End Date and Time", className="muted"),
                                            html.Div(
                                                className="trend-date-time-row",
                                                children=[
                                                    dcc.DatePickerSingle(
                                                        id="data-analysis-trend-end-date",
                                                        display_format="YYYY-MM-DD",
                                                        clearable=False,
                                                        className="trend-date-picker",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="data-analysis-trend-end-time",
                                                        options=_analysis_time_options(),
                                                        value="23:55",
                                                        clearable=False,
                                                        className="trend-time-select",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="data-analysis-trend-summary", className="trend-summary"),
            dcc.Graph(
                id="data-analysis-trend-graph",
                className="trend-graph",
                figure=_build_empty_figure("Generate a dataset, choose variables, and click Generate Plot."),
                config={"displayModeBar": False, "responsive": True},
            ),
        ],
        style={"display": "block"},
    )


def _build_preprocessing_panel() -> html.Div:
    """Build the Data Preprocessing and Feature Engineering panel."""
    return html.Div(
        id="data-analysis-preprocess-panel",
        className="panel preprocessing-panel",
        children=[
            html.H2("Data Preprocessing and Feature Engineering", className="section-title"),
            html.P(
                "Step through the preprocessing pipeline one tab at a time so each stage stays focused and easier to review.",
                className="section-subtitle",
            ),
            dcc.Tabs(
                id="data-analysis-preprocess-tabs",
                className="preprocess-tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="Overview",
                        value="overview",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div("Pipeline Overview", className="trend-card-title"),
                                html.P(
                                    "Use the tabs below to move through the preprocessing workflow in order. Each tab exposes one stage, keeps the configuration visible, and preserves the output summaries for the active step.",
                                    className="trend-card-note",
                                ),
                                html.Div(
                                    className="preprocessing-summary",
                                    children=[
                                        html.Span("1. Overview", className="preprocessing-summary-item"),
                                        html.Span("2. Data Cleaning", className="preprocessing-summary-item"),
                                        html.Span("3. Add Time Features", className="preprocessing-summary-item"),
                                        html.Span("4. Add Solar Features", className="preprocessing-summary-item"),
                                        html.Span("5. Add Weather Features", className="preprocessing-summary-item"),
                                        html.Span("6. Feature Selection", className="preprocessing-summary-item"),
                                        html.Span("7. Dataset Builder", className="preprocessing-summary-item"),
                                        html.Span("8. Dataset Quality", className="preprocessing-summary-item"),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Data Cleaning",
                        value="cleaning",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Data Cleaning", className="trend-card-title"),
                                        html.P(
                                            "Missing values, duplicates, night rows and outliers.",
                                            className="trend-card-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Missing Strategy", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-missing-strategy",
                                                    options=[
                                                        {"label": "Interpolate (linear)", "value": "interpolate"},
                                                        {"label": "Forward Fill", "value": "ffill"},
                                                        {"label": "Fill Zero", "value": "zero"},
                                                        {"label": "Fill Median", "value": "median"},
                                                        {"label": "Drop Rows", "value": "drop"},
                                                    ],
                                                    value="interpolate",
                                                    clearable=False,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Outlier Method", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-outlier-method",
                                                    options=[
                                                        {"label": "None", "value": "noop"},
                                                        {"label": "Z-Score", "value": "zscore"},
                                                        {"label": "IQR", "value": "iqr"},
                                                    ],
                                                    value="noop",
                                                    clearable=False,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Outlier Threshold", className="muted"),
                                                dcc.Input(
                                                    id="data-analysis-preprocess-outlier-threshold",
                                                    type="number",
                                                    min=0.1,
                                                    step=0.1,
                                                    value=3.0,
                                                    className="trend-datetime-input",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Div(
                                            className="preprocessing-check-row",
                                            children=[
                                                dcc.Checklist(
                                                    id="data-analysis-preprocess-remove-duplicates",
                                                    options=[{"label": "Remove duplicate rows", "value": "dupes"}],
                                                    value=["dupes"],
                                                    className="preprocessing-checklist",
                                                    inputClassName="trend-variable-input",
                                                    labelClassName="trend-variable-label",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="preprocessing-check-row",
                                            children=[
                                                dcc.Checklist(
                                                    id="data-analysis-preprocess-filter-night",
                                                    options=[{"label": "Filter night rows (GHI < threshold)", "value": "night"}],
                                                    value=["night"],
                                                    className="preprocessing-checklist",
                                                    inputClassName="trend-variable-input",
                                                    labelClassName="trend-variable-label",
                                                ),
                                            ],
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Night GHI Threshold (W/m2)", className="muted"),
                                                dcc.Input(
                                                    id="data-analysis-preprocess-night-threshold",
                                                    type="number",
                                                    min=0,
                                                    step=1,
                                                    value=10,
                                                    className="trend-datetime-input",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Div(id="data-analysis-preprocess-cleaning-summary", className="preprocessing-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("cleaning"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Add Time Features",
                        value="time",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Time Features", className="trend-card-title"),
                                        html.P(
                                            "Calendar/cyclic encodings plus lag, rolling and difference blocks.",
                                            className="trend-card-note",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-time-cyclic",
                                            options=[{"label": "Calendar / cyclic encodings", "value": "cyclic"}],
                                            value=["cyclic"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        html.P(
                                            "Adds repeating hour, day, and seasonal patterns so the model can learn time-of-day and calendar effects.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Lag Shifts", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-lag-shifts",
                                                    options=[{"label": f"Lag {shift}", "value": shift} for shift in range(1, 13)],
                                                    value=[],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Uses previous time steps of key signals as inputs, which helps when current output depends on recent history.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Rolling Windows", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-rolling-windows",
                                                    options=[{"label": f"{window} step window", "value": window} for window in [3, 6, 12, 24]],
                                                    value=[],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Builds short-window statistics such as moving mean and standard deviation to capture local trend and variability.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Difference Periods", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-diff-periods",
                                                    options=[{"label": f"Diff {period}", "value": period} for period in range(1, 13)],
                                                    value=[],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Measures change from earlier steps, which is useful for detecting ramps, drift, and short-term momentum.",
                                            className="preprocessing-note",
                                        ),
                                        html.Div(id="data-analysis-preprocess-time-summary", className="preprocessing-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("time"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Add Solar Features",
                        value="solar",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Solar Features", className="trend-card-title"),
                                        html.P(
                                            "Solar geometry, clear-sky GHI and PV power curves, clearness index, irradiance ratios and performance ratio.",
                                            className="trend-card-note",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-solar-position",
                                            options=[{"label": "Solar position (elevation/zenith/azimuth)", "value": "position"}],
                                            value=["position"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-clear-sky",
                                            options=[{"label": "Clear-sky GHI and PV power curves (actual + normalized)", "value": "clear_sky"}],
                                            value=["clear_sky"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-clearness",
                                            options=[{"label": "Clearness index (GHI / clear-sky)", "value": "clearness"}],
                                            value=["clearness"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-irradiance-ratios",
                                            options=[{"label": "Irradiance ratios (DNI/GHI, DHI/GHI)", "value": "ratios"}],
                                            value=["ratios"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-performance-ratio",
                                            options=[{"label": "Performance ratio (normalized power / GHI-kW)", "value": "pr"}],
                                            value=["pr"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        html.Div(id="data-analysis-preprocess-solar-summary", className="preprocessing-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("solar"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Add Weather Features",
                        value="weather",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Weather Features", className="trend-card-title"),
                                        html.P(
                                            "Interactions, exponential moving averages and ramp rates.",
                                            className="trend-card-note",
                                        ),
                                        dcc.Checklist(
                                            id="data-analysis-preprocess-weather-interactions",
                                            options=[{"label": "Weather interactions (T x GHI, RH x GHI)", "value": "interactions"}],
                                            value=["interactions"],
                                            className="preprocessing-checklist",
                                            inputClassName="trend-variable-input",
                                            labelClassName="trend-variable-label",
                                        ),
                                        html.P(
                                            "Combines weather variables so the model can learn joint effects, such as hot-and-sunny or humid-and-cloudy conditions.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("EMA Spans", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-ema-spans",
                                                    options=[{"label": f"EMA {span}", "value": span} for span in [3, 6, 12, 24]],
                                                    value=[],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Creates exponentially weighted averages that emphasize the most recent weather history while still keeping past context.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Ramp Rate Periods", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-ramp-periods",
                                                    options=[{"label": f"Ramp {period}", "value": period} for period in range(1, 13)],
                                                    value=[],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Calculates how quickly irradiance or power-related signals change across steps, which helps capture sudden transitions.",
                                            className="preprocessing-note",
                                        ),
                                        html.Div(id="data-analysis-preprocess-weather-summary", className="preprocessing-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("weather"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Feature Selection",
                        value="selection",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Feature Selection", className="trend-card-title"),
                                        html.P(
                                            "Correlation against the target and optional mutual-information ranking.",
                                            className="trend-card-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Target Column", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-target-column",
                                                    options=[
                                                        {"label": "PV Power (Normalized)", "value": "power_average_w_normalized"},
                                                        {"label": "PV Power (Actual kW)", "value": "pv_power_actual_kw"},
                                                    ],
                                                    value="power_average_w_normalized",
                                                    clearable=False,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Selection evaluates the original signals plus any engineered time, solar, and weather features added in earlier tabs.",
                                            className="preprocessing-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Selection Strategy", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-selection-strategy",
                                                    options=SELECTION_STRATEGY_OPTIONS,
                                                    value="all",
                                                    clearable=False,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.P(
                                            "Choose whether the dataset builder should keep every candidate feature, filter by correlation, or follow a manual list.",
                                            className="preprocessing-note",
                                        ),
                                        html.Div(
                                            id="data-analysis-preprocess-selection-top-k-wrap",
                                            children=[
                                                html.Label(
                                                    [
                                                        html.Span("Top K Features", className="muted"),
                                                        dcc.Input(
                                                            id="data-analysis-preprocess-selection-top-k",
                                                            type="number",
                                                            min=1,
                                                            step=1,
                                                            value=10,
                                                            className="trend-datetime-input",
                                                        ),
                                                    ],
                                                    className="preprocessing-field",
                                                ),
                                                html.P(
                                                    "Used when the strategy is Top K by correlation. The highest absolute-correlation features are kept.",
                                                    className="preprocessing-note",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="data-analysis-preprocess-selection-min-corr-wrap",
                                            children=[
                                                html.Label(
                                                    [
                                                        html.Span("Minimum Absolute Correlation", className="muted"),
                                                        dcc.Input(
                                                            id="data-analysis-preprocess-selection-min-corr",
                                                            type="number",
                                                            min=0,
                                                            max=1,
                                                            step=0.01,
                                                            value=0.2,
                                                            className="trend-datetime-input",
                                                        ),
                                                    ],
                                                    className="preprocessing-field",
                                                ),
                                                html.P(
                                                    "Used when the strategy is correlation threshold. Only features with |correlation| at or above this value are kept.",
                                                    className="preprocessing-note",
                                                ),
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            id="data-analysis-preprocess-selection-manual-wrap",
                                            children=[
                                                html.Label(
                                                    [
                                                        html.Span("Manual Feature List", className="muted"),
                                                        dcc.Dropdown(
                                                            id="data-analysis-preprocess-selection-manual-features",
                                                            options=[],
                                                            value=[],
                                                            multi=True,
                                                            className="preprocessing-select",
                                                            placeholder="Pick specific engineered or source features",
                                                        ),
                                                    ],
                                                    className="preprocessing-field",
                                                ),
                                                html.P(
                                                    "Used when the strategy is manual. The list updates from the current candidate features after your earlier preprocessing choices.",
                                                    className="preprocessing-note",
                                                ),
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div(id="data-analysis-preprocess-selection-summary", className="preprocessing-selection-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("selection"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Dataset Builder",
                        value="builder",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card",
                                    children=[
                                        html.Div("Dataset Builder", className="trend-card-title"),
                                        html.P(
                                            "Forecast-horizon target creation, chronological split and feature scaling.",
                                            className="trend-card-note",
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Forecast Horizon Steps", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-horizon-steps",
                                                    options=[{"label": f"{step} step{'s' if step > 1 else ''} ahead", "value": step} for step in [1, 3, 6, 12, 24]],
                                                    value=[1, 6, 12],
                                                    multi=True,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Div(
                                            className="preprocessing-split-grid",
                                            children=[
                                                html.Label(
                                                    [
                                                        html.Span("Train Fraction", className="muted"),
                                                        dcc.Input(
                                                            id="data-analysis-preprocess-train-fraction",
                                                            type="number",
                                                            min=0.1,
                                                            max=0.95,
                                                            step=0.05,
                                                            value=0.7,
                                                            className="trend-datetime-input",
                                                        ),
                                                    ],
                                                    className="preprocessing-field",
                                                ),
                                                html.Label(
                                                    [
                                                        html.Span("Validation Fraction", className="muted"),
                                                        dcc.Input(
                                                            id="data-analysis-preprocess-val-fraction",
                                                            type="number",
                                                            min=0,
                                                            max=0.45,
                                                            step=0.05,
                                                            value=0.15,
                                                            className="trend-datetime-input",
                                                        ),
                                                    ],
                                                    className="preprocessing-field",
                                                ),
                                            ],
                                        ),
                                        html.Label(
                                            [
                                                html.Span("Scaling Method", className="muted"),
                                                dcc.Dropdown(
                                                    id="data-analysis-preprocess-scaling-method",
                                                    options=[
                                                        {"label": "Min-Max Scaling", "value": "minmax"},
                                                        {"label": "Standard Scaling", "value": "standard"},
                                                        {"label": "None", "value": "none"},
                                                    ],
                                                    value="minmax",
                                                    clearable=False,
                                                    className="preprocessing-select",
                                                ),
                                            ],
                                            className="preprocessing-field",
                                        ),
                                        html.Div(id="data-analysis-preprocess-builder-summary", className="preprocessing-summary"),
                                    ],
                                ),
                                _preprocess_tab_actions("builder"),
                            ],
                        ),
                    ),
                    dcc.Tab(
                        label="Dataset Quality",
                        value="quality",
                        children=html.Div(
                            className="preprocessing-step-panel",
                            children=[
                                html.Div(
                                    className="trend-card preprocessing-card preprocessing-report-card",
                                    children=[
                                        html.Div("Data Quality Report", className="trend-card-title"),
                                        html.P(
                                            "Row and column summaries, timestamp integrity and split sizes.",
                                            className="trend-card-note",
                                        ),
                                        html.Div(id="data-analysis-preprocess-quality-report", className="preprocessing-report"),
                                    ],
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            html.Div(
                className="preprocessing-generate-row",
                children=[
                    html.Button(
                        "Generate Dataset",
                        id="data-analysis-generate-dataset-btn",
                        className="action-btn action-btn-primary",
                    ),
                    html.Span(
                        "Run the configured preprocessing and feature-engineering pipeline before plotting.",
                        className="preprocessing-action-note",
                    ),
                ],
            ),
            html.Div(id="data-analysis-generate-dataset-status", className="preprocessing-export-status"),
        ],
        style={"display": "block"},
    )


def build_layout(system: SystemRecord | None) -> html.Div:
    _ = system
    options = build_system_file_options()

    return html.Div(
        className="data-analysis-page",
        children=[
            dcc.Store(id="data-analysis-mode", data=None),
            dcc.Store(id="data-analysis-preprocess-config", data=_default_preprocess_config()),
            dcc.Store(id="data-analysis-preprocess-state", data=None),
            dcc.Store(id="data-analysis-generated-dataset", data=None),
            html.Div(
                className="panel data-analysis-intro-panel",
                children=[
                    html.H2("System Ingestion", className="section-title"),
                    html.P(
                        "Choose whether to register a new source or analyze an existing system dataset.",
                        className="data-analysis-intro",
                    ),
                    html.Div(
                        className="action-button-row",
                        children=[
                            html.Button("Add New System", id="data-analysis-add-btn", className="action-btn"),
                            html.Button("Analyze Existing System", id="data-analysis-analyze-btn", className="action-btn"),
                        ],
                    ),
                    html.Div(
                        id="data-analysis-existing-controls",
                        className="analysis-controls",
                        children=[
                            html.Label("Available Systems", className="muted"),
                            dcc.Dropdown(
                                id="data-analysis-existing-select",
                                options=options,
                                value=options[0]["value"] if options else None,
                                clearable=False,
                                placeholder="Select an available system",
                                className="analysis-select",
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="data-analysis-add-panel",
                        className="panel nested-panel",
                        children=[
                            html.H3("Add New System", className="section-title"),
                            html.P(ADD_SYSTEM_MESSAGE, className="section-subtitle"),
                            html.Div(
                                className="new-system-form",
                                children=[
                                    html.Div(
                                        className="new-system-form-grid",
                                        children=[
                                            html.Label(
                                                [
                                                    html.Span("System ID", className="muted"),
                                                    dcc.Input(
                                                        id="new-system-id",
                                                        type="number",
                                                        min=1,
                                                        step=1,
                                                        className="new-system-input",
                                                        placeholder="e.g. 14",
                                                    ),
                                                ],
                                                className="new-system-field",
                                            ),
                                            html.Label(
                                                [
                                                    html.Span("Capacity (kW)", className="muted"),
                                                    dcc.Input(
                                                        id="new-system-capacity",
                                                        type="number",
                                                        min=0.1,
                                                        step=0.1,
                                                        className="new-system-input",
                                                        placeholder="e.g. 5.0",
                                                    ),
                                                ],
                                                className="new-system-field",
                                            ),
                                            html.Label(
                                                [
                                                    html.Span("City", className="muted"),
                                                    dcc.Input(
                                                        id="new-system-city",
                                                        type="text",
                                                        className="new-system-input",
                                                        placeholder="e.g. Lahore",
                                                    ),
                                                ],
                                                className="new-system-field",
                                            ),
                                            html.Label(
                                                [
                                                    html.Span("Latitude", className="muted"),
                                                    dcc.Input(
                                                        id="new-system-lat",
                                                        type="number",
                                                        step="any",
                                                        className="new-system-input",
                                                        placeholder="e.g. 31.5204",
                                                    ),
                                                ],
                                                className="new-system-field",
                                            ),
                                            html.Label(
                                                [
                                                    html.Span("Longitude", className="muted"),
                                                    dcc.Input(
                                                        id="new-system-lon",
                                                        type="number",
                                                        step="any",
                                                        className="new-system-input",
                                                        placeholder="e.g. 74.3587",
                                                    ),
                                                ],
                                                className="new-system-field",
                                            ),
                                        ],
                                    ),
                                    html.Label(
                                        [
                                            html.Span("PV Power Data File (CSV)", className="muted"),
                                            dcc.Upload(
                                                id="new-system-upload",
                                                className="new-system-upload",
                                                children=html.Div(
                                                    [
                                                        "Drag and Drop or ",
                                                        html.A("Select CSV File", className="new-system-upload-link"),
                                                    ],
                                                    className="new-system-upload-inner",
                                                ),
                                                multiple=False,
                                            ),
                                        ],
                                        className="new-system-field new-system-upload-field",
                                    ),
                                    html.Div(
                                        id="new-system-upload-status",
                                        className="new-system-upload-status",
                                    ),
                                    html.Div(
                                        className="new-system-actions",
                                        children=[
                                            html.Button(
                                                "Create New System Dataset",
                                                id="new-system-create-btn",
                                                className="action-btn action-btn-primary",
                                            ),
                                        ],
                                    ),
                                    html.Div(id="new-system-result", className="new-system-result"),
                                ],
                            ),
                        ],
                        style={"display": "none"},
                    ),
                ],
            ),
            html.Div(
                id="data-analysis-system-info",
                className="panel",
                children=[
                    html.H2("System Information", className="section-title"),
                    html.P(
                        "Select Analyze Existing System to load an operational profile.",
                        className="section-subtitle",
                    ),
                ],
            ),
            _build_preprocessing_panel(),
            _build_trend_panel(),
            html.Div(
                className="panel data-analysis-export-box",
                children=[
                    dcc.Download(id="data-analysis-download-dataset"),
                    html.Div("Dataset Export", className="section-title"),
                    html.P(
                        "Prepare the processed dataset for PV forecasting and Cooling demand, or download a copy for local analysis.",
                        className="section-subtitle",
                    ),
                    html.Div(
                        className="data-analysis-export-actions",
                        children=[
                            html.Button(
                                "Export Dataset for Analysis",
                                id="data-analysis-export-analysis-btn",
                                className="action-btn action-btn-primary",
                            ),
                            html.Button(
                                "Download Dataset",
                                id="data-analysis-download-dataset-btn",
                                className="action-btn",
                            ),
                        ],
                    ),
                    html.Div(id="data-analysis-export-status", className="preprocessing-export-status"),
                ],
            ),
        ]
    )


@callback(
    Output("data-analysis-mode", "data"),
    Input("data-analysis-add-btn", "n_clicks"),
    Input("data-analysis-analyze-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_analysis_mode(add_clicks: int | None, analyze_clicks: int | None):
    _ = add_clicks, analyze_clicks
    triggered = ctx.triggered_id
    if triggered == "data-analysis-add-btn":
        return "add"
    if triggered == "data-analysis-analyze-btn":
        return "analyze"
    return None


@callback(
    Output("data-analysis-existing-controls", "style"),
    Output("data-analysis-add-panel", "style"),
    Output("data-analysis-system-info", "children"),
    Input("data-analysis-mode", "data"),
    Input("data-analysis-existing-select", "value"),
)
def render_analysis_state(mode: str | None, selected_file: str | None):
    hidden = {"display": "none"}
    shown = {"display": "block"}

    if mode == "add":
        return (
            hidden,
            shown,
            [
                html.H2("System Information", className="section-title"),
                html.P(
                    "New system intake selected. Complete upload and naming, then switch to Analyze Existing System.",
                    className="section-subtitle",
                ),
            ],
        )

    if mode == "analyze":
        if not selected_file:
            return (
                shown,
                hidden,
                [
                    html.H2("System Information", className="section-title"),
                    html.P("No dataset selected.", className="section-subtitle"),
                ],
            )

        profile = build_system_dataset_profile(selected_file)
        if profile is None:
            return (
                shown,
                hidden,
                [
                    html.H2("System Information", className="section-title"),
                    html.P("Selected dataset is unavailable or filename is invalid.", className="section-subtitle"),
                ],
            )

        return (
            shown,
            hidden,
            [
                html.H2("System Information", className="section-title"),
                _build_profile_layout(profile),
            ],
        )

    return (
        hidden,
        hidden,
        [
            html.H2("System Information", className="section-title"),
            html.P(
                "Choose Add New System or Analyze Existing System to continue.",
                className="section-subtitle",
            ),
        ],
    )


@callback(
    Output("data-analysis-trend-panel", "style"),
    Output("data-analysis-variable-select", "options"),
    Output("data-analysis-variable-select", "value"),
    Output("data-analysis-trend-start-date", "date"),
    Output("data-analysis-trend-start-time", "value"),
    Output("data-analysis-trend-end-date", "date"),
    Output("data-analysis-trend-end-time", "value"),
    Output("data-analysis-trend-summary", "children"),
    Output("data-analysis-trend-graph", "figure"),
    Input("data-analysis-mode", "data"),
    Input("data-analysis-existing-select", "value"),
    Input("data-analysis-generate-dataset-btn", "n_clicks"),
    Input("data-analysis-generate-plot-btn", "n_clicks"),
    State("data-analysis-variable-select", "value"),
    State("data-analysis-trend-start-date", "date"),
    State("data-analysis-trend-start-time", "value"),
    State("data-analysis-trend-end-date", "date"),
    State("data-analysis-trend-end-time", "value"),
    Input("data-analysis-preprocess-config", "data"),
    Input("data-analysis-trend-graph", "relayoutData"),
    Input("system-select", "value"),
    Input("data-analysis-generated-dataset", "data"),
    running=[(Output("data-analysis-generate-plot-btn", "disabled"), True, False)],
)
def render_trend_analysis_state(
    mode: str | None,
    selected_file: str | None,
    _generate_dataset_clicks: int | None,
    _generate_plot_clicks: int | None,
    selected_variables: list[str] | str | None,
    start_date: str | None,
    start_time: str | None,
    end_date: str | None,
    end_time: str | None,
    preprocess_config: dict | None,
    relayout_data: dict | None,
    _system_number: int | None,
    generated_dataset: dict | None,
):
    hidden = {"display": "none"}
    shown = {"display": "block"}
    empty_figure = _build_empty_figure("Generate a dataset, choose variables, and click Generate Plot.")
    triggered = ctx.triggered_id

    if mode != "analyze" or not selected_file:
        return shown, [], [], None, "00:00", None, "23:55", html.P("Select a system and click Generate Dataset to load variables.", className="section-subtitle"), empty_figure

    generated_matches_selection = generated_dataset and generated_dataset.get("file_name") == selected_file
    if triggered == "system-select" or not _generate_dataset_clicks or not generated_matches_selection:
        return shown, [], [], None, "00:00", None, "23:55", html.P("Click Generate Dataset to load variables for plotting.", className="section-subtitle"), empty_figure

    try:
        result = get_pipeline_result(selected_file, preprocess_config)
        df = result.dataframe.copy()
    except Exception:
        df = None
    if df is None or df.empty or "_ts" not in df.columns:
        return shown, [], [], None, "00:00", None, "23:55", html.P("The processed dataset could not be loaded. Review the previous pipeline step for details.", className="section-subtitle"), _build_empty_figure("Processed dataset unavailable.")

    available_columns = _analysis_variable_columns(df)
    options = _analysis_variable_options(df)
    current_selection = selected_variables if isinstance(selected_variables, list) else ([selected_variables] if isinstance(selected_variables, str) else [])
    current_selection = [column for column in current_selection if column in available_columns]
    if not current_selection:
        current_selection = _default_variable_selection(available_columns)

    min_ts = pd.Timestamp(df["_ts"].min())
    max_ts = pd.Timestamp(df["_ts"].max())

    start_value = f"{start_date} {start_time}" if start_date and start_time else None
    end_value = f"{end_date} {end_time}" if end_date and end_time else None
    parsed_start = _parse_datetime_input(start_value)
    parsed_end = _parse_datetime_input(end_value)

    if triggered in {"data-analysis-existing-select", "data-analysis-mode", "data-analysis-generate-dataset-btn"}:
        parsed_start = min_ts
        parsed_end = max_ts
    else:
        if parsed_start is None:
            parsed_start = min_ts
        if parsed_end is None:
            parsed_end = max_ts
        if parsed_start < min_ts:
            parsed_start = min_ts
        if parsed_start > max_ts:
            parsed_start = max_ts
        if parsed_end < min_ts:
            parsed_end = min_ts
        if parsed_end > max_ts:
            parsed_end = max_ts

    if parsed_start > parsed_end:
        parsed_start, parsed_end = parsed_end, parsed_start

    start_value = _format_datetime_local(parsed_start)
    end_value = _format_datetime_local(parsed_end)

    visible_range = None
    if triggered == "data-analysis-trend-graph" and relayout_data:
        range_start = relayout_data.get("xaxis.range[0]")
        range_end = relayout_data.get("xaxis.range[1]")
        range_values = relayout_data.get("xaxis.range")
        if isinstance(range_values, list) and len(range_values) == 2:
            range_start, range_end = range_values
        if range_start and range_end and not relayout_data.get("xaxis.autorange"):
            visible_range = (str(range_start), str(range_end))

    should_plot = triggered in {"data-analysis-generate-plot-btn", "data-analysis-trend-graph"}
    figure = _build_trend_figure(df, current_selection, start_value, end_value, visible_range) if should_plot else _build_empty_figure("Choose variables and click Generate Plot.")
    start_ts = parsed_start
    end_ts = parsed_end

    summary = html.Div(
        children=[
            html.Span(f"{len(available_columns)} variables available", className="trend-summary-item"),
            html.Span(f"{len(current_selection)} selected", className="trend-summary-item"),
            html.Span(f"Window: {start_ts:%Y-%m-%d %H:%M} to {end_ts:%Y-%m-%d %H:%M}", className="trend-summary-item"),
        ]
    )

    return (
        shown,
        options,
        current_selection,
        parsed_start.strftime("%Y-%m-%d"),
        parsed_start.strftime("%H:%M"),
        parsed_end.strftime("%Y-%m-%d"),
        parsed_end.strftime("%H:%M"),
        summary,
        figure,
    )


def _assemble_preprocess_config(
    missing_strategy: str | None,
    outlier_method: str | None,
    outlier_threshold: float | None,
    remove_duplicates: list[str] | None,
    filter_night: list[str] | None,
    night_threshold: float | None,
    time_cyclic: list[str] | None,
    lag_shifts: list[int] | None,
    rolling_windows: list[int] | None,
    diff_periods: list[int] | None,
    solar_position: list[str] | None,
    clear_sky: list[str] | None,
    clearness: list[str] | None,
    irradiance_ratios: list[str] | None,
    performance_ratio: list[str] | None,
    weather_interactions: list[str] | None,
    ema_spans: list[int] | None,
    ramp_periods: list[int] | None,
    target_column: str | None,
    selection_strategy: str | None,
    selection_top_k: float | None,
    selection_min_corr: float | None,
    manual_features: list[str] | None,
    horizon_steps: list[int] | None,
    train_fraction: float | None,
    val_fraction: float | None,
    scaling_method: str | None,
) -> dict:
    """Assemble the preprocessing configuration payload from UI controls."""
    selected = lambda values: bool(values)

    return {
        "cleaning": {
            "missing_strategy": missing_strategy or "interpolate",
            "outlier_method": outlier_method or "noop",
            "outlier_threshold": float(outlier_threshold or 3.0),
            "outlier_columns": ["power_average_w_normalized", "ghi_pyr", "air_temperature"],
            "remove_duplicates": selected(remove_duplicates),
            "filter_night": selected(filter_night),
            "night_ghi_threshold": float(night_threshold or 10.0),
        },
        "time_features": {
            "cyclic": selected(time_cyclic),
            "lag_shifts": [int(v) for v in (lag_shifts or [])],
            "lag_columns": DEFAULT_LAG_COLUMNS,
            "rolling_windows": [int(v) for v in (rolling_windows or [])],
            "rolling_columns": DEFAULT_LAG_COLUMNS,
            "rolling_stats": ["mean", "std"],
            "diff_periods": [int(v) for v in (diff_periods or [])],
            "diff_columns": DEFAULT_LAG_COLUMNS,
        },
        "solar_features": {
            "solar_position": selected(solar_position),
            "clear_sky_ghi": selected(clear_sky),
            "clearness_index": selected(clearness),
            "irradiance_ratios": selected(irradiance_ratios),
            "performance_ratio": selected(performance_ratio),
        },
        "weather_features": {
            "interactions": selected(weather_interactions),
            "ema_spans": [int(v) for v in (ema_spans or [])],
            "ema_columns": ["ghi_pyr", "air_temperature"],
            "ramp_periods": [int(v) for v in (ramp_periods or [])],
            "ramp_columns": ["ghi_pyr", "power_average_w_normalized"],
        },
        "feature_selection": {
            "target_column": target_column or "power_average_w_normalized",
            "strategy": selection_strategy or "all",
            "top_k": max(int(selection_top_k or 10), 1),
            "min_abs_correlation": float(selection_min_corr or 0.2),
            "manual_features": list(manual_features or []),
        },
        "dataset_builder": {
            "horizon_steps": [int(v) for v in (horizon_steps or [])],
            "train_fraction": float(train_fraction or 0.7),
            "val_fraction": float(val_fraction or 0.15),
            "scaling_method": scaling_method or "minmax",
        },
    }


def _preprocess_summary_chips(result) -> list:
    """Build reusable summary-chip elements from a PipelineResult."""
    report = result.quality_report
    split = report["splits"]
    return [
        html.Span(f"{report['original_rows']:,} -> {report['final_rows']:,} rows", className="preprocessing-summary-item"),
        html.Span(f"{report['original_columns']} -> {report['final_columns']} cols", className="preprocessing-summary-item"),
        html.Span(f"{len(report['added_columns'])} features added", className="preprocessing-summary-item"),
        html.Span(f"Train {split['train_rows']:,} / Val {split['val_rows']:,} / Test {split['test_rows']:,}", className="preprocessing-summary-item"),
    ]


def _preprocess_tab_actions(tab_key: str) -> html.Div:
    return html.Div(
        className="preprocessing-action-row preprocessing-action-box",
        children=[
            html.Button(
                "Update",
                id=f"data-analysis-preprocess-{tab_key}-update-btn",
                className="action-btn action-btn-primary",
            ),
            html.Button(
                "Reset",
                id=f"data-analysis-preprocess-{tab_key}-reset-btn",
                className="action-btn",
            ),
            html.Span("Changes apply only after Update.", className="preprocessing-action-note"),
        ],
    )


def _build_cleaning_section(
    missing_strategy: str | None,
    outlier_method: str | None,
    outlier_threshold: float | None,
    remove_duplicates: list[str] | None,
    filter_night: list[str] | None,
    night_threshold: float | None,
) -> dict:
    selected = lambda values: bool(values)
    return {
        "missing_strategy": missing_strategy or "interpolate",
        "outlier_method": outlier_method or "noop",
        "outlier_threshold": float(outlier_threshold or 3.0),
        "outlier_columns": ["power_average_w_normalized", "ghi_pyr", "air_temperature"],
        "remove_duplicates": selected(remove_duplicates),
        "filter_night": selected(filter_night),
        "night_ghi_threshold": float(night_threshold or 10.0),
    }


def _build_time_features_section(
    time_cyclic: list[str] | None,
    lag_shifts: list[int] | None,
    rolling_windows: list[int] | None,
    diff_periods: list[int] | None,
) -> dict:
    selected = lambda values: bool(values)
    return {
        "cyclic": selected(time_cyclic),
        "lag_shifts": [int(v) for v in (lag_shifts or [])],
        "lag_columns": DEFAULT_LAG_COLUMNS,
        "rolling_windows": [int(v) for v in (rolling_windows or [])],
        "rolling_columns": DEFAULT_LAG_COLUMNS,
        "rolling_stats": ["mean", "std"],
        "diff_periods": [int(v) for v in (diff_periods or [])],
        "diff_columns": DEFAULT_LAG_COLUMNS,
    }


def _build_solar_features_section(
    solar_position: list[str] | None,
    clear_sky: list[str] | None,
    clearness: list[str] | None,
    irradiance_ratios: list[str] | None,
    performance_ratio: list[str] | None,
) -> dict:
    selected = lambda values: bool(values)
    return {
        "solar_position": selected(solar_position),
        "clear_sky_ghi": selected(clear_sky),
        "clearness_index": selected(clearness),
        "irradiance_ratios": selected(irradiance_ratios),
        "performance_ratio": selected(performance_ratio),
    }


def _build_weather_features_section(
    weather_interactions: list[str] | None,
    ema_spans: list[int] | None,
    ramp_periods: list[int] | None,
) -> dict:
    selected = lambda values: bool(values)
    return {
        "interactions": selected(weather_interactions),
        "ema_spans": [int(v) for v in (ema_spans or [])],
        "ema_columns": ["ghi_pyr", "air_temperature"],
        "ramp_periods": [int(v) for v in (ramp_periods or [])],
        "ramp_columns": ["ghi_pyr", "power_average_w_normalized"],
    }


def _build_feature_selection_section(
    target_column: str | None,
    selection_strategy: str | None,
    selection_top_k: float | None,
    selection_min_corr: float | None,
    manual_features: list[str] | None,
) -> dict:
    return {
        "target_column": target_column or "power_average_w_normalized",
        "strategy": selection_strategy or "all",
        "top_k": max(int(selection_top_k or 10), 1),
        "min_abs_correlation": float(selection_min_corr or 0.2),
        "manual_features": list(manual_features or []),
    }


def _build_dataset_builder_section(
    horizon_steps: list[int] | None,
    train_fraction: float | None,
    val_fraction: float | None,
    scaling_method: str | None,
) -> dict:
    return {
        "horizon_steps": [int(v) for v in (horizon_steps or [])],
        "train_fraction": float(train_fraction or 0.7),
        "val_fraction": float(val_fraction or 0.15),
        "scaling_method": scaling_method or "minmax",
    }


def _default_preprocess_config() -> dict:
    return json.loads(json.dumps(DEFAULT_CONFIG))


@callback(
    Output("data-analysis-preprocess-config", "data"),
    Input("data-analysis-preprocess-cleaning-update-btn", "n_clicks"),
    Input("data-analysis-preprocess-time-update-btn", "n_clicks"),
    Input("data-analysis-preprocess-solar-update-btn", "n_clicks"),
    Input("data-analysis-preprocess-weather-update-btn", "n_clicks"),
    Input("data-analysis-preprocess-selection-update-btn", "n_clicks"),
    Input("data-analysis-preprocess-builder-update-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    State("data-analysis-preprocess-missing-strategy", "value"),
    State("data-analysis-preprocess-outlier-method", "value"),
    State("data-analysis-preprocess-outlier-threshold", "value"),
    State("data-analysis-preprocess-remove-duplicates", "value"),
    State("data-analysis-preprocess-filter-night", "value"),
    State("data-analysis-preprocess-night-threshold", "value"),
    State("data-analysis-preprocess-time-cyclic", "value"),
    State("data-analysis-preprocess-lag-shifts", "value"),
    State("data-analysis-preprocess-rolling-windows", "value"),
    State("data-analysis-preprocess-diff-periods", "value"),
    State("data-analysis-preprocess-solar-position", "value"),
    State("data-analysis-preprocess-clear-sky", "value"),
    State("data-analysis-preprocess-clearness", "value"),
    State("data-analysis-preprocess-irradiance-ratios", "value"),
    State("data-analysis-preprocess-performance-ratio", "value"),
    State("data-analysis-preprocess-weather-interactions", "value"),
    State("data-analysis-preprocess-ema-spans", "value"),
    State("data-analysis-preprocess-ramp-periods", "value"),
    State("data-analysis-preprocess-target-column", "value"),
    State("data-analysis-preprocess-selection-strategy", "value"),
    State("data-analysis-preprocess-selection-top-k", "value"),
    State("data-analysis-preprocess-selection-min-corr", "value"),
    State("data-analysis-preprocess-selection-manual-features", "value"),
    State("data-analysis-preprocess-horizon-steps", "value"),
    State("data-analysis-preprocess-train-fraction", "value"),
    State("data-analysis-preprocess-val-fraction", "value"),
    State("data-analysis-preprocess-scaling-method", "value"),
    prevent_initial_call=True,
)
def update_preprocess_config(
    _cleaning_clicks: int | None,
    _time_clicks: int | None,
    _solar_clicks: int | None,
    _weather_clicks: int | None,
    _selection_clicks: int | None,
    _builder_clicks: int | None,
    config: dict | None,
    missing_strategy: str | None,
    outlier_method: str | None,
    outlier_threshold: float | None,
    remove_duplicates: list[str] | None,
    filter_night: list[str] | None,
    night_threshold: float | None,
    time_cyclic: list[str] | None,
    lag_shifts: list[int] | None,
    rolling_windows: list[int] | None,
    diff_periods: list[int] | None,
    solar_position: list[str] | None,
    clear_sky: list[str] | None,
    clearness: list[str] | None,
    irradiance_ratios: list[str] | None,
    performance_ratio: list[str] | None,
    weather_interactions: list[str] | None,
    ema_spans: list[int] | None,
    ramp_periods: list[int] | None,
    target_column: str | None,
    selection_strategy: str | None,
    selection_top_k: float | None,
    selection_min_corr: float | None,
    manual_features: list[str] | None,
    horizon_steps: list[int] | None,
    train_fraction: float | None,
    val_fraction: float | None,
    scaling_method: str | None,
):
    trigger = ctx.triggered_id
    if trigger is None:
        return no_update

    current_config = dict(config or _default_preprocess_config())

    if trigger == "data-analysis-preprocess-cleaning-update-btn":
        current_config["cleaning"] = _build_cleaning_section(
            missing_strategy,
            outlier_method,
            outlier_threshold,
            remove_duplicates,
            filter_night,
            night_threshold,
        )
        return current_config

    if trigger == "data-analysis-preprocess-time-update-btn":
        current_config["time_features"] = _build_time_features_section(
            time_cyclic,
            lag_shifts,
            rolling_windows,
            diff_periods,
        )
        return current_config

    if trigger == "data-analysis-preprocess-solar-update-btn":
        current_config["solar_features"] = _build_solar_features_section(
            solar_position,
            clear_sky,
            clearness,
            irradiance_ratios,
            performance_ratio,
        )
        return current_config

    if trigger == "data-analysis-preprocess-weather-update-btn":
        current_config["weather_features"] = _build_weather_features_section(
            weather_interactions,
            ema_spans,
            ramp_periods,
        )
        return current_config

    if trigger == "data-analysis-preprocess-selection-update-btn":
        current_config["feature_selection"] = _build_feature_selection_section(
            target_column,
            selection_strategy,
            selection_top_k,
            selection_min_corr,
            manual_features,
        )
        return current_config

    if trigger == "data-analysis-preprocess-builder-update-btn":
        current_config["dataset_builder"] = _build_dataset_builder_section(
            horizon_steps,
            train_fraction,
            val_fraction,
            scaling_method,
        )
        return current_config

    return no_update


@callback(
    Output("data-analysis-preprocess-missing-strategy", "value"),
    Output("data-analysis-preprocess-outlier-method", "value"),
    Output("data-analysis-preprocess-outlier-threshold", "value"),
    Output("data-analysis-preprocess-remove-duplicates", "value"),
    Output("data-analysis-preprocess-filter-night", "value"),
    Output("data-analysis-preprocess-night-threshold", "value"),
    Input("data-analysis-preprocess-cleaning-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_cleaning(_n_clicks: int | None, config: dict | None):
    cleaning = (config or _default_preprocess_config())["cleaning"]
    return (
        cleaning["missing_strategy"],
        cleaning["outlier_method"],
        cleaning["outlier_threshold"],
        ["dupes"] if cleaning["remove_duplicates"] else [],
        ["night"] if cleaning["filter_night"] else [],
        cleaning["night_ghi_threshold"],
    )


@callback(
    Output("data-analysis-preprocess-time-cyclic", "value"),
    Output("data-analysis-preprocess-lag-shifts", "value"),
    Output("data-analysis-preprocess-rolling-windows", "value"),
    Output("data-analysis-preprocess-diff-periods", "value"),
    Input("data-analysis-preprocess-time-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_time(_n_clicks: int | None, config: dict | None):
    time_features = (config or _default_preprocess_config())["time_features"]
    return (
        ["cyclic"] if time_features["cyclic"] else [],
        time_features["lag_shifts"],
        time_features["rolling_windows"],
        time_features["diff_periods"],
    )


@callback(
    Output("data-analysis-preprocess-solar-position", "value"),
    Output("data-analysis-preprocess-clear-sky", "value"),
    Output("data-analysis-preprocess-clearness", "value"),
    Output("data-analysis-preprocess-irradiance-ratios", "value"),
    Output("data-analysis-preprocess-performance-ratio", "value"),
    Input("data-analysis-preprocess-solar-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_solar(_n_clicks: int | None, config: dict | None):
    solar_features = (config or _default_preprocess_config())["solar_features"]
    return (
        ["position"] if solar_features["solar_position"] else [],
        ["clear_sky"] if solar_features["clear_sky_ghi"] else [],
        ["clearness"] if solar_features["clearness_index"] else [],
        ["ratios"] if solar_features["irradiance_ratios"] else [],
        ["pr"] if solar_features["performance_ratio"] else [],
    )


@callback(
    Output("data-analysis-preprocess-weather-interactions", "value"),
    Output("data-analysis-preprocess-ema-spans", "value"),
    Output("data-analysis-preprocess-ramp-periods", "value"),
    Input("data-analysis-preprocess-weather-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_weather(_n_clicks: int | None, config: dict | None):
    weather_features = (config or _default_preprocess_config())["weather_features"]
    return (
        ["interactions"] if weather_features["interactions"] else [],
        weather_features["ema_spans"],
        weather_features["ramp_periods"],
    )


@callback(
    Output("data-analysis-preprocess-selection-top-k-wrap", "style"),
    Output("data-analysis-preprocess-selection-min-corr-wrap", "style"),
    Output("data-analysis-preprocess-selection-manual-wrap", "style"),
    Input("data-analysis-preprocess-selection-strategy", "value"),
)
def toggle_selection_strategy_controls(strategy: str | None):
    shown = {"display": "block"}
    hidden = {"display": "none"}
    if strategy == "top_k_corr":
        return shown, hidden, hidden
    if strategy == "corr_threshold":
        return hidden, shown, hidden
    if strategy == "manual":
        return hidden, hidden, shown
    return hidden, hidden, hidden


@callback(
    Output("data-analysis-preprocess-target-column", "value"),
    Output("data-analysis-preprocess-selection-strategy", "value"),
    Output("data-analysis-preprocess-selection-top-k", "value"),
    Output("data-analysis-preprocess-selection-min-corr", "value"),
    Output("data-analysis-preprocess-selection-manual-features", "value"),
    Input("data-analysis-preprocess-selection-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_selection(_n_clicks: int | None, config: dict | None):
    feature_selection = (config or _default_preprocess_config())["feature_selection"]
    return (
        feature_selection["target_column"],
        feature_selection.get("strategy", "all"),
        feature_selection.get("top_k", 10),
        feature_selection.get("min_abs_correlation", 0.2),
        feature_selection.get("manual_features", []),
    )


@callback(
    Output("data-analysis-preprocess-horizon-steps", "value"),
    Output("data-analysis-preprocess-train-fraction", "value"),
    Output("data-analysis-preprocess-val-fraction", "value"),
    Output("data-analysis-preprocess-scaling-method", "value"),
    Input("data-analysis-preprocess-builder-reset-btn", "n_clicks"),
    State("data-analysis-preprocess-config", "data"),
    prevent_initial_call=True,
)
def reset_preprocess_builder(_n_clicks: int | None, config: dict | None):
    dataset_builder = (config or _default_preprocess_config())["dataset_builder"]
    return (
        dataset_builder["horizon_steps"],
        dataset_builder["train_fraction"],
        dataset_builder["val_fraction"],
        dataset_builder["scaling_method"],
    )


@callback(
    Output("data-analysis-preprocess-panel", "style"),
    Output("data-analysis-preprocess-cleaning-summary", "children"),
    Output("data-analysis-preprocess-time-summary", "children"),
    Output("data-analysis-preprocess-solar-summary", "children"),
    Output("data-analysis-preprocess-weather-summary", "children"),
    Output("data-analysis-preprocess-selection-summary", "children"),
    Output("data-analysis-preprocess-selection-manual-features", "options"),
    Output("data-analysis-preprocess-builder-summary", "children"),
    Output("data-analysis-preprocess-quality-report", "children"),
    Output("data-analysis-preprocess-state", "data"),
    Input("data-analysis-mode", "data"),
    Input("data-analysis-existing-select", "value"),
    Input("data-analysis-preprocess-config", "data"),
)
def render_preprocessing_state(
    mode: str | None,
    selected_file: str | None,
    config: dict | None,
):
    hidden = {"display": "none"}
    shown = {"display": "block"}
    empty_note = lambda text: html.P(text, className="preprocessing-note")
    active_config = config or _default_preprocess_config()

    if mode != "analyze" or not selected_file:
        return (
            shown,
            empty_note("Select Analyze Existing System and a dataset to enable preprocessing."),
            empty_note("No time features computed."),
            empty_note("No solar features computed."),
            empty_note("No weather features computed."),
            empty_note("No feature selection summary yet."),
            [],
            empty_note("No dataset summary yet."),
            empty_note("No quality report yet."),
            None,
        )

    try:
        result = get_pipeline_result(selected_file, active_config)
    except Exception as exc:
        message = html.P(f"Pipeline error: {exc}", className="preprocessing-note")
        return (
            shown,
            message,
            message,
            message,
            message,
            message,
            [],
            message,
            message,
            None,
        )

    report = result.quality_report
    log = {entry["step"]: entry for entry in report["pipeline_log"]}

    cleaning_chips = []
    if "clean_missing" in log:
        cleaning_chips.append(html.Span(f"{len(log['clean_missing']['strategies'])} columns cleaned", className="preprocessing-summary-item"))
    if "remove_duplicates" in log and log["remove_duplicates"].get("dropped"):
        cleaning_chips.append(html.Span(f"{log['remove_duplicates']['dropped']:,} duplicate rows removed", className="preprocessing-summary-item"))
    if "filter_night" in log and log["filter_night"].get("dropped"):
        cleaning_chips.append(html.Span(f"{log['filter_night']['dropped']:,} night rows removed", className="preprocessing-summary-item"))
    if "detect_outliers" in log:
        cleaning_chips.append(html.Span(f"{log['detect_outliers']['flagged_total']:,} outlier rows flagged", className="preprocessing-summary-item"))
    if not cleaning_chips:
        cleaning_chips = [html.Span("Cleaning applied", className="preprocessing-summary-item")]
    cleaning_summary = html.Div(children=cleaning_chips, className="preprocessing-summary")

    time_inner = []
    if "time_cyclic" in log:
        time_inner.append(html.Span(f"{log['time_cyclic']['columns']} cyclic columns", className="preprocessing-summary-item"))
    for step in ("time_lag", "time_rolling", "time_diff"):
        if step in log and log[step].get("columns"):
            time_inner.append(html.Span(f"{len(log[step]['columns'])} {step.replace('time_', '')} columns", className="preprocessing-summary-item"))
    time_summary = html.Div(children=time_inner or [html.Span("No time features selected", className="preprocessing-summary-item")], className="preprocessing-summary")

    solar_inner = []
    if "solar_features" in log:
        solar_columns = log["solar_features"].get("columns") or []
        if solar_columns:
            solar_inner.append(html.Span(f"{len(solar_columns)} solar columns", className="preprocessing-summary-item"))
    if "clearness_index" in report["added_columns"]:
        solar_inner.append(html.Span("clearness index", className="preprocessing-summary-item"))
    solar_summary = html.Div(children=solar_inner or [html.Span("No solar features selected", className="preprocessing-summary-item")], className="preprocessing-summary")

    weather_inner = []
    if "weather_features" in log:
        weather_columns = log["weather_features"].get("columns") or []
        if weather_columns:
            weather_inner.append(html.Span(f"{len(weather_columns)} weather columns", className="preprocessing-summary-item"))
    weather_summary = html.Div(children=weather_inner or [html.Span("No weather features selected", className="preprocessing-summary-item")], className="preprocessing-summary")

    # Feature selection summary
    target = active_config["feature_selection"]["target_column"]
    selection_cfg = active_config["feature_selection"]
    selected_features = result.selected_feature_columns
    selected_set = set(selected_features)
    manual_options = [
        {"label": row["column"], "value": row["column"]}
        for row in result.feature_summary
    ]
    rows_by_column = {row["column"]: row for row in result.feature_summary}
    strategy = selection_cfg.get("strategy", "all")
    if strategy == "manual":
        display_columns = [
            column for column in selection_cfg.get("manual_features", []) if column in rows_by_column
        ]
    elif strategy in {"top_k_corr", "corr_threshold"}:
        display_columns = [row["column"] for row in result.feature_summary if row["column"] in selected_set]
    else:
        display_columns = [row["column"] for row in result.feature_summary]

    display_rows_source = [rows_by_column[column] for column in display_columns if column in rows_by_column]
    feature_rows = [
        [
            row["column"],
            f"{row['correlation']:.4f}" if row["correlation"] is not None else "-",
            f"{row['variance']:.4g}",
            f"{row['missing_pct']:.2f}%",
        ]
        for row in display_rows_source
    ]
    selection_parts = []
    selection_parts.append(
        html.Div(
            className="preprocessing-summary",
            children=[
                html.Span(f"Target: {target}", className="preprocessing-summary-item"),
                html.Span(
                    f"Strategy: {SELECTION_STRATEGY_LABELS.get(selection_cfg.get('strategy', 'all'), 'All candidate features')}",
                    className="preprocessing-summary-item",
                ),
                html.Span(f"Candidates: {len(result.feature_summary)}", className="preprocessing-summary-item"),
                html.Span(f"Selected: {len(selected_features)}", className="preprocessing-summary-item"),
            ],
        )
    )
    if feature_rows:
        selection_parts.append(
            html.Div(className="preprocessing-selection-table", children=simple_table(
                ["Feature", "Correlation", "Variance", "Missing"],
                feature_rows,
            ))
        )
    if selection_cfg.get("strategy") == "top_k_corr":
        selection_parts.append(
            html.P(
                f"Top-K filter active: the dataset builder keeps the {selection_cfg.get('top_k', 10)} strongest features by absolute correlation.",
                className="preprocessing-note",
            )
        )
    elif selection_cfg.get("strategy") == "corr_threshold":
        selection_parts.append(
            html.P(
                f"Threshold filter active: only features with |correlation| >= {selection_cfg.get('min_abs_correlation', 0.2):.2f} are kept.",
                className="preprocessing-note",
            )
        )
    elif selection_cfg.get("strategy") == "manual":
        selection_parts.append(
            html.P(
                "Manual filter active: only the features chosen in the manual list move forward into dataset building and scaling.",
                className="preprocessing-note",
            )
        )
    else:
        selection_parts.append(
            html.P(
                "All candidate features are currently being passed into dataset building and scaling.",
                className="preprocessing-note",
            )
        )
    if strategy == "all":
        selection_parts.append(
            html.P(
                "Showing all candidate features because the full candidate set is active for dataset building and scaling.",
                className="preprocessing-note",
            )
        )
    elif not selected_features:
        selection_parts.append(
            html.P(
                "No features match the current selection rule. Adjust the threshold, top K, or manual list before exporting.",
                className="preprocessing-note",
            )
        )
    if selected_features:
        preview = ", ".join(selected_features[:8])
        suffix = " ..." if len(selected_features) > 8 else ""
        selection_parts.append(
            html.P(f"Selected feature preview: {preview}{suffix}", className="preprocessing-note")
        )
    if result.mutual_info["available"]:
        mi_rows = result.mutual_info.get("rows") or []
        if mi_rows:
            mi_text = ", ".join(f"{row['column']} ({row['mi']:.3f})" for row in mi_rows[:5])
            selection_parts.append(html.P(f"Mutual information top: {mi_text}", className="preprocessing-note"))
    else:
        selection_parts.append(
            html.P("scikit-learn not installed — mutual-information ranking unavailable.", className="preprocessing-note")
        )
    selection_summary = html.Div(children=selection_parts or [empty_note("No feature summary yet.")])

    builder_chips = _preprocess_summary_chips(result)
    horizon_steps = active_config["dataset_builder"]["horizon_steps"]
    builder_parts = [
        html.Div(children=builder_chips, className="preprocessing-summary"),
    ]
    builder_parts.append(
        html.P(
            f"Selection strategy in use: {SELECTION_STRATEGY_LABELS.get(selection_cfg.get('strategy', 'all'), 'All candidate features')} ({len(selected_features)} features).",
            className="preprocessing-note",
        )
    )
    if selected_features:
        builder_preview = ", ".join(selected_features[:8])
        builder_suffix = " ..." if len(selected_features) > 8 else ""
        builder_parts.append(
            html.P(
                f"Active feature set for dataset builder: {builder_preview}{builder_suffix}",
                className="preprocessing-note",
            )
        )
    if horizon_steps:
        builder_parts.append(
            html.P(f"Forecast horizons: {', '.join(f'{h} step' for h in horizon_steps)}", className="preprocessing-note")
        )
    builder_summary = html.Div(children=builder_parts)

    # Quality report
    ts_info = report["timestamp"]
    split = report["splits"]
    quality_rows = [
        ["Rows after preprocessing", f"{report['final_rows']:,}"],
        ["Columns after preprocessing", f"{report['final_columns']:,}"],
        ["Added columns", f"{report['column_delta']:+,d}"],
        ["Selected modeling features", f"{len(selected_features):,}"],
        ["Selection strategy", SELECTION_STRATEGY_LABELS.get(selection_cfg.get("strategy", "all"), "All candidate features")],
        [
            "Selected feature preview",
            ", ".join(selected_features[:6]) + (" ..." if len(selected_features) > 6 else "") if selected_features else "-",
        ],
        ["Timestamp start", ts_info["start"] or "-"],
        ["Timestamp end", ts_info["end"] or "-"],
        ["Median interval", f"{ts_info['median_interval_minutes']:.2f} min" if ts_info["median_interval_minutes"] else "-"],
        ["Monotonic timestamps", "Yes" if ts_info["monotonic"] else "No"],
        ["Gaps > 1.5x interval", f"{ts_info['gap_count_gt_1_5x']:,}"],
        ["Train / Val / Test", f"{split['train_rows']:,} / {split['val_rows']:,} / {split['test_rows']:,}"],
        ["Scaling method", active_config["dataset_builder"]["scaling_method"]],
    ]
    quality_report_children = simple_table(["Metric", "Value"], quality_rows)

    state_payload = {
        "file_name": selected_file,
        "rows_after": report["final_rows"],
        "columns_after": report["final_columns"],
        "added_columns_count": len(report["added_columns"]),
        "selected_features_count": len(selected_features),
        "split_train": report["splits"]["train_rows"],
        "split_val": report["splits"]["val_rows"],
        "split_test": report["splits"]["test_rows"],
    }

    return (
        shown,
        cleaning_summary,
        time_summary,
        solar_summary,
        weather_summary,
        selection_summary,
        manual_options,
        builder_summary,
        quality_report_children,
        state_payload,
    )


@callback(
    Output("data-analysis-generate-dataset-status", "children"),
    Output("data-analysis-generated-dataset", "data"),
    Input("data-analysis-generate-dataset-btn", "n_clicks"),
    Input("system-select", "value"),
    State("data-analysis-mode", "data"),
    State("data-analysis-existing-select", "value"),
    State("data-analysis-preprocess-config", "data"),
    running=[(Output("data-analysis-generate-dataset-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def generate_analysis_dataset(
    n_clicks: int | None,
    system_number: int | None,
    mode: str | None,
    selected_file: str | None,
    config: dict | None,
):
    if ctx.triggered_id == "system-select":
        return html.Div(), None
    if not n_clicks:
        return html.Div(), None
    if mode != "analyze" or not selected_file:
        return html.P("Select Analyze Existing System and a dataset first.", className="preprocessing-export-error"), None
    try:
        result = get_pipeline_result(selected_file, config)
    except Exception as exc:
        return html.P(f"Dataset generation failed: {exc}", className="preprocessing-export-error"), None
    return (
        html.P(
            f"Dataset generated: {result.quality_report['final_rows']:,} rows, "
            f"{len(result.selected_feature_columns)} modeling variables available for plotting.",
            className="preprocessing-export-success",
        ),
        {"system_number": system_number, "file_name": selected_file},
    )


@callback(
    Output("data-analysis-export-status", "children"),
    Input("data-analysis-export-analysis-btn", "n_clicks"),
    Input("data-analysis-existing-select", "value"),
    Input("data-analysis-preprocess-config", "data"),
    running=[(Output("data-analysis-export-analysis-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def export_dataset_for_analysis(
    n_clicks: int | None,
    selected_file: str | None,
    config: dict | None,
):
    if not n_clicks:
        return html.Div()
    if not selected_file:
        return html.P("No dataset selected for export.", className="new-system-status-muted")

    try:
        _export_info, result = export_preprocessing_artifacts(selected_file, config or _default_preprocess_config())
    except Exception as exc:
        return html.Div(
            className="preprocessing-export-error",
            children=[
                html.H4("Export Failed", className="new-system-result-title"),
                html.P(str(exc), className="new-system-result-text"),
            ],
        )

    return html.Div(
        className="preprocessing-export-success",
        children=[
            html.H4("Dataset Ready", className="new-system-result-title"),
            html.P("The processed dataset is ready for PV forecasting and Cooling demand.", className="new-system-result-text"),
            html.P(
                f"{result.quality_report['final_rows']:,} rows and "
                f"{result.quality_report['final_columns']} columns exported.",
                className="new-system-result-note",
            ),
            html.P(
                f"Selection strategy applied: {SELECTION_STRATEGY_LABELS.get(result.config.get('feature_selection', {}).get('strategy', 'all'), 'All candidate features')} "
                f"with {len(result.selected_feature_columns)} modeling features.",
                className="new-system-result-note",
            ),
        ],
    )


@callback(
    Output("data-analysis-download-dataset", "data"),
    Input("data-analysis-download-dataset-btn", "n_clicks"),
    State("data-analysis-existing-select", "value"),
    State("data-analysis-preprocess-config", "data"),
    running=[(Output("data-analysis-download-dataset-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def download_processed_dataset(
    n_clicks: int | None,
    selected_file: str | None,
    config: dict | None,
):
    if not n_clicks or not selected_file:
        return None
    try:
        result = get_pipeline_result(selected_file, config or _default_preprocess_config())
    except Exception:
        return None
    dataset_stem = selected_file.rsplit(".", 1)[0]
    return dcc.send_data_frame(
        result.dataframe.to_csv,
        f"{dataset_stem}_processed_dataset.csv",
        index=False,
    )


@callback(
    Output("new-system-upload-status", "children"),
    Input("new-system-upload", "contents"),
    Input("new-system-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents: str | None, filename: str | None):
    if not contents or not filename:
        return html.P("No file selected.", className="new-system-status-muted")
    return html.P(
        f"File ready: {filename}",
        className="new-system-status-ready",
    )


@callback(
    Output("new-system-result", "children"),
    Input("new-system-create-btn", "n_clicks"),
    Input("new-system-upload", "contents"),
    Input("new-system-upload", "filename"),
    Input("new-system-id", "value"),
    Input("new-system-capacity", "value"),
    Input("new-system-city", "value"),
    Input("new-system-lat", "value"),
    Input("new-system-lon", "value"),
    prevent_initial_call=True,
)
def create_new_system(
    n_clicks: int | None,
    contents: str | None,
    filename: str | None,
    system_id: int | None,
    capacity_kw: float | None,
    city: str | None,
    lat: float | None,
    lon: float | None,
):
    if not n_clicks:
        return html.Div()

    # Validate required inputs
    missing = []
    if not system_id:
        missing.append("System ID")
    if not capacity_kw:
        missing.append("Capacity")
    if not city or not str(city).strip():
        missing.append("City")
    if lat is None:
        missing.append("Latitude")
    if lon is None:
        missing.append("Longitude")
    if not contents or not filename:
        missing.append("PV Power Data File")

    if missing:
        return html.Div(
            className="new-system-result-error",
            children=[
                html.H4("Missing Required Information", className="new-system-result-title"),
                html.P(f"Please provide: {', '.join(missing)}.", className="new-system-result-text"),
            ],
        )

    # Save uploaded file to a temporary location
    temp_path = None
    try:
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        temp_dir = PV_DATA_DIR / ".tmp_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        with open(temp_path, "wb") as handle:
            handle.write(decoded)
    except Exception as exc:
        return html.Div(
            className="new-system-result-error",
            children=[
                html.H4("Upload Error", className="new-system-result-title"),
                html.P(f"Could not save the uploaded file: {exc}", className="new-system-result-text"),
            ],
        )

    # Create the new system dataset
    try:
        result = create_new_system_dataset(
            system_id=int(system_id),
            capacity_kw=float(capacity_kw),
            city=str(city).strip(),
            lat=float(lat),
            lon=float(lon),
            uploaded_file_path=temp_path,
        )

        # Clean up temp file
        try:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        return html.Div(
            className="new-system-result-success",
            children=[
                html.H4("Dataset Created Successfully", className="new-system-result-title"),
                html.P(
                    f"Created {result['row_count']:,} rows of data.",
                    className="new-system-result-text",
                ),
                html.P(
                    f"Time range: {result['start_time']} to {result['end_time']}",
                    className="new-system-result-text",
                ),
                html.P(
                    f"Saved as: {result['file_name']}",
                    className="new-system-result-text",
                ),
                html.P(
                    "The new dataset is now available in the M2_PVnowcasting_module/data folder. "
                    "Switch to Analyze Existing System to inspect it.",
                    className="new-system-result-note",
                ),
            ],
        )
    except NewSystemError as exc:
        # Clean up temp file
        try:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return html.Div(
            className="new-system-result-error",
            children=[
                html.H4("Dataset Creation Failed", className="new-system-result-title"),
                html.P(str(exc), className="new-system-result-text"),
            ],
        )
    except Exception as exc:
        # Clean up temp file
        try:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return html.Div(
            className="new-system-result-error",
            children=[
                html.H4("Unexpected Error", className="new-system-result-title"),
                html.P(f"An unexpected error occurred: {exc}", className="new-system-result-text"),
            ],
        )

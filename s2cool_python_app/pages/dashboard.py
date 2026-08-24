from __future__ import annotations

from dash import dcc, html
import plotly.graph_objects as go

from components.ui import info_rows, kpi_card, panel, pill, simple_table
from models.domain import SystemRecord
from services.config_service import load_cooling_sites, load_systems
from services.dataset_service import (
    build_dataset_snapshot,
    count_existing_cooling_outputs,
    count_existing_forecast_outputs,
    load_preview_frame,
)


def _build_preview_figure(system_number: int | None) -> go.Figure:
    df = load_preview_frame(system_number, max_rows=288)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No dataset preview available")
        return fig

    if "power_average_w_normalized" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["_ts"],
                y=df["power_average_w_normalized"],
                mode="lines",
                name="Normalized PV Power",
                line={"color": "#ff9a1f", "width": 2},
            )
        )
    if "air_temperature" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["_ts"],
                y=df["air_temperature"],
                mode="lines",
                name="Air Temperature",
                line={"color": "#5d7fd6", "width": 2},
                yaxis="y2",
            )
        )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 10},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": 1.12},
        yaxis={"title": "Normalized Power"},
        yaxis2={"title": "Temperature (C)", "overlaying": "y", "side": "right"},
    )
    return fig


def build_layout(system: SystemRecord | None) -> html.Div:
    snapshot = build_dataset_snapshot(system.system_number if system else None)
    total_systems = len(load_systems())
    total_sites = len(load_cooling_sites())
    forecast_outputs = count_existing_forecast_outputs()
    cooling_outputs = count_existing_cooling_outputs()

    status_tone = "green" if snapshot.status == "ready" else "orange"
    dataset_status = snapshot.status.replace("_", " ").title()

    system_name = system.name if system else "No system selected"
    system_note = f"{system.location} | {system.capacity_kw:.3g} kW" if system else "Select a system to begin"

    return html.Div(
        children=[
            html.Div(
                className="kpi-grid",
                children=[
                    kpi_card("Selected system", system_name, system_note),
                    kpi_card("Registered PV systems", str(total_systems), "Loaded from systems registry"),
                    kpi_card("Cooling site profiles", str(total_sites), "Ready for generalized site modeling"),
                    kpi_card("Existing research artifacts", str(forecast_outputs + cooling_outputs), "Forecast, backtest, and summary files discovered"),
                ],
            ),
            panel(
                "Operational snapshot",
                "Phase 1 focuses on a clean research dashboard with registry-driven onboarding, diagnostics, and forecast workflows.",
                [
                    html.Div(
                        className="two-col-grid",
                        children=[
                            html.Div(
                                className="panel",
                                children=[
                                    html.H3("Current data readiness", className="section-title"),
                                    html.Div([pill(dataset_status, status_tone)]),
                                    info_rows(
                                        [
                                            ("Latest dataset", snapshot.path.name if snapshot.path else "Not found"),
                                            ("Rows", f"{snapshot.row_count:,}"),
                                            ("Columns", str(snapshot.column_count)),
                                            ("Date span", f"{snapshot.start_ts} to {snapshot.end_ts}"),
                                            (
                                                "Median interval",
                                                f"{snapshot.median_interval_min:.1f} min" if snapshot.median_interval_min is not None else "Unknown",
                                            ),
                                            ("Missing cells", f"{snapshot.missing_cells:,} ({snapshot.missing_pct:.2f}%)"),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                className="panel",
                                children=[
                                    html.H3("Phase 1 module priorities", className="section-title"),
                                    simple_table(
                                        ["Module", "Immediate role"],
                                        [
                                            ["Dashboard", "Surface data freshness, model status, and recent artifacts"],
                                            ["Data Analysis", "Validate schema, clean data, and compare raw vs processed views"],
                                            ["PV Forecasting", "Configure and run generalized multihorizon nowcasting"],
                                            ["Cooling Demand", "Configure site physics and hybrid demand estimation"],
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            panel(
                "Recent PV data preview",
                "A lightweight chart from the latest available dataset for the selected system.",
                [dcc.Graph(figure=_build_preview_figure(system.system_number if system else None), config={"displayModeBar": False})],
            ),
        ]
    )

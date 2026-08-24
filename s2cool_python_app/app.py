from __future__ import annotations

import os
import shutil
import threading
import webbrowser
from pathlib import Path

from dash import Dash, Input, Output, callback, dcc, html

from pages import cooling_demand, dashboard, data_analysis, pv_forecasting
from services.config_service import get_system, get_system_options

APP_TITLE = "S2Cool Research Dashboard"
TAB_ITEMS = [
    ("dashboard", "Dashboard"),
    ("data-analysis", "Data Analysis"),
    ("pv-forecasting", "PV Forecasting"),
    ("cooling-demand", "Cooling Demand"),
    ("niec-optimisation", "NIEC Optimisation"),
    ("energy-management", "Energy Management"),
    ("control-output", "Control Output"),
    ("kpi-reports", "KPI Reports"),
    ("user-guide", "User Guide"),
]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE
server = app.server

system_options = get_system_options()
default_system = system_options[0]["value"] if system_options else None

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    className="topbar-title",
                    children=[
                        html.Div(
                            className="brand-block",
                            children=[
                                html.Img(
                                    src="/assets/s2cool_logo.png",
                                    className="brand-logo",
                                    alt="S2Cool logo",
                                ),
                                html.Div(
                                    className="brand-copy",
                                    children=[
                                        html.Div(
                                            className="brand-title-line",
                                            children=[
                                                html.H1("S2Cool"),
                                                html.Span("|", className="brand-separator"),
                                                html.P("AI Decision Support Application"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="topbar-icons",
                            children=[
                                html.Span(className="top-icon-dot"),
                                html.Span(className="top-icon-dot"),
                                html.Span(className="top-icon-dot"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="nav-strip",
            children=[
                dcc.Tabs(
                    id="app-tabs",
                    className="dash-tabs",
                    value="dashboard",
                    children=[dcc.Tab(label=label, value=value) for value, label in TAB_ITEMS],
                ),
            ],
        ),
        html.Div(
            id="context-strip",
            className="context-strip",
            children=[
                html.Div(
                    className="context-left",
                    children=[
                        html.Span("S2Cool", className="context-link"),
                        html.Span("|", className="context-divider"),
                        html.Span("Hybrid NIEC + MVC Control", className="context-title"),
                        dcc.Dropdown(
                            id="mode-select",
                            className="inline-select small-select",
                            options=[
                                {"label": "Select Mode", "value": "select"},
                                {"label": "PV + Grid", "value": "pv-grid"},
                                {"label": "PV Priority", "value": "pv-priority"},
                            ],
                            value="select",
                            clearable=False,
                            searchable=False,
                        ),
                        html.Span("|", className="context-divider"),
                        html.Span("Lab Unit #01 - Status:", className="context-label"),
                        html.Span("RUNNING", className="status-running"),
                        dcc.Dropdown(
                            id="system-select",
                            className="inline-select medium-select",
                            options=system_options,
                            value=default_system,
                            clearable=False,
                            searchable=False,
                        ),
                        html.Div("All Systems Normal", className="health-pill"),
                    ],
                ),
                html.Div(
                    className="context-right",
                    children=[
                        html.Span("Updated: 12:15 PM", className="updated-label"),
                        html.Span("", className="header-circle"),
                        html.Span("", className="header-circle"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="content-wrap",
            children=[
                html.Div(
                    id="content-header",
                    className="content-header",
                    children=[
                        html.Div(
                            className="content-heading",
                            children=[
                                html.H2("Research Workspace"),
                                html.P("Dashboard shell aligned to the control-center style so you can continue building the analysis modules on top of it."),
                            ],
                        ),
                        html.Div(
                            className="content-actions",
                            children=[
                                html.Div("Active Study Layout", className="content-chip"),
                            ],
                        ),
                    ],
                ),
                html.Div(id="page-content"),
            ],
        ),
    ],
)

# Keep dynamic tab component IDs available for callback validation.
app.validation_layout = html.Div(
    children=[
        app.layout,
        dashboard.build_layout(get_system(default_system)),
        data_analysis.build_layout(get_system(default_system)),
        pv_forecasting.build_layout(get_system(default_system)),
        cooling_demand.build_layout(get_system(default_system)),
    ]
)


@callback(Output("page-content", "children"), Input("app-tabs", "value"), Input("system-select", "value"))
def render_page(tab_name: str, system_number: int | None):
    system = get_system(system_number)
    if tab_name == "dashboard":
        return dashboard.build_layout(system)
    if tab_name == "data-analysis":
        return data_analysis.build_layout(system)
    if tab_name == "pv-forecasting":
        return pv_forecasting.build_layout(system)
    if tab_name == "cooling-demand":
        return cooling_demand.build_layout(system)
    return html.Div(
        className="panel placeholder-panel",
        children=[
            html.H2("Planned Module", className="section-title"),
            html.P(
                "This top-level navigation item is kept in the shell to match the intended control-room layout. The implementation focus remains on Dashboard, Data Analysis, PV Forecasting, and Cooling Demand.",
                className="section-subtitle",
            ),
        ],
    )


@callback(Output("content-header", "style"), Input("app-tabs", "value"))
def toggle_content_header(tab_name: str):
    if tab_name in {"data-analysis", "pv-forecasting", "cooling-demand"}:
        return {"display": "none"}
    return None


@callback(Output("context-strip", "style"), Input("app-tabs", "value"))
def toggle_context_strip(tab_name: str):
    if tab_name in {"data-analysis", "pv-forecasting", "cooling-demand"}:
        return {"display": "none"}
    return None


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    APP_URL = f"http://{host}:{port}/"

    def open_mozilla_firefox() -> None:
        """Open the local Dash app in Firefox after the server starts."""
        candidates = [
            shutil.which("firefox"),
            str(Path(os.environ.get("PROGRAMFILES", "")) / "Mozilla Firefox" / "firefox.exe"),
            str(Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Mozilla Firefox" / "firefox.exe"),
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Mozilla Firefox" / "firefox.exe"),
        ]
        firefox_path = next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)
        try:
            browser = webbrowser.BackgroundBrowser(firefox_path) if firefox_path else webbrowser.get("firefox")
        except webbrowser.Error:
            browser = webbrowser.get()
        browser.open_new(APP_URL)

    if host == "127.0.0.1" and os.environ.get("RENDER") != "true":
        threading.Timer(1.2, open_mozilla_firefox).start()
    app.run(debug=debug, host=host, port=port)

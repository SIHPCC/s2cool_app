from __future__ import annotations

from dash import html


def kpi_card(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(note, className="kpi-note") if note else html.Div(),
        ],
    )


def panel(title: str, subtitle: str = "", children: list | None = None) -> html.Div:
    if title.startswith("STEP ") and " · " in title:
        title = title.split(" · ", 1)[1]
    return html.Div(
        className="panel",
        children=[
            html.H2(title, className="section-title"),
            html.P(subtitle, className="section-subtitle") if subtitle else html.Div(),
            *(children or []),
        ],
    )


def pill(text: str, tone: str = "green") -> html.Span:
    class_name = f"pill pill-{tone}"
    return html.Span(text, className=class_name)


def info_rows(rows: list[tuple[str, str]]) -> html.Div:
    return html.Div(
        className="info-list",
        children=[
            html.Div(
                className="info-row",
                children=[html.Span(label, className="muted"), html.Strong(value)],
            )
            for label, value in rows
        ],
    )


def simple_table(headers: list[str], rows: list[list[str]]) -> html.Table:
    return html.Table(
        className="data-table",
        children=[
            html.Thead(html.Tr([html.Th(header) for header in headers])),
            html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows]),
        ],
    )

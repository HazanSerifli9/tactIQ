import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
from utils.wyscout_loader import load_wyscout_team_averages

dash.register_page(__name__, path="/wyscout", title="TactIQ | Wyscout Stats")

# Columns to display in the comparison table
DISPLAY_COLS = [
    ("xg_for",                 "xG For",              True),
    ("goals_conceded",         "Goals Conceded",       False),
    ("ppda",                   "PPDA",                 False),
    ("pass_accuracy_pct",      "Pass Acc. %",          True),
    ("shots_total",            "Shots",                True),
    ("shots_on_target",        "Shots on Target",      True),
    ("interceptions",          "Interceptions",        True),
    ("clearances",             "Clearances",           True),
    ("progressive_passes",     "Progressive Passes",   True),
    ("smart_passes",           "Smart Passes",         True),
    ("match_tempo",            "Match Tempo",          True),
    ("avg_passes_per_poss",    "Passes / Possession",  True),
]

CHART_OPTIONS = [
    {"label": "xG For",              "value": "xg_for"},
    {"label": "Goals Conceded",      "value": "goals_conceded"},
    {"label": "PPDA",                "value": "ppda"},
    {"label": "Pass Accuracy %",     "value": "pass_accuracy_pct"},
    {"label": "Shots",               "value": "shots_total"},
    {"label": "Shots on Target",     "value": "shots_on_target"},
    {"label": "Interceptions",       "value": "interceptions"},
    {"label": "Progressive Passes",  "value": "progressive_passes"},
    {"label": "Smart Passes",        "value": "smart_passes"},
    {"label": "Match Tempo",         "value": "match_tempo"},
]

_CARD = {"background": "var(--card-bg, #1a1a2e)", "border": "1px solid var(--border-color, #2a2a4a)",
         "borderRadius": "12px", "padding": "20px"}


def _round2(v):
    try:
        return round(float(v), 2)
    except Exception:
        return "-"


def layout():
    df = load_wyscout_team_averages()

    if df.empty:
        return html.Div(
            html.P("No Wyscout data found in raw_data/Wyscout/.", style={"color": "var(--text-secondary)"}),
            style={"padding": "60px", "textAlign": "center"},
        )

    teams = df["wyscout_team"].tolist()

    return html.Div(
        className="container",
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
        children=[
            # ── Header ──────────────────────────────────────────────────────
            html.Header(
                style={"textAlign": "center", "marginBottom": "50px"},
                children=[
                    html.Div("WYSCOUT · 2024/25 SÜPER LİG", style={
                        "color": "var(--accent-color, #fbbf24)",
                        "fontWeight": "600",
                        "textTransform": "uppercase",
                        "letterSpacing": "2px",
                        "fontSize": "0.78rem",
                    }),
                    html.H1("Team Season Averages", style={"fontSize": "3rem", "marginBottom": "10px"}),
                    html.P(
                        f"Per-match averages from Wyscout data · {len(teams)} teams loaded",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
            ),

            # ── Bar chart ───────────────────────────────────────────────────
            html.Div(style={**_CARD, "marginBottom": "32px"}, children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
                    children=[
                        html.Label("Metric", style={
                            "color": "var(--text-secondary)",
                            "fontSize": "0.78rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "1px",
                        }),
                        dcc.Dropdown(
                            id="wyscout-metric-select",
                            options=CHART_OPTIONS,
                            value="xg_for",
                            clearable=False,
                            searchable=False,
                            className="goz-dropdown",
                            style={"minWidth": "220px"},
                        ),
                    ],
                ),
                dcc.Graph(id="wyscout-bar-chart", config={"displayModeBar": False}),
            ]),

            # ── Comparison table ────────────────────────────────────────────
            html.Div(style=_CARD, children=[
                html.Div("Season averages per match", style={
                    "fontSize": "0.75rem",
                    "color": "var(--text-secondary)",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginBottom": "16px",
                }),
                html.Div(
                    style={"overflowX": "auto"},
                    children=[
                        html.Table(
                            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"},
                            children=[
                                html.Thead(
                                    html.Tr([
                                        html.Th("Team", style=_th_style(left=True)),
                                        html.Th("Matches", style=_th_style()),
                                    ] + [
                                        html.Th(label, style=_th_style())
                                        for _, label, _ in DISPLAY_COLS
                                    ]),
                                ),
                                html.Tbody([
                                    _team_row(df[df["wyscout_team"] == team].iloc[0])
                                    for team in sorted(teams)
                                    if not df[df["wyscout_team"] == team].empty
                                ]),
                            ],
                        )
                    ],
                ),
            ]),
        ],
    )


def _th_style(left=False):
    return {
        "padding": "10px 14px",
        "textAlign": "left" if left else "center",
        "color": "var(--accent-color, #fbbf24)",
        "fontWeight": "600",
        "fontSize": "0.72rem",
        "textTransform": "uppercase",
        "letterSpacing": "0.8px",
        "borderBottom": "1px solid var(--border-color, #2a2a4a)",
        "whiteSpace": "nowrap",
    }


def _td_style(highlight=False, left=False):
    return {
        "padding": "10px 14px",
        "textAlign": "left" if left else "center",
        "borderBottom": "1px solid rgba(255,255,255,0.05)",
        "color": "#fbbf24" if highlight else "var(--text-primary, #e2e8f0)",
        "fontWeight": "600" if highlight else "400",
    }


def _team_row(row):
    highlight = str(row["wyscout_team"]) == "Göztepe"
    cells = [
        html.Td(row["wyscout_team"], style=_td_style(highlight, left=True)),
        html.Td(int(row["n_matches"]), style=_td_style(highlight)),
    ]
    for col, _, _ in DISPLAY_COLS:
        val = _round2(row.get(col, "-"))
        cells.append(html.Td(val, style=_td_style(highlight)))
    return html.Tr(cells)


# ── Callback ────────────────────────────────────────────────────────────────

@callback(
    Output("wyscout-bar-chart", "figure"),
    Input("wyscout-metric-select", "value"),
)
def update_chart(metric):
    df = load_wyscout_team_averages()
    if df.empty or metric not in df.columns:
        return go.Figure()

    label = next((o["label"] for o in CHART_OPTIONS if o["value"] == metric), metric)

    plot_df = df[["wyscout_team", metric]].dropna().copy()
    plot_df[metric] = plot_df[metric].astype(float)
    plot_df = plot_df.sort_values(metric, ascending=True)

    colors = [
        "#fbbf24" if t == "Göztepe" else "rgba(255,255,255,0.25)"
        for t in plot_df["wyscout_team"]
    ]

    fig = go.Figure(go.Bar(
        x=plot_df[metric],
        y=plot_df["wyscout_team"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{label} — per match average", font=dict(size=13, color="#aaa"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=10, r=30, t=40, b=10),
        height=500,
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig

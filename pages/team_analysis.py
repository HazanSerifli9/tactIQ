import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
from utils.stats_visuals import (
    generate_team_ball_winning_plot,
    generate_team_common_zonal_actions_plot,
    generate_team_threat_creation_plot,
    generate_team_space_allowed_plot,
    generate_tactical_style_scatter,
    generate_style_metrics_table,
)

dash.register_page(__name__, path='/team-analysis', title='TactIQ | Team Analysis')

_IMG_STYLE = {
    "maxWidth": "100%", "height": "auto",
    "borderRadius": "12px",
    "boxShadow": "0 8px 32px rgba(0,0,0,0.4)",
    "border": "1px solid #374151",
}

_LOADING_STYLE = {
    "padding": "40px", "textAlign": "center",
    "color": "#6b7280", "border": "1px dashed #374151",
    "borderRadius": "12px",
}

_SECTION_STYLE = {"marginBottom": "80px", "textAlign": "center"}


def _plot_card(title, subtitle, chart_id, margin_bottom="80px"):
    return html.Div([
        html.H3(title, style={
            "textAlign": "center", "marginBottom": "6px",
            "color": "white", "fontSize": "1.4rem", "fontWeight": "700",
        }),
        html.P(subtitle, style={
            "textAlign": "center", "color": "#9ca3af",
            "fontSize": "0.9rem", "marginBottom": "20px",
        }),
        dcc.Loading(
            html.Div(id=chart_id),
            type="circle",
            color="#FDE636",
        ),
    ], style={"marginBottom": margin_bottom, "textAlign": "center"})


def layout():
    return html.Div([
        dcc.Interval(id="ta-load-trigger", interval=1, max_intervals=1),

        html.Header([
            html.Div("Tactical Analysis", style={
                "color": "#FDE636", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "3px",
                "marginBottom": "10px", "fontSize": "0.85rem",
            }),
            html.H1("Team Tactical Insights", style={
                "fontSize": "2.5rem", "marginBottom": "12px", "textAlign": "center",
            }),
            html.P(
                "Tactical style map and performance metrics for Süper Lig teams using the Stephanatos methodology",
                style={"color": "#9ca3af", "textAlign": "center",
                       "fontSize": "1rem", "marginBottom": "10px"},
            ),
            dcc.Link("← Back to Home", href="/", style={
                "color": "#6b7280", "display": "block",
                "textAlign": "center", "marginBottom": "30px",
            }),
        ]),

        html.Div([
            html.Div([
                html.Span("★ NEW", style={
                    "background": "#FDE636", "color": "#111827",
                    "fontSize": "0.7rem", "fontWeight": "800",
                    "padding": "2px 8px", "borderRadius": "4px",
                    "marginRight": "10px", "verticalAlign": "middle",
                }),
                html.Span("Stephanatos Tactical Style Analysis",
                          style={"verticalAlign": "middle"}),
            ], style={
                "textAlign": "center", "color": "#FDE636",
                "fontSize": "0.8rem", "fontWeight": "700",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "40px",
            }),
        ]),

        _plot_card(
            "Tactical Style Map",
            "X-axis: Possession quality (Direct → Controlled) | "
            "Y-axis: Defensive territory (Deep Block → High Press). "
            "Each team's style is calculated from data normalised against their league position.",
            "ta-style-scatter",
            margin_bottom="60px",
        ),

        _plot_card(
            "Tactical Style Metrics — League Percentile Table",
            "Each column shows a 0-100 percentile score (green = top of league, red = bottom). "
            "Left group: In-possession metrics | Right group: Out-of-possession (defensive) metrics.",
            "ta-style-table",
            margin_bottom="80px",
        ),

        html.Hr(style={"borderColor": "#374151", "marginBottom": "60px"}),

        html.H2("Detailed Team Analyses", style={
            "textAlign": "center", "color": "white",
            "fontSize": "1.8rem", "marginBottom": "40px",
        }),

        _plot_card("Team Ball Winning Zones",
                   "Average pitch zone where each team wins the ball back — ranked top to bottom",
                   "ta-ball-winning"),

        _plot_card("Most Common Team Actions by Zone",
                   "Colour map of the most frequent action per zone across the pitch",
                   "ta-zonal-actions"),

        _plot_card("Team Threat Creation Zones (xT)",
                   "Zones where teams generate the most threat — ranked by xT/90",
                   "ta-threat-creation"),

        _plot_card("Space Allowed vs League Average",
                   "How much space opponents find at pass reception — delta vs. league average",
                   "ta-space-allowed",
                   margin_bottom="40px"),

    ], className="container", style={
        "maxWidth": "1400px", "margin": "0 auto", "padding": "20px 30px",
    })


def _img_or_placeholder(src):
    if src:
        return html.Img(src=src, style=_IMG_STYLE)
    return html.Div("Could not load chart.", style=_LOADING_STYLE)


@callback(Output("ta-style-scatter", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_style_scatter(_):
    try:
        return _img_or_placeholder(generate_tactical_style_scatter())
    except Exception as e:
        print(f"Error generating style scatter: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)


@callback(Output("ta-style-table", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_style_table(_):
    try:
        return _img_or_placeholder(generate_style_metrics_table())
    except Exception as e:
        print(f"Error generating style metrics table: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)


@callback(Output("ta-ball-winning", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_ball_winning(_):
    try:
        return _img_or_placeholder(generate_team_ball_winning_plot())
    except Exception as e:
        print(f"Error generating ball winning plot: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)


@callback(Output("ta-zonal-actions", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_zonal_actions(_):
    try:
        return _img_or_placeholder(generate_team_common_zonal_actions_plot())
    except Exception as e:
        print(f"Error generating zonal actions plot: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)


@callback(Output("ta-threat-creation", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_threat_creation(_):
    try:
        return _img_or_placeholder(generate_team_threat_creation_plot())
    except Exception as e:
        print(f"Error generating team threat creation plot: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)


@callback(Output("ta-space-allowed", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_space_allowed(_):
    try:
        return _img_or_placeholder(generate_team_space_allowed_plot())
    except Exception as e:
        print(f"Error generating team space allowed plot: {e}")
        return html.Div("Error loading chart.", style=_LOADING_STYLE)

import dash
from dash import html, dcc, Input, Output, callback
from utils.stats_visuals import (
    generate_team_ball_winning_plot,
    generate_team_common_zonal_actions_plot,
    generate_team_threat_creation_plot,
    generate_team_space_allowed_plot,
    generate_tactical_style_scatter,
    generate_style_metrics_table,
)

dash.register_page(__name__, path='/team-analysis', title='TactIQ | Team Analysis')


def _plot_card(title, subtitle, chart_id, *, tight=False, xtight=False):
    klass = "tq-plot-card"
    if xtight:
        klass += " tq-plot-card--xtight"
    elif tight:
        klass += " tq-plot-card--tight"
    return html.Div([
        html.H3(title, className="tq-plot-title"),
        html.P(subtitle, className="tq-plot-subtitle"),
        dcc.Loading(html.Div(id=chart_id), type="circle", color="#fbbf24"),
    ], className=klass)


def layout():
    return html.Div([
        dcc.Interval(id="ta-load-trigger", interval=1, max_intervals=1),

        html.Header([
            html.Div("Tactical Analysis", className="tq-eyebrow"),
            html.H1("Team Tactical Insights", className="tq-page-title"),
            html.P(
                "Tactical style map and performance metrics for Süper Lig teams using the Stephanatos methodology",
                className="tq-page-subtitle",
            ),
            dcc.Link("← Back to Home", href="/", className="tq-back-link"),
        ], className="tq-page-header"),

        html.Div([
            html.Span("★ NEW", className="tq-badge-new"),
            html.Span("Stephanatos Tactical Style Analysis", style={"verticalAlign": "middle"}),
        ], className="tq-badge-row"),

        _plot_card(
            "Tactical Style Map",
            "X-axis: Possession quality (Direct → Controlled) | "
            "Y-axis: Defensive territory (Deep Block → High Press). "
            "Each team's style is calculated from data normalised against their league position.",
            "ta-style-scatter",
            tight=True,
        ),

        _plot_card(
            "Tactical Style Metrics — League Percentile Table",
            "Each column shows a 0-100 percentile score (green = top of league, red = bottom). "
            "Left group: In-possession metrics | Right group: Out-of-possession (defensive) metrics.",
            "ta-style-table",
        ),

        html.Hr(className="tq-section-divider"),

        html.H2("Detailed Team Analyses", className="tq-section-title"),

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
                   xtight=True),

    ], className="tq-page")


def _img_or_placeholder(src):
    if src:
        return html.Img(src=src, className="tq-plot-img")
    return html.Div("Could not load chart.", className="tq-plot-placeholder")


@callback(Output("ta-style-scatter", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_style_scatter(_):
    try:
        return _img_or_placeholder(generate_tactical_style_scatter())
    except Exception as e:
        print(f"Error generating style scatter: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")


@callback(Output("ta-style-table", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_style_table(_):
    try:
        return _img_or_placeholder(generate_style_metrics_table())
    except Exception as e:
        print(f"Error generating style metrics table: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")


@callback(Output("ta-ball-winning", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_ball_winning(_):
    try:
        return _img_or_placeholder(generate_team_ball_winning_plot())
    except Exception as e:
        print(f"Error generating ball winning plot: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")


@callback(Output("ta-zonal-actions", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_zonal_actions(_):
    try:
        return _img_or_placeholder(generate_team_common_zonal_actions_plot())
    except Exception as e:
        print(f"Error generating zonal actions plot: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")


@callback(Output("ta-threat-creation", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_threat_creation(_):
    try:
        return _img_or_placeholder(generate_team_threat_creation_plot())
    except Exception as e:
        print(f"Error generating team threat creation plot: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")


@callback(Output("ta-space-allowed", "children"), Input("ta-load-trigger", "n_intervals"))
def _load_space_allowed(_):
    try:
        return _img_or_placeholder(generate_team_space_allowed_plot())
    except Exception as e:
        print(f"Error generating team space allowed plot: {e}")
        return html.Div("Error loading chart.", className="tq-plot-placeholder")

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.stats_visuals import (
    generate_team_ball_winning_plot, 
    generate_team_common_zonal_actions_plot,
    generate_team_cross_success_plot,
    generate_team_threat_creation_plot,
    generate_team_setpiece_concession_plot,
    generate_team_space_allowed_plot,
    generate_tactical_style_scatter,
    generate_style_metrics_table,
)

dash.register_page(__name__, path='/team-analysis', title='TactIQ | Team Analysis')

def layout():

    # ── New: Tactical Style Analyses ───────────────────────────────────
    try:
        style_scatter_img = generate_tactical_style_scatter()
    except Exception as e:
        print(f"Error generating style scatter: {e}")
        style_scatter_img = None

    try:
        style_table_img = generate_style_metrics_table()
    except Exception as e:
        print(f"Error generating style metrics table: {e}")
        style_table_img = None

    # ── Existing visualisations ─────────────────────────────────────────
    try:
        team_ball_winning_img = generate_team_ball_winning_plot()
    except Exception as e:
        print(f"Error generating ball winning plot: {e}")
        team_ball_winning_img = None

    try:
        team_zonal_actions_img = generate_team_common_zonal_actions_plot()
    except Exception as e:
        print(f"Error generating zonal actions plot: {e}")
        team_zonal_actions_img = None

    try:
        team_cross_success_img = generate_team_cross_success_plot()
    except Exception as e:
        print(f"Error generating team cross success plot: {e}")
        team_cross_success_img = None

    try:
        team_threat_creation_img = generate_team_threat_creation_plot()
    except Exception as e:
        print(f"Error generating team threat creation plot: {e}")
        team_threat_creation_img = None

    try:
        team_setpiece_concession_img = generate_team_setpiece_concession_plot()
    except Exception as e:
        team_setpiece_concession_img = None

    try:
        team_space_allowed_img = generate_team_space_allowed_plot()
    except Exception as e:
        print(f"Error generating team space allowed plot: {e}")
        team_space_allowed_img = None

    # Helper: görsel kartı oluşturur
    def plot_card(title, subtitle, img_src, margin_bottom="80px"):
        return html.Div([
            html.H3(title, style={
                "textAlign": "center", "marginBottom": "6px",
                "color": "white", "fontSize": "1.4rem", "fontWeight": "700"
            }),
            html.P(subtitle, style={
                "textAlign": "center", "color": "#9ca3af",
                "fontSize": "0.9rem", "marginBottom": "20px"
            }),
            html.Img(
                src=img_src,
                style={"maxWidth": "100%", "height": "auto",
                       "borderRadius": "12px",
                       "boxShadow": "0 8px 32px rgba(0,0,0,0.4)",
                       "border": "1px solid #374151"}
            ) if img_src else html.Div(
                "Loading data…",
                style={"padding": "40px", "textAlign": "center",
                       "color": "#6b7280", "border": "1px dashed #374151",
                       "borderRadius": "12px"}
            )
        ], style={"marginBottom": margin_bottom, "textAlign": "center"})

    # ── Page layout ────────────────────────────────────────────────────
    return html.Div([
        html.Header([
            html.Div("Tactical Analysis", style={
                "color": "#FDE636", "fontWeight": "600",
                "textTransform": "uppercase", "letterSpacing": "3px",
                "marginBottom": "10px", "fontSize": "0.85rem"
            }),
            html.H1("Team Tactical Insights", style={
                "fontSize": "2.5rem", "marginBottom": "12px", "textAlign": "center"
            }),
            html.P(
                "Tactical style map and performance metrics for Süper Lig teams using the Stephanatos methodology",
                style={"color": "#9ca3af", "textAlign": "center",
                       "fontSize": "1rem", "marginBottom": "10px"}
            ),
            dcc.Link("← Back to Home", href="/", style={
                "color": "#6b7280", "display": "block",
                "textAlign": "center", "marginBottom": "30px"
            })
        ]),

        # ── Section 1: Tactical Style Map ──────────────────────────────
        html.Div([
            html.Div([
                html.Span("★ NEW", style={
                    "background": "#FDE636", "color": "#111827",
                    "fontSize": "0.7rem", "fontWeight": "800",
                    "padding": "2px 8px", "borderRadius": "4px",
                    "marginRight": "10px", "verticalAlign": "middle"
                }),
                html.Span("Stephanatos Tactical Style Analysis",
                          style={"verticalAlign": "middle"})
            ], style={
                "textAlign": "center", "color": "#FDE636",
                "fontSize": "0.8rem", "fontWeight": "700",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "40px"
            }),
        ]),

        plot_card(
            "Tactical Style Map",
            "X-axis: Possession quality (Direct → Controlled) | "
            "Y-axis: Defensive territory (Deep Block → High Press). "
            "Each team's style is calculated from data normalised against their league position.",
            style_scatter_img,
            margin_bottom="60px"
        ),

        plot_card(
            "Tactical Style Metrics — League Percentile Table",
            "Each column shows a 0-100 percentile score (green = top of league, red = bottom). "
            "Left group: In-possession metrics | Right group: Out-of-possession (defensive) metrics.",
            style_table_img,
            margin_bottom="80px"
        ),

        # Divider
        html.Hr(style={"borderColor": "#374151", "marginBottom": "60px"}),

        # ── Section 2: Existing analyses ───────────────────────────────
        html.H2("Detailed Team Analyses", style={
            "textAlign": "center", "color": "white",
            "fontSize": "1.8rem", "marginBottom": "40px"
        }),

        plot_card("Team Ball Winning Zones",
                  "Average pitch zone where each team wins the ball back — ranked top to bottom",
                  team_ball_winning_img),

        plot_card("Most Common Team Actions by Zone",
                  "Colour map of the most frequent action per zone across the pitch",
                  team_zonal_actions_img),

        plot_card("Team Cross Effectiveness",
                  "Breakdown of crossing outcomes: Goal / Shot / Teammate / Unsuccessful",
                  team_cross_success_img),

        plot_card("Team Threat Creation Zones (xT)",
                  "Zones where teams generate the most threat — ranked by xT/90",
                  team_threat_creation_img),

        plot_card("Defending Indirect Set Pieces",
                  "Heat map of shots conceded from indirect set-piece situations",
                  team_setpiece_concession_img),

        plot_card("Space Allowed vs League Average",
                  "How much space opponents find at pass reception — delta vs. league average",
                  team_space_allowed_img, margin_bottom="40px"),

    ], className="container", style={
        "maxWidth": "1400px", "margin": "0 auto", "padding": "20px 30px"
    })



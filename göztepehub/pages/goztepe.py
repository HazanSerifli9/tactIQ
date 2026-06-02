import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.data import extract_fixture_data, calculate_standings

dash.register_page(__name__, path='/', title='TactIQ | Göztepe Hub')


def layout():
    # Calculate Göztepe PPG
    matches = extract_fixture_data(lite=True)
    df = calculate_standings(matches)
    
    ppg = 0.0
    gf_pg = 0.0
    ga_pg = 0.0
    played = 0
    if not df.empty:
        goz_df = df[df['Team'] == 'Göztepe Spor Kulübü']
        if not goz_df.empty:
            pts = goz_df['Points'].values[0]
            played = goz_df['Played'].values[0]
            if played > 0:
                ppg = round(pts / played, 2)
                gf_pg = round(goz_df['GF'].values[0] / played, 2)
                ga_pg = round(goz_df['GA'].values[0] / played, 2)

    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    # HERO
                    html.Section(
                        className="hero goztepe-hero",
                        children=[
                            html.Div(
                                className="hero-content",
                                children=[
                                    html.Img(src="/assets/goztepelogo.png", style={"height": "100px", "marginBottom": "20px", "filter": "drop-shadow(0 4px 20px rgba(251, 191, 36, 0.4))"}),
                                    html.Span("SÜPER LİG", className="section-label"),
                                    html.H1("Göztepe Hub", className="hero-title"),
                                    html.Div([
                                        html.Div(f"{ppg}", style={"fontSize": "2.5rem", "fontWeight": "bold", "color": "#fbbf24", "lineHeight": "1"}),
                                        html.Div("Points Per Game", style={"fontSize": "0.9rem", "textTransform": "uppercase", "letterSpacing": "1px"})
                                    ], style={"marginTop": "20px", "background": "rgba(0,0,0,0.5)", "padding": "15px 30px", "borderRadius": "10px", "display": "inline-block"}),
                                    
                                    html.Div(
                                        className="feature-grid",
                                        style={"marginTop": "50px", "display": "flex", "gap": "20px", "justifyContent": "center", "flexWrap": "wrap"},
                                        children=[
                                            dcc.Link(
                                                html.Div(
                                                    className="feature-card goztepe-card",
                                                    style={"cursor": "pointer", "width": "300px"},
                                                    children=[
                                                        html.Div(className="card-glow gold-glow"),
                                                        html.Div("📊", className="card-icon"),
                                                        html.Div("Pre-Match Analysis", className="card-title"),
                                                        html.Div(
                                                            "Compare Göztepe against upcoming opponents.",
                                                            className="card-desc",
                                                        ),
                                                        html.Div("VIEW →", className="card-link"),
                                                    ]
                                                ),
                                                href="/pre-match",
                                                style={"textDecoration": "none"}
                                            ),
                                            dcc.Link(
                                                html.Div(
                                                    className="feature-card goztepe-card",
                                                    style={"cursor": "pointer", "width": "300px"},
                                                    children=[
                                                        html.Div(className="card-glow gold-glow"),
                                                        html.Div("📝", className="card-icon"),
                                                        html.Div("Post-Match Analysis", className="card-title"),
                                                        html.Div(
                                                            "Detailed reports of past Göztepe matches.",
                                                            className="card-desc",
                                                        ),
                                                        html.Div("VIEW →", className="card-link"),
                                                    ]
                                                ),
                                                href="/post-match",
                                                style={"textDecoration": "none"}
                                            ),
                                            dcc.Link(
                                                html.Div(
                                                    className="feature-card goztepe-card",
                                                    style={"cursor": "pointer", "width": "300px"},
                                                    children=[
                                                        html.Div(className="card-glow gold-glow"),
                                                        html.Div("📈", className="card-icon"),
                                                        html.Div("Trends", className="card-title"),
                                                        html.Div(
                                                            "View Göztepe's form and performance trends.",
                                                            className="card-desc",
                                                        ),
                                                        html.Div("VIEW →", className="card-link"),
                                                    ]
                                                ),
                                                href="/trends",
                                                style={"textDecoration": "none"}
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            # FOOTER
            html.Footer(
                className="footer",
                children=[
                    html.Div(
                        className="footer-inner",
                        children=[
                            html.Div("© tactIQ — Precision analytics for Süper Lig.", className="footer-text"),
                            html.Img(src="/assets/superlig_logo.jpg", className="superlogo", alt="Süper Lig"),
                        ],
                    )
                ],
            ),
        ],
    )

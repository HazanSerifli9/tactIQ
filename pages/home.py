import dash
from dash import html, dcc

dash.register_page(__name__, path="/", title="TactIQ | Home")

def layout():
    superlig_logo_src = "/assets/superlig_logo.jpg"

    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    # HERO
                    html.Section(
                        className="hero",
                        children=[
                            html.Div(
                                className="hero-content",
                                children=[
                                    html.Span("ANALYTICS PLATFORM", className="section-label"),
                                    html.H1("tactIQ // SÜPER LİG", className="hero-title"),
                                    html.P(
                                        "Match intelligence, player metrics, and tactical breakdowns — built for deeper insight.",
                                        className="hero-subtitle",
                                    ),
                                    html.Div(
                                        className="feature-grid",
                                        children=[
                                            dcc.Link(
                                                className="feature-card goztepe-card",
                                                href="/team-analysis",
                                                children=[
                                                    html.Div(className="card-glow gold-glow"),
                                                    html.Div("🛡️", className="card-icon"),
                                                    html.Div("Team Analysis", className="card-title"),
                                                    html.Div(
                                                        "Team-level analysis: defensive shapes, pressing, and attacking channels.",
                                                        className="card-desc",
                                                    ),
                                                    html.Div("ENTER →", className="card-link"),
                                                ],
                                            ),
                                            dcc.Link(
                                                className="feature-card",
                                                href="/player-analysis",
                                                children=[
                                                    html.Div(className="card-glow"),
                                                    html.Div("👤", className="card-icon"),
                                                    html.Div("Player Analysis", className="card-title"),
                                                    html.Div(
                                                        "Individual production and trends: involvement, and output.",
                                                        className="card-desc",
                                                    ),
                                                    html.Div("EXPLORE →", className="card-link"),
                                                ],
                                            ),
                                            dcc.Link(
                                                className="feature-card",
                                                href="/fixtures",
                                                children=[
                                                    html.Div(className="card-glow"),
                                                    html.Div("⚽", className="card-icon"),
                                                    html.Div("Fixtures", className="card-title"),
                                                    html.Div(
                                                        "Results and quick access to game context.",
                                                        className="card-desc",
                                                    ),
                                                    html.Div("BROWSE →", className="card-link"),
                                                ],
                                            ),
                                            dcc.Link(
                                                className="feature-card",
                                                href="/standings",
                                                children=[
                                                    html.Div(className="card-glow"),
                                                    html.Div("🏆", className="card-icon"),
                                                    html.Div("Standings", className="card-title"),
                                                    html.Div(
                                                        "Live league table: points, goal difference, and form guide.",
                                                        className="card-desc",
                                                    ),
                                                    html.Div("VIEW →", className="card-link"),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),

            # FOOTER (superlogo at the end of page)
            html.Footer(
                className="footer",
                children=[
                    html.Div(
                        className="footer-inner",
                        children=[
                            html.Div("© TactIQ — Precision analytics for Süper Lig.", className="footer-text"),
                            html.Img(src=superlig_logo_src, className="superlogo", alt="Süper Lig"),
                        ],
                    )
                ],
            ),
        ],
    )
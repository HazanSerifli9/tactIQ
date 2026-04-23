import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.data import extract_fixture_data, calculate_standings
from utils.why_we_lose import calc_why_we_lose

dash.register_page(__name__, path='/', title='TactIQ | Göztepe Hub')

def _record_badge(label, rec):
    total = rec['W'] + rec['D'] + rec['L']
    win_pct = round(rec['W'] / total * 100) if total else 0
    color = '#22c55e' if win_pct >= 50 else '#ef4444'
    return html.Div([
        html.Div(label, style={"fontSize": "0.7rem", "textTransform": "uppercase",
                               "letterSpacing": "1px", "color": "#888", "marginBottom": "6px"}),
        html.Div([
            html.Span(f"W {rec['W']}", style={"color": "#22c55e", "fontWeight": "bold", "marginRight": "8px"}),
            html.Span(f"D {rec['D']}", style={"color": "#888", "fontWeight": "bold", "marginRight": "8px"}),
            html.Span(f"L {rec['L']}", style={"color": "#ef4444", "fontWeight": "bold"}),
        ], style={"fontSize": "1.1rem"}),
        html.Div(f"GF {rec['GF']}  –  GA {rec['GA']}",
                 style={"fontSize": "0.8rem", "color": "#888", "marginTop": "4px"}),
        html.Div(f"{win_pct}% win rate",
                 style={"fontSize": "0.85rem", "color": color, "fontWeight": "bold", "marginTop": "4px"}),
    ], style={"flex": "1", "textAlign": "center"})


def _build_why_we_lose():
    try:
        data = calc_why_we_lose()
    except Exception:
        return html.Div()

    hr = data['home_record']
    ar = data['away_record']
    cb = data['conceded_bands']
    sb = data['scored_bands']
    gs = data['game_state_conceded']
    asf = data['after_scoring_first']
    acf = data['after_conceding_first']

    bands = ['1-30', '31-60', '61-90', '90+']

    # ── Chart 1: Goals scored vs conceded by minute band ──
    fig_bands = go.Figure(data=[
        go.Bar(name='Goals Scored', x=bands, y=[sb[b] for b in bands],
               marker_color='#fbbf24', opacity=0.9),
        go.Bar(name='Goals Conceded', x=bands, y=[cb[b] for b in bands],
               marker_color='#ef4444', opacity=0.9),
    ])
    fig_bands.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=30),
        height=200,
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center',
                    font=dict(color='white', size=11)),
        xaxis=dict(color='#888', showgrid=False),
        yaxis=dict(color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        font=dict(color='white'),
    )

    # ── Chart 2: Game state when conceding ──
    total_conceded = sum(gs.values()) or 1
    states = ['When LEADING', 'When DRAWING', 'When TRAILING']
    vals = [gs['Leading'], gs['Drawing'], gs['Trailing']]
    colors = ['#fbbf24', '#888888', '#ef4444']
    pcts = [f"{v/total_conceded*100:.0f}%" for v in vals]

    fig_state = go.Figure(go.Bar(
        y=states, x=vals, orientation='h',
        marker_color=colors,
        text=pcts, textposition='outside',
        textfont=dict(color='white', size=12),
    ))
    fig_state.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=40, t=10, b=10),
        height=160,
        xaxis=dict(color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(color='white', showgrid=False),
        font=dict(color='white'),
    )

    # ── Score-first boxes ──
    def _first_goal_box(title, rec, icon, accent):
        played = rec['played']
        win_pct = round(rec['W'] / played * 100) if played else 0
        return html.Div([
            html.Div(icon, style={"fontSize": "1.8rem", "marginBottom": "6px"}),
            html.Div(title, style={"fontSize": "0.7rem", "textTransform": "uppercase",
                                   "letterSpacing": "1px", "color": "#888", "marginBottom": "8px"}),
            html.Div(f"{win_pct}%", style={"fontSize": "2rem", "fontWeight": "bold",
                                           "color": accent, "lineHeight": "1"}),
            html.Div("win rate", style={"fontSize": "0.75rem", "color": "#888", "marginTop": "2px"}),
            html.Div(f"W {rec['W']}  D {rec['D']}  L {rec['L']}  ({played} games)",
                     style={"fontSize": "0.75rem", "color": "#aaa", "marginTop": "8px"}),
        ], style={"flex": "1", "textAlign": "center", "padding": "16px",
                  "background": "rgba(255,255,255,0.03)", "borderRadius": "8px",
                  "border": f"1px solid {accent}33"})

    card_style = {
        "background": "rgba(14, 18, 24, 0.7)",
        "border": "1px solid rgba(255,255,255,0.08)",
        "borderRadius": "12px",
        "padding": "20px",
    }
    label_style = {
        "fontSize": "0.65rem", "textTransform": "uppercase", "letterSpacing": "2px",
        "color": "#fbbf24", "marginBottom": "14px", "fontWeight": "bold",
    }

    return html.Div([
        html.Div("WHERE WE LOSE POINTS", style={
            "textAlign": "center", "fontSize": "0.7rem", "letterSpacing": "3px",
            "textTransform": "uppercase", "color": "#ef4444", "fontWeight": "bold",
            "marginBottom": "6px",
        }),
        html.H3("Loss Pattern Analysis", style={
            "textAlign": "center", "marginBottom": "30px", "marginTop": "4px",
            "fontSize": "1.4rem",
        }),

        dbc.Row([
            # Home vs Away
            dbc.Col(html.Div([
                html.Div("HOME vs AWAY RECORD", style=label_style),
                html.Div([
                    _record_badge("AT HOME", hr),
                    html.Div(style={"width": "1px", "background": "rgba(255,255,255,0.1)",
                                    "margin": "0 16px"}),
                    _record_badge("AWAY", ar),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style=card_style), md=4, style={"marginBottom": "20px"}),

            # Goals by minute band
            dbc.Col(html.Div([
                html.Div("WHEN DO WE SCORE & CONCEDE?", style=label_style),
                html.Div("Each 30-minute block of the match",
                         style={"fontSize": "0.75rem", "color": "#888", "marginBottom": "8px"}),
                dcc.Graph(figure=fig_bands, config={'displayModeBar': False},
                          style={"height": "210px"}),
            ], style=card_style), md=8, style={"marginBottom": "20px"}),
        ]),

        dbc.Row([
            # First goal impact
            dbc.Col(html.Div([
                html.Div("FIRST GOAL IMPACT", style=label_style),
                html.Div("What happens depending on who scores first",
                         style={"fontSize": "0.75rem", "color": "#888", "marginBottom": "12px"}),
                html.Div([
                    _first_goal_box("WE Score First", asf, "⚽", "#fbbf24"),
                    html.Div(style={"width": "12px"}),
                    _first_goal_box("THEY Score First", acf, "🛡️", "#ef4444"),
                ], style={"display": "flex"}),
            ], style=card_style), md=6, style={"marginBottom": "20px"}),

            # Game state when conceding
            dbc.Col(html.Div([
                html.Div("GAME STATE WHEN WE CONCEDE", style=label_style),
                html.Div("Do we switch off after going ahead?",
                         style={"fontSize": "0.75rem", "color": "#888", "marginBottom": "8px"}),
                dcc.Graph(figure=fig_state, config={'displayModeBar': False},
                          style={"height": "170px"}),
            ], style=card_style), md=6, style={"marginBottom": "20px"}),
        ]),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "40px 20px 20px"})


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
                    _build_why_we_lose(),
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


import dash
import numpy as np
import pandas as pd
import os
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.data import extract_fixture_data, calculate_standings, get_data_dir, TEAM_LOGOS

dash.register_page(__name__, path='/post-match', title='Göztepe Hub | Post-Match')

GOZTEPE = 'Göztepe Spor Kulübü'
_SUFFIXES = ['Spor Kulübü', 'Futbol Kulübü', 'Kulübü', 'Spor A.Ş.', 'A.Ş.', 'S.K.', 'F.K.', 'SK']

def _clean(name):
    r = name
    for s in _SUFFIXES:
        r = r.replace(s, '')
    return r.strip()

PITCH_BG = "#0e1b0f"
LINE_C = "rgba(255,255,255,0.55)"
GOLD = "#fbbf24"
RED = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#a855f7"
GREEN = "rgba(34,197,94,0.9)"


def _load_goztepe_matches():
    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    matches = []
    for fn in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, fn))
            if 'team_name' in df.columns and GOZTEPE in df['team_name'].unique():
                matches.append((fn, df))
        except Exception:
            continue
    return matches


def _get_rival_last5(rival):
    matches = extract_fixture_data(lite=True)
    results = []
    for m in sorted(matches, key=lambda x: x['week'], reverse=True):
        t1, t2 = m['team_names']
        if rival not in (t1, t2):
            continue
        g1, g2 = m['stats']['team1']['goals'], m['stats']['team2']['goals']
        if rival == t1:
            opp = t2; rg, og = g1, g2
        else:
            opp = t1; rg, og = g2, g1
        if rg > og:
            res = 'W'
        elif rg < og:
            res = 'L'
        else:
            res = 'D'
        results.append({'week': m['week'], 'opp': _clean(opp), 'score': f"{rg}-{og}", 'result': res})
        if len(results) >= 5:
            break
    return list(reversed(results))


def _get_h2h_matches(rival):
    goz_matches = _load_goztepe_matches()
    h2h = []
    for fn, df in goz_matches:
        teams = df['team_name'].unique().tolist()
        if rival in teams:
            h2h.append((fn, df))
    return h2h


def _form_badge(result):
    colors = {'W': ('#22c55e', '#000'), 'D': ('#6b7280', '#fff'), 'L': ('#ef4444', '#fff')}
    bg, fg = colors.get(result, ('#6b7280', '#fff'))
    return html.Div(result, style={
        "width": "32px", "height": "32px", "borderRadius": "8px",
        "background": bg, "color": fg, "display": "flex",
        "alignItems": "center", "justifyContent": "center",
        "fontWeight": "700", "fontSize": "0.85rem",
    })


def _build_form_section(rival, opp_name):
    last5 = _get_rival_last5(rival)
    if not last5:
        return html.Div("No recent match data available.", className="goz-card-desc")
    items = []
    for m in last5:
        items.append(html.Div(style={
            "display": "flex", "flexDirection": "column", "alignItems": "center", "gap": "4px",
        }, children=[
            _form_badge(m['result']),
            html.Div(f"W{m['week']}", style={"fontSize": "0.65rem", "color": "var(--text-secondary)"}),
            html.Div(m['score'], style={"fontSize": "0.72rem", "fontWeight": "600", "color": "#fff"}),
            html.Div(m['opp'], style={
                "fontSize": "0.6rem", "color": "var(--text-secondary)",
                "maxWidth": "70px", "textAlign": "center", "overflow": "hidden",
                "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }),
        ]))
    wins = sum(1 for m in last5 if m['result'] == 'W')
    draws = sum(1 for m in last5 if m['result'] == 'D')
    losses = sum(1 for m in last5 if m['result'] == 'L')
    return html.Div(className="goz-form-section", children=[
        html.Div(className="goz-section-header", children=[
            html.Span(f"{opp_name} — Last 5 Matches", className="goz-card-title"),
        ]),
        html.Div(style={"display": "flex", "justifyContent": "center", "gap": "16px", "margin": "16px 0"}, children=items),
        html.Div(style={"textAlign": "center", "marginTop": "8px"}, children=[
            html.Span(f"{wins}W  {draws}D  {losses}L", style={
                "fontSize": "0.85rem", "fontWeight": "600", "color": "var(--text-secondary)",
                "letterSpacing": "1px",
            }),
        ]),
    ])


def _build_h2h_section(rival, opp_name):
    h2h = _get_h2h_matches(rival)
    if not h2h:
        return html.Div(className="goz-form-section", children=[
            html.Div("No head-to-head matches found.", className="goz-card-desc"),
        ])
    sections = []
    goz_short = _clean(GOZTEPE)
    for fn, df in h2h:
        week = int(df['week'].iloc[0]) if 'week' in df.columns else 0
        goz_df = df[df['team_name'] == GOZTEPE]
        opp_df = df[df['team_name'] == rival]
        # Goals
        has_og = 'own goal' in df.columns
        if has_og:
            gg = len(goz_df[(goz_df['type_id'] == 16) & (goz_df['own goal'] != 'Si')]) + len(opp_df[(opp_df['type_id'] == 16) & (opp_df['own goal'] == 'Si')])
            og = len(opp_df[(opp_df['type_id'] == 16) & (opp_df['own goal'] != 'Si')]) + len(goz_df[(goz_df['type_id'] == 16) & (goz_df['own goal'] == 'Si')])
        else:
            gg = len(goz_df[goz_df['type_id'] == 16])
            og = len(opp_df[opp_df['type_id'] == 16])
        goz_logo = TEAM_LOGOS.get(GOZTEPE, "assets/logo.png")
        opp_logo = TEAM_LOGOS.get(rival, "assets/logo.png")
        # Stats
        shot_types = [13, 14, 15, 16]
        g_shots = len(goz_df[goz_df['type_id'].isin(shot_types)])
        o_shots = len(opp_df[opp_df['type_id'].isin(shot_types)])
        g_sot = len(goz_df[goz_df['type_id'].isin([15, 16])])
        o_sot = len(opp_df[opp_df['type_id'].isin([15, 16])])
        g_passes = len(goz_df[goz_df['type_id'] == 1])
        o_passes = len(opp_df[opp_df['type_id'] == 1])
        g_succ = len(goz_df[(goz_df['type_id'] == 1) & (goz_df['outcome'] == 1)])
        o_succ = len(opp_df[(opp_df['type_id'] == 1) & (opp_df['outcome'] == 1)])
        g_acc = round(g_succ / max(g_passes, 1) * 100, 1)
        o_acc = round(o_succ / max(o_passes, 1) * 100, 1)
        g_fouls = len(goz_df[goz_df['type_id'] == 4])
        o_fouls = len(opp_df[opp_df['type_id'] == 4])
        g_corners = len(goz_df[goz_df['type_id'] == 6])
        o_corners = len(opp_df[opp_df['type_id'] == 6])
        g_xg = round(goz_df['xG'].sum(), 2) if 'xG' in goz_df.columns else 0
        o_xg = round(opp_df['xG'].sum(), 2) if 'xG' in opp_df.columns else 0

        # Shot map
        fig_shots = _make_pitch(x0=0, x1=100, height=240)
        for team_df, team_label, base_color in [(goz_df, goz_short, GOLD), (opp_df, opp_name, RED)]:
            for ev_type, color, label in [
                ('Goal', base_color, f"{team_label} Goal"),
                ('Saved Shot', BLUE if team_label == goz_short else PURPLE, f"{team_label} On Target"),
                ('Miss', "rgba(255,255,255,0.25)", f"{team_label} Off Target"),
            ]:
                grp = team_df[team_df['event'] == ev_type]
                if not grp.empty:
                    xg_vals = grp['xG'].tolist() if 'xG' in grp.columns else [0.05] * len(grp)
                    fig_shots.add_trace(go.Scatter(
                        x=grp['x'].tolist(), y=grp['y'].tolist(), mode='markers',
                        marker=dict(color=color, size=[max(7, min(v * 55, 22)) for v in xg_vals],
                                    opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.15)")),
                        name=label, showlegend=True, hoverinfo='skip',
                    ))

        # Avg positions
        fig_pos = _make_pitch(x0=0, x1=100, height=260)
        for team_df, team_name_full, color in [(goz_df, GOZTEPE, GOLD), (opp_df, rival, RED)]:
            players = team_df.groupby('player_name').agg(
                avg_x=('x', 'mean'), avg_y=('y', 'mean'), actions=('event', 'count')
            ).reset_index()
            players = players[players['actions'] >= 10].nlargest(11, 'actions')
            if not players.empty:
                jersey = {}
                if 'Jersey Number' in team_df.columns:
                    for _, r in team_df.drop_duplicates('player_name').iterrows():
                        jn = r.get('Jersey Number')
                        if pd.notna(jn):
                            jersey[r['player_name']] = str(int(jn))
                fig_pos.add_trace(go.Scatter(
                    x=players['avg_x'].tolist(), y=players['avg_y'].tolist(), mode='markers+text',
                    marker=dict(color=color, size=20, opacity=0.85, line=dict(width=1.5, color="#fff")),
                    text=[jersey.get(p, p.split()[-1][:3]) for p in players['player_name']],
                    textposition='top center', textfont=dict(size=8, color="rgba(255,255,255,0.8)"),
                    name=_clean(team_name_full), showlegend=True, hoverinfo='skip',
                ))

        def _stat_row(label, gv, ov):
            return html.Div(style={
                "display": "flex", "alignItems": "center", "padding": "6px 0",
                "borderBottom": "1px solid rgba(255,255,255,0.04)",
            }, children=[
                html.Span(str(gv), style={"flex": "1", "textAlign": "right", "fontWeight": "700",
                    "fontSize": "0.9rem", "color": GOLD, "paddingRight": "12px"}),
                html.Span(label, style={"flex": "1.5", "textAlign": "center", "fontSize": "0.72rem",
                    "color": "var(--text-secondary)", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                html.Span(str(ov), style={"flex": "1", "textAlign": "left", "fontWeight": "700",
                    "fontSize": "0.9rem", "color": RED, "paddingLeft": "12px"}),
            ])

        match_card = html.Div(className="goz-form-section", style={"marginBottom": "24px"}, children=[
            # Scoreline header
            html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "center",
                "gap": "20px", "marginBottom": "20px", "padding": "16px",
                "background": "rgba(255,255,255,0.03)", "borderRadius": "14px",
                "border": "1px solid var(--border-color)"}, children=[
                html.Img(src=f"/{goz_logo}", style={"height": "42px"}),
                html.Div(style={"textAlign": "center"}, children=[
                    html.Div(f"Week {week}", style={"fontSize": "0.7rem", "color": "var(--text-secondary)",
                        "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "4px"}),
                    html.Div(f"{gg}  –  {og}", style={"fontSize": "2rem", "fontWeight": "700",
                        "fontFamily": "'Oswald', sans-serif", "letterSpacing": "4px"}),
                    html.Div(f"{goz_short}  vs  {opp_name}", style={"fontSize": "0.8rem",
                        "color": "var(--text-secondary)", "marginTop": "2px"}),
                ]),
                html.Img(src=f"/{opp_logo}", style={"height": "42px"}),
            ]),
            # Stats table
            html.Div(style={"maxWidth": "420px", "margin": "0 auto 20px"}, children=[
                _stat_row("xG", g_xg, o_xg),
                _stat_row("Shots", g_shots, o_shots),
                _stat_row("On Target", g_sot, o_sot),
                _stat_row("Pass Accuracy", f"{g_acc}%", f"{o_acc}%"),
                _stat_row("Passes", g_passes, o_passes),
                _stat_row("Corners", g_corners, o_corners),
                _stat_row("Fouls", g_fouls, o_fouls),
            ]),
            # Visualizations
            dbc.Row([
                dbc.Col([
                    html.Div("SHOT MAP", style={"fontSize": "0.72rem", "fontWeight": "700",
                        "color": GOLD, "letterSpacing": "1px", "marginBottom": "6px", "textAlign": "center"}),
                    dcc.Graph(figure=fig_shots, config={'displayModeBar': False}, style={"height": "240px"}),
                ], md=6),
                dbc.Col([
                    html.Div("AVERAGE POSITIONS", style={"fontSize": "0.72rem", "fontWeight": "700",
                        "color": GOLD, "letterSpacing": "1px", "marginBottom": "6px", "textAlign": "center"}),
                    dcc.Graph(figure=fig_pos, config={'displayModeBar': False}, style={"height": "260px"}),
                ], md=6),
            ]),
        ])
        sections.append(match_card)

    return html.Div(children=[
        html.Div(className="goz-section-header", style={"marginBottom": "16px"}, children=[
            html.Span(f"Göztepe vs {opp_name} — Match Reports", className="goz-card-title"),
        ]),
    ] + sections)


def _make_pitch(x0=0, x1=100, height=200):
    lc = LINE_C
    lc2 = "rgba(255,255,255,0.25)"
    shapes = [dict(type="rect", x0=x0, y0=0, x1=x1, y1=100,
                   line=dict(color=lc, width=1.5), fillcolor=PITCH_BG, layer="below")]
    if x0 == 0 and x1 == 100:
        shapes.append(dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=lc, width=1)))
    if x0 == 0:
        shapes += [
            dict(type="rect", x0=0, y0=20.35, x1=15.71, y1=79.65, line=dict(color=lc, width=1), fillcolor="rgba(0,0,0,0)"),
            dict(type="rect", x0=0, y0=36.47, x1=5.24, y1=63.53, line=dict(color=lc2, width=1), fillcolor="rgba(0,0,0,0)"),
        ]
    if x1 == 100:
        shapes += [
            dict(type="rect", x0=84.29, y0=20.35, x1=100, y1=79.65, line=dict(color=lc, width=1), fillcolor="rgba(0,0,0,0)"),
            dict(type="rect", x0=94.76, y0=36.47, x1=100, y1=63.53, line=dict(color=lc2, width=1), fillcolor="rgba(0,0,0,0)"),
        ]
    fig = go.Figure()
    fig.update_layout(
        shapes=shapes, plot_bgcolor=PITCH_BG, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=2, r=2, t=2, b=2), height=height,
        xaxis=dict(range=[x0 - 1, x1 + 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-1, 101], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color="rgba(255,255,255,0.75)", size=9), bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hovermode='closest',
    )
    if x0 == 0 and x1 == 100:
        th = np.linspace(0, 2 * np.pi, 72)
        fig.add_trace(go.Scatter(x=50 + 8.71 * np.cos(th), y=50 + 13.46 * np.sin(th),
                                 mode='lines', line=dict(color=lc, width=1), showlegend=False, hoverinfo='skip'))
    return fig


def layout():
    matches = extract_fixture_data(lite=True)
    standings = calculate_standings(matches)
    rivals = sorted([t for t in standings['Team'].unique() if t != GOZTEPE])

    return html.Div(className="page-wrap", children=[
        html.Div(className="goz-hero", children=[
            html.Div(className="goz-hero-content", children=[
                dcc.Link("← GÖZTEPE HUB", href="/", className="goz-back-link"),
                html.H1("POST-MATCH ANALYSIS", className="goz-hub-title"),
                html.P("Head-to-head reports & opponent form review", className="goz-hub-subtitle"),
                html.Div(style={"marginTop": "25px", "width": "100%", "maxWidth": "350px"}, children=[
                    html.Label("SELECT OPPONENT", className="goz-label"),
                    dcc.Dropdown(
                        id='post-match-rival-selector',
                        options=[{'label': r, 'value': r} for r in rivals],
                        value=rivals[0] if rivals else None,
                        className="goz-dropdown", clearable=False,
                    ),
                ]),
            ]),
        ]),
        html.Div(className="content-container", style={"padding": "0 20px 60px"}, children=[
            html.Div(id='post-match-form-container', style={"marginTop": "30px"}),
            html.Div(id='post-match-h2h-container', style={"marginTop": "24px"}),
        ]),
        html.Footer(className="footer", children=[
            html.Div(className="footer-inner", children=[
                html.Div("© TactIQ Göztepe Hub — Precision Analytics", className="footer-text"),
                html.Img(src="/assets/superlig_logo.jpg", className="superlogo"),
            ])
        ])
    ])


@callback(
    [Output('post-match-form-container', 'children'),
     Output('post-match-h2h-container', 'children')],
    [Input('post-match-rival-selector', 'value')]
)
def update_post_match(rival):
    if not rival:
        return html.Div(), html.Div()
    opp_name = _clean(rival)
    form = _build_form_section(rival, opp_name)
    h2h = _build_h2h_section(rival, opp_name)
    return form, h2h

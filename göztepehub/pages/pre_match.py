import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data import extract_fixture_data, calculate_standings, get_match_dataframe
import utils.metrics as metrics
from göztepehub.utils import buildup_analysis
from göztepehub.utils import defensive_analysis
from göztepehub.utils import xg_chain_analysis
import utils.visuals as visuals
from göztepehub.utils.game_phases import get_phase_metrics, LOWER_IS_BETTER


def _aggregate_f3_stats(match_analyses):
    """Aggregate f3_entry_stats from a list of per-match analysis dicts."""
    buckets = []
    for m in (match_analyses or []):
        fs = m.get('f3_entry_stats')
        if not fs:
            continue
        total = fs.get('total_entries', 0)
        if total == 0:
            continue
        em = fs.get('entry_method', {})
        ez = fs.get('entry_zone', {})
        sub = fs.get('subsequent', {})
        buckets.append({
            'total': total,
            'sp': em.get('short_pass', 0),
            'dp': em.get('deep_pass', 0),
            'ca': em.get('carry', 0),
            'lf': ez.get('left', 0),
            'ce': ez.get('center', 0),
            'ri': ez.get('right', 0),
            'bc': sub.get('box_control', 0),
            'cr': sub.get('cross', 0),
            'ae': sub.get('aerial_won', 0),
        })
    if not buckets:
        return None
    total = sum(b['total'] for b in buckets)
    if total == 0:
        return None
    def pct(n):
        return round((n / total) * 100, 1)
    return {
        'total': total,
        'entry_method': {
            'short_pass_pct': pct(sum(b['sp'] for b in buckets)),
            'deep_pass_pct':  pct(sum(b['dp'] for b in buckets)),
            'carry_pct':      pct(sum(b['ca'] for b in buckets)),
        },
        'entry_zone': {
            'left_pct':   pct(sum(b['lf'] for b in buckets)),
            'center_pct': pct(sum(b['ce'] for b in buckets)),
            'right_pct':  pct(sum(b['ri'] for b in buckets)),
        },
        'subsequent': {
            'box_control_pct': pct(sum(b['bc'] for b in buckets)),
            'cross_pct':       pct(sum(b['cr'] for b in buckets)),
            'aerial_won_pct':  pct(sum(b['ae'] for b in buckets)),
        },
    }

dash.register_page(__name__, path='/pre-match', title='Göztepe Hub | Pre-Match')

GOZTEPE = 'Göztepe Spor Kulübü'

def _comparison_card(label, goz_val, opp_val, lower_is_better=False):
    try:
        def to_f(v):
            if isinstance(v, str):
                return float(v.replace('%', '').replace('/ Game', '').strip())
            return float(v)
        g_num = to_f(goz_val)
        o_num = to_f(opp_val)
        goz_better = g_num < o_num if lower_is_better else g_num > o_num
        is_equal = abs(g_num - o_num) < 0.001
        total = g_num + o_num
        g_bar_pct = round((g_num / total) * 100) if total > 0 else 50
        o_bar_pct = 100 - g_bar_pct
    except:
        goz_better = False
        is_equal = True
        g_bar_pct = 50
        o_bar_pct = 50

    g_color = "#fbbf24" if goz_better else ("rgba(255,255,255,0.5)" if is_equal else "rgba(255,255,255,0.35)")
    o_color = "#ef4444" if (not goz_better and not is_equal) else ("rgba(255,255,255,0.5)" if is_equal else "rgba(255,255,255,0.35)")
    g_bar_color = "#fbbf24" if goz_better else "rgba(255,255,255,0.15)"
    o_bar_color = "#ef4444" if (not goz_better and not is_equal) else "rgba(255,255,255,0.15)"

    return html.Div([
        html.Div(label, style={
            "fontSize": "0.68rem", "fontWeight": "600", "color": "rgba(255,255,255,0.45)",
            "textTransform": "uppercase", "letterSpacing": "0.8px", "marginBottom": "8px"
        }),
        html.Div([
            # Göztepe side
            html.Div([
                html.Span(str(goz_val), style={
                    "fontFamily": "'Oswald', sans-serif", "fontSize": "1.15rem",
                    "fontWeight": "700", "color": g_color
                }),
            ], style={"textAlign": "left", "minWidth": "52px"}),
            # Progress bar
            html.Div([
                html.Div(style={
                    "width": f"{g_bar_pct}%", "height": "6px",
                    "background": g_bar_color, "borderRadius": "3px 0 0 3px",
                    "transition": "width 0.4s ease",
                }),
                html.Div(style={
                    "width": f"{o_bar_pct}%", "height": "6px",
                    "background": o_bar_color, "borderRadius": "0 3px 3px 0",
                    "transition": "width 0.4s ease",
                }),
            ], style={"display": "flex", "flex": "1", "margin": "0 10px",
                      "background": "rgba(255,255,255,0.06)", "borderRadius": "3px", "overflow": "hidden"}),
            # Opponent side
            html.Div([
                html.Span(str(opp_val), style={
                    "fontFamily": "'Oswald', sans-serif", "fontSize": "1.15rem",
                    "fontWeight": "700", "color": o_color
                }),
            ], style={"textAlign": "right", "minWidth": "52px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "marginBottom": "8px", "padding": "10px 14px",
        "background": "rgba(255,255,255,0.02)",
        "borderRadius": "10px",
        "border": "1px solid rgba(255,255,255,0.06)",
        "transition": "background 0.2s",
    })

def layout():
    # Get list of rivals
    matches = extract_fixture_data(lite=True)
    standings = calculate_standings(matches)
    rivals = sorted([t for t in standings['Team'].unique() if t != GOZTEPE])

    tab_defs = [
        ("offensive-tab",  "⚔️",  "Offensive"),
        ("defensive-tab",  "🛡️",  "Defensive"),
        ("off-trans-tab",  "⚡",  "Off. Transition"),
        ("def-trans-tab",  "🔒",  "Def. Transition"),
    ]

    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[

                    # ── Hero / Matchup Header ──
                    html.Div(style={
                        "background": "linear-gradient(180deg, rgba(251,191,36,0.07) 0%, rgba(49,51,50,0) 100%)",
                        "borderBottom": "1px solid rgba(255,255,255,0.07)",
                        "padding": "28px 20px 32px",
                        "textAlign": "center",
                        "position": "relative",
                    }, children=[
                        # Back link
                        dcc.Link("← Göztepe Hub", href="/", style={
                            "position": "absolute", "top": "24px", "left": "28px",
                            "color": "#fbbf24", "fontSize": "0.82rem", "fontWeight": "600",
                            "textDecoration": "none", "letterSpacing": "0.3px",
                        }),

                        # Matchup row: logo — VS — opponent badge
                        html.Div([
                            # Göztepe logo
                            html.Div([
                                html.Img(src="/assets/goztepelogo.png", style={
                                    "height": "72px", "filter": "drop-shadow(0 4px 16px rgba(251,191,36,0.45))",
                                }),
                                html.Div("GÖZTEPE", style={
                                    "fontSize": "0.7rem", "fontWeight": "700", "letterSpacing": "2px",
                                    "color": "#fbbf24", "marginTop": "8px", "textTransform": "uppercase",
                                }),
                            ], style={"textAlign": "center", "flex": "1", "maxWidth": "160px"}),

                            # VS divider
                            html.Div([
                                html.Div("PRE-MATCH", style={
                                    "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "3px",
                                    "color": "rgba(255,255,255,0.3)", "marginBottom": "4px",
                                }),
                                html.Div("VS", style={
                                    "fontFamily": "'Oswald', sans-serif", "fontSize": "2.8rem",
                                    "fontWeight": "700", "color": "rgba(255,255,255,0.15)",
                                    "lineHeight": "1",
                                }),
                                html.Div("ANALYSIS", style={
                                    "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "3px",
                                    "color": "rgba(255,255,255,0.3)", "marginTop": "4px",
                                }),
                            ], style={"textAlign": "center", "flex": "0 0 auto", "padding": "0 36px"}),

                            # Opponent placeholder (reactive)
                            html.Div(id="pre-match-opp-badge", style={
                                "textAlign": "center", "flex": "1", "maxWidth": "160px",
                            }),
                        ], style={
                            "display": "flex", "alignItems": "center", "justifyContent": "center",
                            "marginBottom": "28px",
                        }),

                        # Subtitle
                        html.P("4 game phases · Season-level tactical intelligence", style={
                            "color": "rgba(255,255,255,0.35)", "fontSize": "0.82rem",
                            "margin": "0 0 24px", "letterSpacing": "0.5px",
                        }),

                        # Opponent selector
                        html.Div([
                            html.Div("SELECT UPCOMING OPPONENT", style={
                                "fontSize": "0.65rem", "fontWeight": "700", "letterSpacing": "2px",
                                "color": "rgba(255,255,255,0.4)", "marginBottom": "10px",
                            }),
                            dcc.Dropdown(
                                id='pre-match-rival-selector',
                                options=[{'label': r, 'value': r} for r in rivals],
                                value=rivals[0] if rivals else None,
                                className="goz-dropdown",
                                style={"width": "340px", "margin": "0 auto", "color": "#000"},
                                clearable=False,
                            ),
                        ]),
                    ]),

                    # ── Phase Tabs (custom pill-style) ──
                    html.Div(style={
                        "maxWidth": "1600px", "margin": "0 auto", "padding": "24px 24px 0",
                    }, children=[
                        html.Div([
                            dbc.RadioItems(
                                id="pre-match-tabs",
                                options=[
                                    {"label": html.Span([icon, " ", lbl], style={"fontSize": "0.88rem"}),
                                     "value": tid}
                                    for tid, icon, lbl in tab_defs
                                ],
                                value="offensive-tab",
                                inline=True,
                                inputClassName="pm-tab-radio-input",
                                labelClassName="pm-tab-radio-label",
                                className="pm-tab-radio-group",
                            ),
                        ], style={
                            "background": "rgba(255,255,255,0.04)",
                            "borderRadius": "14px",
                            "padding": "6px",
                            "display": "inline-flex",
                            "border": "1px solid rgba(255,255,255,0.08)",
                        }),
                    ]),

                    # ── Main two-column content ──
                    html.Div(style={
                        "maxWidth": "1600px", "margin": "0 auto", "padding": "20px 24px 60px",
                    }, children=[
                        dbc.Row([
                            # ─── Left: KPI panel ───
                            dbc.Col([
                                html.Div(id='pre-match-kpi-container'),
                            ], md=4, style={"paddingRight": "12px"}),

                            # ─── Right: Tab content ───
                            dbc.Col([
                                html.Div(id='pre-match-tab-content'),
                            ], md=8, style={"paddingLeft": "12px"}),
                        ]),
                    ]),
                ],
            ),

            # ── Footer ──
            html.Footer(className="footer", children=[
                html.Div(className="footer-inner", children=[
                    html.Div("© TactIQ — Precision analytics for Süper Lig.", className="footer-text"),
                    html.Img(src="/assets/superlig_logo.jpg", className="superlogo", alt="Süper Lig"),
                ]),
            ]),
        ],
    )

@callback(
    [Output('pre-match-kpi-container', 'children'),
     Output('pre-match-tab-content', 'children'),
     Output('pre-match-opp-badge', 'children')],
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-tabs', 'value')]
)
def update_pre_match(opponent, active_tab):
    if not opponent:
        return html.Div(), html.Div(), html.Div()

    # ── Opponent badge for the hero header ──
    opp_short = opponent.replace(' Spor Kulübü','').replace(' SK','').replace(' FK','').strip()
    opp_badge = html.Div([
        html.Div("🏟️", style={"fontSize": "2.8rem", "lineHeight": "1",
                              "filter": "grayscale(0.3)"}),
        html.Div(opp_short.upper(), style={
            "fontSize": "0.7rem", "fontWeight": "700", "letterSpacing": "2px",
            "color": "rgba(255,255,255,0.55)", "marginTop": "8px", "textTransform": "uppercase",
        }),
    ])

    # ── KPI CONTAINER ──
    current_phase_label = ("Offensive" if active_tab == "offensive-tab"
                           else "Defensive" if active_tab == "defensive-tab"
                           else "Off. Transitions" if active_tab == "off-trans-tab"
                           else "Def. Transitions")
    g_metrics_kpi = get_phase_metrics(current_phase_label, GOZTEPE)
    o_metrics_kpi = get_phase_metrics(current_phase_label, opponent)

    kpis = []
    for label, g_val in g_metrics_kpi.items():
        o_val = o_metrics_kpi.get(label, "N/A")
        kpis.append(_comparison_card(label, g_val, o_val, lower_is_better=(label in LOWER_IS_BETTER)))

    opp_display = opp_short if len(opp_short) <= 12 else opp_short[:12] + "…"

    kpi_content = html.Div(style={
        "background": "rgba(20,24,30,0.55)",
        "border": "1px solid rgba(255,255,255,0.08)",
        "borderRadius": "16px",
        "padding": "18px 16px",
    }, children=[
        # Panel header
        html.Div([
            html.Div([
                html.Div("KEY METRICS", style={
                    "fontSize": "0.62rem", "fontWeight": "700", "letterSpacing": "2px",
                    "color": "rgba(255,255,255,0.35)",
                }),
                html.Div(current_phase_label, style={
                    "fontFamily": "'Oswald', sans-serif", "fontSize": "1.1rem",
                    "fontWeight": "600", "color": "#fff", "lineHeight": "1.2",
                }),
            ]),
        ], style={"marginBottom": "14px", "paddingBottom": "12px",
                  "borderBottom": "1px solid rgba(255,255,255,0.07)"}),
        # Team labels row
        html.Div([
            html.Span("GZT", style={
                "fontSize": "0.65rem", "fontWeight": "700", "color": "#fbbf24",
                "letterSpacing": "1px",
            }),
            html.Span("vs", style={
                "fontSize": "0.6rem", "color": "rgba(255,255,255,0.25)",
                "margin": "0 auto",
            }),
            html.Span(opp_display.upper(), style={
                "fontSize": "0.65rem", "fontWeight": "700", "color": "rgba(255,100,100,0.8)",
                "letterSpacing": "1px",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "marginBottom": "8px", "padding": "0 14px",
        }),
        # Metric cards
        html.Div(kpis),
    ])

    # 3. TAB CONTENT
    tab_content = []
    
    if active_tab == "offensive-tab":
        # ── Last-match data for SEPP + trace ──
        rival_matches = buildup_analysis.get_opponent_matches_list(opponent)
        opp_buildup = goz_buildup = opp_sepp = goz_sepp = None
        fig_trace = go.Figure()
        if rival_matches:
            match_file = rival_matches[-1]['filename']
            df_last = get_match_dataframe(match_file)
            if df_last is not None:
                opp_buildup = buildup_analysis.analyze_buildup_for_match(df_last, opponent)
                goz_buildup = buildup_analysis.analyze_buildup_for_match(df_last, GOZTEPE)
                opp_sepp = metrics.calculate_sepp(df_last, opponent)
                goz_sepp = metrics.calculate_sepp(df_last, GOZTEPE)

                fig_trace.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                                    line_color="#444", fillcolor="rgba(0,0,0,0)")
                if opp_buildup and 'sequences' in opp_buildup:
                    for seq in opp_buildup['sequences'][:4]:
                        xs = [e['x'] for e in seq['events']]
                        ys = [e['y'] for e in seq['events']]
                        fig_trace.add_trace(go.Scatter(
                            x=xs, y=ys, mode='lines+markers',
                            line_width=2, marker_size=5, opacity=0.7))
                fig_trace.update_layout(
                    title=dict(text=f"{opponent} — Build-up Sequences (Last Match)", font=dict(size=12)),
                    template="plotly_dark", height=300, showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=40, b=10, l=10, r=10),
                    xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
                    yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
                )

        # ── Season-level data ──
        match_analyses_list, season_summary = buildup_analysis.get_opponent_buildup_analysis(opponent)
        xg_profile, _ = xg_chain_analysis.analyze_opponent_xg_profile(opponent)

        # F3 entry aggregated from last 5 matches
        agg_f3 = _aggregate_f3_stats((match_analyses_list or [])[-5:])

        # ── Chart: Build-up Direction (season) ──
        fig_direction = go.Figure()
        if season_summary:
            z = season_summary['zone']
            fig_direction.add_trace(go.Bar(
                x=[z['left_pct'], z['center_pct'], z['right_pct']],
                y=['Left', 'Center', 'Right'],
                orientation='h',
                marker_color=['#fbbf24', '#60a5fa', '#34d399'],
                text=[f"{z['left_pct']}%", f"{z['center_pct']}%", f"{z['right_pct']}%"],
                textposition='inside', textfont_color='#111',
            ))
        fig_direction.update_layout(
            title=dict(text=f"{opponent} — Build-up Direction (Season)", font=dict(size=12)),
            template="plotly_dark", height=200,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=10, l=60, r=10),
            xaxis=dict(showgrid=False, range=[0, 100], title="% of build-ups"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )

        # ── Chart: F3 Entry Method ──
        fig_f3_method = go.Figure()
        if agg_f3:
            em = agg_f3['entry_method']
            fig_f3_method.add_trace(go.Bar(
                x=[em['short_pass_pct'], em['deep_pass_pct'], em['carry_pct']],
                y=['Short Pass', 'Deep Pass', 'Ball Carry'],
                orientation='h',
                marker_color=['#60a5fa', '#f97316', '#a78bfa'],
                text=[f"{em['short_pass_pct']}%", f"{em['deep_pass_pct']}%", f"{em['carry_pct']}%"],
                textposition='inside', textfont_color='#111',
            ))
        fig_f3_method.update_layout(
            title="How They Enter the Final Third",
            template="plotly_dark", height=200,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=10, l=80, r=10),
            xaxis=dict(showgrid=False, range=[0, 100], title="% of entries"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )

        # ── Chart: Shot Origin ──
        fig_origins = go.Figure()
        if xg_profile:
            op = xg_profile['origin_pcts']
            labels = ['Open Play', 'From Cross', 'Set Piece', 'Fast Break', 'Through Ball']
            keys   = ['open_play', 'from_cross', 'set_piece', 'fast_break', 'through_ball']
            vals   = [op.get(k, 0) for k in keys]
            colors = ['#60a5fa', '#fbbf24', '#f472b6', '#34d399', '#a78bfa']
            fig_origins.add_trace(go.Bar(
                x=vals, y=labels, orientation='h',
                marker_color=colors,
                text=[f"{v}%" for v in vals],
                textposition='inside', textfont_color='#111',
            ))
        fig_origins.update_layout(
            title="Attack Creation — Shot Origins (Season)",
            template="plotly_dark", height=250,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=10, l=100, r=10),
            xaxis=dict(showgrid=False, range=[0, 100], title="% of shots"),
            yaxis=dict(showgrid=False),
            showlegend=False,
        )

        # ── Assemble tab content ──
        tab_content = [
            # Section 1: SEPP Comparison
            html.Div(className="goz-form-section", children=[
                html.Div(className="goz-section-header", children=[
                    html.Span("Shot Creation (SEPP)", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                ]),
                dbc.Row([
                    dbc.Col([
                        _comparison_card("Shot Creation Rate",
                            f"{goz_sepp['efficiency']}%" if goz_sepp else "—",
                            f"{opp_sepp['efficiency']}%" if opp_sepp else "—"),
                        _comparison_card("Passes per Shot",
                            goz_sepp['sepp_per_shot'] if goz_sepp else "—",
                            opp_sepp['sepp_per_shot'] if opp_sepp else "—",
                            lower_is_better=True),
                        _comparison_card("Final Third Sequences",
                            goz_sepp['sepp_f3'] if goz_sepp else "—",
                            opp_sepp['sepp_f3'] if opp_sepp else "—"),
                    ], md=5),
                    dbc.Col(dcc.Graph(figure=fig_trace, config={'displayModeBar': False}), md=7),
                ]),
            ], style={"marginBottom": "16px"}),

            # Section 2: Build-up Pattern (Season)
            html.Div(className="goz-form-section", children=[
                html.Div(className="goz-section-header", children=[
                    html.Span("Build-up Pattern — Season", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['avg_buildups_per_match']}" if season_summary else "—",
                                    className="goz-stat-number"),
                                html.Div("Avg Build-ups / Game", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['pass_type']['short_pct']}%" if season_summary else "—",
                                    className="goz-stat-number"),
                                html.Div("Short Pass %", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['pass_type']['long_pct']}%" if season_summary else "—",
                                    className="goz-stat-number"),
                                html.Div("Long Ball %", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['outcomes_15s']['f3_entry_pct']}%" if season_summary else "—",
                                    className="goz-stat-number"),
                                html.Div("Reach Final Third (15s)", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['outcomes_15s']['shot_pct']}%" if season_summary else "—",
                                    className="goz-stat-number"),
                                html.Div("Create Shot (15s)", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(
                                    f"{season_summary['outcomes_15s']['turnover_pct']}%" if season_summary else "—",
                                    className="goz-stat-number",
                                    style={"color": "#ef4444"}),
                                html.Div("Turnover %", className="goz-stat-label"),
                            ]),
                        ]),
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(figure=fig_direction, config={'displayModeBar': False}),
                    ], md=8),
                ]),
            ], style={"marginBottom": "16px"}),

            # Section 3: Final Third Entry Analysis
            html.Div(className="goz-form-section", children=[
                html.Div(className="goz-section-header", children=[
                    html.Span("Final Third Entry — How & Where (Last 5 Matches)", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_f3_method, config={'displayModeBar': False}),
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.Div("Entry Zone", className="goz-label", style={"marginBottom": "10px"}),
                            *([
                                html.Div([
                                    html.Span(lbl, style={"color": "var(--text-secondary)", "fontSize": "0.8rem", "width": "55px", "display": "inline-block"}),
                                    html.Div(style={
                                        "display": "inline-block", "height": "14px",
                                        "width": f"{agg_f3['entry_zone'][key]}%",
                                        "background": color, "borderRadius": "3px",
                                        "verticalAlign": "middle", "marginRight": "6px",
                                    }),
                                    html.Span(f"{agg_f3['entry_zone'][key]}%", style={"fontSize": "0.85rem"}),
                                ], style={"marginBottom": "10px"})
                                for lbl, key, color in [
                                    ("Left",   "left_pct",   "#fbbf24"),
                                    ("Center", "center_pct", "#60a5fa"),
                                    ("Right",  "right_pct",  "#34d399"),
                                ]
                            ] if agg_f3 else [html.Div("No data", style={"color": "var(--text-secondary)"})]),
                        ], style={"padding": "14px"}),
                        html.Div([
                            html.Div("After Entry (10s)", className="goz-label", style={"marginBottom": "10px"}),
                            *([
                                html.Div([
                                    html.Span(lbl, style={"color": "var(--text-secondary)", "fontSize": "0.8rem", "width": "90px", "display": "inline-block"}),
                                    html.Span(f"{agg_f3['subsequent'][key]}%", style={"fontSize": "0.9rem", "fontWeight": "bold", "color": color}),
                                ], style={"marginBottom": "8px"})
                                for lbl, key, color in [
                                    ("Box Control",  "box_control_pct", "#fbbf24"),
                                    ("Cross",        "cross_pct",       "#60a5fa"),
                                    ("Aerial Won",   "aerial_won_pct",  "#34d399"),
                                ]
                            ] if agg_f3 else []),
                        ], style={"padding": "14px"}),
                    ], md=6),
                ]),
            ], style={"marginBottom": "16px"}),

            # Section 4: Attack Creation Profile
            html.Div(className="goz-form-section", children=[
                html.Div(className="goz-section-header", children=[
                    html.Span("Attack Creation Profile (Season)", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_origins, config={'displayModeBar': False}),
                    ], md=7),
                    dbc.Col([
                        html.Div([
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['shots_per_game']}" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("Shots / Game", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['xg_per_shot']}" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("xG / Shot", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['xg_per_game']}" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("xG / Game", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['sot_pct']}%" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("Shots on Target %", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['zone_pcts'].get('inside_box', 0)}%" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("Shots Inside Box", className="goz-stat-label"),
                            ], style={"marginBottom": "8px"}),
                            html.Div(className="goz-stat-item", children=[
                                html.Div(f"{xg_profile['conversion_pct']}%" if xg_profile else "—", className="goz-stat-number"),
                                html.Div("Conversion Rate", className="goz-stat-label"),
                            ]),
                        ]),
                    ], md=5),
                ]),
            ]),
        ]
    
    elif active_tab == "defensive-tab":
        def_profile = defensive_analysis.get_opponent_defensive_profile(opponent)
        goz_def_profile = defensive_analysis.get_opponent_defensive_profile(GOZTEPE)
        
        if def_profile:
            flanks = def_profile.get('f3_flanks', {})
            fig_vulnerability = px.bar(
                x=list(flanks.values()), y=list(flanks.keys()),
                orientation='h', color=list(flanks.values()),
                color_continuous_scale='Reds',
                labels={'x': 'Vulnerability (%)', 'y': 'Flank'}
            )
            fig_vulnerability.update_layout(
                title=f"{opponent} - Flank Vulnerability",
                template="plotly_dark", height=250, margin=dict(t=50, b=20, l=20, r=20),
                coloraxis_showscale=False
            )
            
            tab_content = [
                html.Div(className="goz-form-section", children=[
                    html.Div(className="goz-section-header", children=[
                        html.Span("Defensive Structure", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            _comparison_card("Box Aerial Win %", f"{goz_def_profile.get('box_aerial_win_pct')}%", f"{def_profile.get('box_aerial_win_pct')}%"),
                            _comparison_card("Zone 14 Control", f"{100-goz_def_profile.get('z14_success_allowed_pct',0)}%", f"{100-def_profile.get('z14_success_allowed_pct',0)}%"),
                            _comparison_card("Avg Line Height", round(goz_def_profile.get('avg_def_line',0),1), round(def_profile.get('avg_def_line',0),1)),
                        ], md=6),
                        dbc.Col(dcc.Graph(figure=fig_vulnerability, config={'displayModeBar': False}), md=6),
                    ]),
                ], style={"marginBottom": "16px"}),
                html.Div(className="goz-form-section", children=[
                    html.Div(className="goz-section-header", children=[
                        html.Span("30s Before Conceding", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                    ]),
                    html.P("Average opponent pressure in the 30 seconds before a goal was conceded",
                           style={"fontSize": "0.8rem", "color": "var(--text-secondary)", "marginBottom": "16px"}),
                    dbc.Row([
                        dbc.Col(html.Div(className="goz-stat-item", children=[
                            html.Div(def_profile['pre_goal_summary']['avg_opp_passes_30s'], className="goz-stat-number"),
                            html.Div("Passes Allowed", className="goz-stat-label"),
                        ]), md=4),
                        dbc.Col(html.Div(className="goz-stat-item", children=[
                            html.Div(def_profile['pre_goal_summary']['avg_def_actions_30s'], className="goz-stat-number"),
                            html.Div("Defensive Actions", className="goz-stat-label"),
                        ]), md=4),
                        dbc.Col(html.Div(className="goz-stat-item", children=[
                            html.Div(def_profile['pre_goal_summary']['total_failed_clearances'],
                                     className="goz-stat-number", style={"color": "#ef4444"}),
                            html.Div("Failed Clearances", className="goz-stat-label"),
                        ]), md=4),
                    ]),
                ]),
            ]
            
    else:
        tab_content = html.Div(className="goz-form-section", children=[
            html.Div(className="goz-section-header", children=[
                html.Span("Transition Profiling", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
            ]),
            html.P("Coming soon: Detailed counter-attack and recovery maps.",
                   style={"color": "var(--text-secondary)", "textAlign": "center", "padding": "30px 0"}),
        ])

    return kpi_content, html.Div(tab_content), opp_badge

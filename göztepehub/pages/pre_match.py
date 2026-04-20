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
        is_equal = g_num == o_num
    except:
        goz_better = False
        is_equal = True

    g_color = "var(--accent-gold)" if goz_better else "var(--text-primary)"
    o_color = "var(--accent-gold)" if (not goz_better and not is_equal) else "var(--text-primary)"
    g_weight = "bold" if goz_better else "normal"
    o_weight = "bold" if (not goz_better and not is_equal) else "normal"

    return html.Div([
        html.Div(label, className="goz-label", style={"marginBottom": "8px"}),
        html.Div([
            html.Div([
                html.Div(str(goz_val), style={"color": g_color, "fontWeight": g_weight, "fontSize": "1.2rem"}),
                html.Div("Göztepe", style={"fontSize": "0.65rem", "color": "var(--text-secondary)"}),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div("VS", style={"fontSize": "0.75rem", "color": "var(--border-color)", "margin": "0 12px", "alignSelf": "center"}),
            html.Div([
                html.Div(str(opp_val), style={"color": o_color, "fontWeight": o_weight, "fontSize": "1.2rem"}),
                html.Div("Rakip", style={"fontSize": "0.65rem", "color": "var(--text-secondary)"}),
            ], style={"textAlign": "center", "flex": "1"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"marginBottom": "10px", "padding": "12px 16px",
              "background": "rgba(255,255,255,0.02)", "borderRadius": "8px",
              "border": "1px solid var(--border-color)"})

def layout():
    # Get list of rivals
    matches = extract_fixture_data(lite=True)
    standings = calculate_standings(matches)
    rivals = sorted([t for t in standings['Team'].unique() if t != GOZTEPE])
    
    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    # ── Page Header ──
                    html.Div(className="goz-page-header", children=[
                        dcc.Link("← Göztepe Hub", href="/", className="goz-back-link"),
                        html.H1("Pre-Match Analysis", className="goz-page-title"),
                        html.P("Tactical comparison across 4 game phases", className="goz-page-subtitle"),
                        html.Div([
                            html.Label("Upcoming Opponent", className="goz-label",
                                       style={"display": "block", "marginBottom": "6px", "textAlign": "center"}),
                            dcc.Dropdown(
                                id='pre-match-rival-selector',
                                options=[{'label': r, 'value': r} for r in rivals],
                                value=rivals[0] if rivals else None,
                                className="goz-dropdown",
                                style={"width": "320px", "margin": "0 auto", "color": "#000"},
                            ),
                        ], style={"marginTop": "20px"}),
                    ]),

                    # ── Main content ──
                    html.Div(style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 20px 40px"}, children=[
                        dbc.Row([
                            # Left: Radar + KPIs
                            dbc.Col([
                                html.Div(id='pre-match-kpi-container'),
                            ], md=4),

                            # Right: Tabs
                            dbc.Col([
                                dbc.Tabs([
                                    dbc.Tab(label="Offensive",        tab_id="offensive-tab",  label_style={"color": "var(--accent-gold)"}),
                                    dbc.Tab(label="Defensive",        tab_id="defensive-tab",  label_style={"color": "var(--accent-gold)"}),
                                    dbc.Tab(label="Off. Transitions", tab_id="off-trans-tab",  label_style={"color": "var(--accent-gold)"}),
                                    dbc.Tab(label="Def. Transitions", tab_id="def-trans-tab",  label_style={"color": "var(--accent-gold)"}),
                                ], id="pre-match-tabs", active_tab="offensive-tab",
                                   style={"marginBottom": "16px", "borderBottom": "1px solid var(--border-color)"}),
                                html.Div(id='pre-match-tab-content'),
                            ], md=8),
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
     Output('pre-match-tab-content', 'children')],
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-tabs', 'active_tab')]
)
def update_pre_match(opponent, active_tab):
    if not opponent:
        return html.Div(), html.Div()

    # KPI CONTAINER
    current_phase_label = "Offensive" if active_tab == "offensive-tab" else "Defensive" if active_tab == "defensive-tab" else "Off. Transitions" if active_tab == "off-trans-tab" else "Def. Transitions"
    g_metrics_kpi = get_phase_metrics(current_phase_label, GOZTEPE)
    o_metrics_kpi = get_phase_metrics(current_phase_label, opponent)
    
    kpis = []
    for label, g_val in g_metrics_kpi.items():
        o_val = o_metrics_kpi.get(label, "N/A")
        kpis.append(_comparison_card(label, g_val, o_val, lower_is_better=(label in LOWER_IS_BETTER)))
        
    kpi_content = html.Div(className="goz-form-section", children=[
        html.Div(className="goz-section-header", children=[
            html.Span(f"{current_phase_label} — Key Metrics",
                      style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
        ]),
        html.Div(kpis),
    ])

    # 3. TAB CONTENT
    tab_content = []
    
    if active_tab == "offensive-tab":
        rival_matches = buildup_analysis.get_opponent_matches_list(opponent)
        if rival_matches:
            match_file = rival_matches[-1]['filename']
            df = get_match_dataframe(match_file)
            opp_buildup = buildup_analysis.analyze_buildup_for_match(df, opponent)
            goz_buildup = buildup_analysis.analyze_buildup_for_match(df, GOZTEPE)
            
            opp_sepp = metrics.calculate_sepp(df, opponent)
            goz_sepp = metrics.calculate_sepp(df, GOZTEPE)
            
            # Ball Trace Plot (Opponent)
            fig_trace = go.Figure()
            fig_trace.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line_color="#444", fillcolor="rgba(0,0,0,0)")
            
            if opp_buildup and 'sequences' in opp_buildup:
                for seq in opp_buildup['sequences'][:4]: # Top 4 sequences
                    xs = [e['x'] for e in seq['events']]
                    ys = [e['y'] for e in seq['events']]
                    fig_trace.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line_width=2, marker_size=5, opacity=0.7))
            
            fig_trace.update_layout(
                title=f"{opponent} - Build-up Sequences (Last Match)",
                template="plotly_dark", height=450, showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20),
                xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
                yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False)
            )

            tab_content = [
                html.Div(className="goz-form-section", children=[
                    html.Div(className="goz-section-header", children=[
                        html.Span("Shot Creation (SEPP)", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            _comparison_card("Shot Creation Rate", f"{goz_sepp['efficiency']}%", f"{opp_sepp['efficiency']}%"),
                            _comparison_card("Passes per Shot", goz_sepp['sepp_per_shot'], opp_sepp['sepp_per_shot'], lower_is_better=True),
                            _comparison_card("Final Third Sequences", goz_sepp['sepp_f3'], opp_sepp['sepp_f3']),
                        ], md=5),
                        dbc.Col(dcc.Graph(figure=fig_trace, config={'displayModeBar': False}), md=7),
                    ]),
                ], style={"marginBottom": "16px"}),
                html.Div(className="goz-form-section", children=[
                    html.Div(className="goz-section-header", children=[
                        html.Span("Build-up Style", style={"fontFamily": "'Oswald', sans-serif", "fontSize": "1rem", "fontWeight": "600"}),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div("GÖZTEPE", className="goz-label", style={"marginBottom": "10px"}),
                            html.Div([
                                html.Div(f"Short Pass: {goz_buildup['pass_type']['short_pct']}%" if goz_buildup else "—", style={"marginBottom": "6px"}),
                                html.Div(f"Long Ball: {goz_buildup['pass_type']['long_pct']}%" if goz_buildup else "—", style={"marginBottom": "6px"}),
                                html.Div(f"Final Third Entry: {goz_buildup['outcomes_15s']['f3_entry_pct']}%" if goz_buildup else "—"),
                            ], style={"padding": "14px", "background": "rgba(251,191,36,0.04)",
                                      "border": "1px solid rgba(251,191,36,0.2)", "borderRadius": "10px"}),
                        ], md=6),
                        dbc.Col([
                            html.Div("RAKİP", className="goz-label", style={"marginBottom": "10px"}),
                            html.Div([
                                html.Div(f"Short Pass: {opp_buildup['pass_type']['short_pct']}%" if opp_buildup else "—", style={"marginBottom": "6px"}),
                                html.Div(f"Long Ball: {opp_buildup['pass_type']['long_pct']}%" if opp_buildup else "—", style={"marginBottom": "6px"}),
                                html.Div(f"Final Third Entry: {opp_buildup['outcomes_15s']['f3_entry_pct']}%" if opp_buildup else "—"),
                            ], style={"padding": "14px", "background": "rgba(239,68,68,0.04)",
                                      "border": "1px solid rgba(239,68,68,0.2)", "borderRadius": "10px"}),
                        ], md=6),
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

    return kpi_content, html.Div(tab_content)

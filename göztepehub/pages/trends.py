import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data, calculate_standings
from göztepehub.utils.why_we_lose import calc_why_we_lose
import plotly.graph_objects as go

dash.register_page(__name__, path='/trends', title='Göztepe Hub | Trends')


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


def _build_loss_pattern_block(team_name, accent_color="#fbbf24"):
    try:
        data = calc_why_we_lose(team_name)
    except Exception:
        return html.Div(f"Data not available for {team_name}.", style={"textAlign": "center", "color": "#888", "padding": "40px"})

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
               marker_color=accent_color, opacity=0.9),
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
    colors = [accent_color, '#888888', '#ef4444']
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

    short_team = team_name.replace(" Spor Kulübü", "").replace(" Jimnastik Kulübü", "").replace(" Futbol Kulübü", "")

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
        "color": accent_color, "marginBottom": "14px", "fontWeight": "bold",
    }

    return html.Div([
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
                html.Div(f"WHEN DOES {short_team.upper()} SCORE & CONCEDE?", style=label_style),
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
                    _first_goal_box(f"{short_team} Scores First", asf, "⚽", accent_color),
                    html.Div(style={"width": "12px"}),
                    _first_goal_box("Opponent Scores First", acf, "🛡️", "#ef4444"),
                ], style={"display": "flex"}),
            ], style=card_style), md=6, style={"marginBottom": "20px"}),

            # Game state when conceding
            dbc.Col(html.Div([
                html.Div(f"GAME STATE WHEN {short_team.upper()} CONCEDES", style=label_style),
                html.Div("Do they switch off after going ahead?",
                         style={"fontSize": "0.75rem", "color": "#888", "marginBottom": "8px"}),
                dcc.Graph(figure=fig_state, config={'displayModeBar': False},
                          style={"height": "170px"}),
            ], style=card_style), md=6, style={"marginBottom": "20px"}),
        ]),
    ])


def _build_why_we_lose(compare_team=None):
    if compare_team and compare_team != 'None':
        return html.Div([
            html.Div("GÖZTEPE SPOR KULÜBÜ", style={"textAlign": "center", "fontSize": "0.85rem", "fontWeight": "bold", "color": "#fbbf24", "letterSpacing": "2px", "marginBottom": "16px"}),
            _build_loss_pattern_block("Göztepe Spor Kulübü", "#fbbf24"),
            
            html.Hr(style={"borderTop": "1px solid rgba(255,255,255,0.1)", "margin": "30px 0"}),
            
            html.Div(compare_team.upper(), style={"textAlign": "center", "fontSize": "0.85rem", "fontWeight": "bold", "color": "#0ea5e9", "letterSpacing": "2px", "marginBottom": "16px"}),
            _build_loss_pattern_block(compare_team, "#0ea5e9")
        ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "20px"})
    
    return html.Div([
        _build_loss_pattern_block("Göztepe Spor Kulübü", "#fbbf24")
    ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "20px"})


def _build_home_away_tab(compare_team=None):
    try:
        data = calc_why_we_lose('Göztepe Spor Kulübü')
    except Exception:
        return html.Div("Data not available.", style={"textAlign": "center", "color": "#888", "padding": "40px"})

    hr = data['home_record']
    ar = data['away_record']

    comp_data = None
    if compare_team and compare_team != 'None':
        try:
            comp_data = calc_why_we_lose(compare_team)
        except Exception:
            pass

    def ppg(rec):
        total = rec['W'] + rec['D'] + rec['L']
        if total == 0:
            return 0.0
        return round((rec['W'] * 3 + rec['D']) / total, 2)

    def win_pct(rec):
        total = rec['W'] + rec['D'] + rec['L']
        return round(rec['W'] / total * 100) if total else 0

    # ── Big stat comparison ──
    card_s = {"flex": "1", "minWidth": "140px", "textAlign": "center",
              "background": "rgba(14,18,24,0.8)", "border": "1px solid rgba(255,255,255,0.1)",
              "borderRadius": "12px", "padding": "20px 14px"}

    def stat_col(label, h_val, a_val, h_color="#fbbf24", a_color="white"):
        return html.Div([
            html.Div(label, style={"fontSize": "0.65rem", "color": "#666",
                                   "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    html.Div("HOME", style={"fontSize": "0.6rem", "color": "#888", "marginBottom": "4px"}),
                    html.Div(str(h_val), style={"fontSize": "1.8rem", "fontWeight": "bold", "color": h_color}),
                ], style={"flex": "1"}),
                html.Div("vs", style={"color": "#444", "alignSelf": "center", "margin": "0 8px"}),
                html.Div([
                    html.Div("AWAY", style={"fontSize": "0.6rem", "color": "#888", "marginBottom": "4px"}),
                    html.Div(str(a_val), style={"fontSize": "1.8rem", "fontWeight": "bold", "color": a_color}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "justifyContent": "center", "alignItems": "center"}),
        ], style=card_s)

    h_ppg = ppg(hr)
    a_ppg = ppg(ar)
    ppg_color_h = "#22c55e" if h_ppg > a_ppg else "#ef4444"
    ppg_color_a = "#22c55e" if a_ppg > h_ppg else "#ef4444"

    # ── Per-match goals bar chart ──
    h_total = hr['W'] + hr['D'] + hr['L']
    a_total = ar['W'] + ar['D'] + ar['L']
    h_gf_pg = round(hr['GF'] / h_total, 2) if h_total else 0
    h_ga_pg = round(hr['GA'] / h_total, 2) if h_total else 0
    a_gf_pg = round(ar['GF'] / a_total, 2) if a_total else 0
    a_ga_pg = round(ar['GA'] / a_total, 2) if a_total else 0

    if comp_data:
        chr = comp_data['home_record']
        car = comp_data['away_record']
        ch_total = chr['W'] + chr['D'] + chr['L']
        ca_total = car['W'] + car['D'] + car['L']
        ch_gf_pg = round(chr['GF'] / ch_total, 2) if ch_total else 0
        ch_ga_pg = round(chr['GA'] / ch_total, 2) if ch_total else 0
        ca_gf_pg = round(car['GF'] / ca_total, 2) if ca_total else 0
        ca_ga_pg = round(car['GA'] / ca_total, 2) if ca_total else 0

        short_comp = compare_team.replace(" Spor Kulübü", "").replace(" Jimnastik Kulübü", "").replace(" Futbol Kulübü", "")[:10]
        x_labels = ['Göztepe (H)', 'Göztepe (A)', f'{short_comp} (H)', f'{short_comp} (A)']
        
        y_gf = [h_gf_pg, a_gf_pg, ch_gf_pg, ca_gf_pg]
        y_ga = [h_ga_pg, a_ga_pg, ch_ga_pg, ca_ga_pg]
        
        y_won = [hr['W'], ar['W'], chr['W'], car['W']]
        y_drawn = [hr['D'], ar['D'], chr['D'], car['D']]
        y_lost = [hr['L'], ar['L'], chr['L'], car['L']]
    else:
        x_labels = ['Home', 'Away']
        y_gf = [h_gf_pg, a_gf_pg]
        y_ga = [h_ga_pg, a_ga_pg]
        y_won = [hr['W'], ar['W']]
        y_drawn = [hr['D'], ar['D']]
        y_lost = [hr['L'], ar['L']]

    fig_goals = go.Figure(data=[
        go.Bar(name='Goals Scored / Game', x=x_labels, y=y_gf,
               marker_color='#fbbf24', opacity=0.9),
        go.Bar(name='Goals Conceded / Game', x=x_labels, y=y_ga,
               marker_color='#ef4444', opacity=0.9),
    ])
    fig_goals.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=30), height=220,
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                    font=dict(color='white', size=11)),
        xaxis=dict(color='white', showgrid=False),
        yaxis=dict(color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        font=dict(color='white'),
    )

    # ── W/D/L stacked bar ──
    fig_record = go.Figure(data=[
        go.Bar(name='Won', x=x_labels, y=y_won,
               marker_color='#22c55e', opacity=0.9),
        go.Bar(name='Drawn', x=x_labels, y=y_drawn,
               marker_color='#888888', opacity=0.9),
        go.Bar(name='Lost', x=x_labels, y=y_lost,
               marker_color='#ef4444', opacity=0.9),
    ])
    fig_record.update_layout(
        barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=30), height=220,
        legend=dict(orientation='h', y=-0.3, x=0.5, xanchor='center',
                    font=dict(color='white', size=11)),
        xaxis=dict(color='white', showgrid=False),
        yaxis=dict(color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        font=dict(color='white'),
    )

    card_box = {"background": "rgba(14,18,24,0.7)", "border": "1px solid rgba(255,255,255,0.08)",
                "borderRadius": "12px", "padding": "20px"}
    lbl = {"fontSize": "0.65rem", "textTransform": "uppercase", "letterSpacing": "2px",
           "color": "#fbbf24", "fontWeight": "bold", "marginBottom": "14px"}

    goz_pills = html.Div([
        stat_col("Points Per Game", h_ppg, a_ppg, ppg_color_h, ppg_color_a),
        stat_col("Win %", f"{win_pct(hr)}%", f"{win_pct(ar)}%"),
        stat_col("Goals For", hr['GF'], ar['GF']),
        stat_col("Goals Against", hr['GA'], ar['GA']),
        stat_col("Games Played", h_total, a_total),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
              "marginBottom": "24px", "justifyContent": "center"})

    pills_container = [
        html.Div("GÖZTEPE SPLIT", style={"fontSize": "0.75rem", "fontWeight": "bold", "color": "#fbbf24", "textAlign": "center", "marginBottom": "10px", "letterSpacing": "1px"}),
        goz_pills
    ]

    if comp_data:
        ch_ppg = ppg(chr)
        ca_ppg = ppg(car)
        cppg_color_h = "#22c55e" if ch_ppg > ca_ppg else "#ef4444"
        cppg_color_a = "#22c55e" if ca_ppg > ch_ppg else "#ef4444"

        comp_pills = html.Div([
            stat_col("Points Per Game", ch_ppg, ca_ppg, cppg_color_h, cppg_color_a),
            stat_col("Win %", f"{win_pct(chr)}%", f"{win_pct(car)}%"),
            stat_col("Goals For", chr['GF'], car['GF']),
            stat_col("Goals Against", chr['GA'], car['GA']),
            stat_col("Games Played", ch_total, ca_total),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "24px", "justifyContent": "center"})
        
        pills_container.extend([
            html.Div(f"{compare_team.upper()} SPLIT", style={"fontSize": "0.75rem", "fontWeight": "bold", "color": "#0ea5e9", "textAlign": "center", "marginTop": "20px", "marginBottom": "10px", "letterSpacing": "1px"}),
            comp_pills
        ])

    return html.Div([
        html.Div("HOME vs AWAY SPLIT", style={
            "textAlign": "center", "fontSize": "0.7rem", "letterSpacing": "3px",
            "textTransform": "uppercase", "color": "#ef4444", "fontWeight": "bold", "marginBottom": "6px",
        }),
        html.H3("How different is the home vs away performance?", style={
            "textAlign": "center", "marginBottom": "28px", "marginTop": "4px", "fontSize": "1.3rem",
        }),

        html.Div(pills_container),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div("RESULTS (W / D / L)", style=lbl),
                dcc.Graph(figure=fig_record, config={'displayModeBar': False}),
            ], style=card_box), md=6, style={"marginBottom": "20px"}),

            dbc.Col(html.Div([
                html.Div("GOALS PER GAME", style=lbl),
                dcc.Graph(figure=fig_goals, config={'displayModeBar': False}),
            ], style=card_box), md=6, style={"marginBottom": "20px"}),
        ]),
    ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "20px"})


def get_all_teams():
    matches = extract_fixture_data(lite=True)
    df = calculate_standings(matches)
    if not df.empty:
        teams = df['Team'].tolist()
        if 'Göztepe Spor Kulübü' in teams:
            teams.remove('Göztepe Spor Kulübü')
        return sorted(teams)
    return []

def layout():
    teams = get_all_teams()
    opts = [{'label': 'None (Göztepe Only)', 'value': 'None'}] + [{'label': t, 'value': t} for t in teams]
    
    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    html.H3("Göztepe Data Hub & Trends", style={"textAlign": "center", "marginBottom": "20px", "marginTop": "20px"}),
                    
                    html.Div([
                        html.Label("Compare With:", style={"fontWeight": "bold", "marginBottom": "10px", "marginRight": "10px", "color": "#ccc"}),
                        dcc.Dropdown(id='trends-compare-dropdown', options=opts, value='None', 
                                     style={"color": "#000", "minWidth": "300px", "display": "inline-block"})
                    ], style={"textAlign": "center", "marginBottom": "30px"}),

                    dbc.Tabs(
                        id="trends-tabs",
                        active_tab="Team Form",
                        children=[
                            dbc.Tab(label="Team Form & Trajectory", tab_id="Team Form", label_style={"color": "#fbbf24"}),
                            dbc.Tab(label="Home vs Away", tab_id="HomeAway", label_style={"color": "#ef4444"}),
                            dbc.Tab(label="Loss Pattern Analysis", tab_id="LossPatterns", label_style={"color": "#f97316"}),
                        ],
                        className="mb-4",
                        style={"borderBottom": "1px solid #333", "justifyContent": "center"}
                    ),
                    
                    html.Div(id="trends-tab-content")
                ]
            )
        ]
    )


@callback(
    Output("trends-tab-content", "children"),
    [Input("trends-tabs", "active_tab"),
     Input("trends-compare-dropdown", "value")]
)
def render_trends_tab(active_tab, compare_team):
    if active_tab == "LossPatterns":
        return _build_why_we_lose(compare_team)

    if active_tab == "HomeAway":
        return _build_home_away_tab(compare_team)

    # Default to Team Form
    return _build_team_form_tab(compare_team)


def extract_team_data(team_name, full_matches):
    """ Helper to extract time-series data for a given team """
    data = []
    
    # Needs to process in chronological order
    full_matches.sort(key=lambda x: (x['week']))
    
    for match in full_matches:
        if team_name in match['team_names']:
            stats = match.get('stats', {})
            # determine which side is team_name
            t1 = match['team_names'][0]
            if t1 == team_name:
                t_stats = stats.get('team1', {})
                opp_stats = stats.get('team2', {})
            else:
                t_stats = stats.get('team2', {})
                opp_stats = stats.get('team1', {})
                
            # If we don't have deep stats, skip it or inject 0s
            if 'shots_on_target' not in t_stats:
                continue
                
            data.append({
                'week': match['week'],
                'goals_scored': t_stats.get('goals', 0),
                'goals_conceded': opp_stats.get('goals', 0),
                'shots_on_target': t_stats.get('shots_on_target', 0),
                'pass_accuracy': t_stats.get('pass_accuracy', 0),
                'fouls': t_stats.get('fouls', 0),
                'cards': t_stats.get('cards', 0)
            })
    return data

def _build_team_form_tab(compare_team):
    compare_team = None if compare_team == 'None' else compare_team
    
    # 1. Fetch Lite Data for PPG
    lite_matches = extract_fixture_data(lite=True)
    all_weeks = sorted(list(set(m['week'] for m in lite_matches)))
    
    goz_ppg, comp_ppg = [], []
    valid_weeks_ppg = []
    
    for w in all_weeks:
         subset = [m for m in lite_matches if m['week'] <= w]
         st = calculate_standings(subset)
         
         goz = st[st['Team'] == 'Göztepe Spor Kulübü']
         pts = goz['Points'].values[0] if not goz.empty else 0
         games = goz['Played'].values[0] if not goz.empty else 0
         if games > 0:
             valid_weeks_ppg.append(w)
             goz_ppg.append(round(pts / games, 2))
             
             if compare_team:
                 comp = st[st['Team'] == compare_team]
                 c_pts = comp['Points'].values[0] if not comp.empty else 0
                 c_games = comp['Played'].values[0] if not comp.empty else 0
                 comp_ppg.append(round(c_pts / c_games, 2) if c_games > 0 else 0)

    # 2. Fetch Full Data for detailed stats
    full_matches = extract_fixture_data(lite=False)
    goz_data = extract_team_data('Göztepe Spor Kulübü', full_matches)
    
    comp_data = []
    if compare_team:
        comp_data = extract_team_data(compare_team, full_matches)

    # Helper function to generate standardized plotly figures
    def make_figure(title, y_title, metric, name1="Göztepe", name2=None, is_bar=False):
        fig = go.Figure()
        
        y1 = [d.get(metric, 0) for d in goz_data]
        x1 = [d['week'] for d in goz_data]
        
        mode = 'lines+markers' if not is_bar else 'lines'
        
        if is_bar:
             fig.add_trace(go.Bar(x=x1, y=y1, name=name1, marker_color='#fbbf24'))
        else:
             fig.add_trace(go.Scatter(x=x1, y=y1, mode=mode, name=name1, 
                                    line=dict(color='#fbbf24', width=3),
                                    marker=dict(size=8, color='#fbbf24'), fill='tozeroy', fillcolor='rgba(251, 191, 36, 0.1)'))
             
        if name2 and comp_data:
            y2 = [d.get(metric, 0) for d in comp_data]
            x2 = [d['week'] for d in comp_data]
            color2 = '#0ea5e9'
            if is_bar:
                 fig.add_trace(go.Bar(x=x2, y=y2, name=name2, marker_color=color2))
            else:
                 fig.add_trace(go.Scatter(x=x2, y=y2, mode=mode, name=name2, 
                                        line=dict(color=color2, width=3),
                                        marker=dict(size=8, color=color2)))
                 
        fig.update_layout(
             title=title, title_font=dict(color='white', size=14),
             paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
             margin=dict(l=20, r=20, t=50, b=60),
             xaxis=dict(title="Week", color='#888', showgrid=False, tickmode='linear', dtick=1),
             yaxis=dict(title=y_title, color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
             height=300, hovermode='x unified', barmode='group' if is_bar else None,
             legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(color='white', size=11))
        )
        return fig

    # Overall PPG Chart (from lite data)
    fig_ppg = go.Figure()
    fig_ppg.add_trace(go.Scatter(
         x=valid_weeks_ppg, y=goz_ppg, mode='lines+markers', name='Göztepe',
         line=dict(color='#fbbf24', width=3), marker=dict(size=8, color='#fbbf24'),
         fill='tozeroy', fillcolor='rgba(251, 191, 36, 0.1)'
    ))
    if compare_team and comp_ppg:
        fig_ppg.add_trace(go.Scatter(
             x=valid_weeks_ppg, y=comp_ppg, mode='lines+markers', name=compare_team,
             line=dict(color='#0ea5e9', width=3), marker=dict(size=8, color='#0ea5e9')
        ))
    fig_ppg.update_layout(
         title="Rolling Points Per Game (Form)", title_font=dict(color='white', size=14),
         paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', margin=dict(l=20, r=20, t=50, b=60),
         xaxis=dict(title="Week", color='#888', showgrid=False, tickmode='linear', dtick=1),
         yaxis=dict(title="PPG", color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 3.1]),
         height=300, hovermode='x unified', 
         legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(color='white', size=11))
    )

    # Attacking Form Dual Axis
    fig_attack = go.Figure()
    x_att = [d['week'] for d in goz_data]
    fig_attack.add_trace(go.Bar(x=x_att, y=[d['goals_scored'] for d in goz_data], name='Goals (Göztepe)', marker_color='#fbbf24'))
    fig_attack.add_trace(go.Scatter(x=x_att, y=[d['shots_on_target'] for d in goz_data], name='Shots on Target (Göz)', 
                                    line=dict(color='#fcd34d', width=2), mode='lines+markers'))
    if compare_team and comp_data:
        x_att2 = [d['week'] for d in comp_data]
        fig_attack.add_trace(go.Bar(x=x_att2, y=[d['goals_scored'] for d in comp_data], name=f'Goals ({compare_team})', marker_color='#0ea5e9'))
        fig_attack.add_trace(go.Scatter(x=x_att2, y=[d['shots_on_target'] for d in comp_data], name=f'Shots on Target ({compare_team})', 
                                        line=dict(color='#7dd3fc', width=2), mode='lines+markers'))
        
    fig_attack.update_layout(
             title="Attacking Efficiency (Goals & SOT)", title_font=dict(color='white', size=14),
             paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
             margin=dict(l=20, r=20, t=50, b=60),
             xaxis=dict(title="Week", color='#888', showgrid=False, tickmode='linear', dtick=1),
             yaxis=dict(title="Count", color='#888', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
             height=300, hovermode='x unified', barmode='group',
             legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(color='white', size=11))
    )

    # 3 Standard metric charts
    fig_defense = make_figure("Defensive Solidity (Goals Conceded)", "Goals", "goals_conceded", name1="Göztepe", name2=compare_team, is_bar=True)
    fig_control = make_figure("Control & Dominance (Pass Accuracy %)", "Accuracy %", "pass_accuracy", name1="Göztepe", name2=compare_team)
    fig_foul = make_figure("Aggression (Fouls Committed)", "Fouls", "fouls", name1="Göztepe", name2=compare_team, is_bar=True)

    
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_ppg, config={'displayModeBar': False}), style={"border": "1px solid #333", "borderRadius": "8px", "overflow": "hidden"}), md=12, style={"marginBottom": "20px"})
        ]),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_attack, config={'displayModeBar': False}), style={"border": "1px solid #333", "borderRadius": "8px", "overflow": "hidden"}), md=6, style={"marginBottom": "20px"}),
            dbc.Col(html.Div(dcc.Graph(figure=fig_control, config={'displayModeBar': False}), style={"border": "1px solid #333", "borderRadius": "8px", "overflow": "hidden"}), md=6, style={"marginBottom": "20px"})
        ]),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig_defense, config={'displayModeBar': False}), style={"border": "1px solid #333", "borderRadius": "8px", "overflow": "hidden"}), md=6, style={"marginBottom": "20px"}),
            dbc.Col(html.Div(dcc.Graph(figure=fig_foul, config={'displayModeBar': False}), style={"border": "1px solid #333", "borderRadius": "8px", "overflow": "hidden"}), md=6, style={"marginBottom": "20px"})
        ])
    ])

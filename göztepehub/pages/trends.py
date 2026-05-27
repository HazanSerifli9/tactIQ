import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data, calculate_standings
from göztepehub.utils.recipient_analysis import get_goztepe_players, get_recipient_analysis
from göztepehub.utils.pitch import draw_pitch
from göztepehub.utils.why_we_lose import calc_why_we_lose
import plotly.graph_objects as go
import numpy as np

dash.register_page(__name__, path='/trends', title='Göztepe Hub | Trends')


def _build_home_away_tab():
    try:
        data = calc_why_we_lose()
    except Exception:
        return html.Div("Data not available.", style={"textAlign": "center", "color": "#888", "padding": "40px"})

    hr = data['home_record']
    ar = data['away_record']

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

    fig_goals = go.Figure(data=[
        go.Bar(name='Goals Scored / Game', x=['Home', 'Away'], y=[h_gf_pg, a_gf_pg],
               marker_color='#fbbf24', opacity=0.9),
        go.Bar(name='Goals Conceded / Game', x=['Home', 'Away'], y=[h_ga_pg, a_ga_pg],
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
        go.Bar(name='Won', x=['Home', 'Away'], y=[hr['W'], ar['W']],
               marker_color='#22c55e', opacity=0.9),
        go.Bar(name='Drawn', x=['Home', 'Away'], y=[hr['D'], ar['D']],
               marker_color='#888888', opacity=0.9),
        go.Bar(name='Lost', x=['Home', 'Away'], y=[hr['L'], ar['L']],
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

    return html.Div([
        html.Div("HOME vs AWAY SPLIT", style={
            "textAlign": "center", "fontSize": "0.7rem", "letterSpacing": "3px",
            "textTransform": "uppercase", "color": "#ef4444", "fontWeight": "bold", "marginBottom": "6px",
        }),
        html.H3("How different are we at home vs away?", style={
            "textAlign": "center", "marginBottom": "28px", "marginTop": "4px", "fontSize": "1.3rem",
        }),

        # Key stat pills
        html.Div([
            stat_col("Points Per Game", h_ppg, a_ppg, ppg_color_h, ppg_color_a),
            stat_col("Win %", f"{win_pct(hr)}%", f"{win_pct(ar)}%"),
            stat_col("Goals For", hr['GF'], ar['GF']),
            stat_col("Goals Against", hr['GA'], ar['GA']),
            stat_col("Games Played", h_total, a_total),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "24px", "justifyContent": "center"}),

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
    
    # Send players
    goz_players = get_goztepe_players()
    player_opts = [{'label': p, 'value': p} for p in goz_players]
    
    return html.Div(
        className="page-wrap",
        children=[
            html.Div(
                className="page-content",
                children=[
                    html.H3("Göztepe Data Hub & Trends", style={"textAlign": "center", "marginBottom": "20px", "marginTop": "20px"}),
                    
                    dbc.Tabs(
                        id="trends-tabs",
                        active_tab="Team Form",
                        children=[
                            dbc.Tab(label="Team Form & Trajectory", tab_id="Team Form", label_style={"color": "#fbbf24"}),
                            dbc.Tab(label="Home vs Away", tab_id="HomeAway", label_style={"color": "#ef4444"}),
                            dbc.Tab(label="Pass Recipient Analysis", tab_id="Recipients", label_style={"color": "#0ea5e9"}),
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
    Input("trends-tabs", "active_tab")
)
def render_trends_tab(active_tab):
    if active_tab == "HomeAway":
        return _build_home_away_tab()

    if active_tab == "Recipients":
        goz_players = get_goztepe_players()
        p_opts = [{'label': p, 'value': p} for p in goz_players]
        # Sort so we have a reliable default
        default_p = "Y. Kayan" if "Y. Kayan" in goz_players else (goz_players[0] if goz_players else None)
        
        return html.Div([
            html.Div("Actions after receiving a pass from a specific player", style={"textAlign": "center", "color": "#888", "marginBottom": "20px"}),
            html.Div([
                html.Label("Sender (Pass From):", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id='trends-sender-dropdown', options=p_opts, value=default_p, 
                    style={"color": "#000", "minWidth": "250px", "marginBottom": "15px"}
                ),
                html.Label("Recipient (To Plot Map):", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id='trends-recipient-dropdown', options=[], value=None,
                    style={"color": "#000", "minWidth": "250px"}
                )
            ], style={"maxWidth": "400px", "margin": "0 auto 30px", "background": "#1e1e1e", "padding": "20px", "borderRadius": "8px", "border": "1px solid #333"}),
            
            dbc.Row([
                dbc.Col(html.Div(id='recipient-map-container'), md=7),
                dbc.Col([
                    html.Div(id='recipient-zones-container', style={"marginBottom": "20px"}),
                    html.Div(id='recipient-table-container')
                ], md=5)
            ])
        ])

    # Default to Team Form
    teams = get_all_teams()
    opts = [{'label': 'None (Göztepe Only)', 'value': 'None'}] + [{'label': t, 'value': t} for t in teams]
    return html.Div([
        html.Div("Compare Göztepe's form against another team over the season.", style={"textAlign": "center", "color": "#888", "marginBottom": "30px"}),
        
        html.Div([
            html.Label("Compare With:", style={"fontWeight": "bold", "marginBottom": "10px", "marginRight": "10px"}),
            dcc.Dropdown(id='trends-compare-dropdown', options=opts, value='None', 
                         style={"color": "#000", "minWidth": "300px", "display": "inline-block"})
        ], style={"textAlign": "center", "marginBottom": "40px"}),
        
        html.Div(id='trends-dashboard-content', style={"padding": "0 20px"})
    ])

# Callback for the drop downs
@callback(
    [Output('trends-recipient-dropdown', 'options'),
     Output('trends-recipient-dropdown', 'value')],
    Input('trends-sender-dropdown', 'value')
)
def update_recipient_dropdown(sender):
    if not sender:
        return [], None
    recs = get_recipient_analysis(sender)
    opts = [{'label': r['Player'], 'value': r['Player']} for r in recs]
    default_val = opts[0]['value'] if opts else None
    return opts, default_val
    

@callback(
    [Output('recipient-table-container', 'children'),
     Output('recipient-map-container', 'children'),
     Output('recipient-zones-container', 'children')],
    [Input('trends-sender-dropdown', 'value'),
     Input('trends-recipient-dropdown', 'value')]
)
def render_recipient_analysis(sender, recipient):
    if not sender:
        return html.Div(), html.Div(), html.Div()
        
    recs = get_recipient_analysis(sender)
    
    # 1. Build Table
    trs = []
    for r in recs:
        is_active = (r['Player'] == recipient)
        bg = "rgba(40, 167, 69, 0.2)" if is_active else "transparent"
        trs.append(html.Tr([
            html.Td(r['Player'], style={"fontWeight": "bold", "color": "#fff" if is_active else "#ccc"}),
            html.Td(r['Receptions'], style={"textAlign": "center"}),
            html.Td(r['PP_per_R'], style={"textAlign": "center", "color": "#22c55e" if r['PP_per_R'] > 0.15 else "#888"}),
            html.Td(r['PC_per_R'], style={"textAlign": "center", "color": "#3b82f6" if r['PC_per_R'] > 0.05 else "#888"}),
            html.Td(r['TO_Won_Att'], style={"textAlign": "center"})
        ], style={"backgroundColor": bg, "borderBottom": "1px solid #333"}))
    
    table = html.Div([
        html.Div(f"TOP RECIPIENTS | {sender.upper()}", style={"textAlign": "center", "fontWeight": "bold", "color": "#ccc", "letterSpacing": "1px", "marginBottom": "10px"}),
        html.P("Rec = Receptions | PP/R = Progressive Passes/Rec | PC/R = Prog. Carries/Rec", style={"textAlign": "center", "fontSize": "0.75rem", "color": "#888"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Player", style={"textAlign": "left", "color": "#888", "padding": "8px"}),
                html.Th("Rec", style={"textAlign": "center", "color": "#888", "padding": "8px"}),
                html.Th("PP/R", style={"textAlign": "center", "color": "#888", "padding": "8px"}),
                html.Th("PC/R", style={"textAlign": "center", "color": "#888", "padding": "8px"}),
                html.Th("TO W/Att", style={"textAlign": "center", "color": "#888", "padding": "8px"})
            ]), style={"borderBottom": "1px solid #555"}),
            html.Tbody(trs)
        ], style={"width": "100%", "marginTop": "10px", "fontSize": "0.85rem", "borderCollapse": "collapse"})
    ], style={"background": "#111", "padding": "20px", "borderRadius": "8px", "border": "1px solid #333"})

    # 2. Build Map
    fig = draw_pitch(theme='dark')
    if recipient:
        target_rec = next((r for r in recs if r['Player'] == recipient), None)
        if target_rec:
            # color mapping
            cmap = {
                "Progressive Pass": "#22c55e", # green
                "Successful Pass": "#3b82f6",  # blue
                "Unsuccessful Pass": "#555555",# grey
                "Progressive Carry": "#ef4444",# red
                "Take-On": "#a855f7",          # purple
                "Shot": "#fbbf24",             # gold
                "Other": "#888888"
            }
            
            # draw lines for actions
            for a in target_rec['actions']:
                color = cmap.get(a['type'], "#aaa")
                
                if a['type'] == "Progressive Carry":
                    mode = "lines+markers"
                    dash = "dot"
                else:
                    mode = "lines+markers"
                    dash = "solid"
                    
                fig.add_trace(go.Scatter(
                    x=[a['start_x'], a['end_x']],
                    y=[a['start_y'], a['end_y']],
                    mode=mode,
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=5, color=color),
                    showlegend=False,
                    opacity=0.7,
                    hoverinfo="text",
                    text=f"{a['type']} by {recipient}"
                ))
            
            # Add custom legend via invisible scatter traces
            for type_name, color in cmap.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    name=type_name,
                    line=dict(color=color, width=3, dash="dot" if type_name=="Progressive Carry" else "solid")
                ))

            fig.update_layout(
                title=dict(
                    text=f"<b>{recipient}</b><br>Actions after receiving from {sender}",
                    font=dict(color="white", size=16),
                    x=0.5, y=0.95, xanchor='center'
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(color="#ccc", size=10), bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(t=80, b=10, l=10, r=10)
            )

    map_container = dcc.Graph(figure=fig, config={'displayModeBar': False}, style={"height": "600px", "border": "1px solid #333", "borderRadius": "8px", "background": "#1e1e1e"})

    # 3. Build Reception Zones
    zone_container = html.Div()
    if recipient and target_rec:
        actions = target_rec['actions']
        
        # Grid settings: 3x3
        x_bins = [0, 33.3, 66.6, 100]
        y_bins = [0, 33.3, 66.6, 100]
        
        zones_data = {}
        for a in actions:
            rx, ry = a['start_x'], a['start_y']
            is_prog = (a['type'] in ["Progressive Pass", "Progressive Carry"])
            
            x_i = next((i for i, x in enumerate(x_bins[1:]) if rx <= x), 2)
            y_i = next((i for i, y in enumerate(y_bins[1:]) if ry <= y), 2)
            z_id = f"{x_i}_{y_i}"
            
            if z_id not in zones_data:
                zones_data[z_id] = {'count': 0, 'prog_count': 0, 
                                    'center_x': (x_bins[x_i]+x_bins[x_i+1])/2, 
                                    'center_y': (y_bins[y_i]+y_bins[y_i+1])/2}
                                    
            zones_data[z_id]['count'] += 1
            if is_prog:
                zones_data[z_id]['prog_count'] += 1
                
        fig_zones = draw_pitch(theme='dark')
        total_actions = len(actions)
        
        # We need numpy to calculate angles for the arcs
        import numpy as np
        
        for z_id, zd in zones_data.items():
            if zd['count'] > 0:
                prog_rate = (zd['prog_count'] / zd['count']) * 100
                
                # Scale from 0 to max_radius (e.g. 10 coords)
                r_coord = max(3, min(12, (zd['count'] / total_actions) * 15))
                cx, cy = zd['center_x'], zd['center_y']
                
                # Draw the filled bubble (Reception Count) using polygons mapped to pitch coordinates
                theta = np.linspace(0, 2*np.pi, 50)
                circle_x = cx + r_coord * np.cos(theta)
                circle_y = cy + r_coord * np.sin(theta)
                
                fig_zones.add_trace(go.Scatter(
                    x=circle_x, y=circle_y,
                    mode="lines",
                    fill="toself", fillcolor="rgba(30, 64, 175, 0.6)", # Blue filling
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                # Draw the Progression Rate Arc
                # Start at top (pi/2) and go clockwise
                start_angle = np.pi / 2
                end_angle = np.pi / 2 - (2 * np.pi * (prog_rate / 100))
                
                if prog_rate > 0:
                    t_arc = np.linspace(start_angle, end_angle, 30)
                    r_arc = r_coord * 1.05 # slightly outside the bubble
                    arc_x = cx + r_arc * np.cos(t_arc)
                    arc_y = cy + r_arc * np.sin(t_arc)
                    
                    fig_zones.add_trace(go.Scatter(
                        x=arc_x, y=arc_y,
                        mode="lines",
                        line=dict(color="#22c55e", width=4), # Green arc
                        hoverinfo="skip",
                        showlegend=False
                    ))
                
                # Text in center of bubble
                fig_zones.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode='text',
                    text=[f"<b>{zd['count']}</b>"],
                    textposition="middle center",
                    textfont=dict(color='white', size=13),
                    hoverinfo='text',
                    hovertext=f"Receptions: {zd['count']}<br>Prog. Rate: {prog_rate:.0f}%",
                    showlegend=False
                ))
                
                # Text for Progression rate below the bubble
                fig_zones.add_trace(go.Scatter(
                    x=[cx], y=[cy - r_coord - 3], # shifted down
                    mode='text',
                    text=[f"{prog_rate:.0f}%"],
                    textfont=dict(color='#22c55e', size=11, family='Arial Black'),
                    showlegend=False, hoverinfo='skip'
                ))

        team_prog = target_rec['PP_per_R'] + target_rec['PC_per_R']
        
        fig_zones.update_layout(
            title=dict(
                text="<b>RECEPTION ZONES</b><br><span style='font-size:12px;color:#888'>Where they receive and how often they progress the ball</span>",
                font=dict(color="white", size=14),
                x=0.5, y=0.92, xanchor='center'
            ),
            margin=dict(t=70, b=40, l=10, r=10),
            height=350
        )
        
        fig_zones.add_annotation(
            text=f"{zd['prog_count'] if 'zd' in locals() else 0} progressive actions from {total_actions} receptions ({team_prog*100:.0f}%)",
            xref="paper", yref="paper", x=0.5, y=-0.1, showarrow=False,
            font=dict(color="#22c55e", size=10)
        )

        zone_container = html.Div([
            dcc.Graph(figure=fig_zones, config={'displayModeBar': False}, style={"border": "1px solid #333", "borderRadius": "8px", "background": "#1e1e1e"})
        ])

    return table, map_container, zone_container


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

@callback(
    Output('trends-dashboard-content', 'children'),
    Input('trends-compare-dropdown', 'value')
)
def update_trends_dashboard(compare_team):
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

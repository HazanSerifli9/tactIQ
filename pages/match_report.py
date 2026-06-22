import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.data import get_match_dataframe
from utils.wyscout_loader import get_wyscout_match_stats
import utils.visuals as visuals
import utils.metrics as metrics
import utils.tempo_data as tempo_data
import utils.tempo_visuals as tempo_visuals

from urllib.parse import unquote

try:
    dash.register_page(__name__, path_template='/analysis/<match_id>')
except Exception:
    pass # Prevent crash if dynamically imported within a callback

def layout(match_id=None):
    if not match_id:
        return html.Div("Match ID not provided.")
    
    decoded_match_id = unquote(match_id)

    df = get_match_dataframe(decoded_match_id)
    if df is None:
        return html.Div(f"Data not found for match: {decoded_match_id}")

    teams = df['team_name'].unique()
    teams = [t for t in teams if isinstance(t, str)]
    
    if len(teams) < 2:
        return html.Div("Insufficient team data in file.")

    if 'team_position' in df.columns:
        home_team = df[df['team_position'] == 'home']['team_name'].iloc[0]
        away_team = df[df['team_position'] == 'away']['team_name'].iloc[0]
    else:
        home_team = teams[0]
        away_team = teams[1]

    def clean_name(name):
        return name.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').strip()

    home_metrics = {
        "xG": metrics.calculate_xg(df, home_team),
        "xA": metrics.calculate_xa(df, home_team),
        "PPDA": metrics.calculate_ppda(df, home_team),
        "Field Tilt": metrics.calculate_field_tilt(df, home_team),
        "xT": metrics.calculate_xt(df, home_team),
        "BDP": metrics.calculate_bdp(df, home_team),
    }

    away_metrics = {
        "xG": metrics.calculate_xg(df, away_team),
        "xA": metrics.calculate_xa(df, away_team),
        "PPDA": metrics.calculate_ppda(df, away_team),
        "Field Tilt": metrics.calculate_field_tilt(df, away_team),
        "xT": metrics.calculate_xt(df, away_team),
        "BDP": metrics.calculate_bdp(df, away_team),
    }

    # Wyscout overrides — prefer Wyscout for any metric it provides (except for Week 34 where we use custom model)
    ws = None
    has_possession_override = False
    match_week = df['week'].iloc[0] if 'week' in df.columns and not df.empty else None

    if match_week != 34:
        try:
            ws = get_wyscout_match_stats(home_team, away_team)
            if ws:
                ws_overrides = {
                    'xG':         'xg_for',
                    'PPDA':       'ppda',
                    'Field Tilt': 'possession_pct',
                }
                for metric_key, ws_key in ws_overrides.items():
                    if ws['home'].get(ws_key) is not None:
                        home_metrics[metric_key] = ws['home'][ws_key]
                        if metric_key == 'Field Tilt':
                            has_possession_override = True
                    if ws['away'].get(ws_key) is not None:
                        away_metrics[metric_key] = ws['away'][ws_key]
                        if metric_key == 'Field Tilt':
                            has_possession_override = True
        except Exception as e:
            print("Wyscout load error:", e)

    def create_headline_card(question, metric_name, h_val, a_val, note):
        return html.Div([
            html.Div(question, style={"fontSize": "0.7rem", "color": "#888", "textTransform": "uppercase",
                                      "letterSpacing": "1px", "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    html.Div(clean_name(home_team), style={"fontSize": "0.62rem", "color": "#fbbf24",
                                                           "textTransform": "uppercase", "letterSpacing": "0.5px",
                                                           "marginBottom": "2px", "fontWeight": "600"}),
                    html.Span(f"{h_val}", style={"fontSize": "2.4rem", "fontWeight": "bold", "color": "#fbbf24"}),
                ], style={"textAlign": "center"}),
                html.Span(" — ", style={"color": "#444", "margin": "0 10px", "fontSize": "1.5rem", "alignSelf": "center"}),
                html.Div([
                    html.Div(clean_name(away_team), style={"fontSize": "0.62rem", "color": "#e5e7eb",
                                                           "textTransform": "uppercase", "letterSpacing": "0.5px",
                                                           "marginBottom": "2px", "fontWeight": "600"}),
                    html.Span(f"{a_val}", style={"fontSize": "2.4rem", "fontWeight": "bold", "color": "white"}),
                ], style={"textAlign": "center"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "margin": "8px 0"}),
            html.Div(metric_name, style={"fontSize": "0.85rem", "color": "#aaa", "fontWeight": "600"}),
            html.Div(note, style={"fontSize": "0.7rem", "color": "#666", "marginTop": "4px", "lineHeight": "1.4"}),
        ], style={
            "background": "rgba(14,18,24,0.8)", "border": "1px solid rgba(255,255,255,0.12)",
            "borderRadius": "16px", "padding": "28px 20px", "textAlign": "center",
            "flex": "1", "minWidth": "200px",
        })

    field_tilt_title = "Possession %" if has_possession_override else "Territorial Control (Field Tilt %)"
    field_tilt_note = "Percentage of ball possession from Wyscout" if has_possession_override else "% of play spent in the attacking half — higher is better"

    kpi_section = html.Div([
        html.P("3 things that decide this match", style={
            "color": "#555", "textAlign": "center", "marginBottom": "10px",
            "fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "2px",
        }),
        html.Div([
            html.Span("●", style={"color": "#fbbf24", "marginRight": "4px"}),
            html.Span(clean_name(home_team), style={"color": "#fbbf24", "fontSize": "0.75rem", "fontWeight": "600", "marginRight": "20px"}),
            html.Span("●", style={"color": "#e5e7eb", "marginRight": "4px"}),
            html.Span(clean_name(away_team), style={"color": "#e5e7eb", "fontSize": "0.75rem", "fontWeight": "600"}),
        ], style={"textAlign": "center", "marginBottom": "20px"}),
        # ── 3 headline questions ──
        html.Div([
            create_headline_card(
                "Did we dominate?",
                field_tilt_title,
                f"{home_metrics['Field Tilt']}%", f"{away_metrics['Field Tilt']}%",
                field_tilt_note
            ),
            create_headline_card(
                "Did we create chances?",
                "Expected Goals (xG)",
                home_metrics["xG"], away_metrics["xG"],
                "Quality × quantity of shots — 1.0 = one expected goal"
            ),
            create_headline_card(
                "Did we defend well?",
                "Pressing Intensity (PPDA)",
                home_metrics["PPDA"], away_metrics["PPDA"],
                "Passes allowed per defensive action — lower = pressed harder"
            ),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "justifyContent": "center", "marginBottom": "24px"}),
    ])

    plots = {}
    
    def safe_plot(func, *args):
        try:
            return func(*args)
        except Exception as e:
            print(f"Error generating plot {func.__name__}: {e}")
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(visuals.TACTIQ_BG)
            ax.set_facecolor(visuals.TACTIQ_BG)
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", ha='center', va='center', color=visuals.TACTIQ_FG)
            ax.axis('off')
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=visuals.TACTIQ_BG)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

    plots["shot_map"] = safe_plot(visuals.plot_match_shot_map, df, home_team, away_team)
    plots["territorial_voronoi"] = safe_plot(visuals.plot_pitch_dominance, df, home_team, away_team)
    
    h_goals = len(df[(df['team_name'] == home_team) & (df['type_id'] == 16)])
    a_goals = len(df[(df['team_name'] == away_team) & (df['type_id'] == 16)])
    plots["player_dashboard"] = safe_plot(visuals.plot_player_dashboard_bars, df, home_team, away_team, h_goals, a_goals)
    
    home_tempo = tempo_data.process_tempo_network(df, home_team)
    away_tempo = tempo_data.process_tempo_network(df, away_team)

    for team in [home_team, away_team]:
        t_data = home_tempo if team == home_team else away_tempo
        plots[f"{team}_hybrid"]          = safe_plot(tempo_visuals.plot_hybrid_pass_network, df, team, t_data)
        plots[f"{team}_xt"]              = safe_plot(visuals.plot_xt_leaders, df, team)
        plots[f"{team}_startxi"]         = safe_plot(visuals.plot_starting_xi, df, team)
        plots[f"{team}_def_profile"]     = safe_plot(visuals.plot_defensive_profile, df, team)
        plots[f"{team}_pressing"]        = safe_plot(visuals.plot_pressing_map,             df, team)
        plots[f"{team}_off_trans"]       = safe_plot(visuals.plot_offensive_transition_map, df, team)
        plots[f"{team}_corners"]         = safe_plot(visuals.plot_set_pieces, df, team, "corners")
        plots[f"{team}_free_kicks"]      = safe_plot(visuals.plot_set_pieces, df, team, "free_kicks")
        plots[f"{team}_goal_kicks"]      = safe_plot(visuals.plot_goal_kicks_distribution, df, team)
        plots[f"{team}_penalties"]       = safe_plot(visuals.plot_penalties, df, team)

    def get_img(key):
        return f"data:image/png;base64,{plots.get(key, '')}"

    def make_tempo_table(t_data):
        profiles = t_data.get("profiles", [])
        if not profiles:
            return html.Div("No tempo data available", style={"color": "#888", "fontSize": "0.85rem"})
        coach_roles = {"Metronome": "Playmaker", "Direct": "Direct Passer", "Recycler": "Safe Passer", "Connector": "Link Player"}
        role_colors = {"Playmaker": "#3b82f6", "Direct Passer": "#ef4444", "Safe Passer": "#a0aec0", "Link Player": "#fbbf24"}
        rows = [html.Tr([
            html.Th("Player",  style={"textAlign": "left",   "padding": "8px 10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
            html.Th("Release", style={"textAlign": "center", "padding": "8px 10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
            html.Th("Carry",   style={"textAlign": "center", "padding": "8px 10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
            html.Th("Style",   style={"textAlign": "center", "padding": "8px 10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
        ])]
        for p in profiles:
            cr = coach_roles.get(p.get("Role", ""), p.get("Role", ""))
            rc = role_colors.get(cr, "#ffffff")
            j  = p.get("jersey_number")
            display = f"#{int(j)}" if j is not None else p["Player"]
            rows.append(html.Tr([
                html.Td(display,           style={"padding": "7px 10px", "color": "#d1d5db"}),
                html.Td(f"{p['TTRP']}s",   style={"textAlign": "center", "padding": "7px 10px", "color": "#fbbf24"}),
                html.Td(f"+{p['Carry']}m", style={"textAlign": "center", "padding": "7px 10px", "color": "#22c55e"}),
                html.Td(cr,                style={"textAlign": "center", "padding": "7px 10px", "color": rc}),
            ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))
        ttrp = t_data.get("team_avg_ttrp", 0)
        conn = t_data.get("team_total_connections", 0)
        rows.append(html.Tr([
            html.Td(f"Avg Release: {ttrp}s  |  {conn} key connections", colSpan=4,
                    style={"textAlign": "center", "padding": "10px", "color": "#ef4444",
                           "fontWeight": "bold", "fontSize": "0.8rem", "background": "rgba(239,68,68,0.05)"})
        ]))
        return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.82rem"})

    _img_style = {"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"}

    def team_label(name, color):
        return html.Div(clean_name(name), style={"color": color, "fontWeight": "700", "fontSize": "0.85rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "10px"})

    def dual_section(title, description, h_key, a_key):
        return html.Div([
            html.H3(title, style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "6px"}),
            html.P(description, style={"color": "#9ca3af", "fontSize": "0.8rem", "marginBottom": "14px"}),
            html.Div([
                html.Div([team_label(home_team, "#fbbf24"),
                          html.Img(src=get_img(h_key), className="plot-img", style=_img_style)],
                         style={"flex": "1", "minWidth": "280px"}),
                html.Div([team_label(away_team, "#e5e7eb"),
                          html.Img(src=get_img(a_key), className="plot-img", style=_img_style)],
                         style={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], className="visualization-card", style={"marginBottom": "30px"})

    # ── OFFENSIVE ──────────────────────────────────────────────────────────────
    phase_offensive = html.Div([
        html.Div([
            html.H3("Pass & Tempo Network", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "6px"}),
            html.P("Who dictates play (node size), passing combinations (line thickness), speed of play (Red = Fast, Blue = Slow).",
                   style={"color": "#9ca3af", "fontSize": "0.8rem", "marginBottom": "14px"}),
            html.Div([
                html.Div([
                    team_label(home_team, "#fbbf24"),
                    html.Img(src=get_img(f"{home_team}_hybrid"), className="plot-img", style=_img_style),
                    html.Div([
                        html.Div("PLAYER TEMPO PROFILES", style={"textAlign": "center", "fontWeight": "bold",
                                                                   "color": "#e5e7eb", "marginBottom": "4px", "marginTop": "14px"}),
                        make_tempo_table(home_tempo),
                    ], style={"background": "rgba(255,255,255,0.03)", "padding": "12px", "borderRadius": "10px",
                               "border": "1px solid var(--border-color)", "marginTop": "12px"}),
                ], style={"flex": "1", "minWidth": "280px"}),
                html.Div([
                    team_label(away_team, "#e5e7eb"),
                    html.Img(src=get_img(f"{away_team}_hybrid"), className="plot-img", style=_img_style),
                    html.Div([
                        html.Div("PLAYER TEMPO PROFILES", style={"textAlign": "center", "fontWeight": "bold",
                                                                   "color": "#e5e7eb", "marginBottom": "4px", "marginTop": "14px"}),
                        make_tempo_table(away_tempo),
                    ], style={"background": "rgba(255,255,255,0.03)", "padding": "12px", "borderRadius": "10px",
                               "border": "1px solid var(--border-color)", "marginTop": "12px"}),
                ], style={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
        ], className="visualization-card", style={"marginBottom": "30px"}),
        dual_section("Top xT Generators",
                     "Players who created the most threat through carries and passes.",
                     f"{home_team}_xt", f"{away_team}_xt"),
    ], style={"padding": "20px 0"})

    # ── DEFENSIVE ──────────────────────────────────────────────────────────────
    phase_defensive = html.Div([
        dual_section("Defensive Profile",
                     "Event-data estimate of engagement height, defensive-action spread, half-to-half shift, and action breakdown.",
                     f"{home_team}_def_profile", f"{away_team}_def_profile"),
    ], style={"padding": "20px 0"})

    # ── OFFENSIVE TRANSITIONS ──────────────────────────────────────────────────
    phase_off_transitions = html.Div([
        dual_section("Offensive Transition Passes",
                     "Forward passes launched from the defensive/mid zone — shows where each team starts attacks after winning possession.",
                     f"{home_team}_off_trans", f"{away_team}_off_trans"),
    ], style={"padding": "20px 0"})

    # ── DEFENSIVE TRANSITIONS ──────────────────────────────────────────────────
    phase_def_transitions = html.Div([
        dual_section("Pressing & Recovery Map",
                     "Where each team's defensive actions (tackles, interceptions, recoveries, clearances) happen — reveals the pressing footprint and recovery zones.",
                     f"{home_team}_pressing", f"{away_team}_pressing"),
    ], style={"padding": "20px 0"})

    # ── SET PIECES ─────────────────────────────────────────────────────────────
    phase_set_pieces = html.Div([
        dual_section("Corner Kicks",
                     "Corner delivery origins and landing zones — shows delivery patterns and target areas.",
                     f"{home_team}_corners", f"{away_team}_corners"),
        dual_section("Free Kicks",
                     "Dangerous free kick positions and delivery zones in the attacking third.",
                     f"{home_team}_free_kicks", f"{away_team}_free_kicks"),
        dual_section("Goal Kicks",
                     "Goal-kick landing zone distribution: Inside penalty box, Short outside box (def third), and Long (beyond def third).",
                     f"{home_team}_goal_kicks", f"{away_team}_goal_kicks"),
        dual_section("Penalties",
                     "Penalty attempts, result, taker, minute, and attempt location when available.",
                     f"{home_team}_penalties", f"{away_team}_penalties"),
    ], style={"padding": "20px 0"})

    # ── FINAL LAYOUT ──────────────────────────────────────────────────────────
    dashboard_img = plots.get('player_dashboard', '')
    dashboard_content = html.Div([
        html.H2("Top Players Dashboard", style={"color": "white", "textAlign": "center", "margin": "40px 0 30px"}),
        html.Div([
            html.Img(src=f"data:image/png;base64,{dashboard_img}",
                     style={"width": "100%", "maxWidth": "1000px", "borderRadius": "12px",
                            "border": "1px solid #444", "display": "block", "margin": "0 auto"})
        ], style={"padding": "20px", "background": "rgba(0,0,0,0.2)", "borderRadius": "16px"})
    ])

    return html.Div([
        dcc.Store(id='current-match-id', data=decoded_match_id),
        html.Header([
            html.Div("Match Deep Dive", style={"color": "var(--accent-color)", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),
            html.Div([
                html.Span(clean_name(home_team), style={"fontSize": "2rem", "fontWeight": "700", "color": "white"}),
                html.Div([
                    html.Span(str(h_goals), style={"fontSize": "3rem", "fontWeight": "900", "color": "#fbbf24"}),
                    html.Span(" — ", style={"fontSize": "2rem", "color": "#555", "margin": "0 8px"}),
                    html.Span(str(a_goals), style={"fontSize": "3rem", "fontWeight": "900", "color": "white"}),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "margin": "0 28px"}),
                html.Span(clean_name(away_team), style={"fontSize": "2rem", "fontWeight": "700", "color": "white"}),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "12px", "flexWrap": "wrap", "gap": "8px"}),
            html.P("Advanced Tactical Visualizations & Pattern Analysis", style={"color": "var(--text-secondary)", "marginBottom": "16px"}),
            dcc.Link("← Back to Hub", href="/analysis", style={"color": "var(--text-secondary)", "display": "inline-block"})
        ], style={"textAlign": "center", "marginBottom": "40px"}),

        dbc.Tabs([
            dbc.Tab(label="Match Overview", children=[
                html.Div([
                    kpi_section,
                    html.Div([
                        html.H3("Shot Map & Stats", style={"color": "var(--accent-color)", "fontSize": "1.5rem", "marginBottom": "20px", "textAlign": "center"}),
                        html.Img(src=f"data:image/png;base64,{plots.get('shot_map', '')}", className="plot-img",
                                 style={"width": "100%", "maxWidth": "1000px", "margin": "0 auto", "display": "block",
                                        "borderRadius": "12px", "border": "1px solid var(--border-color)"})
                    ], className="visualization-card", style={"marginBottom": "40px", "marginTop": "20px"}),
                    html.Div([
                        html.H3("Territorial Flow (Voronoi Control)", style={"color": "var(--accent-color)", "fontSize": "1.5rem", "marginBottom": "20px", "textAlign": "center"}),
                        html.P("Theoretical pitch footprint each team controls based on average positioning.",
                               style={"color": "var(--text-secondary)", "textAlign": "center", "marginBottom": "20px"}),
                        html.Img(src=f"data:image/png;base64,{plots.get('territorial_voronoi', '')}", className="plot-img",
                                 style={"width": "100%", "maxWidth": "1000px", "margin": "0 auto", "display": "block",
                                        "borderRadius": "12px", "border": "1px solid var(--border-color)"})
                    ], className="visualization-card", style={"marginBottom": "40px"}),
                    dashboard_content,
                ], style={"paddingTop": "30px"})
            ], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Offensive",               children=[html.Div(phase_offensive,       style={"paddingTop": "20px"})], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Defensive",               children=[html.Div(phase_defensive,       style={"paddingTop": "20px"})], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Offensive Transitions",   children=[html.Div(phase_off_transitions, style={"paddingTop": "20px"})], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Defensive Transitions",   children=[html.Div(phase_def_transitions, style={"paddingTop": "20px"})], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Set Pieces",              children=[html.Div(phase_set_pieces,      style={"paddingTop": "20px"})], tab_style={"cursor": "pointer"}),
        ], className="mt-4 custom-tabs")

    ], className="container", style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"})

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from utils.data import get_match_dataframe, extract_fixture_data
from utils.wyscout_loader import get_wyscout_match_stats
import utils.visuals as visuals
import utils.metrics as metrics
import utils.ball_trace_visuals as bt_visuals
import utils.tempo_data as tempo_data
import utils.tempo_visuals as tempo_visuals
import utils.obv_model as obv_model
import utils.obv_visuals as obv_visuals
from göztepehub.utils import buildup_analysis
import pandas as pd
import numpy as np

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
        "Prog Passes": metrics.calculate_progressive_passes(df, home_team),
        "BDP": metrics.calculate_bdp(df, home_team),
    }

    away_metrics = {
        "xG": metrics.calculate_xg(df, away_team),
        "xA": metrics.calculate_xa(df, away_team),
        "PPDA": metrics.calculate_ppda(df, away_team),
        "Field Tilt": metrics.calculate_field_tilt(df, away_team),
        "xT": metrics.calculate_xt(df, away_team),
        "Prog Passes": metrics.calculate_progressive_passes(df, away_team),
        "BDP": metrics.calculate_bdp(df, away_team),
    }

    # Wyscout xG & PPDA override (daha güvenilir kaynak)
    try:
        ws = get_wyscout_match_stats(home_team, away_team)
        if ws['home']['xg'] is not None:
            home_metrics['xG'] = ws['home']['xg']
        if ws['away']['xg'] is not None:
            away_metrics['xG'] = ws['away']['xg']
        if ws['home']['ppda'] is not None:
            home_metrics['PPDA'] = ws['home']['ppda']
        if ws['away']['ppda'] is not None:
            away_metrics['PPDA'] = ws['away']['ppda']
    except Exception:
        pass

    # SEPP & Buildup Integration
    h_sepp = metrics.calculate_sepp(df, home_team)
    a_sepp = metrics.calculate_sepp(df, away_team)
    h_buildup = buildup_analysis.analyze_buildup_for_match(df, home_team)
    a_buildup = buildup_analysis.analyze_buildup_for_match(df, away_team)

    def safe_get_b(b_dict, cat, key, default="-"):
        if not b_dict: return default
        inner = b_dict.get(cat, {})
        if not isinstance(inner, dict): return default
        return inner.get(key, default)

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

    def create_kpi_card(title, subtitle, h_val, a_val):
        return html.Div([
            html.Div(title, style={"fontSize": "0.8rem", "color": "#a0aec0", "fontWeight": "600", "marginBottom": "2px"}),
            html.Div(subtitle, style={"fontSize": "0.62rem", "color": "#555", "marginBottom": "10px",
                                      "fontStyle": "italic", "lineHeight": "1.3"}),
            html.Div([
                html.Span(f"{h_val}", style={"fontSize": "1.4rem", "fontWeight": "bold",
                                             "color": "#fbbf24", "marginRight": "8px"}),
                html.Span("-", style={"color": "#555"}),
                html.Span(f"{a_val}", style={"fontSize": "1.4rem", "fontWeight": "bold",
                                             "color": "white", "marginLeft": "8px"}),
            ])
        ], style={"background": "rgba(255,255,255,0.05)", "padding": "15px", "borderRadius": "12px",
                  "border": "1px solid rgba(255,255,255,0.1)", "textAlign": "center",
                  "minWidth": "140px", "flex": "1"})

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
                "Territorial Control (Field Tilt %)",
                f"{home_metrics['Field Tilt']}%", f"{away_metrics['Field Tilt']}%",
                "% of play spent in the attacking half — higher is better"
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

        # ── Toggle button ──
        html.Div([
            dbc.Button("Show All Stats ▾", id="match-details-toggle", outline=True, size="sm",
                       style={"borderColor": "#444", "color": "#666", "fontSize": "0.75rem",
                              "background": "transparent"})
        ], style={"textAlign": "center", "marginBottom": "12px"}),

        # ── Collapsible detailed metrics ──
        dbc.Collapse(id="match-details-collapse", is_open=False, children=[
            html.Div("All Metrics", style={
                "textAlign": "center", "fontSize": "0.65rem", "color": "#555",
                "letterSpacing": "2px", "textTransform": "uppercase", "marginBottom": "16px",
            }),
            html.Div([
                create_kpi_card("Expected Goals (xG)",
                                "Quality of shots created — 1.0 = one expected goal",
                                home_metrics["xG"], away_metrics["xG"]),
                create_kpi_card("Chance Creation (xA)",
                                "Dangerous passes that nearly led to shots",
                                home_metrics["xA"], away_metrics["xA"]),
                create_kpi_card("Pressing Intensity (PPDA)",
                                "Passes allowed per defensive action — lower = pressed harder",
                                home_metrics["PPDA"], away_metrics["PPDA"]),
                create_kpi_card("Territorial Dominance",
                                "% of play in the attacking half — higher is better",
                                f"{home_metrics['Field Tilt']}%", f"{away_metrics['Field Tilt']}%"),
                create_kpi_card("Threat Generated (xT)",
                                "How much danger your moves created — higher is better",
                                home_metrics["xT"], away_metrics["xT"]),
                create_kpi_card("Forward Passes",
                                "Passes that moved play toward the opponent's goal",
                                home_metrics["Prog Passes"], away_metrics["Prog Passes"]),
                create_kpi_card("Defensive Control (BDP)",
                                "How often you disrupted opponent's build-up — higher is better",
                                f"{home_metrics['BDP']}%", f"{away_metrics['BDP']}%"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                      "justifyContent": "center", "marginBottom": "20px"}),

            html.Div("Build-up & Shot Creation", style={
                "textAlign": "center", "fontSize": "0.65rem", "color": "#555",
                "letterSpacing": "2px", "textTransform": "uppercase",
                "marginBottom": "14px", "marginTop": "8px",
            }),
            html.Div([
                create_kpi_card("Shot Sequences (SEPP)",
                                "Possession chains that ended with a shot",
                                h_sepp['sepp_total'], a_sepp['sepp_total']),
                create_kpi_card("Shot Creation Rate",
                                "How efficiently possession built into shots",
                                f"{h_sepp['efficiency']}%", f"{a_sepp['efficiency']}%"),
                create_kpi_card("Total Build-ups",
                                "How many times you built from the back",
                                h_buildup.get('total_buildups', '-') if h_buildup else '-',
                                a_buildup.get('total_buildups', '-') if a_buildup else '-'),
                create_kpi_card("Final Third Entries",
                                "% of build-ups that reached the attacking third",
                                f"{safe_get_b(h_buildup, 'outcomes_15s', 'f3_entry_pct')}%",
                                f"{safe_get_b(a_buildup, 'outcomes_15s', 'f3_entry_pct')}%"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                      "justifyContent": "center", "marginBottom": "30px"}),
        ]),
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
    
    home_bt    = metrics.calculate_ball_trace(df, home_team)
    away_bt    = metrics.calculate_ball_trace(df, away_team)
    home_tempo = tempo_data.process_tempo_network(df, home_team)
    away_tempo = tempo_data.process_tempo_network(df, away_team)
    df_obv     = obv_model.calculate_obv(df)

    for team in [home_team, away_team]:
        bt_data = home_bt   if team == home_team else away_bt
        t_data  = home_tempo if team == home_team else away_tempo
        plots[f"{team}_prog"]            = safe_plot(visuals.plot_progressive_pass_map, df, team)
        plots[f"{team}_hybrid"]          = safe_plot(tempo_visuals.plot_hybrid_pass_network, df, team, t_data)
        plots[f"{team}_xt"]              = safe_plot(visuals.plot_xt_leaders, df, team)
        plots[f"{team}_startxi"]         = safe_plot(visuals.plot_starting_xi, df, team)
        plots[f"{team}_def_profile"]     = safe_plot(visuals.plot_defensive_profile, df, team)
        plots[f"{team}_bt_map"]          = safe_plot(bt_visuals.plot_ball_time_map,      bt_data, team)
        plots[f"{team}_bt_bars"]         = safe_plot(bt_visuals.plot_thirds_flanks_bars, bt_data, team)
        plots[f"{team}_obv_pitch"]       = safe_plot(obv_visuals.plot_obv_pitch,         df_obv,  team)
        plots[f"{team}_obv_leaderboard"] = safe_plot(obv_visuals.plot_obv_leaderboard,   df_obv,  team)

    def create_team_section(team_name, plots_dict, t_data):
        def get_img(key):
            return f"data:image/png;base64,{plots_dict.get(key, '')}"

        def create_tempo_table(profiles):
            if not profiles:
                return html.Div("No tempo data available", style={"color": "#888"})
            rows = [html.Tr([
                html.Th("Player",         style={"textAlign": "left",   "padding": "10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
                html.Th("Release Time",   style={"textAlign": "center", "padding": "10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
                html.Th("Avg Carry",      style={"textAlign": "center", "padding": "10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
                html.Th("Connections",    style={"textAlign": "center", "padding": "10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
                html.Th("Style",          style={"textAlign": "center", "padding": "10px", "borderBottom": "1px solid #444", "color": "#e5e7eb"}),
            ])]
            
            coach_roles = {
                "Metronome": "Playmaker",
                "Direct": "Direct Passer",
                "Recycler": "Safe Passer",
                "Connector": "Link Player"
            }
            role_colors = {"Playmaker": "#3b82f6", "Direct Passer": "#ef4444", "Safe Passer": "#a0aec0", "Link Player": "#fbbf24"}
            
            for p in profiles:
                orig_role = p.get("Role", "")
                coach_role = coach_roles.get(orig_role, orig_role)
                rc = role_colors.get(coach_role, "#ffffff")
                
                j  = p.get("jersey_number")
                display = f"#{int(j)}" if j is not None else p["Player"]
                rows.append(html.Tr([
                    html.Td(display,              style={"padding": "8px", "color": "#d1d5db"}),
                    html.Td(f"{p['TTRP']}s",      style={"textAlign": "center", "padding": "8px", "color": "#fbbf24"}),
                    html.Td(f"+{p['Carry']}m",    style={"textAlign": "center", "padding": "8px", "color": "#22c55e"}),
                    html.Td(p.get("Drawn To", ""),style={"textAlign": "center", "padding": "8px", "color": "#22c55e"}),
                    html.Td(coach_role,           style={"textAlign": "center", "padding": "8px", "color": rc}),
                ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))
            
            ttrp = t_data.get("team_avg_ttrp", 0)
            conn = t_data.get("team_total_connections", 0)
            rows.append(html.Tr([
                html.Td(f"Team Avg Release Time: {ttrp}s | {conn} strong connections", colSpan=5,
                        style={"textAlign": "center", "padding": "12px", "color": "#ef4444",
                               "fontWeight": "bold", "fontSize": "0.85rem",
                               "background": "rgba(239,68,68,0.05)"})
            ]))
            return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85rem"})

        tactical_setup = html.Div([
            html.Div([
                html.H3("Average Position Map", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.P("Overall average player positions. This represents the baseline shape of the team across the entire match.",
                       style={"color": "#9ca3af", "fontSize": "0.8rem", "marginBottom": "12px"}),
                html.Img(src=get_img(f'{team_name}_startxi'), className="plot-img", style={"width": "100%", "maxWidth": "500px", "display": "block", "margin": "0 auto", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
            ], className="visualization-card", style={"marginBottom": "30px", "textAlign": "center"}),
            html.Div([
                html.H3("Defensive Profile", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.P("Block type, compactness, line height shift between halves, and action breakdown.",
                       style={"color": "#9ca3af", "fontSize": "0.8rem", "marginBottom": "12px"}),
                html.Img(src=get_img(f'{team_name}_def_profile'), className="plot-img", style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
            ], className="visualization-card", style={"marginBottom": "30px"}),
        ], style={"padding": "20px 0"})

        possession_tempo = html.Div([
            html.Div([
                html.H3("Pass & Tempo Network", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "4px"}),
                html.P("Shows who dictates play (node size), passing combinations (line thickness), and speed of play (line color: Red = Fast, Blue = Slow).",
                       style={"color": "#9ca3af", "fontSize": "0.8rem", "marginBottom": "12px"}),
                html.Img(src=get_img(f'{team_name}_hybrid'), className="plot-img",
                         style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"}),
                html.Div([
                    html.Div("PLAYER TEMPO PROFILES", style={"textAlign": "center", "fontWeight": "bold",
                                                              "marginBottom": "5px", "color": "#e5e7eb",
                                                              "marginTop": "18px"}),
                    html.Div("Average release time, carry distance & playing style",
                             style={"textAlign": "center", "fontSize": "0.8rem",
                                    "marginBottom": "10px", "color": "#a0aec0"}),
                    create_tempo_table(t_data.get("profiles", [])),
                ], style={"background": "rgba(255,255,255,0.03)", "padding": "15px",
                           "borderRadius": "12px", "border": "1px solid var(--border-color)",
                           "marginTop": "16px"}),
            ], className="visualization-card", style={"marginBottom": "30px"}),
        ], style={"padding": "20px 0"})

        progression_threat = html.Div([
            html.Div([
                html.H3("Progressive Passes", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.Img(src=get_img(f'{team_name}_prog'), className="plot-img", style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
            ], className="visualization-card", style={"marginBottom": "30px"}),
            html.Div([
                html.H3("Top xT Generators", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.Img(src=get_img(f'{team_name}_xt'), className="plot-img", style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
            ], className="visualization-card", style={"marginBottom": "30px"}),
            html.Div([
                html.H3("On-Ball Value (OBV)", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.P("OBV measures how much each action shifts the team's probability of scoring or conceding. Green = positive, Red = negative.",
                       style={"color": "#9ca3af", "fontSize": "0.85rem", "marginBottom": "15px"}),
                html.Div([
                    html.Div([html.Img(src=get_img(f'{team_name}_obv_pitch'), className="plot-img",
                                       style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})],
                             style={"flex": "1", "minWidth": "300px"}),
                    html.Div([html.Img(src=get_img(f'{team_name}_obv_leaderboard'), className="plot-img",
                                       style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})],
                             style={"flex": "1", "minWidth": "300px"}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px"}),
            ], className="visualization-card", style={"marginBottom": "30px"}),
        ], style={"padding": "20px 0"})

        territorial_control = html.Div([
            html.Div([
                html.H3("Ball Trace: Territorial Map", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.Img(src=get_img(f'{team_name}_bt_map'), className="plot-img", style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
            ], className="visualization-card", style={"marginBottom": "30px"}),
            html.Div([
                html.H3("Ball Trace: Flow & Distribution", style={"color": "var(--accent-color)", "fontSize": "1.2rem", "marginBottom": "10px"}),
                html.Img(src=get_img(f'{team_name}_bt_bars'), className="plot-img", style={"width": "100%", "borderRadius": "12px", "border": "1px solid var(--border-color)"}),
            ], className="visualization-card", style={"marginBottom": "30px"}),
        ], style={"padding": "20px 0"})

        return html.Div([
            html.Div([
                html.H2(clean_name(team_name), style={"fontSize": "2rem", "marginBottom": "10px"})
            ], className="team-header", style={"borderBottom": "1px solid var(--border-color)", "paddingBottom": "15px", "marginBottom": "20px"}),
            dbc.Tabs([
                dbc.Tab(tactical_setup,      label="Tactical Setup",        tab_style={"cursor": "pointer"}),
                dbc.Tab(possession_tempo,    label="Possession & Tempo",     tab_style={"cursor": "pointer"}),
                dbc.Tab(progression_threat,  label="Progression & Threat",   tab_style={"cursor": "pointer"}),
                dbc.Tab(territorial_control, label="Territorial Control",    tab_style={"cursor": "pointer"}),
            ], className="sub-tabs mt-3"),
        ], className="team-section", style={"background": "var(--card-bg)", "border": "1px solid var(--border-color)", "borderRadius": "20px", "padding": "30px", "flex": "1"})

    dashboard_img = plots.get('player_dashboard', '')
    
    dashboard_content = html.Div([
         html.H2("Top Players Dashboard", style={"color": "white", "textAlign": "center", "margin": "40px 0 30px"}),
         html.Div([
             html.Img(src=f"data:image/png;base64,{dashboard_img}", style={"width": "100%", "maxWidth": "1000px", "borderRadius": "12px", "border": "1px solid #444", "display": "block", "margin": "0 auto"})
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
                        html.Img(src=f"data:image/png;base64,{plots.get('shot_map', '')}", className="plot-img", style={"width": "100%", "maxWidth": "1000px", "margin": "0 auto", "display": "block", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
                    ], className="visualization-card", style={"marginBottom": "40px", "marginTop": "20px"}),
                    html.Div([
                        html.H3("Territorial Flow (Voronoi Control)", style={"color": "var(--accent-color)", "fontSize": "1.5rem", "marginBottom": "20px", "textAlign": "center"}),
                        html.P(
                            "This visual combines the starting XIs of both teams to calculate the theoretical footprint each team controls on the pitch based on average positioning.",
                            style={"color": "var(--text-secondary)", "textAlign": "center", "marginBottom": "20px"}
                        ),
                        html.Img(src=f"data:image/png;base64,{plots.get('territorial_voronoi', '')}", className="plot-img", style={"width": "100%", "maxWidth": "1000px", "margin": "0 auto", "display": "block", "borderRadius": "12px", "border": "1px solid var(--border-color)"})
                    ], className="visualization-card", style={"marginBottom": "40px"}),
                    dashboard_content
                ], style={"paddingTop": "30px"})
            ], tab_style={"cursor": "pointer"}),
            dbc.Tab(label=f"{clean_name(home_team)} Analysis", children=[
                html.Div([
                    create_team_section(home_team, plots, home_tempo)
                ], style={"paddingTop": "30px"})
            ], tab_style={"cursor": "pointer"}),
            dbc.Tab(label=f"{clean_name(away_team)} Analysis", children=[
                html.Div([
                    create_team_section(away_team, plots, away_tempo)
                ], style={"paddingTop": "30px"})
            ], tab_style={"cursor": "pointer"})
        ], className="mt-4 custom-tabs")

    ], className="container", style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"})


@callback(
    Output('match-details-collapse', 'is_open'),
    Input('match-details-toggle', 'n_clicks'),
    State('match-details-collapse', 'is_open'),
    prevent_initial_call=True
)
def toggle_match_details(n_clicks, is_open):
    return not is_open


import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.data import get_match_dataframe
from utils.wyscout_loader import get_wyscout_match_stats
import utils.visuals as visuals
import utils.metrics as metrics
import utils.analysis as analysis
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

    # Wyscout overrides — prefer Wyscout for any metric it provides
    # (except for Week 34 where we use the custom model)
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
            html.Div(question, className="mr-headline-card__question"),
            html.Div([
                html.Div([
                    html.Div(clean_name(home_team), className="mr-team-label mr-team-label--home"),
                    html.Span(f"{h_val}", className="mr-big-number mr-big-number--home"),
                ], className="mr-headline-card__team"),
                html.Span(" — ", className="mr-headline-card__dash"),
                html.Div([
                    html.Div(clean_name(away_team), className="mr-team-label mr-team-label--away"),
                    html.Span(f"{a_val}", className="mr-big-number mr-big-number--away"),
                ], className="mr-headline-card__team"),
            ], className="mr-headline-card__team-pair"),
            html.Div(metric_name, className="mr-headline-card__metric"),
            html.Div(note, className="mr-headline-card__note"),
        ], className="mr-headline-card")

    field_tilt_title = "Possession %" if has_possession_override else "Territorial Control (Field Tilt %)"
    field_tilt_note = ("Percentage of ball possession from Wyscout" if has_possession_override
                      else "% of play spent in the attacking half — higher is better")

    kpi_section = html.Div([
        html.P("3 things that decide this match", className="mr-3-things"),
        html.Div([
            html.Span("●", className="mr-legend-row__dot",
                      style={"color": "var(--accent-gold)"}),
            html.Span(clean_name(home_team),
                      className="mr-legend-row__name mr-legend-row__name--home"),
            html.Span("●", className="mr-legend-row__dot", style={"color": "#e5e7eb"}),
            html.Span(clean_name(away_team),
                      className="mr-legend-row__name mr-legend-row__name--away"),
        ], className="mr-legend-row"),
        # 3 headline questions
        html.Div([
            create_headline_card(
                "Did we dominate?",
                field_tilt_title,
                f"{home_metrics['Field Tilt']}%", f"{away_metrics['Field Tilt']}%",
                field_tilt_note,
            ),
            create_headline_card(
                "Did we create chances?",
                "Expected Goals (xG)",
                home_metrics["xG"], away_metrics["xG"],
                "Quality × quantity of shots — 1.0 = one expected goal",
            ),
            create_headline_card(
                "Did we defend well?",
                "Pressing Intensity (PPDA)",
                home_metrics["PPDA"], away_metrics["PPDA"],
                "Passes allowed per defensive action — lower = pressed harder",
            ),
        ], className="mr-kpi-grid"),
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
            return html.Div("No tempo data available", className="mr-tempo-empty")
        coach_roles = {"Metronome": "Playmaker", "Direct": "Direct Passer",
                       "Recycler": "Safe Passer", "Connector": "Link Player"}
        role_colors = {"Playmaker": "#3b82f6", "Direct Passer": "#ef4444",
                       "Safe Passer": "#a0aec0", "Link Player": "#fbbf24"}
        rows = [html.Tr([
            html.Th("Player"),
            html.Th("Release"),
            html.Th("Carry"),
            html.Th("Style"),
        ])]
        for p in profiles:
            cr = coach_roles.get(p.get("Role", ""), p.get("Role", ""))
            rc = role_colors.get(cr, "#ffffff")
            j  = p.get("jersey_number")
            display = f"#{int(j)}" if j is not None else p["Player"]
            rows.append(html.Tr([
                html.Td(display),
                html.Td(f"{p['TTRP']}s", className="mr-tempo-table__release"),
                html.Td(f"+{p['Carry']}m", className="mr-tempo-table__carry"),
                html.Td(cr, style={"color": rc}),
            ]))
        ttrp = t_data.get("team_avg_ttrp", 0)
        conn = t_data.get("team_total_connections", 0)
        rows.append(html.Tr([
            html.Td(f"Avg Release: {ttrp}s  |  {conn} key connections",
                    colSpan=4, className="mr-tempo-table__footer"),
        ]))
        return html.Table(rows, className="mr-tempo-table")

    def team_label(name, side):
        return html.Div(clean_name(name), className=f"mr-team-tag mr-team-tag--{side}")

    def _truthy_flag(value):
        return str(value).strip().lower() in {"1", "true", "yes", "si", "y"}

    def _set_piece_summary(team):
        sp_data = analysis.get_set_pieces(df, team)
        corners = len(sp_data.get("corners", []))
        free_kicks = len(sp_data.get("free_kicks", []))
        penalties = len(sp_data.get("penalties", []))

        corner_goals = 0
        free_kick_goals = 0
        penalty_goals = 0
        goal_events = []
        goals_df = df[(df["team_name"] == team) & (df["event"] == "Goal")]
        for _, goal in goals_df.iterrows():
            is_corner = (
                _truthy_flag(goal.get("From corner")) or
                _truthy_flag(goal.get("From Corner"))
            )
            is_penalty = _truthy_flag(goal.get("Penalty"))
            is_free_kick = (
                _truthy_flag(goal.get("Free kick")) or
                _truthy_flag(goal.get("Free Kick")) or
                _truthy_flag(goal.get("Set piece")) or
                _truthy_flag(goal.get("Set Piece"))
            )

            if is_corner:
                corner_goals += 1
                goal_type = "Corner"
            elif is_penalty:
                penalty_goals += 1
                goal_type = "Penalty"
            elif is_free_kick:
                free_kick_goals += 1
                goal_type = "Free kick"
            else:
                continue

            goal_events.append({
                "type": goal_type,
                "player": goal.get("player_name", "Unknown") or "Unknown",
                "minute": int(goal.get("time_min") or 0),
                "x": goal.get("x"),
                "y": goal.get("y"),
            })

        # Some feeds mark penalty attempts in the attempt row, not the goal row.
        penalties_df = sp_data.get("penalties")
        penalty_attempt_goals = (
            len(penalties_df[penalties_df["event"] == "Goal"])
            if penalties_df is not None and not penalties_df.empty else 0
        )
        if penalty_attempt_goals > penalty_goals and penalties_df is not None:
            for _, penalty in penalties_df[penalties_df["event"] == "Goal"].iterrows():
                goal_events.append({
                    "type": "Penalty",
                    "player": penalty.get("player_name", "Unknown") or "Unknown",
                    "minute": int(penalty.get("time_min") or 0),
                    "x": penalty.get("x"),
                    "y": penalty.get("y"),
                })
        penalty_goals = max(penalty_goals, penalty_attempt_goals)

        return {
            "corners": corners,
            "free_kicks": free_kicks,
            "penalties": penalties,
            "total": corners + free_kicks + penalties,
            "corner_goals": corner_goals,
            "free_kick_goals": free_kick_goals,
            "penalty_goals": penalty_goals,
            "goals": corner_goals + free_kick_goals + penalty_goals,
            "goal_events": goal_events,
        }

    home_set_pieces = _set_piece_summary(home_team)
    away_set_pieces = _set_piece_summary(away_team)

    def _set_piece_stat_box(label, value, color):
        return html.Div([
            html.Div(label, style={
                "fontSize": "0.62rem", "fontWeight": "700",
                "textTransform": "uppercase", "letterSpacing": "0.5px",
                "color": "var(--text-secondary)",
            }),
            html.Div(str(value), style={
                "fontSize": "1.55rem", "fontWeight": "900", "color": color,
                "lineHeight": "1.1",
            }),
        ], style={
            "padding": "10px", "background": "rgba(255,255,255,0.035)",
            "border": "1px solid var(--border-color)", "borderRadius": "8px",
            "textAlign": "center", "minWidth": "92px", "flex": "1",
        })

    def _set_piece_summary_pane(team, side, summary):
        main_color = "var(--accent-gold)" if side == "home" else "#e5e7eb"
        return html.Div([
            team_label(team, side),
            html.Div([
                _set_piece_stat_box("Set Pieces", summary["total"], main_color),
                _set_piece_stat_box("Set-Piece Goals", summary["goals"], "#22c55e"),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
            html.Div([
                html.Span(f"Corners {summary['corners']}"),
                html.Span(f"Dangerous FKs {summary['free_kicks']}"),
                html.Span(f"Penalties {summary['penalties']}"),
            ], style={
                "display": "flex", "gap": "8px", "flexWrap": "wrap",
                "fontSize": "0.72rem", "color": "var(--text-secondary)",
            }),
            html.Div([
                html.Span(f"Corner goals {summary['corner_goals']}"),
                html.Span(f"FK goals {summary['free_kick_goals']}"),
                html.Span(f"Penalty goals {summary['penalty_goals']}"),
            ], style={
                "display": "flex", "gap": "8px", "flexWrap": "wrap",
                "fontSize": "0.72rem", "color": "var(--text-secondary)",
                "marginTop": "6px",
            }),
            html.Div("SET-PIECE GOAL LOG", style={
                "fontSize": "0.62rem", "fontWeight": "800",
                "letterSpacing": "0.5px", "color": main_color,
                "marginTop": "12px", "marginBottom": "6px",
            }) if summary["goal_events"] else None,
            html.Div([
                html.Div([
                    html.Span(f"{goal['minute']}'", style={
                        "fontWeight": "900", "color": main_color,
                    }),
                    html.Span(goal["type"], style={
                        "fontWeight": "800", "color": "#22c55e",
                    }),
                    html.Span(goal["player"], style={
                        "fontWeight": "700", "color": "white",
                        "overflow": "hidden", "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    }),
                    html.Span(
                        f"x{float(goal['x']):.1f} y{float(goal['y']):.1f}"
                        if goal.get("x") is not None and goal.get("y") is not None else "",
                        style={"color": "var(--text-secondary)", "fontSize": "0.66rem"},
                    ),
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "34px 76px 1fr auto",
                    "gap": "8px",
                    "alignItems": "center",
                    "padding": "6px 0",
                    "borderTop": "1px solid rgba(255,255,255,0.06)" if idx else "none",
                    "fontSize": "0.72rem",
                }) for idx, goal in enumerate(summary["goal_events"])
            ], style={
                "marginTop": "2px",
                "padding": "0 8px",
                "background": "rgba(255,255,255,0.025)",
                "border": "1px solid var(--border-color)",
                "borderRadius": "8px",
            }) if summary["goal_events"] else html.Div(
                "No set-piece goals in this match.",
                style={
                    "fontSize": "0.7rem", "color": "var(--text-secondary)",
                    "marginTop": "10px",
                }
            ),
        ], className="mr-dual-col__pane")

    def dual_section(title, description, h_key, a_key):
        return html.Div([
            html.H3(title, className="mr-section-title"),
            html.P(description, className="mr-section-desc"),
            html.Div([
                html.Div([team_label(home_team, "home"),
                          html.Img(src=get_img(h_key), className="plot-img mr-plot-img")],
                         className="mr-dual-col__pane"),
                html.Div([team_label(away_team, "away"),
                          html.Img(src=get_img(a_key), className="plot-img mr-plot-img")],
                         className="mr-dual-col__pane"),
            ], className="mr-dual-col"),
        ], className="visualization-card mr-card")

    # ── OFFENSIVE ──────────────────────────────────────────────────────────────
    def _tempo_pane(team, t_data, side):
        return html.Div([
            team_label(team, side),
            html.Img(src=get_img(f"{team}_hybrid"), className="plot-img mr-plot-img"),
            html.Div([
                html.Div("PLAYER TEMPO PROFILES", className="mr-tempo-title"),
                make_tempo_table(t_data),
            ], className="mr-tempo-card"),
        ], className="mr-dual-col__pane")

    phase_offensive = html.Div([
        html.Div([
            html.H3("Pass & Tempo Network", className="mr-section-title"),
            html.P(
                "Who dictates play (node size), passing combinations (line thickness), "
                "speed of play (Red = Fast, Blue = Slow).",
                className="mr-section-desc",
            ),
            html.Div([
                _tempo_pane(home_team, home_tempo, "home"),
                _tempo_pane(away_team, away_tempo, "away"),
            ], className="mr-dual-col"),
        ], className="visualization-card mr-card"),
        dual_section("Top xT Generators",
                     "Players who created the most threat through carries and passes.",
                     f"{home_team}_xt", f"{away_team}_xt"),
    ], className="mr-phase")

    # ── DEFENSIVE ──────────────────────────────────────────────────────────────
    phase_defensive = html.Div([
        dual_section("Defensive Profile",
                     "Event-data estimate of engagement height, defensive-action spread, "
                     "half-to-half shift, and action breakdown.",
                     f"{home_team}_def_profile", f"{away_team}_def_profile"),
    ], className="mr-phase")

    # ── OFFENSIVE TRANSITIONS ──────────────────────────────────────────────────
    phase_off_transitions = html.Div([
        dual_section("Offensive Transition Passes",
                     "Forward passes launched from the defensive/mid zone — shows where each team "
                     "starts attacks after winning possession.",
                     f"{home_team}_off_trans", f"{away_team}_off_trans"),
    ], className="mr-phase")

    # ── DEFENSIVE TRANSITIONS ──────────────────────────────────────────────────
    phase_def_transitions = html.Div([
        dual_section("Pressing & Recovery Map",
                     "Where each team's defensive actions (tackles, interceptions, recoveries, "
                     "clearances) happen — reveals the pressing footprint and recovery zones.",
                     f"{home_team}_pressing", f"{away_team}_pressing"),
    ], className="mr-phase")

    # ── SET PIECES ─────────────────────────────────────────────────────────────
    phase_set_pieces = html.Div([
        html.Div([
            html.H3("Set-Piece Match Summary", className="mr-section-title"),
            html.P(
                "Corners, dangerous free kicks, penalties, and goals scored from those situations.",
                className="mr-section-desc",
            ),
            html.Div([
                _set_piece_summary_pane(home_team, "home", home_set_pieces),
                _set_piece_summary_pane(away_team, "away", away_set_pieces),
            ], className="mr-dual-col"),
        ], className="visualization-card mr-card"),
        dual_section("Corner Kicks",
                     "Corner delivery origins and landing zones — shows delivery patterns and target areas.",
                     f"{home_team}_corners", f"{away_team}_corners"),
        dual_section("Free Kicks",
                     "Dangerous free kick positions and delivery zones in the attacking third.",
                     f"{home_team}_free_kicks", f"{away_team}_free_kicks"),
        dual_section("Goal Kicks",
                     "Goal-kick landing zone distribution: Inside penalty box, Short outside box "
                     "(def third), and Long (beyond def third).",
                     f"{home_team}_goal_kicks", f"{away_team}_goal_kicks"),
        dual_section("Penalties",
                     "Penalty attempts, result, taker, minute, and attempt location when available.",
                     f"{home_team}_penalties", f"{away_team}_penalties"),
    ], className="mr-phase")

    # ── FINAL LAYOUT ──────────────────────────────────────────────────────────
    dashboard_img = plots.get('player_dashboard', '')
    dashboard_content = html.Div([
        html.H2("Top Players Dashboard", className="mr-dashboard-h2"),
        html.Div(
            html.Img(src=f"data:image/png;base64,{dashboard_img}", className="mr-overview-img-wrap"),
            className="mr-dashboard-wrap",
        ),
    ])

    return html.Div([
        dcc.Store(id='current-match-id', data=decoded_match_id),
        html.Header([
            html.Div("Match Deep Dive", className="mr-eyebrow"),
            html.Div([
                html.Span(clean_name(home_team), className="mr-scoreline__team"),
                html.Div([
                    html.Span(str(h_goals), className="mr-scoreline__num mr-scoreline__num--home"),
                    html.Span(" — ", className="mr-scoreline__sep"),
                    html.Span(str(a_goals), className="mr-scoreline__num mr-scoreline__num--away"),
                ], className="mr-scoreline__numbers"),
                html.Span(clean_name(away_team), className="mr-scoreline__team"),
            ], className="mr-scoreline"),
            html.P("Advanced Tactical Visualizations & Pattern Analysis",
                   className="mr-header__subtitle"),
            dcc.Link("← Back to Hub", href="/analysis", className="mr-back-link"),
        ], className="mr-header"),

        dbc.Tabs([
            dbc.Tab(label="Match Overview", children=[
                html.Div([
                    kpi_section,
                    html.Div([
                        html.H3("Shot Map & Stats",
                                className="mr-section-title mr-section-title--lg"),
                        html.Img(src=f"data:image/png;base64,{plots.get('shot_map', '')}",
                                 className="plot-img mr-overview-img-wrap"),
                    ], className="visualization-card mr-card--overview"),
                    html.Div([
                        html.H3("Territorial Flow (Voronoi Control)",
                                className="mr-section-title mr-section-title--lg"),
                        html.P("Theoretical pitch footprint each team controls based on average positioning.",
                               className="mr-section-desc mr-section-desc--center"),
                        html.Img(src=f"data:image/png;base64,{plots.get('territorial_voronoi', '')}",
                                 className="plot-img mr-overview-img-wrap"),
                    ], className="visualization-card mr-card--overview"),
                    dashboard_content,
                ], className="mr-overview-pane")
            ], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Offensive",             children=[html.Div(phase_offensive,       className="mr-tab-pane")], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Defensive",             children=[html.Div(phase_defensive,       className="mr-tab-pane")], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Offensive Transitions", children=[html.Div(phase_off_transitions, className="mr-tab-pane")], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Defensive Transitions", children=[html.Div(phase_def_transitions, className="mr-tab-pane")], tab_style={"cursor": "pointer"}),
            dbc.Tab(label="Set Pieces",            children=[html.Div(phase_set_pieces,      className="mr-tab-pane")], tab_style={"cursor": "pointer"}),
        ], className="mt-4 custom-tabs"),

    ], className="tq-page")

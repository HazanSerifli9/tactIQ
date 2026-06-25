import dash
import numpy as np
import pandas as pd
import os
import time
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64
import io
from mplsoccer import Pitch
from utils.data import extract_fixture_data, calculate_standings, get_data_dir, TEAM_LOGOS

dash.register_page(__name__, path='/post-match', title='Göztepe Hub | Post-Match')

GOZTEPE = 'Göztepe Spor Kulübü'
_SUFFIXES = ['Spor Kulübü', 'Futbol Kulübü', 'Kulübü', 'Spor A.Ş.', 'A.Ş.', 'S.K.', 'F.K.', 'SK']

_POST_MATCH_CACHE = {}
_RADAR_CACHE = {}
_LEAGUE_CACHE = {'df': None, 'timestamp': 0}

def _clean(name):
    r = name
    for s in _SUFFIXES:
        r = r.replace(s, '')
    return r.strip()

def _truthy_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({'si', 'yes', 'true', '1', 'y'})

def _corner_taken_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    taken_cols = [
        col for col in df.columns
        if isinstance(col, str) and 'corner' in col.lower() and 'taken' in col.lower()
    ]
    for col in taken_cols:
        mask |= _truthy_series(df[col])

    if not taken_cols and 'event' in df.columns:
        event_l = df['event'].astype(str).str.strip().str.lower()
        mask |= event_l.isin({'corner', 'corner taken', 'corner kick'})

    if not taken_cols and not mask.any() and 'type_id' in df.columns:
        mask |= pd.to_numeric(df['type_id'], errors='coerce').eq(6)

    return mask

def _count_corner_deliveries(df: pd.DataFrame) -> int:
    corners = df[_corner_taken_mask(df)].copy()
    if corners.empty:
        return 0
    subset = [
        col for col in ['team_name', 'period_id', 'time_min', 'time_sec', 'player_name', 'x', 'y']
        if col in corners.columns
    ]
    if subset:
        corners = corners.drop_duplicates(subset=subset)
    return len(corners)

PITCH_BG = "#0e1b0f"
LINE_C = "rgba(255,255,255,0.55)"
GOLD = "#fbbf24"
RED = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#a855f7"
GREEN = "rgba(34,197,94,0.9)"

RADAR_METRICS = [
    ('passes_pg',   'Passes/Game',    True),
    ('pass_acc',    'Pass Accuracy',  True),
    ('shots_pg',    'Shots/Game',     True),
    ('xg_pg',       'xG/Game',        True),
    ('press_rec_pg','High Turnovers', True),
    ('xga_pg',      'Def. Solidity',  False),
]

BENCH_METRICS = [
    ('passes_pg',   'Passes / Game',      True,  '{:.0f}'),
    ('pass_acc',    'Pass Accuracy',       True,  '{:.1f}%'),
    ('shots_pg',    'Shots / Game',        True,  '{:.1f}'),
    ('xg_pg',       'xG / Game',          True,  '{:.2f}'),
    ('press_rec_pg','High Turnovers',      True,  '{:.1f}'),
    ('xga_pg',      'xGA / Game',         False, '{:.2f}'),
    ('tackles_pg',  'Tackles / Game',     True,  '{:.1f}'),
    ('ball_rec_pg', 'Ball Recoveries',    True,  '{:.1f}'),
]


def _load_goztepe_matches():
    try:
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
    except Exception as e:
        print(f"Match loading error: {e}")
        return []


def _load_league_benchmarks():
    if time.time() - _LEAGUE_CACHE.get('timestamp', 0) < 3600:
        return _LEAGUE_CACHE['df']

    try:
        from utils.xg_model import predict_xg
    except ImportError:
        predict_xg = lambda d: d

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    acc = {}

    for fn in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, fn))
            df = predict_xg(df)
        except Exception:
            continue
        for team in df['team_name'].unique():
            tdf = df[df['team_name'] == team]
            odf = df[df['team_name'] != team]
            n_pass  = len(tdf[tdf['type_id'] == 1])
            n_succ  = len(tdf[(tdf['type_id'] == 1) & (tdf['outcome'] == 1)])
            n_shots = len(tdf[tdf['type_id'].isin([13, 14, 15, 16])])
            if 'own goal' in tdf.columns:
                n_goals = len(tdf[(tdf['type_id'] == 16) & (tdf['own goal'] != 'Si')])
            else:
                n_goals = len(tdf[tdf['type_id'] == 16])
            n_tack  = len(tdf[tdf['type_id'] == 7])
            n_rec   = len(tdf[tdf['type_id'] == 49])
            n_press = len(tdf[(tdf['type_id'] == 49) & (tdf['x'] > 50)])
            xg  = tdf['xG'].sum() if 'xG' in tdf.columns else 0
            xga = odf['xG'].sum() if 'xG' in odf.columns else 0
            if team not in acc:
                acc[team] = dict(m=0, p=0, s=0, sh=0, g=0, t=0, r=0, pr=0, xg=0, xga=0)
            a = acc[team]
            a['m'] += 1; a['p'] += n_pass; a['s'] += n_succ; a['sh'] += n_shots
            a['g'] += n_goals; a['t'] += n_tack; a['r'] += n_rec; a['pr'] += n_press
            a['xg'] += xg; a['xga'] += xga

    rows = []
    for team, a in acc.items():
        m = max(a['m'], 1)
        rows.append(dict(team=team,
            passes_pg=a['p']/m, pass_acc=a['s']/max(a['p'],1)*100,
            shots_pg=a['sh']/m, goals_pg=a['g']/m,
            xg_pg=a['xg']/m, xga_pg=a['xga']/m,
            tackles_pg=a['t']/m, ball_rec_pg=a['r']/m, press_rec_pg=a['pr']/m))

    result = pd.DataFrame(rows).set_index('team')
    _LEAGUE_CACHE['df'] = result
    _LEAGUE_CACHE['timestamp'] = time.time()
    return result


def _build_benchmark_radar(league_df, rival):
    cache_key = f"radar_{rival}"
    if cache_key in _RADAR_CACHE:
        return _RADAR_CACHE[cache_key]

    def pct_series(col, hib):
        s = league_df[col]
        return (s.rank(pct=True) if hib else s.rank(pct=True, ascending=False)) * 100

    pct_df = pd.DataFrame(
        {col: pct_series(col, hib) for col, _, hib in RADAR_METRICS},
        index=league_df.index
    )

    N = len(RADAR_METRICS)
    labels = [lbl for _, lbl, _ in RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    def row_vals(team):
        if team not in pct_df.index:
            return [50] * N
        return pct_df.loc[team, [c for c, *_ in RADAR_METRICS]].tolist()

    goz_v   = row_vals(GOZTEPE) + row_vals(GOZTEPE)[:1]
    rival_v = row_vals(rival) + row_vals(rival)[:1]
    avg_v   = [50] * (N + 1)
    ang     = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'}, facecolor=PITCH_BG)
    ax.set_facecolor(PITCH_BG)

    for ring in [25, 50, 75, 100]:
        ax.plot(ang, [ring] * (N + 1), color='white', alpha=0.07, linewidth=0.5)

    ax.plot(ang, avg_v,   color='white',  alpha=0.30, linewidth=1.2, linestyle='--', label='League Avg')
    ax.fill(ang, avg_v,   color='white',  alpha=0.03)
    ax.plot(ang, rival_v, color='#3b82f6', linewidth=1.8, label=_clean(rival))
    ax.fill(ang, rival_v, color='#3b82f6', alpha=0.12)
    ax.plot(ang, goz_v,   color='#ef4444', linewidth=2.3, label='Göztepe')
    ax.fill(ang, goz_v,   color='#ef4444', alpha=0.18)
    ax.scatter(angles, goz_v[:-1], color='#ef4444', s=45, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=9, color='white', fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25th', '50th', '75th', '100th'], size=6.5, color=(1, 1, 1, 0.35))
    ax.spines['polar'].set_visible(False)
    ax.grid(color='white', alpha=0.07, linewidth=0.5)

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.18), fontsize=9)
    legend.get_frame().set_facecolor(PITCH_BG)
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_edgecolor((1, 1, 1, 0.2))
    for t in legend.get_texts():
        t.set_color('white')

    result = _fig_to_base64(fig)
    _RADAR_CACHE[cache_key] = result
    if len(_RADAR_CACHE) > 20:
        _RADAR_CACHE.pop(next(iter(_RADAR_CACHE)))
    return result


def _build_benchmarking_section(rival, opp_name):
    try:
        league_df = _load_league_benchmarks()
    except Exception as e:
        return html.Div(f"Benchmark data unavailable: {e}", className="goz-card-desc")

    radar_b64 = _build_benchmark_radar(league_df, rival)

    def rank_of(team, col, hib):
        if team not in league_df.index:
            return '-'
        s = league_df[col]
        return int(s.rank(ascending=not hib, method='min')[team])

    def val_of(team, col):
        if team not in league_df.index:
            return 0
        return league_df.loc[team, col]

    n_teams = len(league_df)

    metric_rows = []
    for col, label, hib, fmt in BENCH_METRICS:
        gv = val_of(GOZTEPE, col)
        rv = val_of(rival, col)
        gr = rank_of(GOZTEPE, col, hib)
        rr = rank_of(rival, col, hib)
        rival_better = (hib and rv > gv) or (not hib and rv < gv)
        delta = rv - gv if hib else gv - rv
        delta_str = f"+{abs(delta):.1f}" if delta > 0 else f"−{abs(delta):.1f}"

        metric_rows.append(html.Div(style={
            "display": "flex", "alignItems": "center", "gap": "6px",
            "padding": "7px 10px", "borderRadius": "8px", "marginBottom": "5px",
            "background": "rgba(239,68,68,0.07)" if rival_better else "rgba(34,197,94,0.05)",
            "border": "1px solid rgba(239,68,68,0.18)" if rival_better else "1px solid rgba(34,197,94,0.14)",
        }, children=[
            html.Span(label, style={"flex": "1.8", "fontSize": "0.76rem",
                "color": "var(--text-secondary)", "fontWeight": "500"}),
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Span(fmt.format(gv), style={"fontWeight": "700", "color": RED, "fontSize": "0.88rem"}),
                html.Span(f" #{gr}", style={"fontSize": "0.62rem", "color": "var(--text-secondary)"}),
            ]),
            html.Span("▲" if rival_better else "▼", style={
                "fontSize": "0.75rem", "width": "14px", "textAlign": "center",
                "color": RED if rival_better else "#22c55e",
            }),
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Span(fmt.format(rv), style={"fontWeight": "700", "color": BLUE, "fontSize": "0.88rem"}),
                html.Span(f" #{rr}", style={"fontSize": "0.62rem", "color": "var(--text-secondary)"}),
            ]),
            html.Span(delta_str, style={
                "fontSize": "0.68rem", "width": "36px", "textAlign": "right",
                "color": RED if rival_better else "#22c55e", "fontWeight": "600",
            }),
        ]))

    goz_rank_xg  = rank_of(GOZTEPE, 'xg_pg',   True)
    goz_rank_pass = rank_of(GOZTEPE, 'passes_pg', True)
    goz_rank_acc  = rank_of(GOZTEPE, 'pass_acc', True)

    return html.Div(className="goz-form-section", style={"marginTop": "24px"}, children=[
        html.Div(className="goz-section-header", style={"marginBottom": "20px"}, children=[
            html.Span("LEAGUE BENCHMARKING", className="goz-card-title"),
            html.P(
                f"Season-wide per-game averages · Göztepe vs {opp_name} vs all {n_teams} teams",
                className="goz-card-desc",
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div("STYLE PROFILE — PERCENTILE RANK", style={
                    "fontSize": "0.72rem", "fontWeight": "700", "color": GOLD,
                    "letterSpacing": "1px", "marginBottom": "8px", "textAlign": "center",
                }),
                html.Img(src=radar_b64, style={"width": "100%", "borderRadius": "8px"}),
            ], md=6),
            dbc.Col([
                html.Div(style={
                    "display": "flex", "gap": "6px", "padding": "0 4px",
                    "marginBottom": "10px", "alignItems": "center",
                }, children=[
                    html.Span("Metric", style={"flex": "1.8", "fontSize": "0.65rem",
                        "color": "var(--text-secondary)", "textTransform": "uppercase"}),
                    html.Span("Göztepe", style={"flex": "1", "textAlign": "center",
                        "fontSize": "0.65rem", "color": RED, "fontWeight": "700", "textTransform": "uppercase"}),
                    html.Span("", style={"width": "14px"}),
                    html.Span(opp_name[:10], style={"flex": "1", "textAlign": "center",
                        "fontSize": "0.65rem", "color": BLUE, "fontWeight": "700", "textTransform": "uppercase",
                        "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                    html.Span("Δ", style={"width": "36px", "textAlign": "right",
                        "fontSize": "0.65rem", "color": "var(--text-secondary)"}),
                ]),
                html.Div(children=metric_rows),
                html.Div(style={"marginTop": "16px", "padding": "10px 12px", "borderRadius": "8px",
                    "background": "rgba(255,255,255,0.03)", "border": "1px solid var(--border-color)",
                    "fontSize": "0.72rem", "color": "var(--text-secondary)", "lineHeight": "1.8"}, children=[
                    html.Span("Context: ", style={"fontWeight": "700", "color": GOLD}),
                    f"Göztepe rank #{goz_rank_xg}/18 in xG, #{goz_rank_pass}/18 in passing volume, "
                    f"#{goz_rank_acc}/18 in pass accuracy. "
                    "▲ = rival outperforms Göztepe on this metric.",
                ]),
            ], md=6),
        ]),
    ])


def _get_goztepe_anchor_week(rival, selected_file=None):
    weeks = []
    for fn, df in _get_h2h_matches(rival):
        if selected_file and fn != selected_file:
            continue
        if 'week' in df.columns and not df.empty:
            try:
                weeks.append(int(df['week'].iloc[0]))
            except Exception:
                continue
    return max(weeks) if weeks else None


def _get_rival_last5(rival, anchor_week=None):
    matches = extract_fixture_data(lite=True)
    results = []
    for m in sorted(matches, key=lambda x: x['week'], reverse=True):
        t1, t2 = m['team_names']
        if rival not in (t1, t2):
            continue
        if anchor_week is not None and int(m['week']) >= int(anchor_week):
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


def _build_form_section(rival, opp_name, selected_file=None):
    anchor_week = _get_goztepe_anchor_week(rival, selected_file)
    last5 = _get_rival_last5(rival, anchor_week)
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
        html.Div(className="goz-section-header", style={
            "flexDirection": "column",
            "alignItems": "flex-start",
            "gap": "4px",
        }, children=[
            html.Div(
                f"{opp_name} — Last 5 Matches Before Göztepe"
                if anchor_week is not None else f"{opp_name} — Last 5 Matches",
                className="goz-card-title",
                style={"margin": 0},
            ),
            html.P(
                f"Form sample ends before the selected Göztepe match week: W{anchor_week}"
                if anchor_week is not None else "Latest available match sample",
                className="goz-card-desc",
                style={"margin": 0},
            ),
        ]),
        html.Div(style={"display": "flex", "justifyContent": "center", "gap": "16px", "margin": "16px 0"}, children=items),
        html.Div(style={"textAlign": "center", "marginTop": "8px"}, children=[
            html.Span(f"{wins}W  {draws}D  {losses}L", style={
                "fontSize": "0.85rem", "fontWeight": "600", "color": "var(--text-secondary)",
                "letterSpacing": "1px",
            }),
        ]),
    ])


def _match_score(df, rival):
    goz_df = df[df['team_name'] == GOZTEPE]
    opp_df = df[df['team_name'] == rival]
    has_og = 'own goal' in df.columns
    if has_og:
        gg = len(goz_df[(goz_df['type_id'] == 16) & (goz_df['own goal'] != 'Si')]) + len(opp_df[(opp_df['type_id'] == 16) & (opp_df['own goal'] == 'Si')])
        og = len(opp_df[(opp_df['type_id'] == 16) & (opp_df['own goal'] != 'Si')]) + len(goz_df[(goz_df['type_id'] == 16) & (goz_df['own goal'] == 'Si')])
    else:
        gg = len(goz_df[goz_df['type_id'] == 16])
        og = len(opp_df[opp_df['type_id'] == 16])
    return gg, og


def _home_away_teams(df, fallback_rival):
    if 'team_position' in df.columns:
        home = df[df['team_position'] == 'home']['team_name']
        away = df[df['team_position'] == 'away']['team_name']
        if not home.empty and not away.empty:
            return home.iloc[0], away.iloc[0]

    teams = [t for t in df['team_name'].unique().tolist() if pd.notna(t)]
    home = teams[0] if teams else GOZTEPE
    away = next((t for t in teams if t != home), fallback_rival)
    return home, away


def _ordered_match_label(df, rival):
    gg, og = _match_score(df, rival)
    home, away = _home_away_teams(df, rival)
    home_goals = gg if home == GOZTEPE else og
    away_goals = og if home == GOZTEPE else gg
    return f"{_clean(home)} {home_goals}-{away_goals} {_clean(away)}", gg, og


def _build_match_options(rival):
    options = []
    res_colors = {'W': "#22c55e", 'D': BLUE, 'L': RED}
    for fn, df in sorted(_get_h2h_matches(rival), key=lambda item: int(item[1]['week'].iloc[0]) if 'week' in item[1].columns and not item[1].empty else 0):
        week = int(df['week'].iloc[0]) if 'week' in df.columns and not df.empty else 0
        date = str(df['local_date'].iloc[0]) if 'local_date' in df.columns and not df.empty else ''
        match_label, gg, og = _ordered_match_label(df, rival)
        if gg > og:
            result = 'W'
        elif gg < og:
            result = 'L'
        else:
            result = 'D'
        color = res_colors[result]
        label = html.Div(style={'textAlign': 'center', 'lineHeight': '1.3'}, children=[
            html.Div(f"Week {week}", style={
                'fontSize': '0.72rem', 'fontWeight': '700', 'color': 'var(--text-secondary)',
            }),
            html.Div(match_label, style={
                'fontSize': '0.86rem', 'fontWeight': '800', 'color': color,
            }),
            html.Div(date, style={
                'fontSize': '0.62rem', 'fontWeight': '600', 'color': 'var(--text-secondary)',
            }),
        ])
        options.append({
            'label': label,
            'value': fn,
        })
    return options


def _build_post_match_report(rival, selected_file):
    if not rival or not selected_file:
        return html.Div()
    matches = [(fn, df) for fn, df in _get_h2h_matches(rival) if fn == selected_file]
    if not matches:
        return html.Div()
    _, df = matches[0]
    opp_name = _clean(rival)
    goz_df = df[df['team_name'] == GOZTEPE]
    opp_df = df[df['team_name'] == rival]
    match_label, gg, og = _ordered_match_label(df, rival)
    week = int(df['week'].iloc[0]) if 'week' in df.columns and not df.empty else 0
    goz_logo = TEAM_LOGOS.get(GOZTEPE, "assets/logo.png")
    opp_logo = TEAM_LOGOS.get(rival, "assets/logo.png")
    shot_types = [13, 14, 15, 16]
    g_shots = len(goz_df[goz_df['type_id'].isin(shot_types)])
    o_shots = len(opp_df[opp_df['type_id'].isin(shot_types)])
    g_sot = len(goz_df[goz_df['type_id'].isin([15, 16])])
    o_sot = len(opp_df[opp_df['type_id'].isin([15, 16])])
    g_xg = round(goz_df['xG'].sum(), 2) if 'xG' in goz_df.columns else 0
    o_xg = round(opp_df['xG'].sum(), 2) if 'xG' in opp_df.columns else 0
    g_pass = len(goz_df[goz_df['type_id'] == 1])
    o_pass = len(opp_df[opp_df['type_id'] == 1])
    g_pass_ok = len(goz_df[(goz_df['type_id'] == 1) & (goz_df['outcome'] == 1)])
    o_pass_ok = len(opp_df[(opp_df['type_id'] == 1) & (opp_df['outcome'] == 1)])
    g_acc = round(g_pass_ok / max(g_pass, 1) * 100, 1)
    o_acc = round(o_pass_ok / max(o_pass, 1) * 100, 1)
    box_entries = cross_cnt = ground_cnt = 0
    if 'Pass End X' in goz_df.columns and 'Pass End Y' in goz_df.columns:
        be = goz_df[
            (goz_df['type_id'] == 1) &
            (goz_df['Pass End X'].notna()) &
            (goz_df['Pass End X'] > 83) &
            (goz_df['Pass End Y'].between(21, 79))
        ]
        box_entries = len(be)
        cross_cnt = len(be[be['Cross'] == 'Si']) if 'Cross' in be.columns else 0
        ground_cnt = box_entries - cross_cnt

        prog = goz_df[
            (goz_df['type_id'] == 1) &
            (goz_df['outcome'] == 1) &
            (goz_df['Pass End X'].notna()) &
            ((goz_df['Pass End X'] - goz_df['x']) > 12) &
            (goz_df['Pass End X'] > 60)
        ]
        if not prog.empty:
            hero_s = prog.groupby('player_name').size().sort_values(ascending=False)
            hero_name = hero_s.index[0].split()[-1]
            hero_count = int(hero_s.iloc[0])
        else:
            hero_name, hero_count = 'N/A', 0
    else:
        hero_name, hero_count = 'N/A', 0
    risky_losses = len(goz_df[
        (((goz_df['type_id'] == 1) & (goz_df['outcome'] == 0)) | (goz_df['type_id'] == 50)) &
        (goz_df['x'] < 50)
    ])

    did = []
    todo = []
    if gg >= og:
        did.append(f"Result control: {gg}-{og}.")
    if g_xg >= o_xg:
        did.append(f"Chance quality matched or beat {opp_name}: xG {g_xg} vs {o_xg}.")
    else:
        todo.append(f"Improve chance quality: xG {g_xg} vs {o_xg}.")
    if g_sot < 4:
        todo.append(f"Increase shots on target: only {g_sot}.")
    if box_entries < 12:
        todo.append(f"Create more box access: {box_entries} entries.")
    if risky_losses > 12:
        todo.append(f"Reduce high-risk losses in own half: {risky_losses}.")
    if not did:
        did.append("Kept enough data structure for a clear review.")
    if not todo:
        todo.append("Maintain the same base principles and refine final-action detail.")

    return html.Div(className="report-page", children=[
        html.Div(className="report-header", children=[
            html.Img(src="/assets/logo.png", className="report-logo"),
            html.Div([
                html.Div("tactIQ", className="report-brand"),
                html.Div("Post-Match Report", className="report-kicker"),
            ]),
        ]),
        html.H1(match_label, className="report-title"),
        html.P("Coach notes and match review for Göztepe.", className="report-subtitle"),
        html.Div(className="report-scorecard", children=[
            html.Div(className="report-score-week", children=f"Week {week}" if week else "Selected Match"),
            html.Div(className="report-scoreline", children=[
                html.Img(src=goz_logo, className="report-team-logo"),
                html.Div(className="report-score", children=f"{gg} - {og}"),
                html.Img(src=opp_logo, className="report-team-logo"),
            ]),
            html.Div(className="report-score-label", children=f"Göztepe vs {opp_name}"),
            html.Div(className="report-stat-table", children=[
                html.Div([html.Strong(str(g_xg)), html.Span("xG"), html.Strong(str(o_xg))]),
                html.Div([html.Strong(str(g_shots)), html.Span("Shots"), html.Strong(str(o_shots))]),
                html.Div([html.Strong(str(g_sot)), html.Span("On Target"), html.Strong(str(o_sot))]),
                html.Div([html.Strong(f"{g_acc}%"), html.Span("Pass Accuracy"), html.Strong(f"{o_acc}%")]),
                html.Div([html.Strong(str(g_pass)), html.Span("Passes"), html.Strong(str(o_pass))]),
            ]),
            html.Div(className="report-card-row", children=[
                html.Div(className="report-card-metric", children=[
                    html.Strong(str(box_entries)),
                    html.Span("Box Entries"),
                    html.Em(f"Cross {cross_cnt} · Ground {ground_cnt}"),
                ]),
                html.Div(className="report-card-metric", children=[
                    html.Strong(hero_name),
                    html.Span("Invisible Hero"),
                    html.Em(f"{hero_count} progressive passes"),
                ]),
            ]),
        ]),
        html.Div(className="report-section", children=[
            html.H3("Match Snapshot"),
            html.Div(className="report-grid", children=[
                html.Div(className="report-pill", children=[html.Strong(f"{gg}-{og}"), html.Span("Score")]),
                html.Div(className="report-pill", children=[html.Strong(f"{g_xg} - {o_xg}"), html.Span("xG")]),
                html.Div(className="report-pill", children=[html.Strong(f"{g_shots} - {o_shots}"), html.Span("Shots")]),
                html.Div(className="report-pill", children=[html.Strong(f"{g_sot} - {o_sot}"), html.Span("Shots on target")]),
                html.Div(className="report-pill", children=[html.Strong(f"{g_acc}%"), html.Span("Göztepe pass accuracy")]),
                html.Div(className="report-pill", children=[html.Strong(str(box_entries)), html.Span("Box entries")]),
            ]),
        ]),
        html.Div(className="report-section", children=[
            html.H3("Coach Notes"),
            html.Ul([html.Li(item) for item in did]),
        ]),
        html.Div(className="report-section", children=[
            html.H3("Next Actions"),
            html.Ul([html.Li(item) for item in todo]),
        ]),
    ])


def _status_item(text, detail=None, color=GOLD):
    return html.Div(style={
        "padding": "10px 11px",
        "borderRadius": "8px",
        "background": "rgba(255,255,255,0.035)",
        "border": f"1px solid {color}44",
        "marginBottom": "8px",
    }, children=[
        html.Div(text, style={"fontSize": "0.78rem", "fontWeight": "700", "color": "var(--text-primary)"}),
        html.Div(detail, style={"fontSize": "0.68rem", "color": "var(--text-secondary)", "marginTop": "3px"}) if detail else None,
    ])


def _status_column(title, items, color):
    safe_items = items[:5] if items else [("No strong signal in this match.", None)]
    return dbc.Col(html.Div(style={
        "height": "100%",
        "padding": "14px",
        "borderRadius": "10px",
        "background": "rgba(255,255,255,0.025)",
        "border": "1px solid var(--border-color)",
    }, children=[
        html.Div(title, style={
            "fontSize": "0.74rem", "fontWeight": "800", "color": color,
            "letterSpacing": "1px", "textTransform": "uppercase", "marginBottom": "12px",
        }),
        html.Div([_status_item(text, detail, color) for text, detail in safe_items]),
    ]), md=4)


def _build_coaching_status_section(metrics, opp_name, as_card=True):
    did = []
    todo = []
    did_not = []

    if metrics['gg'] > metrics['og']:
        did.append(("Protected the result", f"Won the match {metrics['gg']}-{metrics['og']}."))
    elif metrics['gg'] == metrics['og']:
        did.append(("Stayed in the game", f"Finished level at {metrics['gg']}-{metrics['og']}."))

    if metrics['g_xg'] >= metrics['o_xg']:
        did.append(("Matched or beat chance quality", f"xG {metrics['g_xg']} vs {metrics['o_xg']}."))
    else:
        did_not.append(("Chance quality was behind", f"xG {metrics['g_xg']} vs {metrics['o_xg']}."))

    if metrics['g_shots'] >= metrics['o_shots']:
        did.append(("Kept shot volume competitive", f"Shots {metrics['g_shots']} vs {metrics['o_shots']}."))
    else:
        did_not.append(("Did not win the shot count", f"Shots {metrics['g_shots']} vs {metrics['o_shots']}."))

    if metrics['g_sot'] >= 4:
        did.append(("Put enough shots on target", f"{metrics['g_sot']} shots on target."))
    else:
        todo.append(("Increase shots on target", f"Only {metrics['g_sot']} on target; aim for 4+."))

    if metrics['g_acc'] >= 78:
        did.append(("Moved the ball cleanly", f"Pass accuracy {metrics['g_acc']}%."))
    else:
        todo.append(("Clean up possession", f"Pass accuracy was {metrics['g_acc']}%."))

    if metrics['box_total'] >= 12:
        did.append(("Entered the box often", f"{metrics['box_total']} box entries."))
    else:
        todo.append(("Create more box access", f"{metrics['box_total']} box entries; build more cutbacks and central entries."))

    if metrics['ppda'] < 13:
        did.append(("Pressed with useful intensity", f"PPDA {metrics['ppda']}."))
    else:
        did_not.append(("Press did not bite enough", f"PPDA {metrics['ppda']} against {_clean(opp_name)}."))

    if metrics['risky_losses'] > 12:
        did_not.append(("Lost too many balls in our half", f"{metrics['risky_losses']} high-risk losses."))
        todo.append(("Reduce first-pass risk", "Use safer outlets after recoveries and restarts."))

    if metrics['g_sp_shots'] == 0:
        did_not.append(("No set-piece shot threat", "Set pieces did not produce a shot."))
        todo.append(("Turn set pieces into shots", "Add one near-post and one second-ball routine."))
    elif metrics['g_sp_shots'] < metrics['o_sp_shots']:
        todo.append(("Improve set-piece edge", f"Set-piece shots {metrics['g_sp_shots']} vs {metrics['o_sp_shots']}."))

    if metrics['gg'] <= metrics['og']:
        todo.append(("Find the decisive action", f"Result was {metrics['gg']}-{metrics['og']}; attack needs one clearer final action."))

    content = [
        html.Div(className="goz-section-header", style={"marginBottom": "16px"}, children=[
            html.Span("Coaching Checklist", className="goz-card-title"),
            html.P("What we did, what we have to do next, and what we did not do in the selected match.",
                   className="goz-card-desc"),
        ]),
        dbc.Row([
            _status_column("What We Did", did, "#22c55e"),
            _status_column("What We Have To Do", todo, GOLD),
            _status_column("What We Did Not", did_not, RED),
        ], className="g-3"),
    ]

    if not as_card:
        return html.Div(children=content)

    return html.Div(className="goz-form-section", style={"marginBottom": "24px"}, children=content)


def _build_h2h_section(rival, opp_name, active_tab, selected_file=None, transition_filter='all'):
    h2h = _get_h2h_matches(rival)
    if selected_file:
        h2h = [(fn, df) for fn, df in h2h if fn == selected_file]
    if not h2h:
        return html.Div(className="goz-form-section", children=[
            html.Div("No selected head-to-head match found.", className="goz-card-desc"),
        ])

    try:
        from utils.xg_model import predict_xg
    except ImportError:
        predict_xg = lambda d: d

    sections = []
    goz_short = _clean(GOZTEPE)
    for fn, df in h2h:
        df = predict_xg(df)
        week = int(df['week'].iloc[0]) if 'week' in df.columns else 0
        goz_df = df[df['team_name'] == GOZTEPE]
        opp_df = df[df['team_name'] == rival]
        gg, og = _match_score(df, rival)
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
        len(goz_df[goz_df['type_id'] == 4])
        len(opp_df[opp_df['type_id'] == 4])
        _count_corner_deliveries(goz_df)
        _count_corner_deliveries(opp_df)
        g_xg = round(goz_df['xG'].sum(), 2) if 'xG' in goz_df.columns else 0
        o_xg = round(opp_df['xG'].sum(), 2) if 'xG' in opp_df.columns else 0

        # PPDA
        opp_passes_their_half = len(opp_df[(opp_df['type_id'] == 1) & (opp_df['x'] < 50)])
        goz_press = len(goz_df[goz_df['type_id'].isin([7, 8, 4]) & (goz_df['x'] > 50)])
        ppda = round(opp_passes_their_half / max(goz_press, 1), 1)

        # Box Entries
        has_pe = 'Pass End X' in goz_df.columns and 'Pass End Y' in goz_df.columns
        if has_pe:
            be = goz_df[
                (goz_df['type_id'] == 1) &
                (goz_df['Pass End X'].notna()) &
                (goz_df['Pass End X'] > 83) &
                (goz_df['Pass End Y'].between(21, 79))
            ]
            box_total = len(be)
            cross_cnt = len(be[be['Cross'] == 'Si']) if 'Cross' in be.columns else 0
            ground_cnt = box_total - cross_cnt
        else:
            box_total = cross_cnt = ground_cnt = 0

        # Invisible Hero
        if has_pe:
            prog = goz_df[
                (goz_df['type_id'] == 1) &
                (goz_df['outcome'] == 1) &
                (goz_df['Pass End X'].notna()) &
                ((goz_df['Pass End X'] - goz_df['x']) > 12) &
                (goz_df['Pass End X'] > 60)
            ]
            if not prog.empty:
                hero_s = prog.groupby('player_name').size().sort_values(ascending=False)
                hero_name = hero_s.index[0].split()[-1]
                hero_count = int(hero_s.iloc[0])
            else:
                hero_name, hero_count = 'N/A', 0
        else:
            hero_name, hero_count = 'N/A', 0

        risky_losses = len(goz_df[
            (((goz_df['type_id'] == 1) & (goz_df['outcome'] == 0)) | (goz_df['type_id'] == 50)) &
            (goz_df['x'] < 50)
        ])
        g_sp_df_all = goz_df[(goz_df['type_id'].isin(shot_types)) & ((goz_df['Set piece'] == 'Si') | (goz_df['type_id'] == 9))] if 'Set piece' in goz_df.columns else pd.DataFrame()
        o_sp_df_all = opp_df[(opp_df['type_id'].isin(shot_types)) & ((opp_df['Set piece'] == 'Si') | (opp_df['type_id'] == 9))] if 'Set piece' in opp_df.columns else pd.DataFrame()
        checklist_metrics = {
            'gg': gg, 'og': og, 'g_xg': g_xg, 'o_xg': o_xg,
            'g_shots': g_shots, 'o_shots': o_shots, 'g_sot': g_sot,
            'g_acc': g_acc, 'box_total': box_total, 'ppda': ppda,
            'risky_losses': risky_losses, 'g_sp_shots': len(g_sp_df_all),
            'o_sp_shots': len(o_sp_df_all),
        }

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

        # KPI PILL helper
        def _pill(label, value, color=GOLD, sub=None):
            return html.Div(style={
                "padding": "10px 14px", "borderRadius": "10px",
                "background": "rgba(255,255,255,0.04)",
                "border": f"1px solid {color}44",
                "textAlign": "center", "minWidth": "110px", "flex": "1",
            }, children=[
                html.Div(str(value), style={"fontSize": "1.45rem", "fontWeight": "700", "color": color}),
                html.Div(label, style={"fontSize": "0.6rem", "color": "rgba(255,255,255,0.45)",
                                       "textTransform": "uppercase", "letterSpacing": "0.5px", "marginTop": "2px"}),
                html.Div(sub, style={"fontSize": "0.6rem", "color": "rgba(255,255,255,0.3)", "marginTop": "1px"}) if sub else None,
            ])

        tab_widgets = []

        if active_tab == "checklist-tab":
            tab_widgets = [
                _build_coaching_status_section(checklist_metrics, rival, as_card=False)
            ]

        elif active_tab == "offensive-tab":
            shot_map_b64 = _build_post_match_shot_map(goz_df, opp_df, goz_short, opp_name, g_xg, o_xg)

            # Zone activity map
            zone_map_b64 = _build_zone_map_img(goz_df, opp_df, goz_short, opp_name)

            # Box entry map
            box_entry_b64 = _build_box_entry_map(goz_df, opp_df, goz_short, opp_name)

            # Channel Usage
            passes = goz_df[(goz_df['type_id'] == 1) & goz_df['y'].notna()]
            l_cnt = len(passes[passes['y'] < 33])
            c_cnt = len(passes[(passes['y'] >= 33) & (passes['y'] < 67)])
            r_cnt = len(passes[passes['y'] >= 67])
            tot = max(l_cnt + c_cnt + r_cnt, 1)
            fig_ch = go.Figure(go.Bar(
                x=['Left', 'Centre', 'Right'],
                y=[l_cnt/tot*100, c_cnt/tot*100, r_cnt/tot*100],
                marker_color=[BLUE, GOLD, RED],
                text=[f"{v/tot*100:.0f}%" for v in [l_cnt, c_cnt, r_cnt]],
                textposition='outside', textfont=dict(color='white', size=9),
            ))
            fig_ch.update_layout(
                plot_bgcolor=PITCH_BG, paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=5, r=5, t=15, b=25), height=175,
                xaxis=dict(showgrid=False, tickfont=dict(color='rgba(255,255,255,0.55)', size=8)),
                yaxis=dict(range=[0, 115], showgrid=False, visible=False),
            )

            tab_widgets = [
                html.Div(style={"maxWidth": "420px", "margin": "0 auto 20px"}, children=[
                    _stat_row("xG", g_xg, o_xg),
                    _stat_row("Shots", g_shots, o_shots),
                    _stat_row("On Target", g_sot, o_sot),
                    _stat_row("Pass Accuracy", f"{g_acc}%", f"{o_acc}%"),
                    _stat_row("Passes", g_passes, o_passes),
                ]),
                html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "18px"}, children=[
                    _pill("Box Entries", box_total, GOLD, f"Cross {cross_cnt} · Ground {ground_cnt}"),
                    _pill("Invisible Hero", hero_name, BLUE, f"{hero_count} progressive passes"),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div("SHOT QUALITY MAP", style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": GOLD, "letterSpacing": "1px", "marginBottom": "6px", "textAlign": "center"}),
                        html.Img(src=shot_map_b64, style={"width": "100%", "maxWidth": "100%", "borderRadius": "8px"}),
                    ], md=6),
                    dbc.Col([
                        html.Div("BOX ENTRY MAP", style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": GOLD, "letterSpacing": "1px", "marginBottom": "3px", "textAlign": "center"}),
                        html.Div("Pass start positions · Gold = ground · Blue = cross",
                                 style={"fontSize": "0.58rem", "color": "rgba(255,255,255,0.3)",
                                        "textAlign": "center", "marginBottom": "6px"}),
                        html.Img(src=box_entry_b64, style={"width": "100%", "borderRadius": "8px"}),
                    ], md=6),
                ]),
                dbc.Row(style={"marginTop": "20px"}, children=[
                    dbc.Col([
                        html.Div("ZONE ACTIVITY MAP", style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": GOLD, "letterSpacing": "1px", "marginBottom": "6px",
                            "textAlign": "center"}),
                        html.Img(src=zone_map_b64, style={"width": "100%", "borderRadius": "8px"}),
                    ], md=6),
                    dbc.Col([
                        html.Div("CHANNEL USAGE", style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": GOLD, "letterSpacing": "1px", "textAlign": "center", "marginBottom": "3px"}),
                        html.Div("% of passes by pitch corridor", style={"fontSize": "0.58rem",
                            "color": "rgba(255,255,255,0.3)", "textAlign": "center", "marginBottom": "5px"}),
                        dcc.Graph(figure=fig_ch, config={'displayModeBar': False}, style={'width': '100%'}),
                    ], md=6),
                ]),
            ]

        elif active_tab == "defensive-tab":
            try:
                from pages.rival_scout import _build_defensive as _build_rival_defensive
                tab_widgets = [
                    _build_rival_defensive([(fn, df)], GOZTEPE, goz_short)
                ]
            except Exception as e:
                tab_widgets = [
                    html.Div(className="goz-form-section", children=[
                        html.Div("Defensive analysis could not be loaded", className="goz-card-title"),
                        html.P(str(e), className="goz-card-desc"),
                    ])
                ]

        elif active_tab == "off-trans-tab":
            try:
                from pages.rival_scout import _build_off_transitions as _build_rival_off_transitions
                tab_widgets = [
                    _build_rival_off_transitions([(fn, df)], GOZTEPE, goz_short, transition_filter)
                ]
            except Exception as e:
                tab_widgets = [
                    html.Div(className="goz-form-section", children=[
                        html.Div("Offensive transition analysis could not be loaded", className="goz-card-title"),
                        html.P(str(e), className="goz-card-desc"),
                    ])
                ]

        elif active_tab == "def-trans-tab":
            try:
                from pages.rival_scout import _build_def_transitions as _build_rival_def_transitions
                tab_widgets = [
                    _build_rival_def_transitions([(fn, df)], GOZTEPE, goz_short, transition_filter)
                ]
            except Exception as e:
                tab_widgets = [
                    html.Div(className="goz-form-section", children=[
                        html.Div("Defensive transition analysis could not be loaded", className="goz-card-title"),
                        html.P(str(e), className="goz-card-desc"),
                    ])
                ]

        elif active_tab == "set-pieces-tab":
            try:
                from göztepehub.pages.rival_scout import _build_set_pieces as _build_rival_set_pieces
                tab_widgets = [
                    _build_rival_set_pieces([(fn, df)], GOZTEPE, goz_short)
                ]
            except Exception as e:
                tab_widgets = [
                    html.Div(className="goz-form-section", children=[
                        html.Div("Set-piece analysis could not be loaded", className="goz-card-title"),
                        html.P(str(e), className="goz-card-desc"),
                    ])
                ]

        match_card = html.Div(className="goz-form-section", style={"marginBottom": "24px"}, children=[
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
        ] + tab_widgets)
        sections.append(match_card)

    return html.Div(children=[
        html.Div(className="goz-section-header", style={"marginBottom": "16px"}, children=[
            html.Span(f"Göztepe vs {opp_name} — Selected Match", className="goz-card-title"),
        ]),
    ] + sections)


def _fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1b0f', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"


def _make_mplsoccer_pitch(half=False):
    pitch = Pitch(pitch_type='opta', pitch_color='#0e1b0f', line_color=(1.0, 1.0, 1.0, 0.55), linewidth=1.5, half=half)
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.patch.set_facecolor('#0e1b0f')
    return pitch, fig, ax


def _build_post_match_shot_map(goz_df, opp_df, goz_short, opp_name, g_xg, o_xg):
    pitch = Pitch(
        pitch_type='opta',
        pitch_color='#0e1b0f',
        line_color=(1.0, 1.0, 1.0, 0.52),
        linewidth=1.5,
        half=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor='#0e1b0f')

    def normalize_shots(df):
        shots = df[df['type_id'].isin([13, 14, 15, 16])].copy()
        shots = shots.dropna(subset=['x', 'y'])
        if shots.empty:
            return shots
        left_half = shots['x'] < 50
        shots.loc[left_half, 'x'] = 100 - shots.loc[left_half, 'x']
        shots.loc[left_half, 'y'] = 100 - shots.loc[left_half, 'y']
        return shots

    def draw_panel(ax, df, title, team_color, saved_color, total_xg):
        pitch.draw(ax=ax)
        shots = normalize_shots(df)
        goals = shots[shots['event'] == 'Goal']
        saved = shots[shots['event'] == 'Saved Shot']
        misses = shots[shots['event'].isin(['Miss', 'Post'])]

        groups = [
            (misses, (1, 1, 1, 0.42), 'o', 'Miss / Post', 0.65),
            (saved, saved_color, 'o', 'Saved', 0.86),
            (goals, team_color, '*', 'Goal', 0.95),
        ]
        for grp, color, marker, label, alpha in groups:
            if grp.empty:
                continue
            xg_vals = grp['xG'].fillna(0.06).astype(float) if 'xG' in grp.columns else pd.Series([0.06] * len(grp), index=grp.index)
            sizes = [max(55, min(360, 70 + val * 650)) for val in xg_vals]
            pitch.scatter(
                grp['x'].values, grp['y'].values,
                s=sizes, color=color, marker=marker, alpha=alpha,
                edgecolors='white', linewidth=0.7, ax=ax, zorder=5,
            )

        ax.set_title(title.upper(), color=team_color, fontsize=12, fontweight='bold', pad=10)
        ax.text(75, 101.5, f"{len(shots)} shots  |  {len(goals)} goals  |  xG {total_xg}",
                color='white', fontsize=8, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#111827', edgecolor='none', alpha=0.85))
        if shots.empty:
            ax.text(75, 50, 'No shots', color='white', alpha=0.65,
                    fontsize=12, fontweight='bold', ha='center', va='center')

    draw_panel(axes[0], goz_df, goz_short, GOLD, BLUE, g_xg)
    draw_panel(axes[1], opp_df, opp_name, RED, PURPLE, o_xg)

    legend_items = [
        (GOLD, '*', 'Goal'),
        (BLUE, 'o', 'Saved / On target'),
        ((1, 1, 1, 0.42), 'o', 'Miss / Post'),
    ]
    for i, (color, marker, label) in enumerate(legend_items):
        x = 0.34 + i * 0.16
        fig.text(x, 0.045, marker, color=color, fontsize=14, ha='right', va='center', fontweight='bold')
        fig.text(x + 0.008, 0.045, label, color='white', alpha=0.72, fontsize=8, ha='left', va='center')
    fig.text(0.5, 0.01, 'Shot size reflects xG · all shots normalized toward the same goal',
             color='white', alpha=0.45, fontsize=8, ha='center')

    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.12, wspace=0.08)
    return _fig_to_base64(fig)


def _mpl_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(handles, labels, loc='upper left', fontsize=8)
    frame = legend.get_frame()
    frame.set_facecolor('#0e1b0f')
    frame.set_alpha(0.75)
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')


def _build_coach_report(goz_df, opp_df, goz_short, opp_name):

    # ── PPDA (lower = more aggressive press) ─────────────────────────────
    opp_passes_their_half = len(opp_df[(opp_df['type_id'] == 1) & (opp_df['x'] < 50)])
    goz_press = len(goz_df[goz_df['type_id'].isin([7, 8, 4]) & (goz_df['x'] > 50)])
    ppda = round(opp_passes_their_half / max(goz_press, 1), 1)

    # ── BOX ENTRIES ───────────────────────────────────────────────────────
    has_pe = 'Pass End X' in goz_df.columns and 'Pass End Y' in goz_df.columns
    if has_pe:
        be = goz_df[
            (goz_df['type_id'] == 1) &
            (goz_df['Pass End X'].notna()) &
            (goz_df['Pass End X'] > 83) &
            (goz_df['Pass End Y'].between(21, 79))
        ]
        box_total = len(be)
        cross_cnt = len(be[be['Cross'] == 'Si']) if 'Cross' in be.columns else 0
        ground_cnt = box_total - cross_cnt
    else:
        box_total = cross_cnt = ground_cnt = 0

    # ── INVISIBLE HERO (progressive attacking passes) ─────────────────────
    if has_pe:
        prog = goz_df[
            (goz_df['type_id'] == 1) &
            (goz_df['outcome'] == 1) &
            (goz_df['Pass End X'].notna()) &
            ((goz_df['Pass End X'] - goz_df['x']) > 12) &
            (goz_df['Pass End X'] > 60)
        ]
        if not prog.empty:
            hero_s = prog.groupby('player_name').size().sort_values(ascending=False)
            hero_name = hero_s.index[0].split()[-1]
            hero_count = int(hero_s.iloc[0])
        else:
            hero_name, hero_count = 'N/A', 0
    else:
        hero_name, hero_count = 'N/A', 0

    # ── FATIGUE CURVE (pass accuracy per 15-min window) ───────────────────
    windows = [('0-15', 0, 15), ('15-30', 15, 30), ('30-45', 30, 45),
               ('45-60', 45, 60), ('60-75', 60, 75), ('75+', 75, 200)]
    f_labels, f_acc = [], []
    for lbl, s, e in windows:
        w = goz_df[(goz_df['time_min'] >= s) & (goz_df['time_min'] < e) & (goz_df['type_id'] == 1)]
        acc = round(len(w[w['outcome'] == 1]) / max(len(w), 1) * 100, 1)
        f_labels.append(lbl); f_acc.append(acc)

    baseline = f_acc[0] if f_acc[0] > 0 else 75
    f_colors = [GOLD if a >= baseline - 4 else RED for a in f_acc]
    fig_fat = go.Figure(go.Bar(
        x=f_labels, y=f_acc, marker_color=f_colors,
        text=[f"{a}%" for a in f_acc], textposition='outside',
        textfont=dict(color='white', size=9),
    ))
    fig_fat.update_layout(
        plot_bgcolor=PITCH_BG, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=5, t=15, b=25), height=175,
        xaxis=dict(showgrid=False, tickfont=dict(color='rgba(255,255,255,0.55)', size=8)),
        yaxis=dict(range=[0, 115], showgrid=False, visible=False),
    )

    # ── BALL LOSS MAP ─────────────────────────────────────────────────────
    losses = goz_df[
        ((goz_df['type_id'] == 1) & (goz_df['outcome'] == 0)) |
        (goz_df['type_id'] == 50)
    ][['x', 'y']].dropna()

    pitch_l, fig_l, ax_l = _make_mplsoccer_pitch()
    if not losses.empty:
        safe = losses[losses['x'] >= 50]
        risky = losses[losses['x'] < 50]
        if not safe.empty:
            pitch_l.scatter(safe['x'].values, safe['y'].values,
                            s=28, color=GOLD, alpha=0.45, ax=ax_l, edgecolors='none')
        if not risky.empty:
            pitch_l.scatter(risky['x'].values, risky['y'].values,
                            s=40, color=RED, alpha=0.8, ax=ax_l,
                            edgecolors='white', linewidth=0.4, label='High-risk loss')
    _mpl_legend(ax_l)
    loss_map_b64 = _fig_to_base64(fig_l)

    # ── CHANNEL USAGE ─────────────────────────────────────────────────────
    passes = goz_df[(goz_df['type_id'] == 1) & goz_df['y'].notna()]
    l_cnt = len(passes[passes['y'] < 33])
    c_cnt = len(passes[(passes['y'] >= 33) & (passes['y'] < 67)])
    r_cnt = len(passes[passes['y'] >= 67])
    tot = max(l_cnt + c_cnt + r_cnt, 1)
    fig_ch = go.Figure(go.Bar(
        x=['Left', 'Centre', 'Right'],
        y=[l_cnt/tot*100, c_cnt/tot*100, r_cnt/tot*100],
        marker_color=[BLUE, GOLD, RED],
        text=[f"{v/tot*100:.0f}%" for v in [l_cnt, c_cnt, r_cnt]],
        textposition='outside', textfont=dict(color='white', size=9),
    ))
    fig_ch.update_layout(
        plot_bgcolor=PITCH_BG, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=5, t=15, b=25), height=175,
        xaxis=dict(showgrid=False, tickfont=dict(color='rgba(255,255,255,0.55)', size=8)),
        yaxis=dict(range=[0, 115], showgrid=False, visible=False),
    )

    # ── PASS DISTANCE — PANIC BUTTON (1H vs 2H) ───────────────────────────
    def avg_dist(df, t0, t1):
        if not has_pe:
            return 0.0
        w = df[(df['type_id'] == 1) & (df['time_min'] >= t0) & (df['time_min'] < t1)].copy()
        w = w[w['Pass End X'].notna() & w['Pass End Y'].notna()]
        if w.empty:
            return 0.0
        return round(np.sqrt((w['Pass End X'] - w['x'])**2 + (w['Pass End Y'] - w['y'])**2).mean(), 1)

    d1h = avg_dist(goz_df, 0, 45)
    d2h = avg_dist(goz_df, 45, 200)
    d_delta = round(d2h - d1h, 1)
    d_color = RED if d_delta > 5 else GOLD if d_delta > 0 else "#22c55e"

    # ── KPI PILL helper ──────────────────────────────────────────────────
    def _pill(label, value, color=GOLD, sub=None):
        return html.Div(style={
            "padding": "10px 14px", "borderRadius": "10px",
            "background": "rgba(255,255,255,0.04)",
            "border": f"1px solid {color}44",
            "textAlign": "center", "minWidth": "110px", "flex": "1",
        }, children=[
            html.Div(str(value), style={"fontSize": "1.45rem", "fontWeight": "700", "color": color}),
            html.Div(label, style={"fontSize": "0.6rem", "color": "rgba(255,255,255,0.45)",
                                   "textTransform": "uppercase", "letterSpacing": "0.5px", "marginTop": "2px"}),
            html.Div(sub, style={"fontSize": "0.6rem", "color": "rgba(255,255,255,0.3)", "marginTop": "1px"}) if sub else None,
        ])

    ppda_color = "#22c55e" if ppda < 8 else GOLD if ppda < 13 else RED

    return html.Div(style={"marginTop": "22px", "padding": "16px 14px",
                           "background": "rgba(255,255,255,0.02)",
                           "border": "1px solid rgba(255,185,0,0.15)",
                           "borderRadius": "12px"}, children=[
        html.Div("TACTICAL INTELLIGENCE REPORT", style={
            "fontSize": "0.68rem", "fontWeight": "700", "color": GOLD,
            "letterSpacing": "2px", "marginBottom": "14px", "textAlign": "center",
        }),
        html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                        "marginBottom": "18px"}, children=[
            _pill("PPDA", ppda, ppda_color, "lower = sharper press"),
            _pill("Box Entries", box_total, GOLD, f"Cross {cross_cnt} · Ground {ground_cnt}"),
            _pill("Invisible Hero", hero_name, BLUE, f"{hero_count} progressive passes"),
            _pill("Pass Dist Δ (2H−1H)", f"{'+' if d_delta >= 0 else ''}{d_delta}", d_color, "panic signal if rising"),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div("BALL LOSSES", style={"fontSize": "0.63rem", "fontWeight": "700",
                    "color": GOLD, "letterSpacing": "1px", "textAlign": "center", "marginBottom": "3px"}),
                html.Div("Red = own half (high-risk)", style={"fontSize": "0.58rem",
                    "color": "rgba(255,255,255,0.3)", "textAlign": "center", "marginBottom": "5px"}),
                html.Img(src=loss_map_b64, style={"width": "100%", "borderRadius": "8px"}),
            ], md=4),
            dbc.Col([
                html.Div("FATIGUE CURVE — PASS ACCURACY", style={"fontSize": "0.63rem",
                    "fontWeight": "700", "color": GOLD, "letterSpacing": "1px",
                    "textAlign": "center", "marginBottom": "3px"}),
                html.Div("Drop after 60' = fitness / structure breakdown", style={"fontSize": "0.58rem",
                    "color": "rgba(255,255,255,0.3)", "textAlign": "center", "marginBottom": "5px"}),
                dcc.Graph(figure=fig_fat, config={'displayModeBar': False}, style={'width': '100%'}),
            ], md=4),
            dbc.Col([
                html.Div("CHANNEL USAGE", style={"fontSize": "0.63rem", "fontWeight": "700",
                    "color": GOLD, "letterSpacing": "1px", "textAlign": "center", "marginBottom": "3px"}),
                html.Div("% of passes by pitch corridor", style={"fontSize": "0.58rem",
                    "color": "rgba(255,255,255,0.3)", "textAlign": "center", "marginBottom": "5px"}),
                dcc.Graph(figure=fig_ch, config={'displayModeBar': False}, style={'width': '100%'}),
            ], md=4),
        ]),
    ])


def _build_xg_timeline(goz_df, opp_df, goz_short, opp_name):
    SHOT_TYPES = [13, 14, 15, 16]

    def cumulative(df):
        s = df[df['type_id'].isin(SHOT_TYPES)].copy()
        s = s[['time_min', 'xG', 'type_id']].dropna(subset=['xG'])
        s = s.sort_values('time_min')
        s['cumxG'] = s['xG'].cumsum()
        return s

    goz_s = cumulative(goz_df)
    opp_s = cumulative(opp_df)
    max_min = max(
        goz_df['time_min'].max() if not goz_df.empty else 90,
        opp_df['time_min'].max() if not opp_df.empty else 90,
        90
    )

    fig = go.Figure()

    for shots, color, label in [(goz_s, GOLD, 'Göztepe'), (opp_s, RED, opp_name)]:
        if shots.empty:
            continue
        last_xg = shots['cumxG'].iloc[-1]
        x_vals = [0] + shots['time_min'].tolist() + [max_min]
        y_vals = [0] + shots['cumxG'].tolist() + [last_xg]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name=f'{label} xG',
            line=dict(color=color, width=2.5, shape='hv'),
        ))
        goals = shots[shots['type_id'] == 16]
        if not goals.empty:
            fig.add_trace(go.Scatter(
                x=goals['time_min'], y=goals['cumxG'],
                mode='markers',
                name=f'{label} Gol',
                marker=dict(color=color, size=11, symbol='star',
                            line=dict(color='white', width=1)),
                showlegend=True,
            ))

    fig.add_vline(x=45, line_dash='dash',
                  line_color='rgba(255,255,255,0.25)', line_width=1)

    fig.update_layout(
        plot_bgcolor=PITCH_BG, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=10, t=10, b=30), height=220,
        xaxis=dict(
            title='Dakika', color='rgba(255,255,255,0.5)',
            showgrid=False, zeroline=False,
            range=[0, max_min + 2], tickfont=dict(size=9),
        ),
        yaxis=dict(
            title='Cumulative xG', color='rgba(255,255,255,0.5)',
            showgrid=True, gridcolor='rgba(255,255,255,0.07)',
            zeroline=False, tickfont=dict(size=9), rangemode='tozero',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            font=dict(color='rgba(255,255,255,0.8)', size=9),
            bgcolor='rgba(0,0,0,0)',
        ),
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False},
                     style={'width': '100%'})


def _build_box_entry_map(goz_df, opp_df, goz_short, opp_name):
    BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 83, 21, 79

    def get_entries(df):
        if 'Pass End X' not in df.columns or 'Pass End Y' not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
        ent = df[
            (df['type_id'] == 1) &
            (df['outcome'] == 1) &
            (df['Pass End X'].notna()) &
            (df['Pass End X'] > BOX_X_MIN) &
            (df['Pass End Y'].between(BOX_Y_MIN, BOX_Y_MAX)) &
            ~((df['x'] > BOX_X_MIN) & (df['y'].between(BOX_Y_MIN, BOX_Y_MAX)))
        ].copy()
        crosses = ent[ent['Cross'] == 'Si'] if 'Cross' in ent.columns else pd.DataFrame()
        ground  = ent[ent['Cross'] != 'Si'] if 'Cross' in ent.columns else ent
        return ground, crosses

    goz_gnd, goz_cross = get_entries(goz_df)
    opp_gnd, opp_cross = get_entries(opp_df)

    pitch = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
                  line_color=(1.0, 1.0, 1.0, 0.55), linewidth=1.5, half=True)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(14, 5))
    fig.patch.set_facecolor(PITCH_BG)

    total_goz = len(goz_gnd) + len(goz_cross)
    total_opp = len(opp_gnd) + len(opp_cross)

    for ax, gnd, crs, base_color, label in [
        (axes[0], goz_gnd, goz_cross, GOLD, f'Göztepe — {total_goz} box entry'),
        (axes[1], opp_gnd, opp_cross, RED,  f'{opp_name} — {total_opp} box entry'),
    ]:
        ax.set_facecolor(PITCH_BG)
        ax.set_title(label, color='white', fontsize=9, pad=5, fontweight='bold')
        if not gnd.empty:
            pitch.scatter(gnd['x'].values, gnd['y'].values, s=55, color=base_color,
                          alpha=0.75, ax=ax, edgecolors='white', linewidth=0.5,
                          label='Ground', zorder=3)
        if not crs.empty:
            pitch.scatter(crs['x'].values, crs['y'].values, s=55, color=BLUE,
                          alpha=0.75, ax=ax, edgecolors='white', linewidth=0.5,
                          label='Cross', zorder=3)
        _mpl_legend(ax)

    plt.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def _build_pressing_map(goz_df, opp_df, goz_short, opp_name):
    PRESS_TYPES = [49, 7, 8]  # Ball recovery, Tackle, Interception

    goz_p = goz_df[goz_df['type_id'].isin(PRESS_TYPES)][['x', 'y']].dropna()
    opp_p = opp_df[opp_df['type_id'].isin(PRESS_TYPES)][['x', 'y']].dropna()

    pitch = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
                  line_color=(1.0, 1.0, 1.0, 0.55), linewidth=1.5)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(14, 5))
    fig.patch.set_facecolor(PITCH_BG)

    for ax, team_p, cmap, label in [
        (axes[0], goz_p, 'YlOrBr', f'Göztepe — {len(goz_p)} pressing actions'),
        (axes[1], opp_p, 'Reds',   f'{opp_name} — {len(opp_p)} pressing actions'),
    ]:
        ax.set_facecolor(PITCH_BG)
        ax.set_title(label, color='white', fontsize=9, pad=5, fontweight='bold')
        if not team_p.empty:
            pitch.kdeplot(team_p['x'].values, team_p['y'].values,
                          ax=ax, cmap=cmap, levels=60, fill=True, alpha=0.75)

    plt.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def _build_zone_map_img(goz_df, opp_df, goz_short, opp_name):
    EXCLUDE = {32, 34, 30, 31, 35, 37, 38}

    def clean(df):
        return df[df['x'].between(0.5, 99.5) & df['y'].between(0.5, 99.5) & ~df['type_id'].isin(EXCLUDE)]

    gd, od = clean(goz_df), clean(opp_df)
    COLS, ROWS = 6, 5

    def count_grid(df):
        if df.empty:
            return np.zeros((ROWS, COLS), dtype=int)
        xi = np.clip((df['x'].values / 100 * COLS).astype(int), 0, COLS - 1)
        yi = np.clip((df['y'].values / 100 * ROWS).astype(int), 0, ROWS - 1)
        g = np.zeros((ROWS, COLS), dtype=int)
        np.add.at(g, (yi, xi), 1)
        return g

    goz_g = count_grid(gd)
    opp_g = count_grid(od)
    n_goz, n_opp = len(gd), len(od)
    poss_goz = round(n_goz / max(n_goz + n_opp, 1) * 100)
    poss_opp = 100 - poss_goz

    GOZ_RGB = np.array([0.545, 0.082, 0.118])
    OPP_RGB = np.array([0.102, 0.231, 0.420])

    fig = plt.figure(figsize=(13, 9.5), facecolor='#b4c8dc')
    ax_pitch = fig.add_axes([0, 0.12, 1, 0.88])
    pitch = Pitch(pitch_type='opta', pitch_color='#3a6e3a', line_color=(1, 1, 1, 0.45), linewidth=1.5)
    pitch.draw(ax=ax_pitch)

    x_edges = np.linspace(0, 100, COLS + 1)
    y_edges = np.linspace(0, 100, ROWS + 1)

    for ri in range(ROWS):
        for ci in range(COLS):
            x0, x1 = x_edges[ci], x_edges[ci + 1]
            y0, y1 = y_edges[ri], y_edges[ri + 1]
            gc, oc = int(goz_g[ri, ci]), int(opp_g[ri, ci])
            tot = gc + oc
            if tot == 0:
                face = (0.22, 0.44, 0.22, 0.50)
            else:
                rgb = (gc / tot) * GOZ_RGB + (oc / tot) * OPP_RGB
                face = (*rgb.tolist(), 0.82)
            ax_pitch.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=face, edgecolor='white', linewidth=0.7, zorder=2
            ))
            ax_pitch.text(
                (x0 + x1) / 2, (y0 + y1) / 2, f"{gc} / {oc}",
                ha='center', va='center', fontsize=9.5,
                color='white', fontweight='bold', zorder=3
            )

    ax_bar = fig.add_axes([0.04, 0.01, 0.92, 0.10])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    ax_bar.set_facecolor('#1a2535')

    ax_bar.add_patch(mpatches.Rectangle((0, 0.1), poss_goz, 0.8, facecolor='#7a1520', zorder=2))
    ax_bar.add_patch(mpatches.Rectangle((poss_goz, 0.1), poss_opp, 0.8, facecolor='#1a3b6b', zorder=2))

    ax_bar.text(poss_goz / 2, 0.5, f'{poss_goz}%',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=3)
    ax_bar.text(poss_goz + poss_opp / 2, 0.5, f'{poss_opp}%',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=3)

    ax_bar.text(8, 0.5, '→', ha='center', va='center', fontsize=26, color='#f5d060', alpha=0.75, zorder=4)
    ax_bar.text(92, 0.5, '←', ha='center', va='center', fontsize=26, color='white', alpha=0.75, zorder=4)
    ax_bar.text(1, 0.15, goz_short, ha='left', va='bottom', fontsize=8, color='#f5d060', fontweight='bold', zorder=3)
    ax_bar.text(99, 0.15, opp_name, ha='right', va='bottom', fontsize=8, color='white', fontweight='bold', zorder=3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='#b4c8dc', edgecolor='none')
    buf.seek(0)
    img_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    plt.close(fig)
    return img_b64


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
                        options=[{'label': f"{i+1}. {_clean(r)}", 'value': r} for i, r in enumerate(rivals)],
                        value=rivals[0] if rivals else None,
                        className="goz-dropdown",
                        clearable=False,
                        searchable=True,
                        placeholder="Select opponent...",
                    ),
                ]),
                html.Div(className="report-actions", children=[
                    html.Button("Report", type="button", className="btn-print btn-report-print"),
                ]),
            ]),
        ]),
        dcc.Loading(
            id="post-match-loading",
            type="circle",
            color=GOLD,
            children=html.Div(className="content-container", style={"padding": "0 20px 60px"}, children=[
                html.Div(id="post-match-report-container", className="report-only"),
                html.Div(className="report-screen", children=[
                html.Div(style={'margin': '28px 0 16px'}, children=[
                    html.Label('SELECT MATCH', className='goz-label',
                               style={'marginBottom': '10px', 'display': 'block'}),
                    dbc.RadioItems(
                        id='post-match-match-selector',
                        options=[],
                        value=None,
                        inline=True,
                        className='pm-tab-radio-group',
                        inputClassName='pm-tab-radio-input',
                        labelClassName='pm-tab-radio-label',
                        style={'gap': '10px'},
                    ),
                ]),
                html.Div(id='post-match-form-container', style={"marginTop": "24px"}),
                html.Div(style={"display": "flex", "justifyContent": "center", "margin": "30px 0"}, children=[
                    dbc.RadioItems(
                        id="post-match-tabs",
                        options=[
                            {"label": "📋 Checklist",        "value": "checklist-tab"},
                            {"label": "⚔️ Offensive",        "value": "offensive-tab"},
                            {"label": "🛡️ Defensive",        "value": "defensive-tab"},
                            {"label": "⚡ Off. Transitions",  "value": "off-trans-tab"},
                            {"label": "🔄 Def. Transitions",  "value": "def-trans-tab"},
                            {"label": "🎯 Set Pieces",        "value": "set-pieces-tab"},
                        ],
                        value="checklist-tab",
                        inline=True,
                        className="pm-tab-radio-group",
                        inputClassName="pm-tab-radio-input",
                        labelClassName="pm-tab-radio-label",
                    ),
                ]),
                html.Div(id="post-match-transition-filter-wrap", style={"display": "none"}, children=[
                    dbc.RadioItems(
                        id="post-match-transition-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Goals", "value": "goals"},
                            {"label": "Shots", "value": "shots"},
                            {"label": "F3", "value": "f3"},
                            {"label": "Lost / Recovered", "value": "negative"},
                            {"label": "Retained / Survived", "value": "safe"},
                        ],
                        value="all",
                        inline=True,
                        className="pm-tab-radio-group",
                        inputClassName="pm-tab-radio-input",
                        labelClassName="pm-tab-radio-label",
                    ),
                ]),
                html.Div(id='post-match-h2h-container', style={"marginTop": "24px"}),
                ]),
            ])
        ),
        html.Footer(className="footer", children=[
            html.Div(className="footer-inner", children=[
                html.Div("© tactIQ Göztepe Hub — Precision Analytics", className="footer-text"),
                html.Img(src="/assets/superlig_logo.jpg", className="superlogo"),
            ])
        ])
    ])


@callback(
    [Output('post-match-match-selector', 'options'),
     Output('post-match-match-selector', 'value')],
    Input('post-match-rival-selector', 'value'),
)
def update_post_match_options(rival):
    if not rival:
        return [], None
    options = _build_match_options(rival)
    value = options[-1]['value'] if options else None
    return options, value


@callback(
    Output('post-match-transition-filter-wrap', 'style'),
    Input('post-match-tabs', 'value'),
)
def toggle_post_match_transition_filter(active_tab):
    if active_tab in ('off-trans-tab', 'def-trans-tab'):
        return {"display": "flex", "justifyContent": "center", "margin": "0 0 18px"}
    return {"display": "none"}


@callback(
    [Output('post-match-form-container', 'children'),
     Output('post-match-h2h-container', 'children'),
     Output('post-match-report-container', 'children')],
    [Input('post-match-rival-selector', 'value'),
     Input('post-match-match-selector', 'value'),
     Input('post-match-tabs', 'value'),
     Input('post-match-transition-filter', 'value')],
)
def update_post_match(rival, selected_file, active_tab, transition_filter):
    if not rival:
        return html.Div("Select an opponent", className="goz-card-desc"), html.Div("Select an opponent", className="goz-card-desc"), html.Div()
    if not selected_file:
        return html.Div("Select a match", className="goz-card-desc"), html.Div("Select a match", className="goz-card-desc"), html.Div()

    if not active_tab:
        active_tab = "checklist-tab"

    cache_key = f"postmatch_{rival}_{selected_file}_{active_tab}_{transition_filter}_{int(time.time()//300)}"
    if cache_key in _POST_MATCH_CACHE:
        return _POST_MATCH_CACHE[cache_key]

    try:
        opp_name = _clean(rival)
        form = _build_form_section(rival, opp_name, selected_file)
        h2h = _build_h2h_section(rival, opp_name, active_tab, selected_file, transition_filter)
        report = _build_post_match_report(rival, selected_file)
        result = [form, h2h, report]
        _POST_MATCH_CACHE[cache_key] = result
        if len(_POST_MATCH_CACHE) > 10:
            _POST_MATCH_CACHE.pop(next(iter(_POST_MATCH_CACHE)))
        return result
    except Exception as e:
        error_msg = html.Div([
            html.Div("⚠️ Data could not be loaded", className="goz-card-title"),
            html.P(f"Technical error: {str(e)}", className="goz-card-desc")
        ], className="goz-form-section")
        return [error_msg] * 2

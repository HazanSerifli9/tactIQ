import dash
import numpy as np
import pandas as pd
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from utils.data import get_data_dir

dash.register_page(__name__, path='/rival-scout', title='Göztepe Hub | Rival Scout')

GOZTEPE = 'Göztepe Spor Kulübü'

RIVALS = {
    'Galatasaray': 'Galatasaray Spor Kulübü',
    'Fenerbahçe':  'Fenerbahçe Spor Kulübü',
    'Beşiktaş':    'Beşiktaş Jimnastik Kulübü',
    'Trabzonspor': 'Trabzonspor Kulübü',
    'Başakşehir':  'İstanbul Başakşehir Futbol Kulübü',
}

# Exact 5 matches per rival (hardcoded by user selection)
RIVAL_FILES = {
    'Galatasaray Spor Kulübü': [
        'gs-alanya.parquet',
        'fb-gs.parquet',
        'konya-gs.parquet',
        'gs-genclerbirligi.parquet',
        'ts-gs.parquet',
    ],
    'Fenerbahçe Spor Kulübü': [
        'kasimpasa-fb.parquet',
        'fb-fatih.parquet',
        'bjk-fb.parquet',
        'eyup-fb.parquet',
        'antalya-fb.parquet',
    ],
    'Beşiktaş Jimnastik Kulübü': [
        'kayseri-bjk.parquet',
        'bjk-genclerbirligi.parquet',
        'bjk-samsunspor.parquet',
        'ts-bjk.parquet',
        'basaksehir-bjk.parquet',
    ],
    'Trabzonspor Kulübü': [
        'kayseri-ts.parquet',
        'rize-ts.parquet',
        'ts-fb.parquet',
        'basaksehir-ts.parquet',
        'genclerbirligi-ts.parquet',
    ],
    'İstanbul Başakşehir Futbol Kulübü': [
        'konya-basaksehir.parquet',
        'antalya-basaksehir.parquet',
        'rize-basaksehir.parquet',
        'gs-basaksehir.parquet',
        'basaksehir-eyup.parquet',
    ],
}


def _discover_rivals():
    """Build rival options from the parquet data, keeping the famous-five labels first."""
    discovered = {}
    data_dir = get_data_dir()
    try:
        filenames = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    except Exception:
        filenames = []

    for filename in filenames:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename), columns=['team_name'])
        except Exception:
            try:
                df = pd.read_parquet(os.path.join(data_dir, filename))
            except Exception:
                continue
        if 'team_name' not in df.columns:
            continue
        for team in df['team_name'].dropna().unique():
            if team != GOZTEPE:
                discovered[_short(team)] = team

    ordered = dict(RIVALS)
    for label in sorted(discovered):
        ordered.setdefault(label, discovered[label])
    return ordered

_SUFFIXES = [
    'Spor Kulübü', 'Futbol Kulübü', 'Jimnastik Kulübü', 'Kulübü',
    'Spor A.Ş.', 'A.Ş.', 'S.K.', 'F.K.', 'SK', 'Jimnastik',
]

def _short(name):
    r = name
    for s in _SUFFIXES:
        r = r.replace(s, '')
    return r.strip()

# ── Visual constants ────────────────────────────────────────────
PITCH_BG = "#0e1b0f"
GOLD     = "#fbbf24"
RED      = "#ef4444"
BLUE     = "#3b82f6"
PURPLE   = "#a855f7"
GREEN    = "#22c55e"
ORANGE   = "#f97316"

SHOT_EVENTS = {'Goal', 'Miss', 'Saved Shot', 'Post'}
DEF_EVENTS  = {'Tackle', 'Challenge', 'Interception', 'Clearance', 'Ball recovery'}
SKIP_EVENTS = {
    'Team setp up', 'Start', 'End', 'Start delay', 'End delay',
    'Injury Time Announcement', 'Card', 'Player Off', 'Player on',
    'Formation change', 'Collection End', 'Deleted event', 'Referee Drop Ball',
    'Contentious referee decision', 'Unknown',
}

INSIDE_BOX_COLS = [
    'Small box-centre', 'Small box-right', 'Small box-left',
    'Box-centre', 'Box-right', 'Box-left', 'Box-deep right', 'Box-deep left',
]
SMALL_BOX_COLS  = ['Small box-centre', 'Small box-right', 'Small box-left']
OUTSIDE_BOX_COLS = [
    'Out of box-centre', 'Out of box-right', 'Out of box-left',
    'Out of box-deep right', 'Out of box-deep left',
    '35+ centre', '35+ right', '35+ left',
]

# ── Match cache ─────────────────────────────────────────────────
_MATCH_CACHE: dict = {}


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

def _load_rival_matches(team_name: str) -> list:
    """Return list of (filename, df) for this team, preferring curated files when present."""
    if team_name in _MATCH_CACHE:
        return _MATCH_CACHE[team_name]

    data_dir  = get_data_dir()
    filenames = RIVAL_FILES.get(team_name, [])
    matches   = []

    if not filenames:
        try:
            filenames = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        except Exception:
            filenames = []

    for filename in filenames:
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_parquet(path)
            if 'team_name' in df.columns and team_name in df['team_name'].unique():
                matches.append((filename, df))
        except Exception:
            continue

    def sort_key(item):
        _, df = item
        week = int(df['week'].iloc[0]) if 'week' in df.columns and not df.empty else 0
        date = str(df['local_date'].iloc[0]) if 'local_date' in df.columns and not df.empty else ''
        return (week, date)

    matches = sorted(matches, key=sort_key, reverse=True)[:5]

    _MATCH_CACHE[team_name] = matches
    return matches


def _match_summary(filename: str, df: pd.DataFrame, team_name: str) -> dict:
    opponents = [t for t in df['team_name'].unique() if t != team_name]
    opponent  = opponents[0] if opponents else '?'

    team_df = df[df['team_name'] == team_name]
    opp_df  = df[df['team_name'] == opponent]

    goals_for     = len(team_df[team_df['event'] == 'Goal'])
    goals_against = len(opp_df[opp_df['event'] == 'Goal'])
    shots_for     = len(team_df[team_df['event'].isin(SHOT_EVENTS)])
    shots_against = len(opp_df[opp_df['event'].isin(SHOT_EVENTS)])
    passes_for    = len(team_df[team_df['event'] == 'Pass'])
    sot_for       = len(team_df[team_df['event'] == 'Saved Shot']) + goals_for

    result = 'W' if goals_for > goals_against else ('D' if goals_for == goals_against else 'L')
    return {
        'label': filename.replace('.parquet', ''),
        'opponent': _short(opponent),
        'score': f"{goals_for}–{goals_against}",
        'result': result,
        'shots_for': shots_for,
        'shots_against': shots_against,
        'sot_for': sot_for,
        'passes': passes_for,
    }


# ── Zone helpers ────────────────────────────────────────────────

def _zone_y(y: float) -> str:
    if y < 33.3:  return 'Left'
    if y <= 66.6: return 'Center'
    return 'Right'

def _zone_x(x: float) -> str:
    if x < 33.3:  return 'Defensive 3rd'
    if x <= 66.6: return 'Middle 3rd'
    return 'Final 3rd'

def _safe_pct(num, denom) -> float:
    return round(num / max(denom, 1) * 100, 1)


# ════════════════════════════════════════════════════════════════
# OFFENSIVE DATA
# ════════════════════════════════════════════════════════════════

def _compute_buildup(matches, team_name):
    total_passes = long_passes = gk_passes = gk_short = 0
    zone_counts  = {'Left': 0, 'Center': 0, 'Right': 0}
    player_d3    = {}
    coords       = []

    for _, df in matches:
        team   = df[df['team_name'] == team_name]
        passes = team[team['event'] == 'Pass']
        total_passes += len(passes)
        if 'Long ball' in passes.columns:
            long_passes += int(passes['Long ball'].notna().sum())

        for _, row in passes.iterrows():
            zone_counts[_zone_y(float(row['y']))] += 1
            is_long = pd.notna(row.get('Long ball'))
            coords.append({'x': float(row['x']), 'y': float(row['y']),
                           'type': 'Long' if is_long else 'Short'})

        gk = passes[passes['x'] < 15]
        gk_passes += len(gk)
        if 'Pass End X' in gk.columns:
            gk_short += int((gk['Pass End X'].fillna(0) < gk['x'] + 30).sum())

        d3 = passes[passes['x'] < 33]
        for name in d3['player_name']:
            player_d3[name] = player_d3.get(name, 0) + 1

    n  = max(len(matches), 1)
    zt = max(sum(zone_counts.values()), 1)
    return {
        'passes_per_game': round(total_passes / n, 1),
        'long_pct':        _safe_pct(long_passes, total_passes),
        'short_pct':       _safe_pct(total_passes - long_passes, total_passes),
        'left_pct':        _safe_pct(zone_counts['Left'],   zt),
        'center_pct':      _safe_pct(zone_counts['Center'], zt),
        'right_pct':       _safe_pct(zone_counts['Right'],  zt),
        'gk_short_pct':    _safe_pct(gk_short, gk_passes),
        'top_builders':    sorted(player_d3.items(), key=lambda x: -x[1])[:5],
        'coords':          coords[:800],
    }


def _compute_final_third(matches, team_name):
    entries      = []
    after_entry  = {'cross': 0, 'shot': 0, 'pass': 0, 'duel': 0, 'other': 0}

    for _, df in matches:
        team = df[df['team_name'] == team_name].reset_index(drop=True)
        recs = team.to_dict('records')

        for i, row in enumerate(recs):
            if row['event'] != 'Pass':
                continue
            ex = row.get('Pass End X')
            if ex is None or pd.isna(ex):
                continue
            if float(ex) >= 66 and float(row['x']) < 66:
                ey     = row.get('Pass End Y', row['y']) or row['y']
                method = 'Deep Pass' if pd.notna(row.get('Long ball')) else 'Short Pass'
                entries.append({'x': float(ex), 'y': float(ey), 'method': method})

                # What happens next?
                for j in range(i + 1, min(i + 8, len(recs))):
                    nxt = recs[j]
                    if nxt['event'] in SKIP_EVENTS:
                        continue
                    if nxt['event'] in SHOT_EVENTS:
                        after_entry['shot'] += 1
                    elif nxt['event'] == 'Pass' and pd.notna(nxt.get('Cross')):
                        after_entry['cross'] += 1
                    elif nxt['event'] in {'Challenge', 'Tackle', 'Aerial'}:
                        after_entry['duel'] += 1
                    elif nxt['event'] == 'Pass':
                        after_entry['pass'] += 1
                    else:
                        after_entry['other'] += 1
                    break

        # Take Ons near F3 as ball carry proxy
        for _, row in team[
            (team['event'] == 'Take On') & (team['x'] >= 60)
        ].iterrows():
            entries.append({'x': float(row['x']), 'y': float(row['y']), 'method': 'Ball Carry'})

    method_c = {'Short Pass': 0, 'Deep Pass': 0, 'Ball Carry': 0}
    zone_c   = {'Left': 0, 'Center': 0, 'Right': 0}
    for e in entries:
        method_c[e['method']] = method_c.get(e['method'], 0) + 1
        zone_c[_zone_y(e['y'])] += 1

    n  = max(len(matches), 1)
    tm = max(sum(method_c.values()), 1)
    tz = max(sum(zone_c.values()), 1)
    ta = max(sum(after_entry.values()), 1)

    return {
        'per_game': round(len(entries) / n, 1),
        'method':   {k: _safe_pct(v, tm) for k, v in method_c.items()},
        'zone':     {k: _safe_pct(v, tz) for k, v in zone_c.items()},
        'after':    {k: _safe_pct(v, ta) for k, v in after_entry.items()},
        'coords':   entries[:400],
    }


def _compute_15s_outcomes(matches, team_name):
    outcomes = {'f3_entry': 0, 'shot': 0, 'on_target': 0, 'goal': 0, 'turnover': 0}
    total    = 0

    TRIGGERS = {'Ball recovery', 'Interception', 'Tackle'}

    for _, df in matches:
        recs = df.reset_index(drop=True).to_dict('records')
        for i, row in enumerate(recs):
            if (row['team_name'] != team_name or
                    row['event'] not in TRIGGERS or
                    int(row.get('outcome', 1)) != 1 or
                    float(row['x']) >= 50):
                continue

            total += 1
            t0 = float(row['time_min']) * 60 + float(row.get('time_sec') or 0)
            f3, shot, ot, goal, turned = False, False, False, False, False

            for j in range(i + 1, len(recs)):
                r2 = recs[j]
                t2 = float(r2['time_min']) * 60 + float(r2.get('time_sec') or 0)
                if t2 - t0 > 15:
                    break
                if r2['event'] in SKIP_EVENTS:
                    continue
                if r2['team_name'] == team_name:
                    if float(r2.get('x', 0)) >= 66:
                        f3 = True
                    if r2['event'] in SHOT_EVENTS:
                        shot = True
                        if r2['event'] in {'Saved Shot', 'Goal'}:
                            ot = True
                        if r2['event'] == 'Goal':
                            goal = True
                else:
                    if r2['event'] not in SKIP_EVENTS:
                        turned = True
                        break

            if f3:     outcomes['f3_entry']  += 1
            if shot:   outcomes['shot']       += 1
            if ot:     outcomes['on_target']  += 1
            if goal:   outcomes['goal']        += 1
            if turned: outcomes['turnover']   += 1

    t = max(total, 1)
    return {k: _safe_pct(v, t) for k, v in outcomes.items()} | {'total': total}


def _compute_playmaker(matches, team_name):
    prog_dict  = {}
    total_dict = {}
    fb_l = fb_r = wl = wr = 0
    ppg_list = []

    for _, df in matches:
        team   = df[df['team_name'] == team_name]
        passes = team[team['event'] == 'Pass']
        ppg_list.append(len(passes))

        for _, row in passes.iterrows():
            name = row.get('player_name', '?') or '?'
            ex   = row.get('Pass End X')
            if ex is not None and not pd.isna(ex):
                total_dict[name] = total_dict.get(name, 0) + 1
                if float(ex) - float(row['x']) > 10 and float(row['x']) < 66:
                    prog_dict[name] = prog_dict.get(name, 0) + 1

            x, y = float(row['x']), float(row['y'])
            if 25 < x < 66 and y < 25:  fb_l += 1
            if 25 < x < 66 and y > 75:  fb_r += 1

        f3 = team[team['x'] >= 66]
        wl += len(f3[f3['y'] < 25])
        wr += len(f3[f3['y'] > 75])

    n  = max(len(matches), 1)
    top_pm = sorted(prog_dict.items(), key=lambda x: -x[1])
    pm_name  = top_pm[0][0] if top_pm else '—'
    pm_prog  = top_pm[0][1] if top_pm else 0
    pm_total = total_dict.get(pm_name, 0)

    return {
        'pm_name':          pm_name,
        'pm_prog':          pm_prog,
        'pm_total':         pm_total,
        'passes_per_game':  round(sum(ppg_list) / n, 1),
        'fb_left_pg':       round(fb_l / n, 1),
        'fb_right_pg':      round(fb_r / n, 1),
        'wing_left_pg':     round(wl / n, 1),
        'wing_right_pg':    round(wr / n, 1),
        'top_progressive':  top_pm[:5],
    }


def _compute_cross_map(matches, team_name):
    """Collect all cross event locations and zone breakdowns."""
    coords = []

    for _, df in matches:
        team   = df[df['team_name'] == team_name]
        passes = team[team['event'] == 'Pass']
        if 'Cross' not in passes.columns:
            continue
        crosses = passes[passes['Cross'].notna()]
        for _, row in crosses.iterrows():
            coords.append({
                'x':       float(row['x']),
                'y':       float(row['y']),
                'success': int(row.get('outcome', 0) or 0) == 1,
            })

    total = len(coords)
    n     = max(len(matches), 1)

    # Zone breakdown (attacking half only, by flank)
    zones = {
        'Left Flank (deep)':   0,
        'Left Flank (wide)':   0,
        'Right Flank (wide)':  0,
        'Right Flank (deep)':  0,
        'Central':             0,
    }
    for c in coords:
        x, y = c['x'], c['y']
        if y < 33:
            zones['Left Flank (deep)' if x >= 80 else 'Left Flank (wide)'] += 1
        elif y > 67:
            zones['Right Flank (deep)' if x >= 80 else 'Right Flank (wide)'] += 1
        else:
            zones['Central'] += 1

    return {
        'coords':      coords,
        'total':       total,
        'per_game':    round(total / n, 1),
        'success_pct': _safe_pct(sum(1 for c in coords if c['success']), total),
        'zones':       {k: _safe_pct(v, max(total, 1)) for k, v in zones.items()},
        'zones_raw':   zones,
    }


def _compute_z14(matches, team_name):
    passes_z14 = duel_won = duel_lost = shots_z14 = 0

    for _, df in matches:
        team = df[df['team_name'] == team_name]

        if 'Pass End X' in df.columns and 'Pass End Y' in df.columns:
            p = team[team['event'] == 'Pass']
            z14 = p[
                (p['Pass End X'].fillna(0) >= 66) &
                (p['Pass End Y'].fillna(50).between(33, 67))
            ]
            passes_z14 += len(z14)

        duels = team[
            team['event'].isin({'Challenge', 'Tackle', 'Aerial'}) &
            (team['x'] >= 66) &
            (team['y'].between(33, 67))
        ]
        duel_won  += int((duels['outcome'] == 1).sum())
        duel_lost += int((duels['outcome'] == 0).sum())

        shots = team[
            team['event'].isin(SHOT_EVENTS) &
            (team['x'] >= 66) &
            (team['y'].between(33, 67))
        ]
        shots_z14 += len(shots)

    n  = max(len(matches), 1)
    dt = max(duel_won + duel_lost, 1)
    return {
        'passes_pg':   round(passes_z14 / n, 1),
        'duel_win':    _safe_pct(duel_won, dt),
        'duels_pg':    round((duel_won + duel_lost) / n, 1),
        'shots_pg':    round(shots_z14 / n, 1),
    }


def _compute_shot_origin(matches, team_name):
    origins  = {'cross': 0, 'set_piece': 0, 'fast_break': 0, 'open_play': 0}
    zones    = {'small_box': 0, 'inside_box': 0, 'outside_box': 0}
    headers  = on_target = goals = total_shots = 0
    shot_coords = []
    finishers = {}

    for _, df in matches:
        team  = df[df['team_name'] == team_name]
        shots = team[team['event'].isin(SHOT_EVENTS)]
        total_shots += len(shots)

        for _, row in shots.iterrows():
            ev = row['event']
            if ev in {'Saved Shot', 'Goal'}:
                on_target += 1
            if ev == 'Goal':
                goals += 1
                name = row.get('player_name', '?') or '?'
                finishers[name] = finishers.get(name, 0) + 1

            if pd.notna(row.get('Cross')):
                origins['cross'] += 1
            elif any(pd.notna(row.get(c)) for c in ['Set piece', 'Free kick', 'From corner']):
                origins['set_piece'] += 1
            elif pd.notna(row.get('Fast break')):
                origins['fast_break'] += 1
            else:
                origins['open_play'] += 1

            if pd.notna(row.get('Head')):
                headers += 1

            in_small = any(pd.notna(row.get(c)) for c in SMALL_BOX_COLS if c in row.index)
            in_box   = any(pd.notna(row.get(c)) for c in INSIDE_BOX_COLS if c in row.index)
            in_out   = any(pd.notna(row.get(c)) for c in OUTSIDE_BOX_COLS if c in row.index)

            if in_small:
                zones['small_box']  += 1
                zones['inside_box'] += 1
            elif in_box:
                zones['inside_box'] += 1
            elif in_out:
                zones['outside_box'] += 1
            else:
                rx = float(row['x'])
                if rx >= 83:
                    zones['small_box']  += 1
                    zones['inside_box'] += 1
                elif rx >= 66:
                    zones['inside_box'] += 1
                else:
                    zones['outside_box'] += 1

            shot_coords.append({
                'x': float(row['x']), 'y': float(row['y']),
                'event': ev, 'in_box': in_box or in_small,
            })

    n = max(len(matches), 1)
    t = max(total_shots, 1)
    return {
        'total_pg':      round(total_shots / n, 1),
        'on_target_pct': _safe_pct(on_target, total_shots),
        'goals_pg':      round(goals / n, 1),
        'origins':       {k: _safe_pct(v, t) for k, v in origins.items()},
        'zones':         {k: _safe_pct(v, t) for k, v in zones.items()},
        'header_pct':    _safe_pct(headers, total_shots),
        'coords':        shot_coords,
        'finishers':     sorted(finishers.items(), key=lambda x: -x[1])[:5],
    }


# ════════════════════════════════════════════════════════════════
# DEFENSIVE DATA
# ════════════════════════════════════════════════════════════════

def _compute_defensive(matches, team_name):
    def_total = ppda_def = ppda_opp = 0
    x_vals    = []
    player_def = {}
    heat_coords = []

    for _, df in matches:
        team = df[df['team_name'] == team_name]
        opp  = df[df['team_name'] != team_name]

        def_ev = team[team['event'].isin(DEF_EVENTS)]
        def_total += len(def_ev)

        for _, row in def_ev.iterrows():
            x_vals.append(float(row['x']))
            name = row.get('player_name', '?') or '?'
            player_def[name] = player_def.get(name, 0) + 1
            heat_coords.append({'x': float(row['x']), 'y': float(row['y']),
                                 'type': row['event']})

        ppda_def += len(def_ev[def_ev['x'] > 50])

        opp_passes = opp[opp['event'] == 'Pass']
        ppda_opp  += len(opp_passes[opp_passes['x'] < 50])

    n        = max(len(matches), 1)
    avg_line = round(float(np.mean(x_vals)), 1) if x_vals else 0.0
    ppda     = round(ppda_opp / max(ppda_def, 1), 2)

    if ppda < 8:
        press_label = 'High Press'
    elif ppda < 14:
        press_label = 'Mid-Block'
    else:
        press_label = 'Low / Passive'

    return {
        'def_pg':      round(def_total / n, 1),
        'avg_line':    avg_line,
        'ppda':        ppda,
        'press_label': press_label,
        'heat_coords': heat_coords[:600],
        'top_defenders': sorted(player_def.items(), key=lambda x: -x[1])[:8],
    }


def _compute_aerials(matches, team_name):
    a_won = a_lost = ab_won = ab_lost = g_won = g_lost = 0

    for _, df in matches:
        team = df[df['team_name'] == team_name]

        aerials = team[team['event'] == 'Aerial']
        a_won   += int((aerials['outcome'] == 1).sum())
        a_lost  += int((aerials['outcome'] == 0).sum())

        box_a    = aerials[aerials['x'] >= 83]
        ab_won  += int((box_a['outcome'] == 1).sum())
        ab_lost += int((box_a['outcome'] == 0).sum())

        for ev in ['Challenge', 'Tackle']:
            ev_df = team[team['event'] == ev]
            g_won  += int((ev_df['outcome'] == 1).sum())
            g_lost += int((ev_df['outcome'] == 0).sum())

    n = max(len(matches), 1)
    return {
        'aerial_win':     _safe_pct(a_won,  a_won + a_lost),
        'aerial_pg':      round((a_won + a_lost) / n, 1),
        'box_aerial_win': _safe_pct(ab_won, ab_won + ab_lost),
        'box_aerial_tot': ab_won + ab_lost,
        'ground_win':     _safe_pct(g_won,  g_won + g_lost),
        'ground_pg':      round((g_won + g_lost) / n, 1),
    }


def _compute_vulnerability(matches, team_name):
    f3_c = {'Left': 0, 'Center': 0, 'Right': 0}
    z14_c = offside = 0

    for _, df in matches:
        opp      = df[df['team_name'] != team_name]
        team_ev  = df[df['team_name'] == team_name]

        if 'Pass End X' in df.columns and 'Pass End Y' in df.columns:
            opp_passes = opp[opp['event'] == 'Pass']
            entries = opp_passes[
                (opp_passes['Pass End X'].fillna(0) >= 66) &
                (opp_passes['x'] < 66)
            ]
            for _, row in entries.iterrows():
                ey = float(row.get('Pass End Y') or row['y'])
                f3_c[_zone_y(ey)] += 1

            z14 = opp_passes[
                (opp_passes['Pass End X'].fillna(0) >= 66) &
                (opp_passes['Pass End Y'].fillna(50).between(33, 67))
            ]
            z14_c += len(z14)

        offside += len(team_ev[team_ev['event'] == 'Offside provoked'])

    n  = max(len(matches), 1)
    ft = max(sum(f3_c.values()), 1)
    return {
        'f3_conceded':  {k: _safe_pct(v, ft) for k, v in f3_c.items()},
        'f3_pg':        round(sum(f3_c.values()) / n, 1),
        'z14_pg':       round(z14_c / n, 1),
        'offside_pg':   round(offside / n, 1),
    }


def _compute_pre_goal(matches, team_name):
    windows       = []
    goals_conceded = 0

    for _, df in matches:
        opp = df[df['team_name'] != team_name]
        goals = opp[opp['event'] == 'Goal']
        goals_conceded += len(goals)

        ts_col = df['time_min'] * 60 + df['time_sec'].fillna(0)

        for _, grow in goals.iterrows():
            t_goal = float(grow['time_min']) * 60 + float(grow.get('time_sec') or 0)
            win    = df[(ts_col >= t_goal - 30) & (ts_col <= t_goal)]

            def_acts = len(win[
                (win['team_name'] == team_name) &
                (win['event'].isin(DEF_EVENTS))
            ])
            opp_pas = len(win[
                (win['team_name'] != team_name) &
                (win['event'] == 'Pass')
            ])
            failed_clear = len(win[
                (win['team_name'] == team_name) &
                (win['event'] == 'Clearance') &
                (win['outcome'] == 0)
            ])
            windows.append({'def_acts': def_acts, 'opp_pas': opp_pas, 'fc': failed_clear})

    n  = max(len(matches), 1)
    nw = max(len(windows), 1)
    return {
        'goals_conceded': goals_conceded,
        'goals_pg':       round(goals_conceded / n, 1),
        'avg_def_acts':   round(sum(w['def_acts'] for w in windows) / nw, 1),
        'avg_opp_pas':    round(sum(w['opp_pas']  for w in windows) / nw, 1),
        'avg_fc':         round(sum(w['fc']        for w in windows) / nw, 1),
    }


# ════════════════════════════════════════════════════════════════
# TRANSITIONS DATA
# ════════════════════════════════════════════════════════════════

def _compute_transitions(matches, team_name):
    RECOVERY  = {'Ball recovery', 'Interception', 'Tackle'}
    LOSS_EVS  = {'Dispossessed', 'Error'}

    wz = {'Defensive 3rd': 0, 'Middle 3rd': 0, 'Final 3rd': 0}
    lz = {'Defensive 3rd': 0, 'Middle 3rd': 0, 'Final 3rd': 0}
    wo = {'f3': 0, 'shot': 0, 'opp_shot': 0}
    lo = {'opp_f3': 0, 'opp_shot': 0, 'recovered': 0}
    wp = {}; lp = {}
    wc = []; lc = []
    wt = lt = 0

    for _, df in matches:
        recs = df.to_dict('records')
        for i, row in enumerate(recs):
            if row['event'] in SKIP_EVENTS:
                continue

            t0 = float(row['time_min']) * 60 + float(row.get('time_sec') or 0)

            # ── Ball WIN ──────────────────────────────────────────
            if (row['team_name'] == team_name and
                    row['event'] in RECOVERY and
                    int(row.get('outcome', 1)) == 1):
                wt += 1
                wz[_zone_x(float(row['x']))] += 1
                name = row.get('player_name', '?') or '?'
                wp[name] = wp.get(name, 0) + 1
                wc.append({'x': float(row['x']), 'y': float(row['y'])})

                for j in range(i + 1, len(recs)):
                    r2 = recs[j]
                    t2 = float(r2['time_min']) * 60 + float(r2.get('time_sec') or 0)
                    if t2 - t0 > 15: break
                    if r2['event'] in SKIP_EVENTS: continue
                    if r2['team_name'] == team_name:
                        if float(r2.get('x', 0)) >= 66:
                            wo['f3'] += 1
                        if r2['event'] in SHOT_EVENTS:
                            wo['shot'] += 1
                    else:
                        if r2['event'] in SHOT_EVENTS:
                            wo['opp_shot'] += 1
                        break

            # ── Ball LOSS ─────────────────────────────────────────
            if (row['team_name'] == team_name and
                    row['event'] in LOSS_EVS):
                lt += 1
                lz[_zone_x(float(row['x']))] += 1
                name = row.get('player_name', '?') or '?'
                lp[name] = lp.get(name, 0) + 1
                lc.append({'x': float(row['x']), 'y': float(row['y'])})

                for j in range(i + 1, len(recs)):
                    r2 = recs[j]
                    t2 = float(r2['time_min']) * 60 + float(r2.get('time_sec') or 0)
                    if t2 - t0 > 15: break
                    if r2['event'] in SKIP_EVENTS: continue
                    if r2['team_name'] != team_name:
                        if float(r2.get('x', 0)) >= 66:
                            lo['opp_f3'] += 1
                        if r2['event'] in SHOT_EVENTS:
                            lo['opp_shot'] += 1
                    else:
                        if r2['event'] in RECOVERY:
                            lo['recovered'] += 1
                        break

    n = max(len(matches), 1)

    def _pct_dict(d, total):
        t = max(total, 1)
        return {k: _safe_pct(v, t) for k, v in d.items()}

    return {
        'win': {
            'total': wt, 'per_game': round(wt / n, 1),
            'zones': _pct_dict(wz, wt),
            'outcomes': _pct_dict(wo, wt),
            'top': sorted(wp.items(), key=lambda x: -x[1])[:8],
            'coords': wc[:400],
        },
        'loss': {
            'total': lt, 'per_game': round(lt / n, 1),
            'zones': _pct_dict(lz, lt),
            'outcomes': _pct_dict(lo, lt),
            'top': sorted(lp.items(), key=lambda x: -x[1])[:8],
            'coords': lc[:400],
        },
    }


# ════════════════════════════════════════════════════════════════
# PITCH / VISUAL HELPERS
# ════════════════════════════════════════════════════════════════

def _make_pitch(half=False, figsize=(10, 6.5)):
    p = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
              line_color=(1, 1, 1, 0.5), linewidth=1.4, half=half)
    fig, ax = p.draw(figsize=figsize)
    fig.patch.set_facecolor(PITCH_BG)
    return p, fig, ax


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=PITCH_BG, edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def _legend(ax):
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    leg = ax.legend(h, l, loc='upper left', fontsize=8,
                    framealpha=0.7, facecolor=PITCH_BG, edgecolor='white')
    for t in leg.get_texts():
        t.set_color('white')


# ════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════

def _card(*children, title=None, icon=''):
    hdr = []
    if title:
        hdr = [html.Div(className='goz-section-header', style={'marginBottom': '14px'}, children=[
            html.Span(f'{icon}  {title}' if icon else title, className='goz-card-title')
        ])]
    return html.Div(className='goz-form-section', style={'marginBottom': '16px'},
                    children=hdr + list(children))


def _pill(label, value, color=None):
    color = color or GOLD
    return html.Div(style={
        'textAlign': 'center', 'padding': '12px 10px',
        'background': 'rgba(255,255,255,0.04)',
        'borderRadius': '12px', 'border': '1px solid var(--border-color)',
        'flex': '1', 'minWidth': '80px',
    }, children=[
        html.Div(str(value), style={
            'fontSize': '1.45rem', 'fontWeight': '700', 'color': color, 'lineHeight': '1',
        }),
        html.Div(label, style={
            'fontSize': '0.67rem', 'color': 'var(--text-secondary)',
            'marginTop': '5px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px',
        }),
    ])


def _bar(label, pct, color=None):
    color = color or GOLD
    try:
        pct = float(str(pct).replace('%', '').strip())
    except Exception:
        pct = 0
    pct = min(max(pct, 0), 100)
    return html.Div(style={'marginBottom': '8px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '3px'}, children=[
            html.Span(label, style={'fontSize': '0.77rem', 'color': 'var(--text-secondary)'}),
            html.Span(f'{pct:.0f}%', style={'fontSize': '0.77rem', 'fontWeight': '700', 'color': color}),
        ]),
        html.Div(style={'height': '4px', 'background': 'rgba(255,255,255,0.07)', 'borderRadius': '2px'}, children=[
            html.Div(style={'width': f'{pct}%', 'height': '100%', 'background': color, 'borderRadius': '2px'}),
        ]),
    ])


def _player_table(rows, value_key_label='Recoveries', color=None):
    color = color or GOLD
    items = []
    for i, (name, count) in enumerate(rows):
        items.append(html.Div(style={
            'display': 'flex', 'justifyContent': 'space-between',
            'padding': '5px 0', 'borderBottom': '1px solid rgba(255,255,255,0.05)',
        }, children=[
            html.Span(f'{i+1}. {name}', style={'fontSize': '0.76rem', 'color': 'var(--text-secondary)'}),
            html.Span(str(count), style={'fontSize': '0.76rem', 'fontWeight': '700', 'color': color}),
        ]))
    return html.Div(items)


def _label_badge(text, color):
    return html.Span(text, style={
        'background': color, 'color': '#000',
        'borderRadius': '6px', 'padding': '2px 10px',
        'fontSize': '0.75rem', 'fontWeight': '700',
    })


# ════════════════════════════════════════════════════════════════
# MATCH SUMMARY CARD (single selected match)
# ════════════════════════════════════════════════════════════════

def _build_selected_match_card(fname, df, team_name, rival_label):
    """Rich summary card for the currently selected match."""
    s = _match_summary(fname, df, team_name)
    res_colors = {'W': GREEN, 'D': GOLD, 'L': RED}
    c = res_colors.get(s['result'], GOLD)

    # Extra stats for the single-match card
    team_df = df[df['team_name'] == team_name]
    opp_df  = df[df['team_name'] != team_name]
    tackles_won    = len(team_df[(team_df['event'] == 'Tackle')   & (team_df['outcome'] == 1)])
    aerial_won     = len(team_df[(team_df['event'] == 'Aerial')   & (team_df['outcome'] == 1)])
    aerial_total   = len(team_df[team_df['event'] == 'Aerial'])
    ball_recoveries= len(team_df[team_df['event'] == 'Ball recovery'])
    long_balls     = int(team_df[team_df['event'] == 'Pass']['Long ball'].notna().sum()) if 'Long ball' in team_df.columns else 0
    passes         = len(team_df[team_df['event'] == 'Pass'])
    long_pct       = _safe_pct(long_balls, passes)

    # Opponent stats
    opp_shots      = len(opp_df[opp_df['event'].isin(SHOT_EVENTS)])
    opp_passes     = len(opp_df[opp_df['event'] == 'Pass'])

    return html.Div(style={
        'background': 'var(--card-bg)',
        'border': f'1px solid {c}44',
        'borderRadius': '16px',
        'padding': '20px 24px',
        'marginBottom': '20px',
    }, children=[
        # Header row
        html.Div(style={
            'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'marginBottom': '18px',
        }, children=[
            html.Div(style={'flex': '1'}, children=[
                html.Div(f'{rival_label}', style={
                    'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                    'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '4px',
                }),
                html.Div(f'vs {s["opponent"]}', style={
                    'fontSize': '1.2rem', 'fontWeight': '700', 'color': 'var(--text-primary)',
                }),
            ]),
            html.Div(s['score'], style={
                'fontSize': '2.4rem', 'fontWeight': '800', 'color': c, 'lineHeight': '1',
            }),
            _label_badge(s['result'], c),
        ]),
        # Stats pills
        html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
            _pill('Shots',        s['shots_for'],                   BLUE),
            _pill('SOT',          s['sot_for'],                     GREEN),
            _pill('Passes',       s['passes'],                      GOLD),
            _pill('Long Ball %',  f"{long_pct}%",                   PURPLE),
            _pill('Tackles Won',  tackles_won,                      ORANGE),
            _pill('Aerial Won',   f"{aerial_won}/{aerial_total}",   BLUE),
            _pill('Ball Rec.',    ball_recoveries,                  GREEN),
            _pill('Opp Shots',    opp_shots,                        RED),
            _pill('Opp Passes',   opp_passes,                       RED),
        ]),
    ])


# ════════════════════════════════════════════════════════════════
# OFFENSIVE TAB
# ════════════════════════════════════════════════════════════════

def _build_offensive(matches, team_name, rival_label):
    sections = []

    # ── 1. Build-Up Style ──────────────────────────────────────
    try:
        bu = _compute_buildup(matches, team_name)
        p, fig, ax = _make_pitch()
        for c in bu['coords'][:600]:
            color = RED if c['type'] == 'Long' else GOLD
            p.scatter([c['x']], [c['y']], ax=ax, color=color, s=12, alpha=0.45)
        ax.scatter([], [], color=GOLD, s=30, label='Short')
        ax.scatter([], [], color=RED,  s=30, label='Long')
        _legend(ax)
        img = _fig_b64(fig)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div('PASS TYPE', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Short Pass', bu['short_pct'], GOLD),
                    _bar('Long Ball',  bu['long_pct'],  RED),
                    html.Div(style={'marginTop': '14px', 'marginBottom': '8px'}, children=[
                        html.Div('BUILD-UP ZONE', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('Left Channel',     bu['left_pct'],   BLUE),
                        _bar('Central Corridor', bu['center_pct'], GOLD),
                        _bar('Right Channel',    bu['right_pct'],  PURPLE),
                    ]),
                    html.Div(style={
                        'background': 'rgba(255,255,255,0.03)', 'borderRadius': '10px',
                        'padding': '10px', 'border': '1px solid var(--border-color)',
                    }, children=[
                        html.Div('TOP BUILD-UP PLAYERS', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _player_table(bu['top_builders'], color=GOLD),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div(style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center', 'marginTop': '6px'}, children=[
                        html.Span('● Short pass origins', style={'fontSize': '0.65rem', 'color': GOLD}),
                        html.Span('● Long ball origins', style={'fontSize': '0.65rem', 'color': RED}),
                    ]),
                    html.Div(style={'marginTop': '14px', 'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'}, children=[
                        _pill('Passes / Game', bu['passes_per_game'], GOLD),
                        _pill('GK Short %',    f"{bu['gk_short_pct']}%", BLUE),
                    ]),
                ], md=8),
            ]),
            title=f'{rival_label} — Build-Up Style', icon='⚽'
        ))
    except Exception:
        pass

    # ── 2. Final Third Entry ───────────────────────────────────
    try:
        f3 = _compute_final_third(matches, team_name)
        p, fig, ax = _make_pitch(half=True)
        METHOD_C = {'Short Pass': GOLD, 'Deep Pass': RED, 'Ball Carry': BLUE}
        for e in f3['coords']:
            c = METHOD_C.get(e['method'], GOLD)
            p.scatter([e['x']], [e['y']], ax=ax, color=c, s=35, alpha=0.65, marker='D')
        for label, c in METHOD_C.items():
            ax.scatter([], [], color=c, s=40, marker='D', label=label)
        _legend(ax)
        img = _fig_b64(fig)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    _pill('Entries / Game', f3['per_game'], GOLD),
                    html.Div(style={'marginTop': '14px'}, children=[
                        html.Div('ENTRY METHOD', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('Short Pass',  f3['method']['Short Pass'],  GOLD),
                        _bar('Deep Pass',   f3['method']['Deep Pass'],   RED),
                        _bar('Ball Carry',  f3['method']['Ball Carry'],  BLUE),
                    ]),
                    html.Div(style={'marginTop': '14px'}, children=[
                        html.Div('ENTRY ZONE', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('Left',    f3['zone']['Left'],   BLUE),
                        _bar('Central', f3['zone']['Center'], GOLD),
                        _bar('Right',   f3['zone']['Right'],  PURPLE),
                    ]),
                    html.Div(style={'marginTop': '14px'}, children=[
                        html.Div('AFTER ENTRY', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('→ Cross',      f3['after']['cross'],  PURPLE),
                        _bar('→ Shot',       f3['after']['shot'],   GREEN),
                        _bar('→ Duel',       f3['after']['duel'],   ORANGE),
                        _bar('→ Possession', f3['after']['pass'],   BLUE),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div('◆ Final third entry points by method', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=7),
            ]),
            title='Final Third Entry', icon='🎯'
        ))
    except Exception:
        pass

    # ── 3. 15-Second Outcomes ──────────────────────────────────
    try:
        out = _compute_15s_outcomes(matches, team_name)
        sections.append(_card(
            html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                _pill('→ F3 Entry',   f"{out['f3_entry']}%",   GOLD),
                _pill('→ Shot',       f"{out['shot']}%",        BLUE),
                _pill('→ On Target',  f"{out['on_target']}%",   GREEN),
                _pill('→ Goal',       f"{out['goal']}%",        ORANGE),
                _pill('→ Turnover',   f"{out['turnover']}%",    RED),
            ]),
            html.Div(f"Based on {out['total']} ball-win sequences from own half — 15-second window", style={
                'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                'marginTop': '10px', 'textAlign': 'center',
            }),
            title='15-Second Outcomes (from Ball Win)', icon='⏱'
        ))
    except Exception:
        pass

    # ── 4. Playmaker & Tempo ───────────────────────────────────
    try:
        pm = _compute_playmaker(matches, team_name)
        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={
                        'background': 'rgba(251,191,36,0.08)',
                        'border': '1px solid rgba(251,191,36,0.22)',
                        'borderRadius': '12px', 'padding': '14px', 'marginBottom': '12px',
                    }, children=[
                        html.Div('🎖 KEY PLAYMAKER', style={
                            'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                            'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '6px',
                        }),
                        html.Div(pm['pm_name'], style={
                            'fontSize': '1.05rem', 'fontWeight': '700', 'color': GOLD,
                        }),
                        html.Div(style={'display': 'flex', 'gap': '14px', 'marginTop': '6px'}, children=[
                            html.Span(f"{pm['pm_total']} total passes", style={'fontSize': '0.72rem', 'color': 'var(--text-secondary)'}),
                            html.Span(f"{pm['pm_prog']} progressive", style={'fontSize': '0.72rem', 'color': BLUE}),
                        ]),
                    ]),
                    html.Div('TOP PROGRESSIVE PASSERS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _player_table(pm['top_progressive'], color=BLUE),
                ], md=5),
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '12px'}, children=[
                        _pill('Passes / Game',   pm['passes_per_game'], GOLD),
                        _pill('Left FB Passes',  pm['fb_left_pg'],      BLUE),
                        _pill('Right FB Passes', pm['fb_right_pg'],     BLUE),
                    ]),
                    html.Div('WINGER TOUCHES — WIDE FINAL THIRD', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                        _pill('Left Wing / Game',  pm['wing_left_pg'],  PURPLE),
                        _pill('Right Wing / Game', pm['wing_right_pg'], PURPLE),
                    ]),
                    html.Div(style={
                        'marginTop': '12px', 'fontSize': '0.68rem',
                        'color': 'var(--text-secondary)', 'lineHeight': '1.5',
                    }, children=[
                        html.Span('⚠ Numerical superiority by zone & exact triangle formation detection require tracking data (not available from event logs).', ),
                    ]),
                ], md=7),
            ]),
            title='Playmaker, Tempo & Width', icon='🧠'
        ))
    except Exception:
        pass

    # ── 5. Zone 14 Control ─────────────────────────────────────
    try:
        z14 = _compute_z14(matches, team_name)
        sections.append(_card(
            html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                _pill('Z14 Passes / Game', z14['passes_pg'],   GOLD),
                _pill('Z14 Shots / Game',  z14['shots_pg'],    BLUE),
                _pill('Z14 Duels / Game',  z14['duels_pg'],    PURPLE),
                _pill('Z14 Duel Win %',    f"{z14['duel_win']}%", GREEN),
            ]),
            html.Div('Zone 14 = central channel of the final third (x > 66, y 33–67)', style={
                'fontSize': '0.67rem', 'color': 'var(--text-secondary)',
                'marginTop': '8px', 'textAlign': 'center',
            }),
            title='Zone 14 Control', icon='🔑'
        ))
    except Exception:
        pass

    # ── 6. Cross Map ───────────────────────────────────────────
    try:
        cm = _compute_cross_map(matches, team_name)
        coords = cm['coords']

        if coords:
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]

            p, fig, ax = _make_pitch(figsize=(12, 7.5))

            # Grid heatmap — bins=(6 x, 5 y)
            stat = p.bin_statistic(xs, ys, statistic='count', bins=(6, 5))
            p.heatmap(stat, ax=ax, cmap='YlOrRd', edgecolors=PITCH_BG,
                      linewidth=2.5, alpha=0.72)
            p.label_heatmap(stat, color='white', fontsize=13, ax=ax,
                            ha='center', va='center', fontweight='bold',
                            str_format='{:.0f}')

            # Individual cross origins
            succ = [c for c in coords if c['success']]
            fail = [c for c in coords if not c['success']]
            if succ:
                p.scatter([c['x'] for c in succ], [c['y'] for c in succ],
                          ax=ax, color=GREEN, s=50, alpha=0.85,
                          zorder=6, marker='^', label='Successful cross')
            if fail:
                p.scatter([c['x'] for c in fail], [c['y'] for c in fail],
                          ax=ax, color=RED, s=30, alpha=0.55,
                          zorder=6, marker='x', label='Failed cross')
            _legend(ax)

            cross_img = _fig_b64(fig)

            sections.append(_card(
                dbc.Row([
                    dbc.Col([
                        html.Div(style={
                            'display': 'flex', 'gap': '8px',
                            'flexWrap': 'wrap', 'marginBottom': '16px',
                        }, children=[
                            _pill('Total Crosses',  cm['total'],                 GOLD),
                            _pill('Per Match',      cm['per_game'],              BLUE),
                            _pill('Success %',      f"{cm['success_pct']}%",     GREEN),
                        ]),
                        html.Div('CROSS ORIGIN ZONE', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '10px',
                        }),
                        _bar('Left Flank — Deep (byline)',  cm['zones']['Left Flank (deep)'],  BLUE),
                        _bar('Left Flank — Wide',           cm['zones']['Left Flank (wide)'],  BLUE),
                        _bar('Right Flank — Wide',          cm['zones']['Right Flank (wide)'], PURPLE),
                        _bar('Right Flank — Deep (byline)', cm['zones']['Right Flank (deep)'], PURPLE),
                        _bar('Central Corridor',            cm['zones']['Central'],            GOLD),
                        html.Div(style={
                            'marginTop': '14px',
                            'background': 'rgba(255,255,255,0.03)',
                            'borderRadius': '10px', 'padding': '12px',
                            'border': '1px solid var(--border-color)',
                            'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                            'lineHeight': '1.6',
                        }, children=[
                            html.Strong('Raw counts per zone:', style={'color': GOLD, 'display': 'block', 'marginBottom': '5px'}),
                        ] + [
                            html.Div(f"{k}: {v}", style={'marginBottom': '2px'})
                            for k, v in cm['zones_raw'].items() if v > 0
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.Img(src=cross_img, style={'width': '100%', 'borderRadius': '8px'}),
                        html.Div(style={
                            'display': 'flex', 'gap': '20px',
                            'justifyContent': 'center', 'marginTop': '6px',
                        }, children=[
                            html.Span('▲ Successful cross', style={'fontSize': '0.65rem', 'color': GREEN}),
                            html.Span('✕ Failed cross',     style={'fontSize': '0.65rem', 'color': RED}),
                            html.Span('■ Cell count',       style={'fontSize': '0.65rem', 'color': GOLD}),
                        ]),
                    ], md=8),
                ]),
                title=f'{rival_label} — Cross Map & Origin Zones', icon='🎯'
            ))
    except Exception:
        pass

    # ── 7. Shot Origin / xG Chain (kept for reference) ──────────
    try:
        so = _compute_shot_origin(matches, team_name)
        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Shots / Game',  so['total_pg'],             GOLD),
                        _pill('On Target %',   f"{so['on_target_pct']}%",  GREEN),
                        _pill('Goals / Game',  so['goals_pg'],             ORANGE),
                        _pill('Header %',      f"{so['header_pct']}%",     BLUE),
                    ]),
                    html.Div('SHOT ORIGIN (xG Chain Source)', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Open Play / Frontal',  so['origins']['open_play'],   GOLD),
                    _bar('From Cross',            so['origins']['cross'],        BLUE),
                    _bar('Set Piece',             so['origins']['set_piece'],    RED),
                    _bar('Fast Break',            so['origins']['fast_break'],   PURPLE),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('SHOT ZONE', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('Small Box (6-yd)',  so['zones']['small_box'],   GREEN),
                        _bar('Inside Box',        so['zones']['inside_box'],  GOLD),
                        _bar('Outside Box',       so['zones']['outside_box'], RED),
                    ]),
                ], md=5),
                dbc.Col([
                    # Shot map
                    (lambda: (
                        lambda coords: (
                            lambda p, fig, ax: (
                                [p.scatter([c['x']], [c['y']], ax=ax,
                                           color=({'Goal': GOLD, 'Saved Shot': GREEN,
                                                   'Post': PURPLE}.get(c['event'], (1,1,1,0.25))),
                                           s=55, alpha=0.8, marker='*') for c in coords],
                                _legend(ax),
                                html.Img(src=_fig_b64(fig), style={'width': '100%', 'borderRadius': '8px'})
                            )[-1]
                        )(*_make_pitch(half=True))
                    )(so['coords'])
                    )() if so['coords'] else html.Div(),
                    html.Div('★ Goal  ● On Target  ○ Off Target', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('TOP FINISHERS', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _player_table(so['finishers'], color=ORANGE),
                    ]) if so['finishers'] else html.Div(),
                    html.Div('⚠ xGOT and xG values require a trained model — not available from raw event data.',
                             style={'fontSize': '0.65rem', 'color': 'var(--text-secondary)', 'marginTop': '8px'}),
                ], md=7),
            ]),
            title='Shot Profile & xG Chain Origin', icon='🔫'
        ))
    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No offensive data available.', className='goz-card-desc')
    ])


# ════════════════════════════════════════════════════════════════
# DEFENSIVE TAB
# ════════════════════════════════════════════════════════════════

def _build_defensive(matches, team_name, rival_label):
    sections = []

    # ── 1. Defensive Shape & Pressing ─────────────────────────
    try:
        df_m = _compute_defensive(matches, team_name)
        p, fig, ax = _make_pitch()
        TYPE_C = {'Tackle': GOLD, 'Interception': BLUE, 'Clearance': RED,
                  'Challenge': PURPLE, 'Ball recovery': GREEN}
        by_type: dict = {}
        for c in df_m['heat_coords']:
            by_type.setdefault(c['type'], []).append(c)
        for t, pts in by_type.items():
            p.scatter([c['x'] for c in pts], [c['y'] for c in pts],
                      ax=ax, color=TYPE_C.get(t, (1,1,1,0.3)),
                      s=18, alpha=0.5, label=t)
        ax.axvline(x=df_m['avg_line'], color=GREEN, linestyle='--', linewidth=1.8, alpha=0.85)
        ylim = ax.get_ylim()
        ax.text(df_m['avg_line'], ylim[1] * 0.96, f"Avg Line: {df_m['avg_line']}",
                color=GREEN, fontsize=8, ha='center', va='top')
        _legend(ax)
        img = _fig_b64(fig)

        press_c = GREEN if 'High' in df_m['press_label'] else (GOLD if 'Mid' in df_m['press_label'] else RED)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Def. Actions / Game', df_m['def_pg'],       GOLD),
                        _pill('Avg Line Height',     df_m['avg_line'],      BLUE),
                        _pill('PPDA',                df_m['ppda'],          PURPLE),
                    ]),
                    html.Div(style={
                        'background': 'rgba(255,255,255,0.03)', 'borderRadius': '10px',
                        'padding': '12px', 'border': '1px solid var(--border-color)', 'marginBottom': '12px',
                    }, children=[
                        html.Div('PRESSING STYLE', style={
                            'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                            'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        html.Div(df_m['press_label'], style={
                            'fontSize': '1.1rem', 'fontWeight': '700', 'color': press_c,
                        }),
                        html.Div('PPDA < 8 = High Press · 8–14 = Mid-Block · >14 = Low/Passive', style={
                            'fontSize': '0.65rem', 'color': 'var(--text-secondary)', 'marginTop': '4px',
                        }),
                    ]),
                    html.Div('TOP DEFENSIVE PLAYERS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _player_table(df_m['top_defenders'], color=GOLD),
                    html.Div(style={'marginTop': '10px', 'fontSize': '0.67rem', 'color': 'var(--text-secondary)', 'lineHeight': '1.5'}, children=[
                        html.Span('⚠ Player speed, exact space coverage, and block gaps require tracking data.'),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div('● Defensive actions · ── Average defensive line', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=7),
            ]),
            title=f'{rival_label} — Defensive Shape & Pressing', icon='🛡'
        ))
    except Exception:
        pass

    # ── 2. Duels — Aerial & Ground ─────────────────────────────
    try:
        ae = _compute_aerials(matches, team_name)
        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div('AERIAL DUELS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '12px'}, children=[
                        _pill('Win %',      f"{ae['aerial_win']}%",  GREEN),
                        _pill('Per Game',   ae['aerial_pg'],          GOLD),
                    ]),
                    _bar('Aerial Win Rate',      ae['aerial_win'],     GREEN),
                    _bar('Box Aerial Win Rate',  ae['box_aerial_win'], ORANGE),
                    html.Div(f"Total box aerials in sample: {ae['box_aerial_tot']}", style={
                        'fontSize': '0.68rem', 'color': 'var(--text-secondary)', 'marginTop': '6px',
                    }),
                ], md=6),
                dbc.Col([
                    html.Div('GROUND DUELS (Challenge + Tackle)', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '12px'}, children=[
                        _pill('Win %',    f"{ae['ground_win']}%", GREEN),
                        _pill('Per Game', ae['ground_pg'],         GOLD),
                    ]),
                    _bar('Ground Duel Win Rate', ae['ground_win'], GREEN),
                ], md=6),
            ]),
            title='Duel Profile — Aerial vs Ground', icon='💪'
        ))
    except Exception:
        pass

    # ── 3. Vulnerable Flanks & Offside ────────────────────────
    try:
        vul = _compute_vulnerability(matches, team_name)
        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div('F3 ENTRIES CONCEDED BY FLANK', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Left Channel', vul['f3_conceded']['Left'],   BLUE),
                    _bar('Central',      vul['f3_conceded']['Center'], GOLD),
                    _bar('Right Channel',vul['f3_conceded']['Right'],  PURPLE),
                    html.Div(f"F3 entries conceded / game: {vul['f3_pg']}", style={
                        'fontSize': '0.68rem', 'color': 'var(--text-secondary)', 'marginTop': '8px',
                    }),
                ], md=6),
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'}, children=[
                        _pill('Z14 Passes Conceded / Game', vul['z14_pg'],     RED),
                        _pill('Offside Traps / Game',        vul['offside_pg'], GREEN),
                    ]),
                    html.Div(style={
                        'marginTop': '12px', 'fontSize': '0.68rem',
                        'color': 'var(--text-secondary)', 'lineHeight': '1.5',
                    }, children=[
                        html.P('Offside trap count = Offside Provoked events.'),
                        html.P('High Z14 passes conceded → vulnerable in central corridor.'),
                    ]),
                ], md=6),
            ]),
            title='Vulnerable Flanks & Zone 14 Control', icon='⚠'
        ))
    except Exception:
        pass

    # ── 4. Pre-Goal Structure (Golden 30s) ─────────────────────
    try:
        pg = _compute_pre_goal(matches, team_name)
        sections.append(_card(
            html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                _pill('Goals Conceded',            pg['goals_conceded'],   RED),
                _pill('Goals / Game',              pg['goals_pg'],          RED),
                _pill('Avg Def Acts (30s)',         pg['avg_def_acts'],      GOLD),
                _pill('Avg Opp Passes (30s)',       pg['avg_opp_pas'],       BLUE),
                _pill('Failed Clearances (30s)',    pg['avg_fc'],            ORANGE),
            ]),
            html.Div('Average values in the 30 seconds before each goal conceded', style={
                'fontSize': '0.68rem', 'color': 'var(--text-secondary)',
                'marginTop': '10px', 'textAlign': 'center',
            }),
            title='Pre-Goal Structure — Golden 30s Window', icon='🥅'
        ))
    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No defensive data available.', className='goz-card-desc')
    ])


# ════════════════════════════════════════════════════════════════
# TRANSITIONS TAB
# ════════════════════════════════════════════════════════════════

def _build_transitions(matches, team_name, rival_label):
    sections = []
    try:
        tr = _compute_transitions(matches, team_name)

        # ── Ball Wins (Attacking Transitions) ─────────────────
        p_w, fig_w, ax_w = _make_pitch()
        for c in tr['win']['coords']:
            p_w.scatter([c['x']], [c['y']], ax=ax_w, color=GREEN, s=18, alpha=0.5)
        ax_w.scatter([], [], color=GREEN, s=30, label='Ball Recovery')
        _legend(ax_w)
        img_w = _fig_b64(fig_w)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Ball Wins / Game', tr['win']['per_game'], GREEN),
                        _pill('→ F3 Entry %',     f"{tr['win']['outcomes']['f3']}%",      GOLD),
                        _pill('→ Shot %',         f"{tr['win']['outcomes']['shot']}%",    BLUE),
                        _pill('→ Opp Shot %',     f"{tr['win']['outcomes']['opp_shot']}%", RED),
                    ]),
                    html.Div('RECOVERY ZONE', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Defensive 3rd', tr['win']['zones']['Defensive 3rd'], RED),
                    _bar('Middle 3rd',    tr['win']['zones']['Middle 3rd'],    GOLD),
                    _bar('Final 3rd',     tr['win']['zones']['Final 3rd'],     GREEN),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('TOP RECOVERERS', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _player_table(tr['win']['top'], color=GREEN),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=img_w, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div('● Ball recovery locations', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=7),
            ]),
            title=f'{rival_label} — Attacking Transitions (Ball Wins)', icon='⚡'
        ))

        # ── Ball Losses (Defensive Transitions) ───────────────
        p_l, fig_l, ax_l = _make_pitch()
        for c in tr['loss']['coords']:
            p_l.scatter([c['x']], [c['y']], ax=ax_l, color=RED, s=18, alpha=0.5)
        ax_l.scatter([], [], color=RED, s=30, label='Ball Loss')
        _legend(ax_l)
        img_l = _fig_b64(fig_l)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Ball Losses / Game', tr['loss']['per_game'],                    RED),
                        _pill('→ Opp F3 %',         f"{tr['loss']['outcomes']['opp_f3']}%",    ORANGE),
                        _pill('→ Opp Shot %',        f"{tr['loss']['outcomes']['opp_shot']}%",  RED),
                        _pill('→ Recovered %',       f"{tr['loss']['outcomes']['recovered']}%", GREEN),
                    ]),
                    html.Div('BALL LOSS ZONE', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Defensive 3rd', tr['loss']['zones']['Defensive 3rd'], RED),
                    _bar('Middle 3rd',    tr['loss']['zones']['Middle 3rd'],    GOLD),
                    _bar('Final 3rd',     tr['loss']['zones']['Final 3rd'],     GREEN),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('TOP BALL LOSERS', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _player_table(tr['loss']['top'], color=RED),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=img_l, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div('● Ball loss locations', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=7),
            ]),
            title=f'{rival_label} — Defensive Transitions (Ball Losses)', icon='🔄'
        ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No transition data available.', className='goz-card-desc')
    ])


# ════════════════════════════════════════════════════════════════
# SHOT PROFILE TAB (comprehensive)
# ════════════════════════════════════════════════════════════════

def _build_shot_profile(matches, team_name, rival_label):
    sections = []
    try:
        so = _compute_shot_origin(matches, team_name)

        # Full shot map
        p, fig, ax = _make_pitch(half=True, figsize=(11, 7))
        EV_STYLE = {
            'Goal':       (GOLD,                60, '*'),
            'Saved Shot': (GREEN,               45, 'o'),
            'Post':       (PURPLE,              40, 'D'),
            'Miss':       ((1, 1, 1, 0.2),      25, 'o'),
        }
        for ev, (color, size, marker) in EV_STYLE.items():
            pts = [c for c in so['coords'] if c['event'] == ev]
            if pts:
                p.scatter([c['x'] for c in pts], [c['y'] for c in pts],
                          ax=ax, color=color, s=size, alpha=0.85,
                          marker=marker, label=ev,
                          edgecolors=(1, 1, 1, 0.1), linewidths=0.4)
        _legend(ax)
        img = _fig_b64(fig)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Shots / Game',   so['total_pg'],             GOLD),
                        _pill('On Target %',    f"{so['on_target_pct']}%",  GREEN),
                        _pill('Goals / Game',   so['goals_pg'],             ORANGE),
                        _pill('Header %',       f"{so['header_pct']}%",     BLUE),
                    ]),
                    html.Div('SHOT ORIGIN', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Open Play / Frontal', so['origins']['open_play'],  GOLD),
                    _bar('From Cross',           so['origins']['cross'],      BLUE),
                    _bar('Set Piece',            so['origins']['set_piece'],  RED),
                    _bar('Fast Break',           so['origins']['fast_break'], PURPLE),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('SHOT ZONE', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _bar('Small Box (6-yd area)', so['zones']['small_box'],   GREEN),
                        _bar('Inside Penalty Box',     so['zones']['inside_box'],  GOLD),
                        _bar('Outside Box (30m+)',      so['zones']['outside_box'], RED),
                    ]),
                    html.Div(style={'marginTop': '12px'}, children=[
                        html.Div('TOP FINISHERS', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        _player_table(so['finishers'], color=ORANGE),
                    ]) if so['finishers'] else html.Div(),
                ], md=4),
                dbc.Col([
                    html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                    html.Div('★ Goal  ● On Target  ○ Off Target  ◆ Post', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                    html.Div(style={
                        'marginTop': '14px', 'padding': '12px',
                        'background': 'rgba(255,255,255,0.03)',
                        'borderRadius': '10px', 'border': '1px solid var(--border-color)',
                        'fontSize': '0.68rem', 'color': 'var(--text-secondary)', 'lineHeight': '1.65',
                    }, children=[
                        html.Strong('xG Proxy (location-based estimate):', style={'color': GOLD, 'display': 'block', 'marginBottom': '4px'}),
                        html.Span('Small box ≈ 0.45 · Inside box ≈ 0.15 · Outside box ≈ 0.05 · Headers ÷ 2'),
                        html.Br(),
                        html.Span('For calibrated xG and xGOT values a dedicated model is required.'),
                    ]),
                ], md=8),
            ]),
            title=f'{rival_label} — Full Shot Profile', icon='🎯'
        ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No shot profile data available.', className='goz-card-desc')
    ])


# ════════════════════════════════════════════════════════════════
# DATA FEASIBILITY NOTE
# ════════════════════════════════════════════════════════════════

def _feasibility_note():
    available = [
        'Build-up pass type (short / long)',
        'Build-up zone (left / center / right)',
        'Key build-up players',
        '15-second outcomes after ball win',
        'Final third entry method & zone',
        'Zone 14 passes, duels, shots',
        'Playmaker identification (progressive passes)',
        'Tempo (passes per game)',
        'GK involvement in build-up',
        'Winger touches in wide final third',
        'Full-back pass contribution',
        'Who finishes attacks',
        'Shot origin: cross / set-piece / open-play / fast-break',
        'Shot zone: inside box / outside box',
        'Shots on target %',
        'Header %',
        'xG chain source approximation',
        'Pressing intensity (PPDA proxy)',
        'Defensive line height',
        'Aerial duels (win %, box vs rest)',
        'Ground duels (win %)',
        'Pre-goal 30-second window',
        'Vulnerable flanks (F3 entries conceded)',
        'Zone 14 control (defensive)',
        'Offside trap (provoked events)',
        'Ball win / loss locations & 15-second outcomes',
        'Who wins / loses ball most',
        'Shot location & accuracy',
    ]
    not_available = [
        'Exact triangle formation detection (requires tracking / positional data)',
        'Numerical superiority by zone (requires simultaneous player positions)',
        'Player speed and distance data (requires GPS / tracking)',
        'Block-to-block gap analysis (requires tracking)',
        'Calibrated xG / xGOT values (requires trained model)',
    ]
    return _card(
        dbc.Row([
            dbc.Col([
                html.Div('✅ Available from event data', style={
                    'fontSize': '0.75rem', 'fontWeight': '700', 'color': GREEN,
                    'letterSpacing': '0.5px', 'marginBottom': '10px',
                }),
                html.Ul([html.Li(x, style={'fontSize': '0.72rem', 'color': 'var(--text-secondary)', 'marginBottom': '3px'})
                         for x in available], style={'paddingLeft': '16px', 'margin': '0'}),
            ], md=6),
            dbc.Col([
                html.Div('❌ Requires tracking / model (not in event logs)', style={
                    'fontSize': '0.75rem', 'fontWeight': '700', 'color': RED,
                    'letterSpacing': '0.5px', 'marginBottom': '10px',
                }),
                html.Ul([html.Li(x, style={'fontSize': '0.72rem', 'color': 'var(--text-secondary)', 'marginBottom': '3px'})
                         for x in not_available], style={'paddingLeft': '16px', 'margin': '0'}),
            ], md=6),
        ]),
        title='Event Data Feasibility', icon='📊'
    )


# ════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════

def layout():
    rivals = _discover_rivals()
    default_value = next(iter(rivals.values()), None)
    rival_options = [{'label': label, 'value': value} for label, value in rivals.items()]
    tab_options   = [
        {'label': '⚔️  Offensive',   'value': 'off'},
        {'label': '🛡  Defensive',    'value': 'def'},
        {'label': '⚡  Transitions',  'value': 'trans'},
        {'label': '🎯  Shot Profile', 'value': 'shots'},
        {'label': '📊  Feasibility',  'value': 'info'},
    ]

    return html.Div(className='page-wrap', children=[
        html.Div(className='goz-hero', children=[
            html.Div(className='goz-hero-content', children=[
                dcc.Link('← GÖZTEPE HUB', href='/', className='goz-back-link'),
                html.H1('RIVAL SCOUT', className='goz-hub-title'),
                html.P('Per-match deep dive from the latest available event-data sample',
                       className='goz-hub-subtitle'),
                html.Div(style={'marginTop': '24px', 'width': '100%', 'maxWidth': '380px'}, children=[
                    html.Label('SELECT RIVAL', className='goz-label'),
                    dcc.Dropdown(
                        id='scout-rival-selector',
                        options=rival_options,
                        value=default_value,
                        className='goz-dropdown',
                        clearable=False,
                        searchable=True,
                    ),
                ]),
            ]),
        ]),

        html.Div(className='content-container', style={'padding': '0 20px 60px'}, children=[

            # ── Match selector (populated dynamically) ────────
            html.Div(style={'margin': '28px 0 16px'}, children=[
                html.Label('SELECT MATCH', className='goz-label',
                           style={'marginBottom': '10px', 'display': 'block'}),
                html.Div(id='scout-match-selector-container'),
            ]),

            # ── Selected match summary card ───────────────────
            html.Div(id='scout-match-card'),

            # ── Analysis tab selector ─────────────────────────
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'margin': '20px 0'}, children=[
                dbc.RadioItems(
                    id='scout-tab',
                    options=tab_options,
                    value='off',
                    inline=True,
                    className='pm-tab-radio-group',
                    inputClassName='pm-tab-radio-input',
                    labelClassName='pm-tab-radio-label',
                ),
            ]),

            # ── Tab content ───────────────────────────────────
            html.Div(id='scout-tab-content', style={'marginTop': '8px'}),
        ]),

        html.Footer(className='footer', children=[
            html.Div(className='footer-inner', children=[
                html.Div('© tactIQ Göztepe Hub — Rival Scout', className='footer-text'),
                html.Img(src='/assets/superlig_logo.jpg', className='superlogo'),
            ])
        ])
    ])


# ════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════

@callback(
    Output('scout-match-selector-container', 'children'),
    Input('scout-rival-selector', 'value'),
)
def update_match_options(rival_label):
    """Populate the match selector whenever the rival changes."""
    team_name = rival_label or ''
    if not team_name:
        return html.Div()

    matches = _load_rival_matches(team_name)
    res_colors = {'W': GREEN, 'D': GOLD, 'L': RED}

    options = []
    for fname, df in matches:
        s = _match_summary(fname, df, team_name)
        rc = res_colors.get(s['result'], GOLD)
        label = html.Div(style={'textAlign': 'center', 'lineHeight': '1.3'}, children=[
            html.Div(f"vs {s['opponent']}", style={
                'fontSize': '0.8rem', 'fontWeight': '700', 'color': 'var(--text-primary)',
            }),
            html.Div(s['score'], style={
                'fontSize': '1.15rem', 'fontWeight': '800', 'color': rc,
            }),
            html.Div(s['result'], style={
                'fontSize': '0.65rem', 'fontWeight': '700',
                'color': rc, 'letterSpacing': '1px',
            }),
        ])
        options.append({'label': label, 'value': fname})

    default = options[0]['value'] if options else None

    return dbc.RadioItems(
        id='scout-match-selector',
        options=options,
        value=default,
        inline=True,
        className='pm-tab-radio-group',
        inputClassName='pm-tab-radio-input',
        labelClassName='pm-tab-radio-label',
        style={'gap': '10px'},
    )


@callback(
    [Output('scout-match-card',    'children'),
     Output('scout-tab-content',   'children')],
    [Input('scout-rival-selector', 'value'),
     Input('scout-match-selector', 'value'),
     Input('scout-tab',            'value')],
)
def update_scout_content(rival_label, selected_file, active_tab):
    if not rival_label or not selected_file:
        return html.Div(), html.Div()

    team_name = rival_label or ''
    if not team_name:
        return html.Div(), html.Div()

    # Find the selected match df from cache
    all_matches = _load_rival_matches(team_name)
    match_pair  = [(f, df) for f, df in all_matches if f == selected_file]
    if not match_pair:
        return html.Div(), html.Div()

    fname, df   = match_pair[0]
    single      = [(fname, df)]          # analysis functions accept a list

    rival_display = _short(team_name)
    match_card = _build_selected_match_card(fname, df, team_name, rival_display)

    if   active_tab == 'off':   content = _build_offensive(single, team_name, rival_display)
    elif active_tab == 'def':   content = _build_defensive(single, team_name, rival_display)
    elif active_tab == 'trans': content = _build_transitions(single, team_name, rival_display)
    elif active_tab == 'shots': content = _build_shot_profile(single, team_name, rival_display)
    elif active_tab == 'info':  content = _feasibility_note()
    else:                       content = html.Div()

    return match_card, content

import dash
import numpy as np
import pandas as pd
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'Karagümrük':  'Fatih Karagümrük Spor Kulübü',
}

# Exact 6 matches per rival: 2 wins, 2 draws, 2 losses.
RIVAL_FILES = {
    'Galatasaray Spor Kulübü': [
        'gs-antalyaspor.parquet',
        'gençlerbirliği-gs.parquet',
        'gs-kocaeli.parquet',
        'gs-gaziantep.parquet',
        'kasımpaşa-gs.parquet',
        'samsun-gs.parquet',
    ],
    'Fenerbahçe Spor Kulübü': [
        'konya-fb.parquet',
        'kayseri-fb.parquet',
        'fb-eyup.parquet',
        'fb-rize.parquet',
        'gs-fb.parquet',
        'fatih-fb.parquet',
    ],
    'Beşiktaş Jimnastik Kulübü': [
        'gaziantep-bjk.parquet',
        'bjk-antalya.parquet',
        'rize-bjk.parquet',
        'bjk-alanyaspor.parquet',
        'fb-bjk.parquet',
        'samsun-bjk.parquet',
    ],
    'Trabzonspor Kulübü': [
        'bjk-ts.parquet',
        'ts-gs.parquet',
        'alanya-ts.parquet',
        'kayseri-ts.parquet',
        'ts-gençlerbirliği.parquet',
        'konya-ts.parquet',
    ],
    'İstanbul Başakşehir Futbol Kulübü': [
        'gaziantep-başakşehir.parquet',
        'başakşehir-samsun.parquet',
        'ts-başakşehir.parquet',
        'kocaeli-başakşehir.parquet',
        'fb-başakşehir.parquet',
        'gs-basaksehir.parquet',
    ],
    'Fatih Karagümrük Spor Kulübü': [
        'fatih-alanya.parquet',
        'kocaeli-fatih.parquet',
        'bjk-fatih.parquet',
        'gaziantep-fatih.parquet',
        'konya-fatih.parquet',
        'kayseri-fatih.parquet',
    ],
}


def _discover_rivals():
    """Return the fixed rival scout team list requested for this view."""
    return dict(RIVALS)

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

GOAL_MOUTH_Y_COL = 'Goal Mouth Y Coordinate'
GOAL_MOUTH_Z_COL = 'Goal Mouth Z Coordinate'
GOAL_MOUTH_LEFT_OPT = 45.2
GOAL_MOUTH_RIGHT_OPT = 54.8

SHOT_EVENTS = {'Goal', 'Miss', 'Saved Shot', 'Post'}
DEF_EVENTS  = {'Tackle', 'Challenge', 'Interception', 'Clearance', 'Ball recovery'}
RECOVERY_EVENTS = {'Ball recovery', 'Ball Recovery', 'Interception', 'Tackle'}
LOSS_EVENTS = {'Dispossessed', 'Error', 'Blocked Pass'}
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

    match_limit = 6 if team_name in RIVAL_FILES else 5
    matches = sorted(matches, key=sort_key, reverse=True)[:match_limit]

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

def _safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def _event_seconds(row) -> float:
    return float(row.get('time_min') or 0) * 60 + float(row.get('time_sec') or 0)

def _is_yes(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {'si', 'yes', 'true', '1'}

def _has_qualifier(row, columns) -> bool:
    for col in columns:
        if col in row and _is_yes(row.get(col)):
            return True
    return False

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

def _corner_events(df: pd.DataFrame) -> pd.DataFrame:
    corners = df[_corner_taken_mask(df)].copy()
    if corners.empty:
        return corners

    subset = [
        col for col in ['team_name', 'period_id', 'time_min', 'time_sec', 'player_name', 'x', 'y']
        if col in corners.columns
    ]
    if subset:
        corners = corners.drop_duplicates(subset=subset)
    return corners

def _free_kick_taken_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    taken_cols = [
        col for col in df.columns
        if isinstance(col, str) and 'free' in col.lower()
        and 'kick' in col.lower() and 'taken' in col.lower()
    ]
    for col in taken_cols:
        mask |= _truthy_series(df[col])

    if not taken_cols and 'event' in df.columns:
        mask |= df['event'].astype(str).str.strip().str.lower().isin({
            'free kick', 'free-kick', 'direct free kick', 'indirect free kick',
        })

    return mask

def _free_kick_events(df: pd.DataFrame, min_x: float = 60) -> pd.DataFrame:
    free_kicks = df[_free_kick_taken_mask(df)].copy()
    if free_kicks.empty:
        return free_kicks

    if min_x is not None and 'x' in free_kicks.columns:
        free_kicks = free_kicks[pd.to_numeric(free_kicks['x'], errors='coerce') > min_x]
        if free_kicks.empty:
            return free_kicks

    subset = [
        col for col in ['team_name', 'period_id', 'time_min', 'time_sec', 'player_name', 'x', 'y']
        if col in free_kicks.columns
    ]
    if subset:
        free_kicks = free_kicks.drop_duplicates(subset=subset)
    return free_kicks

def _clean_records(df: pd.DataFrame) -> list:
    if df.empty or 'event' not in df.columns:
        return []
    clean = df[~df['event'].isin(SKIP_EVENTS)].copy()
    clean['_abs_time'] = clean.apply(_event_seconds, axis=1)
    sort_cols = [c for c in ['period_id', '_abs_time', 'event_id'] if c in clean.columns]
    if sort_cols:
        clean = clean.sort_values(sort_cols)
    return clean.to_dict('records')

def _safe_xy(row, x_col='x', y_col='y'):
    try:
        x = float(row.get(x_col))
        y = float(row.get(y_col))
    except Exception:
        return None
    if np.isnan(x) or np.isnan(y):
        return None
    return x, y


# ════════════════════════════════════════════════════════════════
# OFFENSIVE DATA
# ════════════════════════════════════════════════════════════════

def _estimate_dimensions(df, team_name, phase):
    team_df = df[df['team_name'] == team_name]
    team_df = team_df.dropna(subset=['x', 'y'])
    
    if phase == 'low':
        phase_df = team_df[team_df['x'] < 33.3]
    elif phase == 'mid':
        phase_df = team_df[(team_df['x'] >= 33.3) & (team_df['x'] < 66.6)]
    else:
        phase_df = team_df[team_df['x'] >= 66.6]
        
    if len(phase_df) < 5:
        if phase == 'low':
            return 15, 36, 50
        elif phase == 'mid':
            return 41, 34, 51
        else:
            return 58, 35, 39
            
    xs = phase_df['x'].values
    ys = phase_df['y'].values
    
    min_x = np.percentile(xs, 15)
    max_x = np.percentile(xs, 85)
    min_y = np.percentile(ys, 15)
    max_y = np.percentile(ys, 85)
    
    line_height = round(min_x * 1.05)
    team_length = round((max_x - min_x) * 1.05)
    team_width = round((max_y - min_y) * 0.68)
    
    line_height = max(5, min(95, line_height))
    team_length = max(10, min(80, team_length))
    team_width = max(15, min(68, team_width))
    
    return line_height, team_length, team_width

def _draw_possession_line_height(matches, team_name, rival_label):
    if not matches:
        raise ValueError("No match data available.")
    
    fname, df = matches[0]
    
    low_lh, low_len, low_wid = _estimate_dimensions(df, team_name, 'low')
    mid_lh, mid_len, mid_wid = _estimate_dimensions(df, team_name, 'mid')
    f3_lh, f3_len, f3_wid = _estimate_dimensions(df, team_name, 'f3')
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), facecolor='#0e1b0f')
    
    phases = [
        ('Build Up Low', low_lh, low_len, low_wid),
        ('Build Up Mid', mid_lh, mid_len, mid_wid),
        ('Final Third Phase', f3_lh, f3_len, f3_wid)
    ]
    
    for idx, (title, lh, length, width) in enumerate(phases):
        ax = axes[idx]
        ax.set_facecolor('#0e1b0f')
        
        # Draw vertical grass stripes
        for i in range(10):
            y0 = i * 10
            ax.fill_between([0, 100], y0, y0 + 10,
                            color='#18331b' if i % 2 == 0 else '#122615',
                            alpha=0.9, zorder=0)
            
        lc = (1.0, 1.0, 1.0, 0.4)
        lw = 1.4
        
        # Pitch outline
        ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, linewidth=lw, zorder=3)
        # Midfield line
        ax.plot([0, 100], [50, 50], color=lc, linewidth=lw, zorder=3)
        # Center circle
        center_circle = plt.Circle((50, 50), 9.15, color=lc, fill=False, linewidth=lw, zorder=3)
        ax.add_patch(center_circle)
        ax.scatter([50.0], [50.0], color=lc, s=12, zorder=3)
        
        # Bottom penalty area
        ax.plot([21.1, 21.1, 78.9, 78.9], [0, 17, 17, 0], color=lc, linewidth=lw, zorder=3)
        ax.plot([40.9, 40.9, 59.1, 59.1], [0, 5.8, 5.8, 0], color=lc, linewidth=lw - 0.3, zorder=3)
        
        # Top penalty area
        ax.plot([21.1, 21.1, 78.9, 78.9], [100, 83, 83, 100], color=lc, linewidth=lw, zorder=3)
        ax.plot([40.9, 40.9, 59.1, 59.1], [100, 94.2, 94.2, 100], color=lc, linewidth=lw - 0.3, zorder=3)
        
        # DIRECTION indicator on the right side
        ax.annotate('', xy=(95, 90), xytext=(95, 10),
                    arrowprops=dict(arrowstyle="->", color=lc, lw=1.2, alpha=0.6))
        ax.text(97, 50, 'DIRECTION', color=(1, 1, 1, 0.4), fontsize=8, rotation=90, ha='center', va='center', fontweight='bold')
        
        # Draw blue box
        lh_opta = lh / 1.05
        len_opta = length / 1.05
        wid_opta = width / 0.68
        
        py_min = lh_opta
        py_max = lh_opta + len_opta
        px_min = 50.0 - wid_opta / 2.0
        px_max = 50.0 + wid_opta / 2.0
        
        # Cap dimensions to stay inside the pitch
        if py_max > 100:
            py_max = 100.0
            py_min = max(0.0, 100.0 - len_opta)
        if px_min < 0:
            px_min = 0.0
            px_max = min(100.0, wid_opta)
        if px_max > 100:
            px_max = 100.0
            px_min = max(0.0, 100.0 - wid_opta)
            
        rect = plt.Rectangle((px_min, py_min), px_max - px_min, py_max - py_min,
                             facecolor='#3b82f6', alpha=0.35, edgecolor='#3b82f6', linewidth=2.0, zorder=4)
        ax.add_patch(rect)
        
        # Draw brackets and text boxes
        # 1. Width bracket
        wy = py_max + 3.5
        if wy > 98: wy = py_max - 3.5
        ax.plot([px_min, px_max], [wy, wy], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.plot([px_min, px_min], [wy - 1.5, wy + 1.5], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.plot([px_max, px_max], [wy - 1.5, wy + 1.5], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.text(50.0, wy, f"{width}m", color='white', fontsize=9, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f2937', edgecolor='none'), zorder=6)
        
        # 2. Length bracket
        lx = px_min - 3.5
        if lx < 2: lx = px_min + 3.5
        ax.plot([lx, lx], [py_min, py_max], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.plot([lx - 1.5, lx + 1.5], [py_min, py_min], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.plot([lx - 1.5, lx + 1.5], [py_max, py_max], color='white', alpha=0.7, linewidth=1.2, zorder=5)
        ax.text(lx, (py_min + py_max)/2.0, f"{length}m", color='white', fontsize=9, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f2937', edgecolor='none'), zorder=6)
                
        # 3. Line height bracket
        hx = px_max + 3.5
        if hx > 98: hx = px_max - 3.5
        if py_min > 2.0:
            ax.plot([hx, hx], [0, py_min], color='white', alpha=0.7, linewidth=1.2, zorder=5)
            ax.plot([hx - 1.5, hx + 1.5], [0, 0], color='white', alpha=0.7, linewidth=1.2, zorder=5)
            ax.plot([hx - 1.5, hx + 1.5], [py_min, py_min], color='white', alpha=0.7, linewidth=1.2, zorder=5)
            ax.text(hx, py_min/2.0, f"{lh}m", color='white', fontsize=9, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1f2937', edgecolor='none'), zorder=6)
        
        ax.set_xlim(-2, 102)
        ax.set_ylim(-3, 103)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title.upper(), color='#fbbf24', fontsize=12, fontweight='bold', pad=15)
        
    plt.tight_layout()
    img_b64 = _fig_b64(fig)
    return img_b64

def _build_possession_line_height(matches, team_name, rival_label):
    try:
        img_b64 = _draw_possession_line_height(matches, team_name, rival_label)
        
        return _card(
            html.Div(children=[
                html.P("Calculated vertical line height, team length, and horizontal team width "
                       "during low build-up, mid build-up, and final third possession phases.",
                       style={'fontSize': '0.75rem', 'color': 'var(--text-secondary)', 'marginBottom': '16px', 'textAlign': 'center'}),
                html.Img(src=img_b64, style={'width': '100%', 'borderRadius': '8px'}),
            ]),
            title='IN POSSESSION LINE HEIGHT & TEAM LENGTH',
            icon='⚽'
        )
    except Exception:
        import traceback
        traceback.print_exc()

def _build_offensive(matches, team_name, rival_label):
    sections = []
    try:
        shot_img, shots = _draw_shot_map(matches, team_name)
        goals = [s for s in shots if s['event'] == 'Goal']
        on_target = [s for s in shots if s['event'] in {'Goal', 'Saved Shot'}]
        shot_card = _card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Shots', len(shots), BLUE),
                        _pill('On Target', len(on_target), GREEN),
                        _pill('Goals', len(goals), GOLD),
                        _pill('Shot Conv.', f"{_safe_pct(len(goals), len(shots))}%", PURPLE),
                    ]),
                    html.Div('SHOT OUTCOMES', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Goals', _safe_pct(len(goals), len(shots)), GOLD),
                    _bar('Saved', _safe_pct(len([s for s in shots if s['event'] == 'Saved Shot']), len(shots)), GREEN),
                    _bar('Miss / Post', _safe_pct(len([s for s in shots if s['event'] in {'Miss', 'Post'}]), len(shots)), RED),
                ], md=4),
                dbc.Col([
                    html.Img(src=shot_img, style={'width': '100%', 'maxHeight': '430px', 'objectFit': 'contain', 'borderRadius': '8px'}),
                    html.Div('★ Goal  ● Saved  ○ Miss  ◆ Post', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=8),
            ]),
            title=f'{rival_label} — Offensive Phase Shot Map', icon='⚔'
        )
        sections.append(shot_card)
    except Exception:
        pass

    try:
        seq_img, goal_chains = _draw_goal_sequences_filtered(matches, team_name)
        if seq_img:
            sequence_rows = []
            for item in goal_chains[:4]:
                chain = item['chain']
                sequence_rows.append(html.Div(style={
                    'display': 'flex', 'justifyContent': 'space-between',
                    'padding': '6px 0', 'borderBottom': '1px solid rgba(255,255,255,0.05)',
                }, children=[
                    html.Span(item['match'], style={'fontSize': '0.75rem', 'color': 'var(--text-secondary)'}),
                    html.Span(f"{item.get('origin', 'open_play').replace('_', ' ')} · {len(chain)} actions", style={'fontSize': '0.75rem', 'fontWeight': '700', 'color': GOLD}),
                ]))
            seq_card = _card(
                dbc.Row([
                    dbc.Col([
                        html.Div('GOAL CHAINS FOUND', style={
                            'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                            'letterSpacing': '1px', 'marginBottom': '8px',
                        }),
                        html.Div(sequence_rows),
                    ], md=4),
                    dbc.Col([
                        html.Img(src=seq_img, style={'width': '100%', 'maxHeight': '430px', 'objectFit': 'contain', 'borderRadius': '8px'}),
                        html.Div('Last attacking actions in the 10 seconds before each goal', style={
                            'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                            'textAlign': 'center', 'marginTop': '4px',
                        }),
                    ], md=8),
                ]),
                title='Goal Sequence Map', icon='🥅'
            )
            sections.append(seq_card)
    except Exception:
        pass

    possession = _build_possession_line_height(matches, team_name, rival_label)
    if possession:
        sections.append(possession)

    if len(sections) >= 2:
        top_row = dbc.Row([
            dbc.Col(sections[0], md=6),
            dbc.Col(sections[1], md=6),
        ], className='g-3', style={'marginBottom': '12px'})
        rest = sections[2:]
        return html.Div([top_row] + rest)

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No offensive data available.', className='goz-card-desc')
    ])


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
    entries_coords = []
    z14_coords = []

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
                zone = _zone_y(ey)
                f3_c[zone] += 1
                try:
                    entries_coords.append({
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'end_x': float(row['Pass End X']),
                        'end_y': ey,
                        'zone': zone,
                    })
                except Exception:
                    pass

            z14 = opp_passes[
                (opp_passes['Pass End X'].fillna(0) >= 66) &
                (opp_passes['Pass End Y'].fillna(50).between(33, 67))
            ]
            z14_c += len(z14)
            for _, row in z14.iterrows():
                try:
                    z14_coords.append({
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'end_x': float(row['Pass End X']),
                        'end_y': float(row['Pass End Y']),
                    })
                except Exception:
                    pass

        offside += len(team_ev[team_ev['event'] == 'Offside provoked'])

    n  = max(len(matches), 1)
    ft = max(sum(f3_c.values()), 1)
    return {
        'f3_conceded':  {k: _safe_pct(v, ft) for k, v in f3_c.items()},
        'f3_pg':        round(sum(f3_c.values()) / n, 1),
        'z14_pg':       round(z14_c / n, 1),
        'offside_pg':   round(offside / n, 1),
        'entries_raw':   f3_c,
        'entries_coords': entries_coords[:120],
        'z14_coords':    z14_coords[:80],
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
    wz = {'Defensive 3rd': 0, 'Middle 3rd': 0, 'Final 3rd': 0}
    lz = {'Defensive 3rd': 0, 'Middle 3rd': 0, 'Final 3rd': 0}
    wo = {'goal': 0, 'shot': 0, 'f3': 0, 'lost': 0, 'retained': 0}
    lo = {'opp_goal': 0, 'opp_shot': 0, 'opp_f3': 0, 'recovered': 0, 'survived': 0}
    wp = {}; lp = {}
    wc = []; lc = []
    w_after = []; l_after = []
    wt = lt = 0

    for _, df in matches:
        recs = _clean_records(df)
        for i, row in enumerate(recs):
            if row['event'] in SKIP_EVENTS:
                continue

            t0 = float(row.get('_abs_time', _event_seconds(row)))

            # ── Ball WIN ──────────────────────────────────────────
            if (row['team_name'] == team_name and
                    row['event'] in RECOVERY_EVENTS and
                    int(row.get('outcome', 1)) == 1):
                xy = _safe_xy(row)
                if not xy:
                    continue
                wt += 1
                wz[_zone_x(xy[0])] += 1
                name = row.get('player_name', '?') or '?'
                wp[name] = wp.get(name, 0) + 1

                reached_f3 = shot = goal = lost = False
                final_xy = None
                final_event = None

                for j in range(i + 1, len(recs)):
                    r2 = recs[j]
                    t2 = float(r2.get('_abs_time', _event_seconds(r2)))
                    if t2 - t0 > 10: break
                    if r2['event'] in SKIP_EVENTS: continue
                    if r2['team_name'] == team_name:
                        xy2 = _safe_xy(r2)
                        if xy2:
                            final_xy = xy2
                            final_event = r2['event']
                        if float(r2.get('x', 0) or 0) >= 66:
                            reached_f3 = True
                        if r2['event'] in SHOT_EVENTS:
                            shot = True
                            if r2['event'] == 'Goal':
                                goal = True
                            break
                    else:
                        lost = True
                        break

                if goal:
                    outcome = 'goal'
                elif shot:
                    outcome = 'shot'
                elif reached_f3:
                    outcome = 'f3'
                elif lost:
                    outcome = 'lost'
                else:
                    outcome = 'retained'
                wo[outcome] += 1
                wc.append({'x': xy[0], 'y': xy[1], 'outcome': outcome})
                if final_xy:
                    w_after.append({
                        'x0': xy[0], 'y0': xy[1], 'x1': final_xy[0], 'y1': final_xy[1],
                        'event': final_event, 'outcome': outcome,
                    })

            # ── Ball LOSS ─────────────────────────────────────────
            if (row['team_name'] == team_name and
                    row['event'] in LOSS_EVENTS):
                xy = _safe_xy(row)
                if not xy:
                    continue
                lt += 1
                lz[_zone_x(xy[0])] += 1
                name = row.get('player_name', '?') or '?'
                lp[name] = lp.get(name, 0) + 1

                opp_reached_f3 = opp_shot = opp_goal = recovered = False
                final_xy = None
                final_event = None

                for j in range(i + 1, len(recs)):
                    r2 = recs[j]
                    t2 = float(r2.get('_abs_time', _event_seconds(r2)))
                    if t2 - t0 > 10: break
                    if r2['event'] in SKIP_EVENTS: continue
                    if r2['team_name'] != team_name:
                        xy2 = _safe_xy(r2)
                        if xy2:
                            final_xy = xy2
                            final_event = r2['event']
                        if float(r2.get('x', 0) or 0) >= 66:
                            opp_reached_f3 = True
                        if r2['event'] in SHOT_EVENTS:
                            opp_shot = True
                            if r2['event'] == 'Goal':
                                opp_goal = True
                            break
                    else:
                        if r2['event'] in RECOVERY_EVENTS:
                            recovered = True
                        break

                if opp_goal:
                    outcome = 'opp_goal'
                elif opp_shot:
                    outcome = 'opp_shot'
                elif opp_reached_f3:
                    outcome = 'opp_f3'
                elif recovered:
                    outcome = 'recovered'
                else:
                    outcome = 'survived'
                lo[outcome] += 1
                lc.append({'x': xy[0], 'y': xy[1], 'outcome': outcome})
                if final_xy:
                    l_after.append({
                        'x0': xy[0], 'y0': xy[1], 'x1': final_xy[0], 'y1': final_xy[1],
                        'event': final_event, 'outcome': outcome,
                    })

    n = max(len(matches), 1)

    def _pct_dict(d, total):
        t = max(total, 1)
        return {k: _safe_pct(v, t) for k, v in d.items()}

    return {
        'win': {
            'total': wt, 'per_game': round(wt / n, 1),
            'zones': _pct_dict(wz, wt),
            'outcomes': _pct_dict(wo, wt),
            'outcome_counts': wo,
            'top': sorted(wp.items(), key=lambda x: -x[1])[:8],
            'coords': wc[:400],
            'after': w_after[:160],
        },
        'loss': {
            'total': lt, 'per_game': round(lt / n, 1),
            'zones': _pct_dict(lz, lt),
            'outcomes': _pct_dict(lo, lt),
            'outcome_counts': lo,
            'top': sorted(lp.items(), key=lambda x: -x[1])[:8],
            'coords': lc[:400],
            'after': l_after[:160],
        },
    }


def _compute_set_pieces(matches, team_name):
    corners = 0
    fks = 0
    penalties = 0
    sp_shots = 0
    sp_goals = 0
    sp_shot_coords = []

    for _, df in matches:
        team = df[df['team_name'] == team_name]

        corners += len(_corner_events(team))

        fks += len(_free_kick_events(team))

        # Shots
        shots = team[team['event'].isin(SHOT_EVENTS)]
        for _, row in shots.iterrows():
            is_sp = any(pd.notna(row.get(c)) for c in ['Set piece', 'Free kick', 'From corner']) or row.get('event') == 'Penalty' or row.get('type_id') == 9
            if is_sp:
                sp_shots += 1
                if row.get('event') == 'Goal':
                    sp_goals += 1
                if row.get('type_id') == 9:
                    penalties += 1
                sp_shot_coords.append({
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'event': row['event']
                })

    n = max(len(matches), 1)
    return {
        'corners_pg': round(corners / n, 1),
        'fks_pg': round(fks / n, 1),
        'penalties': penalties,
        'sp_shots_pg': round(sp_shots / n, 1),
        'sp_goals': sp_goals,
        'coords': sp_shot_coords
    }
# ════════════════════════════════════════════════════════════════
# PITCH / VISUAL HELPERS
# ════════════════════════════════════════════════════════════════

class MockPitch:
    def __init__(self, half=False, pitch_color='#0e1b0f', line_color='white', linewidth=1.4):
        self.half = half
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.linewidth = linewidth

    def draw(self, figsize=(10, 6.5)):
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.pitch_color)
        ax.set_facecolor(self.pitch_color)
        
        # 1. Draw grass stripes (same green shades as pre_match.py)
        if self.half:
            for i in range(5):
                x0 = 50 + i * 10
                ax.fill_between([x0, x0 + 10], 0, 100,
                                color='#18331b' if i % 2 == 0 else '#122615',
                                alpha=0.9, zorder=0)
        else:
            for i in range(10):
                x0 = i * 10
                ax.fill_between([x0, x0 + 10], 0, 100,
                                color='#18331b' if i % 2 == 0 else '#122615',
                                alpha=0.9, zorder=0)

        # 2. Draw lines
        lc = (1.0, 1.0, 1.0, 0.4) # Semi-transparent white
        lw = self.linewidth

        if self.half:
            # Half-pitch outline (x from 50 to 100, y from 0 to 100)
            ax.plot([50, 100, 100, 50, 50], [0, 0, 100, 100, 0], color=lc, linewidth=lw, zorder=3)
            # Penalty box
            ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, linewidth=lw, zorder=3)
            # Six yard box
            ax.plot([100, 94.2, 94.2, 100], [40.9, 40.9, 59.1, 59.1], color=lc, linewidth=lw - 0.3, zorder=3)
            # Halfway line arc
            from matplotlib.patches import Arc
            halfway_arc = Arc((50, 50), 18.3, 18.3, theta1=270, theta2=90, color=lc, linewidth=lw, zorder=3)
            ax.add_patch(halfway_arc)
            # Penalty spot
            ax.scatter([88.5], [50.0], color=lc, s=15, zorder=3)
            
            ax.set_xlim(48, 102)
            ax.set_ylim(-3, 103)
        else:
            # Full pitch outline (x from 0 to 100, y from 0 to 100)
            ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, linewidth=lw, zorder=3)
            # Midfield line
            ax.plot([50, 50], [0, 100], color=lc, linewidth=lw, zorder=3)
            # Center circle
            center_circle = plt.Circle((50, 50), 9.15, color=lc, fill=False, linewidth=lw, zorder=3)
            ax.add_patch(center_circle)
            # Center spot
            ax.scatter([50.0], [50.0], color=lc, s=15, zorder=3)
            
            # Left penalty area
            ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, linewidth=lw, zorder=3)
            # Left six-yard box
            ax.plot([0, 5.8, 5.8, 0], [40.9, 40.9, 59.1, 59.1], color=lc, linewidth=lw - 0.3, zorder=3)
            # Left penalty spot
            ax.scatter([11.5], [50.0], color=lc, s=15, zorder=3)

            # Right penalty area
            ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, linewidth=lw, zorder=3)
            # Right six-yard box
            ax.plot([100, 94.2, 94.2, 100], [40.9, 40.9, 59.1, 59.1], color=lc, linewidth=lw - 0.3, zorder=3)
            # Right penalty spot
            ax.scatter([88.5], [50.0], color=lc, s=15, zorder=3)
            
            ax.set_xlim(-2, 102)
            ax.set_ylim(-3, 103)

        ax.set_aspect('equal')
        ax.axis('off')
        return fig, ax

    def scatter(self, x, y, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        # Filter out any kwargs that matplotlib scatter doesn't support but mplsoccer might
        kwargs.pop('ax', None)
        return ax.scatter(x, y, **kwargs)

    def bin_statistic(self, x, y, statistic='count', bins=(6, 5)):
        nx, ny = bins
        xedges = np.linspace(0, 100, nx + 1)
        yedges = np.linspace(0, 100, ny + 1)
        counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        counts = counts.T
        
        x_grid, y_grid = np.meshgrid(xedges, yedges)
        cx, cy = np.meshgrid((xedges[:-1] + xedges[1:]) / 2.0, (yedges[:-1] + yedges[1:]) / 2.0)
        
        return {
            'statistic': counts,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'cx': cx,
            'cy': cy
        }

    def heatmap(self, stat, ax, cmap='YlOrRd', edgecolors='none', linewidth=0, alpha=1.0, **kwargs):
        mesh = ax.pcolormesh(stat['x_grid'], stat['y_grid'], stat['statistic'],
                             cmap=cmap, edgecolors=edgecolors, linewidth=linewidth, alpha=alpha, **kwargs)
        return mesh

    def label_heatmap(self, stat, color='white', fontsize=12, ax=None, ha='center', va='center', fontweight='bold', str_format='{:.0f}', **kwargs):
        if ax is None:
            ax = plt.gca()
        ny, nx = stat['statistic'].shape
        for i in range(ny):
            for j in range(nx):
                val = stat['statistic'][i, j]
                if val > 0:
                    text = str_format.format(val)
                    ax.text(stat['cx'][i, j], stat['cy'][i, j], text,
                            color=color, fontsize=fontsize, ha=ha, va=va, fontweight=fontweight, **kwargs)

def _make_pitch(half=False, figsize=(10, 6.5)):
    p = MockPitch(half=half, pitch_color=PITCH_BG)
    fig, ax = p.draw(figsize=figsize)
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

def _draw_pitch_title(ax, title, color=GOLD):
    ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=10)

def _draw_shot_map(matches, team_name):
    p, fig, ax = _make_pitch(half=True, figsize=(11, 7))
    shots = []
    for _, df in matches:
        for _, row in df[(df['team_name'] == team_name) & (df['event'].isin(SHOT_EVENTS))].iterrows():
            xy = _safe_xy(row)
            if not xy:
                continue
            shots.append({'x': xy[0], 'y': xy[1], 'event': row['event'], 'player': row.get('player_name', '?') or '?'})

    styles = {
        'Goal': (GOLD, '*', 140, 'Goal'),
        'Saved Shot': (GREEN, 'o', 64, 'Saved'),
        'Post': (PURPLE, 'D', 64, 'Post'),
        'Miss': ((1, 1, 1, 0.45), 'o', 48, 'Miss'),
    }
    for event, (color, marker, size, label) in styles.items():
        pts = [s for s in shots if s['event'] == event]
        if pts:
            p.scatter([s['x'] for s in pts], [s['y'] for s in pts], ax=ax,
                      color=color, s=size, marker=marker, alpha=0.9,
                      edgecolors='white', linewidths=0.5, label=label)
    _draw_pitch_title(ax, 'SHOT MAP')
    _legend(ax)
    return _fig_b64(fig), shots

def _draw_goal_sequences(matches, team_name):
    return _draw_goal_sequences_filtered(matches, team_name)

def _goal_origin_label(goal_row, chain):
    if _has_qualifier(goal_row, ['Penalty']) or goal_row.get('event') == 'Penalty':
        return 'penalty'
    if _has_qualifier(goal_row, ['From corner']):
        return 'corner'
    if _has_qualifier(goal_row, ['Free kick', 'Direct free']):
        return 'free_kick'
    if _has_qualifier(goal_row, ['Set piece']):
        return 'set_piece'
    if _has_qualifier(goal_row, ['Fast break']) or any(_has_qualifier(ev, ['Fast break']) for ev in chain):
        return 'fast_break'
    if any(_has_qualifier(ev, ['Cross']) for ev in chain):
        return 'cross'
    return 'open_play'

def _draw_goal_sequences_filtered(matches, team_name, origin_filter='all', max_seconds=10):
    goal_chains = []
    for fname, df in matches:
        match_info = _match_summary(fname, df, team_name)
        match_label = f"vs {match_info['opponent']} ({match_info['score']})"
        recs = _clean_records(df)
        for i, row in enumerate(recs):
            if row.get('team_name') != team_name or row.get('event') != 'Goal':
                continue
            t_goal = float(row.get('_abs_time', _event_seconds(row)))
            chain = []
            raw_chain = []
            for prev in recs[max(0, i - 18):i + 1]:
                if prev.get('team_name') != team_name:
                    continue
                if t_goal - float(prev.get('_abs_time', _event_seconds(prev))) > max_seconds:
                    continue
                raw_chain.append(prev)
                xy = _safe_xy(prev)
                if not xy:
                    continue
                chain.append({
                    'x': xy[0], 'y': xy[1], 'event': prev.get('event'),
                    'player': prev.get('player_name', '?') or '?',
                })
            if chain:
                origin = _goal_origin_label(row, raw_chain)
                if origin_filter not in (None, 'all') and origin != origin_filter:
                    continue
                goal_chains.append({'match': match_label, 'chain': chain[-8:], 'origin': origin})

    if not goal_chains:
        return None, []

    p, fig, ax = _make_pitch(figsize=(11, 7))
    colors = [GOLD, BLUE, GREEN, PURPLE]
    for idx, item in enumerate(goal_chains[:4]):
        chain = item['chain']
        color = colors[idx % len(colors)]
        for a, b in zip(chain, chain[1:]):
            ax.annotate('', xy=(b['x'], b['y']), xytext=(a['x'], a['y']),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.7, alpha=0.75),
                        zorder=5)
        p.scatter([c['x'] for c in chain[:-1]], [c['y'] for c in chain[:-1]], ax=ax,
                  color=color, s=32, alpha=0.85, edgecolors='white', linewidths=0.3,
                  label=None)
        goal = chain[-1]
        p.scatter([goal['x']], [goal['y']], ax=ax, color=GOLD, marker='*', s=150,
                  edgecolors='white', linewidths=0.7)
    _draw_pitch_title(ax, 'GOAL SEQUENCES')
    return _fig_b64(fig), goal_chains

def _estimate_def_block(team: pd.DataFrame, phase: str):
    def_ev = team[team['event'].isin(DEF_EVENTS)].dropna(subset=['x', 'y'])
    if phase == 'high':
        phase_df = def_ev[def_ev['x'] >= 60]
        fallback = (58, 36, 38)
    elif phase == 'mid':
        phase_df = def_ev[(def_ev['x'] >= 33.3) & (def_ev['x'] < 60)]
        fallback = (38, 24, 40)
    else:
        phase_df = def_ev[def_ev['x'] < 33.3]
        fallback = (16, 18, 36)
    if len(phase_df) < 4:
        return fallback
    min_x, max_x = np.percentile(phase_df['x'], [15, 85])
    min_y, max_y = np.percentile(phase_df['y'], [15, 85])
    line_height = round(float(min_x))
    team_length = round(float(max_x - min_x))
    team_width = round(float((max_y - min_y) * 0.68))
    return max(5, min(92, line_height)), max(10, min(55, team_length)), max(15, min(65, team_width))

def _draw_defensive_phase_blocks(matches, team_name):
    _, df = matches[0]
    team = df[df['team_name'] == team_name]
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), facecolor=PITCH_BG)
    panels = [
        ('High Block / Press', 'high', BLUE),
        ('Mid Block', 'mid', GOLD),
        ('Low Block', 'low', RED),
    ]
    for ax, (title, phase, color) in zip(axes, panels):
        p = MockPitch(pitch_color=PITCH_BG)
        _, panel_ax = p.draw(figsize=(4.6, 7.0))
        # Move the generated artists onto the target axis by redrawing directly.
        plt.close(panel_ax.figure)
        ax.set_facecolor(PITCH_BG)
        for i in range(10):
            ax.fill_between([0, 100], i * 10, i * 10 + 10,
                            color='#18331b' if i % 2 == 0 else '#122615', alpha=0.9, zorder=0)
        lc = (1.0, 1.0, 1.0, 0.42)
        ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=lc, linewidth=1.3)
        ax.plot([50, 50], [0, 100], color=lc, linewidth=1.3)
        ax.add_patch(plt.Circle((50, 50), 9.15, color=lc, fill=False, linewidth=1.3))
        ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=lc, linewidth=1.3)
        ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=lc, linewidth=1.3)
        lh, length, width = _estimate_def_block(team, phase)
        y_min = 50 - (width / 0.68) / 2
        y_max = 50 + (width / 0.68) / 2
        rect = plt.Rectangle((lh, y_min), length, y_max - y_min,
                             facecolor='#8fb0ff', alpha=0.34,
                             edgecolor='#8fb0ff', linewidth=1.8, zorder=4)
        ax.add_patch(rect)
        ax.text(lh + length / 2, y_max + 5, f'{width}m', color='white', fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#1f2937', edgecolor='none'), zorder=5)
        ax.text(lh - 4, 50, f'{length}m', color='white', fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#1f2937', edgecolor='none'), zorder=5)
        ax.text(lh / 2, 10, f'{lh}m', color='white', fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#1f2937', edgecolor='none'), zorder=5)
        ax.text(96, 50, 'DIRECTION', color=(1, 1, 1, 0.35), fontsize=8,
                rotation=90, ha='center', va='center', fontweight='bold')
        ax.set_xlim(-2, 102)
        ax.set_ylim(-3, 103)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title.upper(), color=color, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return _fig_b64(fig)

def _transition_filter_outcome(mode, transition_filter):
    if transition_filter in (None, 'all'):
        return None
    if mode == 'win':
        return {
            'goals': 'goal',
            'shots': 'shot',
            'f3': 'f3',
            'negative': 'lost',
            'safe': 'retained',
        }.get(transition_filter)
    return {
        'goals': 'opp_goal',
        'shots': 'opp_shot',
        'f3': 'opp_f3',
        'negative': 'recovered',
        'safe': 'survived',
    }.get(transition_filter)


def _draw_transition_after_map(tr_data, mode='win', transition_filter='all'):
    base_color = GREEN if mode == 'win' else RED
    after_color = GOLD if mode == 'win' else ORANGE
    outcome_colors = {
        'goal': GOLD, 'shot': BLUE, 'f3': PURPLE, 'lost': RED, 'retained': GREEN,
        'opp_goal': GOLD, 'opp_shot': RED, 'opp_f3': ORANGE, 'recovered': GREEN, 'survived': BLUE,
    }
    filter_outcome = _transition_filter_outcome(mode, transition_filter)
    coords = [
        c for c in tr_data['coords']
        if filter_outcome is None or c.get('outcome') == filter_outcome
    ]
    after = [
        a for a in tr_data.get('after', [])
        if filter_outcome is None or a.get('outcome') == filter_outcome
    ]
    p, fig, ax = _make_pitch(figsize=(11, 7))
    for c in coords:
        p.scatter([c['x']], [c['y']], ax=ax,
                  color=outcome_colors.get(c.get('outcome'), base_color),
                  s=34, alpha=0.72, edgecolors='white', linewidths=0.25)
    for a in after[:90]:
        ax.annotate('', xy=(a['x1'], a['y1']), xytext=(a['x0'], a['y0']),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=outcome_colors.get(a.get('outcome'), after_color),
                        lw=1.6,
                        alpha=0.55,
                    ),
                    zorder=4)
    if mode == 'win':
        legend_items = [
            ('goal', 'Goal'), ('shot', 'Shot'), ('f3', 'F3 entry'),
            ('lost', 'Lost'), ('retained', 'Retained'),
        ]
    else:
        legend_items = [
            ('opp_goal', 'Opp goal'), ('opp_shot', 'Opp shot'), ('opp_f3', 'Opp F3'),
            ('recovered', 'Recovered'), ('survived', 'Survived'),
        ]
    for key, label in legend_items:
        ax.scatter([], [], color=outcome_colors[key], s=36, label=label)
    _draw_pitch_title(ax, 'NEXT 10 SECONDS AFTER RECOVERY' if mode == 'win' else 'NEXT 10 SECONDS AFTER LOSS')
    if filter_outcome is not None and not coords:
        ax.text(50, 50, 'NO EVENTS FOR SELECTED FILTER',
                ha='center', va='center', color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='#111827',
                          edgecolor='none', alpha=0.85))
    _legend(ax)
    return _fig_b64(fig)


def _draw_vulnerability_map(vul):
    p, fig, ax = _make_pitch(half=True, figsize=(10.5, 7))

    zones = [
        ('Left', 0, 33.3, BLUE),
        ('Center', 33.3, 66.6, GOLD),
        ('Right', 66.6, 100, PURPLE),
    ]
    raw_counts = vul.get('entries_raw', {})
    for name, y0, y1, color in zones:
        ax.fill_between([50, 100], y0, y1, color=color, alpha=0.16, zorder=1)
        ax.text(54, (y0 + y1) / 2, f"{raw_counts.get(name, 0)}",
                color='white', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#1f2937',
                          edgecolor='none', alpha=0.85),
                zorder=8)

    z14_rect = plt.Rectangle((66, 33.3), 17, 33.3, linewidth=1.8,
                             edgecolor=RED, facecolor=RED, alpha=0.18, zorder=2)
    ax.add_patch(z14_rect)
    ax.text(74.5, 50, f"Z14\n{vul.get('z14_pg', 0)} / game",
            color='white', fontsize=9, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#991b1b',
                      edgecolor='none', alpha=0.85),
            zorder=9)

    zone_colors = {'Left': BLUE, 'Center': GOLD, 'Right': PURPLE}
    for entry in vul.get('entries_coords', [])[:55]:
        color = zone_colors.get(entry.get('zone'), GOLD)
        ax.annotate('', xy=(entry['end_x'], entry['end_y']), xytext=(entry['x'], entry['y']),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.75),
                    zorder=5)
        ax.scatter([entry['end_x']], [entry['end_y']], color=color, s=30,
                   edgecolors='white', linewidths=0.45, zorder=7)

    _draw_pitch_title(ax, 'DEFENSIVE VULNERABILITY MAP', RED)
    return _fig_b64(fig)

def _outcome_bar(label, count, total, color):
    pct = _safe_pct(count, total)
    return html.Div(style={'marginBottom': '9px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '3px'}, children=[
            html.Span(label, style={'fontSize': '0.77rem', 'color': 'var(--text-secondary)'}),
            html.Span(f'{count}  ·  {pct:.0f}%', style={'fontSize': '0.77rem', 'fontWeight': '700', 'color': color}),
        ]),
        html.Div(style={'height': '5px', 'background': 'rgba(255,255,255,0.07)', 'borderRadius': '3px'}, children=[
            html.Div(style={'width': f'{pct}%', 'height': '100%', 'background': color, 'borderRadius': '3px'}),
        ]),
    ])

def _compute_set_piece_details(matches, team_name):
    data = {k: [] for k in ['corners', 'free_kicks', 'goal_kicks', 'penalties', 'shots']}
    for _, df in matches:
        team = df[df['team_name'] == team_name]
        penalty_times = []
        for _, row in _corner_events(team).iterrows():
            start = _safe_xy(row)
            end = _safe_xy(row, 'Pass End X', 'Pass End Y')
            data['corners'].append({
                'x': start[0] if start else None, 'y': start[1] if start else None,
                'ex': end[0] if end else None, 'ey': end[1] if end else None,
                'event': row.get('event'), 'player': row.get('player_name', '?') or '?',
                'outcome': row.get('outcome'),
            })

        for _, row in _free_kick_events(team).iterrows():
            start = _safe_xy(row)
            end = _safe_xy(row, 'Pass End X', 'Pass End Y')
            data['free_kicks'].append({
                'x': start[0] if start else None, 'y': start[1] if start else None,
                'ex': end[0] if end else None, 'ey': end[1] if end else None,
                'event': row.get('event'), 'player': row.get('player_name', '?') or '?',
                'outcome': row.get('outcome'),
            })

        for _, row in team.iterrows():
            event = row.get('event')
            type_id = row.get('type_id')
            start = _safe_xy(row)
            end = _safe_xy(row, 'Pass End X', 'Pass End Y')
            rec = {
                'x': start[0] if start else None, 'y': start[1] if start else None,
                'ex': end[0] if end else None, 'ey': end[1] if end else None,
                'event': event, 'player': row.get('player_name', '?') or '?',
                'outcome': row.get('outcome'),
                'goal_mouth_y': row.get(GOAL_MOUTH_Y_COL),
                'goal_mouth_z': row.get(GOAL_MOUTH_Z_COL),
            }
            if _has_qualifier(row, ['Goal Kick']) or type_id == 124:
                data['goal_kicks'].append(rec)
            event_l = str(event).lower()
            is_penalty_incident = (
                (_has_qualifier(row, ['Penalty']) or event == 'Penalty')
                and (
                    event == 'Penalty'
                    or event in SHOT_EVENTS
                    or event in {'Save', 'Penalty faced'}
                    or 'penalty' in event_l
                )
            )
            if is_penalty_incident:
                period = int(row.get('period_id') or 0)
                minute = float(row.get('time_min') or 0)
                duplicate = any(p == period and abs(minute - m) <= 5 for p, m in penalty_times)
                if duplicate:
                    continue
                penalty_times.append((period, minute))
                rec['x'], rec['y'] = 88.5, 50.0
                data['penalties'].append(rec)
            if event in SHOT_EVENTS and _has_qualifier(row, ['Set piece', 'From corner', 'Free kick', 'Penalty']):
                data['shots'].append(rec)
    return data

def _draw_set_piece_map(records, title, color=GOLD, half=False, penalties=False):
    p, fig, ax = _make_pitch(half=half, figsize=(11, 7))
    completed = missed = 0
    for rec in records:
        if rec['x'] is None or rec['y'] is None:
            continue
        if penalties:
            marker = '*' if rec['event'] == 'Goal' else 'o'
            p.scatter([rec['x']], [rec['y']], ax=ax, color=color, marker=marker,
                      s=120 if marker == '*' else 58, alpha=0.9, edgecolors='white', linewidths=0.6)
            continue
        if rec['ex'] is not None and rec['ey'] is not None:
            is_complete = int(rec.get('outcome') or 0) == 1
            completed += int(is_complete)
            missed += int(not is_complete)
            arrow_color = color if is_complete else (1, 1, 1, 0.35)
            ax.annotate('', xy=(rec['ex'], rec['ey']), xytext=(rec['x'], rec['y']),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, alpha=0.75),
                        zorder=5)
        p.scatter([rec['x']], [rec['y']], ax=ax, color=color, s=34, alpha=0.9, edgecolors='white', linewidths=0.4)
    _draw_pitch_title(ax, title, color)
    if not records:
        ax.text(50, 50, 'No events recorded', color='white', alpha=0.65, fontsize=12,
                ha='center', va='center', fontweight='bold')
    if not penalties:
        ax.scatter([], [], color=color, s=30, label=f'Completed {completed}')
        ax.scatter([], [], color=(1, 1, 1, 0.45), s=30, label=f'Incomplete {missed}')
        _legend(ax)
    return _fig_b64(fig)


def _goal_mouth_xy(rec):
    mouth_y = _safe_float(rec.get('goal_mouth_y'))
    mouth_z = _safe_float(rec.get('goal_mouth_z'))
    if mouth_y is None:
        return None
    x = (mouth_y - GOAL_MOUTH_LEFT_OPT) / (GOAL_MOUTH_RIGHT_OPT - GOAL_MOUTH_LEFT_OPT)
    y = (mouth_z or 0.0) / 38.0
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


def _draw_set_piece_goal_mouth(records, title='PENALTY GOAL MOUTH'):
    placements = [rec for rec in records if _goal_mouth_xy(rec) is not None]
    if not placements:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=PITCH_BG)
    ax.set_facecolor('#101913')

    ax.plot([0, 1], [0, 0], color='white', alpha=0.92, linewidth=3)
    ax.plot([0, 0], [0, 1], color='white', alpha=0.92, linewidth=3)
    ax.plot([1, 1], [0, 1], color='white', alpha=0.92, linewidth=3)
    ax.plot([0, 1], [1, 1], color='white', alpha=0.92, linewidth=3)
    for x in np.linspace(0.2, 0.8, 4):
        ax.plot([x, x], [0, 1], color='white', alpha=0.12, linewidth=1)
    for y in np.linspace(0.25, 0.75, 3):
        ax.plot([0, 1], [y, y], color='white', alpha=0.12, linewidth=1)
    ax.axvline(0.5, color=GOLD, alpha=0.25, linestyle='--', linewidth=1)

    labels = {'Goal': 0, 'Saved Shot': 0, 'Miss / Post': 0}
    for rec in placements:
        x, y = _goal_mouth_xy(rec)
        event = rec.get('event')
        if event == 'Goal':
            color, marker, label = GREEN, 'o', 'Goal'
        elif event == 'Saved Shot':
            color, marker, label = BLUE, 's', 'Saved Shot'
        else:
            color, marker, label = RED, 'X', 'Miss / Post'
        labels[label] = labels.get(label, 0) + 1
        ax.scatter([x], [y], s=210, color=color, marker=marker,
                   edgecolors='white', linewidths=1.1, alpha=0.95, zorder=5,
                   label=label)

    ax.text(0.17, -0.08, 'LEFT', color=(1, 1, 1, 0.55), fontsize=8,
            fontweight='bold', ha='center')
    ax.text(0.50, -0.08, 'CENTRE', color=(1, 1, 1, 0.55), fontsize=8,
            fontweight='bold', ha='center')
    ax.text(0.83, -0.08, 'RIGHT', color=(1, 1, 1, 0.55), fontsize=8,
            fontweight='bold', ha='center')

    ax.set_title(title, color=GOLD, fontsize=12, fontweight='bold', pad=10)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    if unique:
        legend = ax.legend(unique.values(), unique.keys(), loc='upper center',
                           bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False,
                           fontsize=8)
        for text in legend.get_texts():
            text.set_color('white')
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.18, 1.22)
    ax.axis('off')
    return _fig_b64(fig)

def _mode_value(values, default=None):
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return default
    return pd.Series(vals).mode().iloc[0]

def _format_formation(value):
    if value is None or pd.isna(value):
        return 'Unknown'
    raw = str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
    raw = raw.replace('-', '').strip()
    if len(raw) >= 3 and raw.isdigit():
        return '-'.join(raw)
    return raw or 'Unknown'

def _slot_coordinates(formation):
    formation = _format_formation(formation).replace('-', '')
    if formation == '4231':
        return {
            1: (8, 50), 2: (24, 82), 3: (24, 18), 4: (43, 58), 5: (24, 60), 6: (24, 40),
            7: (66, 82), 8: (43, 42), 9: (86, 50), 10: (66, 50), 11: (66, 18),
        }
    if formation == '433':
        return {
            1: (8, 50), 2: (24, 82), 3: (24, 18), 4: (48, 50), 5: (24, 60), 6: (24, 40),
            7: (78, 82), 8: (54, 66), 9: (86, 50), 10: (54, 34), 11: (78, 18),
        }
    if formation == '4141':
        return {
            1: (8, 50), 2: (24, 82), 3: (24, 18), 4: (42, 50), 5: (24, 60), 6: (24, 40),
            7: (66, 82), 8: (66, 58), 9: (86, 50), 10: (66, 42), 11: (66, 18),
        }
    if formation == '442':
        return {
            1: (8, 50), 2: (24, 82), 3: (24, 18), 4: (48, 58), 5: (24, 60), 6: (24, 40),
            7: (58, 82), 8: (48, 42), 9: (82, 58), 10: (82, 42), 11: (58, 18),
        }
    return {
        1: (8, 50), 2: (24, 82), 3: (24, 18), 4: (46, 58), 5: (24, 60), 6: (24, 40),
        7: (66, 82), 8: (46, 42), 9: (86, 50), 10: (66, 50), 11: (66, 18),
    }

def _position_coordinate(position, used_count):
    base = {
        'GK': (8, 50), 'RB': (24, 82), 'LB': (24, 18), 'CB': (24, 50),
        'RWB': (34, 86), 'LWB': (34, 14), 'CDM': (43, 50), 'CM': (52, 50),
        'CAM': (66, 50), 'RW': (70, 82), 'LW': (70, 18), 'CF': (86, 50), 'ST': (86, 50),
    }
    x, y = base.get(str(position), (52, 50))
    offset = ((used_count % 3) - 1) * 10
    if str(position) in {'CB', 'CDM', 'CM', 'CF', 'ST'}:
        y += offset
    return x, max(12, min(88, y))

def _compute_probable_xi(matches, team_name):
    players = {}
    formations = []
    n_matches = len(matches)

    for match_idx, (_, df) in enumerate(matches):
        team = df[df['team_name'] == team_name].dropna(subset=['player_name'])
        if team.empty:
            continue
        if 'formation' in team.columns:
            formations.extend(team['formation'].dropna().tolist())

        match_counts = team['player_name'].value_counts()
        match_players = match_counts.index.tolist()
        gk_rows = team[team.get('position', pd.Series(dtype=str)) == 'GK']
        if not gk_rows.empty:
            gk = gk_rows['player_name'].value_counts().index[0]
            ordered = [gk] + [p for p in match_players if p != gk]
        else:
            ordered = match_players

        likely_xi = ordered[:11]
        for name in likely_xi:
            p_df = team[team['player_name'] == name]
            info = players.setdefault(name, {
                'name': name, 'apps': 0, 'actions': 0, 'recent_bonus': 0,
                'positions': [], 'slots': [], 'jerseys': [],
            })
            info['apps'] += 1
            info['actions'] += int(match_counts.get(name, 0))
            info['recent_bonus'] += max(0, n_matches - match_idx)
            if 'position' in p_df.columns:
                info['positions'].extend(p_df['position'].dropna().tolist())
            if 'Team Player Formation' in p_df.columns:
                info['slots'].extend(p_df['Team Player Formation'].dropna().tolist())
            if 'Jersey Number' in p_df.columns:
                info['jerseys'].extend(p_df['Jersey Number'].dropna().tolist())

    formation = _mode_value(formations, None)
    coords = _slot_coordinates(formation)

    candidates = []
    for info in players.values():
        slot = _mode_value(info['slots'], None)
        try:
            slot = int(slot) if slot is not None else None
        except Exception:
            slot = None
        jersey = _mode_value(info['jerseys'], None)
        try:
            jersey = int(jersey) if jersey is not None else None
        except Exception:
            jersey = None
        position = _mode_value(info['positions'], 'UNK')
        score = info['apps'] * 1000 + info['recent_bonus'] * 25 + info['actions']
        candidates.append({**info, 'slot': slot, 'jersey': jersey, 'position': position, 'score': score})

    selected = []
    used_names = set()
    for slot in range(1, 12):
        slot_candidates = [p for p in candidates if p['slot'] == slot and p['name'] not in used_names]
        if not slot_candidates:
            continue
        pick = sorted(slot_candidates, key=lambda p: p['score'], reverse=True)[0]
        selected.append(pick)
        used_names.add(pick['name'])

    for pick in sorted(candidates, key=lambda p: p['score'], reverse=True):
        if len(selected) >= 11:
            break
        if pick['name'] not in used_names:
            selected.append(pick)
            used_names.add(pick['name'])

    used_pos_counts = {}
    for i, player in enumerate(selected[:11]):
        if player['slot'] in coords:
            x, y = coords[player['slot']]
        else:
            count = used_pos_counts.get(player['position'], 0)
            x, y = _position_coordinate(player['position'], count)
            used_pos_counts[player['position']] = count + 1
        player['x'] = x
        player['y'] = y
        player['confidence'] = _safe_pct(player['apps'], max(n_matches, 1))

    selected = sorted(selected[:11], key=lambda p: (p.get('x', 50), p.get('y', 50)))
    return {
        'formation': _format_formation(formation),
        'players': selected,
        'sample_matches': n_matches,
    }

def _short_player_name(name):
    parts = str(name).split()
    if len(parts) <= 1:
        return str(name)
    return f"{parts[0][0]}. {parts[-1]}"

def _draw_probable_xi(xi_data, rival_label):
    p, fig, ax = _make_pitch(figsize=(11, 7))
    players = xi_data['players']
    for player in players:
        x, y = player['x'], player['y']
        ax.scatter([x], [y], s=520, color='#111827', edgecolors=GOLD, linewidths=2.0, zorder=5)
        label = str(player['jersey']) if player['jersey'] is not None else str(player.get('position', ''))
        ax.text(x, y + 0.8, label, color='white', fontsize=9, fontweight='bold',
                ha='center', va='center', zorder=6)
        ax.text(x, y - 5.3, _short_player_name(player['name']), color='white', fontsize=7.5,
                ha='center', va='top', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=(0, 0, 0, 0.45), edgecolor='none'))
        ax.text(x, y + 6.0, str(player.get('position', '')), color=GOLD, fontsize=6.5,
                ha='center', va='bottom', fontweight='bold', zorder=6)
    _draw_pitch_title(ax, f'{rival_label.upper()} PROJECTED XI vs GÖZTEPE')
    return _fig_b64(fig)

def _build_projected_xi(matches, team_name, rival_label):
    try:
        xi = _compute_probable_xi(matches, team_name)
        if not xi['players']:
            return html.Div()
        img = _draw_probable_xi(xi, rival_label)
        rows = []
        for p_info in xi['players']:
            rows.append(html.Div(style={
                'display': 'flex', 'justifyContent': 'space-between',
                'padding': '5px 0', 'borderBottom': '1px solid rgba(255,255,255,0.05)',
            }, children=[
                html.Span(f"{p_info.get('position', 'UNK')} · {_short_player_name(p_info['name'])}", style={
                    'fontSize': '0.74rem', 'color': 'var(--text-secondary)',
                }),
                html.Span(f"{p_info['confidence']}%", style={
                    'fontSize': '0.74rem', 'fontWeight': '700', 'color': GOLD,
                }),
            ]))
        return _card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Formation', xi['formation'], GOLD),
                        _pill('Sample', f"{xi['sample_matches']} matches", BLUE),
                        _pill('Purpose', 'vs Göztepe', GREEN),
                    ]),
                    html.Div('PROBABLE STARTERS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    html.Div(rows),
                    html.Div('Confidence = selected in recent sampled XIs, not confirmed team news.', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'marginTop': '8px', 'lineHeight': '1.4',
                    }),
                ], md=4),
                dbc.Col([
                    html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                ], md=8),
            ]),
            title=f'{rival_label} — Projected XI Against Göztepe', icon='👥'
        )
    except Exception:
        return html.Div()


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
    res_colors = {'W': GREEN, 'D': BLUE, 'L': RED}
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


def _build_defensive(matches, team_name, rival_label):
    sections = []

    # ── 0. Defensive Phase Blocks ─────────────────────────────
    try:
        block_img = _draw_defensive_phase_blocks(matches, team_name)
        df_m = _compute_defensive(matches, team_name)
        sections.append(_card(
            html.Div(children=[
                html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                    _pill('Avg Def. Line', df_m['avg_line'], BLUE),
                    _pill('PPDA', df_m['ppda'], PURPLE),
                    _pill('Block Type', df_m['press_label'], GOLD),
                ]),
                html.Img(src=block_img, style={'width': '100%', 'borderRadius': '8px'}),
                html.Div('Estimated from defensive actions in high, mid, and low zones for the selected match.', style={
                    'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                    'textAlign': 'center', 'marginTop': '6px',
                }),
            ]),
            title=f'{rival_label} — Defensive Phase Blocks', icon='🛡'
        ))
    except Exception:
        pass

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

    # ── 2. Vulnerable Flanks & Offside ────────────────────────
    try:
        vul = _compute_vulnerability(matches, team_name)
        vul_img = _draw_vulnerability_map(vul)
        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('F3 Entries / Game', vul['f3_pg'], RED),
                        _pill('Z14 Passes / Game', vul['z14_pg'], GOLD),
                        _pill('Offside Traps / Game', vul['offside_pg'], GREEN),
                    ]),
                    html.Div('F3 ENTRIES CONCEDED BY FLANK', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _bar('Left Channel', vul['f3_conceded']['Left'],   BLUE),
                    _bar('Central',      vul['f3_conceded']['Center'], GOLD),
                    _bar('Right Channel',vul['f3_conceded']['Right'],  PURPLE),
                    html.Div(style={
                        'marginTop': '12px', 'fontSize': '0.68rem',
                        'color': 'var(--text-secondary)', 'lineHeight': '1.5',
                    }, children=[
                        html.P('Arrows show opponent passes entering the final third.'),
                        html.P('Red central box marks Zone 14 passes conceded.'),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Img(src=vul_img, style={'width': '100%', 'maxHeight': '470px', 'objectFit': 'contain', 'borderRadius': '8px'}),
                    html.Div('Blue = left channel · Yellow = central · Purple = right channel', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=8),
            ]),
            title='Defensive Vulnerability Map', icon='⚠'
        ))
    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No defensive data available.', className='goz-card-desc')
    ])


def _build_off_transitions(matches, team_name, rival_label, transition_filter='all'):
    sections = []
    try:
        tr = _compute_transitions(matches, team_name)

        # ── Ball Wins (Attacking Transitions) ─────────────────
        img_w = _draw_transition_after_map(tr['win'], mode='win', transition_filter=transition_filter)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Ball Wins', tr['win']['total'], GREEN),
                        _pill('10s → F3 %',       f"{tr['win']['outcomes']['f3']}%",   GOLD),
                        _pill('10s → Shot %',     f"{tr['win']['outcomes']['shot']}%", BLUE),
                        _pill('10s → Goal %',     f"{tr['win']['outcomes']['goal']}%", GREEN),
                        _pill('10s → Lost %',     f"{tr['win']['outcomes']['lost']}%", RED),
                    ]),
                    html.Div('WHAT HAPPENED AFTER 10 SECONDS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _outcome_bar('Goal', tr['win']['outcome_counts']['goal'], tr['win']['total'], GOLD),
                    _outcome_bar('Shot', tr['win']['outcome_counts']['shot'], tr['win']['total'], BLUE),
                    _outcome_bar('Final third entry', tr['win']['outcome_counts']['f3'], tr['win']['total'], PURPLE),
                    _outcome_bar('Lost again', tr['win']['outcome_counts']['lost'], tr['win']['total'], RED),
                    _outcome_bar('Retained / no danger', tr['win']['outcome_counts']['retained'], tr['win']['total'], GREEN),
                    html.Div('RECOVERY ZONE', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': GOLD,
                        'letterSpacing': '1px', 'marginBottom': '8px', 'marginTop': '14px',
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
                    html.Div('● Recovery point · arrows show the next actions within 10 seconds', style={
                        'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                        'textAlign': 'center', 'marginTop': '4px',
                    }),
                ], md=7),
            ]),
            title=f'{rival_label} — Attacking Transitions (Ball Wins)', icon='⚡'
        ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No transition data available.', className='goz-card-desc')
    ])


def _build_def_transitions(matches, team_name, rival_label, transition_filter='all'):
    sections = []
    try:
        tr = _compute_transitions(matches, team_name)

        # ── Ball Losses (Defensive Transitions) ───────────────
        img_l = _draw_transition_after_map(tr['loss'], mode='loss', transition_filter=transition_filter)

        sections.append(_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                        _pill('Ball Losses', tr['loss']['total'],                              RED),
                        _pill('10s → Opp F3 %',     f"{tr['loss']['outcomes']['opp_f3']}%",    ORANGE),
                        _pill('10s → Opp Shot %',   f"{tr['loss']['outcomes']['opp_shot']}%",  RED),
                        _pill('10s → Opp Goal %',   f"{tr['loss']['outcomes']['opp_goal']}%",  GOLD),
                        _pill('10s → Recovered %',  f"{tr['loss']['outcomes']['recovered']}%", GREEN),
                    ]),
                    html.Div('WHAT HAPPENED AFTER 10 SECONDS', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                        'letterSpacing': '1px', 'marginBottom': '8px',
                    }),
                    _outcome_bar('Opponent goal', tr['loss']['outcome_counts']['opp_goal'], tr['loss']['total'], GOLD),
                    _outcome_bar('Opponent shot', tr['loss']['outcome_counts']['opp_shot'], tr['loss']['total'], RED),
                    _outcome_bar('Opponent final third entry', tr['loss']['outcome_counts']['opp_f3'], tr['loss']['total'], ORANGE),
                    _outcome_bar('Recovered back', tr['loss']['outcome_counts']['recovered'], tr['loss']['total'], GREEN),
                    _outcome_bar('Survived / no danger', tr['loss']['outcome_counts']['survived'], tr['loss']['total'], BLUE),
                    html.Div('BALL LOSS ZONE', style={
                        'fontSize': '0.7rem', 'fontWeight': '700', 'color': RED,
                        'letterSpacing': '1px', 'marginBottom': '8px', 'marginTop': '14px',
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
                    html.Div('● Loss point · arrows show opponent actions within 10 seconds', style={
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


def _build_set_pieces(matches, team_name, rival_label):
    sections = []
    try:
        details = _compute_set_piece_details(matches, team_name)

        set_piece_cards = [
            ('Corners', 'corners', GOLD, True, False),
            ('Dangerous Free Kicks', 'free_kicks', BLUE, True, False),
            ('Goal Kicks', 'goal_kicks', GREEN, False, False),
            ('Penalties', 'penalties', RED, True, True),
        ]
        maps = []
        for title, key, color, half, penalties in set_piece_cards:
            img = _draw_set_piece_map(details[key], title.upper(), color=color, half=half, penalties=penalties)
            maps.append(dbc.Col([
                html.Img(src=img, style={'width': '100%', 'borderRadius': '8px'}),
                html.Div(f"{len(details[key])} {title.lower()} recorded", style={
                    'fontSize': '0.65rem', 'color': 'var(--text-secondary)',
                    'textAlign': 'center', 'marginTop': '4px',
                }),
            ], md=6, style={'marginBottom': '14px'}))

        sections.append(_card(
            html.Div(children=[
                html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                    _pill('Corners', len(details['corners']), GOLD),
                    _pill('Dangerous FKs', len(details['free_kicks']), BLUE),
                    _pill('Goal Kicks', len(details['goal_kicks']), GREEN),
                    _pill('Penalties', len(details['penalties']), RED),
                ]),
                dbc.Row(maps),
            ]),
            title='Set Piece Delivery Maps', icon='📍'
        ))

        penalty_goal_mouth = _draw_set_piece_goal_mouth(details['penalties'])
        penalties_with_placement = [
            p for p in details['penalties']
            if _goal_mouth_xy(p) is not None
        ]
        penalty_goals = len([p for p in details['penalties'] if p.get('event') == 'Goal'])
        penalty_saved = len([p for p in details['penalties'] if p.get('event') == 'Saved Shot'])
        penalty_missed = len([p for p in details['penalties'] if p.get('event') in {'Miss', 'Post'}])
        if penalty_goal_mouth:
            sections.append(_card(
                dbc.Row([
                    dbc.Col([
                        html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '14px'}, children=[
                            _pill('Penalties', len(details['penalties']), RED),
                            _pill('With Placement', len(penalties_with_placement), GOLD),
                            _pill('Scored', penalty_goals, GREEN),
                            _pill('Saved', penalty_saved, BLUE),
                            _pill('Miss/Post', penalty_missed, RED),
                        ]),
                        html.Div('Shows only penalty placement in the goal mouth when Opta goal-mouth coordinates exist.',
                                 style={'fontSize': '0.72rem', 'color': 'var(--text-secondary)', 'lineHeight': '1.5'}),
                    ], md=4),
                    dbc.Col([
                        html.Img(src=penalty_goal_mouth, style={'width': '100%', 'maxHeight': '420px', 'objectFit': 'contain', 'borderRadius': '8px'}),
                    ], md=8),
                ]),
                title='Penalty Goal Mouth', icon='🥅'
            ))
    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className='goz-form-section', children=[
        html.Div('No set piece data available.', className='goz-card-desc')
    ])


# ════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════

def layout():
    rivals = _discover_rivals()
    default_value = next(iter(rivals.values()), None)
    rival_options = [{'label': label, 'value': value} for label, value in rivals.items()]
    tab_options   = [
        {'label': '👥  Projected XI',     'value': 'projected-xi'},
        {'label': '⚔️  Offensive',        'value': 'off'},
        {'label': '🛡  Defensive',         'value': 'def'},
        {'label': '⚡  Off. Transitions',  'value': 'off-trans'},
        {'label': '🔄  Def. Transitions',  'value': 'def-trans'},
        {'label': '🎯  Set Pieces',        'value': 'set-pieces'},
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
                dbc.RadioItems(
                    id='scout-match-selector',
                    options=[],
                    value=None,
                    inline=True,
                    className='pm-tab-radio-group',
                    inputClassName='pm-tab-radio-input',
                    labelClassName='pm-tab-radio-label',
                    style={'gap': '10px'},
                ),
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
            html.Div(id='scout-transition-filter-wrap', style={'display': 'none'}, children=[
                dbc.RadioItems(
                    id='scout-transition-filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Goals', 'value': 'goals'},
                        {'label': 'Shots', 'value': 'shots'},
                        {'label': 'F3', 'value': 'f3'},
                        {'label': 'Lost / Recovered', 'value': 'negative'},
                        {'label': 'Retained / Survived', 'value': 'safe'},
                    ],
                    value='all',
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
    [Output('scout-match-selector', 'options'),
     Output('scout-match-selector', 'value')],
    Input('scout-rival-selector', 'value'),
)
def update_match_options(rival_label):
    """Populate the match selector whenever the rival changes."""
    team_name = rival_label or ''
    if not team_name:
        return [], None

    matches = _load_rival_matches(team_name)
    res_colors = {'W': GREEN, 'D': BLUE, 'L': RED}

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

    return options, default


@callback(
    Output('scout-transition-filter-wrap', 'style'),
    Input('scout-tab', 'value'),
)
def toggle_scout_transition_filter(active_tab):
    if active_tab in ('off-trans', 'def-trans'):
        return {'display': 'flex', 'justifyContent': 'center', 'margin': '0 0 18px'}
    return {'display': 'none'}


@callback(
    [Output('scout-match-card',    'children'),
     Output('scout-tab-content',   'children')],
    [Input('scout-rival-selector', 'value'),
     Input('scout-match-selector', 'value'),
     Input('scout-tab',            'value'),
     Input('scout-transition-filter', 'value')],
)
def update_scout_content(rival_label, selected_file, active_tab, transition_filter):
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

    if   active_tab == 'projected-xi': content = _build_projected_xi(all_matches, team_name, rival_display)
    elif active_tab == 'off':        content = _build_offensive(single, team_name, rival_display)
    elif active_tab == 'def':        content = _build_defensive(single, team_name, rival_display)
    elif active_tab == 'off-trans':  content = _build_off_transitions(single, team_name, rival_display, transition_filter)
    elif active_tab == 'def-trans':  content = _build_def_transitions(single, team_name, rival_display, transition_filter)
    elif active_tab == 'set-pieces': content = _build_set_pieces(single, team_name, rival_display)
    else:                            content = html.Div()

    return match_card, content

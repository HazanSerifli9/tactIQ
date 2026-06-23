from shared.matplotlib_config import configure_matplotlib

configure_matplotlib()

import threading
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch
import matplotlib.patheffects as path_effects
import os
import numpy as np
import io
import base64
from PIL import Image
from typing import Optional, List
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Optional import for Highlight Text
try:
    import highlight_text as htext
except ImportError:
    htext = None

from utils.data import get_data_dir, TEAM_LOGOS
from utils.cache import disk_cache

# --- Theme Configuration ---
TACTIQ_BG = '#313332'
TACTIQ_TEXT = 'white'
TACTIQ_ACCENT = '#fbbf24' # Gold
TACTIQ_COLORS = ["#313332","#47516B", "#848178", "#B2A66F", "#fbbf24"]

CustomCmap = mpl.colors.LinearSegmentedColormap.from_list("", TACTIQ_COLORS)

# Common Action Colors
COLOR_SHOT = "khaki"
COLOR_CROSS = "mediumpurple"
COLOR_FWD = "palegreen"
COLOR_BACK = "lightsalmon"
COLOR_SIDE = "#6a6a6a"

STYLE_COLORS = {
    "Control-ball":            "#22c55e",
    "Trigger-happy Control":   "#84cc16",
    "High Press & Vertical":   "#f97316",
    "Mid-block & Counter":     "#60a5fa",
    "Low Block & Play Out":    "#a78bfa",
    "Chaos-ball":              "#f43f5e",
    "Mixed":                   "#9ca3af",
}

CommonActionCmap = mpl.colors.ListedColormap([COLOR_SHOT, COLOR_CROSS, COLOR_FWD, COLOR_BACK, COLOR_SIDE])

def get_team_color(team_name: str) -> str:
    # return a constant red color for all teams
    return "#e63946"

# --- Data Caching ---
_EVENTS_CACHE: Optional[pd.DataFrame] = None
_EVENTS_CACHE_SIG = None
_EVENTS_CACHE_LOCK = threading.Lock()

def load_all_events() -> pd.DataFrame:
    """
    Load all event parquet files from the data directory into a single DataFrame.
    Automatically reloads when new files are added or removed.
    """
    global _EVENTS_CACHE, _EVENTS_CACHE_SIG
    
    data_dir = get_data_dir()
    
    # Check if directory has changed since last cache
    from utils.data import _get_dir_signature
    current_sig = _get_dir_signature(data_dir)
    
    if _EVENTS_CACHE is not None and _EVENTS_CACHE_SIG == current_sig:
        return _EVENTS_CACHE

    with _EVENTS_CACHE_LOCK:
        if _EVENTS_CACHE is not None and _EVENTS_CACHE_SIG == current_sig:
            return _EVENTS_CACHE

        # Directory changed or first load — rebuild cache
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]

        events_list = []

        for file in all_files:
            try:
                df = pd.read_parquet(file)
                df['match_id'] = os.path.basename(file)
                events_list.append(df)
            except Exception:
                continue

        if not events_list:
            _EVENTS_CACHE = pd.DataFrame()
        else:
            _EVENTS_CACHE = pd.concat(events_list, ignore_index=True)
            for col in ['x', 'y', 'Pass End X', 'Pass End Y']:
                if col in _EVENTS_CACHE.columns:
                    _EVENTS_CACHE[col] = pd.to_numeric(_EVENTS_CACHE[col], errors='coerce')

        _EVENTS_CACHE_SIG = current_sig
        return _EVENTS_CACHE

def _plot_logo(ax, team_name: str):
    """Helper to plot team logo on an axis."""
    logo_path = TEAM_LOGOS.get(team_name)
    if logo_path:
        try:
            full_logo_path = os.path.abspath(logo_path)
            if os.path.exists(full_logo_path):
                team_logo = Image.open(full_logo_path)
                ax_pos = ax.get_position()
                logo_ax = ax.figure.add_axes([ax_pos.x1 - 0.025, ax_pos.y1, 0.025, 0.025])
                logo_ax.axis("off")
                logo_ax.imshow(team_logo)
        except Exception:
            pass

def _save_plot_to_base64(fig) -> str:
    """Helper to save figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{encoded_image}"

# --- Plot Generators ---

@disk_cache
def generate_team_ball_winning_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    """Generates 4-col grid of Team Ball Winning Locations. If teams_list is provided, only those teams are plotted."""
    events_df = load_all_events()
    if events_df.empty: return None
    
    ball_types = ['Interception', 'Tackle', 'Ball recovery', 'BlockedPass']
    ball_wins = events_df[
        (events_df['outcome'] == 1) & 
        (events_df['event'].isin(ball_types))
    ].copy()

    teams = ball_wins['team_name'].dropna().unique()
    if teams_list:
        teams = [t for t in teams if t in teams_list]

    team_stats = []
    
    for team in teams:
        t_events = ball_wins[ball_wins['team_name'] == team]
        if not t_events.empty:
            mean_h = t_events['x'].mean()
            team_stats.append({
                'team': team, 'events': t_events, 'mean_height': mean_h
            })
    
    team_stats.sort(key=lambda x: x['mean_height'], reverse=True)

    mpl.rcParams['xtick.color'] = 'w'
    mpl.rcParams['ytick.color'] = 'w'

    ncols = 4
    nrows = int(np.ceil(len(team_stats)/ncols))
    
    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=ncols, grid_height=0.8, title_height=0.13, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 3.5 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    
    axes_list = ax['pitch'].flatten()

    for idx, stats in enumerate(team_stats):
        if idx >= len(axes_list): break
            
        curr_ax = axes_list[idx]
        
        # Heatmap
        bin_statistic = pitch.bin_statistic(stats['events']['x'], stats['events']['y'], statistic='count', bins=(6, 5), normalize=True)
        pitch.heatmap(bin_statistic, curr_ax, cmap=CustomCmap, edgecolor='w', lw=0.5, zorder=0, alpha=0.7)
        
        # Mean Height Line
        mean_h = stats['mean_height']
        color = get_team_color(stats['team'])
        pitch.lines(mean_h, 0, mean_h, 100, color=color, lw=3, zorder=2, ax=curr_ax)
        
        # Label
        path_eff = [path_effects.Stroke(linewidth=3, foreground='k'), path_effects.Normal()]
        curr_ax.text(mean_h + 2, 8, f"{mean_h:.1f}", fontsize=10, color='w', path_effects=path_eff)
        
        display_name = stats['team'][:15] + '...' if len(stats['team']) > 15 else stats['team']
        curr_ax.set_title(f"{idx + 1}: {display_name}", loc="left", color='w', fontsize=12)
        _plot_logo(curr_ax, stats['team'])

    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')

    fig.text(0.04, 0.965, "Süper Lig - Teams Ranked by Average Ball Win Height", fontweight="bold", fontsize=20, color='w')
    
    legend_elements = [
        Patch(facecolor=TACTIQ_ACCENT, edgecolor='none', label='High Density'),
        Patch(facecolor='#47516B', edgecolor='none', label='Low Density'),
        Line2D([0], [0], color='#e63946', lw=3, label='Average Ball Win Height')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.945), ncol=3, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)

    return _save_plot_to_base64(fig)


@disk_cache
def generate_team_common_zonal_actions_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    events_df = load_all_events()
    if events_df.empty: return None
    
    mask = events_df['event'].str.contains('Pass|Goal|Missed|Saved|Post', case=False, na=False)
    full_df = events_df[mask].copy()
    
    teams = full_df['team_name'].dropna().unique()
    if teams_list:
        teams = [t for t in teams if t in teams_list]

    team_actions_grid = []
    pitch = Pitch(pitch_type='opta')
    
    for team in teams:
        team_df = full_df[full_df['team_name'] == team].copy()
        shots_mask = team_df['event'].isin(['Goal', 'Missed', 'Saved', 'Post', 'Attempt Saved', 'SavedShot', 'ShotOnPost'])
        if 'type_id' in team_df.columns: shots_mask |= team_df['type_id'].isin([13, 14, 15, 16])
        shots = team_df[shots_mask]
        
        is_pass = team_df['event'] == 'Pass'
        is_cross = team_df['Cross'].isin(['Si', 1, '1', True]) if 'Cross' in team_df.columns else pd.Series(False, index=team_df.index)
        is_long = team_df['Long ball'].isin(['Si', 1, '1', True]) if 'Long ball' in team_df.columns else pd.Series(False, index=team_df.index)
            
        cross_long = team_df[is_pass & (is_cross | is_long)]
        standard_passes = team_df[is_pass & ~(is_cross | is_long)].copy()
        
        if not standard_passes.empty:
            start_x = standard_passes['x'].values
            start_y = standard_passes['y'].values
            end_x = standard_passes.get('Pass End X', pd.Series(np.nan, index=standard_passes.index)).values
            end_y = standard_passes.get('Pass End Y', pd.Series(np.nan, index=standard_passes.index)).values
            
            valid_coords = ~np.isnan(end_x) & ~np.isnan(end_y)
            directions = np.full(len(standard_passes), 'side', dtype=object)
            
            if np.any(valid_coords):
                dx = 120 * (end_x[valid_coords] - start_x[valid_coords])
                dy = 80 * (end_y[valid_coords] - start_y[valid_coords])
                ang = np.degrees(np.arctan2(dx, dy))
                
                is_fwd = (ang > 30) & (ang < 150)
                is_back = (ang > -150) & (ang < -30)
                
                dir_subset = np.full(np.sum(valid_coords), 'side', dtype=object)
                dir_subset[is_fwd] = 'fwd'
                dir_subset[is_back] = 'back'
                directions[valid_coords] = dir_subset
                
            standard_passes['direction'] = directions
            fwd_passes = standard_passes[standard_passes['direction'] == 'fwd']
            back_passes = standard_passes[standard_passes['direction'] == 'back']
            side_passes = standard_passes[standard_passes['direction'] == 'side']
        else:
            fwd_passes, back_passes, side_passes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        bins_List = []
        for d in [shots, cross_long, fwd_passes, back_passes, side_passes]:
            if not d.empty:
                bins_List.append(pitch.bin_statistic(d['x'], d['y'], statistic='count', bins=(6, 5))['statistic'])
            else:
                bins_List.append(np.zeros((5, 6)))
        stats_stack = np.stack(bins_List)
        max_action_idx = np.argmax(stats_stack, axis=0).astype(float)
        max_action_idx[np.sum(stats_stack, axis=0) == 0] = np.nan
        
        team_bin_result = pitch.bin_statistic(shots['x'], shots['y'], statistic='count', bins=(6, 5))
        team_bin_result['statistic'] = max_action_idx
        team_actions_grid.append({'team': team, 'bins': team_bin_result})

    team_actions_grid.sort(key=lambda x: x['team'])
    nrows = int(np.ceil(len(team_actions_grid)/4))
    fig, ax = pitch.grid(nrows=nrows, ncols=4, grid_height=0.8, title_height=0.13, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 3.5 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()

    for idx, item in enumerate(team_actions_grid):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]
        pitch.heatmap(item['bins'], curr_ax, cmap=CommonActionCmap, edgecolor=TACTIQ_BG, lw=0.5, zorder=0.6, alpha=0.7, vmin=0, vmax=4)
        
        # Add Attacking Direction Arrow
        curr_ax.annotate('', xy=(85, -5), xytext=(15, -5),
            arrowprops=dict(arrowstyle='->', color='w', lw=1),
            ha='center', va='center', zorder=5, annotation_clip=False)
        curr_ax.text(50, -9, 'Attack Direction', color='w', ha='center', va='center', fontsize=8, zorder=5)

        display_name = item['team'][:15] + '...' if len(item['team']) > 15 else item['team']
        curr_ax.set_title(f"{idx + 1}: {display_name}", loc="left", color='w', fontsize=12)
        _plot_logo(curr_ax, item['team'])

    for i in range(len(team_actions_grid), len(axes_list)): axes_list[i].axis('off')
    for i in range(len(team_actions_grid), len(axes_list)): axes_list[i].axis('off')
    
    fig.text(0.04, 0.965, "Süper Lig - Most Common Team Actions by Zone", fontweight="bold", fontsize=20, color='w')
    legend_elements = [
        Line2D([0], [0], color=COLOR_SHOT, lw=4, label='Shot'),
        Line2D([0], [0], color=COLOR_CROSS, lw=4, label='Cross/Long'),
        Line2D([0], [0], color=COLOR_FWD, lw=4, label='Fwd Pass'),
        Line2D([0], [0], color=COLOR_BACK, lw=4, label='Back Pass'),
        Line2D([0], [0], color=COLOR_SIDE, lw=4, label='Side Pass')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.945), ncol=5, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)

    return _save_plot_to_base64(fig)

@disk_cache
def generate_league_top_players_plot() -> Optional[str]:
    from utils.stats import calculate_player_rankings
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    events_df = load_all_events()
    if events_df.empty: return None
    
    sh_sq_df, passer_df, defender_df = calculate_player_rankings(events_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.set_facecolor(TACTIQ_BG)
    
    def plot_bars(df, ax, title, color):
        if df.empty: ax.axis('off'); return
        df = df.head(10).iloc[::-1].reset_index(drop=True)
        ax.barh(df.index, df['total'], color=color, height=0.6)
        ax.set_title(title, color='w', fontsize=14, fontweight='bold', pad=20)
        ax.set_facecolor(TACTIQ_BG)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_yticks([]); ax.tick_params(axis='x', colors='w', labelsize=10); ax.grid(axis='x', color='w', alpha=0.1)
        max_val = df['total'].max()
        for i, row in df.iterrows():
            ax.text(row['total'] + (max_val * 0.02), i, str(int(row['total'])), color='w', va='center', fontweight='bold')
            ax.text(-0.02, i, row['shortName'], color='w', ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=11)
            logo_path = TEAM_LOGOS.get(row.get('team', ''))
            if logo_path and os.path.exists(logo_path):
                img = Image.open(logo_path)
                imagebox = OffsetImage(img, zoom=0.035)
                ab = AnnotationBbox(imagebox, (max_val * 0.07, i), frameon=True, bboxprops=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.8), xycoords='data')
                ax.add_artist(ab)

    plot_bars(sh_sq_df, axes[0], "Top Attacking Threats", "#e63946")
    plot_bars(passer_df, axes[1], "Top Creators", "#f1faee")
    plot_bars(defender_df, axes[2], "Top Defenders", "#a8dadc")
    plt.tight_layout(pad=4.0)
    fig.text(0.5, 0.02, "Aggregated League-wide Player Performance | Powered by TactIQ", color='w', ha='center', fontsize=12, fontstyle='italic')
    return _save_plot_to_base64(fig)


@disk_cache
def generate_team_threat_creation_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    from utils.visuals import calculate_xt
    events_df = load_all_events()
    if events_df.empty: return None
    
    # Use a copy to avoid modifying the cached dataframe and causing fragmentation
    events_df = events_df.copy()

    
    if 'player_name' in events_df.columns: events_df['playerName'] = events_df['player_name']
    if 'event' in events_df.columns: events_df['typeId'] = events_df['event']
    
    sp_mask = events_df['event'].str.contains('Corner|Free kick|Set piece', case=False, na=False)
    in_play_df = events_df[~sp_mask].copy()
    xt_df = calculate_xt(in_play_df)
    
    team_plots = []
    team_plots = []
    minutes_col = 'minute' if 'minute' in events_df.columns else 'min'
    
    all_teams = events_df['team_name'].unique()
    if teams_list:
        all_teams = [t for t in all_teams if t in teams_list]
        
    for team in all_teams:
        t_xt = xt_df[xt_df['team_name'] == team]
        t_events = events_df[events_df['team_name'] == team]
        total_mins = t_events.groupby('match_id')[minutes_col].max().sum() if minutes_col in t_events.columns else 90 * t_events['match_id'].nunique()
        xt_per_90 = 90 * (t_xt['xT'].sum() / total_mins) if total_mins > 0 else 0
        team_plots.append({'team': team, 'data': t_xt, 'xt_per_90': xt_per_90})
        
    team_plots.sort(key=lambda x: x['xt_per_90'], reverse=True)
    
    nrows = int(np.ceil(len(team_plots)/4))
    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=4, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 4 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()
    
    for idx, item in enumerate(team_plots):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]
        t_data = item['data']
        bin_statistic = pitch.bin_statistic(t_data['x_scaled'], t_data['y_scaled'], statistic='sum', bins=(12, 8), values=t_data['xT'])
        
        # Use a more vibrant colormap instead of dark colors (CustomCmap)
        threat_cmap = mpl.colors.LinearSegmentedColormap.from_list("ThreatCmap", [TACTIQ_BG, "#118ab2", "#06d6a0", "#ffd166"])
        pitch.heatmap(bin_statistic, curr_ax, cmap=threat_cmap, edgecolor='w', lw=0.5, zorder=0, alpha=0.85)
        
        # Add Attacking Direction Arrow
        curr_ax.annotate('', xy=(85, -5), xytext=(15, -5),
            arrowprops=dict(arrowstyle='->', color='w', lw=1),
            ha='center', va='center', zorder=5, annotation_clip=False)
        curr_ax.text(50, -9, 'Attack Direction', color='w', ha='center', va='center', fontsize=8, zorder=5)
        
        display_name = item['team'][:15] + '...' if len(item['team']) > 15 else item['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12)
        curr_ax.text(2, 4, f"xT/90: {item['xt_per_90']:.3f}", color='w', fontsize=9, fontweight='bold', bbox=dict(facecolor=TACTIQ_BG, alpha=0.7, edgecolor='none'))
        _plot_logo(curr_ax, item['team'])

    for i in range(len(team_plots), len(axes_list)): axes_list[i].axis('off')
    
    fig.text(0.04, 0.965, "Süper Lig - Teams Ranked by In-Play Threat Creation (xT/90)", fontweight="bold", fontsize=20, color='w')
    
    legend_elements = [
        Patch(facecolor='#ffd166', edgecolor='none', label='High Threat'),
        Patch(facecolor='#06d6a0', edgecolor='none', label='Moderate Threat'),
        Patch(facecolor='#118ab2', edgecolor='none', label='Low Threat')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.945), ncol=3, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)
    
    return _save_plot_to_base64(fig)


@disk_cache
def generate_style_metrics_table() -> Optional[str]:
    """
    Heatmap-style percentile table for 10 tactical metrics per team.
    Returns Base64 PNG string.
    """
    from utils.style_classifier import build_team_style_profiles
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    events_df = load_all_events()
    if events_df.empty:
        return None

    profiles = build_team_style_profiles(events_df)
    if profiles.empty:
        return None

    metric_cols = [
        'p_pass_acc', 'p_fwd_pass', 'p_long_ball', 'p_seq10', 'p_xg_per_shot',
        'p_avg_poss', 'p_ppda', 'p_def_height', 'p_shot_dist', 'p_def_act90',
    ]

    metric_labels = [
        'Pass Acc.', 'Fwd Pass %', 'Short Play', '10+ Seq.', 'xG/Shot',
        'Poss. Length', 'PPDA', 'Def. Line', 'Shot Dist.', 'Def. Acts.',
    ]

    # Best control-style teams at the top
    profiles = profiles.sort_values('possession_score', ascending=True)

    teams = profiles['team'].tolist()

    short_teams = [
        t.replace(' Kulübü', '')
         .replace(' Spor', '')
         .replace(' Jimnastik', '')
         .replace(' Futbol', '')
         .strip()
        for t in teams
    ]

    data_matrix = profiles[metric_cols].values

    n_teams = len(teams)
    n_metrics = len(metric_cols)

    # ── Layout constants ────────────────────────────────────────────────
    CELL_W = 1.35
    CELL_H = 0.72
    CELL_GAP = 0.10

    ROW_STEP = CELL_H + CELL_GAP
    COL_STEP = CELL_W + CELL_GAP

    LEFT = 11.0
    LOGO_X = LEFT - 4.8
    NAME_X = LEFT - 0.35
    STYLE_X = LEFT + n_metrics * COL_STEP + 0.4

    HEADER_Y = n_teams * ROW_STEP + 1.2

    fig_w = 22
    fig_h = max(14, n_teams * 0.75 + 10)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    BG = '#111827'
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'StyleMap',
        ['#ef4444', '#374151', '#22c55e']
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    # ── Subtle column-group background bands ───────────────────────────
    on_x0 = LEFT - CELL_GAP / 2
    on_x1 = LEFT + 5 * COL_STEP - CELL_GAP / 2

    off_x0 = on_x1
    off_x1 = LEFT + n_metrics * COL_STEP

    band_h = n_teams * ROW_STEP + 0.15

    ax.add_patch(
        plt.Rectangle(
            (on_x0, -CELL_GAP),
            on_x1 - on_x0,
            band_h,
            color='#fbbf24',
            alpha=0.04,
            zorder=0
        )
    )

    ax.add_patch(
        plt.Rectangle(
            (off_x0, -CELL_GAP),
            off_x1 - off_x0,
            band_h,
            color='#60a5fa',
            alpha=0.04,
            zorder=0
        )
    )

    # ── Data rows ──────────────────────────────────────────────────────
    for ti, (team_vals, short_name, team_full) in enumerate(
        zip(data_matrix, short_teams, teams)
    ):
        row_y = ti * ROW_STEP

        style = profiles.iloc[ti]['style_label']
        scolor = STYLE_COLORS.get(style, '#9ca3af')

        # Alternating row background
        if ti % 2 == 0:
            ax.add_patch(
                plt.Rectangle(
                    (0, row_y - CELL_GAP / 2),
                    STYLE_X + 8,
                    ROW_STEP,
                    color='white',
                    alpha=0.025,
                    zorder=0
                )
            )

        # Team logo
        logo_path = TEAM_LOGOS.get(team_full)

        if logo_path:
            abs_path = os.path.abspath(logo_path)

            if os.path.exists(abs_path):
                try:
                    img = Image.open(abs_path).convert('RGBA')
                    imagebox = OffsetImage(img, zoom=0.032)

                    ab = AnnotationBbox(
                        imagebox,
                        (LOGO_X, row_y + CELL_H * 0.45),
                        frameon=False,
                        zorder=5
                    )

                    ax.add_artist(ab)

                except Exception:
                    pass

        # Team name
        ax.text(
            NAME_X,
            row_y + CELL_H * 0.45,
            short_name,
            ha='right',
            va='center',
            color='white',
            fontsize=10,
            fontweight='bold'
        )

        # Metric cells
        for mi, val in enumerate(team_vals):
            cx = LEFT + mi * COL_STEP

            ax.add_patch(
                plt.Rectangle(
                    (cx, row_y),
                    CELL_W,
                    CELL_H,
                    color=cmap(norm(val)),
                    zorder=1,
                    linewidth=0
                )
            )

            ax.text(
                cx + CELL_W * 0.5,
                row_y + CELL_H * 0.5,
                f'{val:.0f}',
                ha='center',
                va='center',
                color='white',
                fontsize=9,
                fontweight='bold',
                zorder=2
            )

        # Tactical style label
        ax.text(
            STYLE_X,
            row_y + CELL_H * 0.45,
            style,
            ha='left',
            va='center',
            color=scolor,
            fontsize=9,
            fontweight='bold'
        )

    # ── Column divider between on-ball and off-ball metrics ────────────
    div_x = LEFT + 5 * COL_STEP - CELL_GAP / 2

    ax.plot(
        [div_x, div_x],
        [-CELL_GAP / 2, n_teams * ROW_STEP],
        color='#fbbf24',
        lw=1.8,
        alpha=0.6,
        zorder=4
    )

    # ── Rotated column headers ─────────────────────────────────────────
    for mi, label in enumerate(metric_labels):
        ax.text(
            LEFT + mi * COL_STEP + CELL_W * 0.5,
            HEADER_Y,
            label,
            ha='left',
            va='bottom',
            color='#e5e7eb',
            fontsize=9,
            fontweight='bold',
            rotation=45,
            rotation_mode='anchor'
        )

    # Separator line between table and headers
    ax.plot(
        [LEFT - CELL_GAP / 2, LEFT + n_metrics * COL_STEP],
        [n_teams * ROW_STEP + 0.08, n_teams * ROW_STEP + 0.08],
        color='#4b5563',
        lw=0.8
    )

    # ── Group labels (figure coordinates — immune to axis scaling) ─────
    # Convert data-x midpoints to figure-x fractions
    _xlim_max   = STYLE_X + 11
    _ax_x_span  = 0.97 - 0.03          # matches subplots_adjust left/right below
    on_mid_fig  = 0.03 + (LEFT + 2.5 * COL_STEP) / _xlim_max * _ax_x_span
    off_mid_fig = 0.03 + (LEFT + 7.5 * COL_STEP) / _xlim_max * _ax_x_span

    fig.text(on_mid_fig,  0.905, '◄── ON THE BALL ──►',
             ha='center', va='center', color='#fbbf24',
             fontsize=11, fontweight='bold')
    fig.text(off_mid_fig, 0.905, '◄── OFF THE BALL ──►',
             ha='center', va='center', color='#60a5fa',
             fontsize=11, fontweight='bold')

    # ── Chart title (figure coordinates) ──────────────────────────────
    fig.text(0.5, 0.975,
             'Süper Lig — Tactical Style Percentile Metrics',
             ha='center', va='top', color='white',
             fontsize=18, fontweight='bold')

    fig.text(0.5, 0.945,
             'Each value is a 0–100 league percentile  |  Season 2025/26',
             ha='center', va='top', color='#9ca3af', fontsize=10)

    # ── Axis limits ────────────────────────────────────────────────────
    ax.set_xlim(0, STYLE_X + 11)
    ax.set_ylim(-0.6, n_teams * ROW_STEP + 2.8)
    ax.axis('off')

    # ── Colorbar ───────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation='horizontal',
        fraction=0.015,
        pad=0.01,
        aspect=50
    )

    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels([
        '0  (Bottom of league)',
        '50  (Average)',
        '100  (Top of league)'
    ])

    cbar.ax.tick_params(colors='#9ca3af', labelsize=9)
    cbar.outline.set_edgecolor('#374151')

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.87,   # top 13% reserved for fig.text title/subtitle/group labels
        bottom=0.08
    )

    return _save_plot_to_base64(fig)


@disk_cache
def generate_tactical_style_scatter() -> Optional[str]:
    """2D scatter: possession_score (x) vs defensive_score (y), one dot per team."""
    from utils.style_classifier import build_team_style_profiles
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    events_df = load_all_events()
    if events_df.empty:
        return None

    profiles = build_team_style_profiles(events_df)
    if profiles.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    mid_x = profiles['possession_score'].mean()
    mid_y = profiles['defensive_score'].mean()

    ax.axvline(mid_x, color='#4b5563', lw=1.2, ls='--', alpha=0.7)
    ax.axhline(mid_y, color='#4b5563', lw=1.2, ls='--', alpha=0.7)

    quadrant_labels = [
        (mid_x / 2,                    mid_y / 2,                    "Low Block & Counter"),
        (mid_x / 2,                    mid_y + (100 - mid_y) / 2,    "High Press & Vertical"),
        (mid_x + (100 - mid_x) / 2,   mid_y / 2,                    "Trigger-happy Control"),
        (mid_x + (100 - mid_x) / 2,   mid_y + (100 - mid_y) / 2,   "Control-ball"),
    ]
    for qx, qy, qlabel in quadrant_labels:
        ax.text(qx, qy, qlabel, ha='center', va='center', color='#4b5563',
                fontsize=11, fontstyle='italic', alpha=0.7)

    for _, row in profiles.iterrows():
        color = STYLE_COLORS.get(row['style_label'], '#9ca3af')
        px, py = row['possession_score'], row['defensive_score']
        short = (row['team'].replace(' Kulübü', '').replace(' Spor', '')
                            .replace(' Jimnastik', '').replace(' Futbol', '').strip())

        logo_path = TEAM_LOGOS.get(row['team'])
        logo_shown = False
        if logo_path:
            abs_path = os.path.abspath(logo_path)
            if os.path.exists(abs_path):
                try:
                    img = Image.open(abs_path).convert('RGBA')
                    imagebox = OffsetImage(img, zoom=0.055)
                    ab = AnnotationBbox(
                        imagebox, (px, py),
                        frameon=True,
                        bboxprops=dict(boxstyle='circle,pad=0.25', fc='white',
                                       ec=color, linewidth=2.0),
                        zorder=4
                    )
                    ax.add_artist(ab)
                    logo_shown = True
                except Exception:
                    pass

        if not logo_shown:
            ax.scatter(px, py, color=color, s=220, zorder=4,
                       edgecolors='white', linewidths=0.6)

        ax.annotate(short, (px, py),
                    textcoords='offset points', xytext=(0, -18),
                    color='white', fontsize=8, ha='center', zorder=5,
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()])

    ax.set_xlabel('← Direct Play      Possession Score      Controlled Play →',
                  color='#9ca3af', fontsize=11)
    ax.set_ylabel('← Deep Block      Defensive Score      High Press →',
                  color='#9ca3af', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(colors='#6b7280')
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')

    legend_elements = [
        Patch(facecolor=STYLE_COLORS[s], edgecolor='none', label=s)
        for s in STYLE_COLORS
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False,
              labelcolor='white', fontsize=9, handlelength=1.2)

    ax.set_title('Tactical Style Map — Süper Lig', color='white',
                 fontsize=16, fontweight='bold', pad=16)

    plt.tight_layout()
    return _save_plot_to_base64(fig)


@disk_cache
def generate_team_space_allowed_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    """
    Pitch grid showing where opponents receive passes against each team,
    as a delta heatmap vs. the league average (red = more space allowed).
    """
    events_df = load_all_events()
    if events_df.empty:
        return None

    succ_passes = events_df[
        (events_df['event'] == 'Pass') &
        (events_df['outcome'] == 1) &
        events_df['Pass End X'].notna() &
        events_df['Pass End Y'].notna()
    ].copy()

    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='opta',
                  line_color='white', linewidth=1, stripe=False)

    league_bin = pitch.bin_statistic(
        succ_passes['Pass End X'], succ_passes['Pass End Y'],
        statistic='count', bins=(8, 6), normalize=True
    )
    league_avg = league_bin['statistic'].copy()

    all_teams = events_df['team_name'].dropna().unique()
    if teams_list:
        all_teams = [t for t in all_teams if t in teams_list]

    team_stats = []
    for team in all_teams:
        match_ids = events_df[events_df['team_name'] == team]['match_id'].unique()
        opp_passes = succ_passes[
            succ_passes['match_id'].isin(match_ids) &
            (succ_passes['team_name'] != team)
        ]
        if opp_passes.empty:
            continue
        bin_stat = pitch.bin_statistic(
            opp_passes['Pass End X'], opp_passes['Pass End Y'],
            statistic='count', bins=(8, 6), normalize=True
        )
        delta = bin_stat['statistic'] - league_avg
        delta_bin = dict(bin_stat)
        delta_bin['statistic'] = delta
        danger_score = float(np.nanmean(np.abs(delta)))
        team_stats.append({'team': team, 'bins': delta_bin, 'danger': danger_score})

    if not team_stats:
        return None

    team_stats.sort(key=lambda x: x['danger'], reverse=True)

    ncols = 4
    nrows = int(np.ceil(len(team_stats) / ncols))

    fig, ax = pitch.grid(nrows=nrows, ncols=ncols, grid_height=0.8,
                         title_height=0.13, endnote_height=0.04,
                         space=0.12, axis=False)
    fig.set_size_inches(14, 3.5 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()

    delta_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'DeltaCmap', ['#3b82f6', TACTIQ_BG, '#ef4444']
    )
    vabs = 0.04

    for idx, stats in enumerate(team_stats):
        if idx >= len(axes_list):
            break
        curr_ax = axes_list[idx]
        pitch.heatmap(stats['bins'], curr_ax, cmap=delta_cmap,
                      vmin=-vabs, vmax=vabs,
                      edgecolor='#374151', lw=0.4, zorder=0, alpha=0.95)
                      
        # Add Attacking Direction Arrow
        curr_ax.annotate('', xy=(85, -5), xytext=(15, -5),
            arrowprops=dict(arrowstyle='->', color='w', lw=0.8),
            ha='center', va='center', zorder=5, annotation_clip=False)
        curr_ax.text(50, -8, 'Attacking Direction →', color='#aaa', ha='center', va='center', fontsize=7.5, zorder=5)

        # Add defending indicator on the left side (rotated 90 degrees)
        curr_ax.text(-3, 50, 'Goal Defended', color='#f87171', ha='center', va='center', fontsize=7, fontweight='bold', rotation=90, zorder=5, clip_on=False)

        display_name = (stats['team'][:15] + '...'
                        if len(stats['team']) > 15 else stats['team'])
        curr_ax.set_title(f"{idx + 1}: {display_name}",
                          loc='left', color='w', fontsize=12)
        _plot_logo(curr_ax, stats['team'])

    for i in range(len(team_stats), len(axes_list)):
        axes_list[i].axis('off')

    fig.text(0.04, 0.965,
             'Süper Lig — Space Allowed vs. League Average (Pass Reception)',
             fontweight='bold', fontsize=18, color='w')

    legend_elements = [
        Patch(facecolor='#ef4444', edgecolor='none', label='More space allowed'),
        Patch(facecolor=TACTIQ_BG, edgecolor='white', label='League average'),
        Patch(facecolor='#3b82f6', edgecolor='none', label='Less space allowed'),
    ]
    fig.legend(handles=legend_elements, loc='upper left',
               bbox_to_anchor=(0.04, 0.945), ncol=3, frameon=False,
               labelcolor='w', fontsize=11, handlelength=1.5, columnspacing=2)

    return _save_plot_to_base64(fig)

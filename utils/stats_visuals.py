
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch, VerticalPitch
import matplotlib.patheffects as path_effects
import os
import numpy as np
import io
import base64
from PIL import Image
from typing import Optional, List, Dict, Any
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Optional import for Highlight Text
try:
    import highlight_text as htext
except ImportError:
    htext = None

from utils.data import get_data_dir, TEAM_LOGOS

# --- Theme Configuration ---
TACTIQ_BG = '#313332'
TACTIQ_TEXT = 'white'
TACTIQ_ACCENT = '#FDE636' # Gold
TACTIQ_COLORS = ["#313332","#47516B", "#848178", "#B2A66F", "#FDE636"]

CustomCmap = mpl.colors.LinearSegmentedColormap.from_list("", TACTIQ_COLORS)

# Common Action Colors
COLOR_SHOT = "khaki"
COLOR_CROSS = "mediumpurple"
COLOR_FWD = "palegreen"
COLOR_BACK = "lightsalmon"
COLOR_SIDE = "#6a6a6a"

CommonActionCmap = mpl.colors.ListedColormap([COLOR_SHOT, COLOR_CROSS, COLOR_FWD, COLOR_BACK, COLOR_SIDE])

def get_team_color(team_name: str) -> str:
    # return a constant red color for all teams
    return "#e63946"

# --- Data Caching ---
_EVENTS_CACHE: Optional[pd.DataFrame] = None
_EVENTS_CACHE_SIG = None

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

def get_pass_direction(row, side_angle=30):
    start_x, start_y = row['x'], row['y']
    end_x, end_y = row.get('Pass End X'), row.get('Pass End Y')
    if pd.isna(end_x) or pd.isna(end_y): return 'side'
    dx = 120 * (end_x - start_x) 
    dy = 80 * (end_y - start_y)
    ang = np.degrees(np.arctan2(dx, dy))
    if (ang > side_angle) and (ang < 180 - side_angle): return 'fwd'
    elif (ang > -180 + side_angle) and (ang < -side_angle): return 'back'
    else: return 'side'

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
            standard_passes['direction'] = standard_passes.apply(get_pass_direction, axis=1)
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

def generate_team_cross_success_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    events_df = load_all_events()
    if events_df.empty: return None
    full_df = events_df.copy()
    if 'abs_time' not in full_df.columns:
        if 'minute' in full_df.columns and 'second' in full_df.columns: full_df['abs_time'] = full_df['minute'] * 60 + full_df['second']
        else: full_df = full_df.reset_index(); full_df['abs_time'] = full_df['index']
    
    is_cross = full_df['Cross'].isin(['Si', 1, '1', True]) if 'Cross' in full_df.columns else (full_df['event'] == 'Pass') & (abs(full_df['Pass End Y'] - full_df['y']) >= 10)
    crosses = full_df[is_cross].copy()
    
    cross_outcomes = []
    for idx, cross in crosses.iterrows():
        outcome = 'Unsuccessful'
        next_evts = full_df[(full_df['match_id'] == cross['match_id']) & (full_df['abs_time'] > cross['abs_time']) & (full_df['abs_time'] <= cross['abs_time'] + 5)]
        team_next_evts = next_evts[next_evts['team_name'] == cross['team_name']]
        if any(team_next_evts['event'].isin(['Goal'])): outcome = 'Goal'
        elif any(team_next_evts['event'].isin(['Missed', 'Attempt Saved', 'SavedShot'])): outcome = 'Shot'
        elif cross['outcome'] in [1, '1', 'Successful']: outcome = 'To Team-mate'
        cross_outcomes.append(outcome)
    crosses['cross_outcome'] = cross_outcomes
            
    teams = crosses['team_name'].unique()
    if teams_list:
        teams = [t for t in teams if t in teams_list]

    team_stats = []
    for team in teams:
        team_crosses = crosses[crosses['team_name'] == team]
        effective = team_crosses[team_crosses['cross_outcome'].isin(['Goal', 'Shot'])]
        pct = (len(effective) / len(team_crosses) * 100) if not team_crosses.empty else 0
        team_stats.append({'team': team, 'pct': pct, 'crosses': team_crosses, 'effective_count': len(effective)})
    team_stats.sort(key=lambda x: x['pct'], reverse=True)
    
    nrows = int(np.ceil(len(team_stats)/4))
    pitch = VerticalPitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1, half=True, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=4, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.1, axis=False)
    fig.set_size_inches(14, 4 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()
    
    for idx, stats in enumerate(team_stats):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]
        t_crosses = stats['crosses']
        for _, cross in t_crosses.iterrows():
            col, alpha, z = 'grey', 0.2, 1
            if cross['cross_outcome'] == 'Goal': col, alpha, z = 'yellow', 0.8, 4
            elif cross['cross_outcome'] == 'Shot': col, alpha, z = 'lightseagreen', 0.7, 3
            elif cross['cross_outcome'] == 'To Team-mate': col, alpha, z = '#a855f7', 0.4, 2 # Purple for Teammate
            pitch.lines(cross['x'], cross['y'], cross['Pass End X'], cross['Pass End Y'], color=col, alpha=alpha, lw=1, zorder=z, ax=curr_ax)
            if cross['cross_outcome'] in ['Goal', 'Shot']: pitch.scatter(cross['Pass End X'], cross['Pass End Y'], color=col, s=15, zorder=z+1, ax=curr_ax)
        display_name = stats['team'][:15] + '...' if len(stats['team']) > 15 else stats['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12)
        curr_ax.text(2, 54, f"{stats['pct']:.1f}%", color="w", fontsize=10, ha="right", va="center")
        _plot_logo(curr_ax, stats['team'])

    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')
    
    fig.text(0.04, 0.965, "Süper Lig - Teams Ranked by In-Play Cross Effectiveness", fontweight="bold", fontsize=20, color='w')
    
    legend_elements = [
        Line2D([0], [0], color='yellow', lw=3, label='Goal'),
        Line2D([0], [0], color='lightseagreen', lw=3, label='Shot'),
        Line2D([0], [0], color='#a855f7', lw=3, label='Teammate'),
        Line2D([0], [0], color='grey', lw=3, label='Fail')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.945), ncol=4, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)

    return _save_plot_to_base64(fig)

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

def generate_team_fullback_interplay_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    from utils.visuals import calculate_xt
    events_df = load_all_events()
    if events_df.empty: return None
    
    # Use a copy to avoid modifying the cached dataframe and causing fragmentation
    events_df = events_df.copy()

    
    if 'minute' in events_df.columns and 'second' in events_df.columns: events_df['abs_time'] = events_df['minute'] * 60 + events_df['second']
    else: events_df['abs_time'] = events_df.index
    
    fb_positions = ['DR', 'DL', 'DMR', 'DML', 'Right Back', 'Left Back', 'RWB', 'LWB']
    def is_fb(pos): return any(p in str(pos) for p in fb_positions)
    
    events_df['next_pos'] = events_df['position'].shift(-1)
    events_df['next_team'] = events_df['team_name'].shift(-1)
    
    fb_interplay_mask = ((events_df['event'] == 'Pass') & events_df['position'].apply(is_fb) & events_df['next_pos'].apply(is_fb) & (events_df['team_name'] == events_df['next_team']))
    fb_passes = events_df[fb_interplay_mask].copy()
    
    leads_to_shot = []
    for idx, row in fb_passes.iterrows():
        following = events_df[(events_df['match_id'] == row['match_id']) & (events_df['abs_time'] > row['abs_time']) & (events_df['abs_time'] <= row['abs_time'] + 10) & (events_df['team_name'] == row['team_name'])]
        leads_to_shot.append(any(following['event'].isin(['Goal', 'Missed', 'Attempt Saved', 'SavedShot'])))
    fb_passes['leads_to_shot'] = leads_to_shot
    
    if 'player_name' in events_df.columns: events_df['playerName'] = events_df['player_name']
    if 'event' in events_df.columns: events_df['typeId'] = events_df['event']
    xt_full = calculate_xt(events_df)
    if not xt_full.empty: fb_passes = fb_passes.merge(xt_full[['general_id', 'xT']], on='general_id', how='inner')
    else: fb_passes['xT'] = 0
    
    team_stats = []
    unique_teams = fb_passes['team_name'].unique()
    if teams_list:
        unique_teams = [t for t in unique_teams if t in teams_list]
        
    for team in unique_teams:
        t_passes = fb_passes[fb_passes['team_name'] == team]
        avg_xt = t_passes['xT'].sum() / events_df[events_df['team_name'] == team]['match_id'].nunique()
        team_stats.append({'team': team, 'passes': t_passes, 'xt_match': avg_xt})
    team_stats.sort(key=lambda x: x['xt_match'], reverse=True)
    
    nrows = int(np.ceil(len(team_stats)/4))
    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=4, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 4 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()
    pass_cmap = plt.cm.get_cmap('viridis')
    
    for idx, stats in enumerate(team_stats):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]
        for _, p in stats['passes'].iterrows():
            col, alpha = pass_cmap(min(p.get('xT',0)/0.05, 1.0)), 0.7
            if p['leads_to_shot']: col, alpha = 'white', 0.9
            pitch.lines(p['x'], p['y'], p['Pass End X'], p['Pass End Y'], color=col, alpha=alpha, comet=True, lw=2, ax=curr_ax, zorder=2)
            pitch.scatter(p['Pass End X'], p['Pass End Y'], color=col, s=20, ax=curr_ax, zorder=3)
        
        display_name = stats['team'][:15] + '...' if len(stats['team']) > 15 else stats['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12)
        curr_ax.text(2, 4, f"xT/match: {stats['xt_match']:.3f}", color='w', fontsize=8)
        _plot_logo(curr_ax, stats['team'])

    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')
    
    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')
    
    fig.text(0.05, 0.95, "Süper Lig - Threat Generated through Full Back Interplay", fontweight="bold", fontsize=20, color='w')
    fig.text(0.05, 0.91, "Legend: White/Bright = Shot Assist | Darker = Prep Pass", color='white', fontsize=10)
    return _save_plot_to_base64(fig)


def generate_team_setpiece_concession_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    """Generates 4-col grid of Team Defending against Set Pieces (Conceded Shots/xG)."""
    events_df = load_all_events()
    if events_df.empty: return None
    
    # Logic: Identify Shots conceded by Team X that were from Set Pieces
    # Using specific columns: 'From corner', 'Direct free', 'Set piece' if they exist, or 'Corner taken' events
    
    teams = events_df['team_name'].dropna().unique()
    if teams_list:
        teams = [t for t in teams if t in teams_list]

    team_stats = []
    
    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1, stripe=False)
    
    # helper to check if col exists and is true-ish
    def is_true(df, col):
        if col not in df.columns: return pd.Series(False, index=df.index)
        return df[col].astype(str).isin(['1', '1.0', 'True', 'true', 'Si'])

    for team in teams:
        # We need OPPONENT shots that are from set pieces
        # Filter for shots against this team
        
        # Identify match IDs where this team played
        team_matches = events_df[events_df['team_name'] == team]['match_id'].unique()
        
        # Get all events from these matches
        matches_df = events_df[events_df['match_id'].isin(team_matches)]
        
        # Opponent events (not this team)
        opp_df = matches_df[matches_df['team_name'] != team]
        
        # Filter for Shots
        shot_types = ['Goal', 'Miss', 'Attempt Saved', 'SavedShot', 'Post', 'Missed']
        # Also check type_id if event names differ (e.g. 13, 14, 15, 16)
        
        is_shot = opp_df['event'].isin(shot_types) | opp_df['type_id'].isin([13, 14, 15, 16])
        shots_df = opp_df[is_shot].copy()
        
        # Filter for Set Piece Origin
        # Logic: Shot has 'From corner' OR 'Direct free' OR 'Set piece' (indirect)
        # OR Preceded by Corner Awarded/Taken? 
        # The columns 'From corner', 'Set piece', 'Direct free', 'Penalty' usually allow direct filtering on the shot event.
        
        from_corner = is_true(shots_df, 'From corner')
        direct_free = is_true(shots_df, 'Direct free')
        set_piece = is_true(shots_df, 'Set piece')
        penalty = is_true(shots_df, 'Penalty')
        
        # We want Indirect Set Pieces (Corners, Indirect Free Kicks). Direct Free Kicks maybe too? 
        # User said "Indirect Set Pieces". Usually means Corners + Free Kicks (delivered).
        # 'Set piece' column often implies Indirect Free Kick or Throw-in setup. 
        # 'From corner' is explicit.
        # Direct Free Kicks are shots themselves.
        # Let's include From Corner and Set Piece. Exclude Penalty.
        
        sp_shots = shots_df[from_corner | set_piece | direct_free] # Including direct free kicks as "Set Piece defending"
        
        if not sp_shots.empty:
            team_stats.append({'team': team, 'shots': sp_shots, 'count': len(sp_shots)})
        else:
             team_stats.append({'team': team, 'shots': pd.DataFrame(), 'count': 0})

    team_stats.sort(key=lambda x: x['count'], reverse=True)

    nrows = int(np.ceil(len(team_stats)/4))
    fig, ax = pitch.grid(nrows=nrows, ncols=4, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 4 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()
    
    for idx, stats in enumerate(team_stats):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]
        
        shots = stats['shots']
        if not shots.empty:
            kde = pitch.kdeplot(shots['x'], shots['y'], ax=curr_ax, cmap='Reds', fill=True, levels=10, alpha=0.6)
            pitch.scatter(shots['x'], shots['y'], c='red', s=20, ax=curr_ax, alpha=0.8)
            
        display_name = stats['team'][:15] + '...' if len(stats['team']) > 15 else stats['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12)
        curr_ax.text(2, 4, f"Conc: {stats['count']}", color='w', fontsize=10)
        _plot_logo(curr_ax, stats['team'])

    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')
    
    for i in range(len(team_stats), len(axes_list)): axes_list[i].axis('off')
    
    fig.text(0.04, 0.965, "Süper Lig - Shots Conceded from Indirect Set Pieces", fontweight="bold", fontsize=20, color='w')
    
    legend_elements = [
        Patch(facecolor='#800026', edgecolor='none', label='High Density'), # Dark Red
        Patch(facecolor='#fc4e2a', edgecolor='none', label='Moderate Density'), # Mid Red
        Patch(facecolor='#ffeda0', edgecolor='none', label='Low Density') # Light Red/Yellow
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.945), ncol=3, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)
    
    return _save_plot_to_base64(fig)


def generate_team_space_allowed_plot(teams_list: Optional[List[str]] = None) -> Optional[str]:
    """Generates the 'How much space does each team allow?' Zonal PPDA 10x6 heatmaps using Opta Event Data."""
    events_df = load_all_events()
    if events_df.empty: return None

    pitch = Pitch(pitch_type='opta')

    unique_teams = events_df['team_name'].dropna().unique()
    if teams_list:
        unique_teams = [t for t in unique_teams if t in teams_list]

    league_zone_stats = []
    team_stats = {}

    for team in unique_teams:
        team_matches = events_df[events_df['team_name'] == team]['match_id'].unique()
        matches_df = events_df[events_df['match_id'].isin(team_matches)]

        # Opponent SUCCESSFUL passes received
        opp_passes = matches_df[(matches_df['team_name'] != team) & (matches_df['event'] == 'Pass') & (matches_df['outcome'].isin([1, '1', 'Successful']))].copy()

        if opp_passes.empty:
            continue

        # In Opta, coordinates are acting-team normalized.
        # Defending team (team) native: X=0 is their goal. X=100 is Opponent Goal.
        # Opponent native: X=0 is their goal. X=100 is Defending team's goal.
        # We align to Defending team's perspective.
        # So we must invert the Opponent's coordinates so their attack is aimed at X=0 (<---).
        opp_passes['end_x'] = opp_passes.get('Pass End X', opp_passes['x'])
        opp_passes['end_y'] = opp_passes.get('Pass End Y', opp_passes['y'])
        
        # Fill missing Pass End coordinates with start coordinates just in case
        opp_passes['end_x'] = opp_passes['end_x'].fillna(opp_passes['x'])
        opp_passes['end_y'] = opp_passes['end_y'].fillna(opp_passes['y'])

        opp_passes['end_x'] = 100 - opp_passes['end_x']
        opp_passes['end_y'] = 100 - opp_passes['end_y']

        # Ensure bounds for binning
        opp_passes = opp_passes[(opp_passes['end_x'] >= 0) & (opp_passes['end_x'] <= 100)]

        # Calculate Zonal Density (10 columns, 6 rows = 60 zones)
        opp_stat = pitch.bin_statistic(opp_passes['end_x'], opp_passes['end_y'], bins=(6, 10), statistic='count')['statistic']

        # Space Score per zone: Successful Passes Allowed per Match
        team_match_count = len(team_matches)
        if team_match_count == 0: team_match_count = 1
        zonal_space = opp_stat / team_match_count

        team_stats[team] = zonal_space
        league_zone_stats.append(zonal_space)

    if not team_stats: return None

    # League Average Space map
    league_avg_space = np.mean(np.stack(league_zone_stats), axis=0)

    plots_data = []
    for team, t_space in team_stats.items():
        delta = t_space - league_avg_space
        plots_data.append({'team': team, 'delta': delta})

    plots_data.sort(key=lambda x: x['team'])

    # Grid Dimensions
    ncols = 4
    nrows = int(np.ceil(len(plots_data) / ncols))
    
    # Pitch rendering setup - matching TactIQ Dark Mode
    pitch = Pitch(pitch_color=TACTIQ_BG, line_color='white', line_zorder=2, pitch_type='opta')

    # Increase endnote_height to give room for colorbar and text
    fig, ax = pitch.grid(nrows=nrows, ncols=ncols, grid_height=0.75, title_height=0.1, endnote_height=0.12, space=0.15, axis=False)
    fig.set_size_inches(14, 3.5 * nrows)
    fig.set_facecolor(TACTIQ_BG)

    axes_list = ax['pitch'].flatten()

    # Custom Diverging Colormap (Red -> Dark Grey -> Blue)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("RdDarkBu", ["#d73027", TACTIQ_BG, "#4575b4"])
    
    # Force symmetry around 0: Find highest absolute variance
    max_abs = max([np.max(np.abs(p['delta'])) for p in plots_data])
    if max_abs == 0: max_abs = 1

    for idx, item in enumerate(plots_data):
        if idx >= len(axes_list): break
        curr_ax = axes_list[idx]

        # Hack mplsoccer's heatmap interface
        dummy_stat = pitch.bin_statistic([50], [50], bins=(6, 10))
        dummy_stat['statistic'] = item['delta']

        # Diverging colors (-max_abs to +max_abs). 
        # RdBu: Lower (negative/less space) = Red. Higher (positive/more space = Blue)
        pitch.heatmap(dummy_stat, curr_ax, cmap=cmap, edgecolors=TACTIQ_BG, lw=0.5, vmin=-max_abs, vmax=max_abs, zorder=1)

        # Attacking Direction Arrow (Opponent attacking our goal -> toward X=0 -> Leftwards) placed BELOW the pitch
        curr_ax.annotate('', xy=(20, -5), xytext=(80, -5),
            arrowprops=dict(arrowstyle='->', color='w', lw=1),
            ha='center', va='center', zorder=5, annotation_clip=False)
        curr_ax.text(50, -9, 'Opponent attacking direction', color='w', ha='center', va='center', fontsize=7, zorder=5)

        display_name = item['team'][:15] + '...' if len(item['team']) > 15 else item['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12, pad=10)
        _plot_logo(curr_ax, item['team'])

    for i in range(len(plots_data), len(axes_list)): axes_list[i].axis('off')

    # Titles and Legends matching the tactIQ app
    fig.text(0.04, 0.955, "How much space does each team allow their opponents?", fontweight="bold", fontsize=20, color='w')
    fig.text(0.04, 0.93, "Süper Lig", fontsize=14, color='#ccc')
    
    fig.text(0.5, 0.02, "When the opponent completes a pass, how much space do they have? Compared to the average passes received for each zone.", color='w', ha='center', fontsize=10)
    
    # Create gradient legend using inset axes to mimic the colorbar under the pitch grid
    # Since endnote is up to 0.12, we can put colorbar around 0.06
    cbar_ax = fig.add_axes([0.43, 0.065, 0.14, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_tick_params(color='w', labelcolor='w')
    
    # Legend texts next to the gradient bar
    fig.text(0.42, 0.072, "Less Space", color='w', ha='right', va='center', fontsize=10)
    fig.text(0.58, 0.072, "More Space", color='w', ha='left', va='center', fontsize=10)

    # Convert to Base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('ascii')
    
    return f"data:image/png;base64,{encoded_image}"


# ============================================================
# TACTICAL STYLE SCATTER — Stephanatos Methodology
# ============================================================

# Stil etiketi → renk eşlemesi
STYLE_COLORS = {
    "Control-ball":           "#22c55e",   # Yeşil
    "Trigger-happy Control":  "#84cc16",   # Sarı-yeşil
    "High Press & Vertical":  "#f97316",   # Turuncu
    "Mid-block & Counter":    "#60a5fa",   # Mavi
    "Low Block & Play Out":   "#a78bfa",   # Mor
    "Chaos-ball":             "#ef4444",   # Kırmızı
    "Mixed":                  "#9ca3af",   # Gri
}


def generate_tactical_style_scatter() -> Optional[str]:
    """
    Süper Lig takımlarının Stephanatos taktik stili 2D scatter plot'u.

    X = Possession Style Score (solda direkt → sağda kontrollü)
    Y = Defensive Territory Score (altta derin blok → üstte yüksek pres)

    Her takım için: logo + kısaltılmış isim + renk-kodlu stil etiketi.
    Arka plan 4 quadrant'a bölünmüş, açıklayıcı etiketlerle.

    Returns:
        Base64 PNG string (data:image/png;base64,...)
    """
    from utils.style_classifier import build_team_style_profiles
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.patheffects as pe

    events_df = load_all_events()
    if events_df.empty:
        return None

    profiles = build_team_style_profiles(events_df)
    if profiles.empty:
        return None

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    # ── Quadrant arka planları ──────────────────────────────────────────
    mid_x = 50
    mid_y = 50

    quad_alpha = 0.06
    ax.fill_between([0, mid_x],  mid_y, 100, color='#f97316', alpha=quad_alpha)   # High Press & Vertical
    ax.fill_between([mid_x, 100], mid_y, 100, color='#22c55e', alpha=quad_alpha)  # Control-ball
    ax.fill_between([0, mid_x],  0, mid_y,   color='#ef4444', alpha=quad_alpha)   # Chaos / Low Block
    ax.fill_between([mid_x, 100], 0, mid_y,  color='#60a5fa', alpha=quad_alpha)   # Mid-block Counter

    # Quadrant arka etiketleri
    quad_label_style = dict(color='#4b5563', fontsize=11, ha='center', va='center',
                            fontstyle='italic', fontweight='bold', alpha=0.5)
    ax.text(25,  85, "High Press\n& Vertical",   **quad_label_style)
    ax.text(75,  85, "Control-ball",              **quad_label_style)
    ax.text(25,  15, "Low Block\n& Play Out",     **quad_label_style)
    ax.text(75,  15, "Mid-block\n& Counter",      **quad_label_style)

    # Orta çizgiler
    ax.axvline(mid_x, color='#374151', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(mid_y, color='#374151', lw=1, linestyle='--', alpha=0.6)

    # ── Takım noktaları ────────────────────────────────────────────────
    for _, row in profiles.iterrows():
        team     = row['team']
        poss_s   = row['possession_score']
        def_s    = row['defensive_score']
        label    = row['style_label']
        color    = STYLE_COLORS.get(label, '#9ca3af')

        # Dış çember (renk-kodlu stil)
        ax.scatter(poss_s, def_s, s=600, color=color, alpha=0.85,
                   edgecolors='white', linewidths=1.5, zorder=4)

        # Logo overlay
        logo_path = TEAM_LOGOS.get(team)
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).convert('RGBA')
                imagebox = OffsetImage(img, zoom=0.045)
                ab = AnnotationBbox(imagebox, (poss_s, def_s),
                                    frameon=False, zorder=5)
                ax.add_artist(ab)
            except Exception:
                pass

        # Takım adı kısaltması (son kelime)
        short_name = team.replace(' Kulübü', '').replace(' Spor', '') \
                        .replace(' Jimnastik', '').replace(' Futbol', '').strip()
        # En fazla iki kelime
        parts = short_name.split()
        display = parts[-1] if len(parts) <= 2 else parts[-1]

        # İsim etiketi — küçük ofset ile
        txt = ax.text(poss_s + 1.5, def_s + 1.5, display,
                      color='white', fontsize=9, fontweight='bold', zorder=6,
                      ha='left', va='bottom')
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground='#111827'),
                              pe.Normal()])

    # ── Eksenler & Etiketler ───────────────────────────────────────────
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Possession Style Score →  (Direkt  ◄──────────►  Kontrollü)",
                  color='#9ca3af', fontsize=12, labelpad=12)
    ax.set_ylabel("Defensive Territory Score →  (Derin Blok  ◄──────────►  Yüksek Pres)",
                  color='#9ca3af', fontsize=12, labelpad=12)
    ax.tick_params(colors='#6b7280', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')
    ax.grid(color='#1f2937', linewidth=0.5, alpha=0.5)

    # ── Başlık ────────────────────────────────────────────────────────
    ax.set_title("Süper Lig — Tactical Style Map",
                 color='white', fontsize=20, fontweight='bold', pad=20)
    fig.text(0.5, 0.945,
             "Stephanatos metodolojisi: Toplu Oyun vs Topsuz Oyun  |  Sezon 2024/25",
             color='#9ca3af', fontsize=11, ha='center')

    # ── Legend: Stil renkleri ──────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=c, edgecolor='white', label=lbl, linewidth=0.8)
        for lbl, c in STYLE_COLORS.items()
    ]
    leg = ax.legend(handles=legend_patches, loc='lower right',
                    frameon=True, framealpha=0.15,
                    edgecolor='#374151', labelcolor='white',
                    fontsize=10, title='Oyun Stili',
                    title_fontsize=11, handlelength=1.2)
    leg.get_title().set_color('#e5e7eb')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_plot_to_base64(fig)


def generate_style_metrics_table() -> Optional[str]:
    """
    Her takım için normalize edilmiş 10 metriği gösteren
    yatay bar ızgara tablosu (heatmap-style).

    Returns:
        Base64 PNG string
    """
    from utils.style_classifier import build_team_style_profiles

    events_df = load_all_events()
    if events_df.empty:
        return None

    profiles = build_team_style_profiles(events_df)
    if profiles.empty:
        return None

    metric_cols = ['p_pass_acc', 'p_fwd_pass', 'p_long_ball',
                   'p_seq10', 'p_xg_per_shot', 'p_avg_poss',
                   'p_ppda', 'p_def_height', 'p_shot_dist', 'p_def_act90']
    metric_labels = [
        'Pas İsabet %\n(Wyscout)', 'İleri Pas %', 'Kısa Oyun\n(uzun top ters)',
        '10+ Sekans', 'xG/Şut\n(Wyscout)', 'Poss. Uzunluğu\n(pas/poss)',
        'PPDA\n(Wyscout)', 'Savunma Hattı', 'Şut Mesafesi\n(ters)', 'Def Aksiyon/90'
    ]

    # Takımları pos_score'a göre sırala
    profiles = profiles.sort_values('possession_score', ascending=True)

    teams = profiles['team'].tolist()
    short_teams = [
        t.replace(' Kulübü', '').replace(' Spor', '').replace(' Jimnastik', '')
         .replace(' Futbol', '').strip()
        for t in teams
    ]

    data_matrix = profiles[metric_cols].values  # shape (n_teams, n_metrics)

    n_teams   = len(teams)
    n_metrics = len(metric_cols)

    fig, ax = plt.subplots(figsize=(16, max(8, n_teams * 0.55 + 2)))
    fig.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    # Çift renk haritası: 0=kırmızı, 50=gri, 100=yeşil
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'StyleMap', ['#ef4444', '#374151', '#22c55e']
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    cell_h = 0.8
    cell_w = 1.0

    for ti, (team_vals, short_name, team_full) in enumerate(
            zip(data_matrix, short_teams, teams)):
        for mi, val in enumerate(team_vals):
            rect = plt.Rectangle(
                (mi * cell_w, ti * cell_h),
                cell_w * 0.95, cell_h * 0.85,
                color=cmap(norm(val)), zorder=1
            )
            ax.add_patch(rect)
            ax.text(
                mi * cell_w + cell_w * 0.475,
                ti * cell_h + cell_h * 0.425,
                f'{val:.0f}',
                ha='center', va='center',
                color='white', fontsize=8, fontweight='bold', zorder=2
            )

        # Stil etiketi
        style  = profiles.iloc[ti]['style_label']
        scolor = STYLE_COLORS.get(style, '#9ca3af')
        ax.text(
            -0.3, ti * cell_h + cell_h * 0.4,
            short_name, ha='right', va='center',
            color='white', fontsize=9, fontweight='bold'
        )
        ax.text(
            n_metrics * cell_w + 0.2,
            ti * cell_h + cell_h * 0.4,
            style, ha='left', va='center',
            color=scolor, fontsize=8.5, fontweight='bold'
        )

    # Başlık sütunları
    for mi, label in enumerate(metric_labels):
        ax.text(
            mi * cell_w + cell_w * 0.475,
            n_teams * cell_h + 0.3,
            label, ha='center', va='bottom',
            color='#d1d5db', fontsize=8, fontweight='bold'
        )
        # Kategori ayırıcısı
        if mi == 5:
            ax.axvline(x=mi * cell_w, color='#FDE636', lw=1.5,
                       ymin=0, ymax=1, alpha=0.5, zorder=3)

    ax.set_xlim(-8, n_metrics * cell_w + 8)
    ax.set_ylim(-0.5, n_teams * cell_h + 1.0)
    ax.axis('off')

    # Kategori başlıkları
    ax.text(2.5 * cell_w, n_teams * cell_h + 0.8,
            '◄── ON THE BALL ──►', ha='center', color='#FDE636',
            fontsize=10, fontweight='bold')
    ax.text(7.5 * cell_w, n_teams * cell_h + 0.8,
            '◄── OFF THE BALL ──►', ha='center', color='#60a5fa',
            fontsize=10, fontweight='bold')

    ax.set_title('Süper Lig — Taktik Stil Metrikleri (Ligde Percentile)',
                 color='white', fontsize=16, fontweight='bold', pad=15)

    # Renk skalası açıklaması
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.02, pad=0.02, aspect=40)
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(['0 (Düşük)', '50 (Orta)', '100 (Yüksek)'])
    cbar.ax.tick_params(colors='#9ca3af', labelsize=9)
    cbar.outline.set_edgecolor('#374151')

    plt.tight_layout()
    return _save_plot_to_base64(fig)
import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from matplotlib.patches import Patch, Rectangle

# Add the project directory to sys.path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_visuals import load_all_events, _plot_logo, _save_plot_to_base64, TACTIQ_BG, TACTIQ_ACCENT

def test_progressive_passes_heatmap():
    print("Testing Progressive Passes League Heatmap...")
    events_df = load_all_events()
    if events_df.empty:
        print("Events DataFrame is empty.")
        return

    events_df = events_df.copy()
    
    # Scale to StatsBomb coordinates (120x80) from assumed Opta (100x100)
    events_df['x_scaled'] = events_df['x'] * 1.2
    events_df['y_scaled'] = events_df['y'] * 0.8
    events_df['end_x_scaled'] = pd.to_numeric(events_df['Pass End X'], errors='coerce').fillna(0) * 1.2
    events_df['end_y_scaled'] = pd.to_numeric(events_df['Pass End Y'], errors='coerce').fillna(0) * 0.8

    # Progressive Pass Metric (pro)
    events_df['pro'] = np.where(
        events_df['end_x_scaled'].notna(),
        np.sqrt((120 - events_df['x_scaled'])**2 + (40 - events_df['y_scaled'])**2) -
        np.sqrt((120 - events_df['end_x_scaled'])**2 + (40 - events_df['end_y_scaled'])**2),
        0
    )

    # Filter for successful in-play progressive passes starting in [40, 115]
    mask = (
        (events_df['event'] == 'Pass') &
        (events_df['outcome'] == 1) &
        (events_df['pro'] >= 9.144) &
        (events_df['x_scaled'].between(40, 115))
    )
    
    # Exclude set pieces
    sp_mask = events_df['event'].str.contains('Corner|Free kick|Set piece', case=False, na=False)
    mask = mask & (~sp_mask)

    df_prog = events_df[mask].copy()
    print(f"Found {len(df_prog)} total progressive passes.")

    all_teams = events_df['team_name'].dropna().unique()
    minutes_col = 'minute' if 'minute' in events_df.columns else 'min'
    
    team_plots = []
    for team in all_teams:
        t_prog = df_prog[df_prog['team_name'] == team]
        t_events = events_df[events_df['team_name'] == team]
        
        # Calculate total minutes played to normalize per 90
        total_mins = t_events.groupby('match_id')[minutes_col].max().sum() if minutes_col in t_events.columns else 90 * t_events['match_id'].nunique()
        prog_per_90 = 90 * (len(t_prog) / total_mins) if total_mins > 0 else 0
        
        team_plots.append({
            'team': team,
            'data': t_prog,
            'prog_per_90': prog_per_90
        })

    # Sort teams by progressive passes per 90
    team_plots.sort(key=lambda x: x['prog_per_90'], reverse=True)

    ncols = 4
    nrows = int(np.ceil(len(team_plots) / ncols))

    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='statsbomb', line_color='#555', linewidth=1, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=ncols, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.12, axis=False)
    fig.set_size_inches(14, 3.5 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()

    # Red/Crimson color gradient matching the Opta Analyst visual
    prog_cmap = mpl.colors.LinearSegmentedColormap.from_list("ProgCmap", [TACTIQ_BG, "#fee2e2", "#f87171", "#dc2626", "#7f1d1d"])

    for idx, item in enumerate(team_plots):
        if idx >= len(axes_list):
            break
        curr_ax = axes_list[idx]
        t_data = item['data']
        
        if not t_data.empty:
            # 2D Binned heatmap of pass origin coordinates
            bin_statistic = pitch.bin_statistic(t_data['x_scaled'], t_data['y_scaled'], statistic='count', bins=(8, 6), normalize=True)
            pitch.heatmap(bin_statistic, curr_ax, cmap=prog_cmap, edgecolor='#374151', lw=0.4, zorder=0, alpha=0.9)

        # Draw the progressive pass start threshold line (x=40)
        pitch.lines(40, 0, 40, 80, color='white', linestyle='--', linewidth=1.2, ax=curr_ax, zorder=2)
        
        # Display team name and progressive passes per 90
        display_name = item['team'][:15] + '...' if len(item['team']) > 15 else item['team']
        curr_ax.set_title(f"{idx+1}: {display_name}", color='w', loc='left', fontsize=12)
        curr_ax.text(42, 4, f"PP/90: {item['prog_per_90']:.1f}", color='w', fontsize=8.5, fontweight='bold', bbox=dict(facecolor=TACTIQ_BG, alpha=0.7, edgecolor='none'))
        _plot_logo(curr_ax, item['team'])

    for i in range(len(team_plots), len(axes_list)):
        axes_list[i].axis('off')

    fig.text(0.04, 0.965, "Süper Lig - Where does each team perform progressive passes?", fontweight="bold", fontsize=20, color='w')
    fig.text(0.04, 0.94, "Ranked by progressive passes per 90 minutes. Dashed line indicates progressive start threshold (x=40).", fontsize=11, color='#aaa')
    
    legend_elements = [
        Patch(facecolor='#dc2626', edgecolor='none', label='High Density'),
        Patch(facecolor='#f87171', edgecolor='none', label='Moderate Density'),
        Patch(facecolor='#fee2e2', edgecolor='none', label='Low Density')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.96, 0.96), ncol=3, frameon=False, labelcolor='w', fontsize=11, handlelength=1.5, columnspacing=2)

    img_b64 = _save_plot_to_base64(fig)
    print(f"Progressive Passes Heatmap success: returned Base64 length {len(img_b64)}")


def test_goal_kicks_heatmap():
    print("Testing Goal Kicks League Distribution...")
    events_df = load_all_events()
    if events_df.empty:
        print("Events DataFrame is empty.")
        return

    events_df = events_df.copy()
    
    # Scale to StatsBomb coordinates (120x80) from assumed Opta (100x100)
    events_df['x_scaled'] = events_df['x'] * 1.2
    events_df['y_scaled'] = events_df['y'] * 0.8
    events_df['end_x_scaled'] = pd.to_numeric(events_df['Pass End X'], errors='coerce').fillna(0) * 1.2
    events_df['end_y_scaled'] = pd.to_numeric(events_df['Pass End Y'], errors='coerce').fillna(0) * 0.8

    # Filter for Goal Kicks
    is_gk = (events_df['event'].astype(str).str.contains('Goal Kick', case=False, na=False))
    is_gk = is_gk | (
        (events_df['event'] == 'Pass') & 
        (events_df['x_scaled'] < 7.2) & 
        (events_df['y_scaled'].between(24, 56))
    )
    df_gk = events_df[is_gk].copy()
    print(f"Found {len(df_gk)} total goal kicks.")

    all_teams = events_df['team_name'].dropna().unique()
    team_plots = []
    
    for team in all_teams:
        t_gk = df_gk[df_gk['team_name'] == team]
        total_gk = len(t_gk)
        
        if total_gk == 0:
            team_plots.append({
                'team': team, 'box_pct': 0, 'short_pct': 0, 'long_pct': 0, 'count': 0
            })
            continue
            
        # Classify landings:
        # 1. Inside Box: x_end <= 18 and y_end between 18 and 62
        in_box = t_gk[(t_gk['end_x_scaled'] <= 18) & (t_gk['end_y_scaled'].between(18, 62))]
        
        # 2. Short Outside Box: x_end <= 40 and not in box
        short_outside = t_gk[(t_gk['end_x_scaled'] <= 40) & ~t_gk.index.isin(in_box.index)]
        
        # 3. Long: x_end > 40
        long_gk = t_gk[t_gk['end_x_scaled'] > 40]
        
        box_pct = round((len(in_box) / total_gk) * 100)
        short_pct = round((len(short_outside) / total_gk) * 100)
        long_pct = 100 - box_pct - short_pct # ensure sums to 100%
        
        team_plots.append({
            'team': team,
            'box_pct': box_pct,
            'short_pct': short_pct,
            'long_pct': long_pct,
            'count': total_gk
        })

    # Sort teams alphabetically
    team_plots.sort(key=lambda x: x['team'])

    ncols = 4
    nrows = int(np.ceil(len(team_plots) / ncols))

    # Defensive half pitch: VerticalPitch with half=True zooms on the defensive third/half
    pitch = VerticalPitch(pitch_color=TACTIQ_BG, pitch_type='statsbomb', line_color='#555', linewidth=1.2, half=True, stripe=False)
    fig, ax = pitch.grid(nrows=nrows, ncols=ncols, grid_height=0.8, title_height=0.1, endnote_height=0.04, space=0.1, axis=False)
    fig.set_size_inches(14, 4 * nrows)
    fig.set_facecolor(TACTIQ_BG)
    axes_list = ax['pitch'].flatten()

    for idx, item in enumerate(team_plots):
        if idx >= len(axes_list):
            break
        curr_ax = axes_list[idx]
        
        box_pct = item['box_pct']
        short_pct = item['short_pct']
        long_pct = item['long_pct']

        # Draw the boundary dashed line at x=40 (which divides short vs long)
        # Note: VerticalPitch has y going from bottom to top (0 to 120), so x_scaled is plotted on vertical axis (y in VerticalPitch)
        curr_ax.axhline(40, color='white', linestyle='--', linewidth=1.2, zorder=2)
        
        # Shade the three zones:
        # 1. Penalty Box: x_scaled <= 18 and y_scaled in [18, 62]
        # In VerticalPitch, StatsBomb X is the y-axis, StatsBomb Y is the x-axis (80 to 0)
        rect_box = Rectangle((18, 0), 44, 18, facecolor='#ef4444', alpha=box_pct/100 * 0.75, zorder=1)
        curr_ax.add_patch(rect_box)

        # 2. Short Outside: x_scaled <= 40, excluding penalty box
        # We can shade the rest of the defensive third up to x=40
        # Bottom side: StatsBomb Y from 0 to 18
        rect_bottom = Rectangle((0, 0), 18, 40, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
        curr_ax.add_patch(rect_bottom)
        # Top side: StatsBomb Y from 62 to 80
        rect_top = Rectangle((62, 0), 18, 40, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
        curr_ax.add_patch(rect_top)
        # Front side: StatsBomb X from 18 to 40, StatsBomb Y from 18 to 62
        rect_front = Rectangle((18, 18), 44, 22, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
        curr_ax.add_patch(rect_front)

        # 3. Long: x_scaled > 40 (defensive midfield and attacking half)
        # Shaded from x=40 to 80 (the top of VerticalPitch half-pitch is 80, but since half=True, it goes up to 80/120)
        rect_long = Rectangle((0, 40), 80, 40, facecolor='#ef4444', alpha=long_pct/100 * 0.75, zorder=1)
        curr_ax.add_patch(rect_long)

        # Text labels on the pitch
        # Penalty box: center at StatsBomb Y = 40, StatsBomb X = 9
        curr_ax.text(40, 9, f"{box_pct}%", color='white', fontsize=11, fontweight='900', ha='center', va='center', zorder=5,
                     path_effects=[mpl.patheffects.Stroke(linewidth=2, foreground=TACTIQ_BG), mpl.patheffects.Normal()])
        
        # Short outside: center at StatsBomb Y = 40, StatsBomb X = 28
        curr_ax.text(40, 26, f"{short_pct}%", color='white', fontsize=10, fontweight='bold', ha='center', va='center', zorder=5,
                     path_effects=[mpl.patheffects.Stroke(linewidth=2, foreground=TACTIQ_BG), mpl.patheffects.Normal()])
        
        # Long: center at StatsBomb Y = 40, StatsBomb X = 55
        curr_ax.text(40, 56, f"{long_pct}%", color='white', fontsize=11, fontweight='900', ha='center', va='center', zorder=5,
                     path_effects=[mpl.patheffects.Stroke(linewidth=2, foreground=TACTIQ_BG), mpl.patheffects.Normal()])

        # Team name
        display_name = item['team'][:15] + '...' if len(item['team']) > 15 else item['team']
        curr_ax.set_title(f"{display_name}", color='w', loc='center', fontsize=11, fontweight='bold')
        _plot_logo(curr_ax, item['team'])

    for i in range(len(team_plots), len(axes_list)):
        axes_list[i].axis('off')

    fig.text(0.04, 0.965, "Süper Lig - Where does each team play their goal-kicks?", fontweight="bold", fontsize=20, color='w')
    fig.text(0.04, 0.94, "Shaded regions represent goal-kick landing zones: Inside penalty box, Short outside box (def third), and Long (beyond def third).", fontsize=11, color='#aaa')

    img_b64 = _save_plot_to_base64(fig)
    print(f"Goal Kicks Distribution Heatmap success: returned Base64 length {len(img_b64)}")

if __name__ == "__main__":
    test_progressive_passes_heatmap()
    test_goal_kicks_heatmap()

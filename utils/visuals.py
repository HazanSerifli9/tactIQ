from shared.matplotlib_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
import seaborn as sns
from io import BytesIO
from scipy.spatial import ConvexHull
from mplsoccer import Pitch
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib
from utils import analysis
from shared.logger import get_logger

# Set Matplotlib backend to Agg to avoid GUI issues
matplotlib.use('Agg')

logger = get_logger(__name__)

# --- Constants & Theme ---
TACTIQ_BG = '#313332'  # Project Standard Dark Grey
TACTIQ_FG = '#ffffff'
TACTIQ_ACCENT = '#00ff87' # Green Accent
TACTIQ_ACCENT_SEC = '#e63946' # Red Accent (Secondary)
TACTIQ_WARNING = '#ff9f1c' # Orange
TACTIQ_HOME = '#e63946' # Match Standard Home Red
TACTIQ_AWAY = '#457b9d' # Match Standard Away Blue

# Common Style Settings
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = TACTIQ_BG
plt.rcParams['axes.facecolor'] = TACTIQ_BG
plt.rcParams['text.color'] = TACTIQ_FG
plt.rcParams['xtick.color'] = TACTIQ_FG
plt.rcParams['ytick.color'] = TACTIQ_FG
plt.rcParams['axes.labelcolor'] = TACTIQ_FG
plt.rcParams['axes.edgecolor'] = '#444444'

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=TACTIQ_BG, dpi=120)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    except Exception as e:
        print(f"Error converting figure to base64: {e}")
        return ""

def setup_pitch(ax, title=""):
    """Helper to setup a basic pitch with consistent styling."""
    ax.set_facecolor(TACTIQ_BG)
    # Draw simple pitch outline
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=TACTIQ_FG, linewidth=1, alpha=0.5)
    ax.plot([50, 50], [0, 100], color=TACTIQ_FG, linewidth=1, alpha=0.5) # Halfway line
    # Box
    ax.plot([0, 17, 17, 0], [21.1, 21.1, 78.9, 78.9], color=TACTIQ_FG, linewidth=1, alpha=0.5)
    ax.plot([100, 83, 83, 100], [21.1, 21.1, 78.9, 78.9], color=TACTIQ_FG, linewidth=1, alpha=0.5)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, color=TACTIQ_FG, fontsize=14, pad=10, fontweight='bold')

def plot_convex_hulls(df, team_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Player Territories")

    team_df = filter_position_events(df[df['team_name'] == team_name].dropna(subset=['player_name', 'x', 'y']))

    # Filter for starters or top players to avoid clutter
    top_players = get_starting_xi(team_df, 'player_name')

    colors = plt.cm.get_cmap('tab20', len(top_players))

    for i, player in enumerate(top_players):
        player_df = team_df[team_df['player_name'] == player]
        points = player_df[['x', 'y']].values
        if len(points) > 2:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                # Close the loop
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                
                c = colors(i)
                ax.fill(hull_points[:,0], hull_points[:,1], color=c, alpha=0.2, label=player)
                ax.plot(hull_points[:,0], hull_points[:,1], color=c, linewidth=1.5)
                
                # Centroid for label
                cx = np.mean(hull_points[:,0])
                cy = np.mean(hull_points[:,1])
                ax.text(cx, cy, player.split()[-1], color=c, fontsize=8, ha='center', va='center', fontweight='bold')
            except Exception as e:
                logger.debug("Convex hull skipped for %s: %s", player, e)
                
    return fig_to_base64(fig)

# type_ids: events that don't carry a real on-pitch position and distort averages
_NON_POSITIONAL_TYPE_IDS = {
    5,   # Out
    17,  # Card
    18,  # Player Off
    19,  # Player On
    27,  # Start delay
    28,  # End delay
    30,  # End (period)
    32,  # Start (period)
    34,  # Team setup
    40,  # Formation change
    43,  # Deleted event
}

def filter_position_events(df):
    """Return events that carry real on-pitch position information.
    Filters out-of-bounds coordinates of Out events (x>100, x<0, etc.) and
    non-positional events that fall at (0,0).
    """
    mask = (
        df['x'].between(0, 100, inclusive='both') &
        df['y'].between(0, 100, inclusive='both')
    )
    if 'type_id' in df.columns:
        mask &= ~df['type_id'].isin(_NON_POSITIONAL_TYPE_IDS)
    return df[mask]


def get_starting_xi(team_df, name_col='player_name'):
    """Helper to reliably get the top 11 players, ensuring GK is included."""
    if team_df.empty:
        return []
    df_valid = team_df.dropna(subset=[name_col])
    if df_valid.empty:
        return []

    counts = df_valid[name_col].value_counts()

    if 'position' in df_valid.columns:
        gk_df = df_valid[df_valid['position'] == 'GK']
        if not gk_df.empty:
            gk_name = gk_df[name_col].value_counts().index[0]
            outfielders = [p for p in counts.index if p != gk_name]
            return [gk_name] + outfielders[:10]

    # Fallback to just top 11 by actions
    return counts.head(11).index.tolist()

def plot_switch_map(df, team_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Switches of Play")
    
    # Filter 'Switch of play'
    if 'Switch of play' in df.columns:
        mask = (df['team_name'] == team_name) & (df['Switch of play'].astype(str).str.contains('Si|1|True', case=False, na=False))
        switches = df[mask]
    else:
        # Fallback: Long passes (>30) with significant Y change (>40)
        mask = (df['team_name'] == team_name) & (df['event_id'] == 1)
        if 'event' in df.columns:
             mask = (df['team_name'] == team_name) & (df['event'] == 'Pass')
        switches = df[mask]
        
    for _, row in switches.iterrows():
        try:
            start_x, start_y = row['x'], row['y']
            end_x, end_y = row['Pass End X'], row['Pass End Y']
            if pd.notna(start_x) and pd.notna(end_x):
                ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                         head_width=1.5, head_length=1.5, fc=TACTIQ_HOME, ec=TACTIQ_HOME, alpha=0.8, width=0.3)
        except Exception as e:
            logger.debug("Switch arrow skipped: %s", e)
            
    return fig_to_base64(fig)

def plot_box_entries(df, team_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Box Entries")
    
    # Box is roughly X >= 83, Y between 21.1 and 78.9 (Opta coords)
    mask = (df['team_name'] == team_name) & (df['Pass End X'] >= 83) & (df['Pass End Y'] >= 21.1) & (df['Pass End Y'] <= 78.9) & (df['x'] < 83)
    entries = df[mask]
    
    for _, row in entries.iterrows():
        try:
            ax.arrow(row['x'], row['y'], row['Pass End X'] - row['x'], row['Pass End Y'] - row['y'],
                     head_width=1.5, head_length=1.5, fc=TACTIQ_ACCENT, ec=TACTIQ_ACCENT, alpha=0.6, width=0.3)
        except Exception as e:
            logger.debug("Box entry arrow skipped: %s", e)
            
    return fig_to_base64(fig)

def plot_progressive_pass_clusters(df, team_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Progressive Passes")
    
    mask = (df['team_name'] == team_name) & ((df['Pass End X'] - df['x']) > 10)
    progs = df[mask]
    
    if not progs.empty:
        ax.scatter(progs['x'], progs['y'], c=TACTIQ_WARNING, s=20, alpha=0.6, edgecolors='none')
        sample = progs.sample(min(len(progs), 30))
        for _, row in sample.iterrows():
             ax.arrow(row['x'], row['y'], row['Pass End X'] - row['x'], row['Pass End Y'] - row['y'],
                     head_width=1, head_length=1, fc=TACTIQ_WARNING, ec=TACTIQ_WARNING, alpha=0.4, width=0.1)

    return fig_to_base64(fig)

def plot_transition_risk_map(df, team_name, filter_type='All'):
    """
    Visualizes defensive transitions (turnovers) and highlights those leading to opponent shots/goals.
    filter_type: 'All', 'Goal', 'Shot'
    """
    # Get Transition Data
    turnovers = analysis.get_turnovers_and_outcomes(df, team_name)
    
    if turnovers.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         setup_pitch(ax, f"{team_name} - Defensive Transitions")
         ax.text(50, 50, "No Data Found", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    
    # 1. Heatmap (Always show context if All?)
    # Only show heatmap if 'All' or if we want density of specific events
    # Let's keep general heatmap for context
    try:
        pitch.kdeplot(turnovers.x, turnovers.y, ax=ax, fill=True, levels=100, thresh=0.05, cut=4, cmap='Reds', alpha=0.4)
    except Exception as e:
        logger.debug("KDE plot skipped (not enough data): %s", e)

    # 2. Plot "Safe" Turnovers (No Consequence) as small dots - Only if All
    if filter_type == 'All':
        safe_turnovers = turnovers[turnovers['consequence'] == 'None']
        pitch.scatter(safe_turnovers.x, safe_turnovers.y, s=20, c='grey', alpha=0.3, ax=ax, label='Turnover (Safe)')
    
    # 3. Plot Dangerous Turnovers (Leading to Shot/Goal)
    dangerous_turnovers = turnovers[turnovers['consequence'] != 'None']
    
    # Apply Filter
    if filter_type == 'Goal':
        dangerous_turnovers = dangerous_turnovers[dangerous_turnovers['consequence'] == 'Goal']
    elif filter_type == 'Shot':
        # Strict filter: Only show shots that are NOT goals (since Goal has its own category)
        # OR: Show all shots?
        # Given the dropdown options "Goal" and "Shot", usually implies separation or hierarchy.
        # Let's try strict separation first to make it clear.
        dangerous_turnovers = dangerous_turnovers[dangerous_turnovers['consequence'] == 'Shot']
    
    for _, row in dangerous_turnovers.iterrows():
        # Color based on severity
        color = TACTIQ_ACCENT_SEC if row['consequence'] == 'Goal' else TACTIQ_WARNING
        
        # Start Point (Turnover)
        pitch.scatter(row['x'], row['y'], s=100, c=color, edgecolors='white', linewidth=1.5, zorder=5, ax=ax)
        
        # Outcome Point (Shot Location) and Arrow
        if pd.notna(row['consequence_x']):
            # Draw Arrow
            pitch.lines(row['x'], row['y'], row['consequence_x'], row['consequence_y'], 
                        lw=2, color=color, comet=True, alpha=0.8, ax=ax, zorder=4)
            
            # Outcome Marker
            marker = '*' if row['consequence'] == 'Goal' else 'o'
            size = 150 if row['consequence'] == 'Goal' else 50
            pitch.scatter(row['consequence_x'], row['consequence_y'], marker=marker, s=size, c=color, ax=ax, zorder=6)

    # Title & Legend
    ax.set_title(f"{team_name} - Defensive Transition Risks\n(Turnovers leading to shots in 15s)", color=TACTIQ_FG, fontsize=16, fontweight='bold', pad=15)
    
    # Add simple legend manually
    ax.text(2, 2, "Grey: Safe Turnover", color='grey', fontsize=10)
    ax.text(2, 6, "Orange: Leads to Shot", color=TACTIQ_WARNING, fontsize=10, fontweight='bold')
    ax.text(2, 10, "Red: Leads to Goal", color=TACTIQ_ACCENT_SEC, fontsize=10, fontweight='bold')

    return fig_to_base64(fig)

# ============================================================
# NEW VISUALIZATIONS: COUNTER ATTACKS & SET PIECES
# ============================================================

def plot_counter_attacks(df, team_name, recent_counters=5):
    """
    Plots the paths of detected counter-attacks.
    """
    counters = analysis.detect_counter_attacks(df)
    
    # Filter for team counters
    team_counters = []
    for seq in counters:
        if df.iloc[seq[0]]['team_name'] == team_name:
            team_counters.append(seq)
            
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Counter Attacks ({len(team_counters)} Detected)")
    
    if not team_counters:
        ax.text(50, 50, "No Counter Attack Sequences Detected", color='gray', ha="center")
        return fig_to_base64(fig)
        
    # Plot Filter: Last N or All?
    # Let's plot all but fade older ones or use heatmap + top lines?
    # For clarity, let's plot lines with alpha.
    
    for seq in team_counters[-recent_counters:]: # Most recent only to avoid clutter? Or random?
        rows = df.iloc[seq]
        
        # Color based on outcome?
        # Check last event
        last_ev = rows.iloc[-1]
        ev_name = str(last_ev.get('event','')).lower()
        is_goal = 'goal' in ev_name
        is_shot = any(s in ev_name for s in ['shot', 'save', 'miss', 'post'])
        
        if is_goal:
            color = TACTIQ_ACCENT_SEC # Red/Orange for Danger/Goal
            alpha = 1.0
            lw = 3
        elif is_shot:
            color = TACTIQ_WARNING # Orange
            alpha = 0.8
            lw = 2
        else:
            color = TACTIQ_ACCENT # Green/Standard
            alpha = 0.5
            lw = 1.5
            
        # Draw path
        xs = rows['x'].values
        ys = rows['y'].values
        
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, marker='o', markersize=3)
        
        # Mark Start
        ax.scatter(xs[0], ys[0], color='white', s=20, zorder=3)
        
        # Mark End
        if is_goal:
            ax.scatter(xs[-1], ys[-1], marker='*', s=150, zorder=5)
        elif is_shot:
            ax.scatter(xs[-1], ys[-1], marker='x', s=50, color=color, zorder=4)
            
    # Legend
    ax.text(2, 95, "Goal", color=TACTIQ_ACCENT_SEC, fontweight='bold')
    ax.text(15, 95, "Shot", color=TACTIQ_WARNING, fontweight='bold')
    ax.text(28, 95, "Box Entry", color=TACTIQ_ACCENT, fontweight='bold')
    
    return fig_to_base64(fig)

def plot_set_pieces(df, team_name, sp_type="corners"):
    """
    Visualizes set pieces (Corners or Free Kicks) on a vertical attacking half-pitch
    with completed golden delivery arrows, landing zone shading, and a stats KPI sidebar.
    """
    from mplsoccer import VerticalPitch
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle
    
    sp_data = analysis.get_set_pieces(df, team_name)
    data = sp_data.get(sp_type, pd.DataFrame())
    
    clean = (team_name.replace(' Kulübü','').replace(' Spor','')
                      .replace(' Futbol','').strip())
    title_text = "Corner Kicks" if sp_type == "corners" else "Dangerous Free Kicks"
    
    # ── Layout: pitch (left) + sidebar (right) ──────────────
    fig = plt.figure(figsize=(16, 7.5))
    fig.patch.set_facecolor(TACTIQ_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.4, 1.1], wspace=0.05)
    ax_pitch = fig.add_subplot(gs[0])
    ax_side  = fig.add_subplot(gs[1])
    
    # Draw vertical attacking half pitch (StatsBomb: y is horizontal, x is vertical)
    # Goal is at x=120, so attacking half is x from 60 to 120
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                          line_color='#555', linewidth=1.2, corner_arcs=True,
                          half=True)
    pitch.draw(ax=ax_pitch)
    
    if data.empty:
        ax_pitch.text(40, 90, f"No {title_text} Data Recorded", color='#777',
                      fontsize=12, fontweight='bold', ha='center', va='center')
        ax_side.set_facecolor(TACTIQ_BG)
        ax_side.axis('off')
        return fig_to_base64(fig)
        
    total_kicks = len(data)
    successful_kicks = 0
    box_entries = 0
    six_yard_entries = 0
    
    # Track takers
    takers = {}
    targets = {}
    delivery_styles = {'Inswinger': 0, 'Outswinger': 0, 'Straight': 0, 'Other': 0}
    
    # We will process and normalize all deliveries
    deliveries = []
    
    for idx, row in data.iterrows():
        # Get taker
        taker = row.get('player_name', 'Unknown')
        takers[taker] = takers.get(taker, 0) + 1

        if sp_type == "corners":
            if _is_truthy_flag(row.get('Inswinger')):
                delivery_styles['Inswinger'] += 1
            elif _is_truthy_flag(row.get('Outswinger')):
                delivery_styles['Outswinger'] += 1
            elif _is_truthy_flag(row.get('Straight')):
                delivery_styles['Straight'] += 1
            else:
                delivery_styles['Other'] += 1

            target = _infer_corner_target(df, row, team_name)
            if target:
                targets[target] = targets.get(target, 0) + 1
        
        raw_x = float(row['x'])
        raw_y = float(row['y'])
        raw_ex = float(row['Pass End X']) if pd.notna(row.get('Pass End X')) else np.nan
        raw_ey = float(row['Pass End Y']) if pd.notna(row.get('Pass End Y')) else np.nan
        
        if pd.isna(raw_ex) or pd.isna(raw_ey):
            continue
            
        # Flip coordinates if attacking from right to left (raw x < 50)
        flip = raw_x < 50
        
        x0 = (100 - raw_x) * 1.2 if flip else raw_x * 1.2
        y0 = (100 - raw_y) * 0.8 if flip else raw_y * 0.8
        x1 = (100 - raw_ex) * 1.2 if flip else raw_ex * 1.2
        y1 = (100 - raw_ey) * 0.8 if flip else raw_ey * 0.8
        
        # Clip inside pitch limits
        x0 = np.clip(x0, 0, 120)
        y0 = np.clip(y0, 0, 80)
        x1 = np.clip(x1, 0, 120)
        y1 = np.clip(y1, 0, 80)
        
        is_success = str(row.get('outcome','')).strip() in ['1', 'True', 'Yes', 'Success']
        if is_success:
            successful_kicks += 1
            
        # Check landing zones
        # 18-yard box is x >= 102 and y between 18 and 62
        in_box = (x1 >= 102) and (18 <= y1 <= 62)
        if in_box:
            box_entries += 1
            
        # 6-yard box is x >= 114 and y between 30 and 50
        in_six = (x1 >= 114) and (30 <= y1 <= 50)
        if in_six:
            six_yard_entries += 1
            
        deliveries.append({
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'is_success': is_success, 'taker': taker
        })

    # Slightly separate near-identical deliveries so repeated corners do not
    # collapse into one visible arrow.
    overlap_groups = {}
    for d in deliveries:
        d['plot_x0'], d['plot_y0'] = d['x0'], d['y0']
        d['plot_x1'], d['plot_y1'] = d['x1'], d['y1']
        key = (
            round(d['x0'] / 2), round(d['y0'] / 2),
            round(d['x1'] / 2), round(d['y1'] / 2),
        )
        overlap_groups.setdefault(key, []).append(d)

    for group in overlap_groups.values():
        if len(group) <= 1:
            continue
        offsets = np.linspace(-(len(group) - 1) * 0.8, (len(group) - 1) * 0.8, len(group))
        for d, offset in zip(group, offsets):
            d['plot_y0'] = np.clip(d['y0'] + offset, 0, 80)
            d['plot_y1'] = np.clip(d['y1'] + offset, 0, 80)
        
    # Draw deliveries
    for d in deliveries:
        x0, y0, x1, y1 = d['plot_x0'], d['plot_y0'], d['plot_x1'], d['plot_y1']
        is_success = d['is_success']
        
        color = TACTIQ_ACCENT if is_success else '#ef4444'
        alpha = 0.8 if is_success else 0.28
        lw = 2.0 if is_success else 1.0
        
        pitch.arrows(x0, y0, x1, y1,
                     color=color, alpha=alpha, lw=lw,
                     headwidth=3.5, headlength=4, headaxislength=3.5,
                     ax=ax_pitch, zorder=4)
                     
        pitch.scatter(x1, y1, color=color, s=40, alpha=alpha + 0.1,
                      edgecolors=TACTIQ_BG, linewidths=0.5, ax=ax_pitch, zorder=5)
                      
    # Shading the key landing areas (6-Yard and Penalty Area) faintly
    # 6-yard box: bottom-left (30, 114), width 20, height 6
    rect_six = Rectangle((30, 114), 20, 6, facecolor='#fbbf24', alpha=0.08, edgecolor='none', zorder=1)
    ax_pitch.add_patch(rect_six)
    # 18-yard box: bottom-left (18, 102), width 44, height 18
    rect_box = Rectangle((18, 102), 44, 18, facecolor='#ef4444', alpha=0.04, edgecolor='none', zorder=1)
    ax_pitch.add_patch(rect_box)
    
    plotted_kicks = len(deliveries)
    count_text = (
        f"{plotted_kicks} plotted · {total_kicks} total"
        if plotted_kicks != total_kicks else f"{total_kicks} total"
    )
    ax_pitch.set_title(f"{clean}  ·  {title_text} Delivery Map  ·  {count_text}",
                       color=TACTIQ_FG, fontsize=12, fontweight='bold', pad=12)
                       
    # ── Sidebar Panel ─────────────────────────────────────────
    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis('off')
    
    # Calculate goals from corners/free kicks
    goals_count = 0
    if not df.empty:
        goals_df = df[(df['team_name'] == team_name) & (df['event'] == 'Goal')]
        for _, r in goals_df.iterrows():
            if sp_type == "corners":
                if (_is_truthy_flag(r.get('From corner')) or _is_truthy_flag(r.get('From Corner'))):
                    goals_count += 1
            else:
                is_fk = (_is_truthy_flag(r.get('Free kick')) or _is_truthy_flag(r.get('Free Kick')) or 
                         _is_truthy_flag(r.get('Set piece')) or _is_truthy_flag(r.get('Set Piece')))
                is_corner = (_is_truthy_flag(r.get('From corner')) or _is_truthy_flag(r.get('From Corner')))
                if is_fk and not is_corner:
                    goals_count += 1

    # Title
    ax_side.text(0.5, 0.94, title_text.upper(), fontsize=14, fontweight='bold', color=TACTIQ_ACCENT, ha='center')
    ax_side.plot([0.15, 0.85], [0.91, 0.91], color='#444', linewidth=1.0)
    if plotted_kicks != total_kicks:
        ax_side.text(0.5, 0.885, f"{total_kicks - plotted_kicks} corner missing landing coordinates" if sp_type == "corners" else f"{total_kicks - plotted_kicks} fk missing landing coordinates",
                     fontsize=8, color='#888', ha='center')
    
    # KPI 0: Goals Scored
    ax_side.text(0.5, 0.83, f"{goals_count}", fontsize=30, fontweight='900', color='#22c55e' if goals_count > 0 else 'white', ha='center')
    ax_side.text(0.5, 0.79, "Goals Scored", fontsize=9, color='#888', ha='center')

    # KPI 1: Completion
    comp_pct = round(successful_kicks / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.71, f"{comp_pct}%", fontsize=24, fontweight='bold', color='white', ha='center')
    ax_side.text(0.5, 0.67, "Delivery Completion Rate", fontsize=9, color='#888', ha='center')
    
    # KPI 2: Danger Zone Entries
    box_pct = round(box_entries / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.59, f"{box_pct}%", fontsize=24, fontweight='bold', color='white', ha='center')
    ax_side.text(0.5, 0.55, "Deliveries into 18-Yard Box", fontsize=9, color='#888', ha='center')
    
    # KPI 3: 6-Yard Box Danger
    six_pct = round(six_yard_entries / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.47, f"{six_pct}%", fontsize=24, fontweight='bold', color='white', ha='center')
    ax_side.text(0.5, 0.43, "Deliveries into 6-Yard Box", fontsize=9, color='#888', ha='center')
    
    if sp_type == "corners":
        ax_side.text(0.5, 0.33, "PRIMARY TAKERS", fontsize=9, color='#777', fontweight='bold', ha='center')
        ax_side.plot([0.3, 0.7], [0.31, 0.31], color='#333', linewidth=0.5)
        sorted_takers = sorted(takers.items(), key=lambda x: x[1], reverse=True)[:2]
        y_taker = 0.27
        for name, cnt in sorted_takers:
            ax_side.text(0.16, y_taker, get_short_name(name), fontsize=8.5, color='#eee', ha='left')
            ax_side.text(0.84, y_taker, f"{cnt}", fontsize=8.5, color=TACTIQ_ACCENT, ha='right', fontweight='bold')
            y_taker -= 0.035

        ax_side.text(0.5, 0.19, "DELIVERY TYPE", fontsize=9, color='#777', fontweight='bold', ha='center')
        ax_side.plot([0.3, 0.7], [0.17, 0.17], color='#333', linewidth=0.5)
        y_style = 0.14
        for label, color in [('Inswinger', '#22c55e'), ('Outswinger', '#f97316'), ('Straight', '#60a5fa')]:
            cnt = delivery_styles.get(label, 0)
            ax_side.text(0.18, y_style, label, fontsize=8.5, color='#eee', ha='left')
            ax_side.text(0.84, y_style, f"{cnt}", fontsize=8.5, color=color, ha='right', fontweight='bold')
            y_style -= 0.032

        ax_side.text(0.5, 0.07, "LIKELY TARGETS", fontsize=9, color='#777', fontweight='bold', ha='center')
        ax_side.plot([0.3, 0.7], [0.055, 0.055], color='#333', linewidth=0.5)
        sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:2]
        if sorted_targets:
            y_start = 0.025
            for name, cnt in sorted_targets:
                ax_side.text(0.16, y_start, get_short_name(name), fontsize=8.2, color='#eee', ha='left')
                ax_side.text(0.84, y_start, f"{cnt}", fontsize=8.2, color=TACTIQ_ACCENT, ha='right', fontweight='bold')
                y_start -= 0.03
        else:
            ax_side.text(0.5, 0.025, "No clear target", fontsize=8.2, color='#888', ha='center')
    else:
        # Top Takers List
        ax_side.text(0.5, 0.32, "PRIMARY TAKERS", fontsize=9, color='#777', fontweight='bold', ha='center')
        ax_side.plot([0.3, 0.7], [0.30, 0.30], color='#333', linewidth=0.5)
        
        sorted_takers = sorted(takers.items(), key=lambda x: x[1], reverse=True)[:3]
        y_start = 0.24
        for name, cnt in sorted_takers:
            s_name = get_short_name(name)
            ax_side.text(0.2, y_start, s_name, fontsize=10, color='#eee', ha='left')
            ax_side.text(0.8, y_start, f"{cnt} kicks", fontsize=10, color=TACTIQ_ACCENT, ha='right', fontweight='bold')
            y_start -= 0.05
        
    return fig_to_base64(fig)


def plot_penalties(df, team_name):
    """Penalty attempts map and result list for a single team."""
    from mplsoccer import VerticalPitch
    from matplotlib import gridspec

    sp_data = analysis.get_set_pieces(df, team_name)
    data = sp_data.get("penalties", pd.DataFrame()).copy()

    clean = (team_name.replace(' Kulübü','').replace(' Spor','')
                      .replace(' Futbol','').strip())

    fig = plt.figure(figsize=(16, 7.5))
    fig.patch.set_facecolor(TACTIQ_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.4, 1.1], wspace=0.05)
    ax_pitch = fig.add_subplot(gs[0])
    ax_side  = fig.add_subplot(gs[1])

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                          line_color='#555', linewidth=1.2, corner_arcs=True,
                          half=True)
    pitch.draw(ax=ax_pitch)

    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis('off')

    if data.empty:
        ax_pitch.text(40, 90, "No Penalties Data Recorded", color='#777',
                      fontsize=12, fontweight='bold', ha='center', va='center')
        ax_side.text(0.5, 0.94, "PENALTIES", fontsize=14, fontweight='bold',
                     color=TACTIQ_ACCENT, ha='center')
        ax_side.text(0.5, 0.78, "0", fontsize=34, fontweight='900',
                     color='white', ha='center')
        ax_side.text(0.5, 0.72, "penalty attempts", fontsize=9, color='#888', ha='center')
        return fig_to_base64(fig)

    attempts = []
    for i, (_, row) in enumerate(data.iterrows()):
        raw_x = pd.to_numeric(row.get('x'), errors='coerce')
        raw_y = pd.to_numeric(row.get('y'), errors='coerce')
        if pd.isna(raw_x) or raw_x <= 0:
            raw_x = 88.5
        if pd.isna(raw_y) or raw_y <= 0:
            raw_y = 50.0

        flip = raw_x < 50
        x = (100 - raw_x) * 1.2 if flip else raw_x * 1.2
        y = (100 - raw_y) * 0.8 if flip else raw_y * 0.8
        x = np.clip(x, 0, 120)
        y = np.clip(y + (i - (len(data) - 1) / 2) * 1.8, 0, 80)

        event = str(row.get('event', 'Penalty'))
        if event == 'Goal' or _is_truthy_flag(row.get('Scored')):
            result, color, marker = "Goal", '#22c55e', '*'
        elif event == 'Saved Shot' or _is_truthy_flag(row.get('Saved')):
            result, color, marker = "Saved", '#f97316', 'o'
        elif event in ['Miss', 'Post'] or _is_truthy_flag(row.get('Missed')):
            result, color, marker = "Missed", '#ef4444', 'x'
        else:
            result, color, marker = event, 'white', 'o'

        attempts.append({
            'player': row.get('player_name', 'Unknown'),
            'minute': 0 if pd.isna(pd.to_numeric(row.get('time_min', 0), errors='coerce')) else int(pd.to_numeric(row.get('time_min', 0), errors='coerce')),
            'result': result,
            'color': color,
            'x': x,
            'y': y,
            'marker': marker,
        })

    for attempt in attempts:
        pitch.scatter(attempt['x'], attempt['y'], color=attempt['color'],
                      s=180 if attempt['result'] == 'Goal' else 95,
                      marker=attempt['marker'], alpha=0.9,
                      edgecolors='white', linewidths=0.8,
                      ax=ax_pitch, zorder=5)
        ax_pitch.text(attempt['y'], attempt['x'] - 4, f"{attempt['minute']}'",
                      color='white', fontsize=8, ha='center', va='center',
                      fontweight='bold', zorder=6)

    ax_pitch.set_title(f"{clean}  ·  Penalties  ·  {len(attempts)} total",
                       color=TACTIQ_FG, fontsize=12, fontweight='bold', pad=12)

    goals = sum(1 for a in attempts if a['result'] == 'Goal')
    ax_side.text(0.5, 0.94, "PENALTIES", fontsize=14, fontweight='bold',
                 color=TACTIQ_ACCENT, ha='center')
    ax_side.plot([0.15, 0.85], [0.91, 0.91], color='#444', linewidth=1.0)
    ax_side.text(0.5, 0.80, f"{goals}/{len(attempts)}", fontsize=34, fontweight='900',
                 color='#22c55e' if goals == len(attempts) else '#fbbf24', ha='center')
    ax_side.text(0.5, 0.75, "penalties scored", fontsize=9, color='#888', ha='center')

    ax_side.text(0.5, 0.62, "ATTEMPTS", fontsize=9, color='#777', fontweight='bold', ha='center')
    ax_side.plot([0.3, 0.7], [0.60, 0.60], color='#333', linewidth=0.5)
    y_start = 0.53
    for attempt in attempts[:6]:
        ax_side.text(0.16, y_start, f"{attempt['minute']}' {get_short_name(attempt['player'])}",
                     fontsize=9.5, color='#eee', ha='left')
        ax_side.text(0.84, y_start, attempt['result'], fontsize=9.5,
                     color=attempt['color'], ha='right', fontweight='bold')
        y_start -= 0.06

    ax_side.text(0.5, 0.08, "Markers show penalty attempt location", fontsize=8,
                 color='#888', ha='center')
    return fig_to_base64(fig)

# --- Advanced Analysis Code Integration ---

def get_short_name(full_name):
    if pd.isna(full_name):
        return full_name
    if not isinstance(full_name, str):
        return str(full_name)
    parts = full_name.split()
    if len(parts) == 1:
        return full_name
    elif len(parts) == 2:
        return parts[0][0] + ". " + parts[1]
    else:
        return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])

def get_shorter_name(full_name):
    if not isinstance(full_name, str):
        return str(full_name)[:3] if full_name else ""
    parts = full_name.split()
    if len(parts) == 1:
        return parts[0][:2].upper()
    shorter_name = ''.join([p[0].upper() for p in parts[:-1]]) + parts[-1][0].upper()
    return shorter_name

def preprocess_for_network(df):
    """Refactors the user's preprocessing logic."""
    df = df.copy()

    # Ensure necessary columns
    if 'Pass End X' not in df.columns:
        df['Pass End X'] = np.nan
    if 'Pass End Y' not in df.columns:
        df['Pass End Y'] = np.nan

    df['end_x'] = pd.to_numeric(df['Pass End X'], errors='coerce').fillna(0)
    df['end_y'] = pd.to_numeric(df['Pass End Y'], errors='coerce').fillna(0)

    df['shortName'] = df['player_name'].apply(get_short_name)
    df['shorter name'] = df['player_name'].apply(get_shorter_name)

    # Filter out off-pitch / non-positional events
    df = filter_position_events(df)

    # Rescale to StatsBomb (120x80) from assumed Opta (100x100)
    df['x_scaled'] = df['x'] * 1.2
    df['y_scaled'] = df['y'] * 0.8
    df['end_x_scaled'] = df['end_x'] * 1.2
    df['end_y_scaled'] = df['end_y'] * 0.8

    return df

def get_passes_between_df(team_name, passes_df):
    team_passes_df = passes_df[passes_df['team_name'] == team_name]

    if team_passes_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    passes_player_ids_df = team_passes_df.loc[:, ['shortName', "shorter name", 'receiver', 'team_name']].copy()
    passes_player_ids_df['shortName'] = passes_player_ids_df['shortName'].astype(str)
    passes_player_ids_df['receiver'] = passes_player_ids_df['receiver'].astype(str)
    
    passes_player_ids_df['player_pair'] = passes_player_ids_df.apply(
        lambda row: tuple(sorted([row['shortName'], row['receiver']])), axis=1
    )
    
    passes_player_ids_df['pos_max'] = passes_player_ids_df.apply(lambda r: max(r['shortName'], r['receiver']), axis=1)
    passes_player_ids_df['pos_min'] = passes_player_ids_df.apply(lambda r: min(r['shortName'], r['receiver']), axis=1)

    average_locs_and_count_df = team_passes_df.groupby('shortName').agg(
        {'x_scaled': ['median'], 'y_scaled': ['median', 'count']}
    )
    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count']

    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).size().reset_index(name='pass_count')
    
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True, suffixes=['', '_end'])

    return passes_between_df, average_locs_and_count_df

def plot_pass_network(df, team_name):
    # Preprocess
    df_processed = preprocess_for_network(df)
    
    # Identify receivers
    # We need to sort by time/index to shift correctly
    # Assuming df is sorted by event order
    df_processed = df_processed.sort_values(by=['period_id', 'time_min', 'time_sec', 'event_id'] if 'event_id' in df_processed.columns else ['min', 'sec']) if 'period_id' in df_processed.columns else df_processed
    
    df_processed["receiver"] = df_processed["shortName"].shift(-1)
    
    # Filter for successful passes
    if 'event' in df_processed.columns:
        passes_df = df_processed[(df_processed['event'] == 'Pass') & (df_processed['outcome'] == 1)]
    elif 'type_id' in df_processed.columns:
        passes_df = df_processed[(df_processed['type_id'] == 1) & (df_processed['outcome'] == 1)]
    else:
        passes_df = pd.DataFrame()
        
    if passes_df.empty:
         # Return empty placeholder
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Passing Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    # Filter for Top 11 Players (Starters Proxy)
    # Use total activity from the processed DF to identify the main 11 players, 
    # not just those who passed the most (which might exclude strikers).
    
    team_all_df = df_processed[df_processed['team_name'] == team_name]
    top_11 = get_starting_xi(team_all_df, 'shortName')
    
    # Filter DF to only include passes FROM top 11 and TO top 11
    passes_df = passes_df[passes_df['shortName'].isin(top_11) & passes_df['receiver'].isin(top_11)]

    passes_between_df, average_locs_and_count_df = get_passes_between_df(team_name, passes_df)
    
    if passes_between_df.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "Insufficient Data for Network", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    # Use mplsoccer Pitch with Alternative Theme Style
    pitch = Pitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG)
    pitch.draw(ax=ax)
    
    # Plot Logic - Refined
    MAX_LINE_WIDTH = 10
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() * MAX_LINE_WIDTH)
    
    # Lines
    # Use simple lines as per "Alternative Theme" example, usually simpler color
    # Using 'col' from args? Not passed. Using user preferred #e63946
    
    color = np.array(to_rgba(TACTIQ_HOME))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    MIN_TRANSPARENCY = 0.1
    c_transparency = (c_transparency * (0.85 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency
    
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y,
                passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                lw=passes_between_df.width, color=color, zorder=1, ax=ax)
                
    # Nodes - Alternative Style (Donut / Inner Circle)
    # Outer Circle
    MAX_MARKER_SIZE = 1000
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)
    
    pitch.scatter(average_locs_and_count_df.pass_avg_x, average_locs_and_count_df.pass_avg_y,
                  s=average_locs_and_count_df['marker_size'], color=TACTIQ_HOME, edgecolor=TACTIQ_FG, linewidth=1, alpha=1, ax=ax)
                  
    # Inner Circle (White/Background to create ring effect if desired, or just white text on red)
    # The example used colored nodes with white border, or contrasting internal.
    # Let's do simple white text on red node matching Dark theme.
                  
    # Fetch positions from dataframe to use instead of names
    pos_map = {}
    if 'position' in df_processed.columns:
        valid_pos = df_processed[(df_processed['team_name'] == team_name) & 
                                 (df_processed['position'].notna()) & 
                                 (df_processed['position'] != 'N/A')]
        pos_map = valid_pos.groupby('shortName')['position'].first().to_dict()

    for index, row in average_locs_and_count_df.iterrows():
        label = pos_map.get(index)
        if not label or pd.isna(label) or label == 'N/A':
            label = get_shorter_name(index)
            
        pitch.annotate(label, xy=(row.pass_avg_x, row.pass_avg_y), c=TACTIQ_FG, va='center', ha='center', size=10, weight='bold', ax=ax)
        
    ax.set_title(f"{team_name} - Passing Network", color=TACTIQ_FG, size=20)
    
    return fig_to_base64(fig)

def plot_defensive_block(df, team_name):
    # Prepare Data
    df_processed = preprocess_for_network(df)
    
    # Filter defensive actions
    # Using Opta type_ids if available, or event names
    # User logic: 
    # (df['typeId'] == 'Aerial') & (df['x'] <= 80) | ...
    
    def_types = ['BallRecovery', 'BlockedPass', 'Challenge', 'Clearance', 'Error', 'Foul', 'Interception', 'Tackle']
    
    if 'event' in df_processed.columns:
        # Assuming event name column
        mask = (df_processed['team_name'] == team_name) & (
            ((df_processed['event'] == 'Aerial') & (df_processed['x'] <= 80)) |
            (df_processed['event'].isin(def_types))
        )
    elif 'type_id' in df_processed.columns:
        # Mapping needs to be known, but if we don't have it, we might try to guess or use passed logic if names are present elsewhere
        # If 'event' is not present, we can't reliably filter without type ID map. 
        # But earlier functions used type_id. Let's assume standardized naming from extraction OR type_id if we had a map.
        # Fallback to simple logic if only type_id available and we don't know the map:
        # Just use common sense ranges or if descriptions exist. extract_fixture provides 'event' column usually?
        # Actually extract_fixture.py refactor didn't explicitly map event names to type_ids in the DF unless it was already there.
        # Let's try to rely on 'event' column if present.
        mask = (df_processed['team_name'] == team_name) # Placeholder if no event names
        if 'description' in df_processed.columns:
             # Sometimes description has it
             pass
    
    # If we can't filter specific types, we returns generic defensive locations (e.g. own half recoveries)
    if 'event' not in df_processed.columns:
        # Fallback: Recoveries/Interceptions often type 49, 12, etc. 
        # Let's use x < 60 as rough proxy for defensive actions if types unknown
         mask = (df_processed['team_name'] == team_name) & (df_processed['x'] < 60)
         
    defensive_actions_df = df_processed[mask].copy()
    
    if defensive_actions_df.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Defensive Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    # Filter for Top 11 Players by Defensive Activity (or general activity if preferred)
    # User asked to remove subs. Top 11 by event count is safer.
    # Let's use the full DF's activity to find the top 11 players, then filter defensive actions.
    
    # Get top 11 from the *processed* df (filtered by team) to ensure we get main players
    team_all_df = df_processed[df_processed['team_name'] == team_name]
    top_11 = get_starting_xi(team_all_df, 'shortName')
    
    defensive_actions_df = defensive_actions_df[defensive_actions_df['shortName'].isin(top_11)]

    # Calculate mean positions
    average_locs = defensive_actions_df.groupby('shortName').agg({'x_scaled': ['median'], 'y_scaled': ['median', 'count']})
    average_locs.columns = ['x', 'y', 'count']
    average_locs = average_locs.reset_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)
    
    # Heatmap
    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", [TACTIQ_BG, TACTIQ_HOME], N=500)
    try:
        pitch.kdeplot(defensive_actions_df.x_scaled, defensive_actions_df.y_scaled, ax=ax, fill=True, levels=100, thresh=0.02, cut=4, cmap=flamingo_cmap)
    except Exception as e:
        logger.debug("Defensive KDE skipped (not enough data): %s", e)

    # Scatter Nodes
    MAX_MARKER_SIZE = 2000
    if not average_locs.empty:
        average_locs['marker_size'] = (average_locs['count'] / average_locs['count'].max() * MAX_MARKER_SIZE)
        
        for index, row in average_locs.iterrows():
             pitch.scatter(row['x'], row['y'], s=row['marker_size']+50, marker='o', color=TACTIQ_BG, edgecolor=TACTIQ_FG, linewidth=1, alpha=1, zorder=3, ax=ax)
             
        # Annotate
        pos_map = {}
        if 'position' in df_processed.columns:
            valid_pos = df_processed[(df_processed['team_name'] == team_name) & 
                                     (df_processed['position'].notna()) & 
                                     (df_processed['position'] != 'N/A')]
            pos_map = valid_pos.groupby('shortName')['position'].first().to_dict()

        def get_label(name):
            label = pos_map.get(name)
            if not label or pd.isna(label) or label == 'N/A':
                return get_shorter_name(name)
            return label

        average_locs['label'] = average_locs['shortName'].apply(get_label)
        for index, row in average_locs.iterrows():
            pitch.annotate(row['label'], xy=(row.x, row.y), c=TACTIQ_FG, ha='center', va='center', size=10, ax=ax)
            
    # Scatter all actions
    pitch.scatter(defensive_actions_df.x_scaled, defensive_actions_df.y_scaled, s=10, marker='x', color='yellow', alpha=0.2, ax=ax)
    
    # Height Line
    if not average_locs.empty:
        dah = round(average_locs['x'].mean(), 2)
        ax.axvline(x=dah, color='gray', linestyle='--', alpha=0.75, linewidth=2)
        ax.text(dah+1, 85, f"{dah}m", fontsize=15, color=TACTIQ_FG, ha='left', va='center')

    ax.set_title(f"{team_name} - Defensive Block", color=TACTIQ_FG, size=20)
    
    return fig_to_base64(fig)


def plot_defensive_profile(df, team_name):
    """
    Comprehensive defensive profile — distinct from Out-of-Possession avg positions.
    Shows: block classification, compactness ellipse, 1H vs 2H shift, action breakdown.
    """

    df_processed = preprocess_for_network(df)

    def_types = ['BallRecovery', 'BlockedPass', 'Challenge', 'Clearance', 'Error',
                 'Foul', 'Interception', 'Tackle']

    if 'event' in df_processed.columns:
        mask = (df_processed['team_name'] == team_name) & (
            ((df_processed['event'] == 'Aerial') & (df_processed['x'] <= 80)) |
            (df_processed['event'].isin(def_types))
        )
    else:
        mask = (df_processed['team_name'] == team_name) & (df_processed['x'] < 60)

    defensive_df = df_processed[mask].copy()

    if defensive_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Defensive Data", color=TACTIQ_FG, ha="center")
        ax.axis('off')
        return fig_to_base64(fig)

    # Top 11 players
    team_all_df = df_processed[df_processed['team_name'] == team_name]
    top_11 = get_starting_xi(team_all_df, 'shortName')
    defensive_df = defensive_df[defensive_df['shortName'].isin(top_11)]

    # Normalise x so x=0 always means the team's own goal.
    # Per period: if the team's mean x > 60 they are operating in the right half
    # (attacking right→left, defending toward x=120), so flip both axes.
    key = 'period_id' if 'period_id' in defensive_df.columns else None
    periods = defensive_df[key].unique() if key else [None]
    normed_parts = []
    for period in periods:
        if key:
            p_def  = defensive_df[defensive_df[key] == period].copy()
            p_all  = team_all_df[team_all_df[key] == period]
        else:
            p_def  = defensive_df.copy()
            p_all  = team_all_df
        if not p_all.empty and p_all['x_scaled'].mean() > 60:
            p_def['x_scaled'] = 120 - p_def['x_scaled']
            p_def['y_scaled'] = 80  - p_def['y_scaled']
        normed_parts.append(p_def)
    defensive_df = pd.concat(normed_parts) if normed_parts else defensive_df

    if defensive_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Defensive Data for Starting XI", color=TACTIQ_FG, ha="center")
        ax.axis('off')
        return fig_to_base64(fig)

    # Per-player average defensive positions
    avg_locs = defensive_df.groupby('shortName').agg(
        x=('x_scaled', 'median'),
        y=('y_scaled', 'median'),
        count=('x_scaled', 'count')
    ).reset_index()

    # --- Engagement Metrics ---
    # Keep the displayed AVG on the same action-level basis as the 1H/2H lines.
    # Using player medians here can produce an AVG outside both half medians.
    avg_line = defensive_df['x_scaled'].median()                       # StatsBomb 0-120

    # Action spread: exclude GK (lowest x position) for realistic outfield spread.
    outfield_locs = avg_locs.nlargest(len(avg_locs) - 1, 'x') if len(avg_locs) > 1 else avg_locs
    vertical_compact = outfield_locs['x'].max() - outfield_locs['x'].min()
    horizontal_compact = outfield_locs['y'].max() - outfield_locs['y'].min()

    # Convert to real meters (SB 120→105m, 80→68m)
    vert_meters  = vertical_compact * (105 / 120)
    horiz_meters = horizontal_compact * (68 / 80)
    line_meters  = avg_line * (105 / 120)

    # Engagement classification using real pitch meters (0 to 105m).
    # This is event-data engagement height, not a tracking-data back-line height.
    if line_meters > 43.0:
        block_type, block_color = "HIGH ENGAGEMENT", "#ef4444"
    elif line_meters > 34.0:
        block_type, block_color = "MID ENGAGEMENT", "#fbbf24"
    else:
        block_type, block_color = "LOW ENGAGEMENT", "#3b82f6"

    # --- 1H vs 2H split ---
    if 'expanded_minute' in defensive_df.columns:
        h1_df = defensive_df[defensive_df['expanded_minute'] < 45]
        h2_df = defensive_df[defensive_df['expanded_minute'] >= 45]
    elif 'time_min' in defensive_df.columns:
        # period_id based split
        if 'period_id' in defensive_df.columns:
            h1_df = defensive_df[defensive_df['period_id'] == 1]
            h2_df = defensive_df[defensive_df['period_id'] == 2]
        else:
            h1_df = defensive_df[defensive_df['time_min'] < 45]
            h2_df = defensive_df[defensive_df['time_min'] >= 45]
    else:
        half = len(defensive_df) // 2
        h1_df = defensive_df.iloc[:half]
        h2_df = defensive_df.iloc[half:]

    h1_line = h1_df['x_scaled'].median() if not h1_df.empty else avg_line
    h2_line = h2_df['x_scaled'].median() if not h2_df.empty else avg_line
    h1_meters = h1_line * (105 / 120)
    h2_meters = h2_line * (105 / 120)

    # --- Recovery vs Pressure split ---
    recovery_types = ['BallRecovery', 'Interception']
    challenge_types = ['Tackle', 'Challenge', 'Foul', 'Aerial']

    if 'event' in defensive_df.columns:
        recoveries = defensive_df[defensive_df['event'].isin(recovery_types)]
        challenges = defensive_df[defensive_df['event'].isin(challenge_types)]
    else:
        recoveries = defensive_df.iloc[:0]
        challenges = defensive_df.iloc[:0]

    # ===================== CREATE FIGURE =====================
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(TACTIQ_BG)

    # Grid: pitch (65%) + info panel (35%)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.04)

    # ---------- LEFT: Defensive Action Pitch ----------
    ax_pitch = fig.add_subplot(gs[0])
    pitch = Pitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                  line_color=TACTIQ_FG, linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax_pitch)

    # KDE heatmap (tinted with block color)
    flamingo_cmap = LinearSegmentedColormap.from_list(
        "DefProfile", [TACTIQ_BG, block_color], N=500)
    try:
        pitch.kdeplot(defensive_df.x_scaled, defensive_df.y_scaled,
                      ax=ax_pitch, fill=True, levels=100, thresh=0.02,
                      cut=4, cmap=flamingo_cmap)
    except Exception:
        pass

    # Scatter all defensive actions (tiny markers)
    pitch.scatter(defensive_df.x_scaled, defensive_df.y_scaled,
                  s=10, marker='x', color='yellow', alpha=0.15, ax=ax_pitch)

    # Defensive line (average)
    ax_pitch.axvline(x=avg_line, color='white', linestyle='-', alpha=0.7, linewidth=2.5)
    ax_pitch.text(avg_line + 1.5, 83, f"AVG {line_meters:.0f}m",
                  fontsize=13, color='white', ha='left', fontweight='bold',
                  bbox=dict(facecolor=TACTIQ_BG, alpha=0.8, edgecolor='none',
                            boxstyle='round,pad=0.3'))

    # 1H / 2H lines
    ax_pitch.axvline(x=h1_line, color='#22c55e', linestyle=':', alpha=0.6, linewidth=2)
    ax_pitch.axvline(x=h2_line, color='#f97316', linestyle=':', alpha=0.6, linewidth=2)

    # Half legend at bottom
    ax_pitch.text(h1_line, -3, "1H", fontsize=10, color='#22c55e',
                  ha='center', fontweight='bold')
    ax_pitch.text(h2_line, -3, "2H", fontsize=10, color='#f97316',
                  ha='center', fontweight='bold')

    ax_pitch.set_title(f"{team_name} — Defensive Profile",
                       color=TACTIQ_FG, size=18, fontweight='bold', pad=12)

    # ---------- RIGHT: Metrics Panel ----------
    ax_info = fig.add_subplot(gs[1])
    ax_info.set_facecolor(TACTIQ_BG)
    ax_info.axis('off')
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    # -- Block type badge --
    ax_info.text(0.5, 0.94, block_type, fontsize=24, fontweight='900',
                 color=block_color, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor=TACTIQ_BG,
                           edgecolor=block_color, linewidth=3))

    # -- Action spread section --
    ax_info.text(0.5, 0.83, "A C T I O N   S P R E A D", fontsize=9,
                 color='#777', ha='center', fontweight='bold')

    ax_info.plot([0.15, 0.85], [0.815, 0.815], color='#444', linewidth=0.5)

    ax_info.text(0.5, 0.77, f"↕  {vert_meters:.1f}m", fontsize=18,
                 color='white', ha='center', fontweight='bold')
    ax_info.text(0.5, 0.75, "vertical defensive-action spread", fontsize=8,
                 color='#888', ha='center')

    ax_info.text(0.5, 0.69, f"↔  {horiz_meters:.1f}m", fontsize=18,
                 color='white', ha='center', fontweight='bold')
    ax_info.text(0.5, 0.67, "horizontal defensive-action spread", fontsize=8,
                 color='#888', ha='center')

    # Action-spread rating. Smaller spread means the team's defensive actions
    # were concentrated in a tighter area; this is not tracking-data compactness.
    compact_score = max(0, min(100, 100 - ((vert_meters + horiz_meters) / 2 - 15) * 2))
    if compact_score >= 70:
        compact_label, compact_clr = "Tight Spread", "#22c55e"
    elif compact_score >= 40:
        compact_label, compact_clr = "Moderate Spread", "#fbbf24"
    else:
        compact_label, compact_clr = "Wide Spread", "#ef4444"

    ax_info.text(0.5, 0.61, compact_label, fontsize=14,
                 color=compact_clr, ha='center', fontweight='bold')

    # -- Engagement height by half section --
    ax_info.text(0.5, 0.52, "E N G A G E M E N T   H E I G H T", fontsize=9,
                 color='#777', ha='center', fontweight='bold')
    ax_info.plot([0.15, 0.85], [0.505, 0.505], color='#444', linewidth=0.5)

    ax_info.text(0.25, 0.46, "1st Half", fontsize=11,
                 color='#22c55e', ha='center', fontweight='600')
    ax_info.text(0.25, 0.42, f"{h1_meters:.0f}m", fontsize=22,
                 color='#22c55e', ha='center', fontweight='bold')

    ax_info.text(0.75, 0.46, "2nd Half", fontsize=11,
                 color='#f97316', ha='center', fontweight='600')
    ax_info.text(0.75, 0.42, f"{h2_meters:.0f}m", fontsize=22,
                 color='#f97316', ha='center', fontweight='bold')

    # Shift arrow
    diff_m = h2_meters - h1_meters
    if abs(diff_m) > 0.5:
        arrow = "▲" if diff_m > 0 else "▼"
        shift_text = f"{arrow} {abs(diff_m):.1f}m {'engaged higher' if diff_m > 0 else 'engaged deeper'}"
        shift_clr = '#22c55e' if diff_m > 0 else '#ef4444'
    else:
        shift_text = "≈ Stable"
        shift_clr = '#aaa'

    ax_info.text(0.5, 0.36, shift_text, fontsize=12,
                 color=shift_clr, ha='center', fontweight='bold')

    # -- Action Breakdown section --
    ax_info.text(0.5, 0.27, "A C T I O N S", fontsize=9,
                 color='#777', ha='center', fontweight='bold')
    ax_info.plot([0.15, 0.85], [0.255, 0.255], color='#444', linewidth=0.5)

    total_actions = len(defensive_df)
    n_recoveries = len(recoveries)
    n_challenges = len(challenges)

    ax_info.text(0.5, 0.21, f"{total_actions}", fontsize=28,
                 color='white', ha='center', fontweight='bold')
    ax_info.text(0.5, 0.18, "total defensive actions", fontsize=8,
                 color='#888', ha='center')

    # Mini bar for recovery vs challenge ratio
    if total_actions > 0:
        rec_pct = n_recoveries / total_actions
        chall_pct = n_challenges / total_actions

        bar_y = 0.13
        bar_h = 0.02
        bar_l = 0.12
        bar_w = 0.76

        # Background bar
        ax_info.add_patch(patches.FancyBboxPatch(
            (bar_l, bar_y), bar_w, bar_h,
            boxstyle='round,pad=0.003', facecolor='#1a1a1a',
            edgecolor='none'))

        # Recovery portion (green)
        if rec_pct > 0:
            ax_info.add_patch(patches.FancyBboxPatch(
                (bar_l, bar_y), bar_w * rec_pct, bar_h,
                boxstyle='round,pad=0.003', facecolor='#22c55e',
                edgecolor='none', alpha=0.8))

        # Challenge portion (orange) — starts after recoveries
        if chall_pct > 0:
            ax_info.add_patch(patches.FancyBboxPatch(
                (bar_l + bar_w * rec_pct, bar_y), bar_w * chall_pct, bar_h,
                boxstyle='round,pad=0.003', facecolor='#f97316',
                edgecolor='none', alpha=0.8))

        ax_info.text(0.2, 0.09, f"Recoveries {n_recoveries}", fontsize=9,
                     color='#22c55e', ha='center')
        ax_info.text(0.5, 0.09, f"Challenges {n_challenges}", fontsize=9,
                     color='#f97316', ha='center')
        other = total_actions - n_recoveries - n_challenges
        ax_info.text(0.8, 0.09, f"Other {other}", fontsize=9,
                     color='#aaa', ha='center')

    # 1H / 2H action counts
    h1_count = len(h1_df)
    h2_count = len(h2_df)
    ax_info.text(0.5, 0.04, f"1H: {h1_count}   •   2H: {h2_count}",
                 fontsize=10, color='#666', ha='center')

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_progressive_pass_map(df, team_name):
    """
    Single-pitch progressive pass map with passes colour-coded by origin zone.
    Zone bands are shaded on the pitch; a sidebar shows per-zone counts and
    top progressors coloured by their dominant zone.
    """
    from mplsoccer import Pitch as MplPitch
    from matplotlib import gridspec
    from matplotlib.patches import Patch

    df = preprocess_for_network(df)

    df['pro'] = np.where(
        df['end_x_scaled'].notna(),
        np.sqrt((120 - df['x_scaled'])**2 + (40 - df['y_scaled'])**2) -
        np.sqrt((120 - df['end_x_scaled'])**2 + (40 - df['end_y_scaled'])**2),
        0
    )
    mask = (
        (df['team_name'] == team_name) &
        (df['pro'] >= 9.144) &
        (df['x_scaled'].between(40, 115))
    )
    for col in ('cross', 'Cross'):
        if col in df.columns:
            mask = mask & (~df[col].astype(str).isin(['True', 'Cross']))

    dfpro = df[mask].copy()
    pro_count = len(dfpro)

    if pro_count == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Progressive Passes Found",
                color=TACTIQ_FG, ha="center", transform=ax.transAxes)
        ax.axis('off')
        return fig_to_base64(fig)

    ZONES = [
        ('Own Half',  '#457b9d', 40,  60),
        ('Mid Third', '#ff9f1c', 60,  80),
        ('Att Third', '#e63946', 80, 115),
    ]
    zone_color_map = {z[0]: z[1] for z in ZONES}

    def origin_zone(x):
        if x < 60:  return 'Own Half'
        if x < 80:  return 'Mid Third'
        return 'Att Third'

    dfpro['zone'] = dfpro['x_scaled'].apply(origin_zone)

    name_col = 'player_name' if 'player_name' in dfpro.columns else None
    player_counts = None
    if name_col:
        player_counts = (
            dfpro.groupby(name_col).size()
            .reset_index(name='count')
            .sort_values('count', ascending=True)
            .tail(8)
        )
        player_counts['short'] = player_counts[name_col].apply(get_short_name)

        def player_top_zone_color(player):
            sub = dfpro[dfpro[name_col] == player]
            if sub.empty: return '#888'
            return zone_color_map.get(sub['zone'].value_counts().idxmax(), '#888')
        player_counts['bar_color'] = player_counts[name_col].apply(player_top_zone_color)

    clean = (team_name.replace(' Kulübü','').replace(' Spor','')
                      .replace(' Futbol','').strip())

    # ── Layout: pitch (left) + sidebar (right) ──────────────
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.6, 1], wspace=0.04)
    ax_pitch = fig.add_subplot(gs[0])
    ax_side  = fig.add_subplot(gs[1])

    # ── Draw pitch ───────────────────────────────────────────
    pitch = MplPitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                     line_color='#444', linewidth=1.2, corner_arcs=True)
    pitch.draw(ax=ax_pitch)

    # Shade zone bands and draw divider lines
    for zone_name, color, x_lo, x_hi in ZONES:
        ax_pitch.axvspan(x_lo, x_hi, color=color, alpha=0.06, zorder=0)
        ax_pitch.axvline(x_lo, color=color, lw=1.0, alpha=0.35, linestyle='--', zorder=1)
        # Zone label at top of band
        mid_x = (x_lo + x_hi) / 2
        sub_n = len(dfpro[dfpro['zone'] == zone_name])
        pct   = round(sub_n / pro_count * 100) if pro_count else 0
        ax_pitch.text(mid_x, 79, f"{zone_name}\n{sub_n}  ({pct}%)",
                      color=color, fontsize=8, fontweight='bold',
                      ha='center', va='top', zorder=5,
                      bbox=dict(boxstyle='round,pad=0.25', facecolor=TACTIQ_BG,
                                edgecolor=color, alpha=0.7, linewidth=0.8))

    # Draw origin density heatmap using a premium Reds color gradient
    from matplotlib.colors import LinearSegmentedColormap
    prog_cmap = LinearSegmentedColormap.from_list("ProgCmap", [TACTIQ_BG, "#fee2e2", "#f87171", "#dc2626", "#7f1d1d"])
    
    if not dfpro.empty:
        bin_statistic = pitch.bin_statistic(dfpro['x_scaled'], dfpro['y_scaled'], statistic='count', bins=(12, 8), normalize=True)
        pitch.heatmap(bin_statistic, ax=ax_pitch, cmap=prog_cmap, edgecolor='#374151', lw=0.4, zorder=1, alpha=0.88)
        
    # Draw starting progressive line at x=40
    pitch.lines(40, 0, 40, 80, color='white', linestyle='--', linewidth=1.5, ax=ax_pitch, zorder=3)

    # Custom premium legend explaining the color system
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#457b9d', lw=4, label='Own Half (40-60m)'),
        Line2D([0], [0], color='#ff9f1c', lw=4, label='Mid Third (60-80m)'),
        Line2D([0], [0], color='#e63946', lw=4, label='Att Third (80-115m)'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#dc2626', markeredgecolor='none', markersize=7, label='Origin Heatmap'),
    ]
    ax_pitch.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=7.0,
                    framealpha=0.15, facecolor=TACTIQ_BG, edgecolor='#444',
                    labelcolor='white', bbox_to_anchor=(0.5, -0.065))

    ax_pitch.set_title(f"{clean}  ·  Progressive Passes  ·  {pro_count} total",
                       color=TACTIQ_FG, fontsize=11, fontweight='bold', pad=10)

    # ── Sidebar ──────────────────────────────────────────────
    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.set_xlim(0, 1)
    ax_side.axis('off')

    if player_counts is not None and not player_counts.empty:
        ax_bar = ax_side.inset_axes([0.0, 0.0, 1.0, 1.0])
        ax_bar.set_facecolor(TACTIQ_BG)

        y_pos = list(range(len(player_counts)))
        bars = ax_bar.barh(y_pos, player_counts['count'].tolist(),
                           color=player_counts['bar_color'].tolist(),
                           height=0.6, edgecolor='none', alpha=0.88)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(player_counts['short'].tolist())
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 0.15, bar.get_y() + bar.get_height() / 2,
                        str(int(w)), va='center', ha='left',
                        color=TACTIQ_FG, fontsize=9, fontweight='bold')

        ax_bar.set_xlim(0, player_counts['count'].max() + 3)
        ax_bar.tick_params(axis='y', labelsize=8.5, colors=TACTIQ_FG)
        ax_bar.tick_params(axis='x', labelsize=7.5, colors='#555')
        ax_bar.spines[:].set_visible(False)
        ax_bar.xaxis.grid(True, color='#ffffff12', linewidth=0.5)
        ax_bar.set_axisbelow(True)
        ax_bar.set_title('Top Progressors', color=TACTIQ_FG,
                         fontsize=9, fontweight='bold', pad=8)

        legend_handles = [Patch(facecolor=z[1], label=z[0], alpha=0.85) for z in ZONES]
        ax_bar.legend(handles=legend_handles, loc='lower right', fontsize=7.5,
                      facecolor=TACTIQ_BG, edgecolor='#333', labelcolor='white',
                      framealpha=0.7)

    return fig_to_base64(fig)



def plot_pressing_map(df, team_name):
    """
    Defensive Transitions Map — scatter plot showing WHERE the team loses possession (turnovers),
    and a donut chart analyzing the subsequent transition outcomes (what happened after they lost the ball):
    - Recovered Back (won possession back within 10s)
    - Opponent Retained (opponent kept possession for 10s without shooting)
    - Opponent Shot (opponent got a shot within 10s)
    """
    from mplsoccer import Pitch as MplPitch
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    import numpy as np

    df = preprocess_for_network(df)

    # Ensure expanded_minute or abs_time
    if 'expanded_minute' not in df.columns:
        if 'minute' in df.columns and 'second' in df.columns:
            df['abs_time'] = df['minute'] * 60 + df['second']
        else:
            df['abs_time'] = df.index
    else:
        df['abs_time'] = df['expanded_minute'] * 60

    # Define robust possession loss mask (turnovers)
    is_loss = (
        ((df['event'] == 'Pass') & (df['outcome'] == 0)) |
        (df['event'] == 'Dispossessed') |
        (df['event'] == 'Turnover') |
        ((df['event'] == 'TakeOn') & (df['outcome'] == 0))
    )
    if 'type_id' in df.columns:
        is_loss = is_loss | (
            ((df['type_id'] == 1) & (df['outcome'] == 0)) | # failed pass
            ((df['type_id'] == 3) & (df['outcome'] == 0))   # failed take-on
        )
    loss_df = df[(df['team_name'] == team_name) & is_loss & df['x_scaled'].notna()].copy()

    clean = team_name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()

    if loss_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Defensive Transition Data", color=TACTIQ_FG, ha='center', transform=ax.transAxes)
        ax.axis('off')
        return fig_to_base64(fig)

    total_losses = len(loss_df)

    # Analyze 10s window transition outcomes for losses
    shot_count = 0
    retained_count = 0
    recovery_count = 0

    loss_df['_outcome'] = 'Opponent Retained'

    for idx, row in loss_df.iterrows():
        match_id = row['match_id']
        start_time = row['abs_time']
        
        next_events = df[
            (df['match_id'] == match_id) & 
            (df['abs_time'] > start_time) & 
            (df['abs_time'] <= start_time + 10)
        ].sort_values('abs_time')
        
        outcome_found = False
        has_our_recovery = False
        
        for _, ev in next_events.iterrows():
            if ev['team_name'] != team_name:
                # If opponent got a shot/goal in 10s:
                if ev['event'] in ['Shot', 'Goal', 'Missed', 'Saved', 'Attempt Saved', 'SavedShot'] or ev['type_id'] in [13, 14, 15, 16]:
                    shot_count += 1
                    loss_df.at[idx, '_outcome'] = 'Opponent Shot'
                    outcome_found = True
                    break
            else:
                # If we won it back (Tackle, Interception, BallRecovery) in 10s:
                if ev['event'] in ['Tackle', 'Interception', 'BallRecovery', 'Ball recovery'] or ev['type_id'] in [7, 8, 49]:
                    has_our_recovery = True
                    
        if not outcome_found:
            if has_our_recovery:
                recovery_count += 1
                loss_df.at[idx, '_outcome'] = 'Recovered'
            else:
                retained_count += 1
                loss_df.at[idx, '_outcome'] = 'Opponent Retained'

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.1], wspace=0.06)
    ax_pitch = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    pitch = MplPitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                     line_color='#444', linewidth=1.2, corner_arcs=True)
    pitch.draw(ax=ax_pitch)

    # Zone definitions (thirds)
    ZONE_DEFS = [('Def Third', '#22c55e', 0, 40), ('Mid Third', '#fbbf24', 40, 80), ('Att Third', '#ef4444', 80, 120)]

    for z_name, z_col, x0, x1 in ZONE_DEFS:
        ax_pitch.axvspan(x0, x1, color=z_col, alpha=0.04, zorder=0)
        ax_pitch.axvline(x0, color=z_col, lw=0.8, alpha=0.18, linestyle='--', zorder=1)
        cnt = len(loss_df[(loss_df['x_scaled'] >= x0) & (loss_df['x_scaled'] < x1)])
        pct = round(cnt / total_losses * 100) if total_losses else 0
        ax_pitch.text((x0 + x1) / 2, 79, f"{z_name}\n{cnt} losses ({pct}%)",
                      color=z_col, fontsize=8, fontweight='bold', ha='center', va='top', zorder=6,
                      bbox=dict(boxstyle='round,pad=0.25', facecolor=TACTIQ_BG,
                                edgecolor=z_col, alpha=0.75, linewidth=0.8))

    # Plot turnovers colored by outcome
    OUTCOME_COLORS = {
        'Recovered':          '#22c55e', # Green (Won ball back in 10s!)
        'Opponent Retained':  '#3b82f6', # Blue (Opponent retained possession)
        'Opponent Shot':      '#ef4444', # Red (Danger! Opponent shot in 10s)
    }

    for outcome, color in OUTCOME_COLORS.items():
        sub = loss_df[loss_df['_outcome'] == outcome]
        if not sub.empty:
            if outcome == 'Opponent Shot':
                # Plot high transition risk as prominent diamond symbols
                ax_pitch.scatter(sub['x_scaled'], sub['y_scaled'],
                                 c=color, marker='D', s=65, alpha=0.85, zorder=4,
                                 edgecolors='white', linewidths=0.5)
            else:
                ax_pitch.scatter(sub['x_scaled'], sub['y_scaled'],
                                 c=color, s=55, alpha=0.72, zorder=4,
                                 edgecolors='none')

    ax_pitch.set_title(f"{clean}  ·  Defensive Transitions  ·  {total_losses} turnovers",
                       color=TACTIQ_FG, fontsize=11, fontweight='bold', pad=10)

    # Legend for outcomes
    handles = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#22c55e', markersize=8, label='Recovered (Won Back in 10s)'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#3b82f6', markersize=8, label='Opponent Retained'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='#ef4444', markeredgecolor='white', markersize=7, label='Opponent Shot (Danger!)'),
    ]
    ax_pitch.legend(handles=handles, loc='lower left', fontsize=7.5,
                    facecolor=TACTIQ_BG, edgecolor='#333', labelcolor='white', framealpha=0.7)

    # Sidebar: outcomes donut + zone breakdown
    ax_bar.set_facecolor(TACTIQ_BG)
    ax_bar.axis('off')

    p_shot = round(shot_count / total_losses * 100, 1) if total_losses else 0
    p_retained = round(retained_count / total_losses * 100, 1) if total_losses else 0
    p_recovered = round(recovery_count / total_losses * 100, 1) if total_losses else 0

    percentages = [p_shot, p_retained, p_recovered]
    colors = ['#ef4444', '#3b82f6', '#22c55e'] # Opponent Shot (Red), Opponent Retained (Blue), Recovered (Green)
    counts = [shot_count, retained_count, recovery_count]

    pie_pcts = []
    pie_colors = []
    pie_labels = []

    label_map = {0: "Opp. Shot", 1: "Opp. Retained", 2: "Recovered"}
    for idx, p in enumerate(percentages):
        if counts[idx] > 0:
            pie_pcts.append(p if p > 0.05 else 0.05)
            pie_colors.append(colors[idx])
            display_pct = f"{p:.1f}" if p >= 0.1 else "<0.1"
            pie_labels.append(f"{label_map[idx]}\n{display_pct}%")

    ax_donut = ax_bar.inset_axes([0.0, 0.46, 1.0, 0.48])
    ax_donut.set_facecolor(TACTIQ_BG)
    ax_donut.axis('off')

    if pie_pcts:
        ax_donut.pie(
            pie_pcts,
            labels=pie_labels,
            colors=pie_colors,
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor=TACTIQ_BG, linewidth=3),
            textprops=dict(color='white', fontsize=8.0, fontweight='bold'),
            center=(0, 0.0)
        )
    ax_donut.set_aspect('equal')

    # Center text overlays inside the donut hole
    ax_donut.text(0, 0.0, f"{total_losses}\nlosses", color='white', fontsize=11, fontweight='bold', ha='center', va='center')
    ax_donut.text(0, 1.28, "Defensive Transition Outcomes", color='white', fontsize=11, fontweight='bold', ha='center', va='center')

    # Horizontal bar chart for losses by zone
    ax_zone = ax_bar.inset_axes([0.08, 0.06, 0.86, 0.28])
    ax_zone.set_facecolor(TACTIQ_BG)

    zone_names = ['Def Third', 'Mid Third', 'Att Third']
    zone_counts = [len(loss_df[(loss_df['x_scaled'] >= z[2]) & (loss_df['x_scaled'] < z[3])]) for z in ZONE_DEFS]
    zone_colors = [z[1] for z in ZONE_DEFS]

    y_pos = np.arange(len(zone_names))
    bars = ax_zone.barh(y_pos, zone_counts, color=zone_colors, height=0.5, alpha=0.85, edgecolor='none')
    for bar, cnt in zip(bars, zone_counts):
        ax_zone.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(cnt), va='center', ha='left', color=TACTIQ_FG, fontsize=9, fontweight='bold')

    ax_zone.set_xlim(0, max(zone_counts) + max(2, int(max(zone_counts) * 0.15)))
    ax_zone.spines[:].set_visible(False)
    ax_zone.set_yticks(y_pos)
    ax_zone.set_yticklabels(zone_names)
    ax_zone.tick_params(axis='y', labelsize=8.5, colors=TACTIQ_FG)
    ax_zone.tick_params(axis='x', colors='#555', labelsize=8)
    ax_zone.xaxis.grid(True, color='#ffffff12', linewidth=0.5)
    ax_zone.set_axisbelow(True)
    ax_zone.set_title('Losses by Zone', color=TACTIQ_FG, fontsize=9.5, fontweight='bold', pad=6)

    return fig_to_base64(fig)


def plot_offensive_transition_map(df, team_name):
    """
    Offensive Transitions — scatter recovery dots showing where possession was gained,
    with a 3x3 zone count/share overlay, and a sidebar outcome analysis in a 10s window.
    """
    from mplsoccer import Pitch as MplPitch
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    df = preprocess_for_network(df)

    # Ensure expanded_minute or abs_time
    if 'expanded_minute' not in df.columns:
        if 'minute' in df.columns and 'second' in df.columns:
            df['abs_time'] = df['minute'] * 60 + df['second']
        else:
            df['abs_time'] = df.index
    else:
        df['abs_time'] = df['expanded_minute'] * 60

    # Filter ball gains (Tackle outcome=1, Interception, BallRecovery)
    is_gain = (
        (df['event'] == 'BallRecovery') | 
        (df['event'] == 'Ball recovery') |
        (df['event'] == 'Interception') |
        ((df['event'] == 'Tackle') & (df['outcome'] == 1)) |
        (df['type_id'] == 49) |
        (df['type_id'] == 8) |
        ((df['type_id'] == 7) & (df['outcome'] == 1))
    )
    df_gains = df[(df['team_name'] == team_name) & is_gain].copy()
    total_gains = len(df_gains)

    clean = team_name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()

    if total_gains == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Transition Gains Recorded", color=TACTIQ_FG, ha='center', transform=ax.transAxes)
        ax.axis('off')
        return fig_to_base64(fig)

    # Analyze "What Happened" - updated to 10s window
    shot_count = 0
    retained_count = 0
    loss_count = 0

    for idx, row in df_gains.iterrows():
        match_id = row['match_id']
        start_time = row['abs_time']
        
        next_events = df[
            (df['match_id'] == match_id) & 
            (df['abs_time'] > start_time) & 
            (df['abs_time'] <= start_time + 10)
        ].sort_values('abs_time')
        
        outcome_found = False
        has_successful_pass = False
        
        for _, ev in next_events.iterrows():
            if ev['team_name'] != team_name:
                if ev['event'] in ['Shot', 'Goal', 'Missed', 'Saved']:
                    break
                continue
            if ev['event'] in ['Shot', 'Goal', 'Missed', 'Saved', 'Attempt Saved', 'SavedShot'] or ev['type_id'] in [13, 14, 15, 16]:
                shot_count += 1
                outcome_found = True
                break
            if ev['event'] == 'Pass' and ev['outcome'] == 1:
                has_successful_pass = True

        if not outcome_found:
            if has_successful_pass:
                retained_count += 1
            else:
                loss_count += 1

    # Map recovery events to types for dot coloring
    ACTION_COLORS = {
        'Tackle':        '#ef4444',
        'Interception':  '#3b82f6',
        'BallRecovery':  '#22c55e',
    }
    def get_action_type(r):
        ev = str(r.get('event', ''))
        tid = r.get('type_id')
        if 'tackle' in ev.lower() or tid == 7:
            return 'Tackle'
        elif 'interception' in ev.lower() or tid == 8:
            return 'Interception'
        else:
            return 'BallRecovery'
            
    df_gains['_action'] = df_gains.apply(get_action_type, axis=1)

    # 9-zone counts: x split at 40, 80; y split at 26.67, 53.33
    zone_limits = [
        ('Def Left', 0, 40, 53.33, 80),
        ('Def Center', 0, 40, 26.67, 53.33),
        ('Def Right', 0, 40, 0, 26.67),
        ('Mid Left', 40, 80, 53.33, 80),
        ('Mid Center', 40, 80, 26.67, 53.33),
        ('Mid Right', 40, 80, 0, 26.67),
        ('Att Left', 80, 120, 53.33, 80),
        ('Att Center', 80, 120, 26.67, 53.33),
        ('Att Right', 80, 120, 0, 26.67)
    ]
    
    zone_counts = {}
    for z_name, x0, x1, y0, y1 in zone_limits:
        cnt = len(df_gains[
            (df_gains['x_scaled'] >= x0) & (df_gains['x_scaled'] < x1) &
            (df_gains['y_scaled'] >= y0) & (df_gains['y_scaled'] < y1)
        ])
        zone_counts[z_name] = cnt

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.1], wspace=0.06)
    ax_pitch = fig.add_subplot(gs[0])
    ax_side  = fig.add_subplot(gs[1])

    pitch = MplPitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG,
                     line_color='#555', linewidth=1.2, corner_arcs=True)
    pitch.draw(ax=ax_pitch)

    # Grid lines
    ax_pitch.axvline(40, color='white', linestyle=':', alpha=0.35, linewidth=1.2, zorder=2)
    ax_pitch.axvline(80, color='white', linestyle=':', alpha=0.35, linewidth=1.2, zorder=2)
    ax_pitch.axhline(26.67, color='white', linestyle=':', alpha=0.35, linewidth=1.2, zorder=2)
    ax_pitch.axhline(53.33, color='white', linestyle=':', alpha=0.35, linewidth=1.2, zorder=2)
    
    path_eff = [path_effects.Stroke(linewidth=2.5, foreground=TACTIQ_BG), path_effects.Normal()]
    
    # Plot individual recovery scatter dots
    for action, color in ACTION_COLORS.items():
        sub = df_gains[df_gains['_action'] == action]
        if not sub.empty:
            ax_pitch.scatter(sub['x_scaled'], sub['y_scaled'],
                             c=color, s=55, alpha=0.75, zorder=4,
                             edgecolors='none')

    # Display zone labels faintly inside grid (text overlays)
    for z_name, x0, x1, y0, y1 in zone_limits:
        cnt = zone_counts[z_name]
        pct = round(cnt / total_gains * 100) if total_gains else 0
        
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        
        # Add a very small, extremely faint background rect to text to ensure readability over scatter dots
        rect_txt = Rectangle((cx - 7, cy - 4), 14, 8, facecolor=TACTIQ_BG, alpha=0.55, edgecolor='none', zorder=3)
        ax_pitch.add_patch(rect_txt)
        
        # Format percentage text
        ax_pitch.text(cx, cy + 0.5, f"{pct}%", color='#ddd', fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=5, path_effects=path_eff)
        # Display gain count below percentage
        ax_pitch.text(cx, cy - 2.2, f"{cnt} gains", color='#aaa', fontsize=6.0, ha='center', va='center', zorder=5, path_effects=path_eff)

    ax_pitch.set_title(f"{clean}  ·  Transition Recovery Map (9 Zones)  ·  {total_gains} total gains",
                       color=TACTIQ_FG, fontsize=11, fontweight='bold', pad=10)

    # Event Legend
    handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=c,
                      markersize=7, label=a) for a, c in ACTION_COLORS.items()]
    ax_pitch.legend(handles=handles, loc='lower left', fontsize=7.5,
                    facecolor=TACTIQ_BG, edgecolor='#333', labelcolor='white', framealpha=0.7)

    # Split sidebar into two inset subplots for Outcomes Donut + Zone breakdown symmetry
    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.axis('off')

    p_shot = round(shot_count / total_gains * 100, 1) if total_gains else 0
    p_retained = round(retained_count / total_gains * 100, 1) if total_gains else 0
    p_lost = round(loss_count / total_gains * 100, 1) if total_gains else 0

    percentages = [p_shot, p_retained, p_lost]
    colors = ['#22c55e', '#3b82f6', '#f97316']
    counts = [shot_count, retained_count, loss_count]
    
    # Filter out categories with 0 count to ensure clean rendering
    pie_pcts = []
    pie_colors = []
    pie_labels = []
    
    label_map = {0: "Shot", 1: "Retained", 2: "Lost"}
    for idx, p in enumerate(percentages):
        if counts[idx] > 0:
            pie_pcts.append(p if p > 0.05 else 0.05)
            pie_colors.append(colors[idx])
            display_pct = f"{p:.1f}" if p >= 0.1 else "<0.1"
            pie_labels.append(f"{label_map[idx]}\n{display_pct}%")
            
    # Draw beautiful donut chart on the top half subplot
    ax_donut = ax_side.inset_axes([0.0, 0.46, 1.0, 0.48])
    ax_donut.set_facecolor(TACTIQ_BG)
    ax_donut.axis('off')

    wedges, texts = ax_donut.pie(
        pie_pcts,
        labels=pie_labels,
        colors=pie_colors,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor=TACTIQ_BG, linewidth=3),
        textprops=dict(color='white', fontsize=8.0, fontweight='bold'),
        center=(0, 0.0)
    )
    
    ax_donut.set_aspect('equal')
    
    # Center text overlays inside the donut hole
    ax_donut.text(0, 0.0, f"{total_gains}\ngains", color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    ax_donut.text(0, 1.28, "Transition Outcomes (10s)", color='white', fontsize=11, fontweight='bold', ha='center', va='center')

    # Draw horizontal bar chart on the bottom half subplot
    ax_zone = ax_side.inset_axes([0.08, 0.06, 0.86, 0.28])
    ax_zone.set_facecolor(TACTIQ_BG)

    t_def = len(df_gains[df_gains['x_scaled'] < 40])
    t_mid = len(df_gains[(df_gains['x_scaled'] >= 40) & (df_gains['x_scaled'] < 80)])
    t_att = len(df_gains[df_gains['x_scaled'] >= 80])
    
    zone_names = ['Def Third', 'Mid Third', 'Att Third']
    zone_counts = [t_def, t_mid, t_att]
    zone_colors = ['#22c55e', '#fbbf24', '#ef4444']
    
    import numpy as np
    y_pos = np.arange(len(zone_names))
    bars = ax_zone.barh(y_pos, zone_counts, color=zone_colors, height=0.5, alpha=0.85, edgecolor='none')
    for bar, cnt in zip(bars, zone_counts):
        ax_zone.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(cnt), va='center', ha='left', color=TACTIQ_FG, fontsize=9, fontweight='bold')
                     
    ax_zone.set_xlim(0, max(zone_counts) + max(2, int(max(zone_counts) * 0.15)))
    ax_zone.spines[:].set_visible(False)
    ax_zone.set_yticks(y_pos)
    ax_zone.set_yticklabels(zone_names)
    ax_zone.tick_params(axis='y', labelsize=8.5, colors=TACTIQ_FG)
    ax_zone.tick_params(axis='x', colors='#555', labelsize=8)
    ax_zone.xaxis.grid(True, color='#ffffff12', linewidth=0.5)
    ax_zone.set_axisbelow(True)
    ax_zone.set_title('Gains by Zone', color=TACTIQ_FG, fontsize=9.5, fontweight='bold', pad=6)

    return fig_to_base64(fig)


def plot_match_shot_map(df, home_team, away_team):
    from matplotlib import gridspec
    match_week = df['week'].iloc[0] if 'week' in df.columns and not df.empty else None

    df = preprocess_for_network(df)
    shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']

    if 'event' in df.columns:
        Shotsdf = df[df['event'].isin(shot_types)].copy()
        Shotsdf['typeId'] = Shotsdf['event']
    else:
        Shotsdf = pd.DataFrame()

    if Shotsdf.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Shot Data Found", color=TACTIQ_FG, ha="center", transform=ax.transAxes)
        return fig_to_base64(fig)

    hShotsdf = Shotsdf[Shotsdf['team_name'] == home_team].copy()
    aShotsdf = Shotsdf[Shotsdf['team_name'] == away_team].copy()

    from utils.wyscout_loader import get_wyscout_match_stats
    ws = None
    if match_week != 34:
        try:
            ws = get_wyscout_match_stats(home_team, away_team)
        except Exception:
            pass

    wyscout_hxg = ws['home'].get('xg_for') if ws else None
    wyscout_axg = ws['away'].get('xg_for') if ws else None

    # Calculate model sums
    xg_col = next((c for c in ['xG', 'expectedGoals', 'xg'] if c in Shotsdf.columns), None)
    h_model_sum = hShotsdf[xg_col].fillna(0).sum() if xg_col else 0
    a_model_sum = aShotsdf[xg_col].fillna(0).sum() if xg_col else 0

    h_scale = (wyscout_hxg / h_model_sum) if (wyscout_hxg is not None and h_model_sum > 0) else 1.0
    a_scale = (wyscout_axg / a_model_sum) if (wyscout_axg is not None and a_model_sum > 0) else 1.0

    def get_xg(row, team):
        val = 0.05
        if xg_col and pd.notna(row.get(xg_col)):
            val = float(row[xg_col])
        if team == home_team:
            return val * h_scale
        else:
            return val * a_scale

    hgoal_count = len(hShotsdf[hShotsdf['typeId'] == 'Goal'])
    agoal_count = len(aShotsdf[aShotsdf['typeId'] == 'Goal'])
    hxg = wyscout_hxg if wyscout_hxg is not None else (round(h_model_sum, 2) if xg_col else 0)
    axg = wyscout_axg if wyscout_axg is not None else (round(a_model_sum, 2) if xg_col else 0)
    hTotalShots = len(hShotsdf)
    aTotalShots = len(aShotsdf)
    hShotsOnT = len(hShotsdf[hShotsdf['typeId'].isin(['Attempt Saved', 'Saved Shot', 'Goal'])])
    aShotsOnT = len(aShotsdf[aShotsdf['typeId'].isin(['Attempt Saved', 'Saved Shot', 'Goal'])])

    hShotsdf['dist'] = np.sqrt((hShotsdf['x_scaled'] - 120)**2 + (hShotsdf['y_scaled'] - 40)**2)
    aShotsdf['dist'] = np.sqrt((aShotsdf['x_scaled'] - 120)**2 + (aShotsdf['y_scaled'] - 40)**2)
    home_avg_dist = round(hShotsdf['dist'].mean(), 1) if not hShotsdf.empty else 0
    away_avg_dist = round(aShotsdf['dist'].mean(), 1) if not aShotsdf.empty else 0

    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY

    # ── Visual encoding (used consistently on pitch, goal mouth, timeline) ──
    # filled + team-col  = on target (saved)
    # filled + gold      = goal
    # hollow + orange    = post
    # hollow + team-col  = miss
    # extra outer ring   = big chance (any type)
    def shot_size(xg_val, typeId):
        base = 200 if typeId in ('Goal', 'Attempt Saved', 'Saved Shot') else 100
        return max(base, min(800, xg_val * 1400 + base))

    # ── Figure layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(TACTIQ_BG)

    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 1], hspace=0.14)
    gs_top   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0],
                                                width_ratios=[2.4, 1], wspace=0.10)
    gs_goals = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[1], hspace=0.38)

    ax_pitch     = fig.add_subplot(gs_top[0])
    ax_home_goal = fig.add_subplot(gs_goals[0])
    ax_away_goal = fig.add_subplot(gs_goals[1])
    ax_timeline  = fig.add_subplot(gs_outer[1])

    # ── Main pitch ──────────────────────────────────────────────
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True,
                  pitch_color=TACTIQ_BG, linewidth=1.5, line_color='#555')
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_ylim(-0.5, 80.5)
    ax_pitch.set_xlim(-0.5, 120.5)

    def _plot_shot(ax_p, x_plot, y_plot, typeId, xg_val, team_col, minute, is_big_chance):
        s = shot_size(xg_val, typeId)

        # Big chance: single bold outer ring (drawn first, stays behind)
        if is_big_chance:
            ax_p.scatter(x_plot, y_plot, s=s * 3.2, color='none',
                         edgecolors='#facc15', linewidths=1.8, zorder=2, alpha=0.55)

        if typeId == 'Goal':
            # Soft glow then filled gold circle
            ax_p.scatter(x_plot, y_plot, s=s * 2.0, color='#fbbf24',
                         edgecolors='none', zorder=3, alpha=0.18)
            ax_p.scatter(x_plot, y_plot, s=s, color='#fbbf24',
                         edgecolors='white', linewidths=1.8, zorder=5)
        elif typeId in ('Attempt Saved', 'Saved Shot'):
            # Filled team-color circle
            ax_p.scatter(x_plot, y_plot, s=s, color=team_col,
                         edgecolors='white', linewidths=1.2, zorder=4, alpha=0.85)
        elif typeId == 'Post':
            # Hollow orange diamond
            ax_p.scatter(x_plot, y_plot, s=s, marker='D', color='none',
                         edgecolors='#f97316', linewidths=2.2, zorder=4)
        else:  # Miss — small hollow circle, transparent
            ax_p.scatter(x_plot, y_plot, s=max(60, s * 0.45), color='none',
                         edgecolors=team_col, linewidths=1.2, zorder=3, alpha=0.4)

        # Minute label — small text above the marker
        if pd.notna(minute):
            r = np.sqrt(s) * 0.030 + 1.0
            col_txt = '#fbbf24' if typeId == 'Goal' else (
                      '#f97316' if typeId == 'Post' else team_col)
            ax_p.text(x_plot, y_plot + r + 0.8, f"{int(minute)}'",
                      color=col_txt, fontsize=5.5, ha='center', va='bottom',
                      zorder=7, fontweight='bold')

    for _, shot in hShotsdf.iterrows():
        _plot_shot(ax_pitch, 120 - shot['x_scaled'], 80 - shot['y_scaled'],
                   shot['typeId'], get_xg(shot, home_team), hcol,
                   shot.get('time_min'), str(shot.get('Big Chance', '')).lower() in ('si', 'yes', '1'))

    for _, shot in aShotsdf.iterrows():
        _plot_shot(ax_pitch, shot['x_scaled'], shot['y_scaled'],
                   shot['typeId'], get_xg(shot, away_team), acol,
                   shot.get('time_min'), str(shot.get('Big Chance', '')).lower() in ('si', 'yes', '1'))

    # Stats butterfly
    labels   = ["Goals", "xG", "Shots", "On Target", "Avg.Dist."]
    values_h = [hgoal_count, hxg, hTotalShots, hShotsOnT, home_avg_dist]
    values_a = [agoal_count, axg, aTotalShots, aShotsOnT, away_avg_dist]
    y_positions = [65 - i * 8 for i in range(len(labels))]
    for lab, vh, va, y in zip(labels, values_h, values_a, y_positions):
        total  = (vh + va) or 1
        norm_h = (vh / total) * 12
        norm_a = (va / total) * 12
        ax_pitch.barh(y, norm_h, height=3.5, left=57 - norm_h, color=hcol, alpha=0.75)
        ax_pitch.barh(y, norm_a, height=3.5, left=63,           color=acol, alpha=0.75)
        ax_pitch.text(60, y, lab,    color='white', fontsize=8.5, ha='center', va='center', fontweight='bold', zorder=5)
        ax_pitch.text(57 - norm_h - 1.8, y, str(vh), color='white', fontsize=11, ha='right', va='center', fontweight='bold')
        ax_pitch.text(63 + norm_a + 1.8, y, str(va), color='white', fontsize=11, ha='left',  va='center', fontweight='bold')

    def _short(name):
        return name.replace(' Spor Kulübü', '').replace(' Kulübü', '').replace(' Futbol Kulübü', '').strip()

    ax_pitch.text(2,   78, _short(home_team), color=hcol, fontsize=11, ha='left',  fontweight='bold')
    ax_pitch.text(118, 78, _short(away_team), color=acol, fontsize=11, ha='right', fontweight='bold')

    from matplotlib.lines import Line2D
    legend_items = [
        # Exactly matches _plot_shot visual encoding
        Line2D([0],[0], marker='o', color='none', markerfacecolor='#fbbf24',
               markeredgecolor='white', markersize=9, label='Goal'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=hcol,
               markeredgecolor='white', markersize=8, label='On Target (saved)'),
        Line2D([0],[0], marker='D', color='none', markerfacecolor='none',
               markeredgecolor='#f97316', markersize=7, label='Post'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='#888', markersize=6, label='Miss'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor='#facc15', markersize=10, markeredgewidth=1.8,
               label='Big Chance'),
    ]
    ax_pitch.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=6.5,
                    framealpha=0.12, facecolor=TACTIQ_BG, edgecolor='#444',
                    labelcolor='white', bbox_to_anchor=(0.5, -0.01))

    # ── Goal mouth panels ──────────────────────────────────────
    def draw_goal_mouth(ax_g, shots_df, team_col, team_name):
        GW, GH = 100, 45   # goal frame size in axis units
        OX, OY =   0,  0

        ax_g.set_facecolor(TACTIQ_BG)
        ax_g.set_xlim(OX - 6, OX + GW + 6)
        ax_g.set_ylim(OY - 9, OY + GH + 6)
        ax_g.axis('off')

        # Net background (dark green-tinted)
        net_bg = patches.Rectangle((OX, OY), GW, GH,
                                    facecolor='#0c1510', linewidth=0, zorder=1)
        ax_g.add_patch(net_bg)

        # Net lines — vertical
        for nx in np.arange(OX + 8, OX + GW, 8):
            ax_g.plot([nx, nx], [OY, OY + GH],
                      color='#1e3322', linewidth=0.5, zorder=2)
        # Net lines — horizontal
        for ny in np.arange(OY + 7, OY + GH, 7):
            ax_g.plot([OX, OX + GW], [ny, ny],
                      color='#1e3322', linewidth=0.5, zorder=2)

        # Draw the 21 zones defined by the user
        zones = [
            # Inside the goal
            {"label": "Low Left", "x0": 51.8, "x1": 54.8, "y0": 0, "y1": 20,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},
            {"label": "High Left", "x0": 51.8, "x1": 54.8, "y0": 20, "y1": 38,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},
            {"label": "Low Centre", "x0": 48.2, "x1": 51.8, "y0": 0, "y1": 20,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},
            {"label": "High Centre", "x0": 48.2, "x1": 51.8, "y0": 20, "y1": 38,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},
            {"label": "Low Right", "x0": 45.2, "x1": 48.2, "y0": 0, "y1": 20,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},
            {"label": "High Right", "x0": 45.2, "x1": 48.2, "y0": 20, "y1": 38,
             "color": (0.0, 0.4, 1.0, 0.15), "is_goal": True},

            # Immediate borders of the goal
            {"label": "Left", "x0": 54.8, "x1": 55.8, "y0": 0, "y1": 38,
             "color": (1.0, 1.0, 1.0, 0.18), "is_goal": False},
            {"label": "High", "x0": 44.2, "x1": 55.8, "y0": 38, "y1": 42,
             "color": (1.0, 1.0, 1.0, 0.18), "is_goal": False},
            {"label": "Right", "x0": 44.2, "x1": 45.2, "y0": 0, "y1": 38,
             "color": (1.0, 1.0, 1.0, 0.18), "is_goal": False},

            # Near-goal zones ("close")
            {"label": "Close Left", "x0": 55.8, "x1": 59.3, "y0": 0, "y1": 40,
             "color": (0.0, 1.0, 0.5, 0.10), "is_goal": False},
            {"label": "Close High Left", "x0": 55.8, "x1": 59.3, "y0": 40, "y1": 60,
             "color": (0.0, 1.0, 0.5, 0.10), "is_goal": False},
            {"label": "Close Right", "x0": 40.7, "x1": 44.2, "y0": 0, "y1": 40,
             "color": (0.0, 1.0, 0.5, 0.10), "is_goal": False},
            {"label": "Close High Right", "x0": 40.7, "x1": 44.2, "y0": 40, "y1": 60,
             "color": (0.0, 1.0, 0.5, 0.10), "is_goal": False},
            {"label": "Close High", "x0": 44.2, "x1": 55.8, "y0": 42, "y1": 60,
             "color": (0.0, 1.0, 0.5, 0.10), "is_goal": False},

            # Far areas ("far" / field)
            {"label": "Left (far)", "x0": 59.3, "x1": 100, "y0": 0, "y1": 40,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "Right (far)", "x0": 0, "x1": 40.7, "y0": 0, "y1": 40,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "HighLeft Top", "x0": 55.8, "x1": 100, "y0": 60, "y1": 100,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "HighLeft Side", "x0": 59.3, "x1": 100, "y0": 40, "y1": 60,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "HighRight Top", "x0": 0, "x1": 44.2, "y0": 60, "y1": 100,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "HighRight Side", "x0": 0, "x1": 40.7, "y0": 40, "y1": 60,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
            {"label": "High", "x0": 44.2, "x1": 55.8, "y0": 60, "y1": 100,
             "color": (0.0, 1.0, 0.5, 0.05), "is_goal": False},
        ]

        for z in zones:
            # Map Y (x0, x1) to Matplotlib X using the Left-to-Right orientation
            x_rect = OX + ((54.8 - z['x1']) / 9.6) * GW
            w_rect = ((z['x1'] - z['x0']) / 9.6) * GW
            # Map Z (y0, y1) to Matplotlib Y
            y_rect = OY + (z['y0'] / 38.0) * GH
            h_rect = ((z['y1'] - z['y0']) / 38.0) * GH

            # Draw zone rectangle (rendered under net lines at zorder=1.5)
            rect = patches.Rectangle((x_rect, y_rect), w_rect, h_rect,
                                     facecolor=z['color'], edgecolor=(1.0, 1.0, 1.0, 0.1),
                                     linewidth=0.5, zorder=1.5, clip_on=True)
            ax_g.add_patch(rect)

            # Draw zone text annotation
            x_center = x_rect + w_rect / 2.0
            y_center = y_rect + h_rect / 2.0

            if z['is_goal']:
                text_color = '#ffffff'
                alpha_val = 0.85
                weight_val = 'bold'
                font_sz = 5.0
            else:
                text_color = '#9ca3af'
                alpha_val = 0.65
                weight_val = 'normal'
                font_sz = 4.5

            ax_g.text(x_center, y_center, z['label'], color=text_color, alpha=alpha_val,
                      fontsize=font_sz, fontweight=weight_val, ha='center', va='center',
                      zorder=3.5, clip_on=True)

        # Ground / turf line (extends beyond posts)
        ax_g.plot([OX - 4, OX + GW + 4], [OY, OY],
                  color='#3a6b3a', linewidth=3.5, zorder=4, solid_capstyle='round')

        # Posts — thick white with cap
        post_kw = dict(color='white', linewidth=4.5, zorder=6,
                       solid_capstyle='round', solid_joinstyle='round')
        ax_g.plot([OX,      OX],      [OY, OY + GH], **post_kw)   # left post
        ax_g.plot([OX + GW, OX + GW], [OY, OY + GH], **post_kw)   # right post
        ax_g.plot([OX,      OX + GW], [OY + GH, OY + GH], **post_kw)  # crossbar

        # Post depth shadow (simulate 3-D depth on posts)
        ax_g.plot([OX, OX], [OY, OY + GH], color='#888', linewidth=1.5, zorder=5)
        ax_g.plot([OX + GW, OX + GW], [OY, OY + GH], color='#888', linewidth=1.5, zorder=5)

        # ── Plot shots on target ───────────────────────────────
        on_target = shots_df[shots_df['typeId'].isin(
            ['Goal', 'Attempt Saved', 'Saved Shot', 'Post'])].copy()
        on_target = on_target.dropna(
            subset=['Goal Mouth Y Coordinate', 'Goal Mouth Z Coordinate'])

        for _, shot in on_target.iterrows():
            raw_y = float(shot['Goal Mouth Y Coordinate'])
            raw_z = float(shot['Goal Mouth Z Coordinate'])
            # Opta: Y is pitch coordinate where left post=54.8, right post=45.2;
            # Z is absolute height where ground=0, crossbar=38.0.
            # Map Y [45.2, 54.8] to [OX, OX + GW] (goal frame width) with corrected L-R direction
            x_pos = OX + ((54.8 - raw_y) / 9.6) * GW
            # Map Z [0, 38.0] to [OY, OY + GH] (goal frame height)
            y_pos = OY + (raw_z / 38.0) * GH

            # Add premium deterministic jitter to overlapping default coordinates (50.0, 19.0)
            if abs(raw_y - 50.0) < 0.05 and abs(raw_z - 19.0) < 0.05:
                try:
                    seed_val = int(float(shot.get('event_id', 93))) % 1000
                except Exception:
                    seed_val = 93
                rng = np.random.default_rng(seed_val)
                x_pos += rng.normal(0, 1.5)
                y_pos += rng.normal(0, 0.9)

            x_pos = np.clip(x_pos, OX + 1, OX + GW - 1)
            y_pos = np.clip(y_pos, OY + 1, OY + GH - 1)

            is_goal = shot['typeId'] == 'Goal'
            is_post = shot['typeId'] == 'Post'

            if is_goal:
                # Glow layer
                ax_g.scatter(x_pos, y_pos, s=600, color='#fbbf24',
                             edgecolors='none', zorder=7, alpha=0.25)
                ax_g.scatter(x_pos, y_pos, s=280, color='#fbbf24',
                             edgecolors='white', linewidths=2, zorder=8)
                text_col = '#1a1a1a'
            elif is_post:
                # Post hit (orange diamond)
                ax_g.scatter(x_pos, y_pos, s=180, color='none', marker='D',
                             edgecolors='#f97316', linewidths=2.2, zorder=7, alpha=0.9)
                text_col = 'white'
            else:
                ax_g.scatter(x_pos, y_pos, s=180, color='none',
                             edgecolors=team_col, linewidths=2.2, zorder=7, alpha=0.9)
                text_col = 'white'

            jersey = shot.get('Jersey Number')
            if pd.notna(jersey):
                ax_g.text(x_pos, y_pos, str(int(jersey)), color=text_col,
                           fontsize=6, ha='center', va='center', zorder=9,
                           fontweight='bold')

        total_on_target = len(on_target[on_target['typeId'].isin(['Goal', 'Attempt Saved', 'Saved Shot'])])
        label = f'{_short(team_name)}  ·  {total_on_target} shot{"s" if total_on_target != 1 else ""} on target'
        ax_g.text(OX + GW / 2, OY - 6, label, color=team_col, fontsize=7.5,
                  ha='center', va='top', fontweight='bold')

    draw_goal_mouth(ax_home_goal, hShotsdf, hcol, home_team)
    draw_goal_mouth(ax_away_goal, aShotsdf, acol, away_team)

    # ── Shot timeline ──────────────────────────────────────────
    ax_timeline.set_facecolor(TACTIQ_BG)
    ax_timeline.set_xlim(0, 95)
    ax_timeline.set_ylim(-1.8, 1.8)
    ax_timeline.axis('off')

    ax_timeline.axhline(0, color='#3a3a3a', linewidth=1.2)

    for min_mark, lbl in [(45, "45'"), (90, "90'")]:
        ax_timeline.axvline(min_mark, color='#3a3a3a', linewidth=0.8, linestyle='--')
        ax_timeline.text(min_mark, -1.65, lbl, color='#555', fontsize=6.5, ha='center')
    ax_timeline.text(0, -1.65, "0'", color='#555', fontsize=6.5, ha='center')
    ax_timeline.text(47.5, 1.65, 'S H O T   T I M E L I N E', color='#444',
                     fontsize=6.5, ha='center', va='top', fontweight='bold')

    ax_timeline.text(2,  0.9, _short(home_team), color=hcol, fontsize=7, ha='left', fontweight='bold', va='center')
    ax_timeline.text(2, -0.9, _short(away_team), color=acol, fontsize=7, ha='left', fontweight='bold', va='center')

    def _plot_timeline(shots_df, y_base, team_col, team_name):
        sign = np.sign(y_base)
        for _, shot in shots_df.iterrows():
            minute = shot.get('time_min')
            if pd.isna(minute):
                continue
            minute  = float(minute)
            xg_val  = get_xg(shot, team_name)
            typeId  = shot['typeId']
            s_tl    = max(30, min(180, xg_val * 400 + 30))

            if typeId == 'Goal':
                fc, ec, mk = '#fbbf24', 'white', 'o'
            elif typeId in ('Attempt Saved', 'Saved Shot'):
                fc, ec, mk = team_col, 'white', 'o'
            elif typeId == 'Post':
                fc, ec, mk = 'none', '#f97316', 'D'
            else:  # Miss
                fc, ec, mk = 'none', team_col, 'o'
                s_tl = max(20, s_tl * 0.5)

            ax_timeline.scatter(minute, y_base, s=s_tl, color=fc, edgecolors=ec,
                                marker=mk, linewidths=1.4, zorder=4)
            label_y = y_base + (0.28 + xg_val * 0.2) * sign
            ax_timeline.text(minute, label_y, f"{int(minute)}'",
                             color='#fbbf24' if typeId == 'Goal' else team_col,
                             fontsize=5, ha='center',
                             va='bottom' if sign > 0 else 'top', alpha=0.85)

    _plot_timeline(hShotsdf,  0.55, hcol, home_team)
    _plot_timeline(aShotsdf, -0.55, acol, away_team)

    return fig_to_base64(fig)

# --- xT Calculation Logic ---

def calculate_xt(df):
    """
    Calculates Expected Threat (xT) for successful passes.
    Uses a standard 12x8 grid (StatsBomb/Opta hybrid, typically 12 zones x, 8 zones y).
    """
    # Grid (8 rows, 12 columns)
    xt_grid = np.array([
        [0.00638303,0.00779616,0.00844854,0.00977659,0.01126267,0.01248344,0.01473596,0.0174506,0.02122129,0.02756312,0.03485072,0.0379259],
        [0.00750072,0.00878589,0.00942382,0.0105949,0.01214719,0.0138454,0.01611813,0.01870347,0.02401521,0.02953272,0.04066992,0.04647721],
        [0.0088799,0.00977745,0.01001304,0.01110462,0.01269174,0.01429128,0.01685596,0.01935132,0.0241224,0.02855202,0.05491138,0.06442595],
        [0.00941056,0.01082722,0.01016549,0.01132376,0.01262646,0.01484598,0.01689528,0.0199707,0.02385149,0.03511326,0.10805102,0.25745362],
        [0.00941056,0.01082722,0.01016549,0.01132376,0.01262646,0.01484598,0.01689528,0.0199707,0.02385149,0.03511326,0.10805102,0.25745362],
        [0.0088799,0.00977745,0.01001304,0.01110462,0.01269174,0.01429128,0.01685596,0.01935132,0.0241224,0.02855202,0.05491138,0.06442595],
        [0.00750072,0.00878589,0.00942382,0.0105949,0.01214719,0.0138454,0.01611813,0.01870347,0.0241224,0.02953272,0.04066992,0.04647721],
        [0.00638303,0.00779616,0.00844854,0.00977659,0.01126267,0.01248344,0.01473596,0.0174506,0.02122129,0.02756312,0.03485072,0.0379259]
    ])
    
    rows, cols = xt_grid.shape
    
    # Preprocess DF
    try:
        df_xt = preprocess_for_network(df)
    except Exception as e:
        logger.warning("preprocess_for_network failed: %s", e)
        return pd.DataFrame()
    
    # Filter for Successful Passes
    if 'event' in df_xt.columns:
         mask = (df_xt['event'] == 'Pass')
         if 'outcome' in df_xt.columns:
             mask = mask & (df_xt['outcome'].astype(str).str.contains('1|True|Successful', case=False, na=False))
    elif 'type_id' in df_xt.columns:
         mask = (df_xt['type_id'] == 1)
         if 'outcome' in df_xt.columns:
             mask = mask & (df_xt['outcome'] == 1)
    else:
         return pd.DataFrame()

    df_passes = df_xt[mask].copy()
    
    if df_passes.empty:
        return pd.DataFrame()

    # Binning
    try:
        # Use simple clipping to ensure no out of bounds
        df_passes['x_scaled'] = df_passes['x_scaled'].clip(0, 119.9)
        df_passes['y_scaled'] = df_passes['y_scaled'].clip(0, 79.9)
        df_passes['end_x_scaled'] = df_passes['end_x_scaled'].clip(0, 119.9)
        df_passes['end_y_scaled'] = df_passes['end_y_scaled'].clip(0, 79.9)

        df_passes['x_bin'] = pd.cut(df_passes['x_scaled'], bins=cols, labels=False).fillna(0).astype(int)
        df_passes['y_bin'] = pd.cut(df_passes['y_scaled'], bins=rows, labels=False).fillna(0).astype(int)
        df_passes['end_x_bin'] = pd.cut(df_passes['end_x_scaled'], bins=cols, labels=False).fillna(0).astype(int)
        df_passes['end_y_bin'] = pd.cut(df_passes['end_y_scaled'], bins=rows, labels=False).fillna(0).astype(int)
        
        # Calculate xT (Vectorized with advanced NumPy indexing for 2000x+ speedup)
        x_bins = df_passes['x_bin'].clip(0, cols - 1).values
        y_bins = df_passes['y_bin'].clip(0, rows - 1).values
        df_passes['start_xt'] = xt_grid[y_bins, x_bins]

        end_x_bins = df_passes['end_x_bin'].clip(0, cols - 1).values
        end_y_bins = df_passes['end_y_bin'].clip(0, rows - 1).values
        df_passes['end_xt'] = xt_grid[end_y_bins, end_x_bins]
        
        df_passes['xT'] = df_passes['end_xt'] - df_passes['start_xt']
        
        return df_passes
    except Exception as e:
        print(f"Error in xT Calculation: {e}")
        return pd.DataFrame()

def plot_xt_leaders(df, team_name):
    try:
        df_xt = calculate_xt(df)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        
        if df_xt.empty:
             ax.text(0.5, 0.5, "No xT Data", color=TACTIQ_FG, ha="center")
             ax.axis('off')
             return fig_to_base64(fig)
             
        team_xt = df_xt[df_xt['team_name'] == team_name]
        
        if team_xt.empty:
             ax.text(0.5, 0.5, "No xT Data for Team", color=TACTIQ_FG, ha="center")
             ax.axis('off')
             return fig_to_base64(fig)
             
        player_xt = team_xt.groupby('shortName')['xT'].sum().sort_values(ascending=True).tail(10)
        
        if player_xt.empty:
             ax.text(0.5, 0.5, "No xT Leaders", color=TACTIQ_FG, ha="center")
             return fig_to_base64(fig)

        # Plot Bar Chart
        colors = [TACTIQ_HOME for _ in range(len(player_xt))]
        ax.barh(player_xt.index, player_xt.values, color=colors)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(TACTIQ_FG)
        ax.spines['left'].set_color(TACTIQ_FG)
        ax.tick_params(axis='x', colors=TACTIQ_FG)
        ax.tick_params(axis='y', colors=TACTIQ_FG)
        
        ax.set_xlabel("Cumulative xT Generated", color=TACTIQ_FG)
        ax.set_title(f"{team_name} - Top xT Generators", color=TACTIQ_FG, fontweight='bold')
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Error plotting xT: {e}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, str(e), color=TACTIQ_FG, ha='center')
        ax.axis('off')
        return fig_to_base64(fig)

# --- Zone 14 Pass Map ---

def plot_zone14_halfspace_passes(df, team_name):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Filter for Team and Successful Passes
    if 'event' in df.columns:
        mask = (df['team_name'] == team_name) & (df['event'] == 'Pass') & (df['outcome'].astype(str).str.contains('1|True|Successful', case=False, na=False))
    elif 'type_id' in df.columns:
        mask = (df['team_name'] == team_name) & (df['type_id'] == 1) & (df['outcome'] == 1)
    else:
        # Fallback if no clean identifying columns
        return pd.DataFrame()
        
    # User Request: Filter x <= 115 (No corners roughly) - applied to START x
    # Note: user used 'x' which is start x.
    # We use x_scaled. 115 in 120 coords is extremely close to goal line. 
    # Let's assume user logic: df['x'] <= 115.
    
    mask = mask & (df['x_scaled'] <= 115)
    
    df_passes = df[mask].copy()
    
    if df_passes.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Zone 14 Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    col = TACTIQ_HOME # Red for highlighting
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)
    
    # Orient Correctly? 
    # User logic: "if title == ateamName: ax.invert_xaxis() ax.invert_yaxis()"
    # Plotting usually standardizes L->R. User wants to invert for away?
    # Dash app usually shows both L->R. 
    # Let's stick to standard L->R for consistency unless user heavily insists on "mirror" view.
    # Standard view is better for comparison.
    
    z14 = 0
    hs = 0
    
    # Plot Arrows
    for index, row in df_passes.iterrows():
        # Zone 14: end_x [80, 100], end_y [26.66, 53.33] (StatsBomb)
        # 80/3 = 26.66, 160/3 = 53.33
        
        # Check Z14
        if 80 <= row['end_x_scaled'] <= 100 and 26.66 <= row['end_y_scaled'] <= 53.33:
            arrow = patches.FancyArrowPatch((row['x_scaled'], row['y_scaled']), (row['end_x_scaled'], row['end_y_scaled']), 
                                            arrowstyle='->', alpha=0.75, mutation_scale=20, color='orange', linewidth=1.5)
            ax.add_patch(arrow)
            z14 += 1
            
        # Check Half Spaces
        # RHS: 80 <= end_x <= 120 (User logic used 120/100 bounds differently? No, used 80->...).
        # User HS logic: 
        # 1. end_x >= 80, end_y [13.33, 26.66]
        # 2. end_x >= 80, end_y [53.33, 66.67]
        
        elif 80 <= row['end_x_scaled'] and (13.33 <= row['end_y_scaled'] <= 26.66):
             arrow = patches.FancyArrowPatch((row['x_scaled'], row['y_scaled']), (row['end_x_scaled'], row['end_y_scaled']), 
                                            arrowstyle='->', alpha=0.75, mutation_scale=20, color=col, linewidth=1.5)
             ax.add_patch(arrow)
             hs += 1
             
        elif 80 <= row['end_x_scaled'] and (53.33 <= row['end_y_scaled'] <= 66.67):
             arrow = patches.FancyArrowPatch((row['x_scaled'], row['y_scaled']), (row['end_x_scaled'], row['end_y_scaled']), 
                                            arrowstyle='->', alpha=0.75, mutation_scale=20, color=col, linewidth=1.5)
             ax.add_patch(arrow)
             hs += 1

    # Coloring Zones
    # Z14
    y_z14 = [26.66, 26.66, 53.33, 53.33]
    x_z14 = [80, 100, 100, 80]
    ax.fill(x_z14, y_z14, 'orange', alpha=0.2, label='Zone14')
    
    # RHS (Bottom in SB usually? depends on orientation. 0 is top left? No, SB 0,0 is top left usually... wait.)
    # mplsoccer StatsBomb: Y 0-80. 
    # Usually 0 is top.
    # 13.33-26.66 is Top Half Space?
    # 53.33-66.67 is Bottom Half Space?
    # Let's plot both.
    
    y_rhs = [13.33, 13.33, 26.66, 26.66]
    x_rhs = [80, 120, 120, 80]
    ax.fill(x_rhs, y_rhs, col, alpha=0.2, label='HalfSpaces')
    
    y_lhs = [53.33, 53.33, 66.67, 66.67]
    x_lhs = [80, 120, 120, 80]
    ax.fill(x_lhs, y_lhs, col, alpha=0.2, label='HalfSpaces')
    
    # Text Counters (Hexagon markers as bg)
    # User coords: 24, 24/56.
    
    [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
    
    # Half Space Count
    ax.scatter(24, 24, color=col, s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
    ax.text(24, 19, "HalfSp", fontsize=20, color=line_color, ha='center', va='center')
    ax.text(24, 26, f"{hs}", fontsize=40, color=line_color, ha='center', va='center')
    
    # Zone 14 Count
    ax.scatter(24, 56, color='orange', s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
    ax.text(24, 51, "Zone14", fontsize=20, color=line_color, ha='center', va='center')
    ax.text(24, 58, f"{z14}", fontsize=40, color=line_color, ha='center', va='center')
    
    ax.set_title(f"{team_name}\nZone 14 & Half Space Passes", color=TACTIQ_FG, fontsize=20, fontweight='bold')

    return fig_to_base64(fig)

# --- Chance Creating Zone ---

def plot_chance_creating_zone(df, team_name):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Check for Key Pass and Assist
    # If not present, try to derive from 'qualifiers' or similar if possible.
    # User assumes columns 'keyPass' and 'assist' exist (likely boolean or 1/0).
    # If using Opta data without pre-calc, we might miss this.
    # Let's check if columns exist. If not, create them with 0 to safely return empty plot or try best effort.
    
    if 'keyPass' not in df.columns:
        # Try to derive: keyPass often typeId 1 (Pass) and qualifier 210? Not standard in raw DF usually unless processed.
        # If 'qualifiers' column exists and is string, maybe check?
        # For robustness, if missing, we set to 0.
        df['keyPass'] = 0
        if 'qualifiers' in df.columns:
             # Qualifier KeyPass is often specific. Let's assume user data might have 'KeyPass' in qualifiers?
             # Or rely on 'assist' logic if available.
             pass
        
    if 'assist' not in df.columns:
        # Try to derive: Goal Assist is often qualifier 29
        # Check 'goal_assist' column if exists
        if 'goal_assist' in df.columns:
            df['assist'] = df['goal_assist'].fillna(0).astype(bool).astype(int)
        else:
            df['assist'] = 0

    # Filter for Team and Key Pass OR Assist
    # RELAXED: If columns are 0, we might get nothing.
    # Check if we can find anything "creative" (e.g. within box or close to it?)
    # But sticking to user definition: KeyPass or Assist.
    
    mask = (df['team_name'] == team_name) & ((df['keyPass'] == 1) | (df['assist'] == 1))
    
    # DEBUG: If empty, maybe print? No console access easily.
    # Let's trust the data has these columns if user requested it. 
    # If not, the plot "not coming" is expected behavior for empty data.
    # Adding text to explain if empty?
    
    df_chances = df[mask].copy()
    
    if df_chances.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Chance Creation Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    
    # Custom Colormaps using team color? User snippet used linear segmented.
    # Let's use generic or team color. 
    # User snippet: pearl_earring_cmaph = LinearSegmentedColormap.from_list("...", [bg_color, hcol], N=20)
    # We will use Red/Blue standard or User Preferred.
    # Let's use a nice heatmap color.
    
    team_color = TACTIQ_HOME # Default Red
    pearl_earring_cmap = LinearSegmentedColormap.from_list("TeamColor", [bg_color, team_color], N=20)
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True, line_zorder=2)
    pitch.draw(ax=ax)
    
    # Heatmap
    # User snippet: df['end_x']>0 check.
    # User snippet: pitch.bin_statistic(df.x, df.y, bins=(6,5), statistic='count', normalize=False)
    # Note: user used x,y not scaled. We use x_scaled, y_scaled (StatsBomb 120x80).
    # Bins (6,5) implies 120/6 = 20 width, 80/5 = 16 height blocks.
    
    bin_statistic = pitch.bin_statistic(df_chances.x_scaled, df_chances.y_scaled, bins=(6,5), statistic='count', normalize=False)
    pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors='#d9d9d9', alpha=0.5)
    
    # Scatter all points
    pitch.scatter(df_chances.x_scaled, df_chances.y_scaled, c='gray', s=5, ax=ax)
    
    cc = 0
    violet = '#8A2BE2' # Key Pass
    green = '#00FF00'  # Assist
    
    path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]

    for index, row in df_chances.iterrows():
        # Arrow logic
        # User snippet used numeric assist check
        color = green if row['assist'] == 1 else violet
        
        if pd.notna(row['end_x_scaled']) and pd.notna(row['end_y_scaled']):
             arrow = patches.FancyArrowPatch((row['x_scaled'], row['y_scaled']), (row['end_x_scaled'], row['end_y_scaled']), 
                                            arrowstyle='->', mutation_scale=20, color=color, linewidth=1.25, alpha=1)
             ax.add_patch(arrow)
             cc += 1

    # Labels for heatmap bins
    pitch.label_heatmap(bin_statistic, color=line_color, fontsize=15, ax=ax, ha='center', va='center', str_format='{:.0f}', exclude_zeros=True, path_effects=path_eff)
    
    # Legend Text
    ax.text(120, 83.5, "Violet = Key Pass\nGreen = Assist", color=team_color, size=12, ha='right', va='center', path_effects=path_eff)
    ax.text(60, -3, f"Total Chances Created = {cc}", color=team_color, fontsize=15, ha='center', va='center', path_effects=path_eff)
    
    ax.set_title(f"{team_name}\nChance Creating Zone", color=TACTIQ_FG, fontsize=20, fontweight='bold', path_effects=path_eff)
    
    return fig_to_base64(fig)

# --- High Turnover Map ---

def plot_high_turnovers(df, home_team, away_team):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Filter for Ball Recovery or Interception
    # User types: 'Ball recovery', 'Interception'
    # Check if these exist in 'type_id' or 'event'
    # If not, try to map or use generic logic
    
    types = ['Ball recovery', 'Interception', 'Ball Recovery'] # Added variations
    
    if 'event' in df.columns:
        mask = df['event'].isin(types)
    elif 'type_id' in df.columns:
        # Assuming mapping for typical Opta/SB type IDs if known?
        # Opta: 49 (Ball Recovery), 12 (Clearance - maybe not), Interception is usually 1? No 1 is pass.
        # Without map, we might struggle if only type_ids. 
        # But 'preprocess' assigns keys. 
        # Let's hope 'event' (name) column is present (standard in this project seems so).
        mask = df['type_id'].isin([49, 12]) # Speculative fallbacks
    else:
        mask = pd.Series([False] * len(df))
    
    # Also filter by x >= 80 (High up)
    # Using scaled x
    mask = mask & (df['x_scaled'] >= 80)
    
    df_to = df[mask].copy()
    
    if df_to.empty and 'event' not in df.columns:
         # Fallback if names missing: use x >= 80 and logic from defensive block?
         pass
         
    # Home and Away
    home_TO = df_to[df_to['team_name'] == home_team].copy()
    away_TO = df_to[df_to['team_name'] == away_team].copy()
    
    # Distance Filter (<= 47m from opponent goal at 120,40)
    # User calc: ((x - 120)**2 + (y - 40)**2)**0.5 <= 47
    
    if not home_TO.empty:
        home_TO['distance'] = ((home_TO['x_scaled'] - 120)**2 + (home_TO['y_scaled'] - 40)**2)**0.5
        home_TO = home_TO[home_TO['distance'] <= 47]
        
    if not away_TO.empty:
        away_TO['distance'] = ((away_TO['x_scaled'] - 120)**2 + (away_TO['y_scaled'] - 40)**2)**0.5
        away_TO = away_TO[away_TO['distance'] <= 47]
        
    hto_count = len(home_TO)
    ato_count = len(away_TO)
    
    if hto_count == 0 and ato_count == 0:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No High Turnovers Found", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)
    
    # Scatter
    # Home: Invert X (120 - x)
    if not home_TO.empty:
        ax.scatter((120 - home_TO.x_scaled), (80 - home_TO.y_scaled), s=250, c=hcol, edgecolor=line_color, marker='o', linewidth=2, zorder=3)
        
    # Away: Normal X
    if not away_TO.empty:
        ax.scatter(away_TO.x_scaled, away_TO.y_scaled, s=250, c=acol, edgecolor=line_color, marker='o', linewidth=2, zorder=3)
        
    # Circles (47m Radius)
    # Left Circle (Home End in Visual)
    left_circle = plt.Circle((0, 40), 47, color=hcol, fill=True, alpha=0.25, linestyle='dashed', linewidth=3)
    ax.add_artist(left_circle)
    
    # Right Circle (Away End in Visual)
    right_circle = plt.Circle((120, 40), 47, color=acol, fill=True, alpha=0.25, linestyle='dashed', linewidth=3)
    ax.add_artist(right_circle)
    
    # Text
    # Home Text (Left)
    ax.text(0, 82, f"{home_team}\nHigh Turnovers: {hto_count}", color=hcol, size=20, ha='left', va='bottom', fontweight='bold')
    
    # Away Text (Right)
    ax.text(120, 82, f"{away_team}\nHigh Turnovers: {ato_count}", color=acol, size=20, ha='right', va='bottom', fontweight='bold')
    
    # Title Removed per user request
    # ax.set_title("High Turnovers (Recoveries within 47m of Goal)", color=TACTIQ_FG, fontsize=20, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# --- Crosses Visualization ---

def plot_crosses(df, home_team, away_team):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Filter for Crosses
    # User logic: typeId=='Pass' & cross=='Cross'.
    # In standardized data, might be 'cross' boolean column or qualifier.
    
    # Check if 'cross' column exists
    # Make check robust: case insensitive search in columns
    cross_col = None
    for c in df.columns:
        if c.lower() == 'cross':
             cross_col = c
             break
             
    if cross_col:
         # Check for True or 'Cross'
         cross_mask = (df[cross_col].astype(str).str.contains('True|Cross|1', case=False, na=False))
    else:
         # Fallback: maybe qualifier 2? 
         # Or if user data has it pre-calculated.
         # Instead of returning empty immediately, let's relax if 'sub_type' or similar says cross.
         # If 'qualifiers' column exists (list/str), check for 'Cross' text?
         mask_qual = pd.Series([False]*len(df))
         if 'qualifiers' in df.columns:
              mask_qual = df['qualifiers'].astype(str).str.contains('Cross', case=False, na=False)
              
         if not mask_qual.any() and not cross_col:
             return pd.DataFrame()
             
         cross_mask = mask_qual

         
    # Pass check
    if 'event' in df.columns:
        pass_mask = (df['event'] == 'Pass')
    elif 'type_id' in df.columns:
        pass_mask = (df['type_id'] == 1)
    else:
        pass_mask = pd.Series([False] * len(df))
        
    # Combine masks and x limit (<= 119)
    # User used x <= 119 (Start x).
    mask = pass_mask & cross_mask & (df['x_scaled'] <= 119)
    
    df_cross = df[mask].copy()
    
    if df_cross.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Cross Data Found", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    home_cross = df_cross[df_cross['team_name'] == home_team].copy()
    away_cross = df_cross[df_cross['team_name'] == away_team].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)
    
    hsuc = 0
    hunsuc = 0
    asuc = 0
    aunsuc = 0
    
    # Plot Home Crosses (Inverted as per user request to show LEFT -> RIGHT vs RIGHT -> LEFT comparison?)
    # User: 120-x, 80-y. This mirrors the pitch content.
    # Home attacks Left->Right normally (0->120).
    # Mirroring puts them Right->Left (120->0).
    # Actually, 0->120 is Left->Right.
    # 120-x maps 0 to 120, 120 to 0.
    # So if Home attacks Left->Right, mirroring makes them appear to attack Right->Left.
    # Away attacks Left->Right (in data).
    # So both would appear to attack Right->Left? 
    # Or does Away KEEP 0->120?
    # User: Away uses x, y.
    # So Home: Right->Left. Away: Left->Right.
    # They face each other?
    # No, arrows point to goal. 
    # Left->Right arrow (0,40 -> 100,40).
    # Inverted: (120,40 -> 20,40). Right->Left arrow.
    # So Home attacks Right->Left (towards 0).
    # Away attacks Left->Right (towards 120).
    # This creates a "meeting in middle" or "attacking outward"? 
    # Usually comparative plots splits pitch: Home Left Side, Away Right Side?
    # High Turnover used: Home (left circle), Away (right circle).
    # Let's execute user logic exactly.
    
    # Check Outcome
    if 'outcome' in df_cross.columns:
        is_succ = lambda r: str(r['outcome']) in ['1', 'True', 'Successful']
    else:
        is_succ = lambda r: False
        
    for index, row in home_cross.iterrows():
        # Invert Coords
        start = (120 - row['x_scaled'], 80 - row['y_scaled'])
        end = (120 - row['end_x_scaled'], 80 - row['end_y_scaled'])
        
        if is_succ(row):
            arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, color='green', linewidth=1.5, alpha=1)
            ax.add_patch(arrow)
            hsuc += 1
        else:
            arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=10, color='red', linewidth=1.5, alpha=0.65)
            ax.add_patch(arrow)
            hunsuc += 1
            
    for index, row in away_cross.iterrows():
        start = (row['x_scaled'], row['y_scaled'])
        end = (row['end_x_scaled'], row['end_y_scaled'])
        
        if is_succ(row):
            arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15, color='green', linewidth=1.5, alpha=1)
            ax.add_patch(arrow)
            asuc += 1
        else:
            arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=10, color='red', linewidth=1.5, alpha=0.65)
            ax.add_patch(arrow)
            aunsuc += 1
            
    # Counts
    # Home Left/Right (from attacking perspective)
    # If attacking Right->Left (inverted):
    # original y >= 40 (Left side of pitch from 0->120 perspective).
    # 80-y <= 40.
    # User logic: len(home_cross[home_cross['y']>=40]) -> "Leftwing".
    # StatsBomb: 0 is top-left? Y goes 0->80 downwards? Or 80->0?
    # Standard: 0-80 Bottom-Top?
    # If 0 is top... then y>=40 is Bottom half (Right Wing for Left->Right attack).
    # User labels y>=40 as "Leftwing". That implies 0 is Bottom? Or they attack Right->Left?
    # Let's stick to user labelling logic on 'y'.
    
    home_left = len(home_cross[home_cross['y_scaled'] >= 40])
    home_right = len(home_cross[home_cross['y_scaled'] < 40])
    away_left = len(away_cross[away_cross['y_scaled'] >= 40])
    away_right = len(away_cross[away_cross['y_scaled'] < 40])
    
    # Text Placements
    # Center is 60.
    # User: 59 (Right align for Home), 61 (Left align for Away).
    
    ax.text(59, 2, f"Crosses from\nLeftwing: {home_left}", color='orange', fontsize=12, va='bottom', ha='right')
    ax.text(59, 78, f"Crosses from\nRightwing: {home_right}", color='orange', fontsize=12, va='top', ha='right')
    
    ax.text(61, 78, f"Crosses from\nLeftwing: {away_left}", color=acol, fontsize=12, va='top', ha='left')
    ax.text(61, 2, f"Crosses from\nRightwing: {away_right}", color=acol, fontsize=12, va='bottom', ha='left')
    
    ax.text(0, -2, f"Successful: {hsuc}", color='green', fontsize=15, ha='left', va='top')
    ax.text(0, -5.5, f"Unsuccessful: {hunsuc}", color='red', fontsize=15, ha='left', va='top')
    
    ax.text(120, -2, f"Successful: {asuc}", color='green', fontsize=15, ha='right', va='top')
    ax.text(120, -5.5, f"Unsuccessful: {aunsuc}", color='red', fontsize=15, ha='right', va='top')
    
    ax.text(0, 82, f"{home_team}\nCrosses", color=hcol, size=20, ha='left', fontweight='bold')
    ax.text(120, 82, f"{away_team}\nCrosses", color=acol, size=20, ha='right', fontweight='bold')
    
    return fig_to_base64(fig)

# --- Top Passer Pass Map ---

def plot_top_passer_map(df, team_name, is_home=True):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Identify Top Passer for this team
    # Logic: Pro Passes > 9.144 & Successful & x[40, 119] + Box Entries + Key Passes
    
    team_df = df[df['team_name'] == team_name].copy()
    
    # Calculate metrics per player
    unique_players = team_df['playerName'].unique()
    unique_players = [p for p in unique_players if isinstance(p, str)]
    
    passer_data = []
    
    # Helper for Short Name
    def get_short_name(name):
        if not isinstance(name, str): return str(name)
        names = name.split()
        if len(names) > 1: return f"{names[0][0]}. {names[-1]}"
        return name

    # Ensure columns exist
    if 'keyPass' not in team_df.columns: team_df['keyPass'] = 0
    if 'assist' not in team_df.columns: team_df['assist'] = 0
    if 'pro' not in team_df.columns:
         team_df['pro'] = np.sqrt((team_df['end_x'] - team_df['x'])**2 + (team_df['end_y'] - team_df['y'])**2)
    
    for p in unique_players:
        p_df = team_df[team_df['playerName'] == p]
        
        # Progressive
        prog = len(p_df[
            (p_df['pro'] > 9.144) & 
            (p_df['outcome'].astype(str).str.contains('Successful|1|True')) & 
            (p_df['x_scaled'] >= 40) & 
            (p_df['x_scaled'] <= 119)
        ])
        # Box Entry (using scaled coords: 103.5 is for 120 pitch? Yes. our scaled is 120x80)
        box = len(p_df[
            (p_df['event'] == 'Pass') & 
            (p_df['outcome'].astype(str).str.contains('Successful|1|True')) & 
            (p_df['end_x_scaled'] >= 103.5) & 
            (p_df['end_y_scaled'] >= 16) & 
            (p_df['end_y_scaled'] <= 64)
        ])
        # Key Pass
        kp = len(p_df[(p_df['event'] == 'Pass') & (p_df['keyPass'] == 1)])
        
        total = prog + box + kp
        passer_data.append({'playerName': p, 'total': total, 'shortName': get_short_name(p)})
        
    if not passer_data:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Passer Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    # Get Top Picker
    top_passer = max(passer_data, key=lambda x: x['total'])
    player_name = top_passer['playerName']
    short_name = top_passer['shortName']
    
    # Prepare Plot Data
    player_df = team_df[(team_df['playerName'] == player_name) & (team_df['event'] == 'Pass')]
    
    # Filters
    is_succ = player_df['outcome'].astype(str).str.contains('Successful|1|True')
    pass_comp = player_df[is_succ]
    pass_incomp = player_df[~is_succ]
    kp_df = player_df[player_df['keyPass'] == 1]
    assist_df = player_df[player_df['assist'] == 1]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME # Red (Home) or User defined
    acol = TACTIQ_AWAY # Blue (Away)
    violet = '#8338ec' # Key Pass
    green_col = TACTIQ_ACCENT  # Assist (variable name 'green' in user code)
    
    # User logic used specific colors for Home vs Away top passer map
    # Home: hcol. Away: acol.
    color = hcol if is_home else acol
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    # User Logic: Away Invert?
    # "def away_player_passmap(ax): ... ax.invert_xaxis() ax.invert_yaxis()"
    if not is_home:
        ax.invert_xaxis()
        ax.invert_yaxis()
        
    # Plot Lines (Comet) and Scatter
    # Using scaled coords: x_scaled, y_scaled
    
    # Successful
    pitch.lines(pass_comp.x_scaled, pass_comp.y_scaled, pass_comp.end_x_scaled, pass_comp.end_y_scaled, 
                lw=3, transparent=True, comet=True, color=color, ax=ax, alpha=0.65)
    pitch.scatter(pass_comp.end_x_scaled, pass_comp.end_y_scaled, s=30, color=bg_color, edgecolor=color, zorder=2, ax=ax)
    
    # Unsuccessful
    pitch.lines(pass_incomp.x_scaled, pass_incomp.y_scaled, pass_incomp.end_x_scaled, pass_incomp.end_y_scaled, 
                lw=3, transparent=True, comet=True, color='gray', ax=ax, alpha=0.25)
    pitch.scatter(pass_incomp.end_x_scaled, pass_incomp.end_y_scaled, s=30, color=bg_color, edgecolor='gray', alpha=0.25, zorder=2, ax=ax)
    
    # Key Pass
    pitch.lines(kp_df.x_scaled, kp_df.y_scaled, kp_df.end_x_scaled, kp_df.end_y_scaled, 
                lw=4, transparent=True, comet=True, color=violet, ax=ax, alpha=0.9)
    pitch.scatter(kp_df.end_x_scaled, kp_df.end_y_scaled, s=40, color=bg_color, edgecolor=violet, linewidth=1.5, zorder=2, ax=ax)
    
    # Assist
    pitch.lines(assist_df.x_scaled, assist_df.y_scaled, assist_df.end_x_scaled, assist_df.end_y_scaled, 
                lw=4, transparent=True, comet=True, color=green_col, ax=ax, alpha=1)
    pitch.scatter(assist_df.end_x_scaled, assist_df.end_y_scaled, s=50, color=bg_color, edgecolor=green_col, linewidth=1.5, zorder=2, ax=ax)
    
    # Stats Text
    # Coordinates depend on Home/Away inversion logic provided by user
    if is_home:
        # Home Text Logic
        ax.text(80, 83, f'Successful Pass: {len(pass_comp)}', color=color, va='center', ha='right', fontsize=12)
        ax.text(120, 83, f'Unsuccessful Pass: {len(pass_incomp)}', color='gray', va='center', ha='right', fontsize=12)
        ax.text(80, 88, f'Key Pass: {len(kp_df)}', color=violet, va='center', ha='right', fontsize=12)
        ax.text(120, 88, f'Assist: {len(assist_df)}', color=green_col, va='center', ha='right', fontsize=12)
        
        ax.text(0, 85, "Attacking Direction ----->", color=color, fontsize=15, va='center', ha='left')
    else:
        # Away Text Logic (Note coords are relative to inverted axis? No, ax.text uses data coords usually. 
        # If inverted, 85 is "Left" visually? 
        # User Logic: ax.text(85, -3, ...)
        
        ax.text(85, -3, f'Successful Pass: {len(pass_comp)}', color=color, va='center', ha='left', fontsize=12)
        ax.text(120, -3, f'Unsuccessful Pass: {len(pass_incomp)}', color='gray', va='center', ha='left', fontsize=12)
        ax.text(85, -8, f'Key Pass: {len(kp_df)}', color=violet, va='center', ha='left', fontsize=12)
        ax.text(120, -8, f'Assist: {len(assist_df)}', color=green_col, va='center', ha='left', fontsize=12)
        
        ax.text(0, -5, "<----- Attacking Direction", color=color, fontsize=15, va='center', ha='right')
        
    ax.set_title(f"{short_name} PassMap", color=color, fontsize=25, fontweight='bold')
    
    return fig_to_base64(fig)

# --- Top Defender Actions Map ---

def plot_top_defender_actions(df, team_name, is_home=True):
    # Preprocess
    df = preprocess_for_network(df)
    
    team_df = df[df['team_name'] == team_name].copy()
    
    # Identify Top Defender
    # Logic: Tackles(Succ) + Interceptions + Clearance
    
    unique_players = team_df['playerName'].unique()
    unique_players = [p for p in unique_players if isinstance(p, str)]
    
    def_data = []
    
    # Helper for Short Name
    def get_short_name(name):
        if not isinstance(name, str): return str(name)
        names = name.split()
        if len(names) > 1: return f"{names[0][0]}. {names[-1]}"
        return name

    # Ensure columns
    if 'typeId' not in team_df.columns:
        if 'event' in team_df.columns: team_df['typeId'] = team_df['event']
        else: return pd.DataFrame() # fail safe
        
    for p in unique_players:
        p_df = team_df[team_df['playerName'] == p]
        
        # Check outcome for Tackle
        outcome_mask = p_df['outcome'].astype(str).str.contains('Successful|1|True', case=False, na=False)
        
        tackles = len(p_df[(p_df['typeId'] == 'Tackle') & outcome_mask])
        interceptions = len(p_df[p_df['typeId'] == 'Interception'])
        clearances = len(p_df[p_df['typeId'] == 'Clearance'])
        
        total = tackles + interceptions + clearances
        def_data.append({'playerName': p, 'total': total, 'shortName': get_short_name(p)})

    if not def_data:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Defender Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    top_def = max(def_data, key=lambda x: x['total'])
    player_name = top_def['playerName']
    short_name = top_def['shortName']

    # Get Player Data
    player_df = team_df[team_df['playerName'] == player_name]
    
    # Categories
    # Outcome logic for Tackle
    is_succ = player_df['outcome'].astype(str).str.contains('Successful|1|True', case=False, na=False)
    
    tk = player_df[(player_df['typeId'] == 'Tackle') & is_succ]
    intc = player_df[player_df['typeId'] == 'Interception']
    br = player_df[player_df['typeId'] == 'Ball recovery']
    cl = player_df[player_df['typeId'] == 'Clearance']
    fl = player_df[player_df['typeId'] == 'Foul']
    # Aerial: x <= 60 (User defined logic for defensive aerials?)
    # Note: user code `home_playerdf['x']<=60`.
    # Our scaled x is 120x80. If user meant 60 on 100 scale, that's diff.
    # Assuming user meant 60 on 120 scale (Halfway line).
    ar = player_df[(player_df['typeId'] == 'Aerial') & (player_df['x_scaled'] <= 60)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME # Red
    acol = TACTIQ_AWAY # Blue
    
    color = hcol if is_home else acol
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    # User specific layouts
    if is_home:
        ax.set_ylim(-13, 80.5)
        # Home Markers
        pitch.scatter(tk.x_scaled, tk.y_scaled, s=250, c=color, lw=2.5, edgecolor=color, marker='+', hatch='/////', ax=ax)
        pitch.scatter(intc.x_scaled, intc.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='s', hatch='/////', ax=ax)
        pitch.scatter(br.x_scaled, br.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='o', hatch='/////', ax=ax)
        pitch.scatter(cl.x_scaled, cl.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='d', hatch='/////', ax=ax)
        pitch.scatter(fl.x_scaled, fl.y_scaled, s=250, c=color, lw=2.5, edgecolor=color, marker='x', hatch='/////', ax=ax)
        pitch.scatter(ar.x_scaled, ar.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='^', hatch='/////', ax=ax)
        
        # Legend (Manual)
        pitch.scatter(51, -3, s=150, c=color, lw=2.5, edgecolor=color, marker='+', hatch='/////', ax=ax)
        pitch.scatter(51, -7, s=150, c='None', lw=2.5, edgecolor=color, marker='s', hatch='/////', ax=ax)
        pitch.scatter(51, -11, s=150, c='None', lw=2.5, edgecolor=color, marker='o', hatch='/////', ax=ax)
        pitch.scatter(78, -3, s=150, c='None', lw=2.5, edgecolor=color, marker='d', hatch='/////', ax=ax)
        pitch.scatter(78, -7, s=150, c=color, lw=2.5, edgecolor=color, marker='x', hatch='/////', ax=ax)
        pitch.scatter(78, -11, s=150, c='None', lw=2.5, edgecolor=color, marker='^', hatch='/////', ax=ax)
        
        ax.text(53, -3, "Tackle", color=color, ha='left', va='center', fontsize=13)
        ax.text(53, -7, "Interception", color=color, ha='left', va='center', fontsize=13)
        ax.text(53, -11, "Ball recovery", color=color, ha='left', va='center', fontsize=13)
        ax.text(81, -3, "Clearance", color=color, ha='left', va='center', fontsize=13)
        ax.text(81, -7, "Foul", color=color, ha='left', va='center', fontsize=13)
        ax.text(81, -11, "Aerial", color=color, ha='left', va='center', fontsize=13)
        
        ax.text(0, -5, "Attacking Direction ----->", color=color, fontsize=15, va='center', ha='left')
        
    else:
        # Away
        ax.set_ylim(-0.5, 93)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        pitch.scatter(tk.x_scaled, tk.y_scaled, s=250, c=color, lw=2.5, edgecolor=color, marker='+', hatch='/////', ax=ax)
        pitch.scatter(intc.x_scaled, intc.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='s', hatch='/////', ax=ax)
        pitch.scatter(br.x_scaled, br.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='o', hatch='/////', ax=ax)
        pitch.scatter(cl.x_scaled, cl.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='d', hatch='/////', ax=ax)
        pitch.scatter(fl.x_scaled, fl.y_scaled, s=250, c=color, lw=2.5, edgecolor=color, marker='x', hatch='/////', ax=ax)
        pitch.scatter(ar.x_scaled, ar.y_scaled, s=250, c='None', lw=2.5, edgecolor=color, marker='^', hatch='/////', ax=ax)
        
        # Legend (Away Positioned at Top)
        pitch.scatter(51, 83, s=150, c=color, lw=2.5, edgecolor=color, marker='+', hatch='/////', ax=ax)
        pitch.scatter(51, 87, s=150, c='None', lw=2.5, edgecolor=color, marker='s', hatch='/////', ax=ax)
        pitch.scatter(51, 91, s=150, c='None', lw=2.5, edgecolor=color, marker='o', hatch='/////', ax=ax)
        pitch.scatter(78, 83, s=150, c='None', lw=2.5, edgecolor=color, marker='d', hatch='/////', ax=ax)
        pitch.scatter(78, 87, s=150, c=color, lw=2.5, edgecolor=color, marker='x', hatch='/////', ax=ax)
        pitch.scatter(78, 91, s=150, c='None', lw=2.5, edgecolor=color, marker='^', hatch='/////', ax=ax)
        
        ax.text(53, 83, "Tackle", color=color, ha='right', va='center', fontsize=13)
        ax.text(53, 87, "Interception", color=color, ha='right', va='center', fontsize=13)
        ax.text(53, 91, "Ball recovery", color=color, ha='right', va='center', fontsize=13)
        ax.text(81, 83, "Clearance", color=color, ha='right', va='center', fontsize=13)
        ax.text(81, 87, "Foul", color=color, ha='right', va='center', fontsize=13)
        ax.text(81, 91, "Aerial", color=color, ha='right', va='center', fontsize=13)
        
        ax.text(0, 85, "<----- Attacking Direction", color=color, fontsize=15, va='center', ha='right')

    ax.set_title(f"{short_name} Defensive Actions", color=color, fontsize=25, fontweight='bold')
    
    return fig_to_base64(fig)

# --- FW Pass Receiving Map ---

def plot_fw_pass_receiving(df, team_name, is_home=True):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Identify "Forward"
    # User code uses 'home_FW_name'. We need to auto-detect.
    # Logic: Player with most 'Goal', 'Shot' events? 
    # Or player with highest average X coord?
    
    team_df = df[df['team_name'] == team_name]
    
    # Simple heuristic: Player with most shots (including Miss, Post, Goal, Saved)
    # If no shots, player with highest avg X?
    
    stats_df = team_df[team_df['event'].isin(['Goal', 'Miss', 'Post', 'Attempt Saved'])]
    if not stats_df.empty:
        fw_name = stats_df['playerName'].mode()
        if not fw_name.empty:
            fw_name = fw_name.iloc[0]
        else:
             # Fallback: Highest Avg X
             avg_x = team_df.groupby('playerName')['x_scaled'].mean().sort_values(ascending=False)
             fw_name = avg_x.index[0]
    else:
         avg_x = team_df.groupby('playerName')['x_scaled'].mean().sort_values(ascending=False)
         if avg_x.empty: return pd.DataFrame()
         fw_name = avg_x.index[0]
         
    # Logic: Passes where next player is FW
    # Check shift(-1) logic.
    # We need full DF context? 
    # User: `df[(df['typeId'] == 'Pass') & (df['outcome'] == 'Successful') & (df['playerName'].shift(-1) == name)]`
    # This implies the player receiver is the next row's player.
    # Standard logic for event data (if possession consistent).
    
    # We need to perform shift on the FULL MATCH DF, then filter for our team's passes.
    # BUT, the receiver might be the FW.
    # Pass (Player A) -> Next Event (Player B). If B is FW, strict logic.
    
    # Let's use the DF passed in (which is match df).
    
    df['next_player'] = df['playerName'].shift(-1)
    
    # Filter
    # TypeId='Pass', Outcome=Successful, NextPlayer=FW
    # Also need to check if Pass is by TEAM? 
    # Usually you receive passes from teammates.
    # User code didn't check team on the pass row explicitly in snippet, but assumes 'home_FW_name' implies home passes?
    # Yes, usually.
    
    # Filter rows
    mask = (
        (df['team_name'] == team_name) & 
        (df['event'] == 'Pass') & 
        (df['outcome'].astype(str).str.contains('Successful|1|True')) & 
        (df['next_player'] == fw_name)
    )
    
    filtered_rows = df[mask].copy()
    
    if filtered_rows.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, f"No stats for FW {fw_name}", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    # Key Pass & Assist received
    # User logic: `keypass_recieved_df = filtered_rows[filtered_rows['keyPass']==1]`
    # Note: If 'keyPass' is on the pass, then it means this pass WAS a key pass.
    # So the FW received a key pass (meaning the FW took a shot next?). Yes.
    
    if 'keyPass' not in filtered_rows.columns: filtered_rows['keyPass'] = 0
    if 'assist' not in filtered_rows.columns: filtered_rows['assist'] = 0
    
    kp_rec = filtered_rows[filtered_rows['keyPass'] == 1]
    assist_rec = filtered_rows[filtered_rows['assist'] == 1]
    
    pr = len(filtered_rows)
    kpr = len(kp_rec)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY
    violet = '#8338ec'
    green_col = TACTIQ_ACCENT
    
    color = hcol if is_home else acol
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
    pitch.draw(ax=ax)
    
    if not is_home:
        ax.invert_xaxis()
        ax.invert_yaxis()
        
    # Stats Lines
    # Standard
    pitch.lines(filtered_rows.x_scaled, filtered_rows.y_scaled, filtered_rows.end_x_scaled, filtered_rows.end_y_scaled, 
                lw=3, transparent=True, comet=True, color=color, ax=ax, alpha=0.5)
    # KP
    pitch.lines(kp_rec.x_scaled, kp_rec.y_scaled, kp_rec.end_x_scaled, kp_rec.end_y_scaled, 
                lw=4, transparent=True, comet=True, color=violet, ax=ax, alpha=0.75)
    # Assist
    pitch.lines(assist_rec.x_scaled, assist_rec.y_scaled, assist_rec.end_x_scaled, assist_rec.end_y_scaled, 
                lw=4, transparent=True, comet=True, color=green_col, ax=ax, alpha=0.75)
                
    # Scatters
    pitch.scatter(filtered_rows.end_x_scaled, filtered_rows.end_y_scaled, s=30, edgecolor=color, linewidth=1, color=bg_color, zorder=2, ax=ax)
    pitch.scatter(kp_rec.end_x_scaled, kp_rec.end_y_scaled, s=40, edgecolor=violet, linewidth=1.5, color=bg_color, zorder=2, ax=ax)
    pitch.scatter(assist_rec.end_x_scaled, assist_rec.end_y_scaled, s=50, edgecolors=green_col, linewidths=1, marker='football', c=bg_color, zorder=2, ax=ax) # football marker might fail if not in mplsoccer simple

    # Avg Lines
    avg_x = filtered_rows['end_x_scaled'].median()
    avg_y = filtered_rows['end_y_scaled'].median()
    
    # Ax lines logic
    # User: ax.axvline(x=avg_endX, ymin=0, ymax=68...)
    # mpl axvline uses ymin/ymax in axes fraction (0-1).
    # User passed 68? Maybe they meant plot coordinates using vlines?
    # ax.axvline usually takes 0-1. 
    # Let's use pitch.lines or check if user code worked with plot coords.
    # Ah, standard matplotlib axvline ymin is 0-1.
    # If they pass >1 it might clip.
    # Let's use `ax.plot` for lines to be safe in data coords.
    
    # Vertical line at avg_x, spanning full width?
    # Statsbomb width is 80.
    ax.plot([avg_x, avg_x], [0, 80], color='gray', linestyle='--', alpha=0.6, linewidth=2)
    # Horizontal line at avg_y, spanning full length?
    ax.plot([0, 120], [avg_y, avg_y], color='gray', linestyle='--', alpha=0.6, linewidth=2)
    
    # Helper Short Name
    def get_short_name(name):
        if not isinstance(name, str): return str(name)
        parts = name.split()
        if len(parts) == 1: return name
        elif len(parts) == 2: return f"{parts[0][0]}. {parts[1]}"
        return f"{parts[0][0]}. {parts[1][0]}. {' '.join(parts[2:])}"

    short_name = get_short_name(fw_name)
    
    ax.set_title(f"{short_name} Passes Received", color=color, fontsize=25, fontweight='bold')
    
    # Text
    # Requires highlight_text module? User used `ax_text`.
    # I don't see highlight_text imported.
    # I will use standard text with simple formatting to avoid import errors.
    
    text_str = f"Passes Received: {pr+kpr}" + (f" (KP Rx: {kpr})" if kpr > 0 else "")
    
    if is_home:
        ax.text(60, -2, text_str, color=line_color, fontsize=15, ha='center', va='center')
        ax.text(0, 85, "Attacking Direction ----->", color=color, fontsize=15, va='center', ha='left')
    else:
        ax.text(60, 82, text_str, color=line_color, fontsize=15, ha='center', va='center')
        ax.text(0, -5, "<----- Attacking Direction", color=color, fontsize=15, va='center', ha='right')
        
    return fig_to_base64(fig)

# --- Player Dashboard Bar Charts ---

def plot_player_dashboard_bars(df, home_team, away_team, home_goals=0, away_goals=0):
    from utils.stats import calculate_player_rankings
    # Get DataFrames
    sh_sq_df, passer_df, defender_df = calculate_player_rankings(df)
    
    # Setup Figure
    # 3 Subplots vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    # Colors
    bg_color = TACTIQ_BG
    line_color = TACTIQ_FG
    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY
    violet = '#8338ec'
    
    # 1. Shot Sequence Bar (Ax 0)
    ax = axes[0]
    
    if not sh_sq_df.empty:
        # Sort so largest at top (barh plots bottom-up, so we want largest at bottom? No, user used nsmallest.
        # Usually barh(y, width). y[0] is bottom.
        # User: top10_... = nsmallest(10). So smallest is at index 0. plotted at y=0 (bottom).
        # Largest is index 9, plotted at top.
        # `sh_sq_df` from stats.py is sorted Descending (Head 10).
        # So index 0 is MAX.
        # We want MAX at TOP.
        # So we should REVERSE the df for plotting (so MAX is at end of list).
        
        plot_df = sh_sq_df.iloc[::-1] # Reverse
        
        names = plot_df['shortName'].tolist()
        sh = plot_df['Shots'].tolist()
        sa = plot_df['Shot Assist'].tolist()
        bs = plot_df['Buildup to shot'].tolist()
        
        # Stacked Logic
        # left1 = sh + sa
        left1 = [x + y for x, y in zip(sh, sa)]
        
        ax.barh(names, sh, label='Shot', color=hcol, left=0)
        ax.barh(names, sa, label='Shot Assist', color=violet, left=sh)
        ax.barh(names, bs, label='Buildup to Shot', color=acol, left=left1)
        
        # Labels
        for i, val in enumerate(names):
            # i counts 0..9
            # sh[i] corresponds to names[i]
            ct_sh = sh[i]
            ct_sa = sa[i]
            ct_bs = bs[i]
            
            current_x = 0
            # Shot Label
            if ct_sh > 0:
                ax.text(current_x + ct_sh/2, i, str(ct_sh), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                current_x += ct_sh
            # SA Label
            if ct_sa > 0:
                ax.text(current_x + ct_sa/2, i, str(ct_sa), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                current_x += ct_sa
            # BS Label
            if ct_bs > 0:
                ax.text(current_x + ct_bs/2, i, str(ct_bs), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
        
        # Grid lines
        max_total = sh_sq_df['total'].max()
        if max_total > 2:
             x_coords = range(2, int(max_total), 2)
             for xc in x_coords:
                 ax.axvline(x=xc, color='gray', linestyle='--', zorder=0, alpha=0.3)
    
    ax.set_title("Shot Sequence Involvement", color=line_color, fontsize=20, fontweight='bold')
    ax.legend(facecolor=bg_color, edgecolor=line_color, labelcolor=line_color)
    
    # 2. Passing Bar (Ax 1)
    ax = axes[1]
    
    if not passer_df.empty:
        plot_df = passer_df.iloc[::-1]
        
        names = plot_df['shortName'].tolist()
        pp = plot_df['Progressive Passes'].tolist()
        tp = plot_df['Passes into pen. box'].tolist()
        kp = plot_df['Key Passes'].tolist()
        
        left1 = [x + y for x, y in zip(pp, tp)]
        
        ax.barh(names, pp, label='Prog. Pass', color=hcol, left=0)
        ax.barh(names, tp, label='Passes into Box', color=acol, left=pp)
        ax.barh(names, kp, label='Key Pass', color=violet, left=left1)
        
        for i, val in enumerate(names):
            ct_pp = pp[i]
            ct_tp = tp[i]
            ct_kp = kp[i]
            
            curr = 0
            if ct_pp > 0:
                ax.text(curr + ct_pp/2, i, str(ct_pp), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                curr += ct_pp
            if ct_tp > 0:
                ax.text(curr + ct_tp/2, i, str(ct_tp), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                curr += ct_tp
            if ct_kp > 0:
                ax.text(curr + ct_kp/2, i, str(ct_kp), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
        
        max_total = passer_df['total'].max()
        if max_total > 2:
             x_coords = range(2, int(max_total), 2)
             for xc in x_coords:
                 ax.axvline(x=xc, color='gray', linestyle='--', zorder=0, alpha=0.3)
                 
    # Dashboard Texts
    # User put these on the Passer chart...
    # ax.text((max_x/2), 12, "Top Players Dashboard"...)
    # 12 is above the top bar (index 9).
    ax.set_title("Top 10 Passers Stats", color=line_color, fontsize=20, fontweight='bold', pad=30)
    ax.legend(facecolor=bg_color, edgecolor=line_color, labelcolor=line_color)
    
    # 3. Defender Bar (Ax 2)
    ax = axes[2]
    
    if not defender_df.empty:
        plot_df = defender_df.iloc[::-1]
        
        names = plot_df['shortName'].tolist()
        tk = plot_df['Tackles'].tolist()
        intc = plot_df['Interceptions'].tolist()
        cl = plot_df['Clearance'].tolist()
        
        left1 = [x + y for x, y in zip(tk, intc)]
        
        ax.barh(names, tk, label='Tackle', color=hcol, left=0)
        ax.barh(names, intc, label='Interception', color=violet, left=tk)
        ax.barh(names, cl, label='Clearance', color=acol, left=left1)
        
        for i, val in enumerate(names):
            ct_tk = tk[i]
            ct_in = intc[i]
            ct_cl = cl[i]
            
            curr = 0
            if ct_tk > 0:
                ax.text(curr + ct_tk/2, i, str(ct_tk), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                curr += ct_tk
            if ct_in > 0:
                ax.text(curr + ct_in/2, i, str(ct_in), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')
                curr += ct_in
            if ct_cl > 0:
                ax.text(curr + ct_cl/2, i, str(ct_cl), ha='center', va='center', color=bg_color, fontsize=11, fontweight='bold')

        max_total = defender_df['total'].max()
        if max_total > 2:
             x_coords = range(2, int(max_total), 2)
             for xc in x_coords:
                 ax.axvline(x=xc, color='gray', linestyle='--', zorder=0, alpha=0.3)
                 
    ax.set_title("Top 10 Defenders Stats", color=line_color, fontsize=20, fontweight='bold')
    ax.legend(facecolor=bg_color, edgecolor=line_color, labelcolor=line_color)
    
    # Styling for all
    for ax in axes:
        ax.set_facecolor(bg_color)
        ax.tick_params(axis='x', colors=line_color, labelsize=12)
        ax.tick_params(axis='y', colors=line_color, labelsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
            
    # Super Title for Dashboard
    fig.suptitle(f"Top Players Dashboard\n{home_team} {home_goals} - {away_goals} {away_team}", color=line_color, fontsize=30, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig_to_base64(fig)

# --- Timeline & Momentum Visuals ---

def plot_match_timeline(df, home_team, away_team):
    """
    Creates a chronological timeline of Goals, Cards, and Subs.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    # Hide axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Timeline Line
    ax.hlines(0, 0, 95, color=TACTIQ_FG, linewidth=2, zorder=1) # 90 mins + Stoppage
    
    # Markers
    # 0, 45, 90
    for tick in [0, 45, 90]:
        ax.vlines(tick, -0.5, 0.5, color=TACTIQ_FG, linewidth=1, linestyle='--', alpha=0.5)
        ax.text(tick, -0.7, f"{tick}'", color=TACTIQ_FG, ha='center', fontsize=10)

    # Event Data Logic
    # Opta usually has: Goals (16), Cards (17), Subs (PlayerOff/On or typeId 18/19?)
    # Assuming standard type_ids or event names
    
    # Colors
    hcol = TACTIQ_HOME
    acol = TACTIQ_AWAY
    
    
    # Filter DF for events
    if 'event' in df.columns:
        # Goal (16), Card (17), Sub (Off=18, On=19 generally)
        # Using string names if available
        mask = df['event'].isin(['Goal', 'Card', 'SubstitutionOff', 'SubstitutionOn'])
        # Also check type_id if needed
        event_df = df[mask].copy()
    elif 'type_id' in df.columns:
        # Fallback to type ids: 16 (Goal), 17 (Card), 18/19 (Sub)
        mask = df['type_id'].isin([16, 17, 18]) # 18 usually Player Off
        event_df = df[mask].copy()
        # Map back to readable
        type_map = {16: 'Goal', 17: 'Card', 18: 'Substitution'}
        if not event_df.empty:
             event_df['event'] = event_df['type_id'].map(type_map)
    else:
        event_df = pd.DataFrame()

    if event_df.empty:
         ax.text(45, 0, "No Timeline Events Found", color=TACTIQ_FG, ha='center')
         return fig_to_base64(fig)

    # Plot Events
    for _, row in event_df.iterrows():
        minute = row['expanded_minute'] if 'expanded_minute' in row else (row['min'] if 'min' in row else row.get('time_min', 0))
        team = row['team_name']
        event_type = row.get('event', '')
        player = get_shorter_name(row.get('player_name', ''))
        
        # Y Offset: Home Top (+1), Away Bottom (-1)
        y_pos = 0.5 if team == home_team else -0.5
        color = hcol if team == home_team else acol
        
        # Marker & Icon
        if event_type == 'Goal':
             # Check Own Goal
             if 'qualifiers' in row and 'OwnGoal' in str(row['qualifiers']): # simplified check
                 pass
             
             # Scatter
             ax.scatter(minute, 0, s=200, marker='o', color=color, edgecolors=TACTIQ_FG, zorder=3)
             # Text
             text_y = y_pos + (0.3 if team == home_team else -0.3)
             ax.text(minute, text_y, f"⚽ {player}", color=TACTIQ_FG, ha='center', va='center', fontsize=9, rotation=45 if team==home_team else -45)
             
        elif event_type == 'Card' or row.get('type_id') == 17:
             # Check Red/Yellow
             # Assuming 'card_type' or qualifiers
             card_color = 'yellow'
             if 'Red' in str(row.get('card_type', '')) or 'SecondYellow' in str(row.get('card_type', '')):
                 card_color = 'red'
                 
             ax.scatter(minute, 0, s=100, marker='s', color=card_color, edgecolors=TACTIQ_FG, zorder=3)
             
        elif 'Sub' in event_type or row.get('type_id') == 18:
             ax.scatter(minute, 0, s=80, marker='^' if team==home_team else 'v', color='gray', edgecolors=TACTIQ_FG, zorder=2)

    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-2, 100)
    
    ax.text(0, 1.2, f"{home_team}", color=hcol, fontweight='bold', ha='left')
    ax.text(0, -1.2, f"{away_team}", color=acol, fontweight='bold', ha='left')
    
    return fig_to_base64(fig)

def plot_game_momentum(df, home_team, away_team):
    """
    Creates a momentum chart based on rolling xG or Threat data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    # Data Prep - Aggregate by Minute
    # Metric: xG per minute? Or simple shot count? User wants "what happened".
    # Rolling xT + Shots is good.
    # Let's use simple Events Count in Final Third as proxy for "Pressure" if xG missing, else xG.
    
    use_xg = 'expectedGoals' in df.columns or 'xG' in df.columns
    metric = 'xG' if use_xg else 'Pressure'
    
    # Minute bins
    df['minute_bin'] = df['min'] if 'min' in df.columns else df['time_min']
    
    # Calculate Metric per Minute per Team
    # Init empty series 0-95
    mins = np.arange(0, 96)
    home_vals = pd.Series(0.0, index=mins)
    away_vals = pd.Series(0.0, index=mins)
    
    if use_xg:
        xg_col = 'expectedGoals' if 'expectedGoals' in df.columns else 'xG'
        h_grp = df[df['team_name'] == home_team].groupby('minute_bin')[xg_col].sum()
        a_grp = df[df['team_name'] == away_team].groupby('minute_bin')[xg_col].sum()
    else:
        # Pressure: Attacks (Passes in final third + Shots)
        # Final third: x > 66 (Standard) or x > 80 (Statsbomb 120 scale)
        # Using x_scaled from preprocess if available, else x
        # Let's assume raw x 0-100.
        thresh = 70
        h_grp = df[(df['team_name'] == home_team) & (df['x'] > thresh)].groupby('minute_bin').size()
        a_grp = df[(df['team_name'] == away_team) & (df['x'] > thresh)].groupby('minute_bin').size()
        
    home_vals = home_vals.add(h_grp, fill_value=0)
    away_vals = away_vals.add(a_grp, fill_value=0)
    
    # Apply specific logic: Home positive, Away negative? 
    # Or both positive and smoothed?
    # Usual Momentum: (Home Pressure - Away Pressure) smoothed.
    
    net_momentum = home_vals - away_vals
    
    # Gaussian Smoothing (Window 5 mins)
    smooth_momentum = net_momentum.rolling(window=5, center=True, min_periods=1).mean()
    
    # Plot
    # Fill between
    ax.plot(mins, smooth_momentum, color=TACTIQ_FG, alpha=0.1)
    
    ax.fill_between(mins, smooth_momentum, 0, where=(smooth_momentum >= 0), color=TACTIQ_HOME, alpha=0.6, interpolate=True)
    ax.fill_between(mins, smooth_momentum, 0, where=(smooth_momentum < 0), color=TACTIQ_AWAY, alpha=0.6, interpolate=True)
    
    # Styling
    ax.axhline(0, color=TACTIQ_FG, alpha=0.3, linewidth=1)
    ax.set_xlim(0, 95)
    
    # Max Y for scaling
    max_val = max(abs(smooth_momentum.min()), abs(smooth_momentum.max()))
    if max_val == 0: max_val = 1
    ax.set_ylim(-max_val*1.2, max_val*1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(TACTIQ_FG)
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', colors=TACTIQ_FG)
    
    ax.set_xlabel("Minute", color=TACTIQ_FG)
    ax.set_title(f"Match Momentum ({metric})", color=TACTIQ_FG, fontsize=12)
    
    # Annotate Goals
    # Reuse goal logic if possible, or just skip for simplicity now
    
    return fig_to_base64(fig)

def plot_phase_radar(phases_data_home, phases_data_away, home_team, away_team):
    """
    Plots a radar chart comparing 5 tactical phases between two teams.
    """
    categories = list(phases_data_home.keys())
    N = len(categories)
    
    # Values
    values_home = list(phases_data_home.values())
    values_away = list(phases_data_away.values())
    
    # Close the loop
    values_home += values_home[:1]
    values_away += values_away[:1]
    
    # Angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    # Plot Home
    ax.plot(angles, values_home, linewidth=2, linestyle='solid', label=home_team, color=TACTIQ_HOME)
    ax.fill(angles, values_home, color=TACTIQ_HOME, alpha=0.25)
    
    # Plot Away
    ax.plot(angles, values_away, linewidth=2, linestyle='solid', label=away_team, color=TACTIQ_AWAY)
    ax.fill(angles, values_away, color=TACTIQ_AWAY, alpha=0.25)
    
    # Labels
    plt.xticks(angles[:-1], categories, color='white', size=10)
    
    # Y-grids
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=7)
    plt.ylim(0, 100)
    
    # Grid lines
    ax.grid(color='#444444', linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_visible(False)
    
    # Legend
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=TACTIQ_BG, edgecolor='#444444')
    for text in legend.get_texts():
        text.set_color("white")
        
    ax.set_title("Tactical Phase Comparison", color='white', size=14, pad=20, fontweight='bold')
    
    return fig_to_base64(fig)


# ============================================================
# NEW: PASS FLOW & AVERAGE POSITIONING
# ============================================================

from collections import Counter

def _is_truthy_flag(value):
    return str(value).lower().strip() in ['1', 'true', 'yes', 'si', 'y']


def _event_seconds(row):
    minute = pd.to_numeric(row.get('time_min', 0), errors='coerce')
    second = pd.to_numeric(row.get('time_sec', 0), errors='coerce')
    minute = 0 if pd.isna(minute) else minute
    second = 0 if pd.isna(second) else second
    return float(minute) * 60 + float(second)


def _infer_corner_target(match_df, corner_row, team_name, max_seconds=12):
    """
    Event-data estimate of the player targeted by a corner.
    Opta corners in this feed usually have a landing coordinate, but not a
    direct recipient. We use the next same-team action near that landing point.
    """
    required = {'period_id', 'time_min', 'time_sec', 'team_name', 'player_name'}
    if not required.issubset(match_df.columns):
        return None

    end_x = pd.to_numeric(corner_row.get('Pass End X'), errors='coerce')
    end_y = pd.to_numeric(corner_row.get('Pass End Y'), errors='coerce')
    if pd.isna(end_x) or pd.isna(end_y):
        return None

    period = corner_row.get('period_id')
    start_s = _event_seconds(corner_row)
    after = match_df[
        (match_df['team_name'] == team_name) &
        (match_df['period_id'] == period)
    ].copy()
    after['_seconds'] = after.apply(_event_seconds, axis=1)
    after = after[(after['_seconds'] > start_s) & (after['_seconds'] <= start_s + max_seconds)]

    if 'event_id' in after.columns and pd.notna(corner_row.get('event_id')):
        after = after[after['event_id'] != corner_row.get('event_id')]

    preferred_events = ['Goal', 'Miss', 'Saved Shot', 'Post', 'Aerial', 'Ball touch']
    preferred = after[after['event'].isin(preferred_events)] if 'event' in after.columns else after.iloc[:0]
    candidates = preferred if not preferred.empty else after
    candidates = candidates.dropna(subset=['player_name', 'x', 'y'])
    if candidates.empty:
        return None

    candidates['_dist'] = np.hypot(
        pd.to_numeric(candidates['x'], errors='coerce') - end_x,
        pd.to_numeric(candidates['y'], errors='coerce') - end_y,
    )
    candidates['_score'] = candidates['_dist'] + (candidates['_seconds'] - start_s) * 0.25
    target = candidates.sort_values('_score').iloc[0]
    return target.get('player_name')


def identify_zone(x, y, x_bins=6, y_bins=3):
    """
    Assigns a zone ID based on x,y coordinates (Opta 100x100 scale).
    Returns the center (x,y) of that zone.
    """
    x = np.clip(x, 0, 100)
    y = np.clip(y, 0, 100)
    
    x_edge = np.linspace(0, 100, x_bins + 1)
    y_edge = np.linspace(0, 100, y_bins + 1)
    
    x_idx = np.digitize(x, x_edge, right=False) - 1
    y_idx = np.digitize(y, y_edge, right=False) - 1
    
    x_idx = np.clip(x_idx, 0, x_bins - 1)
    y_idx = np.clip(y_idx, 0, y_bins - 1)
    
    x_center = (x_edge[x_idx] + x_edge[x_idx+1]) / 2
    y_center = (y_edge[y_idx] + y_edge[y_idx+1]) / 2
    
    return (x_center, y_center)

def plot_pass_flow_map(df, team_name):
    """
    Plots the flow of passes between zones (6x3 grid).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=1)
    pitch.draw(ax=ax)
    
    # Filter Successful Passes by Team
    mask = (df['team_name'] == team_name) & (df['event'] == 'Pass') & (df['outcome'] == 1)
    team_passes = df[mask].copy()
    
    if team_passes.empty:
        ax.text(50, 50, "No Pass Data", color=TACTIQ_FG, ha='center')
        return fig_to_base64(fig)
        
    # Calculate Zones
    pass_flows = Counter()
    for _, row in team_passes.iterrows():
        start = identify_zone(row['x'], row['y'])
        
        # Determine end x,y (Fallbacks)
        ex = row.get('Pass End X', row['x']) # Use start if end missing to avoid error, but ideally filter
        ey = row.get('Pass End Y', row['y'])
        
        if pd.isna(ex): ex = row['x']
        if pd.isna(ey): ey = row['y']
        
        end = identify_zone(ex, ey)
        
        if start != end:
            pass_flows[(start, end)] += 1
            
    if not pass_flows:
        ax.text(50, 50, "No Zonal Flows", color=TACTIQ_FG, ha='center')
        return fig_to_base64(fig)
        
    max_count = max(pass_flows.values())
    
    # Plot Flows
    color = TACTIQ_ACCENT # Team Color (Green)
    
    for (start, end), count in pass_flows.items():
        if count < 3: continue # Filter rare flows
        
        width = (count / max_count) * 8
        
        pitch.lines(start[0], start[1], end[0], end[1], lw=width, comet=True, ax=ax,
                   color=color, alpha=0.7, transparent=True, zorder=2)
        
        pitch.scatter(end[0], end[1], s=30, color=color, ax=ax, alpha=0.8)
        
    ax.set_title(f"{team_name}\nPass Flow", color=TACTIQ_FG, fontsize=16, fontweight='bold')
    
    return fig_to_base64(fig)


def calculate_hull_data(player_events, std_dev_filter=1.5):
    """Calculates Convex Hull with outlier filtering."""
    if len(player_events) < 5: return None
    
    points = player_events[['x', 'y']].values
    
    # Filter descriptors
    try:
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        lower, upper = mean - (std * std_dev_filter), mean + (std * std_dev_filter)
        mask = np.all((points >= lower) & (points <= upper), axis=1)
        filtered_points = points[mask]
        
        if len(filtered_points) < 3: return None
        
        hull = ConvexHull(filtered_points)
        hull_points = filtered_points[hull.vertices]
        centroid = np.mean(filtered_points, axis=0)
        return {
            'hull_x': hull_points[:, 0],
            'hull_y': hull_points[:, 1],
            'centroid': centroid,
            'points': filtered_points
        }
    except Exception as e:
        logger.debug("Hull data calculation failed: %s", e)
        return None

def plot_average_positions(df, team_name):
    """
    Plots average positions (centroids) and convex hulls for the top 11 players.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)

    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=1)
    pitch.draw(ax=ax)

    team_data = df[df['team_name'] == team_name].copy()

    if team_data.empty:
        ax.text(50, 50, "No Data", color=TACTIQ_FG, ha='center')
        return fig_to_base64(fig)

    team_data = filter_position_events(team_data.dropna(subset=['player_name', 'x', 'y']))
    top_players = get_starting_xi(team_data, 'player_name')
    
    colors = plt.cm.get_cmap('tab20', len(top_players))
    
    for i, player in enumerate(top_players):
        p_events = team_data[team_data['player_name'] == player]
        hull = calculate_hull_data(p_events, std_dev_filter=1.0)
        
        if hull:
            c = colors(i)
            
            # Hull
            # Polygon
            pitch.polygon([list(zip(hull['hull_x'], hull['hull_y']))], ax=ax, color=c, alpha=0.15)
            pitch.polygon([list(zip(hull['hull_x'], hull['hull_y']))], ax=ax, edgecolor=c, facecolor='none', alpha=0.5, lw=1)
            
            # Centroid
            pitch.scatter(hull['centroid'][0], hull['centroid'][1], ax=ax, s=150, color=c, edgecolors='black', zorder=5)
            
            # Initials
            initials = "".join([x[0].upper() for x in str(player).split(' ')[:2]])
            ax.text(hull['centroid'][0], hull['centroid'][1], initials, fontsize=8, ha='center', va='center', fontweight='bold', color='black', zorder=6)
            
    ax.set_title(f"{team_name}\nAverage Positioning", color=TACTIQ_FG, fontsize=16, fontweight='bold')
    
    return fig_to_base64(fig)


# ============================================================
# BUILDING UP VISUALIZATION (Matplotlib Version)
# ============================================================

def plot_building_up_card(sequence_df, title_info):
    """
    Visualizes a building up sequence on a tactical pitch.
    Returns base64 image.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    # Custom Pitch Appearance
    pitch = Pitch(pitch_type='opta', pitch_color='#14532d', line_color='white', linewidth=1)
    pitch.draw(ax=ax)
    
    # Add Tactical Grid (Dotted lines)
    # Vertical Zones (6)
    for i in range(1, 6):
        x = i * 100/6
        ax.plot([x, x], [0, 100], color='white', alpha=0.2, linestyle=':', linewidth=1, zorder=1)
        
    # Horizontal Zones (4)
    for i in range(1, 4):
        y = i * 25
        ax.plot([0, 100], [y, y], color='white', alpha=0.2, linestyle=':', linewidth=1, zorder=1)
        
    # Thirds (Yellow Dashed)
    ax.plot([33.3, 33.3], [0, 100], color='yellow', alpha=0.6, linestyle='--', linewidth=2, zorder=1)
    ax.plot([66.6, 66.6], [0, 100], color='yellow', alpha=0.4, linestyle='--', linewidth=1, zorder=1)
    
    if sequence_df.empty:
        return fig_to_base64(fig)
        
    # Plot Path
    pitch.lines(sequence_df.x, sequence_df.y, sequence_df.x.shift(-1), sequence_df.y.shift(-1), 
                color='yellow', lw=3, transparent=True, comet=True, ax=ax, zorder=3)
    
    # Plot Events
    pitch.scatter(sequence_df.x, sequence_df.y, s=80, color='white', edgecolors='black', zorder=4, ax=ax)
    
    # Add Labels (Indices)
    for idx, (i, row) in enumerate(sequence_df.iterrows()):
        ax.text(row['x'], row['y'], str(idx+1), color='black', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)
        
    # Start Label
    start_row = sequence_df.iloc[0]
    ax.text(start_row['x'], start_row['y']-7, start_row['event'], color='white', ha='center', fontsize=9, zorder=5)

    # Title
    ax.set_title(f"{title_info}", color='white', fontsize=12, pad=10)
    
    return fig_to_base64(fig)


# ============================================================
# TACTICAL SHAPES (Offensive/Defensive Hulls)
# ============================================================

def get_actions(df, action_type='offensive'):
    """
    Filters events based on the 'event' column.
    """
    # Define which events count as offensive or defensive
    off_events = ['Pass', 'Take On', 'Saved Shot', 'Miss', 'Goal', 'Post']
    def_events = ['Tackle', 'Interception', 'Clearance', 'Blocked Pass', 'Foul', 'Challenge', 'Ball recovery', 'Aerial']
    
    if action_type == 'offensive':
        if 'event' in df.columns:
            return df[df['event'].isin(off_events)].copy()
        return df # Fallback
    else:
        if 'event' in df.columns:
            return df[df['event'].isin(def_events)].copy()
        return df # Fallback

def plot_hulls_on_ax(hulls_dict, axis_obj, title, pitch_obj):
    colors = ['cyan', 'tomato', 'lime', 'yellow', 'violet', 'orange', 'white', 'deepskyblue', 'gold', 'lightpink', 'silver']
    color_idx = 0
    for player, data in hulls_dict.items():
        # Debug logging
        #print(f"DEBUG: player={player}, data_keys={data.keys()}")
        
        color = colors[color_idx % len(colors)]
        
        # Plot Scatter points (small dots)
        if 'points' in data:
            pitch_obj.scatter(data['points'][:, 0], data['points'][:, 1], ax=axis_obj, s=10, color=color, alpha=0.3)
        else:
            print("ERROR: 'points' key missing in data!")
        
        # Plot Polygon (Hull)
        # pitch.polygon handles coords list of (x,y)
        if 'hull_x' in data and 'hull_y' in data:
            pitch_obj.polygon([list(zip(data['hull_x'], data['hull_y']))], ax=axis_obj, color=color, alpha=0.2)
            pitch_obj.polygon([list(zip(data['hull_x'], data['hull_y']))], ax=axis_obj, edgecolor=color, facecolor='none', alpha=0.6)
        
        # Plot Centroid
        if 'centroid' in data:
            pitch_obj.scatter(data['centroid'][0], data['centroid'][1], ax=axis_obj, s=200, color=color, edgecolors='black', zorder=5)
            
            # Initials
            if isinstance(player, str):
                names = player.split(' ')
                if len(names) > 1:
                     initials = "".join([x[0].upper() for x in names[:2]])
                else:
                     initials = names[0][:2].upper()
            else:
                initials = "??"
                
            axis_obj.text(data['centroid'][0], data['centroid'][1], initials, 
                          fontsize=9, ha='center', va='center', fontweight='bold', color='black', zorder=6)
        
        color_idx += 1
    
    axis_obj.set_title(f"{title}", color='white', fontsize=18, fontweight='bold', pad=15)

def plot_tactical_shapes(df, team_name):
    """
    Plots Offensive vs Defensive shapes (Convex Hulls) side-by-side on Vertical Pitch.
    """
    from mplsoccer import VerticalPitch
    
    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Data for Tactical Shapes", color='white', ha='center')
        return fig_to_base64(fig)

    team_df = filter_position_events(team_df.dropna(subset=['player_name', 'x', 'y']))

    # Separate actions
    offensive_df = get_actions(team_df, 'offensive')
    defensive_df = get_actions(team_df, 'defensive')

    # Calculate Hulls for the Top 11 players (by event count)
    top_players = get_starting_xi(team_df, 'player_name')

    off_hulls = {}
    def_hulls = {}

    for player in top_players:
        # Offensive Hull
        p_off = offensive_df[offensive_df['player_name'] == player]
        hull_data = calculate_hull_data(p_off, std_dev_filter=1.0) # Using existing helper
        if hull_data:
            off_hulls[player] = hull_data
            
        # Defensive Hull
        p_def = defensive_df[defensive_df['player_name'] == player]
        hull_data = calculate_hull_data(p_def, std_dev_filter=1.0)
        if hull_data:
            def_hulls[player] = hull_data
            
    # Create Vertical Pitch
    pitch = VerticalPitch(pitch_color=TACTIQ_BG, pitch_type='opta', line_color='white', linewidth=1)
    
    # Grid 1x2
    fig, axs = pitch.grid(nrows=1, ncols=2, title_height=0.1, grid_height=0.8, axis=False)
    fig.set_size_inches(14, 10)
    fig.set_facecolor(TACTIQ_BG)
    
    # Plot on axes
    # axs['pitch'] is array of axes
    plot_hulls_on_ax(off_hulls, axs['pitch'][0], "Offensive Shape", pitch)
    plot_hulls_on_ax(def_hulls, axs['pitch'][1], "Defensive Shape", pitch)
    
    # Main Title
    axs['title'].text(0.5, 0.5, f"{team_name} Tactical Shapes", color='white', va='center', ha='center', fontsize=22, fontweight='bold')
    
    return fig_to_base64(fig)

# ============================================================
# COACH'S QUESTIONS VISUALS
# ============================================================

def plot_swot(swot_data):
    """
    Plots a SWOT Quadrant chart.
    swot_data: Dict with Lists for Strengths, Weaknesses, Opportunities, Threats.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    # Hide axes
    ax.axis('off')
    
    # Draw Quadrants
    # Center (0.5, 0.5)
    # Line Vertical
    ax.plot([0.5, 0.5], [0, 1], color=TACTIQ_FG, linewidth=2)
    # Line Horizontal
    ax.plot([0, 1], [0.5, 0.5], color=TACTIQ_FG, linewidth=2)
    
    # Headers
    ax.text(0.25, 0.95, "STRENGTHS", color=TACTIQ_ACCENT, fontsize=16, fontweight='bold', ha='center')
    ax.text(0.75, 0.95, "WEAKNESSES", color=TACTIQ_ACCENT_SEC, fontsize=16, fontweight='bold', ha='center')
    ax.text(0.25, 0.45, "OPPORTUNITIES", color=TACTIQ_WARNING, fontsize=16, fontweight='bold', ha='center')
    ax.text(0.75, 0.45, "THREATS", color='gray', fontsize=16, fontweight='bold', ha='center')
    
    # Content
    def print_items(items, x, y_start, color):
        for i, item in enumerate(items[:5]): # Limit to top 5
            ax.text(x, y_start - (i*0.06), f"• {item}", color=color, fontsize=10, ha='center', wrap=True)

    print_items(swot_data.get('Strengths', []), 0.25, 0.85, 'white')
    print_items(swot_data.get('Weaknesses', []), 0.75, 0.85, 'white')
    print_items(swot_data.get('Opportunities', []), 0.25, 0.35, 'white')
    print_items(swot_data.get('Threats', []), 0.75, 0.35, 'white')
    
    return fig_to_base64(fig)

def plot_goal_kicks(df, team_name):
    """
    Plots Goal Kick distribution.
    """
    from utils.desc_analysis import analyze_goal_kicks
    res = analyze_goal_kicks(df, team_name)
    gk_events = res.get('events', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Goal Kick Distribution")
    
    if gk_events.empty:
        ax.text(50, 50, "No Goal Kick Data", color='gray', ha="center")
        return fig_to_base64(fig)
    
    # Plot Logic
    # Differentiate Short vs Long
    for _, row in gk_events.iterrows():
        length = row.get('length', 0)
        end_x = row.get('Pass End X', row.get('end_x', row['x']))
        end_y = row.get('Pass End Y', row.get('end_y', row['y']))
        
        color = TACTIQ_ACCENT if length > 40 else 'cyan' # Long=Green, Short=Cyan
        alpha = 0.6
        
        ax.arrow(row['x'], row['y'], end_x - row['x'], end_y - row['y'],
                 head_width=1, head_length=1, fc=color, ec=color, alpha=alpha, width=0.2)
                 
        # Landing
        ax.scatter(end_x, end_y, c=color, s=20, alpha=alpha)
        
    # Legend
    ax.text(2, 95, "Long (>40m)", color=TACTIQ_ACCENT, fontweight='bold')
    ax.text(25, 95, "Short", color='cyan', fontweight='bold')
    
    return fig_to_base64(fig)

def plot_goal_kicks_distribution(df, team_name):
    """
    Goal Kick landing zones distribution (Inside Box, Short Outside, Long).
    """
    from mplsoccer import Pitch
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as path_effects
    
    df = preprocess_for_network(df)
    
    # Filter for the specific team
    df_team = df[df['team_name'] == team_name].copy()
    
    # Filter for Goal Kicks. Prefer the explicit Opta qualifier; only use the
    # coordinate fallback when the qualifier is unavailable.
    if 'Goal Kick' in df_team.columns:
        is_gk = (
            (df_team['event'] == 'Pass') &
            df_team['Goal Kick'].apply(_is_truthy_flag)
        )
    else:
        is_gk = (df_team['event'].astype(str).str.contains('Goal Kick', case=False, na=False))
        is_gk = is_gk | (
            (df_team['event'] == 'Pass') &
            (df_team['x_scaled'] < 7.2) &
            (df_team['y_scaled'].between(24, 56))
        )
    t_gk = df_team[is_gk].copy()
    t_gk = t_gk.dropna(subset=['Pass End X', 'Pass End Y'])
    total_gk = len(t_gk)
    
    fig, ax = plt.subplots(figsize=(6.5, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    # Horizontal Pitch
    pitch = Pitch(pitch_color=TACTIQ_BG, pitch_type='statsbomb', line_color='#555', linewidth=1.2)
    pitch.draw(ax=ax)
    
    # Crop to defensive half (goalkeeper on the left x=0, halfway line on the right x=60)
    ax.set_xlim(-2, 70)
    ax.set_ylim(82, -2)  # Inverted Y-axis to match StatsBomb coordinate standard
    
    clean = (team_name.replace(' Kulübü','').replace(' Spor','')
                      .replace(' Futbol','').strip())
                      
    ax.set_title(f"{clean} - Goal Kick Distribution", color=TACTIQ_FG, fontsize=12, fontweight='bold', pad=10)
    
    if total_gk == 0:
        ax.text(30, 40, "No Goal Kicks Recorded", color='gray', ha="center", va="center", fontsize=11, transform=ax.transData)
        return fig_to_base64(fig)
        
    # Classify landings:
    # 1. Inside Box: end_x_scaled <= 18 and end_y_scaled in [18, 62]
    in_box = t_gk[(t_gk['end_x_scaled'] <= 18) & (t_gk['end_y_scaled'].between(18, 62))]
    
    # 2. Short Outside Box: end_x_scaled <= 40 and not inside the box
    short_outside = t_gk[(t_gk['end_x_scaled'] <= 40) & ~t_gk.index.isin(in_box.index)]
    
    # 3. Long: end_x_scaled > 40
    t_gk[t_gk['end_x_scaled'] > 40]
    
    box_pct = round((len(in_box) / total_gk) * 100) if total_gk else 0
    short_pct = round((len(short_outside) / total_gk) * 100) if total_gk else 0
    long_pct = 100 - box_pct - short_pct if total_gk else 0
    
    left_gk = t_gk[t_gk['end_y_scaled'] < 26.67]
    right_gk = t_gk[t_gk['end_y_scaled'] > 53.33]
    t_gk[t_gk['end_y_scaled'].between(26.67, 53.33)]
    
    left_pct = round((len(left_gk) / total_gk) * 100) if total_gk else 0
    right_pct = round((len(right_gk) / total_gk) * 100) if total_gk else 0
    center_pct = 100 - left_pct - right_pct if total_gk else 0

    # Boundary line at x=40
    ax.axvline(40, color='white', linestyle='--', linewidth=1.2, zorder=2)

    # Horizontal channel lines at y = 26.67 and y = 53.33
    ax.axhline(26.67, color='#ffffff3b', linestyle=':', linewidth=1.0, zorder=2)
    ax.axhline(53.33, color='#ffffff3b', linestyle=':', linewidth=1.0, zorder=2)
    
    # 1. Penalty Box Rectangle: lower-left corner (0, 18), width 18, height 44
    rect_box = Rectangle((0, 18), 18, 44, facecolor='#ef4444', alpha=box_pct/100 * 0.75, zorder=1)
    ax.add_patch(rect_box)
    
    # 2. Short Outside: defensive third excluding box
    # Lower section: lower-left (0, 0), width 40, height 18
    rect_bottom = Rectangle((0, 0), 40, 18, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
    ax.add_patch(rect_bottom)
    # Upper section: lower-left (0, 62), width 40, height 18
    rect_top = Rectangle((0, 62), 40, 18, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
    ax.add_patch(rect_top)
    # Front section: lower-left (18, 18), width 22, height 44
    rect_front = Rectangle((18, 18), 22, 44, facecolor='#ef4444', alpha=short_pct/100 * 0.75, zorder=1)
    ax.add_patch(rect_front)
    
    # 3. Long: beyond x=40
    # Lower-left (40, 0), width 20, height 80
    rect_long = Rectangle((40, 0), 20, 80, facecolor='#ef4444', alpha=long_pct/100 * 0.75, zorder=1)
    ax.add_patch(rect_long)
    
    # Overlay Percentage Text Labels
    path_eff = [path_effects.Stroke(linewidth=2, foreground=TACTIQ_BG), path_effects.Normal()]

    # Penalty box label
    ax.text(9, 40, f"INSIDE BOX\n{box_pct}%", color='white', fontsize=9.0, fontweight='900', ha='center', va='center', zorder=5, path_effects=path_eff)
    
    # Short outside label
    ax.text(29, 40, f"SHORT\n{short_pct}%", color='white', fontsize=9.0, fontweight='900', ha='center', va='center', zorder=5, path_effects=path_eff)
    
    # Long label
    ax.text(50, 40, f"LONG\n{long_pct}%", color='white', fontsize=9.0, fontweight='900', ha='center', va='center', zorder=5, path_effects=path_eff)
    
    # Horizontal side channel percentages in the right margin.
    # LEFT channel (y > 53.33) is physically at the top (near y = 13.33)
    # RIGHT channel (y < 26.67) is physically at the bottom (near y = 66.67)
    # CENTER channel is in the middle (y = 40.0)
    ax.text(63.0, 13.33, f"LEFT: {left_pct}%", color='#fbbf24', fontsize=8.2, fontweight='900', ha='left', va='center', zorder=5, path_effects=path_eff, bbox=dict(facecolor=TACTIQ_BG, alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25'))
    ax.text(63.0, 40.0, f"CENTER: {center_pct}%", color='#fbbf24', fontsize=8.2, fontweight='900', ha='left', va='center', zorder=5, path_effects=path_eff, bbox=dict(facecolor=TACTIQ_BG, alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25'))
    ax.text(63.0, 66.67, f"RIGHT: {right_pct}%", color='#fbbf24', fontsize=8.2, fontweight='900', ha='left', va='center', zorder=5, path_effects=path_eff, bbox=dict(facecolor=TACTIQ_BG, alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25'))
    
    # Info footer (centered below the pitch x=30, y=83)
    ax.text(30, 83, f"Total Goal Kicks: {total_gk}  |  Shading represents zone density", color='#888', fontsize=8.5, ha='center', va='top', zorder=5)
    
    return fig_to_base64(fig)

def plot_final_third_entries(df, team_name):
    """
    Visualizes entries into the final third.
    """
    # We just want visual, but utilizing shared logic is complex if we didn't export dataframe.
    # Let's re-implement simple visual filter here or modify `analyze_final_third_entries` to return df.
    # Re-implementing filter for visual simplicity.
    
    df_team = df[df['team_name'] == team_name]
    
    # Entries: Start < 66, End >= 66
    entries = df_team[
        (df_team['event'].isin(['Pass', 'Carry'])) & 
        (df_team['x'] < 66) & 
        ((df_team.get('Pass End X', 0) >= 66) | (df_team.get('end_x', 0) >= 66))
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    setup_pitch(ax, f"{team_name} - Final Third Entries")
    
    if entries.empty:
        ax.text(50, 50, "No Final Third Entries Data", color='gray', ha="center")
        return fig_to_base64(fig)
        
    # Heatmap of ENTRY POINTS (Where they cross the line)
    # We can approximate entry point interpolation, or just use End Points in Final Third.
    # Let's use End Points (where they receive in final third).
    
    Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG)
    
    end_xs = []
    end_ys = []
    
    for _, row in entries.iterrows():
         x_end = row.get('Pass End X', row.get('end_x', 0))
         y_end = row.get('Pass End Y', row.get('end_y', 0))
         if x_end >= 66:
             end_xs.append(x_end)
             end_ys.append(y_end)
             
    if end_xs:
        try:
            sns.kdeplot(
                x=end_xs, 
                y=end_ys, 
                ax=ax, 
                fill=True, 
                alpha=0.4, 
                cmap='plasma',
                thresh=0.05
            )
        except Exception as e:
            logger.debug("Final third entry KDE skipped: %s", e)

    # Draw arrows for top entries?
    # Too cluttered. Just heatmap of target zones + origin dots.
    
    ax.scatter(entries['x'], entries['y'], c='white', s=10, alpha=0.3, label='Origin')
    
    # 3 Zones lines (vertical in final third?)
    # Final third starts at 66.
    ax.vlines(66, 0, 100, color='white', linestyle='--', alpha=0.5)
    
    # Text
    ax.text(70, 95, "Final Third", color='white', alpha=0.5)
    
    return fig_to_base64(fig)


def plot_team_shot_map(df, team_name):
    """
    Plots a shot map for a single team.
    """
    # Preprocess
    df = preprocess_for_network(df)
    
    # Filter shots
    shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post']
    
    if 'event' in df.columns:
         mask = (df['team_name'] == team_name) & (df['event'].isin(shot_types))
         shots_df = df[mask].copy()
         shots_df['typeId'] = shots_df['event']
    elif 'type_id' in df.columns:
         # Fallback to type_ids if known, or skip
         # Assuming 'typeId' exists as per other functions if 'event' is missing
         if 'typeId' in df.columns:
             mask = (df['team_name'] == team_name) & (df['typeId'].isin(shot_types))
             shots_df = df[mask].copy()
         else:
             return fig_to_base64(plt.figure())
    else:
         return fig_to_base64(plt.figure())
         
    if shots_df.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         setup_pitch(ax, f"{team_name} - Shot Map")
         ax.text(50, 50, "No Shot Data", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    # Stats
    goals = len(shots_df[shots_df['typeId'] == 'Goal'])
    total_shots = len(shots_df)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=2)
    pitch.draw(ax=ax)
    
    # Plot Shots
    for _, shot in shots_df.iterrows():
        x_plot = shot['x_scaled']
        y_plot = shot['y_scaled']
        
        s = 200
        
        if shot['typeId'] == 'Goal':
            pitch.scatter(x_plot, y_plot, s=s, edgecolors='green', c='None', marker='football', zorder=5, ax=ax)
        elif shot['typeId'] == 'Attempt Saved':
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=TACTIQ_WARNING, c='None', hatch='///', marker='o', ax=ax)
        elif shot['typeId'] == 'Post':
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=TACTIQ_FG, c='None', marker='o', ax=ax)
        else:
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=TACTIQ_FG, c='None', marker='x', ax=ax)
            
    # Title & Stats
    ax.set_title(f"{team_name} - Shot Map", color=TACTIQ_FG, fontsize=20, fontweight='bold', pad=15)
    
    stats_text = f"Shots: {total_shots} | Goals: {goals}"
    
    ax.text(60, 83, stats_text, color=TACTIQ_FG, ha='center', fontsize=12)
    
    return fig_to_base64(fig)

def create_radar_chart(categories, values_team, values_rival, team_name, rival_name):
    """
    Creates a radar chart comparing two teams.
    """
    N = len(categories)
    
    # Angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Close the loop
    values_team += values_team[:1]
    values_rival += values_rival[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    plt.xticks(angles[:-1], categories, color='white', size=10)
    
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=7)
    plt.ylim(0, 100)
    
    ax.plot(angles, values_team, linewidth=2, linestyle='solid', label=team_name, color=TACTIQ_ACCENT)
    ax.fill(angles, values_team, color=TACTIQ_ACCENT, alpha=0.25)
    
    ax.plot(angles, values_rival, linewidth=2, linestyle='solid', label=rival_name, color=TACTIQ_ACCENT_SEC)
    ax.fill(angles, values_rival, color=TACTIQ_ACCENT_SEC, alpha=0.25)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor=TACTIQ_BG, edgecolor='white')
    
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#444444', linestyle='--', linewidth=0.5)
    
    return fig_to_base64(fig)

def plot_starting_xi(df, team_name):
    """
    Dual-phase formation map: in-possession vs out-of-possession average positions.
    Shows how each player's role shifts between attacking and defensive phases.
    Connected by arrows; jersey numbers included.
    """
    from mplsoccer import VerticalPitch

    # ── Event classification ─────────────────────────────────
    IN_POSS_EVENTS  = {'Pass', 'Take On', 'Shot', 'Miss', 'Goal', 'Attempt Saved',
                       'Saved Shot', 'Cross', 'Corner taken', 'Free kick taken',
                       'Ball touch', 'Carry'}
    OUT_POSS_EVENTS = {'Tackle', 'Interception', 'Clearance', 'Ball recovery',
                       'Foul', 'Aerial', 'Blocked pass', 'Challenge', 'BallRecovery',
                       'BlockedPass', 'Ball Recovery'}

    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(8, 12))
        fig.patch.set_facecolor(TACTIQ_BG); ax.axis('off')
        return fig_to_base64(fig)

    team_data = filter_position_events(team_df.dropna(subset=['player_name', 'x', 'y']))
    top_players = get_starting_xi(team_data, 'player_name')
    if not top_players:
        fig, ax = plt.subplots(figsize=(8, 12))
        fig.patch.set_facecolor(TACTIQ_BG); ax.axis('off')
        return fig_to_base64(fig)

    # ── Per-player dual positions ────────────────────────────
    player_rows = []
    for player in top_players:
        p_all = team_data[team_data['player_name'] == player]

        p_in  = p_all[p_all['event'].isin(IN_POSS_EVENTS)]  if 'event' in p_all.columns else p_all
        p_out = p_all[p_all['event'].isin(OUT_POSS_EVENTS)] if 'event' in p_all.columns else p_all

        # Fallback to all events if phase-specific is empty
        src_in  = p_in  if len(p_in)  >= 5 else p_all
        src_out = p_out if len(p_out) >= 5 else p_all

        ax_in,  ay_in  = src_in['x'].mean(),  src_in['y'].mean()
        ax_out, ay_out = src_out['x'].mean(), src_out['y'].mean()

        # Jersey number
        jersey = None
        for jcol in ('Jersey Number', 'jersey_number'):
            if jcol in p_all.columns:
                v = p_all[jcol].dropna()
                if not v.empty:
                    jersey = int(v.iloc[0])
                    break

        names   = str(player).split()
        surname = names[-1] if len(names) > 1 else player

        if not (pd.isna(ax_in) or pd.isna(ay_in)):
            player_rows.append({
                'name': player, 'surname': surname,
                'jersey': str(jersey) if jersey is not None else surname[:2].upper(),
                'x_in': ax_in,  'y_in': ay_in,
                'x_out': ax_out if not pd.isna(ax_out) else ax_in,
                'y_out': ay_out if not pd.isna(ay_out) else ay_in,
                'n_in': len(src_in), 'n_out': len(src_out),
            })

    if not player_rows:
        fig, ax = plt.subplots(figsize=(8, 12))
        fig.patch.set_facecolor(TACTIQ_BG); ax.axis('off')
        return fig_to_base64(fig)

    pdf = pd.DataFrame(player_rows)

    # Identify GK (lowest x_in)
    gk_idx = pdf['x_in'].idxmin()
    pdf['is_gk'] = False
    pdf.loc[gk_idx, 'is_gk'] = True

    # ── Draw pitch ───────────────────────────────────────────
    pitch = VerticalPitch(pitch_type='opta', pitch_color=TACTIQ_BG,
                          line_color='#3a3a3a', linewidth=1.2)
    fig, ax = pitch.draw(figsize=(7, 10))
    fig.set_facecolor(TACTIQ_BG)

    ATT_COL  = '#fbbf24'   # in-possession (attacking shape)
    DEF_COL  = '#60a5fa'   # out-of-possession (defensive shape)
    GK_COL   = '#6b7280'

    # ── Formation lines (connect players by x-band) ──────────
    # Group into lines by x_in (opta 0-100, attacking = 100)
    def _draw_formation_lines(pdf_sub, x_col, y_col, col, lw=1.2, alpha=0.3):
        sorted_p = pdf_sub.sort_values(x_col)
        # Cluster into lines (gap > 8 opta units)
        lines, cur = [], [sorted_p.iloc[0]]
        for i in range(1, len(sorted_p)):
            if sorted_p.iloc[i][x_col] - sorted_p.iloc[i-1][x_col] > 8:
                lines.append(cur); cur = []
            cur.append(sorted_p.iloc[i])
        lines.append(cur)
        for line in lines:
            if len(line) < 2:
                continue
            ys = sorted([r[y_col] for r in line])
            xs = [line[0][x_col]] * len(ys)
            # VerticalPitch: x→vertical, y→horizontal
            ax.plot([y for y in ys], [x for x in xs],
                    color=col, linewidth=lw, alpha=alpha, zorder=2,
                    solid_capstyle='round')

    _draw_formation_lines(pdf, 'x_in',  'y_in',  ATT_COL, lw=1.5, alpha=0.35)
    _draw_formation_lines(pdf, 'x_out', 'y_out', DEF_COL, lw=1.5, alpha=0.35)

    # ── Shift arrows ─────────────────────────────────────────
    for _, p in pdf.iterrows():
        dx = p['y_out'] - p['y_in']   # horizontal shift (VerticalPitch: y→x-axis)
        dy = p['x_out'] - p['x_in']   # vertical shift
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 2.0:                 # only draw arrow if meaningful shift
            ax.annotate('', xy=(p['y_out'], p['x_out']),
                        xytext=(p['y_in'], p['x_in']),
                        arrowprops=dict(arrowstyle='->', color='white',
                                        lw=0.8, alpha=0.3),
                        zorder=3)

    # ── Player nodes ─────────────────────────────────────────
    for _, p in pdf.iterrows():
        gk = p['is_gk']
        jersey_txt = p['jersey']

        # Out-of-possession node (hollow blue)
        ax.scatter(p['y_out'], p['x_out'], s=380, color='none',
                   edgecolors=GK_COL if gk else DEF_COL,
                   linewidths=2.0, zorder=4, alpha=0.75)

        # In-possession node (filled gold/grey)
        fc = GK_COL if gk else ATT_COL
        ax.scatter(p['y_in'], p['x_in'], s=460, color=fc,
                   edgecolors='white', linewidths=1.5, zorder=5)

        # Jersey number
        ax.text(p['y_in'], p['x_in'], jersey_txt,
                fontsize=9, ha='center', va='center',
                fontweight='bold', color='#1a1a1a' if not gk else 'white',
                zorder=6)

        # Surname label (positioned away from the node)
        ax.text(p['y_in'], p['x_in'] - 4.5, p['surname'],
                fontsize=7.5, ha='center', va='top',
                color='#d1d5db', fontweight='600', zorder=6)

    # ── Legend ───────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=ATT_COL,
               markeredgecolor='white', markersize=9, label='In Possession'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor='none',
               markeredgecolor=DEF_COL, markeredgewidth=2, markersize=9,
               label='Out of Possession'),
        Line2D([0],[0], color='white', linewidth=0.8, alpha=0.4,
               marker='>', markersize=5, label='Position Shift'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=3,
              fontsize=7, framealpha=0.15, facecolor=TACTIQ_BG,
              edgecolor='#333', labelcolor='white',
              bbox_to_anchor=(0.5, -0.02))

    # Attacking direction arrow (left side of pitch)
    ax.annotate('', xy=(2, 88), xytext=(2, 60),
                arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.8))
    ax.text(2, 74, 'ATTACK', color='#4ade80', fontsize=6.5, ha='center',
            va='center', fontweight='bold', rotation=90)

    clean = team_name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()
    ax.set_title(f"{clean}  —  Dual Phase Positions",
                 color='white', fontsize=15, fontweight='bold', pad=12)

    return fig_to_base64(fig)


def plot_player_comparison_scatter(df, team_name, metric_x, metric_y):
    """
    Scatter plot comparing players in a team on two selected metrics.
    metric_x, metric_y should be column names or recognizable event types.
    """
    # Group by player
    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Data", color='white', ha='center')
        ax.axis('off')
        return fig_to_base64(fig)
        
    # Determine how to calculate metrics
    def calculate_metric(player_df, metric_name):
        if metric_name == 'Passes':
            return len(player_df[player_df['type_id'] == 1])
        elif metric_name == 'Pass Accuracy':
            passes = player_df[player_df['type_id'] == 1]
            if len(passes) == 0: return 0
            success = len(passes[passes['outcome'] == 1])
            return (success / len(passes)) * 100
        elif metric_name == 'Shots':
            types = [13, 14, 15, 16]
            return len(player_df[player_df['type_id'].isin(types)])
        elif metric_name == 'Expected Goals (xG)':
            col = 'expectedGoals' if 'expectedGoals' in player_df.columns else ('xG' if 'xG' in player_df.columns else None)
            return player_df[col].sum() if col else 0
        elif metric_name == 'Defensive Actions':
            types = [7, 8, 12, 44, 4, 49]
            if 'event' in player_df.columns:
                 types_str = ['Tackle', 'Interception', 'Clearance', 'Blocked Pass', 'Foul', 'Challenge', 'Ball recovery', 'Aerial']
                 return len(player_df[player_df['event'].isin(types_str)])
            return len(player_df[player_df['type_id'].isin(types)])
        elif metric_name == 'Progressive Passes':
            if 'is_progressive' in player_df.columns:
                 return len(player_df[(player_df['type_id'] == 1) & (player_df['is_progressive'] == True)])
            return 0  # Fallback
        else:
            return 0
            
    players = team_df['player_name'].dropna().unique()
    data = []
    
    for p in players:
        p_df = team_df[team_df['player_name'] == p]
        if len(p_df) < 5: continue # filter low minute players roughly
        val_x = calculate_metric(p_df, metric_x)
        val_y = calculate_metric(p_df, metric_y)
        data.append({'player': p, 'x': val_x, 'y': val_y})
        
    plot_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    if plot_df.empty:
         ax.text(0.5, 0.5, "Insufficient Data", color='white', ha='center')
         return fig_to_base64(fig)
         
    # Scatter
    sns.scatterplot(data=plot_df, x='x', y='y', s=100, color=TACTIQ_ACCENT, ax=ax, edgecolor='white')
    
    # Annotate
    for _, row in plot_df.iterrows():
        names = str(row['player']).split(' ')
        name_short = names[0][0] + ". " + names[-1] if len(names) > 1 else str(row['player'])
        
        # We need to shift text depending on graph scale
        y_max = plot_df['y'].max()
        if y_max == 0: y_max = 1
        
        ax.text(row['x'], row['y'] + (y_max * 0.02), name_short, color='white', fontsize=8, ha='center')
        
    ax.set_xlabel(metric_x, color='white', fontweight='bold')
    ax.set_ylabel(metric_y, color='white', fontweight='bold')
    ax.set_title(f"{team_name} Player Comparison", color='white', fontsize=14, fontweight='bold', pad=15)
    
    # Styling
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, linestyle='--')
    
    return fig_to_base64(fig)


def plot_pitch_dominance(df, home_team, away_team):
    """
    Cell-based pitch dominance map.
    Each zone is colored by which team had more on-ball actions there.
    Home = TACTIQ_HOME (red), Away = TACTIQ_AWAY (blue).
    Intensity = margin of dominance.
    """
    from mplsoccer import Pitch
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    def _clean(name):
        return name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()

    BAL_EVENTS = {'Pass', 'Take On', 'Shot', 'Ball touch', 'Carry',
                  'Cross', 'Clearance', 'Header', 'Miss', 'Goal',
                  'Attempt Saved', 'Post', 'Saved Shot', 'Ball Recovery',
                  'Tackle', 'Interception', 'Foul', 'Aerial'}

    home_df = df[df['team_name'] == home_team].copy()
    away_df = df[df['team_name'] == away_team].copy()

    if 'event' in df.columns:
        home_df = home_df[home_df['event'].isin(BAL_EVENTS)]
        away_df = away_df[away_df['event'].isin(BAL_EVENTS)]

    home_df = home_df[home_df['x'].notna() & home_df['y'].notna()]
    away_df = away_df[away_df['x'].notna() & away_df['y'].notna()]

    # Invert Away team coordinates to represent opposite attacking direction
    away_df['x'] = 100.0 - away_df['x']
    away_df['y'] = 100.0 - away_df['y']

    # ── Grid setup ────────────────────────────────────────────────────────────
    COLS, ROWS = 5, 3
    x_edges = np.linspace(0, 100, COLS + 1)
    y_edges = np.linspace(0, 100, ROWS + 1)

    def count_grid(events_df):
        grid = np.zeros((ROWS, COLS))
        if events_df.empty:
            return grid
        xi = np.clip(np.digitize(events_df['x'], x_edges) - 1, 0, COLS - 1)
        yi = np.clip(np.digitize(events_df['y'], y_edges) - 1, 0, ROWS - 1)
        for xi_, yi_ in zip(xi, yi):
            grid[yi_, xi_] += 1
        return grid

    home_grid = count_grid(home_df)
    away_grid = count_grid(away_df)
    total_grid = home_grid + away_grid
    total_grid[total_grid == 0] = 1

    home_pct_grid = home_grid / total_grid

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    fig.patch.set_facecolor(TACTIQ_BG)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 8])
    ax_bar  = fig.add_subplot(gs[0])
    ax_pitch = fig.add_subplot(gs[1])

    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG,
                  line_color='#555', linewidth=1.2, corner_arcs=True)
    pitch.draw(ax=ax_pitch)

    clean_home = _clean(home_team)
    clean_away = _clean(away_team)

    # ── Color cells ───────────────────────────────────────────────────────────
    cell_w = 100 / COLS
    cell_h = 100 / ROWS

    for row in range(ROWS):
        for col in range(COLS):
            h_pct = home_pct_grid[row, col]
            h_cnt = int(home_grid[row, col])
            a_cnt = int(away_grid[row, col])

            if h_pct >= 0.5:
                # Home dominates → red
                alpha = 0.15 + 0.55 * (h_pct - 0.5) / 0.5
                color = TACTIQ_HOME
            else:
                # Away dominates → blue
                alpha = 0.15 + 0.55 * (0.5 - h_pct) / 0.5
                color = TACTIQ_AWAY

            x0 = x_edges[col] + 0.5
            y0 = y_edges[row] + 0.5
            rect = FancyBboxPatch(
                (x0, y0), cell_w - 1.0, cell_h - 1.0,
                boxstyle='round,pad=0.8',
                facecolor=color, edgecolor='none', alpha=alpha, zorder=2
            )
            ax_pitch.add_patch(rect)

            # Event count labels
            cx = x_edges[col] + cell_w / 2
            cy = y_edges[row] + cell_h / 2
            ax_pitch.text(cx, cy + 4.5, str(h_cnt),
                          ha='center', va='center', fontsize=9,
                          color=TACTIQ_HOME, fontweight='bold', zorder=5)
            ax_pitch.text(cx, cy - 4.5, str(a_cnt),
                          ha='center', va='center', fontsize=9,
                          color=TACTIQ_AWAY, fontweight='bold', zorder=5)

    # Zone column labels
    col_labels = ['Def Third', 'Def-Mid', 'Mid', 'Att-Mid', 'Att Third']
    for ci, label in enumerate(col_labels):
        cx = x_edges[ci] + cell_w / 2
        ax_pitch.text(cx, -4, label, ha='center', va='top',
                      fontsize=7.5, color='#888')

    # ── Top bar: overall event split ──────────────────────────────────────────
    h_total = int(home_grid.sum())
    a_total = int(away_grid.sum())
    grand   = h_total + a_total or 1
    h_share = h_total / grand
    a_share = a_total / grand

    ax_bar.set_facecolor(TACTIQ_BG)
    ax_bar.barh([0], [h_share], color=TACTIQ_HOME, height=0.6, edgecolor='none', alpha=0.85)
    ax_bar.barh([0], [a_share], left=[h_share], color=TACTIQ_AWAY,
                height=0.6, edgecolor='none', alpha=0.85)
    ax_bar.text(h_share / 2, 0, f'{clean_home}  {h_share*100:.0f}%',
                ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')
    ax_bar.text(h_share + a_share / 2, 0, f'{a_share*100:.0f}%  {clean_away}',
                ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.axis('off')
    ax_bar.set_title('Pitch Dominance  —  On-Ball Actions per Zone',
                     color=TACTIQ_FG, fontsize=13, fontweight='bold', pad=6)

    # ── Attacking direction arrows above the pitch ──
    # Home team (Red) attacks Left -> Right
    ax_pitch.annotate('', xy=(30, 105), xytext=(5, 105),
                      arrowprops=dict(arrowstyle='->', color=TACTIQ_HOME, lw=2.0))
    ax_pitch.text(17.5, 106, f'{clean_home} Attack →', color=TACTIQ_HOME, fontsize=8.5, fontweight='bold', ha='center', va='bottom')

    # Away team (Blue) attacks Right -> Left
    ax_pitch.annotate('', xy=(70, 105), xytext=(95, 105),
                      arrowprops=dict(arrowstyle='->', color=TACTIQ_AWAY, lw=2.0))
    ax_pitch.text(82.5, 106, f'← {clean_away} Attack', color=TACTIQ_AWAY, fontsize=8.5, fontweight='bold', ha='center', va='bottom')

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_pitch.text(2, 103, '●', color=TACTIQ_HOME, fontsize=10, ha='left', va='bottom')
    ax_pitch.text(4, 103, f'{clean_home} (top number)      ', color='#aaa', fontsize=8, ha='left', va='bottom')

    ax_pitch.text(32, 103, '●', color=TACTIQ_AWAY, fontsize=10, ha='left', va='bottom')
    ax_pitch.text(34, 103, f'{clean_away} (bottom number)', color='#aaa', fontsize=8, ha='left', va='bottom')

    return fig_to_base64(fig)

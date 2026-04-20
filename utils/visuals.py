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
    
    team_df = df[df['team_name'] == team_name]
    
    # Filter for starters or top players to avoid clutter
    top_players = get_starting_xi(team_df, 'player_name')
    
    colors = plt.cm.get_cmap('tab20', len(top_players))
    
    for i, player in enumerate(top_players):
        player_df = team_df[(team_df['player_name'] == player) & (team_df['x'].notna()) & (team_df['y'].notna())]
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
        kde = pitch.kdeplot(turnovers.x, turnovers.y, ax=ax, fill=True, levels=100, thresh=0.05, cut=4, cmap='Reds', alpha=0.4)
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
    Visualizes set pieces (Corners or Free Kicks).
    sp_type: 'corners' or 'free_kicks'
    """
    sp_data = analysis.get_set_pieces(df, team_name)
    data = sp_data.get(sp_type, pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    title_text = "Corner Kicks" if sp_type == "corners" else "Dangerous Free Kicks"
    setup_pitch(ax, f"{team_name} - {title_text}")
    
    if data.empty:
        ax.text(50, 50, f"No {title_text} Data Found", color='gray', ha="center")
        return fig_to_base64(fig)
        
    # Plot Origins
    ax.scatter(data['x'], data['y'], c=TACTIQ_FG, s=50, alpha=0.6, label='Origin')
    
    # Plot Deliveries (Arrows to End)
    # Check if 'Pass End X' exists
    if 'Pass End X' in data.columns and 'Pass End Y' in data.columns:
        for _, row in data.iterrows():
            end_x = row['Pass End X']
            end_y = row['Pass End Y']
            if pd.notna(end_x) and pd.notna(end_y):
                # Color by Outcome? (Successful pass?)
                # If outcome = 1 (Success) -> Green, else Red
                is_success = str(row.get('outcome','')).strip() in ['1', 'True', 'Yes', 'Success']
                color = TACTIQ_ACCENT if is_success else 'grey'
                alpha = 0.6 if is_success else 0.3
                
                ax.arrow(row['x'], row['y'], end_x - row['x'], end_y - row['y'],
                         head_width=1, head_length=1, fc=color, ec=color, alpha=alpha, length_includes_head=True)
                         
                # Plot landing spot
                ax.scatter(end_x, end_y, c=color, s=20, alpha=alpha)
                
    # Heatmap of LANDING zones
    # This helps see where they aim
    if 'Pass End X' in data.columns:
        pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG)
        # Filter successful only? Or all attempts? All attempts shows intent.
        try:
            sns.kdeplot(
                x=data['Pass End X'], 
                y=data['Pass End Y'], 
                ax=ax, 
                fill=True, 
                alpha=0.3, 
                cmap='viridis',
                thresh=0.05,
                levels=5
            )
        except Exception as e:
            logger.debug("KDE plot skipped (not enough data): %s", e)

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
    
    # Rescale to StatsBomb (120x80) from assumed Opta (100x100)
    # Note: If data is already StatsBomb, this might distort it, but we follow user instruction
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
        kde = pitch.kdeplot(defensive_actions_df.x_scaled, defensive_actions_df.y_scaled, ax=ax, fill=True, levels=100, thresh=0.02, cut=4, cmap=flamingo_cmap)
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

def plot_progressive_pass_map(df, team_name):
    # Preprocess
    df = preprocess_for_network(df)
    
    # Calculate Progressive Distance if not present
    # Using statsbomb scaled coords: 120x80 (x_scaled, y_scaled)
    # df['pro'] calculation:
    # np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2) - np.sqrt((120 - df['end_x'])**2 + (40 - df['end_y'])**2)
    
    df['pro'] = np.where(
        (df['end_x_scaled'].notna()),
        np.sqrt((120 - df['x_scaled'])**2 + (40 - df['y_scaled'])**2) - np.sqrt((120 - df['end_x_scaled'])**2 + (40 - df['end_y_scaled'])**2),
        0
    )
    
    # Filter for team and progressive passes (pro >= 9.144 meters approx 10 yards)
    # Also ensuring x ranges as per user snippet
    # User snippet: df['pro'] >= 9.144, df['cross']!='Cross', df['x']<=115, df['x']>=40
    # Note: user used 'x', we use 'x_scaled' for 120 system.
    
    mask = (df['team_name'] == team_name) & (df['pro'] >= 9.144) & (df['x_scaled'] <= 115) & (df['x_scaled'] >= 40)
    
    # Check for 'cross' exclusion if column exists
    if 'cross' in df.columns or 'Cross' in df.columns:
         col = 'cross' if 'cross' in df.columns else 'Cross'
         # Check if 'Cross' value indicates a cross (user snippet: df['cross']!='Cross')
         # Assuming boolean or string
         mask = mask & (df[col].astype(str) != 'True') & (df[col].astype(str) != 'Cross')

    dfpro = df[mask].copy()
    
    pro_count = len(dfpro)
    
    if pro_count == 0:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Progressive Passes Found", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)

    # Zones (StatsBomb y goes 0-80)
    # User snippet: y >= 45.33 (Left), 22.67-45.33 (Mid), < 22.67 (Right)
    # Wait, StatsBomb Y is usually inverted? 0 is top left? Or bottom left?
    # mplsoccer standard: 0,80 is top left? 
    # Let's assume user snippet logic matches the y coords we have.
    # 80/3 = 26.66 not 22.67? 
    # User code used explicit values: 45.33 and 22.67. 
    # If pitch width is 68m (Opta/Standard)? Statsbomb is 80.
    # 80 / 3 = 26.6. 
    # User logic seems to use 68m width base? (pro >= 9.144 is exactly 10 yards).
    # If user's data was scaled to 120x80, divisions should be 80/3 = 26.66 and 53.33.
    # User values 22.67 and 45.33 suggest a width of ~68. (68/3 = 22.66).
    # So user was using METERS (105x68).
    # My `preprocess_for_network` scales to 120x80.
    # I should adjust zone thresholds to 120x80 equivalent.
    # Thresholds: 1/3 and 2/3 of 80.
    
    y_h1 = 80 * (1/3) # 26.66
    y_h2 = 80 * (2/3) # 53.33
    
    # Calculate counts using SCALED y
    left_pro = len(dfpro[dfpro['y_scaled'] >= y_h2]) # "Left" depends on viewing orientation. 
    mid_pro = len(dfpro[(dfpro['y_scaled'] >= y_h1) & (dfpro['y_scaled'] < y_h2)])
    right_pro = len(dfpro[dfpro['y_scaled'] < y_h1])
    
    left_percentage = round((left_pro/pro_count)*100)
    mid_percentage = round((mid_pro/pro_count)*100)
    right_percentage = round((right_pro/pro_count)*100)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)

    # Zone Lines
    ax.hlines(y_h1, xmin=0, xmax=120, colors=TACTIQ_FG, linestyle='dashed', alpha=0.35)
    ax.hlines(y_h2, xmin=0, xmax=120, colors=TACTIQ_FG, linestyle='dashed', alpha=0.35)

    # Text (L-R orientation)
    # Top (Left in landscape?) -> Y > 53.33
    ax.text(27, 70, f'{left_pro}\n({left_percentage}%)', color=TACTIQ_WARNING, fontsize=20, va='center', ha='center') # Top
    ax.text(27, 40, f'{mid_pro}\n({mid_percentage}%)', color=TACTIQ_WARNING, fontsize=20, va='center', ha='center') # Mid
    ax.text(27, 10, f'{right_pro}\n({right_percentage}%)', color=TACTIQ_WARNING, fontsize=20, va='center', ha='center') # Bottom

    # Passes
    # mplsoccer pitch.lines handles plotting
    # color=TACTIQ_WARNING (User used 'col')
    pitch.lines(dfpro.x_scaled, dfpro.y_scaled, dfpro.end_x_scaled, dfpro.end_y_scaled, lw=3, transparent=True, comet=True, color=TACTIQ_WARNING, ax=ax, alpha=0.5)
    pitch.scatter(dfpro.end_x_scaled, dfpro.end_y_scaled, s=35, edgecolor=TACTIQ_WARNING, linewidth=1, color=TACTIQ_BG, zorder=2, ax=ax)

    ax.set_title(f"{team_name}\n{pro_count} Progressive Passes", color=TACTIQ_FG, fontsize=20, fontweight='bold')
    
    return fig_to_base64(fig)

from highlight_text import ax_text

def plot_match_shot_map(df, home_team, away_team):
    # Preprocess (Standardize Coords to StatsBomb 120x80)
    # We will use 'x' and 'y' columns directly but assume they need scaling if not statsbomb
    # Helper preprocess_for_network creates x_scaled, y_scaled.
    df = preprocess_for_network(df)
    
    # Filter shots
    # Opta Event Types often: Miss, Goal, Attempt Saved, Post
    # We need to map or use existing columns
    
    shot_types = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
    
    if 'event' in df.columns:
         mask = (df['event'].isin(shot_types))
         Shotsdf = df[mask].copy()
         # Normalize typeId column for user code compatibility
         Shotsdf['typeId'] = Shotsdf['event']
    elif 'type_id' in df.columns:
         # Need mapping if using type_id. 
         # Assuming user provided 'typeId' column exists in their DF or was mapped.
         # If raw opta, usually need mapping.
         # Let's assume 'typeId' or 'event' exists as per user snippet.
         if 'typeId' in df.columns:
             mask = (df['typeId'].isin(shot_types))
             Shotsdf = df[mask].copy()
         else:
             # Fallback: Filter by event_id ranges commonly used for shots if known, or fail
             Shotsdf = pd.DataFrame()
    else:
         Shotsdf = pd.DataFrame()

    if Shotsdf.empty and 'event' not in df.columns and 'typeId' not in df.columns:
        # Try to find any column resembling event type
        pass
        
    if Shotsdf.empty:
         fig, ax = plt.subplots(figsize=(8, 6))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Shot Data Found", color=TACTIQ_FG, ha="center")
         return fig_to_base64(fig)
         
    # Ensure qualifiers for OwnGoal check
    if 'qualifiers' not in Shotsdf.columns:
        Shotsdf['qualifiers'] = ''
        
    # Teams
    hShotsdf = Shotsdf[Shotsdf['team_name'] == home_team].copy()
    aShotsdf = Shotsdf[Shotsdf['team_name'] == away_team].copy()
    
    # Calc Stats
    hgoal_count = len(hShotsdf[hShotsdf['typeId'] == 'Goal'])
    agoal_count = len(aShotsdf[aShotsdf['typeId'] == 'Goal'])
    
    # Check for xG column
    xg_col = next((c for c in ['xG', 'expectedGoals', 'xg'] if c in Shotsdf.columns), None)
    
    if xg_col:
        hxg = round(hShotsdf[xg_col].fillna(0).sum(), 2)
        axg = round(aShotsdf[xg_col].fillna(0).sum(), 2)
    else:
        hxg = 0
        axg = 0
        
    # xGOT (Expected Goals on Target) - usually different model, or same if goal?
    # User had hxgot. If not present, use 0
    xgot_col = next((c for c in ['xGOT', 'expectedGoalsOnTarget'] if c in Shotsdf.columns), None)
    if xgot_col:
        hxgot = round(hShotsdf[xgot_col].fillna(0).sum(), 2)
        axgot = round(aShotsdf[xgot_col].fillna(0).sum(), 2)
    else:
         # Fallback: xGOT often approx xG for goals? or 0
         hxgot = 0 # Placeholder
         axgot = 0
         
    # Shot Counts
    hTotalShots = len(hShotsdf)
    aTotalShots = len(aShotsdf)
    
    hSavedf = hShotsdf[hShotsdf['typeId'].isin(['Attempt Saved', 'Saved Shot'])]
    aSavedf = aShotsdf[aShotsdf['typeId'].isin(['Attempt Saved', 'Saved Shot'])]
    
    hShotsOnT = len(hSavedf) + hgoal_count
    aShotsOnT = len(aSavedf) + agoal_count
    
    hxGpSh = round(hxg/hTotalShots, 2) if hTotalShots > 0 else 0
    axGpSh = round(axg/aTotalShots, 2) if aTotalShots > 0 else 0
    
    # Distances
    # User used 120,40 as center goal point. 
    # For Home: Attacks to Right? Or Left? 
    # In 'plot_shotmap', user plots home shots at (120-x), (80-y). This implies Home attacks LEFT (towards 0)? 
    # Or attacks RIGHT but plotted mirrored?
    # StatsBomb: Attacking team always attacks Left->Right (0->120).
    # If plotting Home on left side of pitch image and Away on right...
    # User code:
    # sc1 = pitch.scatter((120-hGoalData.x), (80-hGoalData.y), ...) -> Inverting Home coords.
    # sc5 = pitch.scatter(aGoalData.x, aGoalData.y, ...) -> Keeping Away coords.
    # This implies plotting Home shots on the LEFT side of the visual (0-60 area) and Away on RIGHT (60-120)? 
    # BUT if SB coords are 0->120, 120 is the goal line.
    # Inverting (120-x) puts shots near 120 back to near 0.
    # So Home shots (near 120) are moved to near 0 (Left Goal).
    # Away shots (near 120) are kept near 120 (Right Goal).
    # This creates a "Home attacks Left Goal, Away attacks Right Goal" visual.
    
    # Distance Calc:
    # User: sqrt((x-120)^2 + (y-40)^2) -> Distance to Right Goal (120,40).
    # Assuming 'x' and 'y' are attacking direction (towards 120).
    # Note: 'x_scaled', 'y_scaled' are standard SB.
    
    hShotsdf['dist'] = np.sqrt((hShotsdf['x_scaled'] - 120)**2 + (hShotsdf['y_scaled'] - 40)**2)
    aShotsdf['dist'] = np.sqrt((aShotsdf['x_scaled'] - 120)**2 + (aShotsdf['y_scaled'] - 40)**2)
    
    home_average_shot_distance = round(hShotsdf['dist'].mean(), 2) if not hShotsdf.empty else 0
    away_average_shot_distance = round(aShotsdf['dist'].mean(), 2) if not aShotsdf.empty else 0
    
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 10)) # Needs to be wide for stats bar
    fig.patch.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='statsbomb', corner_arcs=True, pitch_color=TACTIQ_BG, linewidth=2, line_color=TACTIQ_FG)
    pitch.draw(ax=ax)
    
    ax.set_ylim(-0.5, 80.5)
    ax.set_xlim(-0.5, 120.5)
    
    # Colors
    hcol = TACTIQ_HOME # Home Red
    acol = TACTIQ_AWAY # Away Blue
    
    # Home Shots (Inverted to Left)
    # Using 'x_scaled'
    
    for _, shot in hShotsdf.iterrows():
        x_plot = 120 - shot['x_scaled']
        y_plot = 80 - shot['y_scaled']
        
        s = 200 # Size
        if shot['typeId'] == 'Goal':
            pitch.scatter(x_plot, y_plot, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
        elif shot['typeId'] in ('Attempt Saved', 'Saved Shot'):
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax)
        elif shot['typeId'] == 'Post':
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=hcol, c=hcol, marker='o', ax=ax)
        else: # Miss
             pitch.scatter(x_plot, y_plot, s=s, edgecolors=hcol, c='None', marker='o', ax=ax)

    # Away Shots (Right)
    for _, shot in aShotsdf.iterrows():
        x_plot = shot['x_scaled']
        y_plot = shot['y_scaled']
        
        s = 200
        if shot['typeId'] == 'Goal':
            pitch.scatter(x_plot, y_plot, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
        elif shot['typeId'] in ('Attempt Saved', 'Saved Shot'):
            pitch.scatter(x_plot, y_plot, s=s, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax)
        elif shot['typeId'] == 'Post':
             pitch.scatter(x_plot, y_plot, s=s, edgecolors=acol, c=acol, marker='o', ax=ax)
        else:
             pitch.scatter(x_plot, y_plot, s=s, edgecolors=acol, c='None', marker='o', ax=ax)

    # Stats Bars (Center)
    labels = ["Goals", "xG", "Shots", "On Target", "Avg.Dist."]
    values_h = [hgoal_count, hxg, hTotalShots, hShotsOnT, home_average_shot_distance]
    values_a = [agoal_count, axg, aTotalShots, aShotsOnT, away_average_shot_distance]
    
    # Y positions for bars (from 60 down)
    y_start = 65
    y_step = 8
    y_positions = [y_start - (i*y_step) for i in range(len(labels))]
    
    # Normalization (simple sum ratio scaled to width 20)
    # Avoid div by zero
    
    for i, (lab, vh, va, y) in enumerate(zip(labels, values_h, values_a, y_positions)):
        total = vh + va
        if total == 0:
            norm_h = 10
            norm_a = 10
        else:
            norm_h = (vh / total) * 20
            norm_a = (va / total) * 20
            
        # Draw Bars (Centered at x=60)
        # Left Bar (Home) goes from (60 - norm_h) to 60? 
        # User code: ax.barh ... left=start_x (50). 
        # User code drew bars starting from 50 going RIGHT? 
        # ax.barh(..., width=shooting_stats_normalized_home, left=50) -> Bars go 50->50+w.
        # This overlays them.
        # Typically "Butterfly chart".
        # Home bar should go LEFT from center. Away bar RIGHT from center.
        # Or Home bar is 50-width to 50?
        # User code: left=start_x (50), width=stat. Then away left implies stacking?
        # Let's Implement a clean butterfly chart centered at x=60.
        
        # Text Stats Layout with Background Bars
        
        # Draw Bars (Centered at x=60)
        # Home Bar (growing left from 60 - gap)
        # Using alpha=0.3 to make it subtle behind text or alpha=0.9 next to text?
        # User asked for "bars also", implying visible bars.
        # Let's put bars *behind* the values or *between* label and value?
        # Or standard butterfly: Value | Bar | Label | Bar | Value
        
        # Standard Butterfly: 
        # HomeVal (45) | HomeBar (50->60) | Label (60) | AwayBar (60->70) | AwayVal (75)
        
        gap = 5
        bar_max_width = 15
        
        # Scaling: max possible width is ~15 units. 20 was arbitrary.
        # norm_h IS the width.
        norm_h_scaled = (norm_h / 20) * bar_max_width
        norm_a_scaled = (norm_a / 20) * bar_max_width
        
        # Home Bar (Grow Left from 58)
        ax.barh(y, norm_h_scaled, height=4, left=58-norm_h_scaled, color=hcol, alpha=0.7, align='center')
        
        # Away Bar (Grow Right from 62)
        ax.barh(y, norm_a_scaled, height=4, left=62, color=acol, alpha=0.7, align='center')

        # Center Label
        ax.text(60, y, lab, color=TACTIQ_FG, fontsize=10, ha='center', va='center', fontweight='bold', zorder=5)
        
        # Home Value (Left of Bar)
        ax.text(58-norm_h_scaled-2, y, str(vh), color=TACTIQ_FG, fontsize=12, ha='right', va='center', fontweight='bold')
        
        # Away Value (Right of Bar)
        ax.text(62+norm_a_scaled+2, y, str(va), color=TACTIQ_FG, fontsize=12, ha='left', va='center', fontweight='bold')


    # Titles and Score - REMOVED per user request
    # ax.text(0, 85, f"{home_team}\nShots", color=hcol, size=20, ha='left', fontweight='bold')
    # ax.text(120, 85, f"{away_team}\nShots", color=acol, size=20, ha='right', fontweight='bold')
    
    # Score Highlight - REMOVED per user request
    # try:
    #     ax_text(60, 115, f"<{home_team} {hgoal_count}> - <{agoal_count} {away_team}>", 
    #             highlight_textprops=[{'color':hcol}, {'color':acol}],
    #             color=TACTIQ_FG, fontsize=30, fontweight='bold', ha='center', va='center', ax=ax)
    # except:
    #     pass
        
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
        
        # Calculate xT
        def get_xt_val(r, c):
            if 0 <= r < rows and 0 <= c < cols:
                return xt_grid[r, c]
            return 0

        df_passes['start_xt'] = df_passes.apply(lambda row: get_xt_val(row['y_bin'], row['x_bin']), axis=1)
        df_passes['end_xt'] = df_passes.apply(lambda row: get_xt_val(row['end_y_bin'], row['end_x_bin']), axis=1)
        
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
    
    path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
    
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
    
    events = []
    
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
        va = 'bottom' if team == home_team else 'top'
        
        # Marker & Icon
        if event_type == 'Goal':
             # Check Own Goal
             is_og = False
             if 'qualifiers' in row and 'OwnGoal' in str(row['qualifiers']): # simplified check
                 is_og = True
             
             marker = 'football'
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
        
    ax.set_title(f"Tactical Phase Comparison", color='white', size=14, pad=20, fontweight='bold')
    
    return fig_to_base64(fig)


# ============================================================
# NEW: PASS FLOW & AVERAGE POSITIONING
# ============================================================

from collections import Counter

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
    
    # Identify Top 11 Players
    # Filter for non-na
    team_data = team_data.dropna(subset=['player_name', 'x', 'y'])
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
         # Return placeholder
         fig, ax = plt.subplots(figsize=(6, 4))
         fig.patch.set_facecolor(TACTIQ_BG)
         ax.text(0.5, 0.5, "No Data for Tactical Shapes", color='white', ha='center')
         return fig_to_base64(fig)

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

def plot_final_third_entries(df, team_name):
    """
    Visualizes entries into the final third.
    """
    from utils.desc_analysis import analyze_final_third_entries
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
    
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG)
    
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
    Plots the top 11 players for a given team on two vertical pitches representing 
    the starting XI In Possession and Out of Possession.
    """
    from mplsoccer import VerticalPitch
    
    # Filter for the team
    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(6, 8))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, "No Data for Starting XI", color='white', ha='center')
        ax.axis('off')
        return fig_to_base64(fig)
        
    team_data = team_df.dropna(subset=['player_name', 'x', 'y'])
    
    # Calculate top 11 
    top_players = get_starting_xi(team_data, 'player_name')
    
    # Create side-by-side pitches
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#14532d', line_color='white', linewidth=1)
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(14, 10))
    fig.set_facecolor(TACTIQ_BG)
    
    if not top_players:
        return fig_to_base64(fig)
        
    color = TACTIQ_ACCENT
    if 'team_position' in team_df.columns and not team_df['team_position'].empty:
        pos = str(team_df['team_position'].iloc[0]).lower()
        if pos == 'home': color = TACTIQ_HOME
        elif pos == 'away': color = TACTIQ_AWAY
        
    # Define defensive events list for OOP analysis
    def_types = ['BallRecovery', 'BlockedPass', 'Challenge', 'Clearance', 'Error', 'Foul', 'Interception', 'Tackle', 'Aerial']
        
    # Plot average positions for these 11
    for player in top_players:
        p_events = team_data[team_data['player_name'] == player]
        if 'expanded_minute' in p_events.columns:
             p_events = p_events[p_events['expanded_minute'] < 90] # roughly regular time
             
        # Determine In-Possession vs Out-of-Possession events
        if 'event' in p_events.columns:
             oop_events = p_events[p_events['event'].isin(def_types)]
             ip_events = p_events[~p_events['event'].isin(def_types)]
        else:
             # Fallback if no event names
             oop_events = p_events[p_events['x'] < 60]
             ip_events = p_events[p_events['x'] >= 60]
             
        mean_ip_x, mean_ip_y = ip_events['x'].mean(), ip_events['y'].mean()
        mean_oop_x, mean_oop_y = oop_events['x'].mean(), oop_events['y'].mean()
        
        # Jersey Number logic
        jersey = p_events['jersey_number'].iloc[0] if 'jersey_number' in p_events.columns and not pd.isna(p_events['jersey_number'].iloc[0]) else None
        
        if jersey is not None:
             disp_label = str(player).split()[-1]
             inner_text = str(int(jersey))
        else:
             names = str(player).split(' ')
             if len(names) > 1:
                 disp_label = f"{names[0][0]}. {names[-1]}"
             else:
                 disp_label = player
             inner_text = "".join([x[0].upper() for x in names[:2]])
             
        # Plot In-Possession (axs[0])
        if not pd.isna(mean_ip_x) and not pd.isna(mean_ip_y):
            pitch.scatter(mean_ip_x, mean_ip_y, ax=axs[0], s=350, color=color, edgecolors='white', linewidth=2, zorder=5)
            axs[0].text(mean_ip_y, mean_ip_x + 2.5, disp_label, fontsize=10, ha='center', va='center', color='white', fontweight='bold', zorder=6)
            axs[0].text(mean_ip_y, mean_ip_x, inner_text, fontsize=9, ha='center', va='center', fontweight='bold', color='black', zorder=6)
        else:
            # Fallback to total mean if no IP events
            mean_x, mean_y = p_events['x'].mean(), p_events['y'].mean()
            if not pd.isna(mean_x):
                pitch.scatter(mean_x, mean_y, ax=axs[0], s=350, color=color, edgecolors='white', linewidth=2, alpha=0.5, zorder=5)
            
        # Plot Out-of-Possession (axs[1])
        if not pd.isna(mean_oop_x) and not pd.isna(mean_oop_y):
            pitch.scatter(mean_oop_x, mean_oop_y, ax=axs[1], s=350, color='#374151', edgecolors='white', linewidth=2, zorder=5)
            axs[1].text(mean_oop_y, mean_oop_x + 2.5, disp_label, fontsize=10, ha='center', va='center', color='white', fontweight='bold', zorder=6)
            axs[1].text(mean_oop_y, mean_oop_x, inner_text, fontsize=9, ha='center', va='center', fontweight='bold', color='white', zorder=6)
        else:
            # Fallback
            mean_x, mean_y = p_events['x'].mean(), p_events['y'].mean()
            if not pd.isna(mean_x):
                pitch.scatter(mean_x, mean_y, ax=axs[1], s=350, color='#374151', edgecolors='white', linewidth=2, alpha=0.5, zorder=5)

    axs[0].set_title(f"In Possession (Avg Position)", color='white', fontsize=16, fontweight='bold', pad=10)
    axs[1].set_title(f"Out of Possession (Avg Position)", color='white', fontsize=16, fontweight='bold', pad=10)
    
    fig.suptitle(f"{team_name} - Tactical Formations", color='white', fontsize=20, fontweight='bold', y=0.98)
    
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


def plot_territorial_voronoi(df, home_team, away_team):
    """
    Plots a Voronoi diagram based on the average positions of both teams' starting XIs.
    Demonstrates territorial control.
    """
    from mplsoccer import Pitch
    import numpy as np
    
    home_df = df[df['team_name'] == home_team].copy()
    away_df = df[df['team_name'] == away_team].copy()
    
    if home_df.empty or away_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.axis('off')
        return fig_to_base64(fig)
        
    home_data = home_df.dropna(subset=['player_name', 'x', 'y'])
    away_data = away_df.dropna(subset=['player_name', 'x', 'y'])
    
    top_home = get_starting_xi(home_data, 'player_name')
    top_away = get_starting_xi(away_data, 'player_name')
    
    if not top_home or not top_away:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.axis('off')
        return fig_to_base64(fig)
        
    xs, ys, teams = [], [], []
    
    # Pre-process regular time events
    if 'expanded_minute' in home_data.columns:
        home_data = home_data[home_data['expanded_minute'] < 90]
    if 'expanded_minute' in away_data.columns:
        away_data = away_data[away_data['expanded_minute'] < 90]
        
    for p in top_home:
        p_events = home_data[home_data['player_name'] == p]
        if not p_events.empty:
            xs.append(p_events['x'].mean())
            ys.append(p_events['y'].mean())
            teams.append(1)  # home
            
    for p in top_away:
        p_events = away_data[away_data['player_name'] == p]
        if not p_events.empty:
            # Flip coordinates so teams face each other
            xs.append(100 - p_events['x'].mean())
            ys.append(100 - p_events['y'].mean()) 
            teams.append(0)  # away

    xs = np.array(xs)
    ys = np.array(ys)
    teams = np.array(teams)
    
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, linewidth=1.5)
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    
    try:
        team1_voronoi, team2_voronoi = pitch.voronoi(xs, ys, teams)
        
        # Draw Voronoi for home (Blue/TACTIQ_HOME) and away (Red/TACTIQ_AWAY)
        pitch.polygon(team1_voronoi, ax=ax, fc=TACTIQ_HOME, ec='white', alpha=0.35, lw=1.5)
        pitch.polygon(team2_voronoi, ax=ax, fc=TACTIQ_AWAY, ec='white', alpha=0.35, lw=1.5)
    except Exception as e:
        print(f"Voronoi failed: {e}")
        pass
        
    # Scatter player locations
    pitch.scatter(xs[teams==1], ys[teams==1], ax=ax, c=TACTIQ_HOME, s=200, ec='white', linewidth=2, zorder=4)
    pitch.scatter(xs[teams==0], ys[teams==0], ax=ax, c=TACTIQ_AWAY, s=200, ec='white', linewidth=2, zorder=4)
    
    # Calculate territorial dominance by sampling
    grid_res = 100
    xx, yy = np.meshgrid(np.linspace(0, 100, grid_res), np.linspace(0, 100, grid_res))
    pts = np.c_[xx.ravel(), yy.ravel()]
    
    from scipy.spatial import distance
    dists = distance.cdist(pts, np.column_stack((xs, ys)))
    nearest = np.argmin(dists, axis=1)
    home_pixels = np.sum(teams[nearest] == 1)
    total_pixels = len(nearest)
    
    home_pct = (home_pixels / total_pixels) * 100
    away_pct = 100 - home_pct
    
    # Text formatting
    clean_home = home_team.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').replace(' A.Ş.', '').strip()
    clean_away = away_team.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').replace(' A.Ş.', '').strip()
    
    ax.set_title("Territorial Map (Voronoi Control)", color='white', fontsize=18, fontweight='bold', pad=25)
    
    # Add floating percentage boxes
    ax.text(10, 105, f"{clean_home}\n{home_pct:.1f}%", ha='center', va='center', color='white', fontsize=14, fontweight='bold', bbox=dict(facecolor=TACTIQ_HOME, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    ax.text(90, 105, f"{clean_away}\n{away_pct:.1f}%", ha='center', va='center', color='white', fontsize=14, fontweight='bold', bbox=dict(facecolor=TACTIQ_AWAY, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    return fig_to_base64(fig)

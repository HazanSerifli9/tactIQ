import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TACTIQ_BG = '#111'
TACTIQ_FG = '#eee'
TACTIQ_HOME = '#ef4444'
TACTIQ_AWAY = '#3b82f6'
TACTIQ_ACCENT = '#fbbf24'

def _short(name):
    if pd.isna(name): return ""
    return name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()

def get_short_name(full_name):
    if pd.isna(full_name): return ""
    parts = str(full_name).split()
    if len(parts) == 1: return full_name
    return f"{parts[0][0]}. {parts[-1]}"

def test_new_set_pieces(df, team_name, sp_type="corners"):
    from utils import analysis
    
    from mplsoccer import VerticalPitch
    
    sp_data = analysis.get_set_pieces(df, team_name)
    data = sp_data.get(sp_type, pd.DataFrame())
    
    clean = _short(team_name)
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
        return fig
        
    total_kicks = len(data)
    successful_kicks = 0
    box_entries = 0
    six_yard_entries = 0
    
    # Track takers
    takers = {}
    
    # We will process and normalize all deliveries
    deliveries = []
    
    for idx, row in data.iterrows():
        # Get taker
        taker = row.get('player_name', 'Unknown')
        takers[taker] = takers.get(taker, 0) + 1
        
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
        
        # Clip inside pitch
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
        
    # Draw deliveries
    for d in deliveries:
        x0, y0, x1, y1 = d['x0'], d['y0'], d['x1'], d['y1']
        is_success = d['is_success']
        
        # Attacking goal is at top (x=120). Let's plot arrows.
        # Since vertical=True: 
        # - StatsBomb x (length 0-120) is plotted on the vertical axis (from bottom to top).
        # - StatsBomb y (width 0-80) is plotted on the horizontal axis (from right to left).
        # So we pass y, x to ax.plot or ax.annotate! Or we can use pitch.arrows!
        # Let's use pitch.arrows or pitch.lines! Pitch objects in mplsoccer automatically 
        # handle vertical rotation if we pass StatsBomb coordinates (x, y).
        
        color = TACTIQ_ACCENT if is_success else '#ef4444'
        alpha = 0.8 if is_success else 0.28
        lw = 2.0 if is_success else 1.0
        
        # Plot curved connection line using FancyArrowPatch directly on the axis
        # But wait! If we do it on the axis, we need to map the coordinates correctly.
        # pitch.draw sets up the axis limits: vertical=True means:
        # horizontal axis is y (StatsBomb) which goes from 80 (left) to 0 (right) OR 0 (left) to 80 (right).
        # Let's use pitch.arrows! It's built-in, extremely robust, and automatically vertical-scaled.
        pitch.arrows(d['x0'], d['y0'], d['x1'], d['y1'],
                     color=color, alpha=alpha, lw=lw,
                     headwidth=3.5, headlength=4, headaxislength=3.5,
                     ax=ax_pitch, zorder=4)
                     
        # Plot landing dot
        pitch.scatter(d['x1'], d['y1'], color=color, s=40, alpha=alpha + 0.1,
                      edgecolors=TACTIQ_BG, linewidths=0.5, ax=ax_pitch, zorder=5)
                      
    from matplotlib.patches import Rectangle
    # 6-yard box: bottom-left (30, 114), width 20, height 6
    rect_six = Rectangle((30, 114), 20, 6, facecolor='#fbbf24', alpha=0.08, edgecolor='none', zorder=1)
    ax_pitch.add_patch(rect_six)
    # 18-yard box: bottom-left (18, 102), width 44, height 18
    rect_box = Rectangle((18, 102), 44, 18, facecolor='#ef4444', alpha=0.04, edgecolor='none', zorder=1)
    ax_pitch.add_patch(rect_box)
    
    ax_pitch.set_title(f"{clean}  ·  {title_text} Delivery Map  ·  {total_kicks} total",
                       color=TACTIQ_FG, fontsize=12, fontweight='bold', pad=12)
                       
    # ── Sidebar Panel ─────────────────────────────────────────
    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis('off')
    
    # Title
    ax_side.text(0.5, 0.94, title_text.upper(), fontsize=14, fontweight='bold', color=TACTIQ_ACCENT, ha='center')
    ax_side.plot([0.15, 0.85], [0.91, 0.91], color='#444', linewidth=1.0)
    
    # KPI 1: Completion
    comp_pct = round(successful_kicks / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.82, f"{comp_pct}%", fontsize=32, fontweight='900', color='#22c55e' if comp_pct > 40 else '#fbbf24', ha='center')
    ax_side.text(0.5, 0.77, "Delivery Completion Rate", fontsize=9, color='#888', ha='center')
    
    # KPI 2: Danger Zone Entries
    box_pct = round(box_entries / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.65, f"{box_pct}%", fontsize=24, fontweight='bold', color='white', ha='center')
    ax_side.text(0.5, 0.61, "Deliveries into 18-Yard Box", fontsize=9, color='#888', ha='center')
    
    # KPI 3: 6-Yard Box Danger
    six_pct = round(six_yard_entries / total_kicks * 100) if total_kicks else 0
    ax_side.text(0.5, 0.49, f"{six_pct}%", fontsize=24, fontweight='bold', color='white', ha='center')
    ax_side.text(0.5, 0.45, "Deliveries into 6-Yard Box", fontsize=9, color='#888', ha='center')
    
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
        
    return fig

# Load data and test
df = pd.read_parquet("/Users/hazanserifli/Desktop/tactıq/raw_data/alanya-goztepe.parquet")
fig = test_new_set_pieces(df, "Alanyaspor Kulübü", "corners")
fig.savefig("scratch/test_corners.png")
print("New Corners plot tested and saved to scratch/test_corners.png")

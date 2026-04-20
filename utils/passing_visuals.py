
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator, DictFormatter
import matplotlib.patheffects as path_effects
from mplsoccer import Pitch
import io
import base64
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_ACCENT = "#00ff87" 
TACTIQ_SEC_ACCENT = "#ff0055" 

def generate_passing_diamond_plot(data: pd.DataFrame, league_name: str = "Süper Lig", year: str = "2024") -> Optional[str]:
    """generate Floating Axis Diamond Plot based on performance metrics."""
    if data.empty: return None

    mpl.rcParams['xtick.color'] = TACTIQ_TEXT
    mpl.rcParams['ytick.color'] = TACTIQ_TEXT
    mpl.rcParams['text.color'] = TACTIQ_TEXT
    
    # Normalize Data
    max_box = data['box_passes_per_100'].max() or 1
    max_prog = data['prog_passes_per_100'].max() or 1
    
    data['left_norm'] = 0.99 * data['box_passes_per_100'] / max_box
    data['right_norm'] = 0.99 * data['prog_passes_per_100'] / max_prog
    
    fig = plt.figure(figsize=(9, 9), facecolor=TACTIQ_BG)
    transform = Affine2D().rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=MaxNLocator(nbins=11), grid_locator2=MaxNLocator(nbins=11),
        tick_formatter1=DictFormatter({i: str(round((i * max_prog)/0.99, 1)) if i>0 else '' for i in np.arange(0, 1.1, 0.1)}),
        tick_formatter2=DictFormatter({i: str(round((i * max_box)/0.99, 1)) if i>0 else '' for i in np.arange(0, 1.1, 0.1)})
    )
    
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    ax.patch.set_alpha(0)
    ax.set_position([0.075, 0.07, 0.85, 0.8])
    aux_ax = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux_ax.patch = ax.patch
    
    # Styling
    for axis in ['left', 'bottom']:
        ax.axis[axis].line.set_color(TACTIQ_TEXT)
        ax.axis[axis].label.set_color(TACTIQ_TEXT)
        ax.axis[axis].label.set_fontweight("bold")
        ax.axis[axis].LABELPAD += 7
        
    ax.axis['right'].set_visible(False)
    ax.axis['top'].set_visible(False)
    ax.axis['left'].set_label("Box Entries / 100 Passes")
    ax.axis['bottom'].set_label("Progressive Passes / 100 Passes")
    ax.grid(alpha=0.2, color=TACTIQ_TEXT)
    
    # Plot
    aux_ax.scatter(data['right_norm'], data['left_norm'], c=data['left_norm'] + data['right_norm'], cmap='viridis', edgecolor=TACTIQ_TEXT, s=50, lw=0.5, zorder=2)
    
    # Labels
    top_players = data[((data['left_norm'] + data['right_norm']) > 1.4)]
    path_eff = [path_effects.Stroke(linewidth=1.5, foreground=TACTIQ_BG), path_effects.Normal()]
    
    for i, p in top_players.iterrows():
        n = p['name'].split()
        short = f"{n[0][0]}. {n[-1]}" if len(n) > 1 else p['name']
        aux_ax.text(p['right_norm']+0.015, p['left_norm'], short, color=TACTIQ_TEXT, fontsize=7, zorder=3, path_effects=path_eff)
        
    fig.text(0.5, 0.95, "Effective Passers", fontweight="bold", fontsize=18, color=TACTIQ_TEXT, ha="center")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

def generate_progressive_pass_map(summary_df: pd.DataFrame, pass_details: Dict[str, List[Dict[str, Any]]], league_name: str = "Süper Lig") -> Optional[str]:
    """Generate 4x3 Grid of top progressive passers."""
    if summary_df.empty: return None
    
    top_12 = summary_df.head(12)
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='white', line_zorder=2)
    fig, axs = pitch.grid(nrows=3, ncols=4, title_height=0.08, endnote_space=0, grid_width=0.9, grid_height=0.80, axis=False)
    fig.set_facecolor(TACTIQ_BG)
    fig.set_size_inches(16, 12)
    
    for idx, (i, row) in enumerate(top_12.iterrows()):
        ax = axs['pitch'].flatten()[idx]
        passes = pass_details.get(row['name'], [])
        if not passes: continue
        
        x = [p['x'] for p in passes]
        y = [p['y'] for p in passes]
        ex = [p['endX'] for p in passes]
        ey = [p['endY'] for p in passes]
        
        colors = [TACTIQ_SEC_ACCENT if p['box_entry'] else 'cyan' for p in passes]
        pitch.arrows(x, y, ex, ey, width=2, color=colors, alpha=0.3, ax=ax, zorder=1)
        
        ax.set_title(f"{idx+1}. {row['name']}", color=TACTIQ_TEXT, fontsize=10, ha='left', x=0.05, y=0.95)
        
    axs['title'].text(0.5, 0.5, f"Top Progressive Passers | {league_name}", color=TACTIQ_TEXT, fontsize=20, fontweight='bold', ha='center', va='center')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator, DictFormatter
import matplotlib.patheffects as path_effects
import numpy as np
import io
import base64
import pandas as pd
from typing import Optional

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"

def generate_defensive_plot(data: pd.DataFrame, league_name: str = "Süper Lig", year: str = "2024") -> Optional[str]:
    """Generate defensive contribution plot (Diamonds)."""
    if data.empty: return None

    mpl.rcParams['xtick.color'] = TACTIQ_TEXT
    mpl.rcParams['ytick.color'] = TACTIQ_TEXT
    mpl.rcParams['text.color'] = TACTIQ_TEXT
    
    # Normalize
    max_balls_won = data['balls_won_norm'].max() or 1
    max_recovery = data['recovery_norm'].max() or 1
    data['left_norm'] = 0.99 * data['balls_won_norm'] / max_balls_won
    data['right_norm'] = 0.99 * data['recovery_norm'] / max_recovery
    
    fig = plt.figure(figsize=(10, 10), facecolor=TACTIQ_BG)
    transform = Affine2D().rotate_deg(45)
    
    # Ticks
    tick_gen = np.arange(0, 1.1, 0.1)
    d1 = {i: str(round((i * max_recovery)/0.99, 1)) if i>0 else '' for i in tick_gen}
    d2 = {i: str(round((i * max_balls_won)/0.99, 1)) if i>0 else '' for i in tick_gen}
    
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=MaxNLocator(nbins=11), grid_locator2=MaxNLocator(nbins=11),
        tick_formatter1=DictFormatter(d1), tick_formatter2=DictFormatter(d2)
    )
    
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    ax.patch.set_alpha(0)
    ax.set_position([0.075, 0.07, 0.85, 0.8])
    aux_ax = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux_ax.patch = ax.patch
    
    for axis in ['left', 'bottom']:
        ax.axis[axis].line.set_color(TACTIQ_TEXT)
        ax.axis[axis].label.set_color(TACTIQ_TEXT)
        ax.axis[axis].label.set_fontweight("bold")
        ax.axis[axis].LABELPAD += 7
        
    ax.axis['right'].set_visible(False)
    ax.axis['top'].set_visible(False)
    ax.axis['left'].set_label("Balls Won Directly / 100 Opp Passes")
    ax.axis['bottom'].set_label("Ball Recoveries / 100 Opp Passes")
    ax.grid(alpha=0.2, color=TACTIQ_TEXT)
    
    aux_ax.scatter(data['right_norm'], data['left_norm'], c=data['left_norm']+data['right_norm'], 
                   cmap='viridis', edgecolor=TACTIQ_TEXT, s=60, lw=0.5, zorder=2)
                   
    path_eff = [path_effects.Stroke(linewidth=1.5, foreground=TACTIQ_BG), path_effects.Normal()]
    # Label top 20%
    threshold = (data['left_norm'] + data['right_norm']).quantile(0.8)
    for i, p in data[((data['left_norm'] + data['right_norm']) > threshold)].iterrows():
        n = p['name'].split()
        short = f"{n[0][0]}. {n[-1]}" if len(n)>1 else p['name']
        aux_ax.text(p['right_norm'] + 0.015, p['left_norm'], short, color=TACTIQ_TEXT, fontsize=8, zorder=3, path_effects=path_eff)
        
    fig.text(0.5, 0.95, "Defensive Contributions", fontweight="bold", fontsize=18, color=TACTIQ_TEXT, ha="center")
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

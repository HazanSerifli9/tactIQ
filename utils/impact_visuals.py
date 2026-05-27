
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
from typing import Optional

import os
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils.data import TEAM_LOGOS
from utils.cache import disk_cache

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_POS = "#00ff87" 
TACTIQ_NEG = "#ff0055" 

@disk_cache
def generate_impact_chart(df: pd.DataFrame, metric: str = 'creation', top_n: int = 10, league_name: str = "Süper Lig") -> Optional[str]:
    """
    Generate Diverging Bar Chart for Player Impact.
    metric: 'creation' or 'concession'.
    """
    if df.empty: return None
        
    if metric == 'creation':
        df = df.sort_values('impact_creation', ascending=False)
        title = "Impact on Threat Creation"
        subtitle = "Net difference in Team Threat created (On - Off) per 90"
        col = 'impact_creation'
        top, bot = df.head(top_n), df.tail(top_n).iloc[::-1]
    else:
        df = df.sort_values('impact_concession', ascending=True)
        title = "Impact on Threat Prevention"
        subtitle = "Net difference in Opponent Threat conceded (On - Off) per 90. Negative is GOOD."
        col = 'impact_concession'
        top, bot = df.head(top_n), df.tail(top_n).iloc[::-1]

    plot_data = pd.concat([top, bot])
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    values = plot_data[col]
    colors = []
    for v in values:
        if metric == 'creation': colors.append(TACTIQ_POS if v > 0 else TACTIQ_NEG)
        else: colors.append(TACTIQ_POS if v < 0 else TACTIQ_NEG) # Negative Concession is Good
            
    bars = ax.barh(np.arange(len(plot_data)), values, color=colors, height=0.6)
    
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data['name'], color=TACTIQ_TEXT, fontsize=10)
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
    
    for i, (bar, v, idx) in enumerate(zip(bars, values, plot_data.index)):
        width = bar.get_width()
        label_x = width + (0.05 if v >= 0 else -0.05)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f"{v:+.2f}", va='center', ha='left' if v>=0 else 'right', color=TACTIQ_TEXT, fontsize=9)
        
        # Add Logo
        team_name = plot_data.loc[idx, 'team'] if 'team' in plot_data.columns else None
        if team_name:
            logo_path = TEAM_LOGOS.get(team_name)
            if logo_path and os.path.exists(os.path.abspath(logo_path)):
                img = Image.open(os.path.abspath(logo_path))
                # Next to name (left axis)
                # Need to use data coords or axis transform
                # y is i. x is tricky.
                # Let's put it on the far right? Or next to the bar? 
                # Let's put it next to the Y tick label (Left side)
                # Using AnnotationBbox with negative X in axes coords
                imagebox = OffsetImage(img, zoom=0.025)
                # x coord: -0.15 axes coords
                ab = AnnotationBbox(imagebox, (0, i), xybox=(-40, 0), xycoords=('data', 'data'), boxcoords="offset points", frameon=False)
                ax.add_artist(ab)
                
    ax.invert_yaxis()
    ax.axhline(top_n - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Minimalist Spines
    for s in ax.spines.values(): s.set_visible(False)
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color(TACTIQ_TEXT)
    ax.tick_params(axis='x', colors=TACTIQ_TEXT)
    ax.tick_params(axis='y', left=False)
    
    fig.text(0.0, 1.02, title, color=TACTIQ_TEXT, fontsize=20, fontweight='bold', ha='left', transform=ax.transAxes)
    fig.text(0.0, 0.98, subtitle, color=TACTIQ_TEXT, fontsize=12, ha='left', alpha=0.8, transform=ax.transAxes)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
import base64
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import os
from PIL import Image
from utils.data import TEAM_LOGOS
from utils.cache import disk_cache
from shared.logger import get_logger

logger = get_logger(__name__)

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_ACCENT = "#00ff87" 
TACTIQ_SEC_ACCENT = "#ff0055" 

def _plot_logo(ax, team_name: str):
    logo_path = TEAM_LOGOS.get(team_name)
    if logo_path:
        try:
            full_logo_path = os.path.abspath(logo_path)
            if os.path.exists(full_logo_path):
                team_logo = Image.open(full_logo_path)
                # Place logo in top right corner of the pitch ax
                ax_pos = ax.get_position()
                # Relative placement on the Axes itself using inset_axes or just adding an axis is tricky with Pitch grid
                # Easies: imshow in data coords? Or ax.add_artist(OffsetImage)
                # Pitch coords: 0-120 x, 0-80 y.
                # Let's put it at (110, 70)
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                imagebox = OffsetImage(team_logo, zoom=0.04)
                ab = AnnotationBbox(imagebox, (90, 85), frameon=False)
                ax.add_artist(ab)
        except Exception as e:
            logger.debug("Logo render skipped for %s: %s", team_name, e)

@disk_cache
def generate_carry_plot(summary_df: pd.DataFrame, carry_details: Dict[str, List[Dict[str, Any]]], league_name="Süper Lig", year="2024") -> Optional[str]:
    """Generate a 4x3 subplot grid of top progressive carriers."""
    if summary_df.empty: return None
    
    top_12 = summary_df.head(12)
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='white', line_zorder=2)
    fig, axs = pitch.grid(nrows=3, ncols=4, title_height=0.08, endnote_space=0, grid_width=0.9, grid_height=0.80, axis=False)
    fig.set_facecolor(TACTIQ_BG)
    fig.set_size_inches(16, 12)
    
    for idx, (i, row) in enumerate(top_12.iterrows()):
        ax = axs['pitch'].flatten()[idx]
        carries = carry_details.get(row['name'], [])
        if not carries: continue
            
        x_start = [c['x_start'] for c in carries]
        y_start = [c['y_start'] for c in carries]
        x_end = [c['x_end'] for c in carries]
        y_end = [c['y_end'] for c in carries]
        
        pitch.arrows(x_start, y_start, x_end, y_end, width=2, color=TACTIQ_ACCENT, alpha=0.6, ax=ax, zorder=1)
        
        # Highlight Box Entries
        box_idx = [i for i, c in enumerate(carries) if c['box_entry']]
        if box_idx:
            pitch.arrows([x_start[i] for i in box_idx], [y_start[i] for i in box_idx],
                         [x_end[i] for i in box_idx], [y_end[i] for i in box_idx],
                         width=3, color=TACTIQ_SEC_ACCENT, alpha=0.9, ax=ax, zorder=2)

        ax.set_title(f"{idx+1}. {row['name']}", color=TACTIQ_TEXT, fontsize=10, ha='left', x=0.05, y=0.95)
        ax.text(2, 5, f"Runs: {row['prog_carry_count']}", color=TACTIQ_TEXT, fontsize=9, va='bottom', ha='left', zorder=5)
        
        # Add Attacking Direction Arrow
        ax.annotate('', xy=(85, -5), xytext=(15, -5),
            arrowprops=dict(arrowstyle='->', color='w', lw=1),
            ha='center', va='center', zorder=5, annotation_clip=False)
        ax.text(50, -9, 'Attack Direction', color='w', ha='center', va='center', fontsize=8, zorder=5)
        
        if 'team' in row:
            _plot_logo(ax, row['team'])

    # Title & Legend
    axs['title'].text(0.04, 0.5, f"Top Progressive Ball Carriers | {league_name}", color=TACTIQ_TEXT, fontsize=20, fontweight='bold', ha='left', va='center')
    
    legend_elements = [
        Line2D([0], [0], color=TACTIQ_ACCENT, lw=2, label='Progressive Carry'),
        Line2D([0], [0], color=TACTIQ_SEC_ACCENT, lw=3, label='Carry into Box')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.94), ncol=2, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

import os
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils.data import TEAM_LOGOS
from shared.logger import get_logger

logger = get_logger(__name__)
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import io
import base64
import numpy as np
from typing import List, Dict, Any, Optional

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_ACCENT = "#00ff87" 

import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def _plot_logo(ax, team_name):
    logo_path = TEAM_LOGOS.get(team_name)
    if logo_path:
        try:
            full_logo_path = os.path.abspath(logo_path)
            if os.path.exists(full_logo_path):
                team_logo = Image.open(full_logo_path)
                # Place logo neatly in the corner to avoid overlapping dots and density shapes
                imagebox = OffsetImage(team_logo, zoom=0.035)
                ab = AnnotationBbox(imagebox, (90, 110), frameon=False, zorder=5)
                ax.add_artist(ab)
        except Exception as e:
            logger.debug("Logo render skipped for %s: %s", team_name, e)

def generate_box_entry_grid(players_data: List[Dict[str, Any]], league_name: str = "Süper Lig") -> Optional[str]:
    """
    Generate 4x5 Grid of Box Entry Heatmaps.
    """
    if not players_data:
        return None
        
    pitch = VerticalPitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='white', line_zorder=2, half=True)
    fig, axs = pitch.grid(nrows=4, ncols=5, axis=False, title_height=0.08, endnote_height=0, grid_height=0.82, figheight=14)
    fig.set_facecolor(TACTIQ_BG)
    
    axs_flat = axs['pitch'].flatten()
    
    # Use the same vibrant threat colormap for consistency
    threat_cmap = mpl.colors.LinearSegmentedColormap.from_list("ThreatCmap", [TACTIQ_BG, "#118ab2", "#06d6a0", "#ffd166"])
    
    for idx, ax in enumerate(axs_flat):
        if idx < len(players_data):
            p_data = players_data[idx]
            x, y = np.array(p_data['x']), np.array(p_data['y'])
            
            # KDE Heatmap 
            if len(x) > 4:
                try:
                    pitch.kdeplot(x, y, ax=ax, cmap=threat_cmap, fill=True, levels=10, alpha=0.85, zorder=0)
                except Exception as e:
                    logger.debug("Box entry KDE skipped: %s", e)
            
            # Scatter (make them slightly larger and clearer)
            pitch.scatter(x, y, ax=ax, s=15, color='white', alpha=0.7, zorder=1)
            
            # Labels
            name_parts = p_data['name'].split()
            short_name = f"{name_parts[0][0]}. {name_parts[-1]}" if len(name_parts) > 1 else p_data['name']
            
            # Title pushed up slightly
            ax.set_title(f"{idx+1}. {short_name}", color=TACTIQ_TEXT, fontsize=11, fontweight='bold', pad=-15)
            
            # Count label placed neatly with a background box to prevent clash with lines/heatmaps
            ax.text(25, 45, f"Entries: {p_data['count']}", color=TACTIQ_TEXT, fontsize=10, fontweight='bold', ha='center',
                    bbox=dict(facecolor=TACTIQ_BG, alpha=0.7, edgecolor='none'))
            
            # Attacking Direction Arrow (Vertical Pitch: Attack goes up (Y: 50 -> 100))
            ax.annotate('', xy=(10, 80), xytext=(10, 60),
                arrowprops=dict(arrowstyle='->', color='w', lw=1),
                ha='center', va='center', zorder=5, annotation_clip=False)
            ax.text(5, 70, 'Attack', color='w', ha='center', va='center', fontsize=8, rotation=90, zorder=5)

            if 'team' in p_data:
                _plot_logo(ax, p_data['team'])
        else:
            ax.axis('off')
    
    # Fix overall title and add visual legend
    fig.text(0.04, 0.965, f"Top 20 Box Entry Sources | {league_name}", 
             color=TACTIQ_TEXT, fontsize=22, fontweight='bold', ha='left', va='center')
             
    legend_elements = [
        Patch(facecolor='#ffd166', edgecolor='none', label='High Volume Zone'),
        Patch(facecolor='#06d6a0', edgecolor='none', label='Moderate Volume Zone'),
        Patch(facecolor='#118ab2', edgecolor='none', label='Low Volume Zone'),
        Line2D([0], [0], marker='o', color='w', label='Pass Starting Point', markerfacecolor='w', markersize=6, linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.04, 0.94), ncol=4, frameon=False, labelcolor='w', fontsize=12, handlelength=1.5, columnspacing=2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

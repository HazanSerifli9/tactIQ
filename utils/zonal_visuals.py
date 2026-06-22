import os
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils.data import TEAM_LOGOS
from utils.cache import disk_cache
from shared.logger import get_logger

logger = get_logger(__name__)
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
import base64
from typing import List, Dict, Any, Optional

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_ACCENT = "#00ff87" 

def _plot_logo(ax, team_name, x, y, zoom=0.03):
    logo_path = TEAM_LOGOS.get(team_name)
    if logo_path:
        try:
            full_logo_path = os.path.abspath(logo_path)
            if os.path.exists(full_logo_path):
                team_logo = Image.open(full_logo_path)
                imagebox = OffsetImage(team_logo, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=4)
                ax.add_artist(ab)
        except Exception as e:
            logger.debug("Logo render skipped for %s: %s", team_name, e)

@disk_cache
def generate_zonal_map(grid_data: List[Dict[str, Any]], rows: int = 5, cols: int = 6, league_name: str = "Süper Lig") -> Optional[str]:
    """
    Generate Pitch Map with Top Player annotations per zone.
    """
    if not grid_data:
        return None
        
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='white', line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(TACTIQ_BG)
    
    w = 100 / cols
    h = 100 / rows
    
    # Draw Grid Lines
    for c in range(1, cols):
        ax.plot([c*w, c*w], [0, 100], color='white', alpha=0.1, linestyle='--', zorder=1)
    for r in range(1, rows):
        ax.plot([0, 100], [r*h, r*h], color='white', alpha=0.1, linestyle='--', zorder=1)
        
    for item in grid_data:
        r, c = item['row'], item['col']
        p, team, val = item['player'], item['team'], item['value']
        
        x_center = (c + 0.5) * w
        y_center = (r + 0.5) * h
        
        parts = p.split()
        short_name = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else p
        team.split(" ")[0][:3].upper()
        
        # Adjust text positions to fit logo
        pitch.text(x_center, y_center+6, short_name, ax=ax, ha='center', va='center', color=TACTIQ_ACCENT, fontsize=9, fontweight='bold', zorder=3)
        # pitch.text(x_center, y_center-2, short_team, ax=ax, ha='center', va='center', color='gray', fontsize=7, zorder=3)
        pitch.text(x_center, y_center-4, f"{val:.2f} xT", ax=ax, ha='center', va='center', color=TACTIQ_TEXT, fontsize=7, alpha=0.7, zorder=3)
        
        # Plot Logo instead of text team name
        _plot_logo(ax, team, x_center, y_center, zoom=0.025)
        
    # Fix Header Alignment
    # Adjust y position to not overlap with pitch (pitch usually takes up to 0.9 or so in draw() default? no, draw() makes tight layout usually)
    # Pitch draw often leaves little margin. Let's assume standard fig coords (0,0 bottom left).
    fig.text(0.5, 1.06, f"Zonal Threat Kings | {league_name}", 
             color=TACTIQ_TEXT, fontsize=24, fontweight='bold', ha='center', va='bottom')
    fig.text(0.5, 1.01, "Top Threat Creator (Total xT Generated) in each zone", 
             color=TACTIQ_TEXT, fontsize=14, ha='center', va='bottom', alpha=0.9)
             
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

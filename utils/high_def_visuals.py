
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
import base64
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
from shared.logger import get_logger

logger = get_logger(__name__)

# Theme
TACTIQ_BG = "#1e1e1e"
TACTIQ_TEXT = "#ffffff"
TACTIQ_ACCENT = "#00ff87" 

def generate_high_def_map(summary_df: pd.DataFrame, event_details: Dict[str, List[Dict[str, Any]]], league_name: str = "Süper Lig") -> Optional[str]:
    """Generate 4x3 Grid of High Defensive Actions."""
    if summary_df.empty: return None
        
    top_12 = summary_df.head(12)
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='white', line_zorder=2)
    fig, axs = pitch.grid(nrows=3, ncols=4, title_height=0.08, endnote_space=0, grid_width=0.9, grid_height=0.80, axis=False)
    fig.set_facecolor(TACTIQ_BG)
    fig.set_size_inches(16, 12)
    
    for idx, (i, row) in enumerate(top_12.iterrows()):
        ax = axs['pitch'].flatten()[idx]
        events = event_details.get(row['name'], [])
        if not events: continue
            
        x = np.array([e['x'] for e in events])
        y = np.array([e['y'] for e in events])
        
        try:
            pitch.kdeplot(x, y, ax=ax, fill=True, levels=100, cmap='viridis', alpha=0.6, cut=3)
        except Exception as e:
            logger.debug("High def KDE skipped: %s", e)
        
        pitch.scatter(x, y, color='white', alpha=0.4, s=15, ax=ax, zorder=2)
        ax.plot([67, 67], [0, 100], ls='--', color='white', alpha=0.5, lw=1)
        
        ax.set_title(f"{idx+1}. {row['name']}", color=TACTIQ_TEXT, fontsize=10, ha='left', x=0.05, y=0.95)
        ax.text(2, 5, f"Actions: {row['high_def_total']}", color=TACTIQ_TEXT, fontsize=9, va='bottom', ha='left', zorder=5)

    axs['title'].text(0.5, 0.5, f"High Defensive Efficiency (Final Third) | {league_name}", 
                      color=TACTIQ_TEXT, fontsize=20, fontweight='bold', ha='center', va='center')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=TACTIQ_BG)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

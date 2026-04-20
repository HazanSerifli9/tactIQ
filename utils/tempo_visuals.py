import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO
import base64
from typing import Dict, Any
from mplsoccer import Pitch

from utils.visuals import TACTIQ_BG, TACTIQ_FG, TACTIQ_ACCENT

def plot_tempo_network(tempo_data: Dict[str, Any], team_name: str) -> str:
    """
    Render the Tempo Network onto a pitch.
    Returns base64 PNG string.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG, line_alpha=0.3, linewidth=1.5, corner_arcs=True)
    pitch.draw(ax=ax)
    
    nodes = tempo_data.get('nodes', {})
    edges = tempo_data.get('edges', [])
    
    if not nodes or not edges:
        ax.text(50, 50, 'No tempo data available', ha='center', va='center', color=TACTIQ_FG)
        return _fig_to_base64(fig)
        
    # Get max volume to scale edge thickness
    max_count = max([e['count'] for e in edges]) if edges else 1
    
    # Custom colormap for Tempo: Red (Fast) -> Yellow (Medium) -> Blue (Slow)
    cdict = {'red':  ((0.0, 0.93, 0.93), (0.5, 0.98, 0.98), (1.0, 0.23, 0.23)),
             'green':((0.0, 0.26, 0.26), (0.5, 0.74, 0.74), (1.0, 0.50, 0.50)),
             'blue': ((0.0, 0.26, 0.26), (0.5, 0.14, 0.14), (1.0, 0.96, 0.96))}
    tempo_cmap = mcolors.LinearSegmentedColormap('TempoCmap', cdict)
    
    # Filter edges (e.g. min 3-5 passes)
    min_edges_threshold = max(2, max_count * 0.1) # At least 2 passes
    valid_edges = [e for e in edges if e['count'] >= min_edges_threshold and e['sender'] in nodes and e['receiver'] in nodes]
    
    for edge in valid_edges:
        n1 = nodes[edge['sender']]
        n2 = nodes[edge['receiver']]
        
        ttrp = edge['avg_ttrp']
        carry = edge['avg_carry_x']
        count = edge['count']
        
        # Color based on TTRP: Clip between 2.5s and 4.0s
        norm_ttrp = max(0, min((ttrp - 2.5) / 1.5, 1.0))
        color = tempo_cmap(norm_ttrp)
        
        # Line width based on volume
        lw = (count / max_count) * 6 + 1
        
        # Draw the tubular edge
        arrow = patches.FancyArrowPatch((n1['x'], n1['y']), (n2['x'], n2['y']),
                                        connectionstyle="arc3,rad=0.1",
                                        arrowstyle="->,head_length=5,head_width=3",
                                        color=color, alpha=0.5, lw=lw, zorder=1)
        ax.add_patch(arrow)
        
        # Overlay Carry Displacement if carry is significant (>3.0m forward)
        if carry > 3.0:
            carry_arrow = patches.FancyArrowPatch((n1['x'], n1['y']), (n2['x'], n2['y']),
                                        connectionstyle="arc3,rad=0.1",
                                        linestyle='--', color='#22c55e', alpha=0.8, lw=lw*0.5, zorder=2)
            ax.add_patch(carry_arrow)

    # Draw Nodes
    for player, stats in nodes.items():
        x, y = stats['x'], stats['y']
        jersey = stats.get('jersey_number')
        
        if jersey is not None:
            label = str(int(jersey))
        else:
            label = "".join([n[0] for n in player.split()[:2]]).upper()
            
        # Role color logic (we pass profiles to get roles)
        role = "Connector"
        for p in tempo_data.get('profiles', []):
            if p['Player'] == player:
                role = p['Role']
                break
                
        border_color = '#ffffff'
        if role == 'Metronome': border_color = '#3b82f6'
        elif role == 'Direct': border_color = '#ef4444'
        elif role == 'Recycler': border_color = '#a0aec0'
        elif role == 'Connector': border_color = '#fbbf24'
        
        # Node circle
        circle = patches.Circle((x, y), radius=2.5, facecolor='#111827', edgecolor=border_color, lw=2, zorder=4)
        ax.add_patch(circle)
        
        # Initials/Number text
        ax.text(x, y, label, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=5)

    # Legends & Titles
    clean_name = team_name.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').strip()
    ax.text(50, -4, f"Tempo Network | Speed of Play — {clean_name}", ha='center', fontsize=12, color=TACTIQ_FG, fontweight='bold')
    ax.text(50, -7, "Edge color = TTRP | Thickness = Volume | Dashed Green = Carry", ha='center', fontsize=9, color='#a0aec0')
    
    # Colormap legend bar
    sm = plt.cm.ScalarMappable(cmap=tempo_cmap, norm=plt.Normalize(vmin=2.5, vmax=4.0))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.1, aspect=40)
    cbar.set_ticks([2.5, 4.0])
    cbar.set_ticklabels(['Fast (<2.5s)', 'Slow (>4.0s)'])
    cbar.ax.tick_params(colors=TACTIQ_FG, labelsize=8)
    cbar.outline.set_edgecolor('#333')
    
    plt.tight_layout()
    return _fig_to_base64(fig)

def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

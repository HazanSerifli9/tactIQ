"""
Ball Trace Visuals
===================
Visualization functions for Ball Trace territorial analysis.
Uses matplotlib for consistency with existing TactIQ visual style.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
import numpy as np
from io import BytesIO
import base64
from typing import Dict, Any, Optional

# TactIQ Theme Colors
BG_COLOR = '#0a0e17'
CARD_BG = '#111827'
TEXT_COLOR = '#e5e7eb'
ACCENT = '#fbbf24'
ACCENT_2 = '#0ea5e9'
GRID_COLOR = (1.0, 1.0, 1.0, 0.08)

# Zone color scale (low → high ball time)
ZONE_CMAP = plt.cm.YlOrRd  # Yellow → Orange → Red


def _draw_pitch(ax, color_lines='#333', alpha=0.5):
    """Draw a football pitch on the given axes (0-100 coordinate system)."""
    # Pitch outline
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=color_lines, alpha=alpha, lw=1.5)
    # Centre line
    ax.plot([50, 50], [0, 100], color=color_lines, alpha=alpha, lw=1)
    # Centre circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(50 + 9.15*np.cos(theta), 50 + 9.15*np.sin(theta), color=color_lines, alpha=alpha, lw=1)
    # Penalty areas
    ax.plot([0, 16.5, 16.5, 0], [21.1, 21.1, 78.9, 78.9], color=color_lines, alpha=alpha, lw=1)
    ax.plot([100, 83.5, 83.5, 100], [21.1, 21.1, 78.9, 78.9], color=color_lines, alpha=alpha, lw=1)
    # 6-yard boxes
    ax.plot([0, 5.5, 5.5, 0], [36.8, 36.8, 63.2, 63.2], color=color_lines, alpha=alpha, lw=1)
    ax.plot([100, 94.5, 94.5, 100], [36.8, 36.8, 63.2, 63.2], color=color_lines, alpha=alpha, lw=1)
    # Centre spot
    ax.scatter([50], [50], color=color_lines, s=15, alpha=alpha, zorder=3)
    # Penalty spots
    ax.scatter([11.5, 88.5], [50, 50], color=color_lines, s=10, alpha=alpha, zorder=3)
    
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_ball_time_map(trace_data: Dict[str, Any], team_name: str, 
                        opponent_name: str = '') -> str:
    """
    Draw a 3x3 zone heatmap on a pitch showing how long the ball stayed in each zone.
    Returns base64 encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    _draw_pitch(ax, color_lines='#2a3040', alpha=0.6)
    
    zone_details = trace_data.get('zone_details', [])
    if not zone_details:
        ax.text(50, 50, 'No data', ha='center', va='center', color=TEXT_COLOR, fontsize=14)
        return _fig_to_base64(fig)
    
    # Get max minutes for color normalization
    max_min = max(z['minutes'] for z in zone_details) if zone_details else 1
    if max_min == 0:
        max_min = 1
    
    # Zone boundaries
    x_bounds = [0, 33.33, 66.66, 100]
    y_bounds = [0, 33.33, 66.66, 100]
    
    for z in zone_details:
        xi, yi = z['x_idx'], z['y_idx']
        x_start = x_bounds[xi]
        y_start = y_bounds[yi]
        width = x_bounds[xi+1] - x_bounds[xi]
        height = y_bounds[yi+1] - y_bounds[yi]
        
        # Color intensity based on minutes
        intensity = z['minutes'] / max_min
        color = ZONE_CMAP(intensity * 0.8)  # cap at 0.8 to avoid pure white
        
        rect = patches.FancyBboxPatch(
            (x_start + 0.5, y_start + 0.5), width - 1, height - 1,
            boxstyle=patches.BoxStyle.Round(pad=1),
            facecolor=color, alpha=0.55, edgecolor='none',
            zorder=2
        )
        ax.add_patch(rect)
        
        # Text: minutes and percentage
        cx = x_start + width / 2
        cy = y_start + height / 2
        
        if z['minutes'] > 0:
            ax.text(cx, cy + 3, f"{z['minutes']:.1f}'",
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white', zorder=5)
            ax.text(cx, cy - 6, f"{z['pct']:.0f}%",
                    ha='center', va='center', fontsize=11,
                    color=(1.0, 1.0, 1.0, 0.7), zorder=5)
    
    # Draw 3x3 grid lines
    for x in [33.33, 66.66]:
        ax.plot([x, x], [0, 100], color='white', alpha=0.15, lw=1, ls='--', zorder=3)
    for y in [33.33, 66.66]:
        ax.plot([0, 100], [y, y], color='white', alpha=0.15, lw=1, ls='--', zorder=3)
    
    # Labels
    clean_name = team_name.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').strip()
    
    # Title
    ax.text(50, 106, f"BALL TRACE — {clean_name.upper()}",
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=ACCENT, zorder=5)
    
    total_time = trace_data.get('total_ball_time_min', 0)
    terr_dom = trace_data.get('territorial_dominance', 50)
    ax.text(50, -6, f"Total Ball Time: {total_time:.1f}' | Territorial Dominance: {terr_dom:.0f}%",
            ha='center', va='center', fontsize=10, color=TEXT_COLOR, alpha=0.7, zorder=5)
    
    # Zone labels on edges
    ax.text(16.5, -3, 'DEF', ha='center', fontsize=9, color='#666', zorder=5)
    ax.text(50, -3, 'MID', ha='center', fontsize=9, color='#666', zorder=5)
    ax.text(83.3, -3, 'ATK', ha='center', fontsize=9, color='#666', zorder=5)
    ax.text(-1, 16.5, 'L', ha='center', fontsize=9, color='#666', rotation=90, zorder=5)
    ax.text(-1, 50, 'C', ha='center', fontsize=9, color='#666', rotation=90, zorder=5)
    ax.text(-1, 83.3, 'R', ha='center', fontsize=9, color='#666', rotation=90, zorder=5)
    
    # Attack direction arrow
    ax.annotate('', xy=(98, -8), xytext=(80, -8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    ax.text(89, -11, 'ATK →', ha='center', fontsize=8, color='#555')
    
    return _fig_to_base64(fig)


def plot_thirds_flanks_bars(trace_data: Dict[str, Any], team_name: str) -> str:
    """
    Draw side-by-side bar charts for Third distribution and Flank distribution.
    Returns base64 encoded PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    
    clean_name = team_name.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').strip()
    
    # --- Thirds Bar ---
    thirds = trace_data.get('thirds', {})
    thirds_pct = trace_data.get('thirds_pct', {})
    
    labels_t = ['Defensive', 'Middle', 'Attacking']
    keys_t = ['defensive', 'middle', 'attacking']
    vals_t = [thirds.get(k, 0) for k in keys_t]
    pcts_t = [thirds_pct.get(k, 0) for k in keys_t]
    colors_t = ['#3b82f6', '#a78bfa', '#ef4444']
    
    ax1.set_facecolor(CARD_BG)
    bars1 = ax1.barh(labels_t, vals_t, color=colors_t, height=0.55, edgecolor='none', alpha=0.85)
    
    for bar, pct, val in zip(bars1, pcts_t, vals_t):
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2,
                f"  {val:.1f}' ({pct:.0f}%)", va='center', fontsize=10, color=TEXT_COLOR)
    
    ax1.set_title('Third Distribution', fontsize=12, color=ACCENT, fontweight='bold', pad=10)
    ax1.set_xlabel('Minutes', fontsize=9, color='#666')
    ax1.tick_params(colors='#888', labelsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_color('#333')
    ax1.spines['left'].set_color('#333')
    
    # --- Flanks Bar ---
    flanks = trace_data.get('flanks', {})
    flanks_pct = trace_data.get('flanks_pct', {})
    
    labels_f = ['Left', 'Center', 'Right']
    keys_f = ['left', 'center', 'right']
    vals_f = [flanks.get(k, 0) for k in keys_f]
    pcts_f = [flanks_pct.get(k, 0) for k in keys_f]
    colors_f = ['#22c55e', '#eab308', '#f97316']
    
    ax2.set_facecolor(CARD_BG)
    bars2 = ax2.barh(labels_f, vals_f, color=colors_f, height=0.55, edgecolor='none', alpha=0.85)
    
    for bar, pct, val in zip(bars2, pcts_f, vals_f):
        w = bar.get_width()
        ax2.text(w + 0.1, bar.get_y() + bar.get_height()/2,
                f"  {val:.1f}' ({pct:.0f}%)", va='center', fontsize=10, color=TEXT_COLOR)
    
    ax2.set_title('Flank Distribution', fontsize=12, color=ACCENT_2, fontweight='bold', pad=10)
    ax2.set_xlabel('Minutes', fontsize=9, color='#666')
    ax2.tick_params(colors='#888', labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#333')
    ax2.spines['left'].set_color('#333')
    
    fig.suptitle(f'{clean_name} — Ball Time Distribution', fontsize=13,
                 color=TEXT_COLOR, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_ball_timeline(trace_data: Dict[str, Any], team_name: str) -> str:
    """
    Draw a horizontal timeline showing average ball position per minute.
    Color-coded by third (Def=blue, Mid=purple, Atk=red).
    Returns base64 encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 2.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    timeline = trace_data.get('timeline', [])
    if not timeline:
        ax.text(0.5, 0.5, 'No timeline data', transform=ax.transAxes,
                ha='center', va='center', color=TEXT_COLOR, fontsize=12)
        return _fig_to_base64(fig)
    
    minutes = [t['minute'] for t in timeline]
    avg_xs = [t['avg_x'] for t in timeline]
    
    # Color based on third
    color_map = {
        'Defensive': '#3b82f6',
        'Middle': '#a78bfa',
        'Attacking': '#ef4444',
    }
    colors = [color_map.get(t['third'], '#666') for t in timeline]
    
    # Draw bars
    for i, (m, x, c) in enumerate(zip(minutes, avg_xs, colors)):
        ax.bar(m, x, color=c, width=0.8, alpha=0.75, edgecolor='none')
    
    # Reference lines
    ax.axhline(y=50, color='white', alpha=0.15, lw=1, ls='--')
    ax.axhline(y=33.33, color='#3b82f6', alpha=0.2, lw=0.5, ls=':')
    ax.axhline(y=66.66, color='#ef4444', alpha=0.2, lw=0.5, ls=':')
    
    # Labels
    ax.text(-0.02, 50, 'Half', transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=8, color='#666')
    ax.text(-0.02, 33.33, 'Def', transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=7, color='#3b82f6')
    ax.text(-0.02, 66.66, 'Atk', transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=7, color='#ef4444')
    
    clean_name = team_name.replace(' Kulübü', '').replace(' Spor', '').replace(' Futbol', '').strip()
    ax.set_title(f'{clean_name} — Ball Position Timeline (avg X per minute)',
                 fontsize=11, color=TEXT_COLOR, fontweight='bold', pad=8)
    ax.set_xlabel('Minute', fontsize=9, color='#666')
    ax.set_ylabel('Avg Position', fontsize=9, color='#666')
    ax.set_ylim(0, 100)
    ax.tick_params(colors='#888', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(),
                dpi=120, pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

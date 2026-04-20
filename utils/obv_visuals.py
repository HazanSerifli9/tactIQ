"""
OBV Visualizations
====================
Renders On-Ball Value pitch annotations and leaderboard charts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO
import base64
import pandas as pd
from typing import Optional
from mplsoccer import Pitch

from utils.visuals import TACTIQ_BG, TACTIQ_FG, TACTIQ_ACCENT, TACTIQ_ACCENT_SEC
from utils.obv_model import get_player_obv_summary

# OBV diverging colormap: red (negative) → grey → green (positive)
OBV_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'OBV', ['#ef4444', '#374151', '#22c55e']
)


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_obv_pitch(df: pd.DataFrame, team_name: str) -> str:
    """
    Scatter each event on a pitch, coloured and sized by OBV_Net.
    Returns base64 PNG.
    """
    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty or 'obv_net' not in team_df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(50, 50, 'No OBV data — train models first', ha='center', color=TACTIQ_FG)
        return _fig_to_base64(fig)

    # Filter to events with spatial coordinates
    team_df = team_df[team_df['x'].notna() & team_df['y'].notna()].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='#6b7280', line_alpha=0.4, linewidth=1.5, corner_arcs=True)
        pitch.draw(ax=ax)
        ax.text(50, 50, 'No spatial data', ha='center', color=TACTIQ_FG)
        return _fig_to_base64(fig)

    # Normalise OBV for color mapping
    obv_min = team_df['obv_net'].quantile(0.05)
    obv_max = team_df['obv_net'].quantile(0.95)
    norm = mcolors.TwoSlopeNorm(vmin=obv_min, vcenter=0, vmax=max(obv_max, 0.01))

    # Scale marker size by absolute OBV
    sizes = (team_df['obv_net'].abs() / max(team_df['obv_net'].abs().max(), 1e-5)) * 80 + 8

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)
    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color='#6b7280', line_alpha=0.4, linewidth=1.5, corner_arcs=True)
    pitch.draw(ax=ax)

    sc = ax.scatter(
        team_df['x'], team_df['y'],
        c=team_df['obv_net'],
        s=sizes,
        cmap=OBV_CMAP,
        norm=norm,
        alpha=0.72,
        edgecolors='none',
        zorder=4,
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.025, pad=0.02, aspect=35)
    cbar.set_label('OBV_Net', color=TACTIQ_FG, fontsize=9)
    cbar.ax.tick_params(colors=TACTIQ_FG, labelsize=8)
    cbar.outline.set_edgecolor('#444')

    # Title
    clean = team_name.replace(' Kulübü', '').replace(' Spor', '').strip()
    ax.set_title(
        f'{clean}  |  On-Ball Value Pitch Map',
        color=TACTIQ_FG, fontsize=13, fontweight='bold', pad=10
    )
    ax.text(50, -2.5, 'Green = Positive OBV  |  Red = Negative OBV  |  Size ∝ Magnitude',
            ha='center', va='top', color='#9ca3af', fontsize=8.5)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_obv_leaderboard(df: pd.DataFrame, team_name: str, top_n: int = 10) -> str:
    """
    Horizontal bar chart showing per-player OBV_Net ranked for the team.
    Bars are split into Pass / Carry / Defensive components.
    Returns base64 PNG.
    """
    summary = get_player_obv_summary(df, team_name)

    if summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(TACTIQ_BG)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, 'OBV models not trained yet.\nRun: python -m utils.obv_model',
                ha='center', va='center', color=TACTIQ_FG, fontsize=11,
                transform=ax.transAxes)
        ax.axis('off')
        return _fig_to_base64(fig)

    # Shorten player names
    def shorten(name):
        if not isinstance(name, str):
            return str(name)
        parts = name.split()
        if len(parts) == 1:
            return name
        return parts[0][0] + '. ' + parts[-1]

    summary = summary.head(top_n).copy()
    summary['short'] = summary['Player'].apply(shorten)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    y_pos = np.arange(len(summary))
    bar_h = 0.6

    # OBV_Pass (teal)
    ax.barh(y_pos, summary['OBV_Pass'], height=bar_h,
            color='#06b6d4', alpha=0.85, label='OBV_Pass', zorder=3)
    # OBV_Carry (gold) — stacked
    ax.barh(y_pos, summary['OBV_Carry'], height=bar_h,
            left=summary['OBV_Pass'], color='#f59e0b', alpha=0.85, label='OBV_Carry', zorder=3)
    # OBV_Def (purple) — stacked
    left2 = summary['OBV_Pass'] + summary['OBV_Carry']
    ax.barh(y_pos, summary['OBV_Def'], height=bar_h,
            left=left2, color='#a78bfa', alpha=0.85, label='OBV_Def', zorder=3)

    # Net OBV marker
    ax.scatter(summary['OBV_Net'], y_pos, color='white', s=35, zorder=5,
               marker='D', label='OBV_Net')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary['short'], color=TACTIQ_FG, fontsize=10)
    ax.tick_params(colors=TACTIQ_FG, labelsize=9)
    ax.axvline(0, color='#6b7280', lw=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('On-Ball Value', color=TACTIQ_FG, fontsize=10)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.grid(axis='x', color='#374151', linewidth=0.5, alpha=0.5)

    clean = team_name.replace(' Kulübü', '').replace(' Spor', '').strip()
    ax.set_title(f'{clean}  |  Player OBV Leaderboard', color=TACTIQ_FG, fontsize=13,
                 fontweight='bold', pad=10)

    legend = ax.legend(frameon=False, labelcolor=TACTIQ_FG, fontsize=9,
                       loc='lower right')

    fig.tight_layout()
    return _fig_to_base64(fig)

"""
OBV Visualizations
====================
Renders On-Ball Value pitch heat-map and player leaderboard.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from io import BytesIO
import base64
import pandas as pd
from mplsoccer import Pitch

from utils.visuals import TACTIQ_BG, TACTIQ_FG, TACTIQ_ACCENT, TACTIQ_ACCENT_SEC
from utils.obv_model import get_player_obv_summary

OBV_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'OBV', ['#ef4444', '#374151', '#22c55e']
)

COMP_COLORS = {
    'OBV_Pass':  '#06b6d4',
    'OBV_Carry': '#f59e0b',
    'OBV_Def':   '#a78bfa',
}


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _clean_name(team_name: str) -> str:
    return team_name.replace(' Kulübü', '').replace(' Spor', '').strip()


def _short_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    parts = name.split()
    if len(parts) == 1:
        return name
    return parts[0][0] + '. ' + parts[-1]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PITCH HEAT-MAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_obv_pitch(df: pd.DataFrame, team_name: str) -> str:
    """
    Binned pitch heat-map of net OBV contribution per zone.
    Shows WHERE on the pitch value was created / destroyed.
    """
    team_df = df[df['team_name'] == team_name].copy()
    has_data = (
        not team_df.empty
        and 'obv_net' in team_df.columns
        and team_df['obv_net'].notna().any()
    )

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    pitch = Pitch(
        pitch_type='opta', pitch_color=TACTIQ_BG,
        line_color='#555', line_alpha=0.5, linewidth=1.2, corner_arcs=True
    )
    pitch.draw(ax=ax)

    clean = _clean_name(team_name)
    ax.set_title(f'{clean}  |  OBV Zone Map', color=TACTIQ_FG,
                 fontsize=13, fontweight='bold', pad=10)

    if not has_data:
        ax.text(50, 50, 'No OBV data — train models first',
                ha='center', va='center', color='#9ca3af', fontsize=11)
        return _fig_to_base64(fig)

    team_df = team_df[team_df['x'].notna() & team_df['y'].notna()].copy()
    if team_df.empty:
        ax.text(50, 50, 'No spatial data', ha='center', va='center',
                color='#9ca3af', fontsize=11)
        return _fig_to_base64(fig)

    # ── Bin pitch into 10×6 zones, sum OBV per zone ──────────────────────────
    BINS_X, BINS_Y = 10, 6
    x_edges = np.linspace(0, 100, BINS_X + 1)
    y_edges = np.linspace(0, 100, BINS_Y + 1)

    xi = np.clip(np.digitize(team_df['x'], x_edges) - 1, 0, BINS_X - 1)
    yi = np.clip(np.digitize(team_df['y'], y_edges) - 1, 0, BINS_Y - 1)

    grid_sum   = np.zeros((BINS_Y, BINS_X))
    grid_count = np.zeros((BINS_Y, BINS_X))
    for i, (xi_, yi_) in enumerate(zip(xi, yi)):
        grid_sum[yi_, xi_]   += team_df['obv_net'].iloc[i]
        grid_count[yi_, xi_] += 1

    # Only show zones with enough events
    min_events = max(3, team_df.shape[0] // 50)
    grid_sum[grid_count < min_events] = np.nan

    abs_max = np.nanmax(np.abs(grid_sum))
    if abs_max < 1e-6:
        abs_max = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    cell_w = 100 / BINS_X
    cell_h = 100 / BINS_Y

    for yi_ in range(BINS_Y):
        for xi_ in range(BINS_X):
            val = grid_sum[yi_, xi_]
            if np.isnan(val):
                continue
            color = OBV_CMAP(norm(val))
            rect = mpatches.FancyBboxPatch(
                (x_edges[xi_] + 0.5, y_edges[yi_] + 0.5),
                cell_w - 1.0, cell_h - 1.0,
                boxstyle='round,pad=0.5',
                facecolor=color, edgecolor='none', alpha=0.82, zorder=2
            )
            ax.add_patch(rect)
            # Value label for strong cells
            if abs(val) > abs_max * 0.35:
                cx = x_edges[xi_] + cell_w / 2
                cy = y_edges[yi_] + cell_h / 2
                ax.text(cx, cy, f'{val:+.2f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold', zorder=5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=OBV_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.022, pad=0.02, aspect=32)
    cbar.set_label('Net OBV', color=TACTIQ_FG, fontsize=9)
    cbar.ax.tick_params(colors=TACTIQ_FG, labelsize=8)
    cbar.outline.set_edgecolor('#444')

    ax.text(50, -3.5,
            'Each zone = sum of OBV_Net for all events in that area  '
            '|  Green = value created  |  Red = value lost',
            ha='center', va='top', color='#6b7280', fontsize=8)

    return _fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PLAYER OBV LEADERBOARD
# ──────────────────────────────────────────────────────────────────────────────

def plot_obv_leaderboard(df: pd.DataFrame, team_name: str, top_n: int = 10) -> str:
    """
    Diverging bar chart of player OBV_Net (sorted best→worst).
    Right panel: component breakdown (Pass / Carry / Def) as mini stacked bars.
    """
    summary = get_player_obv_summary(df, team_name)

    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(TACTIQ_BG)

    if summary.empty:
        ax = fig.add_subplot(111)
        ax.set_facecolor(TACTIQ_BG)
        ax.text(0.5, 0.5, 'OBV models not trained yet.\nRun: python -m utils.obv_model',
                ha='center', va='center', color=TACTIQ_FG, fontsize=11,
                transform=ax.transAxes)
        ax.axis('off')
        return _fig_to_base64(fig)

    summary = summary.head(top_n).copy()
    summary['short'] = summary['Player'].apply(_short_name)
    summary = summary.sort_values('OBV_Net', ascending=True)

    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1])
    ax_net  = fig.add_subplot(gs[0])   # diverging net bar
    ax_comp = fig.add_subplot(gs[1])   # component breakdown

    for ax in (ax_net, ax_comp):
        ax.set_facecolor(TACTIQ_BG)
        ax.spines[:].set_visible(False)

    y     = np.arange(len(summary))
    bar_h = 0.55
    clean = _clean_name(team_name)

    # ── Left panel: OBV_Net diverging bars ───────────────────────────────────
    colors_net = ['#22c55e' if v >= 0 else '#ef4444'
                  for v in summary['OBV_Net']]

    ax_net.barh(y, summary['OBV_Net'], height=bar_h,
                color=colors_net, edgecolor='none', alpha=0.88, zorder=3)

    ax_net.axvline(0, color='#6b7280', lw=1, linestyle='--', alpha=0.6, zorder=2)

    # Value labels
    for i, (val, yi) in enumerate(zip(summary['OBV_Net'], y)):
        offset = 0.015 if val >= 0 else -0.015
        ha     = 'left' if val >= 0 else 'right'
        ax_net.text(val + offset, yi, f'{val:+.2f}',
                    va='center', ha=ha, color=TACTIQ_FG,
                    fontsize=8.5, fontweight='bold', zorder=4)

    ax_net.set_yticks(y)
    ax_net.set_yticklabels(summary['short'], color=TACTIQ_FG, fontsize=10)
    ax_net.tick_params(axis='x', colors='#666', labelsize=8)
    ax_net.grid(axis='x', color='#ffffff18', linewidth=0.5, zorder=0)
    ax_net.set_xlabel('Net OBV', color='#888', fontsize=9)
    ax_net.set_title(f'{clean}  |  Player OBV Leaderboard\n'
                     '(sorted by net contribution)',
                     color=TACTIQ_FG, fontsize=11, fontweight='bold', pad=8)

    # ── Right panel: component breakdown (Pass / Carry / Def) ────────────────
    # Use absolute values to show what each player contributes per category
    for i, (_, row) in enumerate(summary.iterrows()):
        left = 0.0
        for comp, col in [('Pass', '#06b6d4'), ('Carry', '#f59e0b'), ('Def', '#a78bfa')]:
            val = row.get(f'OBV_{comp}', 0) or 0
            if abs(val) > 1e-4:
                ax_comp.barh(i, val, height=bar_h, left=left,
                             color=col, edgecolor='none', alpha=0.85, zorder=3)
                left += val

    ax_comp.axvline(0, color='#6b7280', lw=1, linestyle='--', alpha=0.6, zorder=2)
    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels([])
    ax_comp.tick_params(axis='x', colors='#666', labelsize=8)
    ax_comp.grid(axis='x', color='#ffffff18', linewidth=0.5, zorder=0)
    ax_comp.set_xlabel('OBV by action type', color='#888', fontsize=9)
    ax_comp.set_title('Breakdown by Action\n ', color=TACTIQ_FG,
                      fontsize=11, fontweight='bold', pad=8)

    # Legend for right panel
    legend_handles = [
        mpatches.Patch(facecolor='#06b6d4', label='Pass', alpha=0.85),
        mpatches.Patch(facecolor='#f59e0b', label='Carry', alpha=0.85),
        mpatches.Patch(facecolor='#a78bfa', label='Defensive', alpha=0.85),
    ]
    ax_comp.legend(handles=legend_handles, loc='lower right', fontsize=8,
                   facecolor=TACTIQ_BG, edgecolor='#333', labelcolor='white',
                   framealpha=0.7)

    return _fig_to_base64(fig)

"""
Ball Trace Visuals  —  TactIQ themed redesign
=================================================
Two figures per team:
  1. plot_ball_time_map   → mplsoccer pitch + 3×3 zone heatmap
  2. plot_thirds_flanks_bars → thirds bar + flank bar + timeline line (combined)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO
import base64
from typing import Dict, Any

from utils.visuals import TACTIQ_BG, TACTIQ_FG, TACTIQ_ACCENT, TACTIQ_HOME, TACTIQ_AWAY

# Navy → gold gradient (warm intensity on dark bg)
BT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'BallTrace', ['#0d1f2d', '#1a4a6b', '#2980b9', '#f39c12', '#fbbf24']
)

ZONE_COLORS  = {'defensive': '#457b9d', 'middle': '#a78bfa', 'attacking': '#e63946'}
FLANK_COLORS = {'left': '#22c55e', 'center': '#fbbf24', 'right': '#f97316'}


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _clean(name: str) -> str:
    return (name.replace(' Kulübü', '').replace(' Spor', '')
                .replace(' Futbol', '').strip())


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ZONE HEAT-MAP on a real mplsoccer pitch
# ──────────────────────────────────────────────────────────────────────────────

def plot_ball_time_map(trace_data: Dict[str, Any], team_name: str,
                       opponent_name: str = '') -> str:
    """
    3×3 zone heat-map on an mplsoccer pitch.
    Zones colored navy→gold by time spent.
    Below: territory-dominance split bar.
    """
    from mplsoccer import Pitch

    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    fig.patch.set_facecolor(TACTIQ_BG)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[9, 1])
    ax_pitch = fig.add_subplot(gs[0])
    ax_dom   = fig.add_subplot(gs[1])

    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG,
                  line_color='#888', linewidth=1.5, corner_arcs=True,
                  line_zorder=6)
    pitch.draw(ax=ax_pitch)

    zone_details = trace_data.get('zone_details', [])
    if not zone_details:
        ax_pitch.text(50, 50, 'No data available', ha='center', va='center',
                      color=TACTIQ_FG, fontsize=12)
        ax_dom.axis('off')
        return _fig_to_base64(fig)

    max_min = max((z['minutes'] for z in zone_details), default=1) or 1
    norm    = mcolors.Normalize(vmin=0, vmax=max_min)

    x_bounds = [0, 33.33, 66.66, 100]
    y_bounds = [0, 33.33, 66.66, 100]

    for z in zone_details:
        xi, yi = z['x_idx'], z['y_idx']
        x0 = x_bounds[xi] + 0.6
        y0 = y_bounds[yi] + 0.6
        w  = (x_bounds[xi + 1] - x_bounds[xi]) - 1.2
        h  = (y_bounds[yi + 1] - y_bounds[yi]) - 1.2
        mins = z.get('minutes', 0)
        pct  = z.get('pct', 0)

        color = BT_CMAP(norm(mins))
        rect  = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle='round,pad=1.0',
            facecolor=color, edgecolor='none', alpha=0.70, zorder=2
        )
        ax_pitch.add_patch(rect)

        cx = x_bounds[xi] + (x_bounds[xi + 1] - x_bounds[xi]) / 2
        cy = y_bounds[yi] + (y_bounds[yi + 1] - y_bounds[yi]) / 2

        if mins > 0:
            ax_pitch.text(cx, cy + 4, f"{mins:.1f}'",
                          ha='center', va='center', fontsize=16,
                          fontweight='bold', color='white', zorder=5)
            ax_pitch.text(cx, cy - 5, f"{pct:.0f}%",
                          ha='center', va='center', fontsize=10,
                          color=(1.0, 1.0, 1.0, 0.6), zorder=5)

    # Grid dividers
    for x in [33.33, 66.66]:
        ax_pitch.plot([x, x], [0, 100], color='white', alpha=0.12,
                      lw=0.8, ls='--', zorder=3)
    for y in [33.33, 66.66]:
        ax_pitch.plot([0, 100], [y, y], color='white', alpha=0.12,
                      lw=0.8, ls='--', zorder=3)

    # Column labels
    for xi, label in enumerate(['DEF', 'MID', 'ATK']):
        cx = x_bounds[xi] + (x_bounds[xi + 1] - x_bounds[xi]) / 2
        ax_pitch.text(cx, -4, label, ha='center', va='top',
                      fontsize=9, color='#888')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=BT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_pitch, orientation='vertical',
                        fraction=0.020, pad=0.01, aspect=30)
    cbar.set_label("Minutes in zone", color='#aaa', fontsize=9)
    cbar.ax.tick_params(colors='#888', labelsize=8)
    cbar.outline.set_edgecolor('#444')

    clean = _clean(team_name)
    total = trace_data.get('total_ball_time_min', 0)
    ax_pitch.set_title(
        f'{clean.upper()}  —  BALL TRACE  |  {total:.1f}\' total ball time',
        color=TACTIQ_ACCENT, fontsize=13, fontweight='bold', pad=10
    )

    # ── Dominance split bar ───────────────────────────────────────────────────
    dom = trace_data.get('territorial_dominance', 50)
    ax_dom.set_facecolor(TACTIQ_BG)
    ax_dom.barh([0], [dom / 100], color=TACTIQ_HOME,
                height=0.5, edgecolor='none', alpha=0.85)
    ax_dom.barh([0], [1 - dom / 100], left=[dom / 100],
                color=TACTIQ_AWAY, height=0.5, edgecolor='none', alpha=0.85)
    ax_dom.text(dom / 200, 0, f'{dom:.0f}%',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    ax_dom.text((dom / 100 + 1) / 2, 0, f'{100 - dom:.0f}%',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    ax_dom.text(0.5, -0.55, 'Territorial Dominance  (this team vs opponent)',
                ha='center', va='top', transform=ax_dom.transAxes,
                fontsize=8, color='#888')
    ax_dom.set_xlim(0, 1)
    ax_dom.set_ylim(-0.4, 0.4)
    ax_dom.axis('off')

    return _fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FLOW DASHBOARD  (thirds + flanks bars + timeline)
# ──────────────────────────────────────────────────────────────────────────────

def plot_thirds_flanks_bars(trace_data: Dict[str, Any], team_name: str) -> str:
    """
    Combined figure:
      top-left  : thirds horizontal bars
      top-right : flanks horizontal bars
      bottom    : ball position timeline (colored line + area fill)
    """
    from matplotlib.collections import LineCollection

    clean = _clean(team_name)

    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(TACTIQ_BG)

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.9])
    ax_thirds   = fig.add_subplot(gs[0, 0])
    ax_flanks   = fig.add_subplot(gs[0, 1])
    ax_timeline = fig.add_subplot(gs[1, :])

    for ax in (ax_thirds, ax_flanks, ax_timeline):
        ax.set_facecolor(TACTIQ_BG)
        ax.spines[:].set_visible(False)

    # ── Thirds bars ───────────────────────────────────────────────────────────
    thirds     = trace_data.get('thirds', {})
    thirds_pct = trace_data.get('thirds_pct', {})
    keys_t   = ['defensive', 'middle', 'attacking']
    labels_t = ['Defensive', 'Middle', 'Attacking']
    vals_t   = [thirds.get(k, 0) for k in keys_t]
    pcts_t   = [thirds_pct.get(k, 0) for k in keys_t]

    bars1 = ax_thirds.barh(labels_t, vals_t,
                           color=[ZONE_COLORS[k] for k in keys_t],
                           height=0.5, edgecolor='none', alpha=0.88)
    for bar, pct, val in zip(bars1, pcts_t, vals_t):
        w = bar.get_width()
        ax_thirds.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                       f"  {val:.1f}'  ({pct:.0f}%)",
                       va='center', ha='left', fontsize=10, color=TACTIQ_FG)

    mx_t = max(vals_t or [1])
    ax_thirds.set_xlim(0, mx_t * 1.5)
    ax_thirds.tick_params(axis='x', colors='#555', labelsize=8)
    ax_thirds.tick_params(axis='y', colors=TACTIQ_FG, labelsize=10)
    ax_thirds.xaxis.grid(True, color='#ffffff12', linewidth=0.5)
    ax_thirds.set_axisbelow(True)
    ax_thirds.set_title('Third Distribution', color=TACTIQ_ACCENT,
                        fontsize=11, fontweight='bold', pad=8)
    ax_thirds.set_xlabel("Minutes", color='#666', fontsize=8)

    # ── Flank bars ────────────────────────────────────────────────────────────
    flanks     = trace_data.get('flanks', {})
    flanks_pct = trace_data.get('flanks_pct', {})
    keys_f   = ['left', 'center', 'right']
    labels_f = ['Left', 'Center', 'Right']
    vals_f   = [flanks.get(k, 0) for k in keys_f]
    pcts_f   = [flanks_pct.get(k, 0) for k in keys_f]

    bars2 = ax_flanks.barh(labels_f, vals_f,
                           color=[FLANK_COLORS[k] for k in keys_f],
                           height=0.5, edgecolor='none', alpha=0.88)
    for bar, pct, val in zip(bars2, pcts_f, vals_f):
        w = bar.get_width()
        ax_flanks.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                       f"  {val:.1f}'  ({pct:.0f}%)",
                       va='center', ha='left', fontsize=10, color=TACTIQ_FG)

    mx_f = max(vals_f or [1])
    ax_flanks.set_xlim(0, mx_f * 1.5)
    ax_flanks.tick_params(axis='x', colors='#555', labelsize=8)
    ax_flanks.tick_params(axis='y', colors=TACTIQ_FG, labelsize=10)
    ax_flanks.xaxis.grid(True, color='#ffffff12', linewidth=0.5)
    ax_flanks.set_axisbelow(True)
    ax_flanks.set_title('Flank Distribution', color='#38bdf8',
                        fontsize=11, fontweight='bold', pad=8)
    ax_flanks.set_xlabel("Minutes", color='#666', fontsize=8)

    # ── Timeline ─────────────────────────────────────────────────────────────
    timeline = trace_data.get('timeline', [])
    thirds_clr = {'Defensive': '#457b9d', 'Middle': '#a78bfa', 'Attacking': '#e63946'}

    if timeline:
        minutes = [t['minute'] for t in timeline]
        avg_xs  = [t['avg_x']  for t in timeline]

        # Rolling smooth
        window   = min(5, len(avg_xs))
        smoothed = np.convolve(avg_xs, np.ones(window) / window, mode='same')

        # Zone background bands
        ax_timeline.axhspan(0,     33.33, color='#457b9d', alpha=0.07, zorder=0)
        ax_timeline.axhspan(33.33, 66.66, color='#a78bfa', alpha=0.07, zorder=0)
        ax_timeline.axhspan(66.66, 100,   color='#e63946', alpha=0.07, zorder=0)

        ax_timeline.axhline(33.33, color='#457b9d', lw=0.5, alpha=0.3, ls='--')
        ax_timeline.axhline(66.66, color='#e63946', lw=0.5, alpha=0.3, ls='--')
        ax_timeline.axhline(50,    color='white',   lw=0.4, alpha=0.12, ls=':')

        # Colored segment line
        pts  = np.array([minutes, smoothed]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t_labels = [timeline[i]['third'] for i in range(len(segs))]
        seg_clrs = [thirds_clr.get(t, '#888') for t in t_labels]

        lc = LineCollection(segs, colors=seg_clrs, linewidth=2.2,
                            alpha=0.90, zorder=4)
        ax_timeline.add_collection(lc)

        # Area fill
        ax_timeline.fill_between(minutes, smoothed, 50,
                                 where=[x >= 50 for x in smoothed],
                                 color='#e63946', alpha=0.10, zorder=2)
        ax_timeline.fill_between(minutes, smoothed, 50,
                                 where=[x < 50 for x in smoothed],
                                 color='#457b9d', alpha=0.10, zorder=2)

        # HT line
        if min(minutes) <= 45 <= max(minutes):
            ax_timeline.axvline(45, color='#fbbf24', lw=0.9, alpha=0.4,
                                ls='--', zorder=5)
            ax_timeline.text(45.6, 94, 'HT', color='#fbbf24',
                             fontsize=7, va='top', alpha=0.7)

        ax_timeline.set_xlim(min(minutes), max(minutes))
        ax_timeline.set_ylim(0, 100)
        ax_timeline.set_yticks([16.67, 50, 83.33])
        ax_timeline.set_yticklabels(['DEF', 'HALF', 'ATK'],
                                    color='#888', fontsize=8)
        ax_timeline.tick_params(axis='x', colors='#666', labelsize=8)
        ax_timeline.xaxis.grid(True, color='#ffffff0d', linewidth=0.5)
        ax_timeline.set_xlabel('Minute', color='#666', fontsize=9)
        ax_timeline.set_title('Ball Position Timeline  —  avg pitch X per minute',
                              color=TACTIQ_FG, fontsize=10, fontweight='bold', pad=5)
    else:
        ax_timeline.text(0.5, 0.5, 'No timeline data',
                         ha='center', va='center',
                         transform=ax_timeline.transAxes,
                         color='#888', fontsize=11)

    fig.suptitle(f'{clean}  —  Ball Flow & Distribution',
                 color=TACTIQ_FG, fontsize=13, fontweight='bold')

    return _fig_to_base64(fig)


# backward-compat alias
def plot_ball_timeline(trace_data: Dict[str, Any], team_name: str) -> str:
    return plot_thirds_flanks_bars(trace_data, team_name)

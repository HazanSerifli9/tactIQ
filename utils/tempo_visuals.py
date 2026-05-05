import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from typing import Dict, Any
from mplsoccer import Pitch

from utils.visuals import (
    TACTIQ_BG, TACTIQ_FG, TACTIQ_ACCENT, TACTIQ_HOME,
    preprocess_for_network, get_passes_between_df,
    get_starting_xi, get_shorter_name, fig_to_base64,
)

ROLE_COLORS = {
    'Playmaker':     '#3b82f6',
    'Direct Passer': '#ef4444',
    'Safe Passer':   '#a0aec0',
    'Link Player':   '#fbbf24',
}

COACH_ROLES = {
    'Metronome': 'Playmaker',
    'Direct':    '#ef4444', # Note: actually 'Direct Passer', mapped below
    'Recycler':  'Safe Passer',
    'Connector': 'Link Player'
}
COACH_ROLES['Direct'] = 'Direct Passer'

_TEMPO_CDICT = {
    'red':  ((0.0, 0.93, 0.93), (0.5, 0.98, 0.98), (1.0, 0.23, 0.23)),
    'green':((0.0, 0.26, 0.26), (0.5, 0.74, 0.74), (1.0, 0.50, 0.50)),
    'blue': ((0.0, 0.26, 0.26), (0.5, 0.14, 0.14), (1.0, 0.96, 0.96)),
}
TEMPO_CMAP = mcolors.LinearSegmentedColormap('TempoCmap', _TEMPO_CDICT)


# ─── Hybrid Pass·Tempo Network ───────────────────────────────────────────────

def plot_hybrid_pass_network(df, team_name: str, t_data: Dict[str, Any]) -> str:
    """
    One combined visualisation merging:
      • Pass volume  → edge thickness + node size
      • Tempo (TTRP) → edge colour (red=fast, blue=slow)
      • Carry events → dashed green overlay
      • Player role  → node border colour
    Right panel shows pass stats (total, accuracy, top pair, best accuracy, forward passes).
    """
    # ── 1. Pass-network data ──────────────────────────────────────────────
    df_proc = preprocess_for_network(df)

    sort_by = ([c for c in ['period_id', 'time_min', 'time_sec', 'event_id']
                if c in df_proc.columns]
               or [c for c in ['min', 'sec'] if c in df_proc.columns])
    if sort_by:
        df_proc = df_proc.sort_values(sort_by)

    df_proc['receiver'] = df_proc['shortName'].shift(-1)

    if 'event' in df_proc.columns:
        passes_df = df_proc[(df_proc['event'] == 'Pass') & (df_proc['outcome'] == 1)]
    elif 'type_id' in df_proc.columns:
        passes_df = df_proc[(df_proc['type_id'] == 1) & (df_proc['outcome'] == 1)]
    else:
        passes_df = pd.DataFrame()

    team_all = df_proc[df_proc['team_name'] == team_name]
    top_11   = get_starting_xi(team_all, 'shortName')

    if not passes_df.empty and top_11:
        passes_df = passes_df[
            passes_df['shortName'].isin(top_11) &
            passes_df['receiver'].isin(top_11)
        ]

    passes_between_df, avg_locs_df = get_passes_between_df(team_name, passes_df)

    # ── 2. Pass stats for right panel ────────────────────────────────────
    team_df = df_proc[df_proc['team_name'] == team_name]
    if 'event' in df_proc.columns:
        all_passes  = team_df[team_df['event'] == 'Pass']
        succ_passes = all_passes[all_passes['outcome'] == 1]
    elif 'type_id' in df_proc.columns:
        all_passes  = team_df[team_df['type_id'] == 1]
        succ_passes = all_passes[all_passes['outcome'] == 1]
    else:
        all_passes = succ_passes = pd.DataFrame()

    total_passes = len(all_passes)
    pass_acc     = (len(succ_passes) / total_passes * 100) if total_passes else 0

    # Top pair
    top_pair, top_pair_count = None, 0
    if not passes_between_df.empty:
        idx = passes_between_df['pass_count'].idxmax()
        r   = passes_between_df.loc[idx]
        top_pair       = (r['pos_min'], r['pos_max'])
        top_pair_count = int(r['pass_count'])

    # Best accuracy player (≥5 passes)
    best_acc_player, best_acc_pct, best_acc_total = None, 0, 0
    if not all_passes.empty and 'shortName' in all_passes.columns:
        acc = (all_passes.groupby('shortName')
                         .agg(total=('outcome', 'count'), succ=('outcome', 'sum'))
                         .query('total >= 5'))
        if not acc.empty:
            acc['pct'] = acc['succ'] / acc['total'] * 100
            br = acc.loc[acc['pct'].idxmax()]
            best_acc_player = br.name
            best_acc_pct    = br['pct']
            best_acc_total  = int(br['total'])

    # Best forward passer (passes ending past half, x > 60 statsbomb)
    best_fwd_player, best_fwd_count = None, 0
    if not succ_passes.empty and 'end_x_scaled' in succ_passes.columns:
        fwd = succ_passes[succ_passes['end_x_scaled'] > 60]
        if not fwd.empty:
            fc = fwd.groupby('shortName').size()
            best_fwd_player = fc.idxmax()
            best_fwd_count  = int(fc.max())

    # ── 3. Tempo lookup maps ──────────────────────────────────────────────
    t_nodes    = t_data.get('nodes', {})
    t_edges    = t_data.get('edges', [])
    t_profiles = t_data.get('profiles', [])

    # shortName → role  (try last-name match or initials match)
    role_map = {}
    for p in t_profiles:
        full = p.get('Player', '')
        role = p.get('Role', 'Connector')
        role_map[get_shorter_name(full)] = role
        # also store by last-name fragment
        parts = full.split()
        if parts:
            role_map[parts[-1]] = role

    # (shortA, shortB) sorted tuple → avg_ttrp
    ttrp_map: Dict[tuple, float] = {}
    carry_map: Dict[tuple, float] = {}
    for e in t_edges:
        sa = get_shorter_name(e.get('sender',   ''))
        sb = get_shorter_name(e.get('receiver', ''))
        if sa and sb:
            key = tuple(sorted([sa, sb]))
            ttrp_map[key]  = e.get('avg_ttrp',    3.25)
            carry_map[key] = e.get('avg_carry_x',  0.0)

    # ── 4. Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor=TACTIQ_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig, wspace=0.02)
    ax_p = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1])

    ax_p.set_facecolor(TACTIQ_BG)
    ax_s.set_facecolor(TACTIQ_BG)
    ax_s.axis('off')

    pitch = Pitch(
        pitch_type='statsbomb', pitch_color=TACTIQ_BG,
        line_color=TACTIQ_FG, line_alpha=0.35,
        linewidth=1.2, corner_arcs=True,
    )
    pitch.draw(ax=ax_p)

    # ── 5. Edges ──────────────────────────────────────────────────────────
    if not passes_between_df.empty:
        max_cnt = passes_between_df['pass_count'].max()

        for _, row in passes_between_df.iterrows():
            p1, p2 = row['pos_min'], row['pos_max']
            cnt    = row['pass_count']
            lw     = 1.5 + (cnt / max_cnt) * 9

            key       = tuple(sorted([p1, p2]))
            ttrp      = ttrp_map.get(key, 3.25)
            carry_val = carry_map.get(key, 0.0)
            norm_t    = max(0.0, min((ttrp - 2.5) / 1.5, 1.0))
            color     = TEMPO_CMAP(norm_t)
            alpha     = 0.25 + (cnt / max_cnt) * 0.65

            x1, y1 = row['pass_avg_x'],     row['pass_avg_y']
            x2, y2 = row['pass_avg_x_end'], row['pass_avg_y_end']

            # Main directed edge
            arrow = patches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle='arc3,rad=0.08',
                arrowstyle=f'->,head_length={lw * 0.6},head_width={lw * 0.4}',
                color=(*color[:3], alpha), lw=lw, zorder=2,
            )
            ax_p.add_patch(arrow)

            # Carry overlay (dashed green) when significant
            if carry_val > 3.0:
                carry_arrow = patches.FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    connectionstyle='arc3,rad=0.08',
                    linestyle='--', color='#22c55e',
                    alpha=0.75, lw=lw * 0.45, zorder=3,
                )
                ax_p.add_patch(carry_arrow)

    # ── 6. Nodes ──────────────────────────────────────────────────────────
    if not avg_locs_df.empty:
        MAX_SZ, MIN_SZ = 1400, 250
        max_c = avg_locs_df['count'].max()

        for player, row in avg_locs_df.iterrows():
            x, y = row['pass_avg_x'], row['pass_avg_y']
            sz   = MIN_SZ + (row['count'] / max_c) * (MAX_SZ - MIN_SZ)

            short  = get_shorter_name(player)
            orig_role = role_map.get(short, role_map.get(player.split()[-1], 'Connector'))
            role = COACH_ROLES.get(orig_role, orig_role)
            border = ROLE_COLORS.get(role, '#fbbf24')

            # Soft glow ring
            pitch.scatter(x, y, s=sz * 1.8, color=border, alpha=0.10, ax=ax_p, zorder=3)
            # Main node
            pitch.scatter(x, y, s=sz, color='#0f172a', edgecolors=border,
                          linewidths=2.5, ax=ax_p, zorder=4)
            # Label
            pitch.annotate(short, xy=(x, y), c='white',
                           va='center', ha='center', size=8,
                           weight='bold', ax=ax_p, zorder=5)

    # ── 7. Pitch title + legend ───────────────────────────────────────────
    clean = team_name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()
    ax_p.set_title(f'{clean}  |  Pass · Tempo Network',
                   color=TACTIQ_FG, fontsize=13, fontweight='bold', pad=8)

    # Role legend (bottom of pitch panel)
    from matplotlib.lines import Line2D
    legend_elements = []
    for role, rc in ROLE_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=role,
                                      markerfacecolor='#0f172a', markeredgecolor=rc,
                                      markersize=12, markeredgewidth=2, linestyle='None'))
    
    ax_p.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06),
                ncol=4, frameon=False, fontsize=9, labelcolor='white')

    # Speed colorbar
    sm   = plt.cm.ScalarMappable(cmap=TEMPO_CMAP, norm=plt.Normalize(2.5, 4.0))
    cbar = fig.colorbar(sm, ax=ax_p, orientation='horizontal',
                        fraction=0.025, pad=0.15, aspect=50)
    cbar.set_ticks([2.5, 4.0])
    cbar.set_ticklabels(['Fast (<2.5 s)', 'Slow (>4.0 s)'])
    cbar.ax.tick_params(colors=TACTIQ_FG, labelsize=7)
    cbar.outline.set_edgecolor('#333')

    # ── 8. Stats panel ────────────────────────────────────────────────────
    T = ax_s.transAxes  # shorthand

    def _txt(x, y, s, size=11, color='white', weight='normal', alpha=1.0):
        ax_s.text(x, y, s, ha='center', va='center', color=color,
                  fontsize=size, fontweight=weight, alpha=alpha, transform=T)

    def _hline(y):
        ax_s.plot([0.05, 0.95], [y, y], color='#2d3748', lw=0.8,
                  transform=T, solid_capstyle='butt')

    # Vertical separator
    ax_s.plot([0.02, 0.02], [0.0, 1.0], color='#2d3748', lw=1, transform=T)

    _txt(0.5, 0.95, 'Total Pass Count', size=9, color='#9ca3af')
    _txt(0.5, 0.86, str(total_passes), size=30, weight='bold')
    _hline(0.80)

    _txt(0.5, 0.76, 'Pass Accuracy', size=9, color='#9ca3af')
    _txt(0.5, 0.67, f'{pass_acc:.1f}%', size=30, weight='bold')
    _hline(0.61)

    if top_pair:
        _txt(0.5, 0.57, 'Most passes played between:', size=9, color='#9ca3af')
        _txt(0.5, 0.50, f'{top_pair[0]}', size=10, weight='bold')
        _txt(0.5, 0.45, f'Total  {top_pair_count}', size=9, color='#fbbf24', weight='bold')
        _txt(0.5, 0.40, f'{top_pair[1]}', size=10, weight='bold')
        _hline(0.35)

    # Two-column sub-stats
    if best_acc_player:
        _txt(0.27, 0.31, 'Passes Succeeded', size=8, color='#9ca3af')
        _txt(0.27, 0.25, best_acc_player, size=9, weight='bold')
        _txt(0.27, 0.18, f'{best_acc_pct:.0f}%', size=20, weight='bold')
        _txt(0.27, 0.12, f'Total : {best_acc_total}', size=8, color='#9ca3af')

    if best_fwd_player:
        _txt(0.73, 0.31, 'Passes Above Half', size=8, color='#9ca3af')
        _txt(0.73, 0.25, best_fwd_player, size=9, weight='bold')
        _txt(0.73, 0.18, f'{best_fwd_count}', size=20, weight='bold')
        _txt(0.73, 0.12, 'successful fwd passes', size=8, color='#9ca3af')

    _txt(0.5, 0.04,
         'Edge colour: Fast (Red) → Slow (Blue)  |  Dashed = Carry',
         size=7, color='#4b5563')

    plt.tight_layout()
    return _fig_to_base64(fig)


# ─── Original Tempo Network (kept for backward compat) ───────────────────────

def plot_tempo_network(tempo_data: Dict[str, Any], team_name: str) -> str:
    """Render the Tempo Network onto a pitch. Returns base64 PNG string."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax.set_facecolor(TACTIQ_BG)

    pitch = Pitch(pitch_type='opta', pitch_color=TACTIQ_BG, line_color=TACTIQ_FG,
                  line_alpha=0.3, linewidth=1.5, corner_arcs=True)
    pitch.draw(ax=ax)

    nodes = tempo_data.get('nodes', {})
    edges = tempo_data.get('edges', [])

    if not nodes or not edges:
        ax.text(50, 50, 'No tempo data available', ha='center', va='center', color=TACTIQ_FG)
        return _fig_to_base64(fig)

    max_count = max([e['count'] for e in edges]) if edges else 1
    min_threshold = max(2, max_count * 0.1)
    valid_edges = [e for e in edges
                   if e['count'] >= min_threshold
                   and e['sender'] in nodes and e['receiver'] in nodes]

    for edge in valid_edges:
        n1, n2 = nodes[edge['sender']], nodes[edge['receiver']]
        ttrp  = edge['avg_ttrp']
        carry = edge['avg_carry_x']
        count = edge['count']

        norm_ttrp = max(0, min((ttrp - 2.5) / 1.5, 1.0))
        color = TEMPO_CMAP(norm_ttrp)
        lw    = (count / max_count) * 6 + 1

        arrow = patches.FancyArrowPatch(
            (n1['x'], n1['y']), (n2['x'], n2['y']),
            connectionstyle='arc3,rad=0.1',
            arrowstyle='->,head_length=5,head_width=3',
            color=color, alpha=0.5, lw=lw, zorder=1,
        )
        ax.add_patch(arrow)

        if carry > 3.0:
            carry_arrow = patches.FancyArrowPatch(
                (n1['x'], n1['y']), (n2['x'], n2['y']),
                connectionstyle='arc3,rad=0.1',
                linestyle='--', color='#22c55e', alpha=0.8, lw=lw * 0.5, zorder=2,
            )
            ax.add_patch(carry_arrow)

    for player, stats in nodes.items():
        x, y   = stats['x'], stats['y']
        jersey = stats.get('jersey_number')
        label  = str(int(jersey)) if jersey is not None else ''.join(
            [n[0] for n in player.split()[:2]]).upper()

        orig_role = 'Connector'
        for p in tempo_data.get('profiles', []):
            if p['Player'] == player:
                orig_role = p['Role']
                break

        role = COACH_ROLES.get(orig_role, orig_role)
        border_color = ROLE_COLORS.get(role, '#ffffff')
        circle = patches.Circle((x, y), radius=2.5, facecolor='#111827',
                                 edgecolor=border_color, lw=2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', color='white',
                fontsize=10, fontweight='bold', zorder=5)

    clean_name = team_name.replace(' Kulübü','').replace(' Spor','').replace(' Futbol','').strip()
    ax.text(50, -4, f'Tempo Network | Speed of Play — {clean_name}',
            ha='center', fontsize=12, color=TACTIQ_FG, fontweight='bold')
    ax.text(50, -7, 'Edge color = TTRP | Thickness = Volume | Dashed Green = Carry',
            ha='center', fontsize=9, color='#a0aec0')

    sm   = plt.cm.ScalarMappable(cmap=TEMPO_CMAP, norm=plt.Normalize(vmin=2.5, vmax=4.0))
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.03, pad=0.1, aspect=40)
    cbar.set_ticks([2.5, 4.0])
    cbar.set_ticklabels(['Fast (<2.5s)', 'Slow (>4.0s)'])
    cbar.ax.tick_params(colors=TACTIQ_FG, labelsize=8)
    cbar.outline.set_edgecolor('#333')

    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

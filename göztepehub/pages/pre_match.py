import dash
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import base64
import io
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from utils.data import extract_fixture_data, calculate_standings
from göztepehub.utils import buildup_analysis
from göztepehub.utils.xg_chain_analysis import analyze_opponent_xg_profile
from göztepehub.utils.game_phases import get_phase_metrics, LOWER_IS_BETTER, _load_all_team_events
from göztepehub.utils.advanced_tactics import (
    identify_playmaker, analyze_15s_rule, analyze_xg_chain_origins, get_goal_typologies
)
from göztepehub.utils.defensive_analysis import get_opponent_defensive_profile
from göztepehub.utils.transitions_analysis import get_opponent_transition_profile

dash.register_page(__name__, path='/pre-match', title='Göztepe Hub | Pre-Match')

GOZTEPE = 'Göztepe Spor Kulübü'

TAB_TO_PHASE = {
    'offensive-tab':  'Offensive',
    'defensive-tab':  'Defensive',
    'off-trans-tab':  'Off. Transitions',
    'def-trans-tab':  'Def. Transitions',
}
TAB_LABELS = {
    'offensive-tab':  'Offensive',
    'defensive-tab':  'Defensive',
    'off-trans-tab':  'Off. Transitions',
    'def-trans-tab':  'Def. Transitions',
}

_SUFFIXES = ['Spor Kulübü', 'Futbol Kulübü', 'Kulübü', 'Spor A.Ş.', 'A.Ş.', 'S.K.', 'F.K.', 'SK']

def _clean(name):
    result = name
    for s in _SUFFIXES:
        result = result.replace(s, '')
    return result.strip()


# ──────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────

def _section_card(*children, title=None, icon=""):
    header = []
    if title:
        header = [html.Div(className="goz-section-header", style={"marginBottom": "16px"}, children=[
            html.Span(f"{icon}  {title}" if icon else title, className="goz-card-title")
        ])]
    return html.Div(className="goz-form-section", style={"marginBottom": "16px"},
                    children=header + list(children))


def _stat_pill(label, value, color="var(--accent-gold)", sub=None):
    return html.Div(style={
        "textAlign": "center", "padding": "14px 12px",
        "background": "rgba(255,255,255,0.04)",
        "borderRadius": "12px", "border": "1px solid var(--border-color)",
        "flex": "1", "minWidth": "80px",
    }, children=[
        html.Div(str(value), style={
            "fontSize": "1.5rem", "fontWeight": "700", "color": color, "lineHeight": "1"
        }),
        html.Div(label, style={
            "fontSize": "0.68rem", "color": "var(--text-secondary)",
            "marginTop": "5px", "textTransform": "uppercase", "letterSpacing": "0.5px"
        }),
        *([] if sub is None else [html.Div(str(sub), style={
            "fontSize": "0.65rem", "color": "var(--text-secondary)", "marginTop": "2px"
        })]),
    ])


def _bar_row(label, pct, color="var(--accent-gold)"):
    try:
        pct = float(str(pct).replace('%', '').strip())
    except Exception:
        pct = 0
    pct = min(max(pct, 0), 100)
    return html.Div(style={"marginBottom": "9px"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}, children=[
            html.Span(label, style={"fontSize": "0.78rem", "color": "var(--text-secondary)"}),
            html.Span(f"{pct:.0f}%", style={"fontSize": "0.78rem", "fontWeight": "700", "color": color}),
        ]),
        html.Div(style={"height": "4px", "background": "rgba(255,255,255,0.07)", "borderRadius": "2px"}, children=[
            html.Div(style={"width": f"{pct}%", "height": "100%", "background": color, "borderRadius": "2px"}),
        ])
    ])


# ──────────────────────────────────────────────────────────────
# Pitch drawing (Plotly)
# ──────────────────────────────────────────────────────────────

_PITCH_BG = "#0e1b0f"
_GOLD     = "#fbbf24"
_RED      = "#ef4444"
_BLUE     = "#3b82f6"
_PURPLE   = "#a855f7"
_GREEN    = "#22c55e"


def _make_mpl_pitch(half=False, show_f3=False, figsize=(10, 6.5)):
    """mplsoccer football pitch — returns (pitch, fig, ax)."""
    pitch = Pitch(
        pitch_type='opta',
        pitch_color=_PITCH_BG,
        line_color=(1.0, 1.0, 1.0, 0.55),
        linewidth=1.5,
        half=half,
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(_PITCH_BG)
    if show_f3:
        ax.axvline(x=66.67, color=_GOLD, linestyle=':', linewidth=1, alpha=0.4)
    return pitch, fig, ax


def _fig_to_b64(fig):
    """Convert matplotlib figure to a base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=_PITCH_BG, edgecolor='none')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def _mpl_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(handles, labels, loc='upper left', fontsize=8)
    frame = legend.get_frame()
    frame.set_facecolor(_PITCH_BG)
    frame.set_alpha(0.75)
    frame.set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')


# ──────────────────────────────────────────────────────────────
# Metric grid card (KPI panel)
# ──────────────────────────────────────────────────────────────

def _metric_card(label, goz_val, opp_val, goz_name, opp_name, lower_is_better=False):
    try:
        def to_f(v):
            if isinstance(v, str):
                return float(v.replace('%', '').strip())
            return float(v)
        g_num, o_num = to_f(goz_val), to_f(opp_val)
        total = g_num + o_num
        g_pct = (g_num / total * 100) if total > 0 else 50
        goz_better = (g_num <= o_num) if lower_is_better else (g_num >= o_num)
    except Exception:
        g_pct = 50
        goz_better = None

    GOZ_C = "var(--accent-gold)"
    OPP_C = "#ef4444"
    goz_border = "2px solid rgba(34,197,94,0.6)" if goz_better is True  else "2px solid transparent"
    opp_border  = "2px solid rgba(34,197,94,0.6)" if goz_better is False else "2px solid transparent"

    def side(name, val, color, border):
        return html.Div(style={
            "flex": "1", "textAlign": "center", "padding": "12px 8px",
            "background": "rgba(255,255,255,0.03)", "borderRadius": "10px", "border": border,
        }, children=[
            html.Div(name, style={
                "fontSize": "0.68rem", "fontWeight": "700", "color": color,
                "letterSpacing": "0.5px", "marginBottom": "6px",
                "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
            }),
            html.Div(str(val), style={
                "fontSize": "1.45rem", "fontWeight": "700", "color": color, "lineHeight": "1",
            }),
        ])

    return html.Div(style={
        "background": "var(--card-bg)", "border": "1px solid var(--border-color)",
        "borderRadius": "14px", "padding": "14px",
    }, children=[
        html.Div(label, style={
            "fontSize": "0.72rem", "fontWeight": "600",
            "color": "var(--text-secondary)", "textTransform": "uppercase",
            "letterSpacing": "0.5px", "textAlign": "center", "marginBottom": "10px",
        }),
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "stretch"}, children=[
            side(goz_name, goz_val, GOZ_C, goz_border),
            html.Div("vs", style={
                "alignSelf": "center", "fontSize": "0.65rem",
                "color": "var(--text-secondary)", "flexShrink": "0",
            }),
            side(opp_name, opp_val, OPP_C, opp_border),
        ]),
        html.Div(style={
            "height": "4px", "marginTop": "12px",
            "background": "rgba(255,255,255,0.05)",
            "borderRadius": "2px", "overflow": "hidden", "display": "flex",
        }, children=[
            html.Div(style={"width": f"{g_pct:.1f}%", "background": GOZ_C}),
            html.Div(style={"width": f"{100-g_pct:.1f}%", "background": "rgba(239,68,68,0.35)"}),
        ]),
    ])


# ──────────────────────────────────────────────────────────────
# Offensive deep analysis builder
# ──────────────────────────────────────────────────────────────

def _build_offensive_analysis(opponent, opp_name):
    """Returns a list of rich analysis sections for the offensive tab."""
    sections = []

    # ── 1. BUILD-UP STYLE ──────────────────────────────────────
    try:
        analyses, season_summary = buildup_analysis.get_opponent_buildup_analysis(opponent)
        pt   = season_summary.get('pass_type', {})
        zone = season_summary.get('zone', {})
        coords = season_summary.get('coords', [])

        short_pct  = float(pt.get('short_pct', 50))
        long_pct   = float(pt.get('long_pct',  50))
        left_pct   = float(zone.get('left_pct',   33))
        center_pct = float(zone.get('center_pct', 34))
        right_pct  = float(zone.get('right_pct',  33))

        # Buildup start pitch — color by pass type
        pitch_bu, fig_bu, ax_bu = _make_mpl_pitch(show_f3=True)
        if coords:
            short_xs = [c['x'] for c in coords if isinstance(c, dict) and c.get('pass_type') == 'Short']
            short_ys = [c['y'] for c in coords if isinstance(c, dict) and c.get('pass_type') == 'Short']
            long_xs  = [c['x'] for c in coords if isinstance(c, dict) and c.get('pass_type') == 'Long']
            long_ys  = [c['y'] for c in coords if isinstance(c, dict) and c.get('pass_type') == 'Long']
            if short_xs:
                pitch_bu.scatter(short_xs, short_ys, ax=ax_bu, color=_GOLD, s=20, alpha=0.6, label="Short build-up")
            if long_xs:
                pitch_bu.scatter(long_xs, long_ys, ax=ax_bu, color=_RED, s=20, alpha=0.6, label="Long ball")
            _mpl_legend(ax_bu)
        bu_b64 = _fig_to_b64(fig_bu)

        # Aggregate F3 entry data from individual match analyses
        n_with = 0
        sp_sum = dp_sum = carry_sum = 0
        el_sum = ec_sum = er_sum   = 0
        box_sum = cross_sum        = 0
        entry_coords = []

        for a in (analyses or []):
            fes = a.get('f3_entry_stats') if isinstance(a, dict) else None
            if fes:
                n_with += 1
                em = fes.get('entry_method', {})   # flat: short_pass_pct, deep_pass_pct, carry_pct
                ez = fes.get('entry_zone',   {})   # flat: left_pct, center_pct, right_pct
                sa = fes.get('subsequent',   {})   # flat: box_control_pct, cross_pct
                entry_coords.extend(fes.get('entry_coords', []))
                sp_sum    += em.get('short_pass_pct', 0)
                dp_sum    += em.get('deep_pass_pct',  0)
                carry_sum += em.get('carry_pct',      0)
                el_sum    += ez.get('left_pct',   0)
                ec_sum    += ez.get('center_pct', 0)
                er_sum    += ez.get('right_pct',  0)
                box_sum   += sa.get('box_control_pct', 0)
                cross_sum += sa.get('cross_pct',        0)

        if n_with:
            avg_sp    = round(sp_sum    / n_with, 1)
            avg_dp    = round(dp_sum    / n_with, 1)
            avg_carry = round(carry_sum / n_with, 1)
            avg_el    = round(el_sum    / n_with, 1)
            avg_ec    = round(ec_sum    / n_with, 1)
            avg_er    = round(er_sum    / n_with, 1)
            avg_box   = round(box_sum   / n_with, 1)
            avg_cross = round(cross_sum / n_with, 1)
        else:
            avg_sp = avg_dp = avg_carry = avg_el = avg_ec = avg_er = avg_box = avg_cross = 0

        buildup_section = _section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"marginBottom": "16px"}, children=[
                        html.Div("PASS TYPE", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "8px",
                        }),
                        _bar_row("Short Pass", short_pct, _GOLD),
                        _bar_row("Long Ball",  long_pct,  _RED),
                    ]),
                    html.Div(children=[
                        html.Div("BUILD-UP ZONE", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "8px",
                        }),
                        _bar_row("Left Channel",   left_pct,   _BLUE),
                        _bar_row("Central Corridor", center_pct, _GOLD),
                        _bar_row("Right Channel",  right_pct,  _PURPLE),
                    ]),
                ], md=4),
                dbc.Col([
                    html.Img(src=bu_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("● Build-up starting positions", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)",
                        "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=8),
            ]),
            title=f"{opp_name} — Build-up Style", icon="⚽"
        )
        sections.append(buildup_section)

    except Exception:
        analyses, season_summary, coords = [], {}, []
        pt, zone, entry_coords = {}, {}, []
        avg_sp = avg_dp = avg_carry = avg_el = avg_ec = avg_er = avg_box = avg_cross = 0
        short_pct = long_pct = left_pct = center_pct = right_pct = 50

    # ── 2. 15-SECOND OUTCOMES ─────────────────────────────────
    try:
        out15 = season_summary.get('outcomes_15s', {})
        # Keys are flat: f3_entry_pct, shot_pct, sot_pct, goal_pct, turnover_pct
        f3_pct   = float(out15.get('f3_entry_pct', 0))
        shot_pct = float(out15.get('shot_pct',     0))
        sot_pct  = float(out15.get('sot_pct',      0))
        goal_pct = float(out15.get('goal_pct',     0))
        turn_pct = float(out15.get('turnover_pct', 0))

        sections.append(_section_card(
            html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                _stat_pill("F3 Entry",    f"{f3_pct}%",   _GOLD),
                _stat_pill("Shot",        f"{shot_pct}%",  _BLUE),
                _stat_pill("On Target",   f"{sot_pct}%",   _PURPLE),
                _stat_pill("Goal",        f"{goal_pct}%",  _GREEN),
                _stat_pill("Turnover",    f"{turn_pct}%",  _RED),
            ]),
            html.Div("What happens within 15 seconds of a build-up?", style={
                "fontSize": "0.7rem", "color": "var(--text-secondary)",
                "marginTop": "10px", "textAlign": "center",
            }),
            title="15-Second Outcomes", icon="⏱️"
        ))
    except Exception:
        pass

    # ── 3. FINAL THIRD ENTRY ──────────────────────────────────
    try:
        pitch_f3, fig_f3, ax_f3 = _make_mpl_pitch(half=True, show_f3=True)

        if entry_coords:
            method_colors = {'Short Pass': _GOLD, 'Deep Pass': _RED, 'Ball Carry': _BLUE}
            for method, color in method_colors.items():
                mxs = [c['x'] for c in entry_coords if isinstance(c, dict) and c.get('method') == method]
                mys = [c['y'] for c in entry_coords if isinstance(c, dict) and c.get('method') == method]
                if mxs:
                    pitch_f3.scatter(mxs, mys, ax=ax_f3, color=color, s=45, alpha=0.7, marker='D', label=method)
            _mpl_legend(ax_f3)
        f3_b64 = _fig_to_b64(fig_f3)

        f3_section = _section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"marginBottom": "14px"}, children=[
                        html.Div("ENTRY METHOD", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "marginBottom": "8px", "letterSpacing": "1px",
                        }),
                        _bar_row("Short Pass", avg_sp,    _GOLD),
                        _bar_row("Deep Pass",  avg_dp,    _RED),
                        _bar_row("Ball Carry", avg_carry, _BLUE),
                    ]),
                    html.Div(style={"marginBottom": "14px"}, children=[
                        html.Div("ENTRY ZONE", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "marginBottom": "8px", "letterSpacing": "1px",
                        }),
                        _bar_row("Left",   avg_el, _BLUE),
                        _bar_row("Central",avg_ec, _GOLD),
                        _bar_row("Right",  avg_er, _PURPLE),
                    ]),
                    html.Div(children=[
                        html.Div("AFTER ENTRY", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "marginBottom": "8px", "letterSpacing": "1px",
                        }),
                        _bar_row("Box Control", avg_box,   _GREEN),
                        _bar_row("Cross",        avg_cross, _PURPLE),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=f3_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("◆ Final third entry points", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)",
                        "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=7),
            ]),
            title="Final Third Entry", icon="🎯"
        )
        sections.append(f3_section)
    except Exception:
        pass

    # ── 4. PLAYMAKER, TEMPO & CLUSTERING ─────────────────────
    try:
        all_events, match_count = _load_all_team_events(opponent)

        if not all_events.empty:
            playmaker = identify_playmaker(all_events, opponent)
            pm_name   = playmaker.get('name', '—')
            pm_passes = playmaker.get('passes', 0)
            pm_prog   = playmaker.get('prog_passes', 0)

            # GK short pass distribution (proxy: passes from x < 20)
            opp_events = all_events[all_events['team_name'] == opponent]
            gk_passes  = opp_events[(opp_events['event'] == 'Pass') & (opp_events['x'] < 20)]
            if len(gk_passes) > 0 and 'Pass End X' in gk_passes.columns:
                gk_short_n = len(gk_passes[gk_passes['Pass End X'].fillna(0) < gk_passes['x'] + 25])
            else:
                gk_short_n = len(gk_passes)
            gk_short_pct = round(gk_short_n / max(len(gk_passes), 1) * 100, 1)

            # Full-back pass contribution
            lb_passes = len(opp_events[(opp_events['event'] == 'Pass') & (opp_events['y'] < 25) & (opp_events['x'] > 30)])
            rb_passes = len(opp_events[(opp_events['event'] == 'Pass') & (opp_events['y'] > 75) & (opp_events['x'] > 30)])

            # Passes per game (tempo)
            passes_per_game = round(len(opp_events[opp_events['event'] == 'Pass']) / max(match_count, 1), 1)

            # Final third clustering (zone 14 vs flanks)
            f3 = opp_events[opp_events['x'] > 66]
            left_cluster   = round(len(f3[f3['y'] < 35])                              / max(len(f3), 1) * 100, 1)
            center_cluster = round(len(f3[(f3['y'] >= 35) & (f3['y'] <= 65)])         / max(len(f3), 1) * 100, 1)
            right_cluster  = round(len(f3[f3['y'] > 65])                              / max(len(f3), 1) * 100, 1)

            # Shot origin (from advanced_tactics helper)
            xg_origins = analyze_xg_chain_origins(all_events, opponent)
            total_shots_ev = max(sum(xg_origins.values()), 1)

            sections.append(_section_card(
                dbc.Row([
                    dbc.Col([
                        html.Div(style={
                            "background": "rgba(251,191,36,0.08)",
                            "border": "1px solid rgba(251,191,36,0.2)",
                            "borderRadius": "12px", "padding": "16px", "marginBottom": "14px",
                        }, children=[
                            html.Div("🎖️ KEY PLAYMAKER", style={
                                "fontSize": "0.7rem", "color": "var(--text-secondary)",
                                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px",
                            }),
                            html.Div(pm_name, style={
                                "fontSize": "1.1rem", "fontWeight": "700", "color": _GOLD, "marginBottom": "6px",
                            }),
                            html.Div(style={"display": "flex", "gap": "16px"}, children=[
                                html.Span(f"{pm_passes} passes", style={"fontSize": "0.75rem", "color": "var(--text-secondary)"}),
                                html.Span(f"{pm_prog} progressive", style={"fontSize": "0.75rem", "color": _BLUE}),
                            ]),
                        ]),
                        html.Div(style={
                            "background": "rgba(255,255,255,0.03)", "borderRadius": "12px",
                            "padding": "14px", "border": "1px solid var(--border-color)",
                        }, children=[
                            html.Div("⏩ GAME TEMPO & RHYTHM", style={
                                "fontSize": "0.7rem", "color": "var(--text-secondary)",
                                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "10px",
                            }),
                            html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                                _stat_pill("Passes / Game",    passes_per_game, _GOLD),
                                _stat_pill("GK Short Pass %", f"{gk_short_pct}%", _BLUE),
                                _stat_pill("Left FB Passes",  lb_passes, _PURPLE),
                                _stat_pill("Right FB Passes", rb_passes, _PURPLE),
                            ]),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.Div(style={
                            "background": "rgba(255,255,255,0.03)", "borderRadius": "12px",
                            "padding": "14px", "border": "1px solid var(--border-color)", "marginBottom": "14px",
                        }, children=[
                            html.Div("📍 FINAL THIRD CLUSTERING", style={
                                "fontSize": "0.7rem", "color": "var(--text-secondary)",
                                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "10px",
                            }),
                            _bar_row("Left Channel",   left_cluster,   _BLUE),
                            _bar_row("Zone 14 / Central", center_cluster, _GOLD),
                            _bar_row("Right Channel",  right_cluster,  _PURPLE),
                        ]),
                        html.Div(style={
                            "background": "rgba(255,255,255,0.03)", "borderRadius": "12px",
                            "padding": "14px", "border": "1px solid var(--border-color)",
                        }, children=[
                            html.Div("🥅 SHOT ORIGIN (chain)", style={
                                "fontSize": "0.7rem", "color": "var(--text-secondary)",
                                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "10px",
                            }),
                            _bar_row("Open Play / Combination", round(xg_origins['Frontal/Open'] / total_shots_ev * 100, 1), _GOLD),
                            _bar_row("From Cross",               round(xg_origins['Cross']        / total_shots_ev * 100, 1), _BLUE),
                            _bar_row("Set Piece",                round(xg_origins['Set Piece']    / total_shots_ev * 100, 1), _RED),
                        ]),
                    ], md=6),
                ]),
                title="Playmaker, Tempo & Attacking Structure", icon="🧠"
            ))
        else:
            raise ValueError("No events")

    except Exception:
        pass

    # ── 5. SHOT MAP & xG PROFILE ──────────────────────────────
    try:
        xg_profile, xg_matches = analyze_opponent_xg_profile(opponent)

        xg_per_game    = xg_profile.get('xg_per_game', 0)
        sot_pct_xg     = xg_profile.get('sot_pct', 0)
        xg_per_shot    = xg_profile.get('xg_per_shot', 0)
        total_shots_xg = xg_profile.get('total_shots', 0)
        # origin_pcts keys: 'open_play', 'from_cross', 'set_piece', 'fast_break', 'through_ball'
        origin_pcts    = xg_profile.get('origin_pcts', {})
        # zone_pcts keys: 'inside_box', 'outside_box'
        zones_pct      = xg_profile.get('zone_pcts', {})

        all_shot_coords = []
        for m in (xg_matches or []):
            all_shot_coords.extend(m.get('shot_coords', []))

        pitch_shots, fig_shots, ax_shots = _make_mpl_pitch(half=True)

        if all_shot_coords:
            _MISS_C = (1.0, 1.0, 1.0, 0.3)
            for ev_type, color, label in [
                ('Goal',       _GOLD,   "Goal"),
                ('Saved Shot', _BLUE,   "On Target"),
                ('Miss',       _MISS_C, "Off Target"),
                ('Post',       _PURPLE, "Post"),
            ]:
                grp = [s for s in all_shot_coords if s.get('event') == ev_type]
                if grp:
                    gx  = [s['x'] for s in grp]
                    gy  = [s['y'] for s in grp]
                    gxg = [s.get('xG', 0) for s in grp]
                    sizes = [max(30, min(v * 300, 500)) for v in gxg]
                    pitch_shots.scatter(gx, gy, ax=ax_shots, color=color, s=sizes,
                                       alpha=0.75, label=label,
                                       edgecolors=(1, 1, 1, 0.15), linewidths=0.5)
            _mpl_legend(ax_shots)
        shots_b64 = _fig_to_b64(fig_shots)

        shot_section = _section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "14px"}, children=[
                        _stat_pill("xG / Game",   round(xg_per_game, 2),         _GOLD),
                        _stat_pill("Shots",        total_shots_xg,                _BLUE),
                        _stat_pill("On Target %", f"{round(sot_pct_xg, 1)}%",    _PURPLE),
                        _stat_pill("xG / Shot",   round(xg_per_shot, 3),          _RED),
                    ]),
                    html.Div(children=[
                        html.Div("SHOT ORIGIN", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "marginBottom": "8px", "letterSpacing": "1px",
                        }),
                        _bar_row("Open Play",   round(origin_pcts.get('open_play',    0), 1), _GOLD),
                        _bar_row("From Cross",  round(origin_pcts.get('from_cross',   0), 1), _BLUE),
                        _bar_row("Set Piece",   round(origin_pcts.get('set_piece',    0), 1), _RED),
                        _bar_row("Fast Break",  round(origin_pcts.get('fast_break',   0), 1), _PURPLE),
                        _bar_row("Through Ball",round(origin_pcts.get('through_ball', 0), 1), _GREEN),
                    ]),
                    html.Div(style={"marginTop": "14px"}, children=[
                        html.Div("SHOT ZONE", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "marginBottom": "8px", "letterSpacing": "1px",
                        }),
                        _bar_row("Inside Box",  round(zones_pct.get('inside_box',  0), 1), _GREEN),
                        _bar_row("Outside Box", round(zones_pct.get('outside_box', 0), 1), _RED),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=shots_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("Circle size = xG value", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)",
                        "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=7),
            ]),
            title="Shot Map & xG Profile", icon="🔫"
        )
        sections.append(shot_section)

    except Exception:
        pass

    return html.Div(sections)


# ──────────────────────────────────────────────────────────────
# Defensive deep analysis builder
# ──────────────────────────────────────────────────────────────

def _build_defensive_analysis(opponent, opp_name):
    """Returns rich analysis sections for the defensive tab."""
    sections = []

    try:
        profile = get_opponent_defensive_profile(opponent)
        if profile is None:
            raise ValueError("No profile")

        # ── 1. DEFENSIVE SHAPE & PRESSING ─────────────────────
        coords = profile.get('heat_coords', [])
        avg_line = round(profile.get('avg_def_line', 0), 1)
        total_actions = profile.get('total_def_actions', 0)

        pitch_def, fig_def, ax_def = _make_mpl_pitch()
        if coords:
            import random
            sample = random.sample(coords, min(len(coords), 300))
            by_type = {}
            for c in sample:
                t = c.get('type', 'Other')
                by_type.setdefault(t, []).append(c)
            type_colors = {'Tackle': _GOLD, 'Interception': _BLUE, 'Clearance': _RED, 'Challenge': _PURPLE}
            for t, pts in by_type.items():
                pitch_def.scatter(
                    [p['x'] for p in pts], [p['y'] for p in pts],
                    ax=ax_def, color=type_colors.get(t, (1.0, 1.0, 1.0, 0.3)),
                    s=20, alpha=0.55, label=t,
                )
            ax_def.axvline(x=avg_line, color=_GREEN, linestyle='--', linewidth=2)
            ylim = ax_def.get_ylim()
            ax_def.text(avg_line, ylim[1] * 0.97, f"Avg Line: {avg_line}",
                        color=_GREEN, fontsize=9, ha='center', va='top')
            _mpl_legend(ax_def)
        def_b64 = _fig_to_b64(fig_def)

        sections.append(_section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "14px"}, children=[
                        _stat_pill("Def. Actions", total_actions, _GOLD),
                        _stat_pill("Avg Line Height", avg_line, _BLUE),
                        _stat_pill("Box Aerials Won", f"{profile.get('box_aerial_win_pct', 0)}%", _PURPLE,
                                   sub=f"{profile.get('box_aerial_total', 0)} total"),
                    ]),
                ], md=5),
                dbc.Col([
                    html.Img(src=def_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("● Defensive actions map  ── Avg line", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)", "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=7),
            ]),
            title=f"{opp_name} — Defensive Shape & Pressing", icon="🛡️"
        ))

        # ── 2. VULNERABILITY MAP ──────────────────────────────
        flanks = profile.get('f3_flanks', {})
        z14_allowed = profile.get('z14_passes_allowed', 0)
        z14_pct = profile.get('z14_success_allowed_pct', 0)
        f3_total = profile.get('f3_entries_total', 0)

        sections.append(_section_card(
            dbc.Row([
                dbc.Col([
                    html.Div("F3 ENTRIES CONCEDED BY FLANK", style={
                        "fontSize": "0.72rem", "fontWeight": "700",
                        "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "8px",
                    }),
                    _bar_row("Left Channel", flanks.get('Left', 0), _BLUE),
                    _bar_row("Central", flanks.get('Center', 0), _GOLD),
                    _bar_row("Right Channel", flanks.get('Right', 0), _PURPLE),
                    html.Div(f"Total F3 entries conceded: {f3_total}", style={
                        "fontSize": "0.7rem", "color": "var(--text-secondary)", "marginTop": "8px",
                    }),
                ], md=6),
                dbc.Col([
                    html.Div(style={
                        "background": "rgba(255,255,255,0.03)", "borderRadius": "12px",
                        "padding": "14px", "border": "1px solid var(--border-color)",
                    }, children=[
                        html.Div("ZONE 14 CONTROL", style={
                            "fontSize": "0.72rem", "fontWeight": "700",
                            "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "10px",
                        }),
                        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                            _stat_pill("Passes Allowed", z14_allowed, _RED),
                            _stat_pill("Opp. Success %", f"{z14_pct}%", _RED),
                        ]),
                    ]),
                ], md=6),
            ]),
            title="Vulnerability Map", icon="⚠️"
        ))

        # ── 3. PRE-GOAL STRUCTURE ─────────────────────────────
        pg = profile.get('pre_goal_summary', {})
        goals_conceded = profile.get('goals_conceded', 0)

        sections.append(_section_card(
            html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                _stat_pill("Goals Conceded", goals_conceded, _RED),
                _stat_pill("Avg Def. Actions (30s)", pg.get('avg_def_actions_30s', 0), _GOLD),
                _stat_pill("Avg Opp. Passes (30s)", pg.get('avg_opp_passes_30s', 0), _BLUE),
                _stat_pill("Failed Clearances", pg.get('total_failed_clearances', 0), _RED),
            ]),
            html.Div("What happens in the 30 seconds before conceding?", style={
                "fontSize": "0.7rem", "color": "var(--text-secondary)", "marginTop": "10px", "textAlign": "center",
            }),
            title="Pre-Goal Structure (30s Window)", icon="🥅"
        ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className="goz-form-section", children=[
        html.Div("Defensive deep analysis data unavailable.", className="goz-card-desc")
    ])


# ──────────────────────────────────────────────────────────────
# Off. Transitions deep analysis builder
# ──────────────────────────────────────────────────────────────

def _build_off_transitions_analysis(opponent, opp_name):
    """Returns rich analysis sections for the offensive transitions tab."""
    sections = []

    try:
        att_profile, _ = get_opponent_transition_profile(opponent)
        if not att_profile:
            raise ValueError("No data")

        total = att_profile.get('total', 0)
        zones = att_profile.get('zones', {})
        outcomes = att_profile.get('outcomes', {})
        top_players = att_profile.get('top_players', [])
        coords = att_profile.get('coords', [])

        # ── 1. RECOVERY MAP ───────────────────────────────────
        pitch_rec, fig_rec, ax_rec = _make_mpl_pitch()
        if coords:
            import random
            sample = random.sample(coords, min(len(coords), 400))
            pitch_rec.scatter([c['x'] for c in sample], [c['y'] for c in sample],
                              ax=ax_rec, color=_GOLD, s=20, alpha=0.5, label="Ball Recovery")
            _mpl_legend(ax_rec)
        rec_b64 = _fig_to_b64(fig_rec)

        sections.append(_section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "14px"}, children=[
                        _stat_pill("Total Recoveries", total, _GOLD),
                        _stat_pill("→ F3 Entry", outcomes.get('f3_entry', 0), _BLUE),
                        _stat_pill("→ Shot", outcomes.get('shot', 0), _PURPLE),
                        _stat_pill("→ Goal", outcomes.get('goal', 0), _GREEN),
                    ]),
                    html.Div("RECOVERY ZONE", style={
                        "fontSize": "0.72rem", "fontWeight": "700",
                        "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "8px",
                    }),
                    _bar_row("Defensive 3rd", round(zones.get('Defensive 3rd', 0) / max(total, 1) * 100, 1), _RED),
                    _bar_row("Middle 3rd", round(zones.get('Middle 3rd', 0) / max(total, 1) * 100, 1), _GOLD),
                    _bar_row("Final 3rd", round(zones.get('Final 3rd', 0) / max(total, 1) * 100, 1), _GREEN),
                ], md=5),
                dbc.Col([
                    html.Img(src=rec_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("● Ball recovery positions", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)", "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=7),
            ]),
            title=f"{opp_name} — Attacking Transitions (Recoveries)", icon="⚡"
        ))

        # ── 2. TOP RECOVERY PLAYERS ───────────────────────────
        if top_players:
            player_rows = []
            for i, (name, count) in enumerate(top_players[:8]):
                pct = round(count / max(total, 1) * 100, 1)
                player_rows.append(html.Div(style={"display": "flex", "justifyContent": "space-between",
                    "padding": "6px 0", "borderBottom": "1px solid rgba(255,255,255,0.05)"}, children=[
                    html.Span(f"{i+1}. {name}", style={"fontSize": "0.78rem", "color": "var(--text-secondary)"}),
                    html.Span(f"{count} ({pct}%)", style={"fontSize": "0.78rem", "fontWeight": "700", "color": _GOLD}),
                ]))
            sections.append(_section_card(
                html.Div(player_rows),
                title="Top Recovery Players", icon="🎖️"
            ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className="goz-form-section", children=[
        html.Div("Offensive transitions deep analysis data unavailable.", className="goz-card-desc")
    ])


# ──────────────────────────────────────────────────────────────
# Def. Transitions deep analysis builder
# ──────────────────────────────────────────────────────────────

def _build_def_transitions_analysis(opponent, opp_name):
    """Returns rich analysis sections for the defensive transitions tab."""
    sections = []

    try:
        _, def_profile = get_opponent_transition_profile(opponent)
        if not def_profile:
            raise ValueError("No data")

        total = def_profile.get('total', 0)
        zones = def_profile.get('zones', {})
        outcomes = def_profile.get('outcomes', {})
        top_players = def_profile.get('top_players', [])
        coords = def_profile.get('coords', [])

        # ── 1. BALL LOSS MAP ──────────────────────────────────
        pitch_loss, fig_loss, ax_loss = _make_mpl_pitch()
        if coords:
            import random
            sample = random.sample(coords, min(len(coords), 400))
            pitch_loss.scatter([c['x'] for c in sample], [c['y'] for c in sample],
                               ax=ax_loss, color=_RED, s=20, alpha=0.5, label="Ball Loss")
            _mpl_legend(ax_loss)
        loss_b64 = _fig_to_b64(fig_loss)

        sections.append(_section_card(
            dbc.Row([
                dbc.Col([
                    html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "14px"}, children=[
                        _stat_pill("Total Ball Losses", total, _RED),
                        _stat_pill("Opp. → F3", outcomes.get('opp_f3_entry', 0), _BLUE),
                        _stat_pill("Opp. → Shot", outcomes.get('opp_shot', 0), _PURPLE),
                        _stat_pill("Opp. → Goal", outcomes.get('opp_goal', 0), _RED),
                    ]),
                    html.Div("LOSS ZONE", style={
                        "fontSize": "0.72rem", "fontWeight": "700",
                        "color": "var(--accent-gold)", "letterSpacing": "1px", "marginBottom": "8px",
                    }),
                    _bar_row("Defensive 3rd", round(zones.get('Defensive 3rd', 0) / max(total, 1) * 100, 1), _RED),
                    _bar_row("Middle 3rd", round(zones.get('Middle 3rd', 0) / max(total, 1) * 100, 1), _GOLD),
                    _bar_row("Final 3rd", round(zones.get('Final 3rd', 0) / max(total, 1) * 100, 1), _GREEN),
                ], md=5),
                dbc.Col([
                    html.Img(src=loss_b64, style={"width": "100%", "borderRadius": "8px"}),
                    html.Div("● Ball loss positions", style={
                        "fontSize": "0.65rem", "color": "var(--text-secondary)", "textAlign": "center", "marginTop": "4px",
                    }),
                ], md=7),
            ]),
            title=f"{opp_name} — Defensive Transitions (Ball Losses)", icon="🔄"
        ))

        # ── 2. TOP BALL-LOSING PLAYERS ────────────────────────
        if top_players:
            player_rows = []
            for i, (name, count) in enumerate(top_players[:8]):
                pct = round(count / max(total, 1) * 100, 1)
                player_rows.append(html.Div(style={"display": "flex", "justifyContent": "space-between",
                    "padding": "6px 0", "borderBottom": "1px solid rgba(255,255,255,0.05)"}, children=[
                    html.Span(f"{i+1}. {name}", style={"fontSize": "0.78rem", "color": "var(--text-secondary)"}),
                    html.Span(f"{count} ({pct}%)", style={"fontSize": "0.78rem", "fontWeight": "700", "color": _RED}),
                ]))
            sections.append(_section_card(
                html.Div(player_rows),
                title="Top Ball-Losing Players", icon="⚠️"
            ))

    except Exception:
        pass

    return html.Div(sections) if sections else html.Div(className="goz-form-section", children=[
        html.Div("Defensive transitions deep analysis data unavailable.", className="goz-card-desc")
    ])


# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────

def layout():
    matches   = extract_fixture_data(lite=True)
    standings = calculate_standings(matches)
    rivals    = sorted([t for t in standings['Team'].unique() if t != GOZTEPE])

    return html.Div(className="page-wrap", children=[
        html.Div(className="goz-hero", children=[
            html.Div(className="goz-hero-content", children=[
                dcc.Link("← GÖZTEPE HUB", href="/", className="goz-back-link"),
                html.H1("PRE-MATCH ANALYSIS", className="goz-hub-title"),
                html.P("Season-level tactical intelligence & phase analysis", className="goz-hub-subtitle"),
                html.Div(style={"marginTop": "25px", "width": "100%", "maxWidth": "350px"}, children=[
                    html.Label("SELECT UPCOMING OPPONENT", className="goz-label"),
                    dcc.Dropdown(
                        id='pre-match-rival-selector',
                        options=[{'label': r, 'value': r} for r in rivals],
                        value=rivals[0] if rivals else None,
                        className="goz-dropdown",
                        clearable=False,
                    ),
                ]),
            ]),
        ]),

        html.Div(className="content-container", style={"padding": "0 20px 60px"}, children=[
            html.Div(style={"display": "flex", "justifyContent": "center", "margin": "30px 0"}, children=[
                dbc.RadioItems(
                    id="pre-match-tabs",
                    options=[
                        {"label": "⚔️ Offensive",        "value": "offensive-tab"},
                        {"label": "🛡️ Defensive",        "value": "defensive-tab"},
                        {"label": "⚡ Off. Transitions",  "value": "off-trans-tab"},
                        {"label": "🔄 Def. Transitions",  "value": "def-trans-tab"},
                    ],
                    value="offensive-tab",
                    inline=True,
                    className="pm-tab-radio-group",
                    inputClassName="pm-tab-radio-input",
                    labelClassName="pm-tab-radio-label",
                ),
            ]),
            html.Div(id='pre-match-kpi-container'),
            html.Div(id='pre-match-tab-content', style={"marginTop": "20px"}),
        ]),

        html.Footer(className="footer", children=[
            html.Div(className="footer-inner", children=[
                html.Div("© TactIQ Göztepe Hub — Precision Analytics", className="footer-text"),
                html.Img(src="/assets/superlig_logo.jpg", className="superlogo"),
            ])
        ])
    ])


# ──────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────

@callback(
    [Output('pre-match-kpi-container', 'children'),
     Output('pre-match-tab-content', 'children')],
    [Input('pre-match-rival-selector', 'value'),
     Input('pre-match-tabs', 'value')]
)
def update_pre_match(opponent, active_tab):
    if not opponent:
        return html.Div(), html.Div()

    phase    = TAB_TO_PHASE.get(active_tab, 'Offensive')
    g_metrics = get_phase_metrics(phase, GOZTEPE)
    o_metrics = get_phase_metrics(phase, opponent)
    goz_name  = _clean(GOZTEPE)
    opp_name  = _clean(opponent)

    cards = [
        _metric_card(lbl, g_metrics[lbl], o_metrics.get(lbl, "N/A"),
                     goz_name, opp_name, lbl in LOWER_IS_BETTER)
        for lbl in g_metrics
    ]

    kpi_container = html.Div(className="goz-form-section", children=[
        html.Div(className="goz-section-header", children=[
            html.Span(f"{TAB_LABELS[active_tab]} Key Metrics", className="goz-card-title"),
        ]),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(220px, 1fr))",
            "gap": "12px",
        }, children=cards),
    ])

    # Tab-specific deep content
    if active_tab == "offensive-tab":
        tab_content = _build_offensive_analysis(opponent, opp_name)
    elif active_tab == "defensive-tab":
        tab_content = _build_defensive_analysis(opponent, opp_name)
    elif active_tab == "off-trans-tab":
        tab_content = _build_off_transitions_analysis(opponent, opp_name)
    elif active_tab == "def-trans-tab":
        tab_content = _build_def_transitions_analysis(opponent, opp_name)
    else:
        tab_content = html.Div(className="goz-form-section", children=[
            html.Div(f"{TAB_LABELS[active_tab]} detaylı analiz yakında.", className="goz-card-desc")
        ])

    return kpi_container, tab_content

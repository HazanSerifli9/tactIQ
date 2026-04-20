"""
SEPP — Shot-Ending Possession Passes
======================================
Only count passes within possessions that end with a shot.
Build-up passes in the back that don't lead to shots are meaningless.

Set piece possessions are separated and excluded from open-play SEPP.

Metrics:
- SEPP Total: Total passes in shot-ending possessions (open play only)
- SEPP / Shot: Average passes per shot-ending possession
- SEPP F3: Passes in the final third within shot-ending possessions
- SEPP Prog: Progressive passes within shot-ending possessions
- Dead Passes: Passes in non-shot possessions (wasted build-up)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from utils.possession_engine import extract_possession_chains, PossessionChain


def _is_progressive_pass(ev: dict) -> bool:
    """Check if a pass is progressive (moves ball ≥25% closer to goal)."""
    x1 = ev.get('x', 0)
    y1 = ev.get('y', 50)
    x2 = ev.get('Pass End X')
    y2 = ev.get('Pass End Y')
    
    if x2 is None or y2 is None:
        return False
    
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except (ValueError, TypeError):
        return False
    
    dist_start = np.sqrt((100 - x1)**2 + (50 - y1)**2)
    dist_end = np.sqrt((100 - x2)**2 + (50 - y2)**2)
    
    return dist_end <= dist_start * 0.75


def _is_final_third_pass(ev: dict) -> bool:
    """Check if a pass originates in the final third (x >= 66.6)."""
    return float(ev.get('x', 0)) >= 66.6


def _get_pass_details(ev: dict) -> dict:
    """Extract pass coordinates for visualization."""
    return {
        'player': ev.get('player_name', ''),
        'x': ev.get('x', 0),
        'y': ev.get('y', 0),
        'end_x': ev.get('Pass End X', 0),
        'end_y': ev.get('Pass End Y', 0),
        'is_progressive': _is_progressive_pass(ev),
        'is_f3': _is_final_third_pass(ev),
        'outcome': ev.get('outcome', 0),
    }


def calculate_sepp(df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
    """
    Calculate SEPP metrics for a team in a single match.
    
    Returns dict with:
        - sepp_total: passes in shot-ending open-play possessions
        - sepp_per_shot: avg passes per such possession
        - sepp_f3: final-third passes in shot-ending possessions
        - sepp_prog: progressive passes in shot-ending possessions
        - dead_passes: passes in non-shot open-play possessions
        - shot_chains: count of open-play possessions ending in shots
        - dead_chains: count of open-play possessions NOT ending in shots
        - set_piece_chains: possessions that started from set pieces
        - efficiency: sepp_total / (sepp_total + dead_passes) as percentage
    """
    chains = extract_possession_chains(df, team_name)
    
    if not chains:
        return _empty_sepp()
    
    # Separate open play vs set pieces
    open_play_chains = [c for c in chains if not c.is_set_piece]
    set_piece_chains = [c for c in chains if c.is_set_piece]
    
    # Shot-ending open play chains
    shot_chains = [c for c in open_play_chains if c.ended_with_shot]
    dead_chains = [c for c in open_play_chains if not c.ended_with_shot]
    
    # SEPP: passes in shot-ending open-play possessions
    sepp_total = 0
    sepp_f3 = 0
    sepp_prog = 0
    sepp_pass_details = []
    
    for chain in shot_chains:
        for ev in chain.events:
            if ev.get('event') == 'Pass':
                sepp_total += 1
                if _is_final_third_pass(ev):
                    sepp_f3 += 1
                if _is_progressive_pass(ev):
                    sepp_prog += 1
                sepp_pass_details.append(_get_pass_details(ev))
    
    # Dead passes: passes in non-shot open-play possessions
    dead_passes = 0
    for chain in dead_chains:
        dead_passes += chain.pass_count
    
    # Set piece shot chains (reported separately)
    sp_shot_chains = [c for c in set_piece_chains if c.ended_with_shot]
    sp_passes = sum(c.pass_count for c in sp_shot_chains)
    
    # Calculations
    n_shot_chains = len(shot_chains)
    sepp_per_shot = round(sepp_total / n_shot_chains, 1) if n_shot_chains > 0 else 0
    total_open_passes = sepp_total + dead_passes
    efficiency = round((sepp_total / total_open_passes) * 100, 1) if total_open_passes > 0 else 0
    
    return {
        'sepp_total': sepp_total,
        'sepp_per_shot': sepp_per_shot,
        'sepp_f3': sepp_f3,
        'sepp_prog': sepp_prog,
        'dead_passes': dead_passes,
        'shot_chains': n_shot_chains,
        'dead_chains': len(dead_chains),
        'set_piece_shot_chains': len(sp_shot_chains),
        'set_piece_shot_passes': sp_passes,
        'efficiency': efficiency,
        'pass_details': sepp_pass_details,
    }


def _empty_sepp() -> Dict[str, Any]:
    return {
        'sepp_total': 0,
        'sepp_per_shot': 0,
        'sepp_f3': 0,
        'sepp_prog': 0,
        'dead_passes': 0,
        'shot_chains': 0,
        'dead_chains': 0,
        'set_piece_shot_chains': 0,
        'set_piece_shot_passes': 0,
        'efficiency': 0,
        'pass_details': [],
    }

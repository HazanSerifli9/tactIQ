import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from utils.possession_engine import extract_possession_chains

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def process_tempo_network(df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
    """
    Generate Tempo Network data: nodes, edges, player profiles.
    Returns:
    {
       'nodes': { 'Player Name': {'x': avg_x, 'y': avg_y, 'passes': count, ...} },
       'edges': { ('P1', 'P2'): {'count': c, 'avg_ttrp': t, 'avg_carry': c} },
       'player_profiles': [ {name, ttrp, carry, drawn_to, role} ]
    }
    """
    from utils.visuals import get_starting_xi
    
    top_11_players = get_starting_xi(df[df['team_name'] == team_name], 'player_name')
    
    chains = extract_possession_chains(df, team_name)
    
    player_data = defaultdict(lambda: {
        'events': 0, 'passes': 0, 'ttrp_sum': 0.0, 'carry_x_sum': 0.0,
        'pass_x_sum': 0.0, 'pass_y_sum': 0.0,
        'drawn_to_sum': 0, # How many carries resulted in passing TO this player
        'prog_passes': 0, 'backwards_passes': 0,
        'jersey_number': None # Added for display
    })
    
    edges_data = defaultdict(lambda: {
        'count': 0, 'ttrp_sum': 0.0, 'carry_x_sum': 0.0
    })

    def _event_time_seconds(ev):
        try:
            return float(ev.get('time_min', 0)) * 60 + float(ev.get('time_sec', 0))
        except:
            return 0.0

    for chain in chains:
        current_holder = None
        holder_start_time = None
        holder_start_x = None
        holder_start_y = None
        pending_pass = None
        
        for ev in chain.events:
            player = ev.get('player_name')
            if not player or pd.isna(player):
                continue
                
            time = _event_time_seconds(ev)
            x = float(ev.get('x', 0))
            y = float(ev.get('y', 0))
            
            if 'jersey_number' in ev and pd.notna(ev.get('jersey_number')):
                player_data[player]['jersey_number'] = int(float(ev['jersey_number']))
            
            if current_holder != player:
                # Flush pending pass
                if pending_pass is not None:
                    sender = pending_pass['sender']
                    edges_data[(sender, player)]['count'] += 1
                    edges_data[(sender, player)]['ttrp_sum'] += pending_pass['ttrp']
                    edges_data[(sender, player)]['carry_x_sum'] += pending_pass['carry_x']
                    
                    if pending_pass['carry_x'] > 5.0:
                        player_data[player]['drawn_to_sum'] += 1
                        
                    # Receive ball exact info
                    holder_start_time = pending_pass['pass_time']
                    holder_start_x = pending_pass['end_x']
                    holder_start_y = pending_pass['end_y']
                    pending_pass = None
                else:
                    holder_start_time = time
                    holder_start_x = x
                    holder_start_y = y
                
                current_holder = player
                
            # If it's a field event we can use, log it
            player_data[player]['events'] += 1
            
            if ev.get('event') == 'Pass' and ev.get('outcome') == 1:
                ttrp = max(0.0, time - holder_start_time) if holder_start_time is not None else 0.0
                carry_x = x - float(holder_start_x) if holder_start_x is not None else 0.0
                
                # Accumulate for sender
                player_data[player]['passes'] += 1
                player_data[player]['ttrp_sum'] += ttrp
                player_data[player]['carry_x_sum'] += carry_x
                player_data[player]['pass_x_sum'] += x
                player_data[player]['pass_y_sum'] += y
                
                end_x = float(ev.get('Pass End X', x))
                end_y = float(ev.get('Pass End Y', y))
                
                # Pass classification
                if end_x < x - 2.0:
                    player_data[player]['backwards_passes'] += 1
                dist_start = np.sqrt((100 - x)**2 + (50 - y)**2)
                dist_end = np.sqrt((100 - end_x)**2 + (50 - end_y)**2)
                if dist_end <= dist_start * 0.75:
                    player_data[player]['prog_passes'] += 1
                    
                pending_pass = {
                    'sender': player,
                    'pass_time': time,
                    'end_x': end_x,
                    'end_y': end_y,
                    'ttrp': ttrp,
                    'carry_x': carry_x
                }
                
    # Build Nodes
    nodes = {}
    for p, stats in player_data.items():
        if p in top_11_players and stats['passes'] > 0:
            nodes[p] = {
                'x': stats['pass_x_sum'] / stats['passes'],
                'y': stats['pass_y_sum'] / stats['passes'],
                'passes': stats['passes'],
                'jersey_number': stats['jersey_number']
            }

    # Build Edges
    edges = []
    for (sender, rec), stats in edges_data.items():
        if stats['count'] > 0:
            edges.append({
                'sender': sender,
                'receiver': rec,
                'count': stats['count'],
                'avg_ttrp': stats['ttrp_sum'] / stats['count'],
                'avg_carry_x': stats['carry_x_sum'] / stats['count']
            })

    # Prepare Player Profiles array
    profiles = []
    total_team_passes = sum(v['passes'] for v in player_data.values())
    
    for p, stats in player_data.items():
        if p not in top_11_players:
            continue
        if stats['passes'] < 5:  # filter noise
            continue
            
        passes = stats['passes']
        avg_ttrp = stats['ttrp_sum'] / passes
        avg_carry = stats['carry_x_sum'] / passes
        back_pct = stats['backwards_passes'] / passes
        prog_pct = stats['prog_passes'] / passes
        participation = passes / total_team_passes if total_team_passes else 0
        
        # Determine Role
        role = "Connector"
        if back_pct > 0.40:
            role = "Recycler"
        elif avg_carry > 3.0 or prog_pct > 0.20:
            role = "Direct"
        elif avg_ttrp < 3.2 and participation > 0.08:
            role = "Metronome"
            
        profiles.append({
            'Player': p,
            'jersey_number': stats['jersey_number'],
            'TTRP': round(avg_ttrp, 1),
            'Carry': round(avg_carry, 1),
            'Drawn To': stats['drawn_to_sum'],
            'Role': role,
            'passes': passes
        })
        
    profiles.sort(key=lambda x: x['passes'], reverse=True)
    
    # Take top 11
    top_profiles = profiles[:11]

    return {
        'nodes': nodes,
        'edges': edges,
        'profiles': top_profiles,
        'team_avg_ttrp': round(np.mean([p['TTRP'] for p in profiles]), 1) if profiles else 0.0,
        'team_total_connections': sum(e['count'] for e in edges)
    }


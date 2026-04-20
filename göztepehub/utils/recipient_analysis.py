import pandas as pd
import numpy as np
import os
import math

from utils.data import get_data_dir

GOZTEPE = 'Göztepe Spor Kulübü'

_MATCHES_CACHE = None

def _load_all_matches():
    global _MATCHES_CACHE
    if _MATCHES_CACHE is not None:
        return _MATCHES_CACHE

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    match_dfs = []
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' not in df.columns:
                continue
            if GOZTEPE in df['team_name'].unique().tolist():
                # Opta data is chronological by default, but let's sort just in case
                df = df.sort_values(by=['time_min', 'time_sec']).reset_index(drop=True)
                match_dfs.append(df)
        except Exception:
            continue

    _MATCHES_CACHE = match_dfs
    return match_dfs

def get_goztepe_players():
    dfs = _load_all_matches()
    players = set()
    for df in dfs:
        goz_df = df[df['team_name'] == GOZTEPE]
        players.update(goz_df['player_name'].dropna().unique().tolist())
    
    # Filter out empty or obviously bad names if any
    return sorted(list([p for p in players if isinstance(p, str) and len(p)>2]))

def is_progressive_pass(start_x, end_x):
    # Overly simple definition: moved ball at least 15m forward or strictly into penalty box
    if start_x < 50 and end_x >= 50:
        return True # entered opponent half
    if end_x - start_x > 15:
        return True
    if end_x > 83.3 and start_x < 83.3: # Entered final bit
        return True
    return False

def get_recipient_analysis(sender_name):
    dfs = _load_all_matches()
    
    # Store aggregated stats per recipient
    stats = {}
    
    for df in dfs:
        # Get all successful passes by sender
        sender_passes = df[(df['team_name'] == GOZTEPE) & 
                           (df['player_name'] == sender_name) & 
                           (df['event'] == 'Pass') & 
                           (df['outcome'] == 1)]
                           
        for idx in sender_passes.index:
            pass_row = df.loc[idx]
            pass_end_x = pass_row.get('Pass End X', pass_row.get('end_x', 0))
            pass_end_y = pass_row.get('Pass End Y', pass_row.get('end_y', 0))
            
            # The next event by Goztepe is the reception
            subsequent = df.loc[idx+1:]
            goz_subsequent = subsequent[subsequent['team_name'] == GOZTEPE]
            
            if goz_subsequent.empty:
                continue
                
            next_event = goz_subsequent.iloc[0]
            recipient = next_event['player_name']
            
            if pd.isna(recipient) or recipient == sender_name:
                continue
                
            if recipient not in stats:
                stats[recipient] = {
                    'receptions': 0, 'prog_passes': 0, 'prog_carries': 0,
                    'take_ons_won': 0, 'take_ons_attempted': 0, 'actions': []
                }
            
            stats[recipient]['receptions'] += 1
            
            e_type = next_event.get('event')
            e_out = next_event.get('outcome')
            e_x = next_event.get('x', pass_end_x)
            e_y = next_event.get('y', pass_end_y)
            
            action_type = "Other"
            end_x = e_x
            end_y = e_y
            
            # progressive carry check (did the player move with the ball before the next event?)
            if e_x - pass_end_x > 10:
                stats[recipient]['prog_carries'] += 1
                action_type = "Progressive Carry"
                end_x = e_x
                end_y = e_y
                # If they carried it and THEN passed it, the event is 'Pass', we still record the carry
                
            if e_type == 'Pass':
                e_end_x = next_event.get('Pass End X', next_event.get('end_x', e_x))
                e_end_y = next_event.get('Pass End Y', next_event.get('end_y', e_y))
                
                if e_out == 1:
                    if is_progressive_pass(e_x, e_end_x):
                        stats[recipient]['prog_passes'] += 1
                        action_type = "Progressive Pass"
                    elif action_type != "Progressive Carry":
                        action_type = "Successful Pass"
                else:
                    if action_type != "Progressive Carry":
                        action_type = "Unsuccessful Pass"
                
                # the line drawn should ideally be the carry + pass. 
                # For simplicity, we draw the pass line, or the carry line if it was a carry.
                if action_type in ["Progressive Pass", "Successful Pass", "Unsuccessful Pass"]:
                    end_x = e_end_x
                    end_y = e_end_y

            elif e_type == 'Take On':
                stats[recipient]['take_ons_attempted'] += 1
                if action_type != "Progressive Carry":
                    action_type = "Take-On"
                if e_out == 1:
                    stats[recipient]['take_ons_won'] += 1
            
            elif e_type in ['Shot', 'Goal']:
                if action_type != "Progressive Carry":
                    action_type = "Shot"
                
            stats[recipient]['actions'].append({
                'start_x': pass_end_x,
                'start_y': pass_end_y,
                'end_x': end_x,
                'end_y': end_y,
                'type': action_type
            })

    # Prepare return list
    recipient_list = []
    for p, v in stats.items():
        if v['receptions'] > 0: # Threshold can be changed
            rec = v['receptions']
            recipient_list.append({
                'Player': p,
                'Receptions': rec,
                'PP_per_R': round(v['prog_passes'] / rec, 2),
                'PC_per_R': round(v['prog_carries'] / rec, 2),
                'TO_Won_Att': f"{v['take_ons_won']}/{v['take_ons_attempted']}",
                'actions': v['actions']
            })
            
    # Sort by most receptions
    recipient_list.sort(key=lambda x: x['Receptions'], reverse=True)
    return recipient_list

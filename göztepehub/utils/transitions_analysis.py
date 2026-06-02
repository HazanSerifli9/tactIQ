import pandas as pd
import numpy as np
import os

from utils.data import get_data_dir

GOZTEPE = 'Göztepe Spor Kulübü'

# Events to skip (non-game events)
SKIP_EVENTS = {
    'Team setp up', 'Start', 'End', 'Start delay', 'End delay',
    'Injury Time Announcement', 'Card', 'Player Off', 'Player on',
    'Formation change', 'Collection End', 'Deleted event',
    'Referee Drop Ball', 'Contentious referee decision',
}

# Events that end a possession
SHOT_EVENTS = {'Goal', 'Miss', 'Saved Shot', 'Post'}
END_EVENTS = SHOT_EVENTS | {'Out', 'Offside Pass', 'Foul throw-in'}

# Box Zone boundaries
F3_X = 66.6

_OPPONENT_MATCHES_CACHE = {}


def _load_opponent_matches(team_name):
    """Load all parquet files where `team_name` plays, EXCLUDING Göztepe matches."""
    if team_name in _OPPONENT_MATCHES_CACHE:
        return _OPPONENT_MATCHES_CACHE[team_name]

    data_dir = get_data_dir()
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    match_dfs = []
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' not in df.columns:
                continue
            teams_in_match = df['team_name'].unique().tolist()
            if team_name in teams_in_match and GOZTEPE not in teams_in_match:
                match_dfs.append((filename, df))
        except Exception:
            continue

    _OPPONENT_MATCHES_CACHE[team_name] = match_dfs
    return match_dfs


def _get_third(x):
    if x < 33.3:
        return 'Defensive 3rd'
    elif x < 66.6:
        return 'Middle 3rd'
    else:
        return 'Final 3rd'


def extract_transitions_for_match(df, team_name):
    """
    Extract attacking (recoveries) and defensive (losses) transitions 
    for the given team in a single match.
    """
    attacking_transitions = []  # when team_name wins the ball
    defensive_transitions = []  # when team_name loses the ball

    # Clean the dataframe first to only have game events
    game_df = df[~df['event'].isin(SKIP_EVENTS)].reset_index(drop=True)

    if game_df.empty:
        return [], []

    current_team = None
    possession_start_idx = 0

    # To group events by sequence
    for i in range(len(game_df)):
        row = game_df.iloc[i]
        team = row['team_name']

        if team != current_team:
            # Possession shifted!
            if current_team is not None:
                # Analyze the shift
                prev_row = game_df.iloc[possession_start_idx - 1] if possession_start_idx > 0 else None
                end_row = game_df.iloc[i - 1]
                end_event = end_row['event']

                # It's a genuine turnover if it didn't end in a shot or out of bounds
                is_turnover = end_event not in END_EVENTS

                if is_turnover:
                    # If current_team was team_name, they LOST the ball -> Defensive Transition
                    if current_team == team_name:
                        defensive_transitions.append({
                            'loss_x': end_row['x'],
                            'loss_y': end_row['y'],
                            'zone': _get_third(end_row['x']),
                            'time_min': end_row['time_min'],
                            'time_sec': end_row['time_sec'],
                            'player': end_row.get('player_name', 'Unknown'),
                            'event': end_row['event'],
                            'seq_start_idx': i,  # The opponent's response starts here
                        })
                    # If the NEW team is team_name, they WON the ball -> Attacking Transition
                    elif team == team_name:
                        # Find exactly where they won it (the first action)
                        attacking_transitions.append({
                            'recovery_x': row['x'],
                            'recovery_y': row['y'],
                            'zone': _get_third(row['x']),
                            'time_min': row['time_min'],
                            'time_sec': row['time_sec'],
                            'player': row.get('player_name', 'Unknown'),
                            'event': row['event'],
                            'seq_start_idx': i,  # Team's response starts here
                        })

            current_team = team
            possession_start_idx = i

        # If a sequence ends naturally (Out, Shot), reset possession so the next event is a fresh start
        if row['event'] in END_EVENTS:
            current_team = None

    # Now, trace the 15-second window for each transition
    _analyze_windows(game_df, attacking_transitions, team_name, True)
    
    # For defensive transitions, we track the OPPONENT's actions in the next 15s
    teams = game_df['team_name'].unique().tolist()
    opponent = [t for t in teams if t != team_name]
    opp_name = opponent[0] if opponent else 'Unknown'
    
    _analyze_windows(game_df, defensive_transitions, opp_name, False)

    return attacking_transitions, defensive_transitions


def _analyze_windows(df, transitions, active_team, is_attacking):
    """
    Fill in the 10-second consequence data for a list of transitions.
    `active_team` is the team we are tracking in the 10 seconds 
    (the team doing the transition, or the opponent capitalizing on the loss).
    """
    for tr in transitions:
        start_idx = tr['seq_start_idx']
        start_row = df.iloc[start_idx]
        start_time = start_row['time_min'] * 60 + start_row['time_sec']
        
        passes = 0
        carries = 0
        duels_won = 0
        reached_f3 = False
        shot = False
        goal = False
        shot_coords = []
        
        events_in_10s = []

        for j in range(start_idx, len(df)):
            ev_row = df.iloc[j]
            if ev_row['team_name'] != active_team:
                # Possession lost before 10s
                break
                
            cur_time = ev_row['time_min'] * 60 + ev_row['time_sec']
            if cur_time - start_time > 10:
                # 10 seconds elapsed
                break
                
            # Track actions
            x = ev_row.get('x', 0)
            if x >= F3_X:
                reached_f3 = True
                
            events_in_10s.append(ev_row)
            
            e_type = ev_row['event']
            if e_type == 'Pass':
                passes += 1
            elif e_type in ('Ball touch', 'Take On'):
                carries += 1
            elif e_type == 'Duel':
                if ev_row.get('outcome') == 1:
                    duels_won += 1
            elif e_type in SHOT_EVENTS:
                shot = True
                shot_coords.append({
                    'x': ev_row.get('x', 88.5),
                    'y': ev_row.get('y', 50.0),
                    'event': e_type,
                    'player': ev_row.get('player_name', 'Unknown')
                })
                if e_type == 'Goal':
                    goal = True

        tr['events_count'] = len(events_in_10s)
        tr['passes'] = passes
        tr['carries'] = carries
        tr['duels_won'] = duels_won
        tr['reached_f3'] = reached_f3
        tr['shot'] = shot
        tr['goal'] = goal
        tr['shot_coords'] = shot_coords


def get_opponent_transition_profile(team_name):
    """
    Analyze transitions for an opponent across all their matches.
    """
    match_dfs = _load_opponent_matches(team_name)

    if not match_dfs:
        return None, None

    all_att = []
    all_def = []

    for filename, df in match_dfs:
        att, deff = extract_transitions_for_match(df, team_name)
        if att or deff:
            # Metadata
            teams = df['team_name'].unique().tolist()
            opponent = [t for t in teams if t != team_name]
            opp_name = opponent[0] if opponent else 'Unknown'
            week = int(df['week'].iloc[0]) if 'week' in df.columns else 0

            for a in att:
                a['match_week'] = week
                a['match_opp'] = opp_name
                a['filename'] = filename
            for d in deff:
                d['match_week'] = week
                d['match_opp'] = opp_name
                d['filename'] = filename

            all_att.extend(att)
            all_def.extend(deff)

    # Aggregate Attacking Transitions (Recoveries)
    att_profile = {}
    if all_att:
        att_profile['total'] = len(all_att)
        att_profile['zones'] = {
            'Defensive 3rd': sum(1 for x in all_att if x['zone'] == 'Defensive 3rd'),
            'Middle 3rd': sum(1 for x in all_att if x['zone'] == 'Middle 3rd'),
            'Final 3rd': sum(1 for x in all_att if x['zone'] == 'Final 3rd'),
        }
        att_profile['outcomes'] = {
            'f3_entry': sum(1 for x in all_att if x['reached_f3']),
            'shot': sum(1 for x in all_att if x['shot']),
            'goal': sum(1 for x in all_att if x['goal']),
        }
        # Player rankings
        players = {}
        for a in all_att:
            p = a['player']
            players[p] = players.get(p, 0) + 1
        
        # Sort top 10
        att_profile['top_players'] = sorted(players.items(), key=lambda item: item[1], reverse=True)[:10]
        att_profile['coords'] = [{'x': a['recovery_x'], 'y': a['recovery_y']} for a in all_att]
        att_profile['shot_coords'] = [sc for a in all_att for sc in a.get('shot_coords', [])]

    # Aggregate Defensive Transitions (Losses)
    def_profile = {}
    if all_def:
        def_profile['total'] = len(all_def)
        def_profile['zones'] = {
            'Defensive 3rd': sum(1 for x in all_def if x['zone'] == 'Defensive 3rd'),
            'Middle 3rd': sum(1 for x in all_def if x['zone'] == 'Middle 3rd'),
            'Final 3rd': sum(1 for x in all_def if x['zone'] == 'Final 3rd'),
        }
        def_profile['outcomes'] = {
            'opp_f3_entry': sum(1 for x in all_def if x['reached_f3']),
            'opp_shot': sum(1 for x in all_def if x['shot']),
            'opp_goal': sum(1 for x in all_def if x['goal']),
        }
        # Player rankings for losing the ball
        players = {}
        for d in all_def:
            p = d['player']
            players[p] = players.get(p, 0) + 1
        
        def_profile['top_players'] = sorted(players.items(), key=lambda item: item[1], reverse=True)[:10]
        def_profile['coords'] = [{'x': d['loss_x'], 'y': d['loss_y']} for d in all_def]
        def_profile['shot_coords'] = [sc for d in all_def for sc in d.get('shot_coords', [])]

    return att_profile, def_profile

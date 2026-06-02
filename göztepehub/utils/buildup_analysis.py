
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

# Cache for fully-analyzed buildup results (expensive computation)
_BUILDUP_ANALYSIS_CACHE = {}


def _load_opponent_matches(team_name):
    """
    Load all parquet files where `team_name` plays, EXCLUDING Göztepe matches.
    Returns list of (filename, DataFrame) tuples.
    """
    from göztepehub.utils.transitions_analysis import _load_opponent_matches as load_matches
    return load_matches(team_name)


def _get_zone(y):
    """Determine zone from y-coordinate."""
    if y < 33.3:
        return 'Left'
    elif y <= 66.6:
        return 'Center'
    else:
        return 'Right'


def _extract_possession_sequences(records, team_name):
    """
    Extract possession sequences from a pre-converted list of row dicts.
    Returns list of (seq, end_record_idx) tuples where end_record_idx is the
    position of the last event in `records` for that sequence.
    """
    sequences = []
    current_seq = []
    current_team = None
    current_end_idx = -1

    for i, row in enumerate(records):
        if row['event'] in SKIP_EVENTS:
            continue

        team = row['team_name']

        if team != current_team:
            if current_seq and current_team == team_name:
                sequences.append((current_seq, current_end_idx))
            current_seq = [row]
            current_team = team
            current_end_idx = i
        else:
            current_seq.append(row)
            current_end_idx = i

        if row['event'] in END_EVENTS:
            if current_seq and current_team == team_name:
                sequences.append((current_seq, current_end_idx))
            current_seq = []
            current_team = None
            current_end_idx = -1

    if current_seq and current_team == team_name:
        sequences.append((current_seq, current_end_idx))

    return sequences


def _analyze_buildup_sequence(seq):
    """
    Analyze a single build-up sequence (starts in own half, x < 50).
    Returns a dict with analysis results, or None if not a valid build-up.
    """
    if not seq:
        return None

    start_x = seq[0]['x']
    if start_x >= 50:
        return None  # Not a build-up from own half

    start_time = seq[0]['time_min'] * 60 + seq[0]['time_sec']
    start_y = seq[0]['y']
    zone = _get_zone(start_y)

    # Find first pass to determine type
    pass_type = 'Short'
    for ev in seq:
        if ev['event'] == 'Pass':
            if ev.get('Long ball') == 'Si':
                pass_type = 'Long'
            break

    # Track the 10-second window
    reached_f3 = False
    had_shot = False
    shot_on_target = False
    goal = False
    events_in_window = []

    for ev in seq:
        ev_time = ev['time_min'] * 60 + ev['time_sec']
        elapsed = ev_time - start_time

        if elapsed <= 10:
            events_in_window.append(ev)
            if ev['x'] > 66.6:
                reached_f3 = True
            if ev['event'] in SHOT_EVENTS:
                had_shot = True
                if ev['event'] in ('Goal', 'Saved Shot'):
                    shot_on_target = True
                if ev['event'] == 'Goal':
                    goal = True
        else:
            break

    # Determine sequence duration and outcome
    last_ev = seq[-1]
    full_duration = last_ev['time_min'] * 60 + last_ev['time_sec'] - start_time
    end_event = last_ev['event']

    # Check if team lost ball (sequence ended because opponent took over)
    turnover = end_event not in SHOT_EVENTS and end_event != 'Out'

    return {
        'start_x': start_x,
        'start_y': start_y,
        'zone': zone,
        'pass_type': pass_type,
        'duration': min(full_duration, 10),
        'events_count': len(events_in_window),
        'reached_f3': reached_f3,
        'had_shot': had_shot,
        'shot_on_target': shot_on_target,
        'goal': goal,
        'turnover': turnover,
        'end_event': end_event,
        'events': events_in_window,
        'start_min': events_in_window[0]['time_min'] if events_in_window else 0,
        'start_sec': events_in_window[0]['time_sec'] if events_in_window else 0,
        'end_min': events_in_window[-1]['time_min'] if events_in_window else 0,
        'end_sec': events_in_window[-1]['time_sec'] if events_in_window else 0,
    }


def _analyze_post_turnover(df, team_name, seq):
    """
    After a turnover at the end of a sequence, check if the opponent
    reaches the team's final third (opponent x > 66.6) within 10 seconds.
    """
    if not seq:
        return False

    last_ev = seq[-1]
    turnover_time = last_ev['time_min'] * 60 + last_ev['time_sec']

    mask = (
        (df['time_min'] == last_ev['time_min']) &
        (df['time_sec'] == last_ev['time_sec']) &
        (df['event'] == last_ev['event'])
    )
    matching = df.index[mask]
    if len(matching) == 0:
        return False
    turnover_idx = matching[0]

    subsequent = df.loc[turnover_idx + 1:]
    for row in subsequent.to_dict('records'):
        ev_time = row['time_min'] * 60 + row['time_sec']
        if ev_time - turnover_time > 10:
            break
        if row['team_name'] != team_name and row['x'] > 66.6:
            return True
        if row['event'] in SHOT_EVENTS and row['team_name'] != team_name:
            return True

    return False


# ============================================================
# FINAL THIRD (F3) ENTRY ANALYSIS
# ============================================================

F3_THRESHOLD = 66.6  # x >= this = final third

# Box zone columns from Opta qualifiers
BOX_ZONE_COLS = [
    'Box-centre', 'Box-left', 'Box-right',
    'Box-deep left', 'Box-deep right',
    'Small box-centre', 'Small box-left', 'Small box-right',
]


def _detect_f3_entry_event(events_in_window):
    """
    Walk through the events in a build-up sequence and find the first event
    that transitions into the final third (x >= 66.6 while the preceding
    event was x < 66.6).

    Returns a dict describing the entry, or None.
    """
    prev_x = None
    for i, ev in enumerate(events_in_window):
        cur_x = ev['x']
        if prev_x is not None and prev_x < F3_THRESHOLD and cur_x >= F3_THRESHOLD:
            # Determine entry method
            if ev['event'] == 'Pass':
                if ev.get('Long ball') == 'Si':
                    method = 'Deep Pass'
                else:
                    method = 'Short Pass'
            elif ev['event'] in ('Ball touch', 'Take On'):
                method = 'Ball Carry'
            else:
                # Treat any other event moving into F3 as a carry
                method = 'Ball Carry'

            # Entry zone based on y-coordinate at the point of entry
            zone = _get_zone(ev['y'])

            entry_time = ev['time_min'] * 60 + ev['time_sec']

            return {
                'method': method,
                'zone': zone,
                'entry_x': cur_x,
                'entry_y': ev['y'],
                'entry_time': entry_time,
                'entry_event_idx': i,
            }
        prev_x = cur_x

    # If the very first event is already in F3 but the build-up started from own half,
    # check Pass End X to detect passes that land in F3
    for i, ev in enumerate(events_in_window):
        if ev['event'] == 'Pass' and ev.get('Pass End X') is not None:
            try:
                end_x = float(ev['Pass End X'])
            except (ValueError, TypeError):
                continue
            if ev['x'] < F3_THRESHOLD and end_x >= F3_THRESHOLD:
                if ev.get('Long ball') == 'Si':
                    method = 'Deep Pass'
                else:
                    method = 'Short Pass'
                end_y = float(ev.get('Pass End Y', ev['y']))
                zone = _get_zone(end_y)
                entry_time = ev['time_min'] * 60 + ev['time_sec']
                return {
                    'method': method,
                    'zone': zone,
                    'entry_x': end_x,
                    'entry_y': end_y,
                    'entry_time': entry_time,
                    'entry_event_idx': i,
                }

    return None


def _analyze_subsequent_actions(events_in_window, entry_info):
    """
    After the F3 entry event, look at subsequent events within 10 seconds
    to classify follow-up actions:
      - box_control: any action inside the penalty box (Box-* qualifiers or x >= 83.5)
      - cross: a Cross qualifier on a pass
      - aerial_won: Aerial event with outcome == 1
    Returns a dict of booleans.
    """
    result = {
        'box_control': False,
        'cross': False,
        'aerial_won': False,
    }

    if entry_info is None:
        return result

    entry_time = entry_info['entry_time']
    start_idx = entry_info['entry_event_idx'] + 1

    for ev in events_in_window[start_idx:]:
        ev_time = ev['time_min'] * 60 + ev['time_sec']
        if ev_time - entry_time > 10:
            break

        # Box control: Opta Box-* qualifier or inside penalty area (x >= 83.5, 16.5 < y < 83.5)
        if not result['box_control']:
            for col in BOX_ZONE_COLS:
                if ev.get(col) == 'Si':
                    result['box_control'] = True
                    break
            if not result['box_control'] and ev['x'] >= 83.5 and 16.5 < ev['y'] < 83.5:
                result['box_control'] = True

        # Cross
        if not result['cross'] and ev.get('Cross') == 'Si':
            result['cross'] = True

        # Aerial duel won
        if not result['aerial_won'] and ev['event'] == 'Aerial' and ev.get('outcome') == 1:
            result['aerial_won'] = True

    return result


def _compute_f3_entry_stats(buildups):
    """
    From the list of build-up dicts (each including events list),
    compute aggregate F3 entry statistics.
    Returns a dict with counts/percentages or None if no entries.
    """
    entries = []
    for b in buildups:
        evs = b.get('events', [])
        if not evs:
            continue
        entry = _detect_f3_entry_event(evs)
        if entry is None:
            continue
        subseq = _analyze_subsequent_actions(evs, entry)
        entries.append({**entry, **subseq})

    total = len(entries)
    if total == 0:
        return None

    def pct(n):
        return round((n / total) * 100, 1) if total > 0 else 0

    # Entry method breakdown
    short_pass = sum(1 for e in entries if e['method'] == 'Short Pass')
    deep_pass = sum(1 for e in entries if e['method'] == 'Deep Pass')
    carry = sum(1 for e in entries if e['method'] == 'Ball Carry')

    # Entry zone breakdown
    left = sum(1 for e in entries if e['zone'] == 'Left')
    center = sum(1 for e in entries if e['zone'] == 'Center')
    right = sum(1 for e in entries if e['zone'] == 'Right')

    # Subsequent actions
    box_ctrl = sum(1 for e in entries if e['box_control'])
    cross = sum(1 for e in entries if e['cross'])
    aerial = sum(1 for e in entries if e['aerial_won'])

    return {
        'total_entries': total,
        'entry_method': {
            'short_pass': short_pass, 'short_pass_pct': pct(short_pass),
            'deep_pass': deep_pass, 'deep_pass_pct': pct(deep_pass),
            'carry': carry, 'carry_pct': pct(carry),
        },
        'entry_zone': {
            'left': left, 'left_pct': pct(left),
            'center': center, 'center_pct': pct(center),
            'right': right, 'right_pct': pct(right),
        },
        'subsequent': {
            'box_control': box_ctrl, 'box_control_pct': pct(box_ctrl),
            'cross': cross, 'cross_pct': pct(cross),
            'aerial_won': aerial, 'aerial_won_pct': pct(aerial),
        },
        'entry_coords': [{'x': e['entry_x'], 'y': e['entry_y'], 'method': e['method']} for e in entries],
    }


def analyze_buildup_for_match(df, team_name):
    """
    Analyze all build-up sequences for a team in a single match.
    Returns a summary dict.
    """
    df_clean = df.copy()
    flip = False
    if 'team_position' in df_clean.columns:
        pos_series = df_clean[df_clean['team_name'] == team_name]['team_position']
        if not pos_series.empty and pos_series.iloc[0] == 'away':
            flip = True

    if flip:
        df_clean['x'] = 100.0 - df_clean['x']
        df_clean['y'] = 100.0 - df_clean['y']
        if 'Pass End X' in df_clean.columns:
            try:
                df_clean['Pass End X'] = 100.0 - pd.to_numeric(df_clean['Pass End X'], errors='coerce')
            except Exception:
                pass
        if 'Pass End Y' in df_clean.columns:
            try:
                df_clean['Pass End Y'] = 100.0 - pd.to_numeric(df_clean['Pass End Y'], errors='coerce')
            except Exception:
                pass

    records = df_clean.to_dict('records')
    seq_with_idx = _extract_possession_sequences(records, team_name)

    # First pass: analyze each sequence, collecting (result, end_record_idx)
    raw_buildups = []
    for seq, end_idx in seq_with_idx:
        result = _analyze_buildup_sequence(seq)
        if result is not None:
            raw_buildups.append((result, end_idx))

    # Batch compute post-turnover danger in a single forward pass per turnover.
    # For each turnover sequence, scan forward in records from its end position.
    # This avoids N separate vectorized mask operations on the dataframe.
    danger_indices = set()
    for i, (result, end_idx) in enumerate(raw_buildups):
        if not result['turnover'] or end_idx < 0:
            continue
        turnover_time = records[end_idx]['time_min'] * 60 + records[end_idx]['time_sec']
        for j in range(end_idx + 1, len(records)):
            ev = records[j]
            ev_time = ev['time_min'] * 60 + ev['time_sec']
            if ev_time - turnover_time > 10:
                break
            # Since coordinate flipping ensures the analyzed team always attacks left-to-right (0->100),
            # the defending goal is at x=0. Opponent entering defensive third means opponent x < 33.3.
            if ev['team_name'] != team_name and (ev['x'] < 33.3 or ev['event'] in SHOT_EVENTS):
                danger_indices.add(i)
                break

    buildups = []
    for i, (result, _) in enumerate(raw_buildups):
        result['opp_f3_after_turnover'] = (i in danger_indices)
        buildups.append(result)

    if not buildups:
        return None

    total = len(buildups)

    # Pass type distribution
    long_count = sum(1 for b in buildups if b['pass_type'] == 'Long')
    short_count = total - long_count

    # Zone distribution
    left_count = sum(1 for b in buildups if b['zone'] == 'Left')
    center_count = sum(1 for b in buildups if b['zone'] == 'Center')
    right_count = sum(1 for b in buildups if b['zone'] == 'Right')

    # 10s outcomes
    f3_entries = sum(1 for b in buildups if b['reached_f3'])
    shots = sum(1 for b in buildups if b['had_shot'])
    sot = sum(1 for b in buildups if b['shot_on_target'])
    goals = sum(1 for b in buildups if b['goal'])
    turnovers = sum(1 for b in buildups if b['turnover'])
    opp_danger = sum(1 for b in buildups if b['opp_f3_after_turnover'])

    def pct(n):
        return round((n / total) * 100, 1) if total > 0 else 0

    # Get opponent name
    teams_in_match = df['team_name'].unique().tolist()
    opponent = [t for t in teams_in_match if t != team_name]
    opponent_name = opponent[0] if opponent else 'Unknown'

    # Get match description / week
    week = int(df['week'].iloc[0]) if 'week' in df.columns else 0
    description = str(df['description'].iloc[0]) if 'description' in df.columns else ''

    # Collect coordinates mapping for visualization
    coords = [{'x': b['start_x'], 'y': b['start_y'], 'pass_type': b['pass_type']} for b in buildups]

    # F3 Entry deep analysis
    f3_entry_stats = _compute_f3_entry_stats(buildups)

    # Add the full sequences for minute-by-minute views
    # Sort them by time
    buildups.sort(key=lambda x: (x.get('start_min', 0), x.get('start_sec', 0)))

    return {
        'opponent': opponent_name,
        'week': week,
        'description': description,
        'total_buildups': total,
        'coords': coords,
        'sequences': buildups,
        'f3_entry_stats': f3_entry_stats,
        'pass_type': {
            'long': long_count, 'short': short_count,
            'long_pct': pct(long_count), 'short_pct': pct(short_count),
        },
        'zone': {
            'left': left_count, 'center': center_count, 'right': right_count,
            'left_pct': pct(left_count), 'center_pct': pct(center_count), 'right_pct': pct(right_count),
        },
        'outcomes_10s': {
            'f3_entry': f3_entries, 'f3_entry_pct': pct(f3_entries),
            'shot': shots, 'shot_pct': pct(shots),
            'sot': sot, 'sot_pct': pct(sot),
            'goal': goals, 'goal_pct': pct(goals),
            'turnover': turnovers, 'turnover_pct': pct(turnovers),
            'opp_danger': opp_danger, 'opp_danger_pct': pct(opp_danger),
        },
    }


def get_opponent_buildup_analysis(team_name):
    """
    Get build-up analysis for a given team across all their matches
    (excluding Göztepe matches).
    Returns a list of match analysis dicts + a season summary.
    """
    if team_name in _BUILDUP_ANALYSIS_CACHE:
        return _BUILDUP_ANALYSIS_CACHE[team_name]

    match_dfs = _load_opponent_matches(team_name)

    if not match_dfs:
        _BUILDUP_ANALYSIS_CACHE[team_name] = ([], None)
        return [], None

    match_analyses = []
    for filename, df in match_dfs:
        result = analyze_buildup_for_match(df, team_name)
        if result:
            result['source_file'] = filename
            match_analyses.append(result)

    match_analyses.sort(key=lambda x: x['week'])

    if not match_analyses:
        _BUILDUP_ANALYSIS_CACHE[team_name] = ([], None)
        return [], None

    # Aggregate all coordinates for the season summary
    all_coords = []
    for m in match_analyses:
        all_coords.extend(m.get('coords', []))

    # Season summary (aggregate)
    total_buildups = sum(m['total_buildups'] for m in match_analyses)
    total_long = sum(m['pass_type']['long'] for m in match_analyses)
    total_short = sum(m['pass_type']['short'] for m in match_analyses)
    total_left = sum(m['zone']['left'] for m in match_analyses)
    total_center = sum(m['zone']['center'] for m in match_analyses)
    total_right = sum(m['zone']['right'] for m in match_analyses)
    total_f3 = sum(m['outcomes_10s']['f3_entry'] for m in match_analyses)
    total_shots = sum(m['outcomes_10s']['shot'] for m in match_analyses)
    total_sot = sum(m['outcomes_10s']['sot'] for m in match_analyses)
    total_goals = sum(m['outcomes_10s']['goal'] for m in match_analyses)
    total_turnovers = sum(m['outcomes_10s']['turnover'] for m in match_analyses)
    total_opp_danger = sum(m['outcomes_10s']['opp_danger'] for m in match_analyses)
    season_f3_entries = []
    for m in match_analyses:
        f3_stats = m.get('f3_entry_stats') or {}
        season_f3_entries.extend(f3_stats.get('entry_coords', []))

    def pct(n):
        return round((n / total_buildups) * 100, 1) if total_buildups > 0 else 0

    season_summary = {
        'matches_analyzed': len(match_analyses),
        'total_buildups': total_buildups,
        'avg_buildups_per_match': round(total_buildups / len(match_analyses), 1),
        'pass_type': {
            'long_pct': pct(total_long), 'short_pct': pct(total_short),
        },
        'zone': {
            'left_pct': pct(total_left), 'center_pct': pct(total_center), 'right_pct': pct(total_right),
        },
        'outcomes_10s': {
            'f3_entry': total_f3,
            'f3_entry_pct': pct(total_f3),
            'shot': total_shots,
            'shot_pct': pct(total_shots),
            'sot': total_sot,
            'sot_pct': pct(total_sot),
            'goal': total_goals,
            'goal_pct': pct(total_goals),
            'turnover': total_turnovers,
            'turnover_pct': pct(total_turnovers),
            'opp_danger': total_opp_danger,
            'opp_danger_pct': pct(total_opp_danger),
        },
        'f3_entry_stats': {
            'total_entries': len(season_f3_entries),
            'entry_coords': season_f3_entries,
        },
        'coords': all_coords,
    }

    _BUILDUP_ANALYSIS_CACHE[team_name] = (match_analyses, season_summary)
    return match_analyses, season_summary

def get_opponent_matches_list(team_name):
    """
    Returns a sorted list of available matches for an opponent (excluding Göztepe).
    Format: [{'filename': str, 'week': int, 'opponent': str, 'description': str}, ...]
    """
    match_dfs = _load_opponent_matches(team_name)
    
    matches = []
    for filename, df in match_dfs:
        teams_in_match = df['team_name'].unique().tolist()
        opponent = [t for t in teams_in_match if t != team_name]
        opponent_name = opponent[0] if opponent else 'Unknown'
        week = int(df['week'].iloc[0]) if 'week' in df.columns else 0
        desc = str(df['description'].iloc[0]) if 'description' in df.columns else ''
        
        matches.append({
            'filename': filename,
            'week': week,
            'opponent': opponent_name,
            'description': desc,
            'label': f"Week {week} vs {opponent_name}"
        })
        
    # Sort by week
    matches.sort(key=lambda x: x['week'])
    return matches

def get_single_match_buildups(filename, team_name):
    """
    Load a single match parquet file and return the build-up analysis.
    """
    data_dir = get_data_dir()
    filepath = os.path.join(data_dir, filename)
    
    try:
        df = pd.read_parquet(filepath)
        result = analyze_buildup_for_match(df, team_name)
        if result:
            result['source_file'] = filename
        return result
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def _precompute_all_teams():
    """Precompute buildup analysis for all teams in background at startup."""
    try:
        from utils.data import extract_fixture_data, calculate_standings
        matches = extract_fixture_data(lite=True)
        standings = calculate_standings(matches)
        teams = [t for t in standings['Team'].unique() if t != GOZTEPE]
        for team in teams:
            if team not in _BUILDUP_ANALYSIS_CACHE:
                get_opponent_buildup_analysis(team)
    except Exception:
        pass


# Keep analysis on-demand. Eagerly processing every club delays the selected
# opponent on the pre-match page and duplicates the league benchmark warm-up.

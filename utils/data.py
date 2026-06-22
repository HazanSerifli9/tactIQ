
import pandas as pd
import os
import math

from shared.logger import get_logger

logger = get_logger(__name__)

try:
    from utils.xg_model import predict_xg
except ImportError:
    predict_xg = lambda df: df

TEAM_LOGOS = {
    "Alanyaspor Kulübü": "assets/alanyasporlogo.png",
    "Antalyaspor Kulübü": "assets/antalyasporlogo.png",
    "Beşiktaş Jimnastik Kulübü": "assets/bjklogo.png",
    "Eyüp Spor Kulübü": "assets/eyuplogo.png",
    "Fatih Karagümrük Spor Kulübü": "assets/fatihlogo.png",
    "Fenerbahçe Spor Kulübü": "assets/fblogo.png",
    "Galatasaray Spor Kulübü": "assets/gslogo.png",
    "Gaziantep Futbol Kulübü": "assets/gazianteplogo.png",
    "Gençlerbirliği Spor Kulübü": "assets/genclerbirligilogo.png",
    "Göztepe Spor Kulübü": "assets/goztepelogo.png",
    "Kasımpaşa Spor Kulübü": "assets/kasımpasalogo.png",
    "Kayseri Spor Kulübü": "assets/Kayserisporlogo.png",
    "Kocaelispor Kulübü": "assets/kocaelisporlogo.png",
    "Konyaspor Kulübü": "assets/konyalogo.png",
    "Samsunspor Kulübü": "assets/samsunlogo.png",
    "Trabzonspor Kulübü": "assets/tslogo.png",
    "Çaykur Rize Spor Kulübü": "assets/rizelogo.png",
    "İstanbul Başakşehir Futbol Kulübü": "assets/basaksehirlogo.png",
    "League": "assets/superlig_logo.jpg"
}

def get_data_dir():
    # Helper to find data dir relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "raw_data")

def _get_dir_signature(directory_path):
    """Get a signature of the directory state (file count + latest mtime) for cache invalidation."""
    try:
        parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
        file_count = len(parquet_files)
        latest_mtime = 0
        for f in parquet_files:
            mtime = os.path.getmtime(os.path.join(directory_path, f))
            if mtime > latest_mtime:
                latest_mtime = mtime
        return (file_count, latest_mtime)
    except Exception as e:
        logger.warning("Could not read dir signature for %s: %s", directory_path, e)
        return (0, 0)

def extract_fixture_data(directory_path=None, target_weeks=None, lite=False):
    if directory_path is None:
        directory_path = get_data_dir()

    matches = []
    
    if not os.path.exists(directory_path):
        print(f"Data directory not found: {directory_path}")
        return []

    files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    # Check if directory has changed since last cache
    current_sig = _get_dir_signature(directory_path)
    
    # Memory cache key based on lite/full
    cache_key = 'lite' if lite else 'full'
    if hasattr(extract_fixture_data, "cache"):
        cached_sig = extract_fixture_data.cache.get('_dir_sig')
        if cached_sig != current_sig:
            # Directory changed, invalidate all caches
            extract_fixture_data.cache = {'_dir_sig': current_sig}
        elif extract_fixture_data.cache.get(cache_key):
             return extract_fixture_data.cache[cache_key]
    else:
        extract_fixture_data.cache = {'_dir_sig': current_sig}

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            # Optimize: Check first if it's a match file before reading whole DF?
            # For now read headers or just read full as before.
            df = pd.read_parquet(file_path)
            try:
                from utils.xg_model import predict_xg
                df = predict_xg(df)
            except Exception as e:
                logger.debug("xG model skipped for %s: %s", filename, e)
            
            if 'week' in df.columns and not df.empty:
                week = int(df['week'].iloc[0])
                
                # If target_weeks is None, extract all. Otherwise filter.
                if target_weeks is None or week in target_weeks:
                    match_name = str(df['description'].iloc[0])
                    
                    # Deduplicate matches based on description and week
                    if any(m['match_name'] == match_name and m['week'] == week for m in matches):
                        continue

                    # Get teams based on position
                    home_team_name = df[df['team_position'] == 'home']['team_name'].iloc[0]
                    away_team_name = df[df['team_position'] == 'away']['team_name'].iloc[0]
                    
                    def get_team_stats(team_name, opponent_name):
                        team_df = df[df['team_name'] == team_name]
                        opponent_df = df[df['team_name'] == opponent_name]
                        
                        # Goals (type_id 16)
                        # Correct logic: 
                        # 1. Goals scored by team (excluding own goals)
                        # 2. Own goals scored by opponent (benefiting team)
                        
                        has_own_goal_col = 'own goal' in df.columns
                        
                        if has_own_goal_col:
                            team_goals_scored = len(team_df[(team_df['type_id'] == 16) & (team_df['own goal'] != 'Si')])
                            opponent_own_goals = len(opponent_df[(opponent_df['type_id'] == 16) & (opponent_df['own goal'] == 'Si')])
                            goals = team_goals_scored + opponent_own_goals
                        else:
                            # Fallback if no own goal column
                            goals = len(team_df[team_df['type_id'] == 16])
                        
                        # Shots (Goal + Miss + Post + Saved Shot)
                        # Exclude own goals from shot counts
                        shot_types = [16, 13, 14, 15]
                        if has_own_goal_col:
                            shots = len(team_df[team_df['type_id'].isin(shot_types) & (team_df['own goal'] != 'Si')])
                        else:
                            shots = len(team_df[team_df['type_id'].isin(shot_types)])
                        
                        # Shots on Target (Goal + Saved Shot)
                        target_types = [16, 15]
                        if has_own_goal_col:
                            shots_on_target = len(team_df[team_df['type_id'].isin(target_types) & (team_df['own goal'] != 'Si')])
                        else:
                            shots_on_target = len(team_df[team_df['type_id'].isin(target_types)])
                        
                        # Passes (type_id 1)
                        total_passes = len(team_df[team_df['type_id'] == 1])
                        success_passes = len(team_df[(team_df['type_id'] == 1) & (team_df['outcome'] == 1)])
                        pass_accuracy = round((success_passes / total_passes * 100), 1) if total_passes > 0 else 0
                        
                        # Fouls (type_id 4)
                        fouls = len(team_df[team_df['type_id'] == 4])
                        
                        # Corners (type_id 6)
                        corners = len(team_df[team_df['type_id'] == 6])
                        
                        # Cards (type_id 17)
                        cards = len(team_df[team_df['type_id'] == 17])
                        
                        xg = round(team_df['xG'].sum(), 2) if 'xG' in team_df.columns else 0
                        
                        return {
                            'goals': goals,
                            'shots': shots,
                            'shots_on_target': shots_on_target,
                            'passes': total_passes,
                            'pass_accuracy': pass_accuracy,
                            'fouls': fouls,
                            'corners': corners,
                            'cards': cards,
                            'xg': xg
                        }


                    if lite:
                        # Skip expensive stats but calculate goals for standings
                        def get_goals_lite(team_name, opponent_name):
                            team_df = df[df['team_name'] == team_name]
                            opponent_df = df[df['team_name'] == opponent_name]
                            has_own_goal_col = 'own goal' in df.columns
                            if has_own_goal_col:
                                team_goals_scored = len(team_df[(team_df['type_id'] == 16) & (team_df['own goal'] != 'Si')])
                                opponent_own_goals = len(opponent_df[(opponent_df['type_id'] == 16) & (opponent_df['own goal'] == 'Si')])
                                goals = team_goals_scored + opponent_own_goals
                            else:
                                goals = len(team_df[team_df['type_id'] == 16])
                            xg = round(team_df['xG'].sum(), 2) if 'xG' in team_df.columns else 0
                            return goals, xg

                        g1, xg1 = get_goals_lite(home_team_name, away_team_name)
                        g2, xg2 = get_goals_lite(away_team_name, home_team_name)
                        stats = {
                          'team1': {'goals': g1, 'xg': xg1},
                          'team2': {'goals': g2, 'xg': xg2}
                        }
                        key_players = []
                    else:
                        stats = {
                            'team1': get_team_stats(home_team_name, away_team_name),
                            'team2': get_team_stats(away_team_name, home_team_name)
                        }

                        # Advanced Top Players (User Logic)
                        try:
                            from utils.stats import calculate_match_stats
                            key_players = calculate_match_stats(df)
                        except Exception as e:
                            logger.warning("Top player calc failed for %s: %s", filename, e)
                            key_players = []

                        # Fallback if advanced calc returns empty or fails
                        if not key_players:
                            # Key Players Extraction (Basic)
                            goals_df = df[(df['event'] == 'Goal') & (df['outcome'] == 1)]
                            if not goals_df.empty:
                                for _, row in goals_df.iterrows():
                                    key_players.append({
                                        'name': row['player_name'],
                                        'team': row['team_name'],
                                        'reason': f"Goal ({row['time_min']}')"
                                    })
                            
                            if len(key_players) < 3:
                                event_counts = df['player_name'].value_counts()
                                for player, count in event_counts.items():
                                    if len(key_players) >= 3:
                                        break
                                    if any(kp['name'] == player for kp in key_players):
                                        continue
                                    player_team = df[df['player_name'] == player]['team_name'].iloc[0]
                                    key_players.append({
                                        'name': player,
                                        'team': player_team,
                                        'reason': "Key Player"
                                    })

                    match_info = {
                        'week': week,
                        'date': str(df['local_date'].iloc[0]),
                        'time': str(df['local_time'].iloc[0]),
                        'match_name': f"{home_team_name} vs {away_team_name}",
                        'venue': str(df['venue_long_name'].iloc[0]) if 'venue_long_name' in df.columns else 'Unknown Venue',
                        'competition': str(df['competition_name'].iloc[0]) if 'competition_name' in df.columns else 'Süper Lig',
                        'competition_logo': TEAM_LOGOS.get("League"),
                        'stats': stats,
                        'team_names': [home_team_name, away_team_name],
                        'logos': [TEAM_LOGOS.get(home_team_name, "assets/logo.png"), TEAM_LOGOS.get(away_team_name, "assets/logo.png")],
                        'source_file': filename,
                        'key_players': key_players
                    }
                    
                    matches.append(match_info)
                        
        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)
            
    matches.sort(key=lambda x: (x['week'], x['date'], x['time']))
    extract_fixture_data.cache[cache_key] = matches
    return matches

def get_match_dataframe(filename):
    data_dir = get_data_dir()
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        df['source_file'] = filename
        
        # Apply Custom xG Model automatically
        try:
            df = predict_xg(df)
        except Exception as e:
            logger.warning("Failed to apply xG model on %s: %s", filename, e)
            
        return df
    return None

def calculate_match_probabilities(xg_h, xg_a):
    if xg_h == 0 and xg_a == 0:
        return 0.0, 1.0, 0.0
    def poisson(k, lmbda):
        return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)
    prob_h_win = 0
    prob_a_win = 0
    prob_draw = 0
    for h in range(8):
        for a in range(8):
            prob = poisson(h, xg_h) * poisson(a, xg_a)
            if h > a: prob_h_win += prob
            elif a > h: prob_a_win += prob
            else: prob_draw += prob
    total = prob_h_win + prob_a_win + prob_draw
    return prob_h_win/total, prob_draw/total, prob_a_win/total

def calculate_standings(matches):
    """
    Calculates the league table from a list of match dictionaries.
    Returns a DataFrame sorted by Points, GD, GF.
    """
    standings = {}

    for match in matches:
        # Check if stats exist (some matches might be future fixtures or empty)
        if 'stats' not in match or not match['stats']:
            continue
            
        # Extract data
        t1_name = match['team_names'][0]
        t2_name = match['team_names'][1]
        
        # Ensure teams are in standings dict
        for t in [t1_name, t2_name]:
            if t not in standings:
                standings[t] = {
                    'Team': t, 'Played': 0, 'Won': 0, 'Drawn': 0, 'Lost': 0, 
                    'GF': 0, 'GA': 0, 'GD': 0, 'Points': 0,
                    'xG': 0.0, 'xGA': 0.0, 'xGD': 0.0, 'xPts': 0.0
                }
        
        # Get scores
        # stats keys are 'team1' and 'team2' corresponding to team_names[0] and [1]
        s1 = match['stats']['team1']['goals']
        s2 = match['stats']['team2']['goals']
        
        xg1 = match['stats']['team1'].get('xg', 0)
        xg2 = match['stats']['team2'].get('xg', 0)
        prob_h, prob_d, prob_a = calculate_match_probabilities(xg1, xg2)
        pts1 = round((prob_h * 3) + (prob_d * 1), 2)
        pts2 = round((prob_a * 3) + (prob_d * 1), 2)

        standings[t1_name]['xG'] += xg1
        standings[t1_name]['xGA'] += xg2
        standings[t1_name]['xGD'] += (xg1 - xg2)
        standings[t1_name]['xPts'] += pts1
        
        standings[t2_name]['xG'] += xg2
        standings[t2_name]['xGA'] += xg1
        standings[t2_name]['xGD'] += (xg2 - xg1)
        standings[t2_name]['xPts'] += pts2

        # Update Stats for Team 1
        standings[t1_name]['Played'] += 1
        standings[t1_name]['GF'] += s1
        standings[t1_name]['GA'] += s2
        standings[t1_name]['GD'] += (s1 - s2)
        
        # Update Stats for Team 2
        standings[t2_name]['Played'] += 1
        standings[t2_name]['GF'] += s2
        standings[t2_name]['GA'] += s1
        standings[t2_name]['GD'] += (s2 - s1)
        
        # Points
        if s1 > s2:
            standings[t1_name]['Won'] += 1
            standings[t1_name]['Points'] += 3
            standings[t2_name]['Lost'] += 1
        elif s2 > s1:
            standings[t2_name]['Won'] += 1
            standings[t2_name]['Points'] += 3
            standings[t1_name]['Lost'] += 1
        else:
            standings[t1_name]['Drawn'] += 1
            standings[t1_name]['Points'] += 1
            standings[t2_name]['Drawn'] += 1
            standings[t2_name]['Points'] += 1

    # Convert to DataFrame
    df = pd.DataFrame(list(standings.values()))
    
    # Sort: Points (desc), GD (desc), GF (desc)
    if not df.empty:
        df = df.sort_values(by=['Points', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
        # Add Rank
        df.index += 1
        df.insert(0, 'Rank', df.index)
        
        # Reorder columns
        cols = ['Rank', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points']
        df = df[cols]
        
    return df

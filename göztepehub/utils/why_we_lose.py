import pandas as pd
import os

from utils.data import get_data_dir

GOZTEPE = 'Göztepe Spor Kulübü'

_cache = None


def _load_goztepe_match_dfs():
    data_dir = get_data_dir()
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
    match_dfs = []
    for filename in files:
        try:
            df = pd.read_parquet(os.path.join(data_dir, filename))
            if 'team_name' in df.columns and GOZTEPE in df['team_name'].values:
                df['_source'] = filename
                match_dfs.append(df)
        except Exception:
            continue
    return match_dfs


def _parse_minute(val):
    try:
        if isinstance(val, str) and '+' in val:
            base, extra = val.split('+', 1)
            return int(base) + int(extra)
        return int(float(val))
    except Exception:
        return None


def _minute_band(minute):
    if minute is None:
        return None
    if minute <= 30:
        return '1-30'
    if minute <= 60:
        return '31-60'
    if minute <= 90:
        return '61-90'
    return '90+'


def calc_why_we_lose():
    global _cache
    if _cache is not None:
        return _cache

    match_dfs = _load_goztepe_match_dfs()

    home_record = {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0}
    away_record = {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0}

    bands = ['1-30', '31-60', '61-90', '90+']
    conceded_bands = {b: 0 for b in bands}
    scored_bands = {b: 0 for b in bands}

    game_state_conceded = {'Leading': 0, 'Drawing': 0, 'Trailing': 0}

    after_scoring_first = {'played': 0, 'W': 0, 'D': 0, 'L': 0}
    after_conceding_first = {'played': 0, 'W': 0, 'D': 0, 'L': 0}

    for df in match_dfs:
        goz_rows = df[df['team_name'] == GOZTEPE]
        if goz_rows.empty:
            continue

        goz_position = (
            goz_rows['team_position'].iloc[0]
            if 'team_position' in df.columns else 'home'
        )

        has_og_col = 'own goal' in df.columns
        goals = df[df['type_id'] == 16].copy()
        if has_og_col:
            goals = goals[goals['own goal'] != 'Si']

        if 'time_min' in goals.columns:
            goals['_min'] = goals['time_min'].apply(_parse_minute)
            goals = goals.sort_values('_min', na_position='last')
        else:
            goals['_min'] = None

        goz_score = 0
        opp_score = 0
        first_goal_team = None

        for _, g in goals.iterrows():
            is_goz = (g['team_name'] == GOZTEPE)
            band = _minute_band(g.get('_min'))

            if is_goz:
                if band:
                    scored_bands[band] += 1
                if first_goal_team is None:
                    first_goal_team = 'goztepe'
                goz_score += 1
            else:
                if band:
                    conceded_bands[band] += 1
                if first_goal_team is None:
                    first_goal_team = 'opponent'

                diff = goz_score - opp_score
                if diff > 0:
                    game_state_conceded['Leading'] += 1
                elif diff == 0:
                    game_state_conceded['Drawing'] += 1
                else:
                    game_state_conceded['Trailing'] += 1

                opp_score += 1

        if goz_score > opp_score:
            result = 'W'
        elif goz_score == opp_score:
            result = 'D'
        else:
            result = 'L'

        rec = home_record if goz_position == 'home' else away_record
        rec[result] += 1
        rec['GF'] += goz_score
        rec['GA'] += opp_score

        if first_goal_team == 'goztepe':
            after_scoring_first['played'] += 1
            after_scoring_first[result] += 1
        elif first_goal_team == 'opponent':
            after_conceding_first['played'] += 1
            after_conceding_first[result] += 1

    _cache = {
        'home_record': home_record,
        'away_record': away_record,
        'conceded_bands': conceded_bands,
        'scored_bands': scored_bands,
        'game_state_conceded': game_state_conceded,
        'after_scoring_first': after_scoring_first,
        'after_conceding_first': after_conceding_first,
    }
    return _cache

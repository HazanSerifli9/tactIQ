"""
Wyscout Loader
==============
Reads Wyscout Excel exports and computes per-team season averages.

Up to five files per team:
  Team Stats {team}.xlsx        → base file
  Team Stats {team} (1).xlsx   → variant 1
  Team Stats {team} (2).xlsx   → variant 2
  Team Stats {team} (3).xlsx   → variant 3
  Team Stats {team} (4).xlsx   → variant 4

File numbers may map to different content across teams, so each file
is auto-categorised by its column signatures.

Usage:
    from utils.wyscout_loader import load_wyscout_team_averages
    df = load_wyscout_team_averages()
    # One row per team; columns are season averages
"""

import os
import pandas as pd
import numpy as np
from functools import lru_cache
from shared.logger import get_logger

logger = get_logger(__name__)

WYSCOUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_data", "Wyscout"
)

# Teams whose Wyscout files are present in WYSCOUT_DIR
WYSCOUT_TEAMS = [
    "Alanyaspor",
    "Antalyaspor",
    "Beşiktaş",
    "Eyüpspor",
    "Fatih Karagümrük",
    "Fenerbahçe",
    "Galatasaray",
    "Gaziantep",
    "Gençlerbirliği",
    "Göztepe",
    "Kasımpaşa",
    "Kayserispor",
    "Kocaelispor",
    "Konyaspor",
    "Rizespor",
    "Samsunspor",
    "Trabzonspor",
    "İstanbul Başakşehir",
]

# ── File-type detection signatures ──────────────────────────────────────────
# Presence of any signature column identifies the file type
FILE_TYPE_SIGNATURES = {
    "summary":   ["Goals", "Possession, %", "Losses / Low / Medium / High"],
    "tempo":     ["PPDA", "Match tempo", "Average passes per possession"],
    "passing":   ["Passes / accurate", "Forward passes / accurate", "Progressive passes / accurate"],
    "defensive": ["Conceded goals", "Shots against / on target", "Interceptions"],
    "attacking": ["xG", "Positional attacks / with shots", "Counterattacks / with shots"],
}

# ── Column renames ───────────────────────────────────────────────────────────
RENAME_ATTACKING = {
    "xG":                             "xg_for",
    "Shots / on target":              "shots_total",
    "Unnamed: 8":                     "shots_on_target",
    "Unnamed: 9":                     "shots_on_target_pct",
    "Positional attacks / with shots":"positional_attacks",
    "Unnamed: 11":                    "positional_with_shots",
    "Counterattacks / with shots":    "counterattacks",
    "Unnamed: 14":                    "counter_with_shots",
    "Corners / with shots":           "corners",
    "Unnamed: 17":                    "corners_with_shots",
    "Free kicks / with shots":        "free_kicks",
    "Unnamed: 20":                    "free_kicks_with_shots",
    "Crosses / accurate":             "crosses",
    "Unnamed: 26":                    "crosses_accurate",
    "Unnamed: 27":                    "crosses_accurate_pct",
    "Offensive duels / won":          "offensive_duels",
    "Unnamed: 29":                    "offensive_duels_won",
    "Unnamed: 30":                    "offensive_duels_won_pct",
    "Offsides":                       "offsides",
}

RENAME_DEFENSIVE = {
    "Conceded goals":                 "goals_conceded",
    "Shots against / on target":      "shots_against",
    "Unnamed: 8":                     "shots_against_on_target",
    "Unnamed: 9":                     "shots_against_on_target_pct",
    "Defensive duels / won":          "defensive_duels",
    "Unnamed: 11":                    "defensive_duels_won",
    "Unnamed: 12":                    "defensive_duels_won_pct",
    "Aerial duels / won":             "aerial_duels",
    "Unnamed: 14":                    "aerial_duels_won",
    "Unnamed: 15":                    "aerial_duels_won_pct",
    "Sliding tackles / successful":   "sliding_tackles",
    "Unnamed: 17":                    "sliding_tackles_succ",
    "Interceptions":                  "interceptions",
    "Clearances":                     "clearances",
    "Fouls":                          "fouls",
    "Yellow cards":                   "yellow_cards",
    "Red cards":                      "red_cards",
}

RENAME_PASSING = {
    "Passes / accurate":               "passes_total",
    "Unnamed: 7":                      "passes_accurate",
    "Unnamed: 8":                      "pass_accuracy_pct",
    "Forward passes / accurate":       "fwd_passes",
    "Unnamed: 10":                     "fwd_passes_accurate",
    "Unnamed: 11":                     "fwd_pass_accuracy_pct",
    "Back passes / accurate":          "back_passes",
    "Unnamed: 13":                     "back_passes_accurate",
    "Long passes / accurate":          "long_passes",
    "Unnamed: 19":                     "long_passes_accurate",
    "Unnamed: 20":                     "long_pass_accuracy_pct",
    "Passes to final third / accurate":"final_third_passes",
    "Unnamed: 22":                     "final_third_passes_accurate",
    "Progressive passes / accurate":   "progressive_passes",
    "Unnamed: 25":                     "progressive_passes_accurate",
    "Smart passes / accurate":         "smart_passes",
    "Unnamed: 28":                     "smart_passes_accurate",
    "Goal kicks":                      "goal_kicks",
}

RENAME_TEMPO = {
    "Match tempo":                    "match_tempo",
    "Average passes per possession":  "avg_passes_per_poss",
    "Long pass %":                    "long_pass_pct",
    "PPDA":                           "ppda",
    "Average shot distance":          "avg_shot_distance",
    "Average pass length":            "avg_pass_length",
}

RENAME_SUMMARY = {
    "Goals":                          "goals_scored",
    "xG":                             "xg_for",
    "Shots / on target":              "shots_total",
    "Unnamed: 9":                     "shots_on_target",
    "Unnamed: 10":                    "shots_on_target_pct",
    "Passes / accurate":              "passes_total",
    "Unnamed: 12":                    "passes_accurate",
    "Unnamed: 13":                    "pass_accuracy_pct",
    "Possession, %":                  "possession_pct",
    "Losses / Low / Medium / High":   "losses_total",
    "Recoveries / Low / Medium / High":"recoveries_total",
    "Duels / won":                    "duels_total",
    "Unnamed: 24":                    "duels_won",
    "Unnamed: 25":                    "duels_won_pct",
}

TYPE_RENAME_MAP = {
    "tempo":     RENAME_TEMPO,
    "passing":   RENAME_PASSING,
    "defensive": RENAME_DEFENSIVE,
    "attacking": RENAME_ATTACKING,
    "summary":   RENAME_SUMMARY,
}


def _detect_file_type(df: pd.DataFrame) -> str:
    """Identify file type from the column names present in the DataFrame."""
    cols = set(df.columns)
    for ftype, signatures in FILE_TYPE_SIGNATURES.items():
        if any(sig in cols for sig in signatures):
            return ftype
    return "unknown"


def _read_and_clean(fpath: str, team_name: str) -> tuple:
    """Read one Excel file, detect its type, and filter to Süper Lig rows for the given team."""
    try:
        df = pd.read_excel(fpath, sheet_name="TeamStats")
    except Exception as e:
        logger.warning("Cannot read %s: %s", fpath, e)
        return "unknown", pd.DataFrame()

    ftype = _detect_file_type(df)

    # First two rows are summary metadata; keep only per-match rows
    df = df[df["Match"].notna()].copy()

    # Süper Lig only
    if "Competition" in df.columns:
        df = df[df["Competition"].str.contains("Süper Lig", na=False)].copy()

    if df.empty:
        return ftype, pd.DataFrame()

    # Keep only rows for this team
    if "Team" in df.columns:
        df = df[df["Team"].str.strip() == team_name].copy()

    return ftype, df


def _load_team_raw(team_name: str) -> pd.DataFrame:
    """Load all Wyscout files for one team, categorise by content type, and merge into a single DataFrame."""
    suffixes = ["", " (1)", " (2)", " (3)", " (4)"]
    type_dfs: dict[str, pd.DataFrame] = {}

    for sfx in suffixes:
        fname = f"Team Stats {team_name}{sfx}.xlsx"
        fpath = os.path.join(WYSCOUT_DIR, fname)
        if not os.path.exists(fpath):
            continue
        ftype, df = _read_and_clean(fpath, team_name)
        if df.empty or ftype == "unknown":
            continue
        # Use the first file found for each type
        if ftype not in type_dfs:
            rename_map = TYPE_RENAME_MAP.get(ftype, {})
            type_dfs[ftype] = df.rename(columns=rename_map)
            logger.debug("Team %s: loaded %s from %s", team_name, ftype, fname)

    if not type_dfs:
        return pd.DataFrame()

    # Prefer attacking, then summary, as the base DataFrame
    if "attacking" in type_dfs:
        merged = type_dfs["attacking"].copy()
    elif "summary" in type_dfs:
        merged = type_dfs["summary"].copy()
    else:
        merged = next(iter(type_dfs.values())).copy()

    possible_keys = ["Match", "Team"]

    for ftype, df_extra in type_dfs.items():
        if df_extra is merged:
            continue
        new_cols = [c for c in df_extra.columns if c not in merged.columns]
        if not new_cols:
            continue
        # Only join on keys that exist in both DataFrames
        key_cols = [c for c in possible_keys if c in df_extra.columns and c in merged.columns]
        if not key_cols:
            continue
        extra_sub = df_extra[key_cols + new_cols]
        merged = merged.merge(extra_sub, on=key_cols, how="left")

    merged["wyscout_team"] = team_name
    return merged


@lru_cache(maxsize=1)
def load_wyscout_team_averages() -> pd.DataFrame:
    """
    Load all Wyscout team files and compute per-team season averages.

    Returns:
        pd.DataFrame — one row per team; columns are season averages:
          wyscout_team, n_matches,
          xg_for, shots_total, shots_on_target_pct,
          goals_conceded, shots_against, defensive_duels_won_pct,
          aerial_duels_won_pct, interceptions, clearances,
          pass_accuracy_pct, fwd_pass_accuracy_pct, long_pass_accuracy_pct,
          final_third_passes, progressive_passes, smart_passes,
          match_tempo, avg_passes_per_poss, long_pass_pct,
          ppda, avg_shot_distance, avg_pass_length
    """
    if not os.path.isdir(WYSCOUT_DIR):
        logger.error("Wyscout directory not found: %s", WYSCOUT_DIR)
        return pd.DataFrame()

    rows = []
    for team_name in WYSCOUT_TEAMS:
        raw = _load_team_raw(team_name)
        if raw.empty:
            logger.warning("No data for team: %s", team_name)
            continue

        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        avg = raw[numeric_cols].mean(numeric_only=True)
        row = {"wyscout_team": team_name, "n_matches": len(raw)}
        row.update(avg.to_dict())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Wyscout team averages loaded: %d teams", len(df))
    return df


def get_wyscout_match_stats(home_team: str, away_team: str) -> dict:
    """
    Return Wyscout match-level tactical indicators (xG, PPDA, tempo, passes, clearances, etc.)
    for the match between the two teams.
    The Match column uses the format "TeamA - TeamB score"; both team names are searched.

    Returns:
        dict: containing 'home' and 'away' dictionaries with various Wyscout metrics.
    """
    name_map = get_wyscout_team_name_map()
    h_ws = name_map.get(home_team, home_team)
    a_ws = name_map.get(away_team, away_team)

    metrics_list = [
        # attacking / offensive
        'xg_for', 'shots_total', 'shots_on_target',
        'positional_attacks', 'positional_with_shots',
        'crosses', 'crosses_accurate',
        # transitions
        'counterattacks', 'counter_with_shots',
        # set pieces
        'corners', 'corners_with_shots',
        'free_kicks', 'free_kicks_with_shots',
        # defensive
        'ppda', 'interceptions', 'clearances',
        'defensive_duels_won_pct', 'aerial_duels_won_pct', 'fouls',
        # possession / tempo
        'pass_accuracy_pct', 'match_tempo', 'avg_passes_per_poss',
        'progressive_passes', 'final_third_passes', 'possession_pct',
    ]

    result = {
        'home': {m: None for m in metrics_list},
        'away': {m: None for m in metrics_list},
    }

    def _find_match_row(team_ws: str, other_ws: str, col: str) -> float | None:
        raw = _load_team_raw(team_ws)
        if raw.empty or col not in raw.columns or 'Match' not in raw.columns:
            return None
        team_col = 'Team' if 'Team' in raw.columns else 'wyscout_team'
        mask = (
            raw['Match'].str.contains(h_ws, case=False, na=False) &
            raw['Match'].str.contains(a_ws, case=False, na=False) &
            (raw[team_col] == team_ws)
        )
        rows = raw[mask]
        if rows.empty:
            return None
        val = rows.iloc[0][col]
        return round(float(val), 2) if pd.notna(val) else None

    for m in metrics_list:
        result['home'][m] = _find_match_row(h_ws, a_ws, m)
        result['away'][m] = _find_match_row(a_ws, h_ws, m)

    # Maintain backward compatibility
    result['home']['xg'] = result['home']['xg_for']
    result['away']['xg'] = result['away']['xg_for']

    logger.info(
        "Wyscout match stats retrieved — %s: xG=%s PPDA=%s | %s: xG=%s PPDA=%s",
        h_ws, result['home']['xg_for'], result['home']['ppda'],
        a_ws, result['away']['xg_for'], result['away']['ppda'],
    )
    return result


def get_wyscout_team_name_map() -> dict:
    """Map event-data team names to Wyscout team names."""
    return {
        # Wyscout short names
        "Alanyaspor":                          "Alanyaspor",
        "Antalyaspor":                         "Antalyaspor",
        "Antalyaspor ":                        "Antalyaspor",
        "Beşiktaş":                            "Beşiktaş",
        "Eyüpspor":                            "Eyüpspor",
        "Eyup":                                "Eyüpspor",
        "Fatih Karagümrük":                    "Fatih Karagümrük",
        "Fenerbahçe":                          "Fenerbahçe",
        "Galatasaray":                         "Galatasaray",
        "Gaziantep":                           "Gaziantep",
        "Gaziantep FK":                        "Gaziantep",
        "Gençlerbirliği":                      "Gençlerbirliği",
        "Göztepe":                             "Göztepe",
        "Kasımpaşa":                           "Kasımpaşa",
        "Kayserispor":                         "Kayserispor",
        "Kayseri":                             "Kayserispor",
        "Kocaelispor":                         "Kocaelispor",
        "Kocaeli":                             "Kocaelispor",
        "Konyaspor":                           "Konyaspor",
        "Konya":                               "Konyaspor",
        "Rizespor":                            "Rizespor",
        "Rize":                                "Rizespor",
        "Samsunspor":                          "Samsunspor",
        "Samsun":                              "Samsunspor",
        "Trabzonspor":                         "Trabzonspor",
        "İstanbul Başakşehir":                 "İstanbul Başakşehir",
        "Başakşehir":                          "İstanbul Başakşehir",
        "Istanbul Basaksehir":                 "İstanbul Başakşehir",
        # Opta / event-data full names
        "Alanyaspor Kulübü":                   "Alanyaspor",
        "Antalyaspor Kulübü":                  "Antalyaspor",
        "Beşiktaş Jimnastik Kulübü":           "Beşiktaş",
        "Eyüp Spor Kulübü":                    "Eyüpspor",
        "Fatih Karagümrük Spor Kulübü":        "Fatih Karagümrük",
        "Fenerbahçe Spor Kulübü":              "Fenerbahçe",
        "Galatasaray Spor Kulübü":             "Galatasaray",
        "Gaziantep Futbol Kulübü":             "Gaziantep",
        "Gençlerbirliği Spor Kulübü":          "Gençlerbirliği",
        "Göztepe Spor Kulübü":                 "Göztepe",
        "Kasımpaşa Spor Kulübü":               "Kasımpaşa",
        "Kayseri Spor Kulübü":                 "Kayserispor",
        "Kocaelispor Kulübü":                  "Kocaelispor",
        "Konyaspor Kulübü":                    "Konyaspor",
        "Samsunspor Kulübü":                   "Samsunspor",
        "Trabzonspor Kulübü":                  "Trabzonspor",
        "Çaykur Rize Spor Kulübü":             "Rizespor",
        "İstanbul Başakşehir Futbol Kulübü":   "İstanbul Başakşehir",
    }

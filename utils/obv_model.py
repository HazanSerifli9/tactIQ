"""
On-Ball Value (OBV) Engine
===========================
Calculates the contribution of each on-ball event to a team's probability
of scoring or conceding, inspired by the Hudl StatsBomb OBV model.

Core Methodology:
  - Train two independent XGBoost models on the project's own event data:
      OBV_For    → P(team generates an xG shot later in this chain)
      OBV_Against → P(next possession generates xG against this team)
  - For every event: OBV_Net = ΔOBV_For - ΔOBV_Against
  - Features are purely spatial + contextual (no sequence history)
    so team strength does NOT inflate/deflate values.

Usage:
  1. Train offline:  python -m utils.obv_model
  2. At runtime:     from utils.obv_model import calculate_obv
                     df = calculate_obv(df)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Tuple, List, Dict, Any

from utils.possession_engine import extract_all_possession_chains, PossessionChain

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Events that carry an absolute OBV but need careful directional tracking
PASS_LABEL   = 'Pass'
CARRY_LABEL  = 'Carry'   # inferred carry; also used when no explicit carry event
SHOT_LABELS  = {'Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot'}
DEF_LABELS   = {'Tackle', 'Interception', 'Challenge', 'Ball Recovery', 'Clearance', 'Foul'}
DRIBBLE_LABELS = {'Take-on', 'Dribble'}

OBV_FOR_MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'tactiq_obv_for_model.json')
OBV_AGAINST_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tactiq_obv_against_model.json')

# -----------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------

def _extract_features(x, y, event: str, outcome: int = 1,
                       end_x: float = None, end_y: float = None) -> Dict[str, float]:
    """
    Extract OBV feature vector for a single pitch state.
    Coordinates: Opta 0-100 system.
    """
    # Distance & angle to goal (attacking end at x=100, y=50)
    gx, gy = 100.0, 50.0
    dist_to_goal = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)

    # Angle to goal (in degrees)
    dx, dy = gx - x, gy - y
    angle_to_goal = np.degrees(np.arctan2(abs(dy), max(dx, 0.1)))

    # Pitch zones (x-based thirds)
    def_third   = 1.0 if x < 33.3 else 0.0
    mid_third   = 1.0 if 33.3 <= x < 66.6 else 0.0
    att_third   = 1.0 if x >= 66.6 else 0.0

    # Flanks (y-based)
    left_flank  = 1.0 if y < 33.3 else 0.0
    right_flank = 1.0 if y > 66.6 else 0.0
    central     = 1.0 if 33.3 <= y <= 66.6 else 0.0

    # Event type binary flags
    is_pass    = 1.0 if event == PASS_LABEL else 0.0
    is_carry   = 1.0 if event == CARRY_LABEL else 0.0
    is_shot    = 1.0 if event in SHOT_LABELS else 0.0
    is_def     = 1.0 if event in DEF_LABELS else 0.0
    is_dribble = 1.0 if event in DRIBBLE_LABELS else 0.0
    is_success = float(outcome == 1)

    # Progression delta (if end coords available)
    prog_x = (end_x - x) if end_x is not None else 0.0
    prog_y = abs(end_y - y) if end_y is not None else 0.0
    has_end_coords = 1.0 if end_x is not None else 0.0

    return {
        'x': x,
        'y': y,
        'dist_to_goal': dist_to_goal,
        'angle_to_goal': angle_to_goal,
        'def_third': def_third,
        'mid_third': mid_third,
        'att_third': att_third,
        'left_flank': left_flank,
        'right_flank': right_flank,
        'central': central,
        'is_pass': is_pass,
        'is_carry': is_carry,
        'is_shot': is_shot,
        'is_def': is_def,
        'is_dribble': is_dribble,
        'is_success': is_success,
        'prog_x': prog_x,
        'prog_y': prog_y,
        'has_end_coords': has_end_coords,
    }

FEATURE_COLS = [
    'x', 'y', 'dist_to_goal', 'angle_to_goal',
    'def_third', 'mid_third', 'att_third',
    'left_flank', 'right_flank', 'central',
    'is_pass', 'is_carry', 'is_shot', 'is_def', 'is_dribble',
    'is_success', 'prog_x', 'prog_y', 'has_end_coords',
]

# -----------------------------------------------------------------------
# Dataset Builder
# -----------------------------------------------------------------------

XG_SHOT_THRESHOLD = 0.05  # minimum xG to count as a "meaningful" shot chain

def _chain_had_xg_shot(chain: PossessionChain) -> float:
    """
    Returns 1.0 if this possession chain contained a shot with xG >= threshold,
    else 0.0.  Used as a binary classification target for OBV models.
    """
    for ev in chain.events:
        # Ended with shot event
        if ev.get('event', '') in SHOT_LABELS:
            xg = ev.get('xG', 0)
            try:
                xg_val = float(xg) if xg is not None and not pd.isna(xg) else 0.0
            except (TypeError, ValueError):
                xg_val = 0.0
            if xg_val >= XG_SHOT_THRESHOLD or ev.get('event') == 'Goal':
                return 1.0
    # No qualifying shot found — but chain ended_with_shot flag is also reliable
    if chain.ended_with_shot:
        return 1.0
    return 0.0


def build_obv_datasets(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build training datasets for OBV_For and OBV_Against.

    For each event in a possession chain:
      - obv_for_target  = xG generated by THIS team in THIS chain
      - obv_against_target = xG allowed by OPPONENT in the NEXT chain

    Returns (df_for, df_against) — each with FEATURE_COLS + target column.
    """
    print(f"Scanning data directory: {data_dir}")
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} match files.")

    rows_for: List[dict] = []
    rows_against: List[dict] = []

    for i, fp in enumerate(parquet_files):
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue

        if df.empty or 'event' not in df.columns:
            continue

        # Apply stored xG values if column exists (model already run on data)
        try:
            chains_by_team = extract_all_possession_chains(df)
        except Exception:
            continue

        teams = list(chains_by_team.keys())
        if len(teams) < 2:
            continue

        for team_idx, team in enumerate(teams):
            opp = teams[1 - team_idx]  # the other team
            my_chains  = chains_by_team[team]
            opp_chains = chains_by_team.get(opp, [])

            for ci, chain in enumerate(my_chains):
                # Binary: did THIS chain produce a qualified shot?
                chain_xg_for = _chain_had_xg_shot(chain)

                # OBV_Against target = did the *immediately following* opponent chain produce a shot?
                opp_next_xg = 0.0
                if opp_chains:
                    # Find next opp chain starting after this chain ends
                    for oc in opp_chains:
                        if oc.period_id == chain.period_id and oc.start_time >= chain.end_time:
                            opp_next_xg = _chain_had_xg_shot(oc)
                            break

                # Create one row per event in this chain
                for ev in chain.events:
                    event_name = ev.get('event', '')
                    x = float(ev.get('x', 50)) if not pd.isna(ev.get('x', 50)) else 50.0
                    y = float(ev.get('y', 50)) if not pd.isna(ev.get('y', 50)) else 50.0
                    outcome = int(ev.get('outcome', 1)) if not pd.isna(ev.get('outcome', 1)) else 1
                    end_x_raw = ev.get('Pass End X')
                    end_y_raw = ev.get('Pass End Y')
                    try:
                        end_x = float(end_x_raw) if end_x_raw is not None and not pd.isna(end_x_raw) else None
                        end_y = float(end_y_raw) if end_y_raw is not None and not pd.isna(end_y_raw) else None
                    except (TypeError, ValueError):
                        end_x, end_y = None, None

                    feats = _extract_features(x, y, event_name, outcome, end_x, end_y)
                    feats['target'] = chain_xg_for
                    rows_for.append(feats)

                    feats_a = _extract_features(x, y, event_name, outcome, end_x, end_y)
                    feats_a['target'] = opp_next_xg
                    rows_against.append(feats_a)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(parquet_files)} files...")

    df_for     = pd.DataFrame(rows_for)
    df_against = pd.DataFrame(rows_against)

    print(f"Dataset built. For: {len(df_for)} rows, Against: {len(df_against)} rows.")
    return df_for, df_against


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def _train_single(df: pd.DataFrame, label: str) -> xgb.XGBClassifier:
    """
    Train one XGBoost binary classifier for OBV.
    Target: 1 = possession chain produced a qualifying shot; 0 = did not.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df['target'].fillna(0).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    pos_weight = max(1, (y == 0).sum() / max((y == 1).sum(), 1))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    proba = model.predict_proba(X_te)[:, 1]
    try:
        auc   = roc_auc_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)
        pos_rate = y_te.mean()
        print(f"  [{label}] ROC-AUC: {auc:.4f} | Brier: {brier:.5f} | Positive rate: {pos_rate:.3f}")
    except Exception as e:
        print(f"  [{label}] Metrics unavailable: {e}")
    return model


def train_obv_models(data_dir: str = None) -> None:
    """
    Full training pipeline. Run once; saves models as JSON files.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_data')

    df_for, df_against = build_obv_datasets(data_dir)


    if df_for.empty:
        print("ERROR: No data extracted. Cannot train OBV models.")
        return

    print("\nTraining OBV_For model...")
    model_for = _train_single(df_for, 'OBV_For')
    model_for.save_model(OBV_FOR_MODEL_PATH)

    print("\nTraining OBV_Against model...")
    model_against = _train_single(df_against, 'OBV_Against')
    model_against.save_model(OBV_AGAINST_MODEL_PATH)

    print(f"\nModels saved:\n  {OBV_FOR_MODEL_PATH}\n  {OBV_AGAINST_MODEL_PATH}")


# -----------------------------------------------------------------------
# Inference — calculate OBV for a match dataframe
# -----------------------------------------------------------------------

_cached_for: xgb.XGBClassifier     = None
_cached_against: xgb.XGBClassifier = None


def _load_models():
    global _cached_for, _cached_against
    if _cached_for is None and os.path.exists(OBV_FOR_MODEL_PATH):
        _cached_for = xgb.XGBClassifier()
        _cached_for.load_model(OBV_FOR_MODEL_PATH)
    if _cached_against is None and os.path.exists(OBV_AGAINST_MODEL_PATH):
        _cached_against = xgb.XGBClassifier()
        _cached_against.load_model(OBV_AGAINST_MODEL_PATH)
    return _cached_for, _cached_against


def _models_available() -> bool:
    return os.path.exists(OBV_FOR_MODEL_PATH) and os.path.exists(OBV_AGAINST_MODEL_PATH)


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate a match DataFrame with OBV columns:
      obv_for       → model score for attacking contribution
      obv_against   → model score for defensive exposure
      obv_net       → obv_for delta − obv_against delta per event

    If models are not yet trained, returns df unchanged with zero columns.
    """
    df = df.copy()
    df['obv_for']     = 0.0
    df['obv_against'] = 0.0
    df['obv_net']     = 0.0

    if not _models_available():
        return df

    model_for, model_against = _load_models()
    if model_for is None or model_against is None:
        return df

    # Build feature matrix for every row
    feat_rows = []
    for _, row in df.iterrows():
        event_name = row.get('event', '')
        x = float(row.get('x', 50)) if not pd.isna(row.get('x', 50)) else 50.0
        y = float(row.get('y', 50)) if not pd.isna(row.get('y', 50)) else 50.0
        outcome = int(row.get('outcome', 1)) if not pd.isna(row.get('outcome', 1)) else 1
        end_x_raw = row.get('Pass End X')
        end_y_raw = row.get('Pass End Y')
        try:
            end_x = float(end_x_raw) if end_x_raw is not None and not pd.isna(end_x_raw) else None
            end_y = float(end_y_raw) if end_y_raw is not None and not pd.isna(end_y_raw) else None
        except (TypeError, ValueError):
            end_x, end_y = None, None

        feat_rows.append(_extract_features(x, y, event_name, outcome, end_x, end_y))

    feat_df = pd.DataFrame(feat_rows, columns=FEATURE_COLS).fillna(0)

    # Predict probabilities (P = likelihood of chain generating a shot)
    raw_for     = model_for.predict_proba(feat_df)[:, 1]
    raw_against = model_against.predict_proba(feat_df)[:, 1]

    # OBV_Net: how much does this event shift the team's net xG probability?
    #   +ve = event improves team's scoring chances and/or reduces opponent's
    #   -ve = event reduces team's chance (turnover in dangerous area, etc.)
    df['obv_for']     = np.round(raw_for, 4)
    df['obv_against'] = np.round(raw_against, 4)
    df['obv_net']     = np.round(raw_for - raw_against, 4)

    return df


# -----------------------------------------------------------------------
# Player Summary Stats
# -----------------------------------------------------------------------

def get_player_obv_summary(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Returns a per-player OBV summary for a team from an annotated match df.

    Columns: Player, OBV_Net, OBV_Pass, OBV_Carry, OBV_Def, Events
    """
    if 'obv_net' not in df.columns:
        df = calculate_obv(df)

    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        return pd.DataFrame()

    team_df['is_pass'   ] = team_df['event'] == PASS_LABEL
    team_df['is_carry'  ] = team_df['event'] == CARRY_LABEL
    team_df['is_def'    ] = team_df['event'].isin(DEF_LABELS)

    group = team_df.groupby('player_name')

    summary = pd.DataFrame({
        'OBV_Net'  : group['obv_net'].sum(),
        'OBV_Pass' : team_df[team_df['is_pass' ]].groupby('player_name')['obv_net'].sum(),
        'OBV_Carry': team_df[team_df['is_carry' ]].groupby('player_name')['obv_net'].sum(),
        'OBV_Def'  : team_df[team_df['is_def'  ]].groupby('player_name')['obv_net'].sum(),
        'Events'   : group['obv_net'].count(),
    }).reset_index()

    summary.rename(columns={'player_name': 'Player'}, inplace=True)
    summary = summary.fillna(0).sort_values('OBV_Net', ascending=False)
    summary['OBV_Net']   = summary['OBV_Net'].round(3)
    summary['OBV_Pass']  = summary['OBV_Pass'].round(3)
    summary['OBV_Carry'] = summary['OBV_Carry'].round(3)
    summary['OBV_Def']   = summary['OBV_Def'].round(3)

    return summary.reset_index(drop=True)


# -----------------------------------------------------------------------
# Entry point — train models
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'raw_data'
    )
    train_obv_models(data_dir)

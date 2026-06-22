import pandas as pd
import numpy as np
import os
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

def extract_xg_features(df):
    """
    Given a dataframe of shot events, extracts spatial and contextual features.
    Returns the dataframe with new feature columns.
    """
    df = df.copy()
    
    if 'event' not in df.columns:
        return df
        
    # Spatial Features (Convert 100x100 to 105x68 meters)
    if 'x_meters' not in df.columns:
        df['x_meters'] = df['x'] * 1.05
        df['y_meters'] = df['y'] * 0.68
    
    goal_x_meters = 100 * 1.05
    goal_y_meters = 50 * 0.68
    
    df['distance_to_goal'] = np.sqrt((goal_x_meters - df['x_meters'])**2 + (goal_y_meters - df['y_meters'])**2)
    
    post1_y = goal_y_meters - (7.32 / 2)
    post2_y = goal_y_meters + (7.32 / 2)
    
    v1_x = goal_x_meters - df['x_meters']
    v1_y = post1_y - df['y_meters']
    
    v2_x = goal_x_meters - df['x_meters']
    v2_y = post2_y - df['y_meters']
    
    dot = v1_x*v2_x + v1_y*v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)
    
    cos_angle = dot / (mag1 * mag2 + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    df['shot_angle'] = np.degrees(np.arccos(cos_angle))
    
    # Categorical Features
    def parse_opta_bool(val):
        if pd.isna(val) or val == 0 or val == '0' or val == False or str(val).lower() == 'no':
            return 0
        return 1

    df['is_header'] = df.get('Head', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_right_foot'] = df.get('Right footed', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_left_foot'] = df.get('Left footed', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    
    df['is_corner'] = df.get('From corner', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_freekick'] = df.get('Free kick', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_fastbreak'] = df.get('Fast break', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_setpiece'] = df.get('Set piece', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_penalty'] = df.get('Penalty', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    
    df['is_big_chance'] = df.get('Big Chance', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_assisted'] = df.get('Assisted', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_cross'] = df.get('Cross', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    df['is_through_ball'] = df.get('Through ball', pd.Series(0, index=df.index)).apply(parse_opta_bool)
    
    return df

def build_xg_dataset(data_dir):
    """Iterates through raw parquets and builds a master DataFrame of all shots."""
    print("Building master shot dataset from all parquets...")
    all_shots = []
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            if 'event' in df.columns:
                shots = df[df['event'].isin(['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot'])].copy()
                if not shots.empty:
                    if 'own goal' in shots.columns:
                        shots = shots[(shots['own goal'] != 'Si') & (shots['own goal'] != '1')]
                        
                    shots['is_goal'] = np.where(shots['event'] == 'Goal', 1, 0)
                    all_shots.append(shots)
        except Exception:
            continue
            
    if not all_shots:
         return pd.DataFrame()
         
    master_df = pd.concat(all_shots, ignore_index=True)
    master_df = extract_xg_features(master_df)
    
    master_df = master_df[master_df['x'] > 0]
    return master_df

def get_model_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "tactiq_xg_model.json")

def train_xg_model(data_dir):
    df = build_xg_dataset(data_dir)
    if df.empty:
        print("No shot data available to train the model.")
        return
        
    print(f"Dataset compiled. Total Shots: {len(df)}")
    
    features = [
        'distance_to_goal', 'shot_angle', 'is_header', 'is_right_foot', 'is_left_foot',
        'is_corner', 'is_freekick', 'is_fastbreak', 'is_setpiece', 'is_penalty',
        'is_big_chance', 'is_assisted', 'is_cross', 'is_through_ball'
    ]
    
    for f in features:
        if f not in df.columns:
            df[f] = 0
            
    X = df[features]
    y = df['is_goal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier...")
    (len(y) - sum(y)) / max(sum(y), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    
    try:
        auc = roc_auc_score(y_test, preds_proba)
        brier = brier_score_loss(y_test, preds_proba)
        
        print("Model Training Complete!")
        print(f"ROC-AUC: {auc:.4f} (Higher is better, >0.75 is good for xG)")
        print(f"Brier Score: {brier:.4f} (Lower is better)")
    except Exception as e:
        print(f"Could not compute metrics: {e}")
    
    model_path = get_model_path()
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

def predict_xg(df):
    """
    Applies the trained xG model to an incoming match dataframe.
    Returns the dataframe with an 'xG' column.
    """
    if df is None or df.empty or 'event' not in df.columns:
        return df
        
    df = extract_xg_features(df)
    
    model_path = get_model_path()
    
    if 'xG' not in df.columns:
        df['xG'] = 0.0
    elif pd.api.types.is_numeric_dtype(df['xG']) and df['xG'].sum() > 0:
        # If xG already extensively exists in this df (perhaps pre-calculated), optionally skip or overwrite
        # We will overwrite to ensure consistency.
        df['xG'] = 0.0
        
    if not os.path.exists(model_path):
        return df
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = [
        'distance_to_goal', 'shot_angle', 'is_header', 'is_right_foot', 'is_left_foot',
        'is_corner', 'is_freekick', 'is_fastbreak', 'is_setpiece', 'is_penalty',
        'is_big_chance', 'is_assisted', 'is_cross', 'is_through_ball'
    ]
    
    shot_mask = df['event'].isin(['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot'])
    
    if not shot_mask.any():
        return df
        
    for f in features:
        if f not in df.columns:
            df[f] = 0
            
    X_pred = df.loc[shot_mask, features]
    
    # Needs to be a dataframe of same shape
    xg_values = model.predict_proba(X_pred)[:, 1]
    
    # StatsBomb static PK value
    if 'is_penalty' in X_pred.columns:
        pk_mask = (X_pred['is_penalty'] == 1).values
        xg_values[pk_mask] = 0.78
        
    df.loc[shot_mask, 'xG'] = np.round(xg_values, 3)
    
    return df

if __name__ == "__main__":
    # If run directly as a script, train the model
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data")
    train_xg_model(data_dir)

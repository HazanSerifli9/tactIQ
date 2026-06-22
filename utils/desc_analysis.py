
import pandas as pd
import numpy as np

# ============================================================
# COACH'S QUESTIONS (DESCRIPTIVE ANALYSIS)
# ============================================================

def analyze_goal_kicks(df, team_name):
    """
    Analyzes Goal Kicks to answer: "How do they use goal kicks? Target player?"
    Returns a dict with distribution (Short/Long) and Top Targets.
    """
    df_team = df[df['team_name'] == team_name].copy()
    
    # Identify Goal Kicks
    # Strategy: 'Pass' with qualifier 'GoalKick' (True) or Event 'Goal Keeper' with sub-type?
    # Based on previous analysis, we look for 'Goal Kick taken' qualifier or similar.
    
    gk_cols = [c for c in df_team.columns if 'goal' in c.lower() and 'kick' in c.lower() and ('taken' in c.lower() or 'type' in c.lower())]
    if not gk_cols:
         # Fallback: Pass from very explicit GK coordinates (box)
         # 120x80: Box area x < 6, y 36-44? (6 yard box)
         # Let's assume user has standard Opta/SB data where Goal Kick checks worked before.
         # For now, let's try to find "Pass" events that are Goal Kicks.
         
         # Logic from detection: 'Goal kick taken'
         # If unknown, we can't do much. But let's assume standard event filter.
         # Or check for 'Pass' followed by recovery/aerial duel of opponent/teammate?
         
         # Let's assume 'subEventName' == 'Goal Kick' or similar if available.
         # Or just use the spatial logic: Pass from 6-yard box.
         pass
         
    # Assuming we found them using a broad filter for now or column check
    # Let's use a coordinate based heuristic + 'Pass' event if no column
    # Standard 6-yard box is x <= 6 (on 120 scale)
    
    # Using 'x_scaled' if available from preprocessing, else x 
    # (Assuming x is 0-100 or 0-120). 
    # If 0-100, 6 yards is ~5.
    
    mask_gk = (df_team['event'] == 'Pass') & (df_team['x'] < 6) & (df_team['y'] > 30) & (df_team['y'] < 70)
    # Refine: often specific event "Goal Kick" exists in some providers.
    if 'event' in df_team.columns:
         mask_gk = mask_gk | (df_team['event'].astype(str).str.contains('Goal Kick', case=False))

    gk_events = df_team[mask_gk].copy()
    
    if gk_events.empty:
        return {"count": 0, "long_pct": 0, "targets": {}}
        
    # Analyze Length
    # Short < 40m? 
    # Parse distance if available or calc
    if 'pass_length' in gk_events.columns:
         gk_events['length'] = gk_events['pass_length']
    elif 'x' in gk_events.columns and ('end_x' in gk_events.columns or 'Pass End X' in gk_events.columns):
         end_x = gk_events.get('end_x', gk_events.get('Pass End X', 0))
         end_y = gk_events.get('end_y', gk_events.get('Pass End Y', 0))
         gk_events['length'] = np.sqrt((end_x - gk_events['x'])**2 + (end_y - gk_events['y'])**2)
    else:
         gk_events['length'] = 0
         
    long_gks = gk_events[gk_events['length'] > 40]
    gk_events[gk_events['length'] <= 40]
    
    long_pct = round((len(long_gks) / len(gk_events)) * 100, 1)
    
    # Identify Target Players (for Long Kicks)
    # The receiver is often not in the event itself unless linked.
    # But usually 'receiver' column exists or we look at next event's player.
    
    targets = {}
    if 'receiver' in gk_events.columns: # If pre-processed
         targets = gk_events['receiver'].value_counts().head(3).to_dict()
    else:
         # Try to find next event
         # Indices
         targets_list = []
         for idx in long_gks.index:
             if idx + 1 in df_team.index:
                  next_ev = df_team.loc[idx+1]
                  # If next event is same team -> receiver?
                  # Or aerial duel?
                  if next_ev['team_name'] == team_name:
                       targets_list.append(next_ev['player_name'])
         
         if targets_list:
              targets = pd.Series(targets_list).value_counts().head(3).to_dict()
              
    return {
        "count": len(gk_events),
        "long_pct": long_pct,
        "targets": targets,
        "events": gk_events # for plotting
    }

def analyze_final_third_entries(df, team_name):
    """
    Analyzes how they enter the final third.
    Returns: Count of Carries vs Passes, and Zone breakdown (Left, Center, Right).
    """
    df_team = df[df['team_name'] == team_name].copy()
    
    # Define Final Third Entry
    # Move from middle third (x < 66) to final third (x >= 66) (0-100 scale)
    # Check 'x' and 'end_x' (for passes) or next event x (for carries)
    
    # Passes into final third
    # Start < 66, End >= 66
    passes_into_3rd = df_team[
        (df_team['event'] == 'Pass') & 
        (df_team['outcome'].isin([1, 'Successful', 'True'])) &
        (df_team['x'] < 66) & 
        ((df_team.get('Pass End X', 0) >= 66))
    ]
    
    # Carries into final third
    # We need 'Carry' events or infer them (movement > 5m without pass/event change)
    # Assuming 'Carry' events exist or we can't easily do it without trajectory logic.
    # Let's check for 'Carry' event.
    carries_into_3rd = pd.DataFrame()
    if 'event' in df_team.columns:
         # 'Carry' or 'Dribble' (though Dribble usually means Take-on)
         # Using 'Carry' if available (SB data has it, Opta usually implicit in coordinates between events)
         # If explicit Carry:
         carries = df_team[df_team['event'] == 'Carry']
         if not carries.empty:
              carries_into_3rd = carries[
                  (carries['x'] < 66) & (carries['end_x'] >= 66) # Assuming end_x for carry
              ]
    
    # Stats
    pass_count = len(passes_into_3rd)
    carry_count = len(carries_into_3rd)
    
    # Zones (of Entry point)
    # y 0-100: Left (>66?), Right (<33?), Center (33-66)
    # Remember Opta Y: 0/100 are touchlines. Usually 0 is Right? 100 Left? Or vice versa.
    # Let's assume standard split.
    
    zones = {"Left": 0, "Center": 0, "Right": 0}
    
    # Combine for zoning
    entries = pd.concat([passes_into_3rd, carries_into_3rd])
    
    if not entries.empty:
        # Check Entry Y (End Y for pass, End Y for carry)
        # For pass: Pass End Y
        pass_y = passes_into_3rd['Pass End Y'] if not passes_into_3rd.empty else []
        carry_y = carries_into_3rd['end_y'] if not carries_into_3rd.empty else [] # Assuming end_y
        
        all_y = list(pass_y) + list(carry_y)
        
        for y in all_y:
            if pd.isna(y): continue
            if y < 33.3:
                zones["Right"] += 1
            elif y > 66.6:
                zones["Left"] += 1
            else:
                zones["Center"] += 1
                
    return {
        "pass_count": pass_count,
        "carry_count": carry_count,
        "zones": zones
    }

import pandas as pd
import sys
import os

# Add the project directory to sys.path so we can import packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_match_dataframe
import utils.visuals as visuals
import utils.tempo_data as tempo_data
import utils.tempo_visuals as tempo_visuals
import utils.obv_model as obv_model
import utils.obv_visuals as obv_visuals

def test_plots():
    filename = "alanya-goztepe.parquet"
    print(f"Loading match data for {filename}...")
    df = get_match_dataframe(filename)
    if df is None:
        print("Failed to load match dataframe.")
        return

    teams = df['team_name'].unique()
    teams = [t for t in teams if isinstance(t, str)]
    print("Teams found:", teams)
    
    if len(teams) < 2:
        print("Insufficient team data")
        return

    if 'team_position' in df.columns:
        home_team = df[df['team_position'] == 'home']['team_name'].iloc[0]
        away_team = df[df['team_position'] == 'away']['team_name'].iloc[0]
    else:
        home_team = teams[0]
        away_team = teams[1]

    print(f"Home: {home_team}, Away: {away_team}")
    h_goals = len(df[(df['team_name'] == home_team) & (df['type_id'] == 16)])
    a_goals = len(df[(df['team_name'] == away_team) & (df['type_id'] == 16)])

    # Test plot functions
    plot_funcs = [
        ("plot_match_shot_map", lambda: visuals.plot_match_shot_map(df, home_team, away_team)),
        ("plot_pitch_dominance", lambda: visuals.plot_pitch_dominance(df, home_team, away_team)),
        ("plot_player_dashboard_bars", lambda: visuals.plot_player_dashboard_bars(df, home_team, away_team, h_goals, a_goals)),
    ]

    for name, func in plot_funcs:
        try:
            print(f"Testing {name}...")
            res = func()
            print(f"  {name} success: returned length {len(res) if res else 0}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Processing tempo and OBV models ---")
    try:
        home_tempo = tempo_data.process_tempo_network(df, home_team)
        away_tempo = tempo_data.process_tempo_network(df, away_team)
        df_obv     = obv_model.calculate_obv(df)
        print("Tempo processing and OBV calculation successful")
    except Exception as e:
        print(f"Model processing FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    team_plot_funcs = [
        ("plot_progressive_pass_map", lambda t: visuals.plot_progressive_pass_map(df, t)),
        ("plot_hybrid_pass_network", lambda t: tempo_visuals.plot_hybrid_pass_network(df, t, home_tempo if t == home_team else away_tempo)),
        ("plot_xt_leaders", lambda t: visuals.plot_xt_leaders(df, t)),
        ("plot_starting_xi", lambda t: visuals.plot_starting_xi(df, t)),
        ("plot_defensive_profile", lambda t: visuals.plot_defensive_profile(df, t)),
        ("plot_obv_pitch", lambda t: obv_visuals.plot_obv_pitch(df_obv, t)),
        ("plot_obv_leaderboard", lambda t: obv_visuals.plot_obv_leaderboard(df_obv, t)),
        ("plot_pressing_map", lambda t: visuals.plot_pressing_map(df, t)),
        ("plot_offensive_transition_map", lambda t: visuals.plot_offensive_transition_map(df, t)),
        ("plot_set_pieces corners", lambda t: visuals.plot_set_pieces(df, t, "corners")),
        ("plot_set_pieces free_kicks", lambda t: visuals.plot_set_pieces(df, t, "free_kicks")),
        ("plot_goal_kicks_distribution", lambda t: visuals.plot_goal_kicks_distribution(df, t)),
    ]

    for name, func in team_plot_funcs:
        for team in [home_team, away_team]:
            try:
                print(f"Testing {name} for {team}...")
                res = func(team)
                print(f"  {name} success: returned length {len(res) if res else 0}")
            except Exception as e:
                print(f"  {name} FAILED for {team}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_plots()

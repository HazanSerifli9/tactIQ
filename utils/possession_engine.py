"""
Possession Chain Engine
========================
Extracts continuous possession sequences from Opta event data.
Each chain tracks: team, events, timing, and how the possession ended.

This is the foundation for:
- SEPP (Shot-Ending Possession Passes)
- Ball Trace (time-zone territorial analysis)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Events that don't count as "field play" — skip during chain building
SKIP_EVENTS = {
    'Team setp up', 'Start', 'End', 'Start delay', 'End delay',
    'Injury Time Announcement', 'Card', 'Player Off', 'Player on',
    'Formation change', 'Collection End', 'Deleted event',
    'Referee Drop Ball', 'Contentious referee decision',
    'Referee stop', 'Referee delay', 'Resume', 'Suspended',
    'Game end', 'Post match complete',
}

# Shot events that end a possession with a shot
SHOT_EVENTS = {'Goal', 'Miss', 'Saved Shot', 'Post', 'Attempt Saved'}

# Events that signal end of play (out of bounds, etc.)
OUT_EVENTS = {'Out', 'Offside Pass', 'Foul throw-in'}

# Set piece qualifiers — if present on the first pass, mark the chain
SET_PIECE_QUALIFIERS = [
    'Corner taken', 'Free kick taken', 'Throw In', 
    'Goal Kick', 'Set piece', 'From corner', 'Free kick',
    'Throw In set piece', 'Penalty',
]


@dataclass
class PossessionChain:
    """Represents a single continuous possession sequence."""
    team: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0       # seconds from period start
    end_time: float = 0.0
    period_id: int = 1
    ended_with_shot: bool = False
    end_event: str = ''
    is_set_piece: bool = False     # started from a set piece
    
    @property
    def duration(self) -> float:
        return max(0, self.end_time - self.start_time)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for e in self.events if e.get('event') == 'Pass')
    
    @property
    def successful_pass_count(self) -> int:
        return sum(1 for e in self.events 
                   if e.get('event') == 'Pass' and e.get('outcome') == 1)


def _event_time_seconds(row) -> float:
    """Convert time_min + time_sec to total seconds."""
    return float(row.get('time_min', 0)) * 60 + float(row.get('time_sec', 0))


def _is_set_piece_start(event_dict: dict) -> bool:
    """Check if an event has set piece qualifiers."""
    for q in SET_PIECE_QUALIFIERS:
        val = event_dict.get(q)
        if val is not None and str(val).strip().lower() in ('si', '1', 'true', 'yes'):
            return True
    return False


def extract_possession_chains(df: pd.DataFrame, team_name: str) -> List[PossessionChain]:
    """
    Extract all possession chains for a given team from a match dataframe.
    
    A possession chain starts when the team gains the ball and ends when:
    - The opponent gains possession
    - A shot is taken
    - The ball goes out of play
    - A foul/set piece interrupts
    
    Args:
        df: Match dataframe with Opta event data
        team_name: Team to extract chains for
        
    Returns:
        List of PossessionChain objects
    """
    if df.empty:
        return []
    
    # Sort by period and time
    df_sorted = df.sort_values(
        by=['period_id', 'time_min', 'time_sec', 'event_id']
    ).reset_index(drop=True)
    
    chains: List[PossessionChain] = []
    current_chain: Optional[PossessionChain] = None
    
    for _, row in df_sorted.iterrows():
        event = row.get('event', '')
        team = row.get('team_name', '')
        
        # Skip non-field events
        if event in SKIP_EVENTS:
            continue
        
        period = row.get('period_id', 1)
        # Skip setup periods (16 = team setup)
        if period == 16:
            continue
            
        ev_time = _event_time_seconds(row)
        ev_dict = row.to_dict()
        
        # --- Decision: should we end the current chain? ---
        should_end_chain = False
        end_reason = ''
        
        if current_chain is not None:
            # Period change
            if period != current_chain.period_id:
                should_end_chain = True
                end_reason = 'period_change'
            # Team change (opponent gained possession)
            elif team != current_chain.team and team != '' and not pd.isna(team):
                should_end_chain = True
                end_reason = 'turnover'
            # Shot event (ends possession)
            elif event in SHOT_EVENTS and team == current_chain.team:
                # Add this event to the chain first, then close
                current_chain.events.append(ev_dict)
                current_chain.end_time = ev_time
                current_chain.ended_with_shot = True
                current_chain.end_event = event
                chains.append(current_chain)
                current_chain = None
                continue
            # Out events
            elif event in OUT_EVENTS:
                should_end_chain = True
                end_reason = 'out'
            # Foul by opponent (gives us a set piece — new chain)
            elif event == 'Foul' and team != current_chain.team:
                should_end_chain = True
                end_reason = 'foul_won'
        
        if should_end_chain and current_chain is not None:
            current_chain.end_time = ev_time
            current_chain.end_event = end_reason
            if len(current_chain.events) > 0:
                chains.append(current_chain)
            current_chain = None
        
        # --- Start new chain or add to existing ---
        if team == team_name:
            if current_chain is None:
                current_chain = PossessionChain(
                    team=team_name,
                    start_time=ev_time,
                    period_id=period,
                )
                # Check for set piece start
                if _is_set_piece_start(ev_dict):
                    current_chain.is_set_piece = True
                    
            current_chain.events.append(ev_dict)
            current_chain.end_time = ev_time
    
    # Don't forget the last chain
    if current_chain is not None and len(current_chain.events) > 0:
        chains.append(current_chain)
    
    return chains


def extract_all_possession_chains(df: pd.DataFrame) -> Dict[str, List[PossessionChain]]:
    """
    Extract possession chains for ALL teams in a match.
    
    Returns:
        Dict mapping team_name -> List[PossessionChain]
    """
    teams = [t for t in df['team_name'].unique() if isinstance(t, str) and len(t) > 2]
    
    result = {}
    for team in teams:
        result[team] = extract_possession_chains(df, team)
    
    return result

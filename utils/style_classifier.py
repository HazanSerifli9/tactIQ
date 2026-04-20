"""
Tactical Style Classifier
==========================
Stephanatos metodolojisine göre Süper Lig takımlarını sınıflandırır.

İki ana boyut:
  - Possession Style Score  → ne kadar kontrollü / direkt oynuyor
  - Defensive Territory Score → savunma hattı nerede, ne kadar agresif

Wyscout verisi önceliği:
  xG, PPDA, shot distance, pass accuracy, long pass %, avg passes/poss
  gibi metrikler artık doğrudan Wyscout'tan alınır.
  Event verisi bu değerleri tamamlayan bileşenleri sağlar (seq10, def_height vb.)

Referans: Nikolas Stephanatos, "Tactical Performance Profiling in Football"
"""

import numpy as np
import pandas as pd
from typing import Optional
from utils.possession_engine import extract_possession_chains
from shared.logger import get_logger

logger = get_logger(__name__)


# ── Metrik ağırlıkları ──────────────────────────────────────────────────────

# On-Ball bileşen ağırlıkları (toplam 1.0)
W_PASS_ACC     = 0.20   # Wyscout: pas isabet %
W_FWD_PASS     = 0.15   # Event: ileri pas %
W_LONG_BALL    = 0.15   # Wyscout: uzun top % (ters: yüksek = direkt)
W_SEQ_10       = 0.15   # Event: 10+ paslı sekans oranı
W_XG_PER_SHOT  = 0.20   # Wyscout: şut başına xG (kaliteli pozisyon)
W_AVG_POSS     = 0.15   # Wyscout: ortalama pas/possesion (yüksek=kontrollü)

# Off-Ball bileşen ağırlıkları (toplam 1.0)
W_PPDA         = 0.35   # Wyscout: PPDA (ters: düşük = yüksek pres)
W_DEF_HEIGHT   = 0.25   # Event: savunma hattı yüksekliği
W_SHOT_DIST    = 0.20   # Wyscout: avg shot distance (ters: yakın = yüksek pres)
W_DEF_ACT90    = 0.20   # Event: savunma aksiyonu / 90dk

STYLE_LABELS = {
    (True,  True):  "Control-ball",
    (True,  False): "Trigger-happy Control",
    (False, True):  "High Press & Vertical",
    (False, False): "Mid-block & Counter",
}

CHAOS_THRESHOLD  = 85
LOWBLOCK_THRESHOLD = 25


def _safe_ratio(num, denom, default=0.0):
    return num / denom if denom > 0 else default


def build_team_style_profiles(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm takımlar için taktik stil profilini hesaplar.

    Wyscout verisi öncelikliyse oradan alınır; yoksa event verisinden hesaplanır.

    Args:
        events_df: load_all_events() çıktısı (tüm maçlar birleşik)

    Returns:
        Her satır bir takım olan DataFrame:
          team, pass_acc_pct, fwd_pass_pct, long_ball_pct,
          seq10_pct, xg_per_shot, avg_passes_per_poss,
          ppda, def_height, avg_shot_distance, def_act90,
          possession_score, defensive_score, style_label
    """
    # ── Wyscout verisini yükle ─────────────────────────────────────────────
    try:
        from utils.wyscout_loader import load_wyscout_team_averages, get_wyscout_team_name_map
        wyscout_df = load_wyscout_team_averages()
        name_map   = get_wyscout_team_name_map()
        has_wyscout = not wyscout_df.empty
    except Exception as e:
        logger.warning("Wyscout verisi yüklenemedi, event verisine geri dönülüyor: %s", e)
        wyscout_df  = pd.DataFrame()
        name_map    = {}
        has_wyscout = False

    if events_df.empty:
        if has_wyscout:
            return _build_from_wyscout_only(wyscout_df)
        return pd.DataFrame()

    teams = [t for t in events_df['team_name'].dropna().unique()
             if isinstance(t, str) and len(t) > 2]

    rows = []

    for team in teams:
        df_team = events_df[events_df['team_name'] == team]
        if df_team.empty:
            continue

        # Wyscout takım adını bul
        wy_name = name_map.get(team, team)
        wy_row  = None
        if has_wyscout and 'wyscout_team' in wyscout_df.columns:
            matches = wyscout_df[wyscout_df['wyscout_team'] == wy_name]
            if not matches.empty:
                wy_row = matches.iloc[0]

        # ── Maç sayısı ────────────────────────────────────────────────────
        match_ids = df_team['match_id'].unique()
        n_matches = max(len(match_ids), 1)

        # ── ON-BALL metrikleri ─────────────────────────────────────────────

        passes = df_team[df_team['event'] == 'Pass']
        total_passes = len(passes)

        # 1. Pas isabet % — Wyscout öncelikli
        if wy_row is not None and pd.notna(wy_row.get('pass_accuracy_pct', np.nan)):
            pass_acc_pct = float(wy_row['pass_accuracy_pct'])
        else:
            successful_passes = len(passes[passes['outcome'] == 1])
            pass_acc_pct = _safe_ratio(successful_passes, total_passes) * 100

        # 2. İleri pas % — Event verisi
        if 'Pass End X' in passes.columns:
            fwd_passes = passes[
                pd.to_numeric(passes['Pass End X'], errors='coerce') >
                pd.to_numeric(passes['x'], errors='coerce')
            ]
            fwd_pass_pct = _safe_ratio(len(fwd_passes), total_passes) * 100
        else:
            fwd_pass_pct = 50.0

        # 3. Uzun top % — Wyscout öncelikli
        if wy_row is not None and pd.notna(wy_row.get('long_pass_pct', np.nan)):
            long_ball_pct = float(wy_row['long_pass_pct'])
        elif 'Long ball' in df_team.columns:
            long_balls = passes[
                passes['Long ball'].astype(str).isin(['Si', '1', '1.0', 'True', 'true'])
            ]
            long_ball_pct = _safe_ratio(len(long_balls), total_passes) * 100
        else:
            long_ball_pct = 0.0

        # 4. 10+ paslı sekanslar — Event verisi
        seq_total, seq_10plus = 0, 0
        for mid in match_ids:
            match_df = events_df[events_df['match_id'] == mid]
            try:
                chains = extract_possession_chains(match_df, team)
                for c in chains:
                    seq_total += 1
                    if c.pass_count >= 10:
                        seq_10plus += 1
            except Exception as e:
                logger.debug("Possession chain extraction skipped: %s", e)

        seq10_pct = _safe_ratio(seq_10plus, seq_total) * 100

        # 5. Şut başına xG — Wyscout öncelikli
        shot_events = ['Goal', 'Miss', 'Attempt Saved', 'Post', 'Saved Shot']
        shots = df_team[df_team['event'].isin(shot_events)]
        n_shots = len(shots)

        if wy_row is not None and pd.notna(wy_row.get('xg_for', np.nan)) and pd.notna(wy_row.get('shots_total', np.nan)):
            wy_shots = float(wy_row['shots_total'])
            wy_xg    = float(wy_row['xg_for'])
            xg_per_shot = _safe_ratio(wy_xg, wy_shots)
        else:
            if 'xG' in shots.columns:
                total_xg = pd.to_numeric(shots['xG'], errors='coerce').sum()
            else:
                total_xg = 0.0
            xg_per_shot = _safe_ratio(total_xg, n_shots)

        # 6. Ortalama pas/possession — Wyscout öncelikli (yüksek = kontrollü)
        if wy_row is not None and pd.notna(wy_row.get('avg_passes_per_poss', np.nan)):
            avg_passes_per_poss = float(wy_row['avg_passes_per_poss'])
        else:
            avg_passes_per_poss = _safe_ratio(total_passes, max(seq_total, 1))

        # ── OFF-BALL metrikleri ────────────────────────────────────────────

        # 7. PPDA — Wyscout öncelikli (düşük = yüksek pres)
        if wy_row is not None and pd.notna(wy_row.get('ppda', np.nan)):
            ppda = float(wy_row['ppda'])
        else:
            opp_mask = events_df['team_name'] != team
            opp_team_matches = events_df[
                events_df['match_id'].isin(match_ids) & opp_mask
            ]
            opp_passes_own_half = opp_team_matches[
                (opp_team_matches['event'] == 'Pass') &
                (pd.to_numeric(opp_team_matches['x'], errors='coerce') < 50)
            ]
            def_actions_in_opp = df_team[
                df_team['event'].isin(['Tackle', 'Interception', 'Challenge', 'Ball Recovery']) &
                (pd.to_numeric(df_team['x'], errors='coerce') > 50)
            ]
            ppda = _safe_ratio(len(opp_passes_own_half), len(def_actions_in_opp), default=20.0)

        # 8. Savunma hattı yüksekliği — Event verisi
        ball_wins = df_team[
            df_team['event'].isin(['Tackle', 'Interception', 'Ball Recovery']) &
            (df_team['outcome'] == 1)
        ]
        if not ball_wins.empty:
            def_height = pd.to_numeric(ball_wins['x'], errors='coerce').mean()
        else:
            def_height = 40.0

        # 9. Ortalama şut mesafesi — Wyscout öncelikli (düşük = yüksek baskı)
        if wy_row is not None and pd.notna(wy_row.get('avg_shot_distance', np.nan)):
            avg_shot_distance = float(wy_row['avg_shot_distance'])
        else:
            avg_shot_distance = 20.0  # Default: orta mesafe

        # 10. Savunma aksiyonları / 90dk — Event verisi
        def_acts = len(df_team[df_team['event'].isin(
            ['Tackle', 'Interception', 'Challenge', 'Ball Recovery', 'Clearance']
        )])
        total_mins = n_matches * 90
        def_act90 = _safe_ratio(def_acts, total_mins) * 90

        rows.append({
            'team':               team,
            'pass_acc_pct':       round(pass_acc_pct, 2),
            'fwd_pass_pct':       round(fwd_pass_pct, 2),
            'long_ball_pct':      round(long_ball_pct, 2),
            'seq10_pct':          round(seq10_pct, 2),
            'xg_per_shot':        round(xg_per_shot, 4),
            'avg_passes_per_poss':round(avg_passes_per_poss, 2),
            'ppda':               round(ppda, 2),
            'def_height':         round(def_height, 2),
            'avg_shot_distance':  round(avg_shot_distance, 2),
            'def_act90':          round(def_act90, 2),
            # Wyscout ek metrikler (görsel tablo için)
            'xg_for':             float(wy_row['xg_for'])   if wy_row is not None and pd.notna(wy_row.get('xg_for'))   else np.nan,
            'xg_against':         float(wy_row.get('shots_against', np.nan)) if wy_row is not None else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    profiles = pd.DataFrame(rows)
    return _normalize_and_label(profiles)


def _build_from_wyscout_only(wyscout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sadece Wyscout verisi ile profil oluştur (event verisi yoksa).
    """
    rows = []
    for _, wy in wyscout_df.iterrows():
        team = wy['wyscout_team']

        pass_acc_pct       = float(wy.get('pass_accuracy_pct', 75))
        fwd_pass_pct       = 50.0   # bilinmiyor
        long_ball_pct      = float(wy.get('long_pass_pct', 20))
        seq10_pct          = 0.0
        xg_per_shot        = _safe_ratio(
            float(wy.get('xg_for', 0)),
            float(wy.get('shots_total', 1)),
            default=0.1
        )
        avg_passes_per_poss = float(wy.get('avg_passes_per_poss', 4))
        ppda               = float(wy.get('ppda', 10))
        def_height         = 40.0
        avg_shot_distance  = float(wy.get('avg_shot_distance', 20))
        def_act90          = 100.0

        rows.append({
            'team':               team,
            'pass_acc_pct':       round(pass_acc_pct, 2),
            'fwd_pass_pct':       round(fwd_pass_pct, 2),
            'long_ball_pct':      round(long_ball_pct, 2),
            'seq10_pct':          round(seq10_pct, 2),
            'xg_per_shot':        round(xg_per_shot, 4),
            'avg_passes_per_poss':round(avg_passes_per_poss, 2),
            'ppda':               round(ppda, 2),
            'def_height':         round(def_height, 2),
            'avg_shot_distance':  round(avg_shot_distance, 2),
            'def_act90':          round(def_act90, 2),
            'xg_for':             float(wy.get('xg_for', np.nan)),
            'xg_against':         float(wy.get('shots_against', np.nan)),
        })

    if not rows:
        return pd.DataFrame()

    return _normalize_and_label(pd.DataFrame(rows))


def _normalize_and_label(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Ham metriklerden percentile puanları ve stil etiketleri üretir.
    """

    def percentile_rank(series: pd.Series) -> pd.Series:
        from scipy.stats import rankdata
        ranks = rankdata(series.fillna(series.median()), method='average')
        return pd.Series(
            (ranks - 1) / max(len(ranks) - 1, 1) * 100,
            index=series.index
        )

    # ON-BALL normalizasyon
    profiles['p_pass_acc']       = percentile_rank(profiles['pass_acc_pct'])
    profiles['p_fwd_pass']       = percentile_rank(profiles['fwd_pass_pct'])
    profiles['p_long_ball']      = 100 - percentile_rank(profiles['long_ball_pct'])  # ters
    profiles['p_seq10']          = percentile_rank(profiles['seq10_pct'])
    profiles['p_xg_per_shot']    = percentile_rank(profiles['xg_per_shot'])
    profiles['p_avg_poss']       = percentile_rank(profiles['avg_passes_per_poss'])

    # OFF-BALL normalizasyon
    profiles['p_ppda']           = 100 - percentile_rank(profiles['ppda'])           # ters: düşük PPDA = yüksek pres
    profiles['p_def_height']     = percentile_rank(profiles['def_height'])
    profiles['p_shot_dist']      = 100 - percentile_rank(profiles['avg_shot_distance'])  # ters: yakın = yüksek pres
    profiles['p_def_act90']      = percentile_rank(profiles['def_act90'])

    # ── Kompozit Skorlar ─────────────────────────────────────────────────
    profiles['possession_score'] = (
        profiles['p_pass_acc']    * W_PASS_ACC   +
        profiles['p_fwd_pass']    * W_FWD_PASS   +
        profiles['p_long_ball']   * W_LONG_BALL  +
        profiles['p_seq10']       * W_SEQ_10     +
        profiles['p_xg_per_shot'] * W_XG_PER_SHOT +
        profiles['p_avg_poss']    * W_AVG_POSS
    ).round(1)

    profiles['defensive_score'] = (
        profiles['p_ppda']       * W_PPDA       +
        profiles['p_def_height'] * W_DEF_HEIGHT +
        profiles['p_shot_dist']  * W_SHOT_DIST  +
        profiles['p_def_act90']  * W_DEF_ACT90
    ).round(1)

    # ── Stil Etiketi ────────────────────────────────────────────────────
    lig_mean_poss = profiles['possession_score'].mean()
    lig_mean_def  = profiles['defensive_score'].mean()
    lig_std_poss  = profiles['possession_score'].std()
    lig_std_def   = profiles['defensive_score'].std()

    threshold = 0.75

    def label(row):
        high_poss = row['possession_score'] > lig_mean_poss + threshold * lig_std_poss
        low_poss  = row['possession_score'] < lig_mean_poss - threshold * lig_std_poss
        high_def  = row['defensive_score']  > lig_mean_def  + threshold * lig_std_def
        low_def   = row['defensive_score']  < lig_mean_def  - threshold * lig_std_def

        if row['defensive_score'] >= CHAOS_THRESHOLD and row['possession_score'] < 40:
            return "Chaos-ball"
        if low_def and low_poss:
            return "Low Block & Play Out"
        if high_poss and high_def:
            return "Control-ball"
        if high_poss and not high_def:
            return "Trigger-happy Control"
        if not high_poss and high_def:
            return "High Press & Vertical"
        if low_def and not low_poss:
            return "Mid-block & Counter"
        return "Mixed"

    profiles['style_label'] = profiles.apply(label, axis=1)

    return profiles.sort_values('possession_score', ascending=False).reset_index(drop=True)

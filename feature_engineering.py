import pandas as pd
import numpy as np


def build_trackman_aggregates(tm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trackman 이력 데이터로부터 투수/타자별 요약 통계 지표를 집계합니다.
    """
    # 투수별 구속, 회전수, 무브먼트의 평균 및 변동성(STD)
    tm_pitcher = tm_df.groupby('pitcher_trackman_id').agg(
        tm_p_speed_mean=('rel_speed', 'mean'),
        tm_p_speed_std=('rel_speed', 'std'),
        tm_p_spin_mean=('spin_rate', 'mean'),
        tm_p_vbreak_mean=('induced_vert_break', 'mean'),
        tm_p_hbreak_mean=('horz_break', 'mean')
    ).reset_index().rename(columns={'pitcher_trackman_id': 'pitcher_id'})

    # 타자별 상대한 구속, 회전수, 무브먼트 평균
    tm_batter = tm_df.groupby('batter_trackman_id').agg(
        tm_b_speed_mean=('rel_speed', 'mean'),
        tm_b_spin_mean=('spin_rate', 'mean'),
        tm_b_vbreak_mean=('induced_vert_break', 'mean')
    ).reset_index().rename(columns={'batter_trackman_id': 'batter_id'})

    return tm_pitcher, tm_batter


def advanced_feature_engineering(
    df: pd.DataFrame, 
    tm_p: pd.DataFrame, 
    tm_b: pd.DataFrame
) -> pd.DataFrame:
    """
    메인 데이터프레임에 Trackman 통계 병합, 상성 지표 계산, 
    볼카운트/도메인 피처, 최근 폼 및 베이지안 스무딩 지표를 파생합니다.
    """
    df = df.copy()

    # 1. Trackman 통계 병합
    df = df.merge(tm_p, on='pitcher_id', how='left')
    df = df.merge(tm_b, on='batter_id', how='left')

    # Cold-Start Fallback (Trackman 결측치를 전체 평균으로 대치)
    for col in tm_p.columns:
        if col != 'pitcher_id':
            df[col] = df[col].fillna(df[col].mean())
    for col in tm_b.columns:
        if col != 'batter_id':
            df[col] = df[col].fillna(df[col].mean())

    # 2. [상성 및 상대 지표] 투수 - 타자 대응 능력 차이
    df['speed_diff'] = df['tm_p_speed_mean'] - df['tm_b_speed_mean']
    df['spin_diff'] = df['tm_p_spin_mean'] - df['tm_b_spin_mean']
    df['vbreak_diff'] = df['tm_p_vbreak_mean'] - df['tm_b_vbreak_mean']

    # 3. [야구 도메인 & 볼카운트 정교화]
    df['count_state'] = df['balls_before'].astype(str) + "B_" + df['strikes_before'].astype(str) + "S"
    df['is_hitter_advantage'] = (df['balls_before'] > df['strikes_before']).astype(int)
    df['is_two_strikes'] = (df['strikes_before'] == 2).astype(int)
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(int)
    df['is_platoon_advantage'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)

    # 4. [최근 경기 흐름 & 폼 피처]
    df['recent_form_diff'] = df['asof_pitcher_prev1_game_success_rate'].fillna(0) - df['asof_pitcher_success_rate'].fillna(0)
    df['recent_3g_diff'] = df['asof_pitcher_prev3_game_success_rate'].fillna(0) - df['asof_pitcher_success_rate'].fillna(0)

    # 5. [Bayesian Smoothing] 소수 투구 수 투수에 대한 성공률 보정
    global_mean_success = 0.55
    m = 20.0
    n = df['asof_pitcher_n'].fillna(0)
    rate = df['asof_pitcher_success_rate'].fillna(global_mean_success)
    df['asof_pitcher_success_smoothed'] = (n * rate + m * global_mean_success) / (n + m)

    # 6. [구종 조합 비율 결측 보정]
    df['fastball_ratio'] = df['asof_pitcher_fastball_rate'].fillna(0)
    df['breaking_ratio'] = df['asof_pitcher_breaking_rate'].fillna(0)
    df['offspeed_ratio'] = df['asof_pitcher_offspeed_rate'].fillna(0)

    # 7. 범주형 변수 타입 변환
    cat_cols = [
        'top_bottom', 'game_type', 'base_state', 'count_state',
        'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id'
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

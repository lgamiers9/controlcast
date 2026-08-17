# 19_actual_files_with_yeongeun_timeaware.py
"""팀원 각자 만든 실제 코드를 그대로 가져와 합친 피처 세트 - trackman 상황별 집계
(손잡이+카운트+아웃 조합별 trackman 평균)까지 시간 인지형(time-aware)으로 포함한 최종안.

trackman 상황별 집계는 원래 연도 무관하게 통째로 평균내는 방식이라 "평가 시점 이후
정보 사용 금지" 규정 위반 소지가 있었음 -> 각 행 시점보다 엄격히 이전 데이터만
반영하는 시간 인지형 방식(merge_asof)으로 고쳐서 포함시킴.

trackman 매칭 시도(투수 ID 기반, 11개 컬럼)는 매칭률 0%로 확인됨(익명 ID 체계가
서로 달라서 애초에 조인이 안 됨) - 시간 인지형으로 고쳐도 소용없어서 계속 제외.

최종 32개 피처 = 카운트/최근폼/구종비율 8개 + 카운트압박/조회테이블 7개 +
제구비율/경험치 4개 + 압박상황/매치업 8개 + trackman 상황별 집계(시간 인지형) 5개
(tm_situ_fastball_rate, tm_situ_breaking_rate, tm_situ_offspeed_rate,
tm_situ_zone_speed_mean, tm_situ_horz_break_mean)

=======================================================================
실행 결과 (2025-08, seed 1~5 평균, 총 소요시간 49초)
=======================================================================
32개 전체(trackman 상황별 집계 시간인지형 5개 포함): mean=685.91  std=12.32
  (참고: trackman 상황별 집계를 뺀 27개 단계는 mean=683.64  std=8.23이었음)

결론: +2.27로 두 세트 표준편차보다 작아서 노이즈 범위 안 - trackman 상황별 집계 5개를
추가해도 실질적인 개선은 확인되지 않음. 오히려 표준편차만 8.23 -> 12.32로 커져서 더
불안정해짐. 규정을 지키는 방식으로 제대로 고쳐 넣었어도 이 조합에서는 큰 도움이 안
된다는 뜻 - 27개로 가나 32개로 가나 실질적 차이는 없음.

=======================================================================
피처 중요도 & 잔차 분석 (seed=42 단일 실행, 2025-08)
=======================================================================
seed=42 BSS=676.62  best_iteration=154 (5시드 평균 685.91과 차이 있음 - 시드 노이즈 범위)

피처 중요도(gain) 상위: game_type(54050.5), season(36690.7), asof_pitcher_success_smoothed
(33326.9, 최근폼/구종비율 계열), asof_pitcher_success_rate(15252.0),
asof_pitcher_reverse_rate(14096.4), batter_team_id(11918.7), pitcher_team_id(10355.7),
f_share(10121.8, 조회테이블 계열), count_state(7281.5, 카운트 계열),
tm_situ_offspeed_rate(6042.3, trackman 상황별 집계) - trackman 상황별 집계도 상위권에
실제로 gain을 먹고 있음이 확인됨.

완전히 죽은 피처(gain 0, 7개): is_experienced_mix, is_disadvantaged_count, is_runner_on,
is_two_outs, is_batter_cold_start, is_cold_start, is_first_pitch

주의: hand_match는 gain 자체는 낮은 편(1551.7)인데, 이 피처 하나만 빼고 다시 검증했을
때는 683.64 -> 575.97로 압도적으로 떨어졌었음(-107.67). 즉 gain 지표만으로는 "대체 불가능한
구조적 중요성"이 안 보일 수 있음 - 다른 피처들과 gain을 나눠 쓰지만 없으면 대체가 안
되는 경우가 있다는 뜻. gain 낮다고 함부로 제거하면 안 됨.

잔차 분석 (홀드아웃 2024, Murphy decomposition):
  BSS=671.14  REL=0.000052(낮음=캘리브레이션 신뢰도 좋음)  RES=0.001729
  mean_pred - r = +0.004602 (예측 평균이 실제보다 0.46%p 높음 - 미세한 과대예측 잔존,
  시즌 추세 과소보정 문제의 잔여분으로 보이나 크기는 작음)

예측확률 10분위 캘리브레이션: 대부분 구간에서 예측-실제 격차(gap)가 ±0.01 안쪽으로 양호.
10개 구간 중 7개가 양의 gap(과대예측)이라 방향은 살짝 일관되게 위로 치우쳐 있지만 절대
크기는 작아서 심각한 미스캘리브레이션은 아님.

전체 예측 range: min=0.3333 max=0.6498 std=0.0423 (실제 r=0.4861 대비 적절한 폭)

=======================================================================
개별 피처 leave-one-out 5시드 ablation (32개 전체, 2025-08, 총 소요시간 1098초)
=======================================================================
baseline(32개 전체): mean=685.91  std=12.32 (노이즈 폭 자체가 커서, 아래 delta들도
±10~17 이내는 노이즈일 가능성 염두에 두고 볼 것)

컬럼 제거                          mean      std     baseline 대비
--------------------------------------------------------------------
count_state                      685.46    28.57          -0.45
is_two_strikes                   683.18    16.67          -2.74
is_full_count                    678.98    18.34          -6.93
recent_form_diff                 682.72    18.76          -3.19
asof_pitcher_success_smoothed    702.78    22.39         +16.87
fastball_ratio                   678.81    12.16          -7.11
breaking_ratio                   686.51    18.11          +0.59
offspeed_ratio                   693.26    10.99          +7.35
count_pressure                   695.19    15.39          +9.27
count_depth                      686.24    17.22          +0.33
form_dev_3                       680.28    19.21          -5.63
skill_gap                        680.66    13.16          -5.25
residual                         683.29    13.31          -2.63
f_share                          691.76     6.36          +5.85
career_span                      675.31     9.09         -10.61
pitcher_control_ratio            697.65     6.54         +11.73
is_batter_cold_start             690.60    14.83          +4.68
season_progress                  693.94    10.90          +8.03
is_experienced_mix               669.74    35.77         -16.17
is_disadvantaged_count           688.08    17.70          +2.17
is_runner_on                     684.29    27.43          -1.63
is_late_inning                   679.33    30.24          -6.59
is_two_outs                      690.86    18.83          +4.94
pressure_score                   687.35    14.39          +1.43
hand_match                       677.67    18.65          -8.25
is_cold_start                    678.09    12.17          -7.82
is_first_pitch                   687.15    10.72          +1.24
tm_situ_fastball_rate            682.30    10.02          -3.62
tm_situ_breaking_rate            669.23    30.80         -16.69
tm_situ_offspeed_rate            688.69    12.40          +2.77
tm_situ_zone_speed_mean          685.27     9.88          -0.65
tm_situ_horz_break_mean          675.62    13.10         -10.30

핵심 발견 - hand_match 영향력 급감:
  27개 세트(trackman 상황별 집계 없음)에서는 hand_match 제거 시 -107.67로 압도적이었는데,
  32개 세트(trackman 상황별 집계 포함)에서는 -8.25로 크게 줄었다. trackman 상황별 집계
  피처들(특히 tm_situ_offspeed_rate, tm_situ_breaking_rate)이 손잡이 매치업 정보를
  어느 정도 대체해주고 있다는 뜻 - 완전 대체는 아니지만(여전히 제거하면 손해), 27개
  세트만큼 "이거 하나에 목숨 걸린" 상태는 아니게 됨.

gain과 ablation이 어긋나는 사례들:
  - asof_pitcher_success_smoothed: importance gain 2위(33326.9)였는데 ablation에서는
    제거 시 +16.87로 가장 크게 개선됨 - 이 피처가 오히려 노이즈를 더하고 있었을 가능성.
  - is_experienced_mix: importance gain 0(seed=42 단일 실행 기준)이었는데 5시드 ablation
    에서는 제거 시 -16.17로 크게 나빠짐 - 단일 실행 gain 0을 보고 "필요없다"고 판단하면
    안 된다는 걸 재확인.

실무 결론: 32개 세트에서도 명확하게 "이건 반드시 빼자"고 할 만큼 일관된 피처는 없음
(baseline 표준편차 자체가 12.32로 커서 대부분의 delta가 노이즈 범위 안). 32개 전체
유지가 여전히 안전한 선택.

=======================================================================
최종 결론
=======================================================================
채택: 32개 전체 유지. 이유:
  1) baseline(32개, 685.91)이 어떤 leave-one-out 결과보다도 표준편차 대비 안정적이고,
     지금까지 여러 세트(50개/27개/32개)에서 반복적으로 "부분 제거가 도움 안 됨"이 확인됨.
  2) gain(중요도)과 ablation(실제 제거 영향) 결과가 서로 어긋나는 피처가 있어서
     (asof_pitcher_success_smoothed, is_experienced_mix), 어느 한쪽 지표만 보고
     피처를 쳐내는 건 위험함이 재확인됨.
  3) hand_match처럼 gain은 낮아도 구조적으로 중요한 피처가 있다는 것도 다시 확인됨 -
     다만 trackman 상황별 집계가 추가되면서 그 의존도가 완화되는 것도 함께 관찰됨
     (다른 피처가 보완할 수 있는 여지가 있다는 뜻).
  4) 팀원 전원의 작업이 다 반영된다는 점도 32개를 유지하는 이유 중 하나.

다음 단계 후보: 피처 쳐내기는 여기서 마무리. 하이퍼파라미터 튜닝이나, gain-ablation
불일치가 큰 피처(asof_pitcher_success_smoothed, is_experienced_mix)에 대한 개별 재현성
검증(다른 시드 조합으로 한 번 더) 정도가 남은 선택지.
"""

import time

import lightgbm as lgb  # Windows에서 pandas보다 먼저 import해야 access violation 회피
import numpy as np
import pandas as pd

TRAIN_PATH = "../../data/train.csv"
TRACKMAN_PATH = "../../data/trackman_history.csv"

ID_COL = "row_id"
TARGET_COL = "control_success"
DROP_COLS = ["pitcher_id", "batter_id"]

EARLY_STOP_TAIL_SEASON = 2023
EARLY_STOP_TAIL_FRAC = 0.15
TRAIN_POOL_SEASONS = (2019, 2020, 2021, 2022, 2023)
HOLDOUT_SEASON = 2024

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "count_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]

BASE_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

SEEDS = [1, 2, 3, 4, 5]

# train/test 숫자코드(1/2) <-> trackman 문자열(Left/Right) 매핑.
# 비율 비교로 검증됨(trackman Right 74.9%/Left 25.1% vs train 2번코드 74.15%/1번코드 25.85%).
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}
SITU_KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]


# =======================
# 시간 인지형(time-aware) 공용 유틸
# =======================

def build_time_aware_lookup(tm, group_cols, metrics):
    tm = tm.copy()
    tm["_tkey"] = tm["season"] * 12 + tm["game_month"]
    agg = tm.groupby(group_cols + ["_tkey"])[metrics].agg(["sum", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index().sort_values(group_cols + ["_tkey"])
    for m in metrics:
        agg[f"{m}_cumsum"] = agg.groupby(group_cols)[f"{m}_sum"].cumsum()
        agg[f"{m}_cumcount"] = agg.groupby(group_cols)[f"{m}_count"].cumsum()
    result = agg[group_cols + ["_tkey"]].copy()
    for m in metrics:
        result[m] = agg[f"{m}_cumsum"] / agg[f"{m}_cumcount"]
    return result.sort_values("_tkey")


def join_time_aware(df, lookup, group_cols, prefix):
    df = df.copy()
    df["_tkey"] = df["season"] * 12 + df["game_month"]
    metrics = [c for c in lookup.columns if c not in group_cols + ["_tkey"]]
    df_sorted = df.sort_values("_tkey")
    lookup_sorted = lookup.sort_values("_tkey")
    merged = pd.merge_asof(
        df_sorted, lookup_sorted[group_cols + ["_tkey"] + metrics],
        on="_tkey", by=group_cols, direction="backward", allow_exact_matches=False,
    )
    merged = merged.rename(columns={m: f"{prefix}_{m}" for m in metrics})
    return merged.sort_index().drop(columns=["_tkey"])


def build_yeongeun_situ_stats(trackman_path=TRACKMAN_PATH):
    """trackman_history.csv 기반 손잡이+카운트+아웃 상황별 집계 - 시간 인지형.
    fastball/breaking/offspeed 비율은 0/1 indicator의 누적평균 = 누적비율이 되도록
    변환해서 build_time_aware_lookup()에 태운다."""
    usecols = SITU_KEY + ["pitch_type_group", "zone_speed", "horz_break", "season", "game_month"]
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig", usecols=usecols)
    tm["pitcher_hand"] = tm["pitcher_hand"].map(INV_HAND_MAP)
    tm["batter_hand"] = tm["batter_hand"].map(INV_HAND_MAP)

    tm["fastball_rate"] = (tm["pitch_type_group"] == "fastball").astype(int)
    tm["breaking_rate"] = (tm["pitch_type_group"] == "breaking").astype(int)
    tm["offspeed_rate"] = (tm["pitch_type_group"] == "offspeed").astype(int)
    tm["zone_speed_mean"] = tm["zone_speed"]
    tm["horz_break_mean"] = tm["horz_break"]

    metrics = ["fastball_rate", "breaking_rate", "offspeed_rate", "zone_speed_mean", "horz_break_mean"]
    return build_time_aware_lookup(tm, SITU_KEY, metrics)


def add_yeongeun_features(df, situ_lookup):
    return join_time_aware(df, situ_lookup, SITU_KEY, "tm_situ")


# =======================
# 카운트압박/상성/조회테이블 계열 - train.csv 전용 조회 테이블 (trackman과 무관)
# =======================

def build_haejin_lookups(lookup_source_df):
    is_f = lookup_source_df["game_type"] == "F"
    f_share = (lookup_source_df.assign(_f=is_f.astype("int8"))
               .groupby("pitcher_id", observed=True)["_f"].mean())
    debut_season = lookup_source_df.groupby("pitcher_id", observed=True)["season"].min()
    return {
        "f_share": f_share.to_dict(),
        "debut_season": debut_season.to_dict(),
        "fallback_f_share": float(f_share.median()),
        "fallback_debut": int(debut_season.median()),
    }


# =======================
# 각 계열 실제 코드 (팀원 각자 작성한 원본 그대로, trackman 매칭 부분만 제외)
# =======================

def add_eunwoo_features(df):
    """카운트상태 / 최근폼 / 구종비율 계열."""
    df = df.copy()
    df["count_state"] = df["balls_before"].astype(str) + "B_" + df["strikes_before"].astype(str) + "S"
    df["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["recent_form_diff"] = (
        df["asof_pitcher_prev1_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)
    )
    global_mean_success = 0.55
    m = 20.0
    n = df["asof_pitcher_n"].fillna(0)
    rate = df["asof_pitcher_success_rate"].fillna(global_mean_success)
    df["asof_pitcher_success_smoothed"] = (n * rate + m * global_mean_success) / (n + m)
    df["fastball_ratio"] = df["asof_pitcher_fastball_rate"].fillna(0)
    df["breaking_ratio"] = df["asof_pitcher_breaking_rate"].fillna(0)
    df["offspeed_ratio"] = df["asof_pitcher_offspeed_rate"].fillna(0)
    return df


def add_haejin_features(df, pressure_lookups):
    """카운트압박 / 상성 / 조회테이블 계열."""
    df = df.copy()
    df["count_pressure"] = (df["balls_before"] - df["strikes_before"]).astype("int8")
    df["count_depth"] = (df["balls_before"] + df["strikes_before"]).astype("int8")
    df["form_dev_3"] = (
        df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    )
    df["skill_gap"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["residual"] = (
        1.0 - df["asof_pitcher_success_rate"] - df["asof_pitcher_middle_rate"] - df["asof_pitcher_reverse_rate"]
    )
    pid = df["pitcher_id"].astype("int64")
    df["f_share"] = pid.map(pressure_lookups["f_share"]).fillna(pressure_lookups["fallback_f_share"]).astype("float32")
    debut = pid.map(pressure_lookups["debut_season"]).fillna(pressure_lookups["fallback_debut"])
    df["career_span"] = (df["season"].astype("int16") - debut).astype("int16")
    return df


def add_haneul_features(df):
    """제구비율 / 경험치 계열."""
    df = df.copy()
    df["pitcher_control_ratio"] = (
        df["asof_pitcher_ball_rate"] / (df["asof_pitcher_middle_rate"] + 1e-6)
    ).astype("float32")
    df["is_batter_cold_start"] = (df["asof_batter_n"] == 0).astype("int8")
    df["season_progress"] = (df["game_month"] - 3).astype("int8")
    df["is_experienced_mix"] = (df["asof_pitcher_pitchmix_n"] >= 322).astype("int8")
    return df


def add_nawoon_features(df):
    """압박상황 / 매치업 계열."""
    df = df.copy()
    df["is_disadvantaged_count"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_runner_on"] = (df["num_runners_on"] > 0).astype(int)
    df["is_late_inning"] = (df["inning"] >= 7).astype(int)
    df["is_two_outs"] = (df["outs_before"] == 2).astype(int)
    df["pressure_score"] = (
        df["is_disadvantaged_count"] + df["is_runner_on"] + df["is_late_inning"] + df["is_two_outs"]
    )
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["is_cold_start"] = (df["asof_pitcher_n"] == 0).astype(int)
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)
    return df


def add_all_features(df, pressure_lookups, situ_lookup):
    df = add_eunwoo_features(df)
    df = add_haejin_features(df, pressure_lookups)
    df = add_haneul_features(df)
    df = add_nawoon_features(df)
    df = add_yeongeun_features(df, situ_lookup)
    return df


# =======================
# 학습/평가 파이프라인
# =======================

def split_train_early_stop_holdout(df):
    pool = df[df["season"].isin(TRAIN_POOL_SEASONS)]
    holdout_df = df[df["season"] == HOLDOUT_SEASON]
    tail_season_df = pool[pool["season"] == EARLY_STOP_TAIL_SEASON]
    tail_cut = int(len(tail_season_df) * (1 - EARLY_STOP_TAIL_FRAC))
    early_stop_df = tail_season_df.iloc[tail_cut:]
    train_df = pd.concat([
        pool[pool["season"] != EARLY_STOP_TAIL_SEASON],
        tail_season_df.iloc[:tail_cut],
    ])
    return train_df, early_stop_df, holdout_df


def build_xy(df, drop_extra=()):
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL, *DROP_COLS, *drop_extra)]
    return df[feature_cols], df[TARGET_COL]


def brier_skill_score(y_true, p):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    brier = np.mean((p - y_true) ** 2)
    r = y_true.mean()
    baseline_brier = r * (1 - r)
    return max(0.0, 100000 * (1 - brier / baseline_brier))


def run_once(train_df, es_df, holdout_df, seed, drop_extra=()):
    cat_cols = [c for c in CAT_COLS if c not in drop_extra]
    X_train, y_train = build_xy(train_df, drop_extra)
    X_es, y_es = build_xy(es_df, drop_extra)
    X_holdout, y_holdout = build_xy(holdout_df, drop_extra)

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    es_set = lgb.Dataset(X_es, label=y_es, categorical_feature=cat_cols, reference=train_set)
    params = dict(BASE_LGB_PARAMS)
    params.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed})

    model = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[es_set], valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p_holdout = model.predict(X_holdout, num_iteration=model.best_iteration)
    return brier_skill_score(y_holdout, p_holdout), model


def main():
    t0 = time.time()
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")

    pressure_lookups = build_haejin_lookups(df[df["season"].isin(TRAIN_POOL_SEASONS)])
    situ_lookup = build_yeongeun_situ_stats()
    df = add_all_features(df, pressure_lookups, situ_lookup)

    new_cols = [c for c in df.columns if c not in (
        "row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
        "run_total_before", "score_diff_home", "score_diff_pitcher_team",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
        "home_win_expectancy", "away_win_expectancy", "li",
        "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate", "control_success",
    )]
    print(f"신규 피처 {len(new_cols)}개: {new_cols}\n")

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    train_df, es_df, holdout_df = split_train_early_stop_holdout(df)
    print(f"train={len(train_df)}  early_stop={len(es_df)}  holdout={len(holdout_df)}\n")

    print("=== trackman 상황별 집계(시간인지형) 포함 전체 5시드 ===")
    scores = []
    for seed in SEEDS:
        s, model = run_once(train_df, es_df, holdout_df, seed)
        scores.append(s)
        print(f"  seed={seed}  BSS={s:.2f}  best_iter={model.best_iteration}")
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"\nmean={mean:.2f}  std={std:.2f}")
    print(f"(참고: trackman 상황별 집계를 뺀 27개 단계는 mean=683.64 std=8.23이었음)")

    print(f"\n=== 개별 피처 leave-one-out 5시드 ablation ({len(new_cols)}개) ===")
    rows = [("baseline(전체포함)", mean, std)]
    for i, col in enumerate(new_cols, 1):
        col_scores = [run_once(train_df, es_df, holdout_df, s, drop_extra=[col])[0] for s in SEEDS]
        col_mean, col_std = float(np.mean(col_scores)), float(np.std(col_scores))
        delta = col_mean - mean
        print(f"[{i}/{len(new_cols)}] {col:<30} mean={col_mean:>8.2f}  std={col_std:>7.2f}  delta={delta:>+8.2f}  "
              f"({time.time()-t0:.0f}s 경과)")
        rows.append((f"{col} 제거", col_mean, col_std))

    print(f"\n{'설정':<32}{'mean':>10}{'std':>10}{'baseline 대비':>14}")
    for label, row_mean, row_std in rows:
        delta = row_mean - mean if not label.startswith("baseline") else 0.0
        print(f"{label:<32}{row_mean:>10.2f}{row_std:>10.2f}{delta:>+14.2f}")

    print(f"\n총 소요시간: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

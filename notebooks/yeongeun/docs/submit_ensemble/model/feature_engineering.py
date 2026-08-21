"""feature_engineering.py — 팀 최종 확정 11개 피처 + (개인 실험) 트랙맨 situ_stats 6개.

submit.zip 안에 그대로 동봉되는 "단일 소스" 모듈. 학습 스크립트(train_ensemble.py)와
제출용 script.py가 반드시 이 파일 하나만 참조해서, 학습 때 만든 피처와 추론 때 만든 피처가
어긋나는 일(train/inference skew)이 없도록 한다.

⚠️ src/features.py(팀 공식본)와의 차이: 팀 회의에서 "yeongeun 개인의 트랙맨 상황별 집계는
최종안에서 제외"하기로 했었는데, C:\\Aimers 개인 실험에서 이 피처들(situ_stats)이 성능에
크게 기여하는 걸 확인해서(LightGBM 단독 기준 690 -> 730) yeongeun 개인 앙상블에는 다시 추가함.
팀 공식 src/features.py는 건드리지 않음 — 이건 어디까지나 개인 branch 실험.

트랙맨 데이터 사용에 대한 규정 검토: trackman_history.csv는 대회가 공식 제공하는 데이터(README
"제공 데이터 구조" 4개 축 중 하나)라 "외부 데이터 금지" 규정에 안 걸림. situ_stats는 투구 이전에
이미 아는 정보(투수/타자 손, 카운트, 아웃)로 그룹핑해 2019~2024 과거 트랙맨 데이터로만 미리
집계한 조회테이블이라 "행 단위 독립 예측" 원칙도 위반하지 않음 (f_share와 동일한 패턴).
단, 평가 서버 data/ 에는 trackman_history.csv가 없으므로 추론 시점엔 절대 새로 읽지 않고,
학습 때 미리 만든 lookup만 사용해야 함 (자세한 내용: docs/COMPETITION_RULES.md).

lookup(f_share, situ_stats)은 둘 다 test 시점엔 train.csv/trackman_history.csv에 접근할 수
없으므로, 학습 때 build_lookups()로 미리 계산해서 model/f_share_lookup.json 으로 저장해두고
추론 때는 그 파일만 로드해서 쓴다.
"""

import pandas as pd

ID = "row_id"
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
CAT_COLS = BASE_CAT_COLS + ["count_state"]

GLOBAL_MEAN_SUCCESS = 0.55  # 폴백 기본값(build_lookups가 못 넘겨줬을 때만 사용) -- 아래 참고
SMOOTHING_M = 20.0

# 2026-08-20: GLOBAL_MEAN_SUCCESS=0.55 하드코딩이 실제 시즌별 평균(2023=0.500, 2024=0.486로
# 계속 하락 추세)과 안 맞아서, 투구 경험 적은(asof_pitcher_n 작은) 투수의 성공확률을 체계적으로
# 과대예측하는 걸 에러분석으로 발견함(0~5회 구간: 예측 0.479 vs 실제 0.433, +4.6%p 편향).
# build_lookups()가 학습데이터 실제 평균으로 동적 계산해서 넘겨주도록 수정 -- add_features_eunwoo()의
# 인자로 받는다 (모듈 상수는 lookups 없이 단독 호출될 때 대비한 폴백으로만 남겨둠).

# prepare_data.py와 동일한 매핑 — train/test는 숫자코드(1/2), trackman은 문자열(Left/Right)
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}
SITU_KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]
SITU_FEATURES = [
    "tm_fastball_rate", "tm_breaking_rate", "tm_offspeed_rate",
    "tm_zone_speed_mean", "tm_horz_break_mean", "tm_n",
]

# 투수/타자 개인별 트랙맨 물리 특성 집계 (situ_stats와 다른 grain -- 상황별이 아니라 선수 개인별)
PB_TRACKMAN_FEATURES = [
    "tm_p_speed_mean", "tm_p_speed_std", "tm_p_spin_mean", "tm_p_vbreak_mean", "tm_p_hbreak_mean",
    "tm_b_speed_mean", "tm_b_spin_mean", "tm_b_vbreak_mean",
    "speed_diff", "spin_diff", "vbreak_diff",
]


def add_features_eunwoo(df, global_mean_success=GLOBAL_MEAN_SUCCESS):
    df = df.copy()
    df["count_state"] = df["balls_before"].astype(str) + "B_" + df["strikes_before"].astype(str) + "S"
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype("int8")

    df["recent_form_diff"] = (
        df["asof_pitcher_prev1_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)
    ).astype("float32")

    n = df["asof_pitcher_n"].fillna(0)
    rate = df["asof_pitcher_success_rate"].fillna(global_mean_success)
    df["asof_pitcher_success_smoothed"] = (
        (n * rate + SMOOTHING_M * global_mean_success) / (n + SMOOTHING_M)
    ).astype("float32")
    return df


EUNWOO_FEATURES = ["count_state", "is_full_count", "recent_form_diff", "asof_pitcher_success_smoothed"]


def build_situ_stats(trackman_df):
    """트랙맨 상황별(투수손×타자손×볼카운트×아웃) 집계. 2019~2024 과거 데이터로 한 번만 계산."""
    usecols = SITU_KEY + ["pitch_type_group", "zone_speed", "horz_break", "trackman_id"]
    tm = trackman_df[usecols].copy()
    situ = tm.groupby(SITU_KEY, observed=True).agg(
        tm_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
        tm_breaking_rate=("pitch_type_group", lambda s: (s == "breaking").mean()),
        tm_offspeed_rate=("pitch_type_group", lambda s: (s == "offspeed").mean()),
        tm_zone_speed_mean=("zone_speed", "mean"),
        tm_horz_break_mean=("horz_break", "mean"),
        tm_n=("trackman_id", "count"),
    ).reset_index()
    situ["pitcher_hand"] = situ["pitcher_hand"].map(INV_HAND_MAP)
    situ["batter_hand"] = situ["batter_hand"].map(INV_HAND_MAP)
    return situ


def build_pb_trackman_stats(trackman_df):
    """투수/타자 개인별 트랙맨 물리 특성(구속·회전수·무브먼트) 집계. 2019~2024 과거 데이터로 한 번만 계산.
    situ_stats(상황별)와 다른 grain -- 선수 개인 단위. eunwoo의 탐색 노트북(feature_xgboost.py)에서
    후보로 만들었던 것 중 채택."""
    tm_p = trackman_df.groupby("pitcher_trackman_id", observed=True).agg(
        tm_p_speed_mean=("rel_speed", "mean"),
        tm_p_speed_std=("rel_speed", "std"),
        tm_p_spin_mean=("spin_rate", "mean"),
        tm_p_vbreak_mean=("induced_vert_break", "mean"),
        tm_p_hbreak_mean=("horz_break", "mean"),
    ).reset_index().rename(columns={"pitcher_trackman_id": "pitcher_id"})
    tm_b = trackman_df.groupby("batter_trackman_id", observed=True).agg(
        tm_b_speed_mean=("rel_speed", "mean"),
        tm_b_spin_mean=("spin_rate", "mean"),
        tm_b_vbreak_mean=("induced_vert_break", "mean"),
    ).reset_index().rename(columns={"batter_trackman_id": "batter_id"})
    return tm_p, tm_b


def build_lookups(train_full, trackman_df=None):
    """f_share(+situ_stats+pb_trackman) 조회테이블. train/trackman 전체로 한 번만 계산 -> 키 조인만
    하면 됨 (row-independent, 대회 '행 단위 독립 예측' 원칙 준수)."""
    is_f = train_full["game_type"] == "F"
    contaminated = is_f & (train_full["season"] <= 2022)
    f_share_old = (train_full.assign(_f=contaminated.astype("int8"))
                   .groupby("pitcher_id", observed=True)["_f"].mean())
    lookups = {
        "f_share_old": {str(k): v for k, v in f_share_old.to_dict().items()},
        "fallback_f_share": float(f_share_old.median()),
        # 하드코딩 0.55 대신 실제 학습데이터 평균 성공률로 동적 계산 (위 주석 참고)
        "global_mean_success": float(train_full[TARGET].mean()),
    }
    if trackman_df is not None:
        situ_stats = build_situ_stats(trackman_df)
        lookups["situ_stats"] = situ_stats.to_dict(orient="records")
        lookups["situ_fallback"] = {c: float(situ_stats[c].mean()) for c in SITU_FEATURES}

        tm_p, tm_b = build_pb_trackman_stats(trackman_df)
        lookups["tm_pitcher"] = {str(k): v for k, v in
                                  tm_p.set_index("pitcher_id").to_dict(orient="index").items()}
        lookups["tm_batter"] = {str(k): v for k, v in
                                 tm_b.set_index("batter_id").to_dict(orient="index").items()}
        lookups["tm_p_fallback"] = {c: float(tm_p[c].mean()) for c in tm_p.columns if c != "pitcher_id"}
        lookups["tm_b_fallback"] = {c: float(tm_b[c].mean()) for c in tm_b.columns if c != "batter_id"}
    return lookups


def add_features_haejin(df, lookups):
    out = df.copy()
    out["count_pressure"] = (out["balls_before"] - out["strikes_before"]).astype("int8")
    out["count_depth"] = (out["balls_before"] + out["strikes_before"]).astype("int8")
    out["form_dev_3"] = (out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_success_rate"]).astype("float32")
    out["skill_gap"] = (out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]).astype("float32")
    out["residual"] = (
        1.0 - out["asof_pitcher_success_rate"] - out["asof_pitcher_middle_rate"] - out["asof_pitcher_reverse_rate"]
    ).astype("float32")

    f_share_old = lookups["f_share_old"]
    fallback = lookups["fallback_f_share"]
    pid_str = out["pitcher_id"].astype("int64").astype(str)
    out["f_share"] = pid_str.map(f_share_old).fillna(fallback).astype("float32")
    return out


HAEJIN_FEATURES = ["count_pressure", "count_depth", "form_dev_3", "skill_gap", "residual", "f_share"]


def add_hand_match(df):
    df = df.copy()
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype("int8")
    return df


def add_situ_features(df, lookups):
    """situ_stats lookup을 SITU_KEY로 조인. lookups에 situ_stats가 없으면(=팀 공식 모드) 그대로 통과."""
    if "situ_stats" not in lookups:
        return df
    situ_df = pd.DataFrame(lookups["situ_stats"])
    out = df.merge(situ_df, on=SITU_KEY, how="left")
    for c in SITU_FEATURES:
        out[c] = out[c].fillna(lookups["situ_fallback"][c]).astype("float32")
    return out


def add_pb_trackman_features(df, lookups):
    """투수/타자 개인별 트랙맨 lookup을 pitcher_id/batter_id로 조인.
    lookups에 없으면(=팀 공식 모드) 그대로 통과. 2019~2024 트랙맨 기록이 없는 선수(신인 등)는
    tm_p_fallback/tm_b_fallback(리그 평균)으로 대체."""
    if "tm_pitcher" not in lookups:
        return df
    out = df.copy()
    tm_p_map, tm_b_map = lookups["tm_pitcher"], lookups["tm_batter"]
    p_fallback, b_fallback = lookups["tm_p_fallback"], lookups["tm_b_fallback"]

    pid_str = out["pitcher_id"].astype("int64").astype(str)
    bid_str = out["batter_id"].astype("int64").astype(str)
    # 컬럼마다 flat dict를 한 번만 만들어서 map(dict)으로 벡터화 (map(lambda)는 147만행에 느림)
    for c in ["tm_p_speed_mean", "tm_p_speed_std", "tm_p_spin_mean", "tm_p_vbreak_mean", "tm_p_hbreak_mean"]:
        flat = {k: v.get(c) for k, v in tm_p_map.items()}
        out[c] = pid_str.map(flat).fillna(p_fallback[c]).astype("float32")
    for c in ["tm_b_speed_mean", "tm_b_spin_mean", "tm_b_vbreak_mean"]:
        flat = {k: v.get(c) for k, v in tm_b_map.items()}
        out[c] = bid_str.map(flat).fillna(b_fallback[c]).astype("float32")

    out["speed_diff"] = (out["tm_p_speed_mean"] - out["tm_b_speed_mean"]).astype("float32")
    out["spin_diff"] = (out["tm_p_spin_mean"] - out["tm_b_spin_mean"]).astype("float32")
    out["vbreak_diff"] = (out["tm_p_vbreak_mean"] - out["tm_b_vbreak_mean"]).astype("float32")
    return out


NEW_FEATURES = EUNWOO_FEATURES + HAEJIN_FEATURES + ["hand_match"] + SITU_FEATURES + PB_TRACKMAN_FEATURES
TRACKMAN_FEATURES = SITU_FEATURES + PB_TRACKMAN_FEATURES  # include_situ=False일 때 한꺼번에 제외할 목록


def add_all_features(df, lookups):
    """train/test 어디서나 동일하게 쓰는 단일 진입점."""
    df = add_features_eunwoo(df, lookups.get("global_mean_success", GLOBAL_MEAN_SUCCESS))
    df = add_features_haejin(df, lookups)
    df = add_hand_match(df)
    df = add_situ_features(df, lookups)
    df = add_pb_trackman_features(df, lookups)
    return df


def get_all_features(base_columns, include_situ=True):
    """row_id/target을 뺀 base 피처 + 신규 피처 목록.

    ⚠️ include_situ는 반드시 실제 add_all_features()에 트랙맨 lookup(situ_stats/tm_pitcher)이 있는
    lookups를 넘겼는지와 일치시켜야 함 (build_lookups(train, trackman_df=None)이면 이 lookup들이
    안 생겨서 tm_* 컬럼이 df에 없는데, 여기서 True(기본값)로 두면 존재하지 않는 컬럼을 피처 목록에
    넣어 KeyError가 남 -- 실제로 재현해서 확인한 버그, 트랙맨 미사용 경로 쓸 때는 반드시 False로 넘길 것.
    """
    new_features = NEW_FEATURES if include_situ else [c for c in NEW_FEATURES if c not in TRACKMAN_FEATURES]
    base = [c for c in base_columns if c not in (ID, TARGET) and c not in NEW_FEATURES]
    return base + new_features

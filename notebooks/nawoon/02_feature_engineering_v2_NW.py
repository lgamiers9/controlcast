# 14_nawoon_feature_engineering.py
"""nawoon 담당 피처 생성 전용 코드 - 학습/평가는 포함하지 않음.

총 30개 피처:
  A) 행 내부 파생 12개 (매칭/조회 불필요, 그 행 자기 컬럼값만 사용)
     same_hand, pitcher_momentum, pitcher_is_home,
     matchup_success_diff, matchup_middle_diff, pitcher_strike_ball_diff,
     pitcher_risk_diff, pitch_mix_entropy, is_weekend,
     is_disadvantaged_count, pressure_score, is_first_pitch
  B) trackman_history.csv 시간 인지형(time-aware) 집계 18개
     tm_month_*(6) + tm_count_*(6) + tm_inning_*(6)
     - data_description.md의 "평가 시점 이후 정보를 포함하는 방식으로 사용할 수 없다"는
       조항을 지키기 위해, 각 행의 (season, game_month) 시점보다 엄격히 이전(<)의
       trackman 데이터만 반영한 누적평균으로 계산함 (merge_asof 사용).
       자세한 배경은 notes/trackman_시점_규정_이슈.md 참고.

시도했으나 최종안에서 제외한 것:
  - is_cold_start, is_blowout, is_runner_on: gain 0(원본 컬럼과 중복) 확인
  - risp, is_close_late: 단독으론 약하게 기여했지만 함께 있을 때 5시드 표준편차를
    12.04→5.63으로 배 가까이 키움(평균은 노이즈 범위 안) - 안정성 위해 제외
  - team_matchup(투수팀x타자팀 169조합): raw pitcher_id/batter_id와 같은 이유로
    과적합 유발(단일 실행 BSS 514.33으로 급락) - 제외

재현 검증: 이 피처들 그대로 학습해 홀드아웃(2024) BSS=683.02 확인됨 (시간 인지형 적용 전 682.97과 거의 동일 
- 미래 정보 누수가 아니라 진짜 계절/상황 패턴이었다는 뜻).
"""

import os

import pandas as pd
import numpy as np

TRAIN_PATH = "../data/train.csv"
TRACKMAN_PATH = "../data/trackman_history.csv"
TRACKMAN_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "zone_speed"]
CACHE_DIR = "../cache"


# =======================
# A) 행 내부 파생 피처 12개
# =======================

def add_nawoon_row_features(df):
    """자기 행 정보만으로 계산되는 파생 피처 - 매칭/조인 불필요, 규정상 100% 안전."""
    df = df.copy()

    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["pitcher_momentum"] = (
        df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    )
    df["pitcher_is_home"] = (df["top_bottom"] == "B").astype(int)

    df["matchup_success_diff"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["matchup_middle_diff"] = df["asof_pitcher_middle_rate"] - df["asof_batter_middle_rate"]
    df["pitcher_strike_ball_diff"] = df["asof_pitcher_strike_rate"] - df["asof_pitcher_ball_rate"]
    df["pitcher_risk_diff"] = df["asof_pitcher_middle_rate"] - df["asof_pitcher_reverse_rate"]

    # 구종 다양성(엔트로피) - 세 비율이 고르게 섞일수록 값이 크고(예측하기 어려움),
    # 한 구종에 쏠릴수록 0에 가까움. 표본 없는 행(전부 NaN)은 그대로 NaN 유지.
    mix_cols = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    mix = df[mix_cols].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(mix > 0, mix * np.log(mix), 0.0), axis=1)
    entropy[np.isnan(mix).all(axis=1)] = np.nan
    df["pitch_mix_entropy"] = entropy

    df["is_weekend"] = df["game_dayofweek"].isin([5, 6]).astype(int)

    is_disadvantaged_count = (df["balls_before"] > df["strikes_before"]).astype(int)
    is_runner_on = (df["num_runners_on"] > 0).astype(int)
    is_late_inning = (df["inning"] >= 7).astype(int)
    is_two_outs = (df["outs_before"] == 2).astype(int)

    df["is_disadvantaged_count"] = is_disadvantaged_count
    # pressure_score를 구성하는 4개 하위 조건 중 is_runner_on/is_late_inning/is_two_outs는
    # 단독으로는 거의 기여가 없어 별도 컬럼으로 노출하지 않고 pressure_score 계산에만 사용.
    df["pressure_score"] = is_disadvantaged_count + is_runner_on + is_late_inning + is_two_outs
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)
    return df


# =======================
# B) trackman_history.csv 시간 인지형 집계 18개
# =======================

def build_time_aware_lookup(tm, group_cols, metrics):
    """group_cols 조합 x 월단위(_tkey=season*12+game_month)로 합계/개수를 구하고,
    각 그룹 안에서 _tkey 순으로 누적합/누적개수를 매겨 "그 시점까지의 누적평균"을 만든다.
    join_time_aware()에서 merge_asof(direction='backward', allow_exact_matches=False)로
    조인하면 각 행 시점보다 엄격히 이전(<) 데이터만 반영된 평균을 얻는다."""
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
    """df 각 행의 (season, game_month) 시점보다 엄격히 이전까지의 누적평균을 merge_asof로 붙인다."""
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


def build_trackman_lookup(path=TRACKMAN_PATH):
    """trackman_history.csv로 game_month / 카운트 상황 / inning 단위 시간 인지형
    (그 행 시점 이전만 반영) 누적평균 lookup을 만든다.

    "월 버킷"(tm_month_*)은 첫 시즌(2019)이 전부 결측(이전 시즌의 같은 달이 없어서) -
    실측 16.09% 결측, 카운트/inning 버킷은 0.9%대. 버그 아니라 시점을 제대로 지킨 결과.
    """
    tm = pd.read_csv(path, encoding="utf-8-sig")
    month_lookup = build_time_aware_lookup(tm, ["game_month"], TRACKMAN_METRICS)
    count_lookup = build_time_aware_lookup(tm, ["balls_before", "strikes_before"], TRACKMAN_METRICS)
    inning_lookup = build_time_aware_lookup(tm, ["inning"], TRACKMAN_METRICS)
    return month_lookup, count_lookup, inning_lookup


def add_trackman_features(df, month_lookup, count_lookup, inning_lookup):
    df = join_time_aware(df, month_lookup, ["game_month"], "tm_month")
    df = join_time_aware(df, count_lookup, ["balls_before", "strikes_before"], "tm_count")
    df = join_time_aware(df, inning_lookup, ["inning"], "tm_inning")
    return df


# =======================
# 통합
# =======================

def add_nawoon_features(df, trackman_path=TRACKMAN_PATH):
    """nawoon 몫 30개 피처를 전부 적용한다. 다른 스크립트에서 이 함수 하나만 불러 쓰면 됨."""
    df = add_nawoon_row_features(df)
    month_lookup, count_lookup, inning_lookup = build_trackman_lookup(trackman_path)
    df = add_trackman_features(df, month_lookup, count_lookup, inning_lookup)
    return df


NEW_FEATURES = [
    "same_hand", "pitcher_momentum", "pitcher_is_home",
    "matchup_success_diff", "matchup_middle_diff", "pitcher_strike_ball_diff",
    "pitcher_risk_diff", "pitch_mix_entropy", "is_weekend",
    "is_disadvantaged_count", "pressure_score", "is_first_pitch",
] + [f"tm_month_{m}" for m in TRACKMAN_METRICS] \
  + [f"tm_count_{m}" for m in TRACKMAN_METRICS] \
  + [f"tm_inning_{m}" for m in TRACKMAN_METRICS]


# =======================
# main - 캐시 생성용 (실제 학습/평가는 scripts/10_final_model_features_time_aware.py)
# =======================

def main():
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    print("원본:", df.shape)

    df = add_nawoon_features(df)
    print("\n=== 신규 피처 결측률(%) ===")
    print((df[NEW_FEATURES].isna().mean() * 100).round(2))

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(os.path.join(CACHE_DIR, "train_feat_nawoon.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {CACHE_DIR}/train_feat_nawoon.csv  shape={df.shape}")


if __name__ == "__main__":
    main()

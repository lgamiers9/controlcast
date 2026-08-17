"""build_integrated_features.py — 팀원 전체 피처 통합 스크립트

각 팀원의 add_features() 로직을 하나로 합쳐서, "개별 피처 단위"가 아니라
"전체 피처를 합친 상태"에서 모델을 돌려볼 수 있는 데이터셋을 만든다.

원본 노트북(Colab 경로 하드코딩)을 그대로 실행하는 게 아니라, 그 안의
피처 생성 로직만 옮겨온 것 — 팀원 본인 노트북에서 검증한 정의와 반드시
동일하게 유지해야 한다 (바뀌면 본인 노트북에 먼저 반영 후 여기 동기화).

제외한 것:
  - eunwoo: tm_p_*/tm_b_*/speed_diff/spin_diff/vbreak_diff (11개)
    pitcher_id/batter_id 기준으로 trackman을 join하는데, train과 trackman은
    ID 체계가 완전히 다르고 교집합이 0(직접 검증 완료)이라 전부 NaN이 되고
    cold-start fallback으로 전 행이 동일한 상수값이 됨 -> 정보량 0.
    회의 때 eunwoo에게 공유 필요.
  - eunwoo: is_platoon_advantage / nawoon: hand_match
    -> 정의가 완전히 동일(pitcher_hand == batter_hand). hand_match만 유지.
  - eunwoo: is_hitter_advantage / nawoon: is_disadvantaged_count
    -> 정의가 완전히 동일(balls_before > strikes_before). is_hitter_advantage만
    유지하고, nawoon의 pressure_score 합산에도 이 컬럼을 재사용.
  - eunwoo: recent_3g_diff / haejin: form_dev_3
    -> 상관계수 1.0000. 둘 다 prev3_game_success_rate - success_rate 계산인데
    eunwoo만 fillna(0) 적용. NaN을 명시적으로 채우지 않는 게 팀 관례
    (LightGBM/HistGBM은 NaN 네이티브 처리, 명시적 결측 대체가 오히려 악화된
    전례 있음 - CLAUDE.md §6-7)라서 haejin의 form_dev_3만 유지.
  - eunwoo: fastball_ratio/breaking_ratio/offspeed_ratio
    -> 상관계수 1.0000. base의 asof_pitcher_fastball/breaking/offspeed_rate를
    fillna(0)만 한 것이라 새 정보가 전혀 없음. 전부 제외.
  - haneul: season_progress
    -> base의 game_month와 상관계수 1.0000 (season_progress = game_month - 3,
    단순 평행이동이라 트리 모델 기준 완전히 같은 분할 정보). 제외.

실행:
    python build_integrated_features.py   (반드시 이 파일이 있는 폴더에서 실행 — DATA_DIR이 상대경로)

산출물:
    cache/integrated_train.parquet   통합 피처가 전부 붙은 학습 데이터
    cache/feature_manifest.json      피처별 담당자/설명/dtype/결측률 메타정보
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "./data"
CACHE_DIR = "./cache"

ID = "row_id"
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]

KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}


# =======================
# yeongeun (본인) — 상황 기준 트랙맨 집계 + hand_match
# =======================

def build_situ_stats(trackman_path):
    usecols = KEY + ["pitch_type_group", "zone_speed", "horz_break", "trackman_id"]
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig", usecols=usecols)
    situ = tm.groupby(KEY).agg(
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


def add_features_yeongeun(df, situ_stats):
    df = df.merge(situ_stats, on=KEY, how="left")
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


YEONGEUN_FEATURES = [
    "tm_fastball_rate", "tm_breaking_rate", "tm_offspeed_rate",
    "tm_zone_speed_mean", "tm_horz_break_mean", "tm_n", "hand_match",
]


# =======================
# eunwoo — 도메인/카운트/최근폼 피처 (trackman ID join 피처 11개는 제외 — 위 docstring 참고)
# =======================

def add_features_eunwoo(df):
    df = df.copy()
    df["count_state"] = df["balls_before"].astype(str) + "B_" + df["strikes_before"].astype(str) + "S"
    df["is_hitter_advantage"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)

    df["recent_form_diff"] = df["asof_pitcher_prev1_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)

    global_mean_success = 0.55
    m = 20.0
    n = df["asof_pitcher_n"].fillna(0)
    rate = df["asof_pitcher_success_rate"].fillna(global_mean_success)
    df["asof_pitcher_success_smoothed"] = (n * rate + m * global_mean_success) / (n + m)
    return df


EUNWOO_FEATURES = [
    "count_state", "is_hitter_advantage", "is_two_strikes", "is_full_count",
    "recent_form_diff", "asof_pitcher_success_smoothed",
]
EUNWOO_CAT_COLS = ["count_state"]


# =======================
# haejin (HJ) — 행내부 5개 + 조회테이블 2개(f_share, career_span)
# =======================

def build_haejin_lookups(train_full):
    is_f = train_full["game_type"] == "F"
    contaminated = is_f & (train_full["season"] <= 2022)
    f_share_old = (train_full.assign(_f=contaminated.astype("int8"))
                   .groupby("pitcher_id", observed=True)["_f"].mean())
    debut_season = train_full.groupby("pitcher_id", observed=True)["season"].min()
    return {
        "f_share_old": f_share_old.to_dict(),
        "debut_season": debut_season.to_dict(),
        "fallback_f_share": float(f_share_old.median()),
        "fallback_debut": int(debut_season.median()),
    }


def add_features_haejin(df, lookups):
    out = df.copy()
    out["count_pressure"] = (out["balls_before"] - out["strikes_before"]).astype("int8")
    out["count_depth"] = (out["balls_before"] + out["strikes_before"]).astype("int8")
    out["form_dev_3"] = (out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_success_rate"]).astype("float32")
    out["skill_gap"] = (out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]).astype("float32")
    out["residual"] = (1.0 - out["asof_pitcher_success_rate"] - out["asof_pitcher_middle_rate"] - out["asof_pitcher_reverse_rate"]).astype("float32")

    pid = out["pitcher_id"].astype("int64")
    out["f_share"] = (pid.map(lookups["f_share_old"]).fillna(lookups["fallback_f_share"]).astype("float32"))
    debut = pid.map(lookups["debut_season"]).fillna(lookups["fallback_debut"])
    out["career_span"] = (out["season"].astype("int16") - debut).astype("int16")
    return out


HAEJIN_FEATURES = ["count_pressure", "count_depth", "form_dev_3", "skill_gap", "residual", "f_share", "career_span"]


# =======================
# haneul (HN) — 행내부 4개
# =======================

def add_features_haneul(df):
    out = df.copy()
    out["pitcher_control_ratio"] = (out["asof_pitcher_ball_rate"] / (out["asof_pitcher_middle_rate"] + 1e-6)).astype("float32")
    out["is_batter_cold_start"] = (out["asof_batter_n"] == 0).astype("int8")
    out["is_experienced_mix"] = (out["asof_pitcher_pitchmix_n"] >= 322).astype("int8")  # 322: haneul 노트북에서 확정한 train 중앙값 기준, 재계산하지 않고 그대로 고정
    return out


HANEUL_FEATURES = ["pitcher_control_ratio", "is_batter_cold_start", "is_experienced_mix"]


# =======================
# nawoon — 압박상황/매치업/콜드스타트 (hand_match, is_disadvantaged_count는 중복이라 제외 — docstring 참고)
# =======================

def add_features_nawoon(df):
    df = df.copy()
    df["is_runner_on"] = (df["num_runners_on"] > 0).astype(int)
    df["is_late_inning"] = (df["inning"] >= 7).astype(int)
    df["is_two_outs"] = (df["outs_before"] == 2).astype(int)
    # is_hitter_advantage(eunwoo)가 is_disadvantaged_count와 동일 정의라 그걸 재사용
    df["pressure_score"] = (
        df["is_hitter_advantage"] + df["is_runner_on"] + df["is_late_inning"] + df["is_two_outs"]
    )
    df["is_cold_start"] = (df["asof_pitcher_n"] == 0).astype(int)  # 주의: 과거 실험에서 유사한 결측 플래그가 -12.44로 악화된 전례 있음 (CLAUDE.md §6-7) — 중요도/조합 테스트에서 눈여겨볼 것
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)
    return df


NAWOON_FEATURES = ["is_runner_on", "is_late_inning", "is_two_outs", "pressure_score", "is_cold_start", "is_first_pitch"]


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Load raw data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    print(f" train: {train.shape}")

    print("yeongeun: situ_stats + hand_match...")
    situ_stats = build_situ_stats(os.path.join(DATA_DIR, "trackman_history.csv"))
    train = add_features_yeongeun(train, situ_stats)

    print("eunwoo: domain/count/recent-form features...")
    train = add_features_eunwoo(train)

    print("haejin: lookups + features...")
    hj_lookups = build_haejin_lookups(train)
    train = add_features_haejin(train, hj_lookups)

    print("haneul: features...")
    train = add_features_haneul(train)

    print("nawoon: features (pressure_score reuses eunwoo's is_hitter_advantage)...")
    train = add_features_nawoon(train)

    owner_map = {}
    for f in YEONGEUN_FEATURES:
        owner_map[f] = "yeongeun"
    for f in EUNWOO_FEATURES:
        owner_map[f] = "eunwoo"
    for f in HAEJIN_FEATURES:
        owner_map[f] = "haejin"
    for f in HANEUL_FEATURES:
        owner_map[f] = "haneul"
    for f in NAWOON_FEATURES:
        owner_map[f] = "nawoon"

    base_features = [c for c in train.columns if c not in [ID, TARGET] and c not in owner_map]
    all_features = base_features + list(owner_map.keys())
    cat_cols = BASE_CAT_COLS + EUNWOO_CAT_COLS

    print(f"\n총 피처 수: base {len(base_features)} + 팀원 신규 {len(owner_map)} = {len(all_features)}")

    before_mb = train.memory_usage(deep=True).sum() / 1024**2
    for c in train.columns:
        if c in (ID, TARGET) or c in cat_cols:
            continue
        if pd.api.types.is_float_dtype(train[c]):
            train[c] = train[c].astype("float32")
        elif pd.api.types.is_integer_dtype(train[c]):
            train[c] = pd.to_numeric(train[c], downcast="integer")
    after_mb = train.memory_usage(deep=True).sum() / 1024**2
    print(f"다운캐스트: {before_mb:.0f}MB -> {after_mb:.0f}MB")

    print("Save cache...")
    train.to_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"), index=False)

    manifest = {
        "base_features": base_features,
        "new_features": owner_map,
        "all_features": all_features,
        "cat_cols": cat_cols,
        "id": ID,
        "target": TARGET,
    }
    with open(os.path.join(CACHE_DIR, "feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== 담당자별 신규 피처 요약 (결측률 %) ===")
    for owner in ["yeongeun", "eunwoo", "haejin", "haneul", "nawoon"]:
        feats = [f for f, o in owner_map.items() if o == owner]
        print(f"\n[{owner}] {len(feats)}개")
        for f in feats:
            miss = train[f].isna().mean() * 100
            print(f"  {f:30s} dtype={str(train[f].dtype):10s} 결측률={miss:.2f}%")

    print(f"\nDone. shape={train.shape}")
    print(f"저장: {CACHE_DIR}/integrated_train.parquet, {CACHE_DIR}/feature_manifest.json")


if __name__ == "__main__":
    main()

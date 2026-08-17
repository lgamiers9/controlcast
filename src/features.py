"""features.py — 팀 회의에서 최종 확정한 11개 신규 피처만 생성하는 파이프라인.

build_integrated_features.py(notebooks/yeongeun/)는 검증 단계에서 후보로 고려했던 피처를
전부 만드는 탐색용 스크립트였고, 이 스크립트는 회의에서 "살리기로 결정"한 11개만 만든다.

최종 채택 11개 (담당별):
  eunwoo (4): count_state, is_full_count, recent_form_diff, asof_pitcher_success_smoothed
  haejin (6): f_share, residual, form_dev_3, skill_gap, count_pressure, count_depth
  통합   (1): hand_match

빠진 것:
  - yeongeun 본인의 트랙맨 상황별 집계(tm_offspeed_rate 등) 전부 — 그래서 trackman_history.csv를
    아예 안 읽는다 (hand_match는 train/test 자체 컬럼만으로 계산 가능해 트랙맨이 필요 없음)
  - haejin의 career_span — CatBoost 분석에서 season 상관 0.660(교란변수) + train/test 값 범위
    불일치(train 최대 5 → test에 6 등장) 구조적 문제가 지적되어 회의에서 제외
  - haneul·nawoon 피처 전부

실행:
    python src/features.py   (레포 루트에서 실행 — DATA_DIR이 상대경로)

산출물:
    cache/final_train.parquet   base 47 + 신규 11 = 58개 피처가 붙은 학습 데이터
    cache/final_features.json   피처 목록 / 범주형 목록 메타정보
"""

import json
import os

import pandas as pd

DATA_DIR = "./data"
CACHE_DIR = "./cache"

ID = "row_id"
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
CAT_COLS = BASE_CAT_COLS + ["count_state"]


# =======================
# eunwoo — 4개
# =======================

def add_features_eunwoo(df):
    df = df.copy()
    df["count_state"] = df["balls_before"].astype(str) + "B_" + df["strikes_before"].astype(str) + "S"
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype("int8")

    df["recent_form_diff"] = (
        df["asof_pitcher_prev1_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)
    ).astype("float32")

    global_mean_success = 0.55
    m = 20.0
    n = df["asof_pitcher_n"].fillna(0)
    rate = df["asof_pitcher_success_rate"].fillna(global_mean_success)
    df["asof_pitcher_success_smoothed"] = ((n * rate + m * global_mean_success) / (n + m)).astype("float32")
    return df


EUNWOO_FEATURES = ["count_state", "is_full_count", "recent_form_diff", "asof_pitcher_success_smoothed"]


# =======================
# haejin — 6개 (f_share는 조회테이블 필요)
# =======================

def build_haejin_lookups(train_full):
    """f_share 조회테이블. train 전체로 미리 계산 -> pitcher_id 조인만 하면 되므로 row-independent."""
    is_f = train_full["game_type"] == "F"
    contaminated = is_f & (train_full["season"] <= 2022)
    f_share_old = (train_full.assign(_f=contaminated.astype("int8"))
                   .groupby("pitcher_id", observed=True)["_f"].mean())
    return {
        "f_share_old": f_share_old.to_dict(),
        "fallback_f_share": float(f_share_old.median()),
    }


def add_features_haejin(df, lookups):
    out = df.copy()
    out["count_pressure"] = (out["balls_before"] - out["strikes_before"]).astype("int8")
    out["count_depth"] = (out["balls_before"] + out["strikes_before"]).astype("int8")
    out["form_dev_3"] = (out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_success_rate"]).astype("float32")
    out["skill_gap"] = (out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]).astype("float32")
    out["residual"] = (
        1.0 - out["asof_pitcher_success_rate"] - out["asof_pitcher_middle_rate"] - out["asof_pitcher_reverse_rate"]
    ).astype("float32")

    pid = out["pitcher_id"].astype("int64")
    out["f_share"] = pid.map(lookups["f_share_old"]).fillna(lookups["fallback_f_share"]).astype("float32")
    return out


HAEJIN_FEATURES = ["count_pressure", "count_depth", "form_dev_3", "skill_gap", "residual", "f_share"]


# =======================
# 통합 — hand_match (트랙맨 불필요, train/test 자체 컬럼만으로 계산)
# =======================

def add_hand_match(df):
    df = df.copy()
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype("int8")
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Load raw data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    print(f" train: {train.shape}")

    print("eunwoo: 4개...")
    train = add_features_eunwoo(train)

    print("haejin: 6개 (f_share 조회테이블 포함)...")
    lookups = build_haejin_lookups(train)
    train = add_features_haejin(train, lookups)

    print("hand_match...")
    train = add_hand_match(train)

    new_features = EUNWOO_FEATURES + HAEJIN_FEATURES + ["hand_match"]
    base_features = [c for c in train.columns if c not in [ID, TARGET] and c not in new_features]
    all_features = base_features + new_features
    print(f"\n총 피처 수: base {len(base_features)} + 신규 {len(new_features)} = {len(all_features)}")

    downcast_cols = [c for c in train.columns if c not in [ID, TARGET] + CAT_COLS]
    for c in downcast_cols:
        if pd.api.types.is_float_dtype(train[c]):
            train[c] = train[c].astype("float32")
        elif pd.api.types.is_integer_dtype(train[c]):
            train[c] = pd.to_numeric(train[c], downcast="integer")

    print("Save cache...")
    train.to_parquet(os.path.join(CACHE_DIR, "final_train.parquet"), index=False)
    with open(os.path.join(CACHE_DIR, "final_features.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_features": base_features,
            "new_features": new_features,
            "all_features": all_features,
            "cat_cols": CAT_COLS,
            "id": ID,
            "target": TARGET,
        }, f, ensure_ascii=False, indent=2)

    print(f"Done. shape={train.shape}")
    print(f"저장: {CACHE_DIR}/final_train.parquet, {CACHE_DIR}/final_features.json")


if __name__ == "__main__":
    main()

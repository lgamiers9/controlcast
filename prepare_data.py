"""prepare_data.py — 원본 데이터 + trackman 피처엔지니어링을 한 번만 수행하고 캐시로 저장.

이후 모델 실험 스크립트(train_rf_exp.py, train_lgb_exp.py 등)는 이 캐시만 불러오면 되므로,
매번 원본 CSV 로드 + trackman merge(느림)를 반복하지 않아도 된다.
피처 정의를 바꿨을 때만 이 스크립트를 다시 실행하면 된다.

실행:
    python prepare_data.py

산출물:
    cache/train_feat.parquet   - 피처엔지니어링 완료된 학습 데이터 (2019~2024 전체)
    cache/situ_stats.csv       - trackman 상황별 집계 lookup 테이블 (제출용 script.py에서도 재사용)
    cache/features.json        - 피처 목록 / 범주형 목록 등 메타정보
"""

import json
import os
import time

import pandas as pd

DATA_DIR = "./data"
CACHE_DIR = "./cache"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

# 비율 비교로 검증한 매핑: train/test의 숫자코드(1/2) <-> trackman의 문자열(Left/Right)
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}
KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]


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


def add_features(df, situ_stats):
    """train/test/script.py 어디서나 동일하게 재사용하는 피처 엔지니어링."""
    df = df.merge(situ_stats, on=KEY, how="left")
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Load raw data...")
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    base_features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                         usecols=base_features + [TARGET])
    print(f" train: {train.shape}")

    print("Build trackman situ_stats...")
    t0 = time.time()
    situ_stats = build_situ_stats(os.path.join(DATA_DIR, "trackman_history.csv"))
    print(f" situ_stats: {situ_stats.shape} ({time.time() - t0:.1f}s)")

    print("Add features...")
    train_feat = add_features(train, situ_stats)
    features = base_features + [c for c in situ_stats.columns if c not in KEY] + ["hand_match"]

    # 메모리 절약 - float32 다운캐스트 (범주형/타깃 제외)
    downcast_cols = [c for c in train_feat.columns if c not in [ID, TARGET] + CAT_COLS]
    train_feat[downcast_cols] = train_feat[downcast_cols].astype("float32")

    print(f"Save cache -> {CACHE_DIR}/ ...")
    train_feat.to_parquet(os.path.join(CACHE_DIR, "train_feat.parquet"), index=False)
    situ_stats.to_csv(os.path.join(CACHE_DIR, "situ_stats.csv"), index=False)
    with open(os.path.join(CACHE_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_features": base_features,
            "features": features,
            "cat_cols": CAT_COLS,
            "id": ID,
            "target": TARGET,
        }, f, ensure_ascii=False, indent=2)

    print(f"Done. train_feat shape={train_feat.shape}, features={len(features)}")


if __name__ == "__main__":
    main()

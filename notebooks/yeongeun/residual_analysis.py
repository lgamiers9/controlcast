"""residual_analysis.py — HistGBM 예측의 보정(calibration) + 상황별 잔차 패턴 확인.

train_histgbm.py가 저장해둔 val(season 2024) 예측(model/val_pred_meta.parquet)을 그대로 쓴다.
BSS는 "확률 자체가 실제 빈도와 얼마나 맞는가"를 채점하므로, 잔차가 특정 구간에서
체계적으로 몰려있으면(과신/과소신) 그 구간 관련 피처/보정이 부족하다는 신호다.

실행:
    python -u residual_analysis.py
"""

import os

import numpy as np
import pandas as pd

CACHE_DIR = "./cache"
MODEL_DIR = "./model"
TARGET = "control_success"
ID = "row_id"


def reliability_table(y, p, n_bins=10):
    bins = pd.qcut(p, n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), pred_mean=("p", "mean"), actual_mean=("y", "mean"),
    )
    g["gap(pred-actual)"] = g["pred_mean"] - g["actual_mean"]
    return g


def segment_residual(df, col, n_bins=None):
    d = df.copy()
    if n_bins and d[col].nunique() > 12:
        d["_seg"] = pd.qcut(d[col], n_bins, duplicates="drop")
    else:
        d["_seg"] = d[col]
    g = d.groupby("_seg", observed=True).agg(
        n=("residual", "size"), mean_residual=("residual", "mean"), mean_abs_residual=("abs_residual", "mean"),
    ).sort_values("mean_abs_residual", ascending=False)
    return g


def main():
    val = pd.read_parquet(os.path.join(MODEL_DIR, "val_pred_meta.parquet"))
    need_cols = [ID, "balls_before", "strikes_before", "outs_before", "base_state", "inning",
                 "num_runners_on", "game_type", "pitcher_hand", "batter_hand", "count_state",
                 "asof_pitcher_n"]
    side = pd.read_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"), columns=need_cols)
    val = val.merge(side, on=ID, how="left")
    del side
    val["residual"] = val[TARGET] - val["pred"]
    val["abs_residual"] = val["residual"].abs()

    print("=== 1. 보정(calibration) 확인 — 예측확률 10분위별 실제 성공률 ===")
    rel = reliability_table(val[TARGET], val["pred"])
    print(rel.round(4).to_string())
    print(f"\n전체 mean_pred={val['pred'].mean():.4f}  전체 실제 성공률={val[TARGET].mean():.4f}  "
          f"전체 gap={val['pred'].mean()-val[TARGET].mean():+.4f}")

    print("\n=== 2. 카운트(볼-스트라이크) 조합별 잔차 (상위 10, 절대잔차 큰 순) ===")
    print(segment_residual(val, "count_state").head(10).round(4).to_string())

    print("\n=== 3. 주자 상황(base_state)별 잔차 ===")
    print(segment_residual(val, "base_state").round(4).to_string())

    print("\n=== 4. 이닝별 잔차 ===")
    print(segment_residual(val, "inning").round(4).to_string())

    print("\n=== 5. 경기유형(game_type)별 잔차 ===")
    print(segment_residual(val, "game_type").round(4).to_string())

    print("\n=== 6. 콜드스타트(asof_pitcher_n==0) 여부별 잔차 ===")
    val["is_cold"] = (val["asof_pitcher_n"] == 0).astype(int)
    print(segment_residual(val, "is_cold").round(4).to_string())

    val.to_parquet(os.path.join(MODEL_DIR, "val_residuals.parquet"), index=False)
    print(f"\n저장: {MODEL_DIR}/val_residuals.parquet")


if __name__ == "__main__":
    main()

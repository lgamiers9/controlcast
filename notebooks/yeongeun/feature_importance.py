"""feature_importance.py — 통합 76피처 HistGBM의 feature importance 산출 (수동 permutation, 진행상황 실시간 표시).

sklearn의 permutation_importance()는 호출 한 번에 전체 피처를 다 계산해서 중간 진행상황을
볼 수 없다 (그래서 이전 시도가 얼마나 됐는지 알 수 없었음). 여기서는 같은 원리를 피처 하나씩
직접 돌면서, 끝날 때마다 바로 print+flush하고 CSV에 append한다 -> 도중에 죽어도 그때까지
결과는 남아있고, 실시간으로 진행률이 보인다.

HistGradientBoostingClassifier는 LightGBM과 달리 .feature_importances_가 없어서 이 방식이 필요.
단, val 분할이 season==2024 단일 시즌이라 'season'은 val 내에서 상수가 되고, permutation
importance는 상수 컬럼을 셔플해도 값이 안 바뀌므로 항상 중요도 0으로 잘못 측정한다
(CLAUDE.md에 기록된 함정). season의 실제 기여도는 season_ablation.py에서 별도로 검증한다.

속도를 위해 val 25만행 전체가 아니라 6만행 샘플 사용.

실행:
    python -u feature_importance.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

CACHE_DIR = "./cache"
MODEL_DIR = "./model"
VAL_SEASON = 2024
TARGET = "control_success"
N_SAMPLE = 60_000
N_REPEATS = 3
RNG = np.random.default_rng(42)


def main():
    print("Load model + val data...", flush=True)
    model = joblib.load(os.path.join(MODEL_DIR, "histgbm_full.pkl"))
    train = pd.read_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"))
    with open(os.path.join(CACHE_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    features = manifest["all_features"]
    cat_cols = manifest["cat_cols"]
    for c in cat_cols:
        train[c] = train[c].astype("category")

    is_val = train["season"] == VAL_SEASON
    X_va_full, y_va_full = train.loc[is_val, features], train.loc[is_val, TARGET]
    X_va = X_va_full.sample(min(N_SAMPLE, len(X_va_full)), random_state=42)
    y_va = y_va_full.loc[X_va.index]
    print(f" val 전체={len(X_va_full)}, 샘플 사용={len(X_va)}, n_repeats={N_REPEATS}", flush=True)

    baseline_pred = model.predict_proba(X_va)[:, 1]
    baseline_brier = brier_score_loss(y_va, baseline_pred)
    print(f" baseline brier={baseline_brier:.6f}", flush=True)

    out_path = os.path.join(MODEL_DIR, "perm_importance.csv")
    rows = []
    t_start = time.time()
    for i, feat in enumerate(features, 1):
        t1 = time.time()
        drops = []
        for r in range(N_REPEATS):
            X_perm = X_va.copy()
            X_perm[feat] = X_perm[feat].sample(frac=1.0, random_state=1000 + r).values
            pred = model.predict_proba(X_perm)[:, 1]
            brier = brier_score_loss(y_va, pred)
            drops.append(brier - baseline_brier)  # 양수 = 섞었더니 성능이 나빠짐 = 중요한 피처
        imp_mean, imp_std = float(np.mean(drops)), float(np.std(drops))
        rows.append({"feature": feat, "importance_mean": imp_mean, "importance_std": imp_std})
        elapsed = time.time() - t1
        print(f" [{i}/{len(features)}] {feat:32s} imp={imp_mean:+.6f} std={imp_std:.6f}  "
              f"({elapsed:.1f}s, 누적 {time.time()-t_start:.0f}s)", flush=True)
        pd.DataFrame(rows).sort_values("importance_mean", ascending=False).to_csv(out_path, index=False)

    imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    print("\n=== 상위 15 ===")
    print(imp.head(15).to_string(index=False))
    print("\n=== 하위 10 (0에 가까울수록 기여 없음 -- season은 함정이니 무시하고 season_ablation.py로 확인) ===")
    print(imp.tail(10).to_string(index=False))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

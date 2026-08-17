"""season_ablation.py — season을 실제로 빼고 재학습해서 permutation importance의 '0' 함정을 검증.

val 분할이 season==2024라 season은 val 내에서 상수 -> permutation_importance가 항상 0을
반환한다 (feature_importance.py 결과에서 재확인됨). season이 진짜 중요한지는 직접 빼고
재학습해서 val BSS가 얼마나 떨어지는지로만 검증할 수 있다 (CLAUDE.md §6-5와 동일 방법론).

실행:
    python -u season_ablation.py
"""

import json
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import brier_skill_score

CACHE_DIR = "./cache"
MODEL_DIR = "./model"
VAL_SEASON = 2024
TARGET = "control_success"


def main():
    print("Load data...", flush=True)
    train = pd.read_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"))
    with open(os.path.join(CACHE_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    features = [f for f in manifest["all_features"] if f != "season"]
    cat_cols = manifest["cat_cols"]
    for c in cat_cols:
        train[c] = train[c].astype("category")

    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va, y_va = train.loc[is_val, features], train.loc[is_val, TARGET]
    print(f" train={X_tr.shape}, val={X_va.shape} (season 컬럼 제외, {len(features)}개 피처)", flush=True)

    print("Fit HistGradientBoostingClassifier (season 제외)...", flush=True)
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=500, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
    )
    model.fit(X_tr, y_tr)
    print(f" n_iter={model.n_iter_}", flush=True)

    p_va = model.predict_proba(X_va)[:, 1]
    score = brier_skill_score(y_va, p_va)
    print(f"\nseason 제외(75피처) val BSS: {score:.4f}", flush=True)
    print("baseline(76피처, season 포함) val BSS는 train_histgbm.py 결과 참고: 511.3006", flush=True)
    print(f"차이: {score - 511.3006:+.4f}", flush=True)


if __name__ == "__main__":
    main()

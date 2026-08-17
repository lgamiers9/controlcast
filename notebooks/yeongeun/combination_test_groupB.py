"""combination_test_groupB.py — permutation importance가 애매했던(음수/노이즈 수준) 6개 피처를
하나씩 빼고 재학습해서 진짜 효과가 있는지 개별 검증.

대상 (permutation importance 기준, importance_mean이 0 근처거나 음수인데 std는 0이 아닌 -
즉 모델이 가끔 쓰긴 했지만 순효과가 애매한 피처):
  pitcher_control_ratio(haneul), pressure_score(nawoon), recent_form_diff(eunwoo),
  tm_n(yeongeun), skill_gap(haejin), tm_breaking_rate(yeongeun)

baseline(76피처 전체) val BSS = 511.3006 (train_histgbm.py 결과)

피처 하나 끝날 때마다 바로 print+flush하고 CSV에 append -> 노트북을 나중에 다시 열어서
로그만 읽어도 어디까지 됐는지 바로 알 수 있음. 도중에 중단돼도 그때까지 결과는 남아있음.

실행:
    python -u combination_test_groupB.py
"""

import json
import os
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluate import brier_skill_score

CACHE_DIR = "./cache"
MODEL_DIR = "./model"
VAL_SEASON = 2024
TARGET = "control_success"
BASELINE_BSS = 511.3006

CANDIDATES = ["pitcher_control_ratio", "pressure_score", "recent_form_diff", "tm_n", "skill_gap", "tm_breaking_rate"]


def main():
    print("Load data...", flush=True)
    train = pd.read_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"))
    with open(os.path.join(CACHE_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    all_features = manifest["all_features"]
    cat_cols = manifest["cat_cols"]
    for c in cat_cols:
        train[c] = train[c].astype("category")
    is_val = train["season"] == VAL_SEASON

    out_path = os.path.join(MODEL_DIR, "combination_test_groupB.csv")
    rows = []
    t_start = time.time()

    for i, drop_feat in enumerate(CANDIDATES, 1):
        t1 = time.time()
        features = [f for f in all_features if f != drop_feat]
        X_tr = train.loc[~is_val, features]
        y_tr = train.loc[~is_val, TARGET]
        X_va = train.loc[is_val, features]
        y_va = train.loc[is_val, TARGET]

        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=500, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=42,
        )
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        score = brier_skill_score(y_va, p_va)
        delta = score - BASELINE_BSS

        rows.append({"dropped_feature": drop_feat, "val_bss": score, "delta_vs_baseline": delta, "n_iter": model.n_iter_})
        elapsed = time.time() - t1
        verdict = "제외해도 무방 (오히려 개선되거나 변화 미미)" if delta >= -0.5 else "유지 권장 (제외 시 악화)"
        print(f" [{i}/{len(CANDIDATES)}] {drop_feat:24s} 제외 -> BSS={score:.4f}  "
              f"(baseline 대비 {delta:+.4f})  {verdict}  ({elapsed:.0f}s, 누적 {time.time()-t_start:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

        del X_tr, y_tr, X_va, y_va, model

    print(f"\n=== 전체 결과 (baseline={BASELINE_BSS}) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

"""finalize_model.py — 3단계를 한 번에: 61피처 재학습 -> 하이퍼파라미터 랜덤서치 -> isotonic 보정.

데이터를 한 번만 로드해서 세 단계에서 재사용 (메모리/시간 절약, 이 컴퓨터는 RAM 7.72GB뿐이라
반복 로드가 부담됨). 각 단계 끝날 때마다 print+flush로 진행상황이 바로 보이게 함.

61피처 = base 47 + REPORT.md에서 "유지" 결론난 14개
  (asof_pitcher_success_smoothed, tm_offspeed_rate, f_share, hand_match, count_state,
   form_dev_3, career_span, tm_fastball_rate, tm_zone_speed_mean, tm_horz_break_mean,
   residual, is_full_count, is_experienced_mix, recent_form_diff)

검증: 팀 공통 규칙 season==2024 홀드아웃. 보정은 val을 calib/eval 반으로 쪼개서
(CLAUDE.md §6-6 방법론과 동일) 데이터 누수 없이 측정.

실행:
    python -u finalize_model.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from evaluate import brier_skill_score

CACHE_DIR = "./cache"
MODEL_DIR = "./model"
VAL_SEASON = 2024
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
EUNWOO_CAT_COLS = ["count_state"]

KEEP_14 = [
    "asof_pitcher_success_smoothed", "tm_offspeed_rate", "f_share", "hand_match", "count_state",
    "form_dev_3", "career_span", "tm_fastball_rate", "tm_zone_speed_mean", "tm_horz_break_mean",
    "residual", "is_full_count", "is_experienced_mix", "recent_form_diff",
]

N_TRIALS = 15
SEARCH_SPACE = {
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [50, 100, 200, 500],
    "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0],
    "max_bins": [128, 255],
}


def fit_eval(X_tr, y_tr, X_va, y_va, **params):
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=500,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=42, **params,
    )
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    score = brier_skill_score(y_va, p_va)
    return model, score, p_va


def main():
    print("Load data (61피처 = base 47 + 유지 14)...", flush=True)
    with open(os.path.join(CACHE_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    base_features = manifest["base_features"]
    features = base_features + KEEP_14
    cat_cols = BASE_CAT_COLS + EUNWOO_CAT_COLS
    need_cols = list(dict.fromkeys(features + [TARGET, "season"]))

    train = pd.read_parquet(os.path.join(CACHE_DIR, "integrated_train.parquet"), columns=need_cols)
    for c in cat_cols:
        train[c] = train[c].astype("category")
    is_val = train["season"] == VAL_SEASON
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va, y_va = train.loc[is_val, features], train.loc[is_val, TARGET]
    print(f" 피처 수={len(features)}, train={X_tr.shape}, val={X_va.shape}", flush=True)

    # ---- 1단계: 61피처, 현재 하이퍼파라미터로 baseline 재확인 ----
    print("\n=== 1단계: 61피처 baseline 재학습 ===", flush=True)
    t0 = time.time()
    default_params = dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0)
    _, base_score, _ = fit_eval(X_tr, y_tr, X_va, y_va, **default_params)
    print(f" 61피처 baseline val BSS = {base_score:.4f}  (76피처 baseline 511.3006 대비 {base_score-511.3006:+.4f})  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2단계: 하이퍼파라미터 랜덤서치 ----
    print(f"\n=== 2단계: 하이퍼파라미터 랜덤서치 ({N_TRIALS}회) ===", flush=True)
    rng = np.random.RandomState(42)
    trial_rows = []
    best = {"score": base_score, "params": default_params}
    t_search = time.time()
    for i in range(1, N_TRIALS + 1):
        t1 = time.time()
        params = {k: v[rng.randint(len(v))] for k, v in SEARCH_SPACE.items()}
        _, score, _ = fit_eval(X_tr, y_tr, X_va, y_va, **params)
        trial_rows.append({**params, "val_bss": score})
        is_best = score > best["score"]
        if is_best:
            best = {"score": score, "params": params}
        print(f" [{i}/{N_TRIALS}] {params}  BSS={score:.4f}{'  <- 최고' if is_best else ''}  "
              f"({time.time()-t1:.0f}s, 누적 {time.time()-t_search:.0f}s)", flush=True)
        pd.DataFrame(trial_rows).to_csv(os.path.join(MODEL_DIR, "tune_histgbm_61.csv"), index=False)

    print(f"\n최적 하이퍼파라미터: {best['params']}  val BSS={best['score']:.4f}", flush=True)
    with open(os.path.join(MODEL_DIR, "best_params_histgbm_61.json"), "w", encoding="utf-8") as f:
        json.dump({"params": best["params"], "val_bss": best["score"], "features": features, "cat_cols": cat_cols}, f, ensure_ascii=False, indent=2)

    # ---- 3단계: 최적 하이퍼파라미터로 재학습 + isotonic 보정 (val을 calib/eval 반으로 분리, 누수 없이 측정) ----
    print("\n=== 3단계: isotonic 보정 (val을 calib/eval 반으로 분리) ===", flush=True)
    t2 = time.time()
    final_model, uncalibrated_score, p_va = fit_eval(X_tr, y_tr, X_va, y_va, **best["params"])
    print(f" 최종모델(최적 하이퍼파라미터) 재학습 완료, val BSS(보정 전)={uncalibrated_score:.4f}  ({time.time()-t2:.0f}s)", flush=True)

    rng2 = np.random.RandomState(42)
    idx = np.arange(len(y_va))
    rng2.shuffle(idx)
    half = len(idx) // 2
    calib_idx, eval_idx = idx[:half], idx[half:]

    y_va_arr = y_va.to_numpy()
    calib_pred, calib_y = p_va[calib_idx], y_va_arr[calib_idx]
    eval_pred, eval_y = p_va[eval_idx], y_va_arr[eval_idx]

    uncalibrated_eval_bss = brier_skill_score(eval_y, eval_pred)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_pred, calib_y)
    eval_pred_calibrated = iso.predict(eval_pred)
    calibrated_eval_bss = brier_skill_score(eval_y, eval_pred_calibrated)

    print(f"\n eval-half BSS (보정 전): {uncalibrated_eval_bss:.4f}", flush=True)
    print(f" eval-half BSS (보정 후): {calibrated_eval_bss:.4f}  (차이 {calibrated_eval_bss-uncalibrated_eval_bss:+.4f})", flush=True)

    joblib.dump(iso, os.path.join(MODEL_DIR, "calibrator_61.pkl"))
    joblib.dump(final_model, os.path.join(MODEL_DIR, "histgbm_61_tuned.pkl"))

    print("\n=== 최종 요약 ===")
    print(f" 76피처(가지치기 전) baseline val BSS      : 511.3006")
    print(f" 61피처(가지치기 후) baseline val BSS       : {base_score:.4f}")
    print(f" 61피처 + 튜닝 val BSS                     : {best['score']:.4f}")
    print(f" 61피처 + 튜닝 + 보정 eval-half BSS         : {calibrated_eval_bss:.4f}")
    print(f"\n저장: model/best_params_histgbm_61.json, model/tune_histgbm_61.csv, "
          f"model/calibrator_61.pkl, model/histgbm_61_tuned.pkl")


if __name__ == "__main__":
    main()

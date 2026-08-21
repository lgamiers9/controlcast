"""train_ensemble.py — XGBoost + HistGradientBoosting + LightGBM + CatBoost 4-모델 앙상블 학습,
Optuna로 각각 하이퍼파라미터 최적화, 로지스틱회귀 스태킹(또는 가중평균, 더 좋은 쪽 자동 채택)까지
찾아서 실제 제출 zip에 들어갈 model/ 아티팩트를 통째로 생성한다.

전제:
    notebooks/yeongeun/data/train.csv 가 있어야 함.
    docs/submit_ensemble/model/feature_engineering.py 가 "단일 소스" — 여기서 만드는 피처와
    제출용 script.py가 추론 때 만드는 피처가 100% 동일한 코드로 계산되도록 이 모듈만 참조한다.
    (src/features.py와 정의는 같지만, 제출 zip에 src/를 동봉할 수 없어서 별도 복사본으로 관리)

검증 방식: train_xgb.py와 동일하게 시즌 기준 분할(기본 2019~2023 학습 / 2024 검증).

파이프라인:
    1. train.csv 로드 -> feature_engineering으로 58개 피처 생성 (base 47 + 신규 11)
    2. f_share lookup, 카테고리 도메인을 model/ 아티팩트로 저장 (추론 때 train.csv 접근 불가하므로)
    3. XGBoost / HistGradientBoosting / LightGBM 각각 Optuna 탐색
       (학습셋 일부 샘플로 속도 확보, XGB·LGB는 외부 홀드아웃 eval_set으로 early stopping,
        HGB는 내부 early_stopping 사용) -> 검증(2024)-이전 전체로 재학습
    4. 세 모델의 검증(2024) 예측을 3-way 블렌딩 가중치 그리드서치로 최적 결합
    5. 최종 프로덕션 모델은 "전체 데이터(2019~2024)"로 다시 학습해서 model/ 에 저장
       (하이퍼파라미터·블렌딩 가중치는 홀드아웃으로 고정, 실제 제출 모델은 데이터를 최대로 활용)

실행:
    cd notebooks/yeongeun
    ../../.venv/Scripts/python.exe train_ensemble.py --quick      # 동작 확인용
    ../../.venv/Scripts/python.exe train_ensemble.py              # 기본 (trial 상한 80, patience 15로 조기종료)

산출물 (docs/submit_ensemble/model/):
    xgb_model.pkl, hgb_model.pkl, lgb_model.pkl, blend_weight.json,
    f_share_lookup.json, cat_categories.json, feature_manifest.json
    + 콘솔에 세 모델/블렌딩 각각의 Val(검증시즌) BSS 출력
"""

import argparse
import json
import os
import sys
import time

# Windows 콘솔 기본 인코딩(cp949)에서 유니코드 출력 깨짐 방지
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "docs", "submit_ensemble", "model")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, MODEL_DIR)
from src.score import brier_skill_score  # noqa: E402
import feature_engineering as fe  # noqa: E402  (docs/submit_ensemble/model/feature_engineering.py, 단일 소스)

DATA_DIR = "./data"
DEFAULT_VAL_SEASON = 2024


def to_jsonable(x):
    return x.item() if hasattr(x, "item") else x


def load_and_build_features(use_trackman=True):
    print("Load raw train.csv...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    print(f" train_raw: {train_raw.shape}")

    trackman_df = None
    if use_trackman:
        print("Load trackman_history.csv...")
        trackman_df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
        print(f" trackman: {trackman_df.shape}")

    lookups = fe.build_lookups(train_raw, trackman_df)
    full = fe.add_all_features(train_raw, lookups)
    features = fe.get_all_features(train_raw.columns, include_situ=use_trackman)
    print(f" features: base+new = {len(features)}")
    return full, features, lookups


def cast_categories(full, cat_cols):
    categories = {}
    for c in cat_cols:
        full[c] = full[c].astype("category")
        categories[c] = [to_jsonable(v) for v in full[c].cat.categories.tolist()]
    return full, categories


def split_by_season(full, features, val_season):
    X = full[features]
    y = full[fe.TARGET]
    is_val = full["season"] == val_season
    return X[~is_val], y[~is_val], X[is_val], y[is_val]


# =======================
# XGBoost
# =======================

def make_xgb(params, n_estimators, early_stopping_rounds=None, seed=42):
    kwargs = dict(
        n_estimators=n_estimators, tree_method="hist", enable_categorical=True,
        eval_metric="logloss", random_state=seed, n_jobs=-1, **params,
    )
    if early_stopping_rounds:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBClassifier(**kwargs)


def xgb_objective_factory(X_tr, y_tr, X_va, y_va, seed):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = make_xgb(params, n_estimators=2000, early_stopping_rounds=50, seed=seed)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        trial.set_user_attr("best_iteration", model.best_iteration)
        return brier_skill_score(y_va.values, p)
    return objective


# =======================
# HistGradientBoosting
# =======================

def make_hgb(params, max_iter, early_stopping=False, seed=42):
    return HistGradientBoostingClassifier(
        max_iter=max_iter, early_stopping=early_stopping,
        n_iter_no_change=30, validation_fraction=0.1,
        random_state=seed, **params,
    )


def hgb_objective_factory(X_tr, y_tr, X_va, y_va, seed):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
        model = make_hgb(params, max_iter=2000, early_stopping=True, seed=seed)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        trial.set_user_attr("n_iter", int(model.n_iter_))
        return brier_skill_score(y_va.values, p)
    return objective


# =======================
# LightGBM
# =======================

def make_lgb(params, n_estimators, seed=42):
    return lgb.LGBMClassifier(
        boosting_type="gbdt", objective="binary", n_estimators=n_estimators,
        random_state=seed, n_jobs=-1, verbosity=-1, **params,
    )


def lgb_objective_factory(X_tr, y_tr, X_va, y_va, seed):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        model = make_lgb(params, n_estimators=2000, seed=seed)
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p = model.predict_proba(X_va)[:, 1]
        trial.set_user_attr("best_iteration", model.best_iteration_)
        return brier_skill_score(y_va.values, p)
    return objective


# =======================
# CatBoost
# =======================

def make_catboost(params, iterations, early_stopping_rounds=None, seed=42):
    # ⚠️ boosting_type="Plain" 필수 — 기본값 "Ordered"는 대용량(수십만행 이상)에서 훨씬 느림
    # (실제로 재현: 147만행 프로덕션 학습이 Ordered로 1시간 넘게 걸림). CatBoost 공식 문서도
    # 큰 데이터셋엔 Plain을 권장함. 정확도 손해는 미미하고 속도는 몇 배 빨라짐.
    kwargs = dict(
        iterations=iterations, cat_features=fe.CAT_COLS, random_seed=seed,
        verbose=False, thread_count=-1, eval_metric="Logloss", boosting_type="Plain", **params,
    )
    if early_stopping_rounds:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return CatBoostClassifier(**kwargs)


def catboost_objective_factory(X_tr, y_tr, X_va, y_va, seed):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),  # 10까지 열어두면 trial당 시간이 너무 늘어남
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }
        model = make_catboost(params, iterations=1000, early_stopping_rounds=50, seed=seed)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        return brier_skill_score(y_va.values, p)
    return objective


def make_no_improvement_callback(patience):
    """최근 patience번 동안 best_value 갱신이 없으면 탐색을 조기 종료.

    trial 개수를 미리 정확히 못 정해도, 쉬운 탐색은 빨리 끝나고
    어려운 탐색은 (상한 trial 수까지) 알아서 더 오래 도는 방식.
    """
    def callback(study, trial):
        if trial.number - study.best_trial.number >= patience:
            print(f"  -> 최근 {patience}trial 동안 개선 없음, 조기 종료 (trial {trial.number})")
            study.stop()
    return callback


def generate_weight_grid(n_models, step=20):
    """0~1 사이, 합이 1인 가중치 조합을 step 간격(기본 0.05)으로 전부 생성.
    모델 개수(3개 또는 4개)에 상관없이 동작하도록 재귀로 구현 -- CatBoost 있고 없고를
    --no-catboost 로 유연하게 바꿀 수 있어야 해서 하드코딩된 중첩 for문 대신 이 방식을 씀."""
    def rec(remaining, slots):
        if slots == 1:
            yield (remaining,)
            return
        for v in range(remaining + 1):
            for rest in rec(remaining - v, slots - 1):
                yield (v,) + rest
    for combo in rec(step, n_models):
        yield np.array(combo) / step


def run_study(name, objective, max_trials, patience, timeout, seed):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(
        objective, n_trials=max_trials, timeout=timeout, show_progress_bar=True,
        callbacks=[make_no_improvement_callback(patience)],
    )
    print(f"\n[{name}] HPO 완료: {len(study.trials)} trials (상한 {max_trials}, patience {patience}), {time.time() - t0:.1f}s")
    print(f"[{name}] Best BSS (HPO 샘플 학습 기준): {study.best_value:.2f}")
    print(f"[{name}] Best params: {study.best_params}")
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xgb-trials", type=int, default=80, help="모델당 trial 상한(cap) — patience로 조기 종료 안 되면 이 값까지 돔")
    ap.add_argument("--hgb-trials", type=int, default=80)
    ap.add_argument("--lgb-trials", type=int, default=80)
    ap.add_argument("--cat-trials", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15, help="최근 N trial 동안 개선 없으면 조기 종료")
    ap.add_argument("--timeout", type=int, default=None, help="모델당 초 단위 HPO 제한시간")
    ap.add_argument("--hpo-sample-size", type=int, default=200_000)
    ap.add_argument("--val-season", type=int, default=DEFAULT_VAL_SEASON)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="동작 확인용: trial 상한 3개, 샘플 20,000개")
    ap.add_argument("--no-trackman", action="store_true",
                     help="트랙맨 situ_stats 6개 피처 제외 (팀 공식 58개 피처만 사용, 이전 방식과 비교용)")
    ap.add_argument("--no-catboost", action="store_true",
                     help="CatBoost 제외 (GPU 없이 로컬 CPU로만 돌릴 때 — CatBoost가 CPU에서 너무 느려서)")
    args = ap.parse_args()
    if args.quick:
        args.xgb_trials = args.hgb_trials = args.lgb_trials = args.cat_trials = 3
        args.patience = 3
        args.hpo_sample_size = 20_000

    os.makedirs(MODEL_DIR, exist_ok=True)

    full, features, lookups = load_and_build_features(use_trackman=not args.no_trackman)
    full, categories = cast_categories(full, fe.CAT_COLS)

    X_tr_full, y_tr_full, X_va, y_va = split_by_season(full, features, args.val_season)
    print(f"train(<{args.val_season})={len(X_tr_full):,}  val({args.val_season})={len(X_va):,}  features={len(features)}")

    if len(X_tr_full) > args.hpo_sample_size:
        idx = X_tr_full.sample(args.hpo_sample_size, random_state=args.seed).index
        X_tr_hpo, y_tr_hpo = X_tr_full.loc[idx], y_tr_full.loc[idx]
    else:
        X_tr_hpo, y_tr_hpo = X_tr_full, y_tr_full
    print(f"HPO 학습 샘플: {len(X_tr_hpo):,}")

    # ---- 1) XGBoost HPO ----
    xgb_study = run_study(
        "XGB", xgb_objective_factory(X_tr_hpo, y_tr_hpo, X_va, y_va, args.seed),
        args.xgb_trials, args.patience, args.timeout, args.seed,
    )
    xgb_iter = xgb_study.best_trial.user_attrs.get("best_iteration") or 500
    xgb_holdout = make_xgb(xgb_study.best_params, n_estimators=int(xgb_iter * 1.2) + 1,
                            early_stopping_rounds=50, seed=args.seed)
    xgb_holdout.fit(X_tr_full, y_tr_full, eval_set=[(X_va, y_va)], verbose=False)
    xgb_final_iter = xgb_holdout.best_iteration
    p_xgb_va = xgb_holdout.predict_proba(X_va)[:, 1]
    bss_xgb = brier_skill_score(y_va.values, p_xgb_va)
    print(f"\n[XGB] 홀드아웃 재학습 Val({args.val_season}) BSS: {bss_xgb:.2f}  (n_estimators={xgb_final_iter})")

    # ---- 2) HistGradientBoosting HPO ----
    hgb_study = run_study(
        "HGB", hgb_objective_factory(X_tr_hpo, y_tr_hpo, X_va, y_va, args.seed),
        args.hgb_trials, args.patience, args.timeout, args.seed,
    )
    hgb_iter = hgb_study.best_trial.user_attrs.get("n_iter") or 300
    hgb_holdout = make_hgb(hgb_study.best_params, max_iter=hgb_iter, early_stopping=False, seed=args.seed)
    hgb_holdout.fit(X_tr_full, y_tr_full)
    p_hgb_va = hgb_holdout.predict_proba(X_va)[:, 1]
    bss_hgb = brier_skill_score(y_va.values, p_hgb_va)
    print(f"\n[HGB] 홀드아웃 재학습 Val({args.val_season}) BSS: {bss_hgb:.2f}  (max_iter={hgb_iter})")

    # ---- 3) LightGBM HPO ----
    lgb_study = run_study(
        "LGB", lgb_objective_factory(X_tr_hpo, y_tr_hpo, X_va, y_va, args.seed),
        args.lgb_trials, args.patience, args.timeout, args.seed,
    )
    lgb_iter = lgb_study.best_trial.user_attrs.get("best_iteration") or 500
    lgb_holdout = make_lgb(lgb_study.best_params, n_estimators=int(lgb_iter * 1.2) + 1, seed=args.seed)
    lgb_holdout.fit(
        X_tr_full, y_tr_full, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_final_iter = lgb_holdout.best_iteration_
    p_lgb_va = lgb_holdout.predict_proba(X_va)[:, 1]
    bss_lgb = brier_skill_score(y_va.values, p_lgb_va)
    print(f"\n[LGB] 홀드아웃 재학습 Val({args.val_season}) BSS: {bss_lgb:.2f}  (n_estimators={lgb_final_iter})")

    # ---- 4) CatBoost HPO (--no-catboost면 스킵, GPU 없는 로컬 CPU에서 너무 느려서) ----
    if not args.no_catboost:
        cat_study = run_study(
            "CAT", catboost_objective_factory(X_tr_hpo, y_tr_hpo, X_va, y_va, args.seed),
            args.cat_trials, args.patience, args.timeout, args.seed,
        )
        cat_iter = cat_study.best_trial.user_attrs.get("best_iteration") or 500
        cat_holdout = make_catboost(cat_study.best_params, iterations=int(cat_iter * 1.2) + 1,
                                     early_stopping_rounds=50, seed=args.seed)
        cat_holdout.fit(X_tr_full, y_tr_full, eval_set=(X_va, y_va), verbose=False)
        cat_final_iter = cat_holdout.get_best_iteration()
        p_cat_va = cat_holdout.predict_proba(X_va)[:, 1]
        bss_cat = brier_skill_score(y_va.values, p_cat_va)
        print(f"\n[CAT] 홀드아웃 재학습 Val({args.val_season}) BSS: {bss_cat:.2f}  (iterations={cat_final_iter})")
    else:
        print("\n[CAT] --no-catboost 지정됨, CatBoost 스킵 (3-way 앙상블: XGB+HGB+LGB)")
        cat_study = cat_final_iter = p_cat_va = bss_cat = None

    # ---- 5) 스태킹 (로지스틱회귀 메타모델) — 검증셋을 절반으로 나눠서 (가중치 학습용 / 최종 평가용)
    #    같은 데이터로 가중치를 찾고 그 데이터로 다시 채점하면 낙관적으로 부풀려짐(4-way 블렌딩 때
    #    겪은 것과 같은 종류의 실수) -> 분리해서 정직하게 평가함.
    print("\n스태킹 메타모델 학습 중 (검증셋을 fit/eval로 분리)...")
    model_names = ["xgb", "hgb", "lgb"] + ([] if args.no_catboost else ["cat"])
    p_list = [p_xgb_va, p_hgb_va, p_lgb_va] + ([] if args.no_catboost else [p_cat_va])
    n_models = len(model_names)

    rng_split = np.random.default_rng(args.seed)
    va_perm = rng_split.permutation(len(y_va))
    half = len(va_perm) // 2
    fit_idx, eval_idx = va_perm[:half], va_perm[half:]

    P_va = np.column_stack(p_list)
    y_va_arr = y_va.values

    stacker = LogisticRegression()
    stacker.fit(P_va[fit_idx], y_va_arr[fit_idx])
    p_stack_eval = stacker.predict_proba(P_va[eval_idx])[:, 1]
    bss_stack = brier_skill_score(y_va_arr[eval_idx], p_stack_eval)

    # 비교용: 같은 fit/eval 분리로 단순 가중평균 그리드서치도 돌려서 공정 비교
    best = None
    for w in generate_weight_grid(n_models, step=20):
        score = brier_skill_score(y_va_arr[fit_idx], P_va[fit_idx] @ w)
        if best is None or score > best[1]:
            best = (w, score)
    best_w, _ = best
    bss_blend = brier_skill_score(y_va_arr[eval_idx], P_va[eval_idx] @ best_w)

    print(f"[Stack] 로지스틱회귀 메타모델 BSS (별도 eval셋): {bss_stack:.2f}")
    weight_str = "/".join(f"{n}{w:.2f}" for n, w in zip(model_names, best_w))
    print(f"[Blend] 가중평균 그리드서치 BSS (같은 eval셋): {bss_blend:.2f}  weights={weight_str}")
    use_stack = bss_stack >= bss_blend
    combine_method = "stack" if use_stack else "blend"
    final_bss_estimate = bss_stack if use_stack else bss_blend
    print(f"-> {'스태킹' if use_stack else '가중평균'} 채택 (BSS {final_bss_estimate:.2f})")
    bss_str = " / ".join(f"{n.upper()} {b:.2f}" for n, b in
                          zip(model_names, [bss_xgb, bss_hgb, bss_lgb] + ([] if args.no_catboost else [bss_cat])))
    print(f"  (참고: {bss_str})")

    # 실제 배포용 스태킹 메타모델은 검증셋 전체(fit+eval)로 다시 학습 (가중치 탐색은 이미 분리해서
    # 검증했으니, 최종 메타모델은 데이터를 최대로 활용)
    final_stacker = LogisticRegression()
    final_stacker.fit(P_va, y_va_arr)

    # ---- 6) 최종 프로덕션 모델: 전체 데이터(검증 시즌 포함)로 재학습 ----
    print("\n전체 데이터(모든 시즌)로 최종 프로덕션 모델 재학습...")
    X_all, y_all = full[features], full[fe.TARGET]
    xgb_prod = make_xgb(xgb_study.best_params, n_estimators=xgb_final_iter, seed=args.seed)
    xgb_prod.fit(X_all, y_all, verbose=False)
    hgb_prod = make_hgb(hgb_study.best_params, max_iter=hgb_iter, early_stopping=False, seed=args.seed)
    hgb_prod.fit(X_all, y_all)
    lgb_prod = make_lgb(lgb_study.best_params, n_estimators=lgb_final_iter, seed=args.seed)
    lgb_prod.fit(X_all, y_all)
    if not args.no_catboost:
        cat_prod = make_catboost(cat_study.best_params, iterations=cat_final_iter, seed=args.seed)
        cat_prod.fit(X_all, y_all, verbose=False)

    # ---- 저장 ----
    # ⚠️ LightGBM만 joblib 대신 자체 포맷(save_model)으로 저장한다.
    #    lightgbm==4.7.0 + joblib 조합이 Windows에서 카테고리형 컬럼을 쓴 모델을 pickle 후
    #    재로드하면 부스터가 깨지는 버그가 있음(access violation) — 실제로 재현해서 확인함.
    #    네이티브 저장(model.booster_.save_model())은 문제없이 동작해서 이 방식으로 통일.
    joblib.dump(xgb_prod, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(hgb_prod, os.path.join(MODEL_DIR, "hgb_model.pkl"))
    lgb_prod.booster_.save_model(os.path.join(MODEL_DIR, "lgb_model.txt"))
    if not args.no_catboost:
        cat_prod.save_model(os.path.join(MODEL_DIR, "cat_model.cbm"))  # CatBoost도 자체 포맷으로 저장
    else:
        # 이전 실행에서 남은 cat_model.cbm이 있으면 지금 앙상블(3-way)과 안 맞으니 제거
        stale = os.path.join(MODEL_DIR, "cat_model.cbm")
        if os.path.exists(stale):
            os.remove(stale)
    joblib.dump(final_stacker, os.path.join(MODEL_DIR, "stacker.pkl"))  # LogisticRegression, joblib 안전

    blend_weights = {f"w_{n}": float(w) for n, w in zip(model_names, best_w)}
    with open(os.path.join(MODEL_DIR, "blend_weight.json"), "w", encoding="utf-8") as f:
        json.dump({
            "combine_method": combine_method,  # "stack" 또는 "blend" -- script.py가 이거 보고 분기
            "models_used": model_names,        # script.py가 이거 보고 몇 개 모델 로드할지 결정
            "blend_weights": blend_weights,
            "val_season": args.val_season,
            "val_bss_xgb": bss_xgb, "val_bss_hgb": bss_hgb, "val_bss_lgb": bss_lgb, "val_bss_cat": bss_cat,
            "val_bss_stack": bss_stack, "val_bss_blend": bss_blend,
            "xgb_best_params": xgb_study.best_params, "xgb_n_estimators": xgb_final_iter,
            "hgb_best_params": hgb_study.best_params, "hgb_max_iter": hgb_iter,
            "lgb_best_params": lgb_study.best_params, "lgb_n_estimators": lgb_final_iter,
            "cat_best_params": (cat_study.best_params if not args.no_catboost else None),
            "cat_iterations": cat_final_iter,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODEL_DIR, "f_share_lookup.json"), "w", encoding="utf-8") as f:
        json.dump(lookups, f, ensure_ascii=False)

    with open(os.path.join(MODEL_DIR, "cat_categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False)

    with open(os.path.join(MODEL_DIR, "feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"all_features": features, "cat_cols": fe.CAT_COLS, "id": fe.ID, "target": fe.TARGET},
                   f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {MODEL_DIR}/ 에 xgb_model.pkl, hgb_model.pkl, lgb_model.pkl, cat_model.cbm, "
          f"stacker.pkl, blend_weight.json, f_share_lookup.json, cat_categories.json, feature_manifest.json")
    print(f"기대 성능 (홀드아웃 Val {args.val_season} 기준, 실제 제출 모델은 전체 데이터로 재학습됨): "
          f"BSS ≈ {final_bss_estimate:.2f}  (combine_method={combine_method})")


if __name__ == "__main__":
    main()

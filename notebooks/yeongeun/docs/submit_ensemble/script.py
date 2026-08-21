"""script.py — 앙상블(XGBoost + HistGradientBoosting + LightGBM [+ CatBoost]) 제출용 추론 스크립트.

model/ 안의 아티팩트:
    feature_engineering.py  — 피처 생성 로직 (train_ensemble.py와 동일 소스, train/inference skew 방지)
    f_share_lookup.json     — pitcher_id -> f_share(+situ_stats+개인별 트랙맨) 조회테이블. 학습 시
                               train 전체로 미리 계산해둔 것 — 추론 시점엔 train.csv/trackman_history.csv에
                               접근할 수 없으므로 반드시 이 파일을 써야 함
    cat_categories.json     — 범주형 컬럼별 카테고리 도메인 (학습 시점 기준으로 고정)
    feature_manifest.json   — 최종 피처 목록 / 범주형 목록 / ID·타깃 컬럼명
    xgb_model.pkl, hgb_model.pkl, lgb_model.txt, (있으면) cat_model.cbm — 학습된 모델들
    stacker.pkl              — 로지스틱회귀 메타모델 (combine_method="stack"일 때 사용)
    blend_weight.json        — combine_method("stack"|"blend") + models_used(실제 쓰인 모델 목록)
                               + weights(가중평균용, 비교/백업용)

⚠️ models_used에 "cat"이 없으면 cat_model.cbm을 아예 안 읽음 — train_ensemble.py를
--no-catboost로 돌렸을 때(3-way 앙상블)와 그대로 돌렸을 때(4-way) 둘 다 이 스크립트 하나로 대응.

data/ (평가 서버가 자동 마운트, 읽기 전용): test.csv, sample_submission.csv
output/submission.csv 로 결과 저장 (평가 서버 규칙, 파일명 고정)

실행 (평가 서버가 자동으로 호출):
    python script.py
"""

import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_DIR = "./model"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

sys.path.insert(0, MODEL_DIR)
import feature_engineering as fe  # noqa: E402  (model/feature_engineering.py)


def load_json(name):
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Load model artifacts...")
    lookups = load_json("f_share_lookup.json")
    categories = load_json("cat_categories.json")
    manifest = load_json("feature_manifest.json")
    combine = load_json("blend_weight.json")
    models_used = combine.get("models_used", ["xgb", "hgb", "lgb", "cat"])
    print(f" models_used: {models_used}  combine_method: {combine['combine_method']}")

    loaded = {}
    loaded["xgb"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    loaded["hgb"] = joblib.load(os.path.join(MODEL_DIR, "hgb_model.pkl"))
    # LightGBM만 네이티브 포맷으로 로드 (joblib pickle이 Windows에서 카테고리형 모델을 깨뜨리는
    # lightgbm==4.7.0 버그 회피 — train_ensemble.py 저장부 주석 참고)
    loaded["lgb"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_model.txt"))
    if "cat" in models_used:
        from catboost import CatBoostClassifier
        cat_model = CatBoostClassifier()
        cat_model.load_model(os.path.join(MODEL_DIR, "cat_model.cbm"))
        loaded["cat"] = cat_model

    print("Load test data...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    feat = fe.add_all_features(test, lookups)
    features = manifest["all_features"]
    for c in manifest["cat_cols"]:
        feat[c] = pd.Categorical(feat[c], categories=categories[c])
    X = feat[features]

    print("Inference...")
    preds = {}
    for name in models_used:
        model = loaded[name]
        if name == "lgb":
            preds[name] = model.predict(X)  # native Booster.predict: binary -> 양성 클래스 확률 그대로 반환
        else:
            preds[name] = model.predict_proba(X)[:, 1]
    P = np.column_stack([preds[n] for n in models_used])

    if combine["combine_method"] == "stack":
        stacker = joblib.load(os.path.join(MODEL_DIR, "stacker.pkl"))
        p = stacker.predict_proba(P)[:, 1]
    else:
        w = np.array([combine["blend_weights"][f"w_{n}"] for n in models_used])
        p = P @ w

    id_col, target_col = manifest["id"], manifest["target"]
    pred_map = dict(zip(test[id_col], p))
    values, n_missing = [], 0
    for rid, cur in zip(sub[id_col], sub[target_col]):
        v = pred_map.get(rid)
        if v is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(v)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[target_col] = values

    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} (rows={len(sub)})")


if __name__ == "__main__":
    main()

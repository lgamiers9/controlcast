# 🌙 작업 요약 (2026-08-18 저녁 ~ 08-19)

> ✅ **최종 결과: BSS 820.84** (XGB+HGB+LGB+CatBoost 4-모델 + 로지스틱회귀 스태킹)
> 제출용 zip 최종 검증 통과 완료. `docs/submit_ensemble.zip` 바로 제출 가능.

---

## 🏆 최종 스코어보드

| 단계 | 구성 | Val(2024) BSS |
|---|---|---|
| 1차 | XGB+HGB+LGB, 트랙맨 미포함(58개 피처), 가중평균 블렌딩 | 707.13 |
| 2차 | + 트랙맨 situ_stats 6개(64개 피처), 가중평균 블렌딩 | 726.57 |
| 참고 | `C:\Aimers`(개인 이전 작업) — LGB+XGB 가중평균 | 736.85 |
| 3차 | + 투수/타자 개인별 트랙맨 11개(75개 피처) + CatBoost + 스태킹 | **820.84** ⭐ |

**820.84는 지금까지 나온 모든 기록(726.57, 736.85 포함)을 크게 앞선 최종 결과**예요. 수료
기준선(549.51) 대비도 압도적으로 여유 있음.

### 3차(최종) 개별 모델 성능
| 모델 | Val(2024) BSS |
|---|---|
| XGBoost 단독 | 682.21 |
| HistGradientBoosting 단독 | 713.84 |
| LightGBM 단독 | 726.57 |
| **CatBoost 단독** | **773.36** ← 새로 추가한 모델, 제일 강력했음 |
| 가중평균 그리드서치 (참고용, CatBoost 100%가 최적) | 804.53 |
| **로지스틱회귀 스태킹 (채택)** | **820.84** |

CatBoost가 개별 모델 중 가장 강력했고(773.36), 단순 가중평균(804.53)보다 **스태킹이 한 번 더
끌어올림**(820.84). 검증셋을 절반으로 나눠서(가중치 학습용/최종 평가용) 공정하게 평가한 결과라
신뢰할 수 있는 수치.

---

## 📈 무엇이 이 결과를 만들었나

1. **투수/타자 개인별 트랙맨 피처 11개 추가** (`tm_p_speed_mean`, `tm_p_spin_mean`,
   `tm_p_vbreak_mean`, `tm_p_hbreak_mean`, `tm_b_*` 3개, `speed_diff`, `spin_diff`,
   `vbreak_diff` 등) — eunwoo의 탐색 노트북에 있던 후보를 정식 채택. 2019~2024 트랙맨
   과거 데이터로 선수별 평균 구속·회전수·무브먼트를 집계해서 pitcher_id/batter_id로 조인.
2. **CatBoost를 4번째 모델로 추가** — 이 데이터셋에서 유독 강력했음(단독 773.36으로 이미
   이전 최고기록들을 앞섬).
3. **스태킹**(로지스틱회귀 메타모델)으로 조합 — 단순 가중평균보다 항상 같거나 나은 성능
   (안 좋으면 자동으로 가중평균 채택하도록 설계해둠).

---

## 🐛 오늘 잡은 버그/이슈들

1. **LightGBM + joblib pickle이 Windows에서 카테고리형 모델을 깨뜨림** (access violation)
   → LightGBM만 자체 포맷(`booster_.save_model()`)으로 저장.
2. **`feature_engineering.py`가 트랙맨 미사용 모드(`include_situ=False`)에서 KeyError 나는 버그**
   → `get_all_features()`에 `include_situ` 플래그 추가해서 트랙맨 컬럼 목록을 조건부로 뺌.
3. **CatBoost가 CPU에서 기본 `boosting_type="Ordered"`로 147만행 학습 시 1시간 넘게 걸림**
   → `boosting_type="Plain"`으로 변경, `depth` 탐색범위 3~10 → 3~8로 축소.
   **129초(20만행) / 트라이얼당 최대 수십초로 단축** — 이게 이번 최종 결과를 가능하게 한
   핵심 수정이었음.
4. **TabNet 관련 버그 4개** (torch/pandas import 순서, season 스케일링 후 분할 버그, epoch당
   전체 검증셋 평가로 인한 병목, best_n_epochs 변수명 불일치) — 아래 TabNet/LSTM 섹션 참고.
5. **4-way 블렌딩 방법론 오류** (프로덕션 모델을 그 모델이 이미 학습한 데이터로 재평가) —
   발견 즉시 수정. 이번 3차 결과는 홀드아웃 모델로만 평가해서 이 문제 없음.

---

## 🧠 시도했지만 채택 안 한 것들 (참고용, 제출 zip엔 안 들어감)

| 실험 | Val(2024) BSS | 비고 |
|---|---|---|
| TabNet (로컬 CPU, 트랙맨 미포함) | 303.07 | |
| TabNet (Colab GPU, 트랙맨 포함) | 429.8 | |
| 투구 시퀀스 LSTM (Colab GPU) | 554.19 | ⚠️ 실제 제출하려면 재설계 필요(아래 참고) |

모두 트리 앙상블보다 낮아서 미채택. 표 데이터에서 트리 계열이 강세인 흔한 패턴.

**시퀀스 LSTM 규정 주의사항**: 지금 버전은 "같은 투수의 직전 투구"를 test.csv 안의 다른 행에서도
가져올 수 있게 설계되어 있어서, 실제 제출하려면 "2024년 말 기준으로 고정한 lookup"만 쓰도록
재설계해야 함 (다른 test 행을 참조하면 안 된다는 규정 때문). 지금은 학습/검증용으로만 유효.

---

## 📁 산출물 위치

```
notebooks/yeongeun/
├── train_ensemble.py              # 4-모델(XGB/HGB/LGB/CatBoost) + 스태킹 학습 스크립트
├── train_ensemble_v3_full_run.log # 최종(BSS 820.84) 학습 로그
├── train_dl.py, colab_train_dl_gpu.ipynb          # TabNet (로컬/Colab GPU)
├── colab_train_seq_gpu.ipynb                       # 투구 시퀀스 LSTM (Colab GPU)
├── colab_train_ensemble_gpu.ipynb                  # 앙상블 GPU 버전 (GPU 할당량 문제로 미사용,
│                                                      다음에 GPU 여유 있을 때 참고용으로 남겨둠)
├── build_colab_*.py                                # 각 노트북 생성 스크립트 (재생성용)
└── docs/
    ├── COMPETITION_RULES.md
    ├── check_submission.py        # 제출 전 자동 검사 (구조/용량/인터넷접근/누수 스캔/실제 실행)
    ├── baseline_submit/           # 공식 baseline
    ├── submit_ensemble.zip        # ✅ 최종 제출용 (BSS 820.84), 검증 통과 완료
    └── submit_ensemble/
        ├── script.py               # 4개 모델 동적 로드 + 스태킹/블렌딩 자동 분기
        ├── requirements.txt        # xgboost, lightgbm, catboost
        └── model/
            ├── xgb_model.pkl, hgb_model.pkl, lgb_model.txt, cat_model.cbm  # 4개 프로덕션 모델
            ├── stacker.pkl                          # 로지스틱회귀 메타모델
            ├── blend_weight.json                    # combine_method, models_used, 각종 점수 기록
            ├── f_share_lookup.json                  # f_share + situ_stats + 개인별 트랙맨 lookup
            ├── cat_categories.json, feature_manifest.json
            └── feature_engineering.py                # 학습/추론 단일 소스 (75개 피처)
```

---

## 🌅 다음에 할 일 (제안)

1. **제출** — `docs/submit_ensemble.zip` 그대로 제출 가능한 상태.
2. **GPU 앙상블 재시도** (선택) — `colab_train_ensemble_gpu.ipynb` 있음. GPU 할당량 풀리면
   trial 수를 더 늘려서(지금은 CatBoost 60개로 제한했음) 추가 개선 여지 확인 가능.
3. **시퀀스 LSTM 규정 맞게 재설계 후 4-way 블렌딩 재시도** — CatBoost가 이미 워낙 강력해서
   추가 이득은 크지 않을 수 있지만, 완전히 다른 종류의 신호라 시도해볼 가치는 있음.
4. **Public LB 점수로 로컬 홀드아웃 추정치(820.84) 검증** — 실제 제출해서 확인 권장.

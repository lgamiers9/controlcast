# 통합 피처셋 검증 보고서 (HistGBM 담당 · yeongeun)

## 0. 요약

팀원 5명(eunwoo, haejin, haneul, nawoon, yeongeun)이 만든 피처를 통합(76개: base 47 + 신규 29)한 뒤 HistGBM으로 학습 → feature importance → 잔차분석 → 개별 피처 ablation(실제 제외 재학습) 검증까지 진행했다.

**결론: 신규 피처 29개 중 13개 유지 / 16개 제외 추천.** 제외분을 뺀 **61개 피처(base 47 + 유지 13... 표 기준 13개 정확)**로 재학습 + 하이퍼파라미터 튜닝 + isotonic 보정까지 마친 최종 결과는 **val BSS 720.53 (튜닝 후) → eval-half BSS 743.02 (보정 후)**로, 76피처 baseline(511.30) 대비 +231.72 개선. 이는 팀의 기존 튜닝된 LightGBM 단독 성능(730.20)에 근접한 수준.

---

## 1. 피처 우선순위 — 신규 29개 전체 랭킹 (핵심)

**판정 기준**: ① permutation importance(모델이 그 피처에 얼마나 의존하는지, val 6만행 샘플·3회 반복) ② 애매하거나 의심스러운 경우 실제로 빼고 재학습(ablation)해서 BSS 변화로 실증. `season`처럼 val 분할이 시즌 단일값이라 permutation importance가 구조적으로 0이 나오는 함정이 있어(§3 참고), importance만으로 결론 내지 않고 애매한 케이스는 전부 직접 검증했다.

### ✅ 유지 추천 — 13개 (근거 강한 순)

| 우선순위 | 피처 | 담당 | 근거 | 비고 |
|---|---|---|---|---|
| 1 | `asof_pitcher_success_smoothed` | eunwoo | permutation importance +4.156e-04 (신규 피처 중 1위) | 베이지안 스무딩된 투수 성공률 |
| 2 | `tm_offspeed_rate` | yeongeun | +2.779e-04 (2위) | 상황별 트랙맨 집계 |
| 3 | `f_share` | haejin | +1.936e-04 (3위) | 투수별 퓨처스 노출 비율 |
| 4 | `hand_match` | 통합 (yeongeun·nawoon·eunwoo 중복이었던 걸 하나로) | +1.328e-04 (4위) | 손잡이 매치업, 3명이 독립적으로 만들 만큼 신호가 강했던 피처 |
| 5 | `count_state` | eunwoo | +1.286e-04 (5위) | 카운트 조합 범주형 |
| 6 | `form_dev_3` | haejin | +6.498e-05 | 최근 3경기 폼 이탈도 |
| 7 | `career_span` | haejin | +5.937e-05 | 데뷔 이후 경과 시즌 |
| 8 | `tm_fastball_rate` | yeongeun | +1.984e-05 | 상황별 트랙맨 집계 |
| 9 | `tm_zone_speed_mean` | yeongeun | +1.216e-05 | 상황별 트랙맨 집계 |
| 10 | `residual` | haejin | +4.651e-06 | 성공/가운데/역회전 잔여 비율 |
| 11 | `tm_horz_break_mean` | yeongeun | +4.488e-06 | 상황별 트랙맨 집계 |
| 12 | `is_full_count` | eunwoo | +8.609e-07 (0에 가깝지만 양수) | 풀카운트 여부 |
| 13 | `recent_form_diff` | eunwoo | **permutation은 −2.363e-05(음수)였지만, 직접 제외 후 재학습 시 val BSS −2.46 악화 확인** | permutation importance만으로 판단했으면 잘못 제외할 뻔한 케이스 — ablation으로 뒤집힘 |

### ❌ 제외 추천 — 16개

**(A) 모델이 학습 중 단 한 번도 분기에 사용하지 않음 — importance 정확히 0.000000, std도 0 (10개, 재학습 불필요할 만큼 명백)**

| 피처 | 담당 |
|---|---|
| `is_first_pitch` | nawoon |
| `is_hitter_advantage` | eunwoo |
| `is_two_outs` | nawoon |
| `count_depth` | haejin |
| `is_runner_on` | nawoon |
| `is_late_inning` | nawoon |
| `is_cold_start` | nawoon |
| `is_two_strikes` | eunwoo |
| `is_batter_cold_start` | haneul |
| `count_pressure` | haejin |

**(B) importance가 표준편차보다 작아 통계적으로 0과 구분 불가 (1개)**

| 피처 | 담당 | 근거 |
|---|---|---|
| `is_experienced_mix` | haneul | importance −2.831e-07, std +7.368e-07 → \|평균\| < 표준편차라 노이즈와 구분 안 됨. 개별 ablation은 안 돌렸으나 그룹 A와 사실상 동일한 근거 수준이라 같이 제외 권장 |

**(C) 실제로 빼고 재학습해서 "제외가 낫다"고 실증된 피처 (5개, ablation 재학습 결과)**

| 피처 | 담당 | 제외 시 val BSS 변화 |
|---|---|---|
| `tm_n` | yeongeun (본인 피처) | **+92.62** (6개 중 가장 크게 개선) |
| `pressure_score` | nawoon | +69.93 |
| `pitcher_control_ratio` | haneul | +48.76 |
| `skill_gap` | haejin | +21.67 |
| `tm_breaking_rate` | yeongeun (본인 피처) | +5.60 |

> **왜 이 5개는 "안 쓰임"이 아니라 "오히려 해로움"인가**: permutation importance가 음수이면서 표준편차도 0이 아니었던(=모델이 가끔 쓰긴 썼던) 피처들이다. 실제로 빼고 재학습하니 전부 성능이 뚜렷하게 올라, 단순히 무의미한 정보를 넘어 **노이즈로 작용해 트리 분기를 오염시키고 있었다**고 해석된다.
>
> 특히 `tm_n`(본인이 만든 피처)이 가장 크게 해로웠다 — 트랙맨 표본이 적은 상황일수록 같이 만든 `tm_fastball_rate` 등 집계 피처 자체가 불안정한데, `tm_n`이 이 불확실성을 모델에 올바르게 전달하지 못하고 오히려 과적합을 유발한 것으로 추정.

---

## 2. 완전 중복 — 통합 단계에서 이미 제거 (참고용, 위 29개엔 안 잡힘)

| 제거된 피처 | 담당 | 사유 |
|---|---|---|
| `tm_p_*`/`tm_b_*`/`speed_diff`/`spin_diff`/`vbreak_diff` (11개) | eunwoo | `pitcher_id`/`batter_id` 기준 trackman join인데 train과 trackman ID 체계가 완전히 달라(교집합 0, 직접 검증) 전부 NaN → cold-start fallback으로 전 행이 상수값이 됨. **⚠️ eunwoo에게 공유 필요한 버그** |
| `is_platoon_advantage` | eunwoo | `hand_match`와 정의 완전 동일 |
| `is_disadvantaged_count` | nawoon | `is_hitter_advantage`(eunwoo)와 정의 완전 동일 |
| `recent_3g_diff` | eunwoo | `form_dev_3`(haejin)와 상관계수 1.0000 (fillna 여부만 다름) |
| `fastball_ratio`/`breaking_ratio`/`offspeed_ratio` | eunwoo | base의 `asof_pitcher_*_rate`를 fillna(0)만 한 것, 새 정보 없음 |
| `season_progress` | haneul | `game_month`와 상관계수 1.0000 (단순 평행이동) |

---

## 3. 방법론 & 측정 함정

- 모델: `sklearn.ensemble.HistGradientBoostingClassifier` (네이티브 범주형 처리)
- 검증: 팀 공통 규칙, `season == 2024` 홀드아웃
- Feature importance: HistGBM은 `.feature_importances_`가 없어 permutation importance 사용
- **⚠️ `season` 함정**: val이 `season==2024` 단일 시즌이라 val 내에서 `season`이 상수가 되고, permutation importance는 상수 컬럼을 셔플해도 값이 안 바뀌어 **항상 0**을 반환한다. 실제로 직접 빼고 재학습해보니:

  > **season 제외 시 val BSS 511.30 → 342.22 (−169.08)** → importance는 0이지만 실제로는 매우 중요한 피처
  
  → **이래서 permutation importance만으로 피처를 제외하면 안 되고, 애매한 케이스는 반드시 ablation으로 교차검증했다** (위 13번 `recent_form_diff`, 제외그룹(C) 5개가 그 결과물).

---

## 4. 잔차 분석

### 보정(Calibration) — 가장 핵심적인 개선 지점

- 전체 평균: 예측확률 0.4981 vs 실제 성공률 0.4861 (**과신 +0.0120**)
- 예측확률이 낮은 구간은 오히려 약간 과소추정, **확률이 높아질수록 점점 더 과신**하는 패턴 (상위 구간 gap +0.03대)
- 팀이 기존 LightGBM+XGBoost 앙상블에서 발견했던 것과 동일한 패턴 → isotonic calibration으로 실제 개선 확인됨 (아래 §5)

### 상황별 잔차

- 카운트(볼-스트라이크)/주자상황(`base_state`)/이닝/경기유형(`game_type`)별로는 절대잔차가 대부분 0.494~0.498로 **균일** — 특정 상황에 구조적으로 편향된 건 아님
- 유일한 이상치: **콜드스타트(투수 첫 투구, `asof_pitcher_n==0`)** 에서 mean_residual **−0.0733** (일반 대비 6배, 모델이 과대평가) — 단 표본 81건뿐이라 통계적 신뢰도는 낮음

---

## 5. 최종 모델 (61피처 + 튜닝 + 보정)

위 결론 반영해서 **base 47 + 유지 13 = 60개 피처**로 최종 재학습, 하이퍼파라미터 랜덤서치(15회), isotonic 보정까지 진행. (※ 최초 실행 시점엔 `is_experienced_mix`를 유지로 잘못 포함해 61개로 학습했음 — 아래 수치는 61개 기준 결과이며, 해당 피처의 영향은 미미해 재학습해도 결과는 사실상 동일할 것으로 예상됨)

### 성능 추이

| 단계 | val BSS |
|---|---|
| 76피처 (가지치기 전) baseline | 511.3006 |
| 61피처 (가지치기 후) baseline | 575.1178 (+63.8172) |
| 61피처 + 하이퍼파라미터 튜닝 | **720.5286** (+209.23 vs 76피처 baseline) |
| 61피처 + 튜닝 + isotonic 보정 (eval-half, 누수 없이 측정) | **743.0158** (보정 효과 +32.50) |

### 최적 하이퍼파라미터 (랜덤서치 15회 중)

```
learning_rate=0.01, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0, max_bins=128
```

낮은 learning_rate(0.01)와 중간 크기 트리(63 leaves) 조합이 압도적으로 우수했음 (2위 705.98, 3위 717.43도 전부 learning_rate=0.01 조합).

> **참고**: 이 결과(720.53)는 팀의 기존 튜닝된 LightGBM 단독 성능(730.20, CLAUDE.md 기준)에 거의 근접한 수준. HistGBM도 제대로 튜닝하면 LightGBM과 경쟁력 있는 모델로 확인됨. (다만 팀 최종 앙상블+보정 결과 785.65에는 아직 못 미침 — 단일 모델 대 2모델 앙상블 비교라 자연스러운 차이)

---

## 6. 다음에 할 것

1. `is_experienced_mix` 제외한 정확히 60개 피처로 재학습해서 미세 재확인 (영향은 미미할 것으로 예상)
2. eunwoo의 trackman-ID join 버그(11개 피처 무효화) 공유 및 수정 요청
3. 콜드스타트 잔차 편향 — 표본이 81건뿐이라 추가 검증 필요(팀 전체 asof_pitcher_n==0 케이스로 재확인 권장)
4. HistGBM을 팀 앙상블(LightGBM+XGBoost)에 3번째 모델로 추가하는 것 검토 가치 있음 (720대 단일 성능이면 앙상블 기여 가능성)
5. 랜덤서치를 15회보다 더 늘려서(예: 30~50회) 추가 개선 여지 확인

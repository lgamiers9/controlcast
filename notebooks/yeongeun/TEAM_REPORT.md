# 팀 전체 피처엔지니어링 통합 검증 — 종합 보고서

## 마스터 테이블 — 피처별 담당자 교차비교 (전체 요약)

범례: ✅ 유지 · ❌ 제외 · ⚠️ 구조적 결함으로 제외 · ➖ 미언급/미검증

| 피처 | 담당 | yeongeun | haneul | 팀원B | 팀원C | 팀원D | 팀 결론 |
|---|---|---|---|---|---|---|---|
| `hand_match` | 통합 | ✅ | ➖ | ✅ | ✅ | ✅ | **유지 (전원일치, 최우선)** |
| `f_share` | haejin | ✅3위 | ➖ | ✅ | ✅2위 | ✅ | **유지 (강한 일치)** |
| `residual` | haejin | ✅ | ➖ | ✅ | ➖ | ✅ | 유지 |
| `form_dev_3` | haejin | ✅ | ➖ | ✅ | ➖ | ✅ | 유지 |
| `form_dev_1` | haejin | (미생성) | ➖ | ➖ | ❌중복 | ➖ | 제외 (form_dev_3와 중복) |
| `skill_gap` | haejin | ❌ (+21.67) | ➖ | ✅ "확실히 기여" | ➖ | ➖ | **⚠️ 신규 쟁점 — yeongeun/팀원B 정반대** |
| `count_pressure` | haejin | ❌ (importance 0) | ➖ | ➖ | ➖ | ✅ | **⚠️ 신규 쟁점 — 모델별로 다름** |
| `count_depth` | haejin | ❌ (importance 0) | ➖ | ➖ | ✅ (3.03σ) | ➖ | **⚠️ 신규 쟁점 — 모델별로 다름** |
| `asof_pitcher_success_smoothed` | eunwoo | ✅1위 | ➖ | ❌ (+16.87) | ✅1위 | ✅1위 | 쟁점 (3:1) |
| `recent_form_diff` | eunwoo | ✅ (ablation로 뒤집힘) | ➖ | ➖ | ➖ | ✅ | 유지 |
| `count_state` | eunwoo | ✅5위 | ➖ | ➖ | ➖ | ➖ | 유지 (잠정) |
| `is_full_count` | eunwoo | ✅ (약함) | ➖ | ➖ | ➖ | ➖ | 유지 (약함) |
| `is_hitter_advantage` | eunwoo | ❌ (그룹A) | ➖ | ➖ | ➖ | 중복정리만 | 대체로 제외 |
| `is_two_strikes` | eunwoo | ❌ (그룹A) | ➖ | ➖ | ➖ | ➖ | 제외 (잠정) |
| `fastball/breaking/offspeed_ratio` | eunwoo | ❌ (base와 100%중복) | ➖ | ✅ (중복 못 잡음) | ➖ | ➖ | **제외 — 팀원B 재확인 필요** |
| eunwoo trackman 선수단위 11개(원본) | eunwoo | ❌ | ❌ | ❌ | ⚠️**ID재매핑으로 살림** | ❌ | 제외 다수, 단 팀원C 해법 공유 필요 |
| `career_span` | haejin | ✅ | ➖ | ✅ (−10.61) | ⚠️**train/test 범위불일치** | ✅ | 쟁점 — 팀원C 지적 최우선 확인 |
| `pitcher_control_ratio` | haneul | ❌ (+48.76) | ❌ (본인) | 애매 | ❌ | ✅ | 제외 우세 (3:1) |
| `is_batter_cold_start` | haneul | ❌ | ❌ (본인) | ➖ | ➖ | ❌ | **제외 (강한 일치)** |
| `season_progress` | haneul | ❌ (game_month와 100%중복) | ❌ (본인) | ✅ (중복 못 잡음) | ➖ | ➖ | **제외 — 팀원B 재확인 필요** |
| `is_experienced_mix` | haneul | ❌ (노이즈수준) | ❌ (본인) | ✅ (ablation −16.17) | ➖ | ➖ | 쟁점 (2:1, 근거 상충) |
| `pressure_score` | nawoon | ❌ (+69.93) | ➖(미검증) | ➖ | ➖ | ➖ | 제외 (yeongeun 강한 근거) |
| `is_runner_on` | nawoon | ❌ (그룹A) | ➖ | ➖ | ➖ | ➖ | 제외 (잠정) |
| `is_late_inning` | nawoon | ❌ | ➖ | ➖ | ➖ | ❌ | **제외 (일치)** |
| `is_two_outs` | nawoon | ❌ | ➖ | ➖ | ➖ | ❌ | **제외 (일치)** |
| `is_cold_start` | nawoon | ❌ | ➖ | ➖ | ➖ | ❌ | **제외 (일치)** |
| `is_first_pitch` | nawoon | ❌ | ➖ | ➖ | ➖ | ➖ | 제외 (잠정) |
| `tm_offspeed_rate` | yeongeun | ✅2위 | ❌ (6개 묶음) | ✅ | ✅3위 | ➖ | 쟁점, 유지 쪽 우세(2:1) |
| `tm_fastball_rate` | yeongeun | ✅ | ❌ (6개 묶음) | ➖ | ✅ | ➖ | 쟁점, 유지 쪽 우세 |
| `tm_zone_speed_mean` | yeongeun | ✅ | ❌ (6개 묶음) | ➖ | ➖ | ➖ | 애매 (2명뿐) |
| `tm_horz_break_mean` | yeongeun | ✅ | ❌ (6개 묶음) | ➖ | ✅(단, `pitcher_hand` 인코딩 의심) | ➖ | 쟁점 + 팀원C 통찰 주목 |
| `tm_breaking_rate` | yeongeun | ❌ (+5.60) | ❌ (6개 묶음) | ➖ | ❌ (다중공선, −2.76σ) | ➖ | **제외 (3명이 서로 다른 방법으로 일치 — 신뢰도 최고)** |
| `tm_n` | yeongeun | ❌ (+92.62, 최다) | ❌ (6개 묶음) | ➖ | ➖ | ➖ | 제외 (강한 근거) |

**한눈에 보이는 패턴**
- **가장 신뢰도 높은 제외**: `tm_breaking_rate` (yeongeun·haneul·팀원C 3명이 전혀 다른 방법으로 독립적으로 제외 도달), `is_late_inning`/`is_two_outs`/`is_cold_start`/`is_batter_cold_start` (전원 일치)
- **가장 신뢰도 높은 유지**: `hand_match`, `f_share` (거의 전원 최상위권)
- **모델마다 결론이 갈리는(=회의에서 꼭 짚어야 할) 피처**: `skill_gap`, `count_pressure`, `count_depth`, `asof_pitcher_success_smoothed`, `career_span`, `pitcher_control_ratio`, `is_experienced_mix`, yeongeun trackman 계열 전반
- **팀원B가 중복 탐지를 놓친 것으로 보이는 피처**: `fastball_ratio`/`breaking_ratio`/`offspeed_ratio`(base와 100%중복), `season_progress`(`game_month`와 100%중복) — 32개 최종안에 재검토 필요

---

## 0. 개요

5명이 각자 독립적으로 "팀 전체 피처 통합 → 검증" 작업을 서로 다른 모델/방법론으로 수행함.

| 담당 | 모델 | 평가지표 | 검증분할 |
|---|---|---|---|
| yeongeun(본인) | HistGradientBoostingClassifier | BSS | season==2024 홀드아웃 |
| haneul | RandomForest | BSS 추정 | 랜덤 8:2 (⚠️ 시즌 미반영, 본인이 한계로 명시) |
| 팀원 B (이름 미상) | 미상 | BSS, 5시드 평균 | 미상 |
| 팀원 C (이름 미상) | CatBoost (GPU) | RES (Murphy 분해) | season==2024, 다중시드(3~7) |
| 팀원 D (이름 미상) | 미상 | Gain + Permutation 이중검증 | 미상 |

⚠️ 팀원 B/C/D는 자료에 이름이 명시돼 있지 않아 특정하지 못함 — 확인되는 대로 반영 필요.

---

## 1. 전원(또는 다수)이 독립적으로 도달한 공통 결론

### 1-1. "개별 중요도만으로 피처를 pruning하면 안 된다" — 가장 중요한 메시지

서로 다른 모델·지표·팀원이 전부 같은 함정을 각자 독립적으로 발견함.

| 담당 | 발견 내용 |
|---|---|
| yeongeun | `season`이 permutation importance 0인데(val 분할 상수 함정), 실제 제외 시 −169.08 |
| 팀원 B | `asof_pitcher_success_smoothed`(gain 2위)가 오히려 제외 시 +16.87 개선, `is_experienced_mix`(gain 0)가 제외 시 −16.17 악화 |
| 팀원 C | 유의미(2σ 이상) 10개만 남긴 세트가 전체 41개 세트보다 오히려 RES 낮음(0.001842 vs 0.001880) — "무기여"로 보였던 31개가 집합적으로는 기여 |
| haneul | 담당자 단위 순차 ablation만 수행해 동시제거 효과는 미확인 — 본인이 스스로 이 한계를 명시 |

**→ 회의에서 "importance가 낮으니 빼자"는 개별 판단은 지양하고, 반드시 ablation(실제 제외 후 재학습) 결과로 교차검증할 것.**

### 1-2. 중복 피처 판정 — 전원 일치

| 중복 쌍 | 채택 | 비고 |
|---|---|---|
| `hand_match` = `is_platoon_advantage`(eunwoo) = nawoon 버전 | `hand_match` | 5/5 전원 일치 |
| `form_dev_3`(haejin) = `recent_3g_diff`(eunwoo) | `form_dev_3` | 5/5 전원 일치 |
| `is_hitter_advantage`(eunwoo) = `is_disadvantaged_count`(nawoon) | `is_hitter_advantage` | 5/5 전원 일치 (다만 이후 중요도 자체가 낮아 최종 제외한 팀원 다수) |

### 1-3. 명확히 제외 — 다수 일치

| 피처 | 근거 | 일치 |
|---|---|---|
| `is_cold_start`, `is_batter_cold_start`, `is_two_outs`, `is_late_inning` | 여러 팀원이 독립적으로 importance ≈ 0 확인 | yeongeun, 팀원 D 등 |
| eunwoo trackman 선수단위 집계 11개 (원본 버전) | ID 매칭률 0% → 죽은 피처 | yeongeun, 팀원 B, 팀원 D (haneul도 제외 시 개선 확인) — **단, 팀원 C는 ID 재매핑으로 살려냄 (§2)** |

---

## 2. 가장 중요한 논쟁거리 — eunwoo의 trackman 매칭 버그, 세 가지 다른 대응

| 대응 | 담당 | 결과 |
|---|---|---|
| ① 통째로 제외 | yeongeun, 팀원 B, 팀원 D | 11개 전부 죽은 피처로 판정, 제외 |
| ② **ID 재매핑으로 살림** | **팀원 C (CatBoost)** | **"슬롯 동시출현 기반 ID 매핑"으로 교체 → 매칭 성공.** `tm_p_speed_mean` 등 다수가 permutation 유의(2~4σ)로 확인됨 |
| ③ 시점 준수 여부 미확인 | haneul | 시간 관계상 검증 못 함, "재확인 필요"로 명시 |

**회의 안건 제안**: 팀원 C가 개발한 ID 재매핑 방법을 팀 전체가 공유해서, eunwoo의 원래 의도(투수 개인 단위 트랙맨 특성)를 살릴 수 있는지 검토할 가치가 큼. 지금까지는 "죽은 피처니까 버린다"가 다수였는데, 살릴 방법이 실제로 있었다는 게 확인됨.

---

## 3. 의견이 갈리는 피처들 — 팀 논의 필요

### `asof_pitcher_success_smoothed` (eunwoo)

| 담당 | 판정 | 근거 |
|---|---|---|
| yeongeun | 유지, 신규 1위 | permutation +4.156e-04 |
| 팀원 C | 유지, 신규 1위 | LossFunctionChange 0.000633 · permutation 0.000353 둘 다 1위 |
| 팀원 D | 유지 | Gain 51.34 · Permutation 0.000681 둘 다 신규 1위 |
| **팀원 B** | **제외 방향 시사** | **ablation(실제 제외) 시 오히려 +16.87 개선** |

3곳에서 압도적 1위인데 1곳에서만 반대 — 팀원 B의 32피처 조합 안에서만 나타나는 상호작용 효과일 가능성. **팀원 B의 실험 조건(같이 있던 다른 피처) 재확인 필요.**

### `career_span` (haejin)

| 담당 | 판정 | 근거 |
|---|---|---|
| yeongeun | 유지 | permutation 양수 |
| 팀원 D | 유지 | Gain 40.20 · Permutation 0.0000472 |
| 팀원 B | 유지(간접) | 32개 세트에서 제외 시 −10.61 악화 |
| **팀원 C** | **⚠️ 구조적 결함으로 제외** | **season 상관 0.660 + 시즌별 부호 반전(교란변수) + train/test 값 범위 불일치 (train 최대 5 → test에 6 등장, 학습 안 된 값이 test에 나타남)** |

팀원 C가 지적한 **train/test 값 범위 불일치는 실제 운영 리스크** — 다른 3명은 importance만 봐서 이 구조적 문제를 못 잡았을 가능성이 있음. **가장 우선 확인해야 할 이슈.**

### `pitcher_control_ratio` (haneul)

| 담당 | 판정 | 근거 |
|---|---|---|
| yeongeun | **제외** | ablation 시 +48.76 개선 |
| 팀원 C | **제외** | "분모 +1e-6, 값 폭발 — 신규 중 최대 음수" |
| 팀원 D | 유지 | Gain 13.72 · Permutation 0.0000801 |
| 팀원 B | 애매 | 단독 제거는 유리해 보였으나 5개 동시 제거 시 이득 없음 |

3명 중 2명(yeongeun, 팀원 C)이 서로 다른 방법으로 독립적으로 "해로움"을 확인 — **제외 쪽에 무게가 실림.**

### yeongeun의 trackman 상황별 집계 6개

| 담당 | 판정 | 근거 |
|---|---|---|
| yeongeun(본인) | 5개 유지 (`tm_offspeed_rate` 2위 등) | permutation 상위권 |
| 팀원 B | 유지 근거 강함 | `hand_match` 의존도를 −107.67 → −8.25로 완화(대체 신호 제공 확인) |
| 팀원 C | 유지 | `tm_offspeed_rate` permutation 3위, `tm_p_*` 계열도 다수 양의 기여 |
| **haneul** | **제외 방향** | 담당자 단위 순차 ablation에서 yeongeun 6개 제외 시 +6.44 개선(1674.50, 최선 기록) |

4명 중 3명이 강하게 "유지"를 지지하고 haneul만 반대인데, haneul 본인이 밝힌 한계(랜덤 분할·단일시드·원본 38컬럼 기준·순차 단독제거만 수행)를 감안하면 **haneul 결과의 신뢰도가 상대적으로 낮다고 판단됨.**

---

## 4. 담당자별 최종 성능 (⚠️ 절대값 비교 금지)

| 담당 | 모델 | 최종 피처 수 | 최종 성능 | 비고 |
|---|---|---|---|---|
| yeongeun | HistGBM (튜닝+보정) | 47+13=60개 | val BSS 720.53 → eval-half 743.02 | season 홀드아웃, 정식 BSS |
| haneul | RandomForest | 65개 | BSS(추정) 1674.50 | ⚠️ 랜덤 8:2 분할이라 절대값 비교 불가(본인이 명시한 한계), baseline 자체도 1599.02로 스케일이 다름 |
| 팀원 B | 미상 | 32개 | BSS 685.91 (5시드 평균, ±12.32) | season 분할 여부 불명, 다중시드 적용 |
| 팀원 C | CatBoost | 79개 | RES 0.001885 (3시드, baseline 대비 +11.5%) | BSS 대신 RES 사용(계산 방식이 근본적으로 다름), season 분할 |
| 팀원 D | 미상 | 미상 | 미상 | Gain+Permutation 기준만 제공, 최종 성능수치 없음 |

**절대값을 서로 비교하면 안 되는 이유**: 검증분할 방식(season vs 랜덤), 평가지표(BSS vs RES), 시드 수(단일 vs 다중), 베이스 피처 수(47 vs 38)가 담당자마다 달라서 숫자 자체는 비교 불가. **오직 "어떤 피처가 유지/제외되는가"의 방향성만 교차검증에 사용할 것.**

---

## 5. eunwoo 공유 필요 사항 (재확인)

- 원래 trackman 매칭 방식(`pitcher_trackman_id`를 `pitcher_id`로 이름만 바꿔 병합)이 매칭률 0%로 11개 피처가 죽어있었음
- 팀원 C가 "슬롯 동시출현 기반 ID 매핑"으로 해결 성공 — 이 방법 공유받아서 eunwoo 본인이 원래 의도(투수 개인 트랙맨 특성)를 살릴 수 있는지 검토 권장

## 6. haejin 피처 — 전원 호평, 이견 없음

`skill_gap`, `residual`, `f_share`, `form_dev_3` 등 haejin 담당 피처는 5명 전원의 분석에서 예외 없이 긍정적으로 평가됨 (haneul의 순차 ablation에서도 haejin 담당 제외 시 유일하게 뚜렷한 하락 −42.35 확인). **팀 내에서 가장 이견 없이 확실한 피처 그룹.**

---

## 7. 회의 액션 아이템

1. 팀원 B/C/D 이름 확정, 위 표에 반영
2. 쟁점 4가지 논의: `asof_pitcher_success_smoothed`, `career_span`, `pitcher_control_ratio`, yeongeun trackman 6개
3. eunwoo trackman ID 재매핑 방법(팀원 C 개발) 팀 공유 및 eunwoo 피처 부활 검토
4. haneul 검증 재수행 시 season 기준 분할 + 다중 시드 적용 권장 (본인도 인지하고 있음)
5. `career_span`의 train/test 값 범위 불일치 이슈 — 다른 담당자도 재확인 권장
6. 최종 피처셋/모델 선택은 이번 회의에서 5개 결과 비교 후 결정

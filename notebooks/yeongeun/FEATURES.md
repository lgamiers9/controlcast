# 통합 피처셋 설명 (`build_integrated_features.py` 산출물)

팀원 전체(eunwoo, haejin, haneul, nawoon, yeongeun)가 만든 피처를 하나로 합친 결과물 문서.
생성 스크립트: [`build_integrated_features.py`](build_integrated_features.py)
데이터: `cache/integrated_train.parquet` (1,475,092행 × 78컬럼) / `cache/feature_manifest.json`

- base 47개 (대회 원본 제공 피처) + 팀원 신규 29개 = **총 76개 피처**
- 중복 검증: 완전 동일 컬럼 0개, 상관계수 1.0000짜리 중복 5개 제거 완료 (아래 "제외한 피처" 참고)

---

## 1. yeongeun — 상황 기준 트랙맨 집계 (7개)

`[pitcher_hand, batter_hand, balls_before, strikes_before, outs_before]` 조합(KEY) 기준으로
`trackman_history.csv`를 집계해서 train/test에 KEY로 left join. (train/test의 `pitcher_id`/`batter_id`는
trackman과 ID 체계가 완전히 달라 직접 join 불가 — 그래서 "상황"으로 우회 집계.)

| 피처 | 정의 | 결측률 |
|---|---|---|
| `tm_fastball_rate` | 해당 상황에서 직구 계열 구종 비율 | 0% |
| `tm_breaking_rate` | 해당 상황에서 변화구 계열 비율 | 0% |
| `tm_offspeed_rate` | 해당 상황에서 오프스피드 계열 비율 | 0% |
| `tm_zone_speed_mean` | 해당 상황 평균 스트라이크존 통과 구속 | 0% |
| `tm_horz_break_mean` | 해당 상황 평균 수평 무브먼트 | 0% |
| `tm_n` | 해당 상황 trackman 표본 수 | 0% |
| `hand_match` | 투수·타자 손잡이 일치 여부 (1=동일) | 0% |

---

## 2. eunwoo — 카운트/최근폼 도메인 피처 (6개)

| 피처 | 정의 | 결측률 |
|---|---|---|
| `count_state` | `"{balls}B_{strikes}S"` 문자열 범주형 (카운트 조합) | 0% |
| `is_hitter_advantage` | `balls_before > strikes_before` (타자 유리 카운트) | 0% |
| `is_two_strikes` | `strikes_before == 2` | 0% |
| `is_full_count` | `balls_before==3 & strikes_before==2` (풀카운트) | 0% |
| `recent_form_diff` | 직전 1경기 성공률 − 누적 성공률 (최근 폼 이탈도) | 0% |
| `asof_pitcher_success_smoothed` | 베이지안 스무딩된 투수 성공률 (표본 적을수록 전체평균 0.55로 수렴, prior weight m=20) | 0% |

⚠️ **제외됨**: `tm_p_*`/`tm_b_*`/`speed_diff`/`spin_diff`/`vbreak_diff` (11개) — `pitcher_id`/`batter_id` 기준
trackman join인데 train과 trackman ID 체계가 완전히 달라(교집합 0, 직접 검증) 전부 NaN → cold-start
fallback으로 전 행이 상수값이 되어 정보량 0. **eunwoo에게 공유 필요.**

⚠️ **주의**: `count_state`는 과거 실험(CLAUDE.md §6-7, §11-3)에서 유사 개념(`count_state` 그 자체)이
트랙맨 피처 없이는 +7.52 도움이 됐지만, 트랙맨 피처와 결합하면 **-40.72로 악화**된 전례가 있음
(트랙맨 집계가 이미 balls/strikes를 KEY로 쓰기 때문에 중복 정보가 됨). 조합 테스트에서 눈여겨볼 것.

---

## 3. haejin — 행내부 연산 + 조회테이블 (7개)

| 피처 | 정의 | 결측률 |
|---|---|---|
| `count_pressure` | `balls_before - strikes_before` | 0% |
| `count_depth` | `balls_before + strikes_before` | 0% |
| `form_dev_3` | 직전 3경기 성공률 − 누적 성공률 | 1.98% |
| `skill_gap` | 투수 누적 성공률 − 타자 누적 성공률 | 0.11% |
| `residual` | `1 − (성공률+가운데비율+역회전비율)` — 세 유형에 안 잡히는 잔여 비율 추정치 (이론상 음수 가능) | 0.05% |
| `f_share` | 투수별 퓨처스(2군) 리그 등판 비율 (2019~2022 오염구간 기준, `pitcher_id` 조회테이블) | 0% |
| `career_span` | `season − 데뷔시즌` | 0% |

`f_share`/`career_span`은 train 전체로 미리 계산한 조회테이블(`pitcher_id` → 값)을 join하는 방식이라,
평가 시에도 그 행 자기 자신만으로 결정되는 값(row-independent) — 규정 위반 아님.

---

## 4. haneul — 행내부 연산 (3개)

| 피처 | 정의 | 결측률 |
|---|---|---|
| `pitcher_control_ratio` | `볼비율 / (가운데비율 + 1e-6)` — 제구 성향 지표 | 0.05% |
| `is_batter_cold_start` | 타자 첫 타석 여부 (`asof_batter_n == 0`) | 0% |
| `is_experienced_mix` | 투수 구종이력 표본 수 ≥ 322 (haneul 노트북에서 train 중앙값으로 확정한 고정 임계값) | 0% |

---

## 5. nawoon — 압박상황/매치업/콜드스타트 (6개)

| 피처 | 정의 | 결측률 |
|---|---|---|
| `is_runner_on` | 출루 주자 있음 여부 (`num_runners_on > 0`) | 0% |
| `is_late_inning` | 7회 이상 | 0% |
| `is_two_outs` | 2아웃 | 0% |
| `pressure_score` | `is_hitter_advantage + is_runner_on + is_late_inning + is_two_outs` 합산 (0~4, 압박 상황 종합 점수) | 0% |
| `is_cold_start` | 투수 첫 투구 여부 (`asof_pitcher_n == 0`) | 0% |
| `is_first_pitch` | 초구(0-0 카운트) 여부 | 0% |

⚠️ **주의**: `is_cold_start`처럼 결측/신인 여부를 명시적 플래그로 만드는 방식은 과거 실험(CLAUDE.md §6-7)에서
`-12.44`로 악화된 전례가 있음 (LightGBM류 트리 모델은 NaN을 이미 네이티브로 처리해서 명시적 플래그가
중복 정보+노이즈만 추가). 조합 테스트에서 확인 필요.

---

## 제외한 피처 (완전 중복, 상관계수 1.0000)

| 제외된 피처 | 담당 | 남긴 피처 | 사유 |
|---|---|---|---|
| `is_platoon_advantage` | eunwoo | `hand_match`(yeongeun/nawoon 공통) | 정의 완전히 동일 |
| `is_disadvantaged_count` | nawoon | `is_hitter_advantage`(eunwoo) | 정의 완전히 동일 (`pressure_score` 계산에도 `is_hitter_advantage` 재사용) |
| `recent_3g_diff` | eunwoo | `form_dev_3`(haejin) | 완전히 같은 공식, NaN 처리(fillna 여부)만 다름 |
| `fastball_ratio` | eunwoo | `asof_pitcher_fastball_rate`(base) | base 컬럼을 fillna(0)만 한 것, 새 정보 없음 |
| `breaking_ratio` | eunwoo | `asof_pitcher_breaking_rate`(base) | 위와 동일 |
| `offspeed_ratio` | eunwoo | `asof_pitcher_offspeed_rate`(base) | 위와 동일 |
| `season_progress` | haneul | `game_month`(base) | 단순 평행이동(`game_month - 3`), 트리 모델 기준 완전히 같은 분할 정보 |
| `tm_p_*`/`tm_b_*`/`speed_diff`/`spin_diff`/`vbreak_diff` (11개) | eunwoo | — | ID 체계 불일치로 상수값 (위 2절 참고) |

## 완전 중복은 아니지만 상관 높은 쌍 (0.85~0.95, 파생 관계라 정상)

| 신규 피처 | 관련 base 피처 | 상관계수 |
|---|---|---|
| `tm_horz_break_mean` | `pitcher_hand` | 0.9500 |
| `asof_pitcher_success_smoothed` | `asof_pitcher_success_rate` | 0.9439 |
| `recent_form_diff` | `asof_pitcher_prev1_game_success_rate` | 0.9106 |
| `is_two_outs` | `outs_before` | 0.8645 |
| `is_two_strikes` | `strikes_before` | 0.8622 |
| `is_runner_on` | `num_runners_on` | 0.8591 |
| `count_depth` | `balls_before` | 0.8583 |

이진화/스무딩/차분 등으로 만든 파생 피처라 원본과 상관은 높지만 정보 손실·변형이 있어 중복은 아님.
다만 feature importance에서 원본과 경쟁할 가능성이 높으니 결과 해석 시 참고.

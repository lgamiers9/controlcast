# -*- coding: utf-8 -*-
"""rf_feature_validation_HN.py

Random Forest 담당 (haneul) — 통합 피처셋 검증 코드
- 팀원 5명(eunwoo, haejin, haneul, nawoon, yeongeun) 피처 통합
- 담당자 단위 ablation(순차 제거) 검증
- Feature Importance (gini importance)
- 잔차 분석

실행 환경: Colab (# %% 셀 구분)
"""

# %% ---------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")

ID, TARGET = "row_id", "control_success"


# %% ---------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

DATA_PATH = '/content/drive/MyDrive/aimers/data'
train = pd.read_csv(f'{DATA_PATH}/train.csv')
trackman = pd.read_csv(f'{DATA_PATH}/trackman_history.csv')
print("train:", train.shape)
print("trackman:", trackman.shape)


# %% ---------------------------------------------------------
# 2. 피처 엔지니어링 — 팀원 5명 담당분 통합
# ------------------------------------------------------------

# ---- 2-1) haneul(본인) 4개 ----
def add_features_HN(df):
    """haneul 담당 4개 피처. 행 내부 연산만 사용, 조회 테이블 없음."""
    out = df.copy()
    out["pitcher_control_ratio"] = out["asof_pitcher_ball_rate"] / (out["asof_pitcher_middle_rate"] + 1e-6)
    out["is_batter_cold_start"] = (out["asof_batter_n"] == 0).astype(int)
    out["season_progress"] = out["game_month"] - 3
    out["is_experienced_mix"] = (out["asof_pitcher_pitchmix_n"] >= 322).astype(int)
    return out

train = add_features_HN(train)


# ---- 2-2) nawoon 7개 ----
def add_features_naun(df):
    """nawoon 담당 7개 피처. 압박 상황·매치업·콜드스타트 관련."""
    df = df.copy()
    df["is_disadvantaged_count"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_runner_on"] = (df["num_runners_on"] > 0).astype(int)
    df["is_late_inning"] = (df["inning"] >= 7).astype(int)
    df["is_two_outs"] = (df["outs_before"] == 2).astype(int)
    df["pressure_score"] = (
        df["is_disadvantaged_count"] + df["is_runner_on"] + df["is_late_inning"] + df["is_two_outs"]
    )
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["is_cold_start"] = (df["asof_pitcher_n"] == 0).astype(int)
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)
    return df

train = add_features_naun(train)


# ---- 2-3) haejin 7개 (조회 테이블 방식) ----
is_f = train["game_type"] == "F"
f_share_all = (train.assign(_f=is_f.astype("int8")).groupby("pitcher_id", observed=True)["_f"].mean())
debut_season = train.groupby("pitcher_id", observed=True)["season"].min()

def add_features_hj(df, f_share_map, debut_map):
    """haejin 담당 7개 피처. count_pressure/depth, form_dev_3, skill_gap,
    residual(라벨 역산), f_share/career_span(조회 테이블 조인)."""
    out = df.copy()
    out["count_pressure"] = out["balls_before"] - out["strikes_before"]
    out["count_depth"] = out["balls_before"] + out["strikes_before"]
    out["form_dev_3"] = out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_success_rate"]
    out["skill_gap"] = out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]
    out["residual"] = (1.0 - out["asof_pitcher_success_rate"]
                        - out["asof_pitcher_middle_rate"] - out["asof_pitcher_reverse_rate"])
    pid = out["pitcher_id"].astype("int64")
    out["f_share"] = pid.map(f_share_map).fillna(f_share_all.median())
    debut = pid.map(debut_map).fillna(debut_season.median())
    out["career_span"] = out["season"] - debut
    return out

train = add_features_hj(train, f_share_all.to_dict(), debut_season.to_dict())


# ---- 2-4) eunwoo 11개 (trackman 개별 매칭 — pitcher_id 체계 불일치로 매칭 실패 확인됨) ----
def build_trackman_aggregates(tm_df):
    """투수/타자별 trackman 요약 통계. 주의: pitcher_id(train)와
    pitcher_trackman_id(trackman)가 서로 다른 ID 체계라 merge 시 매칭률 0%."""
    tm_pitcher = tm_df.groupby('pitcher_trackman_id').agg(
        tm_p_speed_mean=('rel_speed', 'mean'),
        tm_p_speed_std=('rel_speed', 'std'),
        tm_p_spin_mean=('spin_rate', 'mean'),
        tm_p_vbreak_mean=('induced_vert_break', 'mean'),
        tm_p_hbreak_mean=('horz_break', 'mean')
    ).reset_index().rename(columns={'pitcher_trackman_id': 'pitcher_id'})
    tm_batter = tm_df.groupby('batter_trackman_id').agg(
        tm_b_speed_mean=('rel_speed', 'mean'),
        tm_b_spin_mean=('spin_rate', 'mean'),
        tm_b_vbreak_mean=('induced_vert_break', 'mean')
    ).reset_index().rename(columns={'batter_trackman_id': 'batter_id'})
    return tm_pitcher, tm_batter

tm_p, tm_b = build_trackman_aggregates(trackman)
train = train.merge(tm_p, on='pitcher_id', how='left')
train = train.merge(tm_b, on='batter_id', how='left')

# 매칭 실패로 인한 결측 → 평균 대치 (사실상 상수 컬럼이 됨)
for col in ['tm_p_speed_mean', 'tm_p_speed_std', 'tm_p_spin_mean', 'tm_p_vbreak_mean', 'tm_p_hbreak_mean']:
    train[col] = train[col].fillna(train[col].mean())
for col in ['tm_b_speed_mean', 'tm_b_spin_mean', 'tm_b_vbreak_mean']:
    train[col] = train[col].fillna(train[col].mean())

train['speed_diff'] = train['tm_p_speed_mean'] - train['tm_b_speed_mean']
train['spin_diff'] = train['tm_p_spin_mean'] - train['tm_b_spin_mean']
train['vbreak_diff'] = train['tm_p_vbreak_mean'] - train['tm_b_vbreak_mean']


# ---- 2-5) yeongeun 6개 (trackman 상황별 집계) ----
# 주의: 본인이 사용한 버전은 시점 준수(각 투구 시점 이전 데이터만 반영) 여부를
# 직접 확인하지 못함 — trackman_history.csv 전체를 groupby했을 가능성 있음.
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}
KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]

situ = trackman.groupby(KEY).agg(
    tm_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
    tm_breaking_rate=("pitch_type_group", lambda s: (s == "breaking").mean()),
    tm_offspeed_rate=("pitch_type_group", lambda s: (s == "offspeed").mean()),
    tm_zone_speed_mean=("zone_speed", "mean"),
    tm_horz_break_mean=("horz_break", "mean"),
    tm_n=("trackman_id", "count"),
).reset_index()
situ["pitcher_hand"] = situ["pitcher_hand"].map(INV_HAND_MAP)
situ["batter_hand"] = situ["batter_hand"].map(INV_HAND_MAP)

train = train.merge(situ, on=KEY, how='left')

print(f"\n피처 엔지니어링 완료. train shape: {train.shape}")


# %% ---------------------------------------------------------
# 3. 담당자 단위 Ablation 검증
#
#    [방법] 팀 전체 피처를 한 번에 넣는 대신, 담당자별로 순차 제거하며
#    Brier Skill Score 변화를 확인. random_state=42 단일 시드로 검증을
#    진행하였으며, 여러 시드로 반복하며 노이즈 폭을 정밀하게 재는 것보다
#    "어떤 담당자의 피처가 실제로 기여하는지"를 넓게 확인하는 방향으로
#    진행함.
# ------------------------------------------------------------
feature_cols = [
    'season', 'game_month', 'game_dayofweek', 'inning',
    'balls_before', 'strikes_before', 'outs_before',
    'run_top_before', 'run_bot_before', 'run_total_before',
    'score_diff_home', 'score_diff_pitcher_team',
    'runner_on_1b', 'runner_on_2b', 'runner_on_3b', 'num_runners_on',
    'home_win_expectancy', 'away_win_expectancy', 'li',
    'pitcher_hand', 'batter_hand',
    'asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
    'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
    'asof_pitcher_prev5_game_success_rate',
    'asof_pitcher_prev1_game_middle_rate', 'asof_pitcher_prev3_game_middle_rate',
    'asof_pitcher_prev5_game_middle_rate',
    'asof_batter_n', 'asof_batter_success_rate', 'asof_batter_middle_rate',
    'asof_pitcher_pitchmix_n', 'asof_pitcher_fastball_rate',
    'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'
]

haneul_features = ['pitcher_control_ratio', 'is_batter_cold_start', 'season_progress', 'is_experienced_mix']
nawoon_features = ['is_disadvantaged_count', 'is_runner_on', 'is_late_inning', 'is_two_outs',
                    'pressure_score', 'hand_match', 'is_cold_start', 'is_first_pitch']
haejin_features = ['count_pressure', 'count_depth', 'form_dev_3', 'skill_gap', 'residual',
                    'f_share', 'career_span']
eunwoo_features = ['tm_p_speed_mean', 'tm_p_speed_std', 'tm_p_spin_mean', 'tm_p_vbreak_mean',
                    'tm_p_hbreak_mean', 'tm_b_speed_mean', 'tm_b_spin_mean', 'tm_b_vbreak_mean',
                    'speed_diff', 'spin_diff', 'vbreak_diff']
yeongeun_features = ['tm_fastball_rate', 'tm_breaking_rate', 'tm_offspeed_rate',
                      'tm_zone_speed_mean', 'tm_horz_break_mean', 'tm_n']

new_features_all = list(dict.fromkeys(
    haneul_features + nawoon_features + haejin_features + eunwoo_features + yeongeun_features
))
feature_cols_all = feature_cols + new_features_all
print(f"베이스 {len(feature_cols)}개 + 신규 {len(new_features_all)}개 = 전체 {len(feature_cols_all)}개")


def run_rf(cols, label):
    """지정한 피처 목록으로 RF 학습 후 검증 데이터 Brier Skill Score 반환.
    random_state=42로 train/valid 분할 및 모델 시드 고정."""
    X = train[cols].fillna(train[cols].mean(numeric_only=True))
    y = train[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50,
                                 n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_va)[:, 1]
    brier = brier_score_loss(y_va, p)
    r = y_va.mean()
    sc = max(0, 100000 * (1 - brier / (r * (1 - r))))
    print(f"[{label}] Brier Skill Score: {sc:.2f}")
    return sc, rf


# Step 1: baseline (원본 38개)
score_base, _ = run_rf(feature_cols, "baseline 38개")

# Step 2: 팀 전체 통합 (86개)
score_all, rf_all = run_rf(feature_cols_all, "팀 전체 86개")

# Step 3: haneul(본인) 4개 제거
cols_no_mine = [c for c in feature_cols_all if c not in haneul_features]
score_no_mine, _ = run_rf(cols_no_mine, f"haneul 4개 제외 ({len(cols_no_mine)}개)")

# Step 4: eunwoo 11개 추가 제거 (trackman 매칭 실패 확인분)
cols_v2 = [c for c in cols_no_mine if c not in eunwoo_features]
score_v2, _ = run_rf(cols_v2, f"eunwoo 11개 추가 제외 ({len(cols_v2)}개)")

# Step 5: yeongeun 6개 추가 제거
cols_v3 = [c for c in cols_v2 if c not in yeongeun_features]
score_v3, rf_final = run_rf(cols_v3, f"yeongeun 6개 추가 제외 ({len(cols_v3)}개)")

# Step 6: haejin 7개 추가 제거
cols_v4 = [c for c in cols_v3 if c not in haejin_features]
score_v4, _ = run_rf(cols_v4, f"haejin 7개 추가 제외 ({len(cols_v4)}개)")

print("\n=== Ablation 결과 요약 ===")
print(f"baseline(38개):              {score_base:.2f}")
print(f"팀 전체(86개):                {score_all:.2f}  ({score_all-score_base:+.2f})")
print(f"-haneul(82개):                {score_no_mine:.2f}  ({score_no_mine-score_all:+.2f})")
print(f"-eunwoo(71개):                {score_v2:.2f}  ({score_v2-score_no_mine:+.2f})")
print(f"-yeongeun(65개):              {score_v3:.2f}  ({score_v3-score_v2:+.2f})  ← 최선")
print(f"-haejin(58개):                {score_v4:.2f}  ({score_v4-score_v3:+.2f})  ← 하락, haejin 재포함 결정")

# nawoon 담당분(7개)은 시간 관계상 미검증 — 다음 단계 과제


# %% ---------------------------------------------------------
# 4. Feature Importance (86개 전체 기준, gini importance)
# ------------------------------------------------------------
importance = pd.DataFrame({
    'feature': feature_cols_all,
    'importance': rf_all.feature_importances_
}).sort_values('importance', ascending=False)

print("=== Feature Importance 상위 20 ===")
print(importance.head(20).to_string(index=False))

plt.figure(figsize=(8, 10))
sns.barplot(data=importance.head(20), y='feature', x='importance', color='steelblue')
plt.title('통합 피처셋 Feature Importance (상위 20개)')
plt.tight_layout()
plt.show()

# eunwoo·yeongeun trackman 컬럼 순위 확인 (ablation 결과와 교차 검증)
importance_ranked = importance.reset_index(drop=True)
importance_ranked['rank'] = importance_ranked.index + 1
trackman_cols = eunwoo_features + yeongeun_features
print("\n=== trackman 관련 컬럼 순위 (죽은 피처 여부 확인) ===")
print(importance_ranked[importance_ranked['feature'].isin(trackman_cols)][['rank', 'feature', 'importance']]
      .to_string(index=False))


# %% ---------------------------------------------------------
# 5. 잔차 분석 (65개 최종 모델, yeongeun까지 제외한 조합 기준)
# ------------------------------------------------------------
X_final = train[cols_v3].fillna(train[cols_v3].mean(numeric_only=True))
y_final = train[TARGET]
X_tr, X_va, y_tr, y_va = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

pred_final = rf_final.predict_proba(X_va)[:, 1]
residuals = y_va - pred_final

print("=== 잔차 분석 (65개 기준) ===")
print(f"잔차 평균: {residuals.mean():.6f}")
print(f"잔차 표준편차: {residuals.std():.6f}")

plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50)
plt.title('최종 모델(65개) 잔차 분포')
plt.xlabel('실제값 - 예측확률')
plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------
# 6. 결론
# ------------------------------------------------------------
# [최종 채택] 65개 (원본 38개 + haejin 7개, nawoon 7개는 미검증)
#   baseline(1599.02) 대비 1674.50 (+75.48)
#
# [담당자별 판단]
#    haejin 7개 : 반드시 유지 (제외 시 -42.35 급락)
#    haneul(본인) 4개 : 제외 권장 (+5.10)
#    eunwoo 11개 : 제외 권장 (+4.09, trackman ID 체계 불일치로 매칭 실패)
#    yeongeun 6개 : 제외 권장 (+6.44)
#    nawoon 7개 : 미검증
#
# [한계]
#   - random_state=42 단일 시드로만 검증. 담당자 단위 ablation 4건에
#     검증 우선순위를 두었으며, 다중 시드 반복 검증은 다음 단계로 남겨둠
#   - 담당자 단위(그룹) 제거만 수행 — 그룹 내 개별 피처 기여도는 미분리
#   - 시즌 기준 분할(랜덤 대신 시계열) 미적용
#   - 하이퍼파라미터 튜닝, 확률 보정(calibration) 미적용
#   - 리더보드 제출 미진행
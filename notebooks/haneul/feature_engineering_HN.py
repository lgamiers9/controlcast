# ============================================================
# 피처 생성 
#
#   행 내부 연산 (4개)
#     pitcher_control_ratio   asof_pitcher_ball_rate / asof_pitcher_middle_rate
#     is_batter_cold_start    asof_batter_n == 0
#     season_progress         game_month - 3
#     is_experienced_mix      asof_pitcher_pitchmix_n >= 322 (중앙값 기준)
#
#   시도했으나 제외한 9개 (효과 미미/없음, 검증 로그는 하단 참고)
#     scoring_position, is_high_leverage, is_close_game, is_blowout,
#     is_top_inning, pitcher_batter_gap, pitcher_recent_trend,
#     is_pitcher_cold_start, is_batter_cold_start(중복 삭제분)
#
# 규정 준수
#   전부 행 내부 연산(그 행 자기 자신의 컬럼값만 사용), 
#   다른 행/전체 분포 참조 없음. 조회 테이블 없음.
#
# 실행 환경: Colab (# %% 셀 구분)
# ============================================================


# %% ---------------------------------------------------------
# 1. 설정
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")

BASE = Path("/content/drive/MyDrive/aimers")
DATA_DIR = BASE / "data"

ID, TARGET = "row_id", "control_success"

NEW_FEATURES = ["pitcher_control_ratio", "is_batter_cold_start",
                "season_progress", "is_experienced_mix"]

train = pd.read_csv(DATA_DIR / "train.csv")
print("원본:", train.shape)


# %% ---------------------------------------------------------
# 2. 피처 생성 함수
#    train / test 모두 이 함수 하나만 통과시킨다.
# ------------------------------------------------------------
def add_features(df):
    """4개 피처를 추가한다. 행 내부 연산만 사용, 조회 테이블 없음."""
    out = df.copy()

    # 투수의 "볼로 빠지는 비율" 대비 "가운데로 몰리는 비율"
    out["pitcher_control_ratio"] = (
        out["asof_pitcher_ball_rate"] / (out["asof_pitcher_middle_rate"] + 1e-6)
    ).astype("float32")

    # 타자가 상대한 투구 기록이 아예 없는 첫 타석(cold-start) 여부
    out["is_batter_cold_start"] = (out["asof_batter_n"] == 0).astype("int8")

    # 시즌 진행도 (3월=0 ~ 10월=7)
    out["season_progress"] = (out["game_month"] - 3).astype("int8")

    # 투수의 구종 이력 표본이 충분히 쌓였는지 (train 기준 중앙값 322 사용)
    out["is_experienced_mix"] = (out["asof_pitcher_pitchmix_n"] >= 322).astype("int8")

    return out


train = add_features(train)
print(train[NEW_FEATURES].dtypes)


# %% ---------------------------------------------------------
# 3. 분포와 결측 확인
# ------------------------------------------------------------
print("=== 기술통계 ===")
print(train[NEW_FEATURES].describe().T.round(4))

print("\n=== 결측률 (%) ===")
print((train[NEW_FEATURES].isna().mean() * 100).round(4))


# %% ---------------------------------------------------------
# 4. 분위수별 기저율 — 타겟과의 단변량 관계
# ------------------------------------------------------------
def quantile_report(df, col, target=TARGET, n_bins=5):
    s = df[col]
    if s.nunique() <= 12:
        g = df.groupby(col, observed=True)[target].agg(["mean", "size"])
    else:
        b = pd.qcut(s, n_bins, duplicates="drop")
        g = df.groupby(b, observed=True)[target].agg(["mean", "size"])
    lo, hi = g["mean"].min(), g["mean"].max()
    spread = (hi - lo) / lo if lo > 0 else np.nan
    return g, spread

print("=== 단변량 강도 (최대/최소 그룹 상대차) ===\n")
strength = {}
for c in NEW_FEATURES:
    g, spread = quantile_report(train, c)
    strength[c] = spread
    print(f"[{c}]  상대차 {spread:.1%}")
    print(g.round(4))
    print()

strength = pd.Series(strength).sort_values(ascending=False)
print("=== 강도 순위 ===")
print(strength.map(lambda v: f"{v:.1%}"))


# %% ---------------------------------------------------------
# 5. 기존 컬럼과의 중복 확인
# ------------------------------------------------------------
existing = [c for c in train.columns
            if c not in [ID, TARGET] + NEW_FEATURES
            and pd.api.types.is_numeric_dtype(train[c])]

samp = train.sample(min(300_000, len(train)), random_state=42)

print("=== 신규 × 기존 최고 상관 ===")
for c in NEW_FEATURES:
    cors = samp[existing].corrwith(samp[c].astype(float)).abs().dropna()
    top = cors.sort_values(ascending=False).head(3)
    flag = "  ⚠️ 중복 의심" if top.iloc[0] >= 0.90 else ""
    print(f"[{c}]{flag}")
    for k, v in top.items():
        print(f"    {v:.4f}  {k}")
    print()


# %% ---------------------------------------------------------
# 6. RF baseline 대비 검증
#    38개 기존 피처 + 신규 4개 → Brier Skill Score 비교
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

feature_cols = [
    'season','game_month','game_dayofweek','inning',
    'balls_before','strikes_before','outs_before',
    'run_top_before','run_bot_before','run_total_before',
    'score_diff_home','score_diff_pitcher_team',
    'runner_on_1b','runner_on_2b','runner_on_3b','num_runners_on',
    'home_win_expectancy','away_win_expectancy','li',
    'pitcher_hand','batter_hand',
    'asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate',
    'asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate',
    'asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate',
    'asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate',
    'asof_pitcher_pitchmix_n','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'
]

def run_rf(cols, label):
    X = train[cols].fillna(train[cols].mean(numeric_only=True))
    y = train[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_va)[:, 1]
    brier = brier_score_loss(y_va, p)
    r = y_va.mean()
    sc = max(0, 100000 * (1 - brier / (r*(1-r))))
    print(f"[{label}] Brier Skill Score: {sc:.2f}")
    return sc

score_base = run_rf(feature_cols, "기존 38개")
score_new  = run_rf(feature_cols + NEW_FEATURES, "38개 + 신규 4개")
print(f"차이: {score_new - score_base:.2f}")


# %% ---------------------------------------------------------
# 7. 결론
# ------------------------------------------------------------
# 시도한 피처: 총 13개 (초기 3개 + 추가 6개 + 최종 압축 검증용)
# 최종 채택: 4개 (pitcher_control_ratio, is_batter_cold_start,
#            season_progress, is_experienced_mix)
# 근거: 위 4개는 분위수 상대차 3~15% 수준으로 나머지(0.1~2.4%)보다 뚜렷했고,
#       RF 검증에서 38개 baseline(1599.02) 대비 +0.80 개선 확인
# 나머지 9개(scoring_position, is_high_leverage, is_close_game, is_blowout,
#            is_top_inning 등)는 분위수 상대차 2% 미만 + RF 검증 시 오히려
#            -11.02 하락 → 제외
#
# 주의: +0.80은 개선폭이 매우 작아 노이즈 수준일 가능성 있음.
#       팀 전체 피처 통합 후 permutation importance로 재검증 필요.


# %% ---------------------------------------------------------
# 8. 피처 추가된 데이터 캐싱
# ------------------------------------------------------------
train.to_parquet(DATA_DIR / "train_feat_HN.parquet", index=False)
print(f"저장 완료: {DATA_DIR / 'train_feat_HN.parquet'}")
print(f"shape {train.shape}")
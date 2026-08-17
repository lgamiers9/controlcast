# ============================================================
# 팀 피처 통합 — 5명분 전부 합쳐서 모델 학습 + importance + 잔차 분석
#
#   1. 해진 : count_pressure, count_depth, form_dev_3, skill_gap,
#            residual, f_share, career_span   (조회테이블 2개 필요)
#   2. 하늘 : pitcher_control_ratio, is_batter_cold_start,
#            season_progress, is_experienced_mix
#   3. 은우 : trackman 선수별 집계(speed/spin/break) + 상성지표 +
#            베이지안 스무딩 등 (영은의 situ_stats 방식과는 다른 grain)
#   4. 나운 : is_disadvantaged_count, is_runner_on, is_late_inning,
#            is_two_outs, pressure_score, hand_match, is_cold_start,
#            is_first_pitch
#   5. 영은 : situ_stats(hand×count×outs 조합) lookup + hand_match
#            (prepare_data.py)
#
# ⚠️ 통합 전 발견한 충돌
#   - hand_match : 나운과 영은이 "동일한 정의"로 중복 생성
#                  → merge 시 hand_match_x/_y로 갈라짐. 하나만 남긴다.
#   - is_platoon_advantage(은우) : hand_match와 개념적으로 동일(같은 계산식).
#                  이름만 다르므로 상관 0.99+ 예상 → 중복 후보로 별도 표시.
#   - is_batter_cold_start(하늘, asof_batter_n==0) vs
#     is_cold_start(나운, asof_pitcher_n==0) : 이름은 비슷하지만
#     정의가 달라 진짜 중복은 아님 → 그대로 유지하되 상관은 확인.
#
# 실행 환경: Colab
# ============================================================

# %% ---------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 200)
sns.set_theme(style="whitegrid")

# 팀 공용 경로로 통일 (각자 환경에 맞게 수정)
PROJECT_PATH = '/content/drive/MyDrive/Colab Notebooks/공모전/lg_aimers_9'

BASE = Path(PROJECT_PATH)
DATA_DIR = BASE / "data"
CACHE_DIR = BASE / "cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = BASE / "artifacts"; ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
tm_df = pd.read_csv(DATA_DIR / "trackman_history.csv", encoding="utf-8-sig")
print("train:", train.shape, " trackman:", tm_df.shape)

# %% ---------------------------------------------------------
# 1. 해진 — 조회테이블 산출 + 피처
# ------------------------------------------------------------
is_f = train["game_type"] == "F"
f_share_all = (train.assign(_f=is_f.astype("int8"))
                     .groupby("pitcher_id", observed=True)["_f"].mean())
contaminated = is_f & (train["season"] <= 2022)
f_share_old = (train.assign(_f=contaminated.astype("int8"))
                     .groupby("pitcher_id", observed=True)["_f"].mean())
debut_season = train.groupby("pitcher_id", observed=True)["season"].min()

LOOKUPS_HJ = {
    "f_share_all": f_share_all.to_dict(),
    "f_share_old": f_share_old.to_dict(),
    "debut_season": debut_season.to_dict(),
    "fallback_f_share": float(f_share_all.median()),
    "fallback_debut": int(debut_season.median()),
}
joblib_path = ARTIFACT_DIR / "feature_lookups_hj.joblib"
import joblib
joblib.dump(LOOKUPS_HJ, joblib_path, compress=3)


def add_features_hj(df, lookups, f_share_version="old"):
    out = df.copy()
    out["count_pressure"] = (out["balls_before"] - out["strikes_before"]).astype("int8")
    out["count_depth"] = (out["balls_before"] + out["strikes_before"]).astype("int8")
    out["form_dev_3"] = (out["asof_pitcher_prev3_game_success_rate"]
                         - out["asof_pitcher_success_rate"]).astype("float32")
    out["skill_gap"] = (out["asof_pitcher_success_rate"]
                        - out["asof_batter_success_rate"]).astype("float32")
    out["residual"] = (1.0 - out["asof_pitcher_success_rate"]
                       - out["asof_pitcher_middle_rate"]
                       - out["asof_pitcher_reverse_rate"]).astype("float32")
    key = f"f_share_{f_share_version}"
    pid = out["pitcher_id"].astype("int64")
    out["f_share"] = (pid.map(lookups[key])
                         .fillna(lookups["fallback_f_share"]).astype("float32"))
    debut = pid.map(lookups["debut_season"]).fillna(lookups["fallback_debut"])
    out["career_span"] = (out["season"].astype("int16") - debut).astype("int16")
    return out


# %% ---------------------------------------------------------
# 2. 하늘 — 행 내부 연산 4개
# ------------------------------------------------------------
def add_features_haneul(df):
    out = df.copy()
    out["pitcher_control_ratio"] = (
        out["asof_pitcher_ball_rate"] / (out["asof_pitcher_middle_rate"] + 1e-6)
    ).astype("float32")
    out["is_batter_cold_start"] = (out["asof_batter_n"] == 0).astype("int8")
    out["season_progress"] = (out["game_month"] - 3).astype("int8")
    out["is_experienced_mix"] = (out["asof_pitcher_pitchmix_n"] >= 322).astype("int8")
    return out


# %% ---------------------------------------------------------
# 3. 은우 — trackman 선수별 집계 + 상성/폼/스무딩 피처
#    ※ hand_match와 개념 중복인 is_platoon_advantage 포함 (그대로 두고 6번에서 판정)
# ------------------------------------------------------------
def build_trackman_aggregates(tm_df):
    tm_pitcher = tm_df.groupby("pitcher_trackman_id").agg(
        tm_p_speed_mean=("rel_speed", "mean"),
        tm_p_speed_std=("rel_speed", "std"),
        tm_p_spin_mean=("spin_rate", "mean"),
        tm_p_vbreak_mean=("induced_vert_break", "mean"),
        tm_p_hbreak_mean=("horz_break", "mean"),
    ).reset_index().rename(columns={"pitcher_trackman_id": "pitcher_id"})
    tm_batter = tm_df.groupby("batter_trackman_id").agg(
        tm_b_speed_mean=("rel_speed", "mean"),
        tm_b_spin_mean=("spin_rate", "mean"),
        tm_b_vbreak_mean=("induced_vert_break", "mean"),
    ).reset_index().rename(columns={"batter_trackman_id": "batter_id"})
    return tm_pitcher, tm_batter


def add_features_eunwoo(df, tm_p, tm_b):
    df = df.copy()
    df = df.merge(tm_p, on="pitcher_id", how="left")
    df = df.merge(tm_b, on="batter_id", how="left")
    for col in tm_p.columns:
        if col != "pitcher_id":
            df[col] = df[col].fillna(df[col].mean())
    for col in tm_b.columns:
        if col != "batter_id":
            df[col] = df[col].fillna(df[col].mean())
    df["speed_diff"] = df["tm_p_speed_mean"] - df["tm_b_speed_mean"]
    df["spin_diff"] = df["tm_p_spin_mean"] - df["tm_b_spin_mean"]
    df["vbreak_diff"] = df["tm_p_vbreak_mean"] - df["tm_b_vbreak_mean"]
    df["count_state"] = df["balls_before"].astype(str) + "B_" + df["strikes_before"].astype(str) + "S"
    df["is_hitter_advantage"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["is_platoon_advantage"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)  # ≈ hand_match
    df["recent_form_diff"] = df["asof_pitcher_prev1_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)
    df["recent_3g_diff"] = df["asof_pitcher_prev3_game_success_rate"].fillna(0) - df["asof_pitcher_success_rate"].fillna(0)
    global_mean_success, m = 0.55, 20.0
    n = df["asof_pitcher_n"].fillna(0)
    rate = df["asof_pitcher_success_rate"].fillna(global_mean_success)
    df["asof_pitcher_success_smoothed"] = (n * rate + m * global_mean_success) / (n + m)
    df["fastball_ratio"] = df["asof_pitcher_fastball_rate"].fillna(0)
    df["breaking_ratio"] = df["asof_pitcher_breaking_rate"].fillna(0)
    df["offspeed_ratio"] = df["asof_pitcher_offspeed_rate"].fillna(0)
    return df


# %% ---------------------------------------------------------
# 4. 나운 — 압박/매치업/콜드스타트 7개 (hand_match 원 소유자로 지정)
# ------------------------------------------------------------
def add_features_naun(df):
    df = df.copy()
    df["is_disadvantaged_count"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_runner_on"] = (df["num_runners_on"] > 0).astype(int)
    df["is_late_inning"] = (df["inning"] >= 7).astype(int)
    df["is_two_outs"] = (df["outs_before"] == 2).astype(int)
    df["pressure_score"] = (df["is_disadvantaged_count"] + df["is_runner_on"]
                            + df["is_late_inning"] + df["is_two_outs"])
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["is_cold_start"] = (df["asof_pitcher_n"] == 0).astype(int)
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)
    return df


# %% ---------------------------------------------------------
# 5. 영은 — trackman 상황별(situ) lookup (prepare_data.py)
#    ⚠️ hand_match를 여기서도 만들지만 4번(나운)과 정의가 동일 → drop
# ------------------------------------------------------------
HAND_MAP = {1: "Left", 2: "Right"}
INV_HAND_MAP = {v: k for k, v in HAND_MAP.items()}
SITU_KEY = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before", "outs_before"]

def build_situ_stats(tm_df):
    usecols = SITU_KEY + ["pitch_type_group", "zone_speed", "horz_break", "trackman_id"]
    tm = tm_df[usecols].copy() if set(usecols).issubset(tm_df.columns) else None
    if tm is None:
        print("⚠️ trackman_history.csv에 situ_stats 필요 컬럼이 없어 건너뜀")
        return None
    situ = tm.groupby(SITU_KEY).agg(
        tm_fastball_rate=("pitch_type_group", lambda s: (s == "fastball").mean()),
        tm_breaking_rate=("pitch_type_group", lambda s: (s == "breaking").mean()),
        tm_offspeed_rate=("pitch_type_group", lambda s: (s == "offspeed").mean()),
        tm_zone_speed_mean=("zone_speed", "mean"),
        tm_horz_break_mean=("horz_break", "mean"),
        tm_n=("trackman_id", "count"),
    ).reset_index()
    situ["pitcher_hand"] = situ["pitcher_hand"].map(INV_HAND_MAP)
    situ["batter_hand"] = situ["batter_hand"].map(INV_HAND_MAP)
    return situ

def add_features_situ(df, situ_stats):
    if situ_stats is None:
        return df
    df = df.merge(situ_stats, on=SITU_KEY, how="left")
    return df  # hand_match는 4번(나운)에서 이미 생성 → 여기서 재생성하지 않음


# %% ---------------------------------------------------------
# 6. 전체 통합 파이프라인
# ------------------------------------------------------------
def build_full_dataset(raw_df, tm_df):
    df = add_features_hj(raw_df, LOOKUPS_HJ, f_share_version="old")      # 해진
    df = add_features_haneul(df)                                        # 하늘

    tm_p, tm_b = build_trackman_aggregates(tm_df)
    df = add_features_eunwoo(df, tm_p, tm_b)                             # 은우

    df = add_features_naun(df)                                          # 나운

    situ_stats = build_situ_stats(tm_df)
    df = add_features_situ(df, situ_stats)                               # 영은

    return df

full = build_full_dataset(train, tm_df)
print("통합 후 shape:", full.shape)
assert full.columns.duplicated().sum() == 0, "중복 컬럼명 존재 — merge 충돌 확인 필요"
print("중복 컬럼 없음 확인 완료")

full.to_parquet(CACHE_DIR / "train_feat_ALL.parquet", index=False)
print(f"저장 완료: {CACHE_DIR / 'train_feat_ALL.parquet'}")


# %% ---------------------------------------------------------
# 7. 중복/상관 사전 점검 — 모델 태우기 전에 명백한 중복 표시만 해둔다
#    (실제 제거 여부는 9번 importance 결과까지 보고 최종 판단)
# ------------------------------------------------------------
NEW_FEATURES = [
    # 해진
    "count_pressure", "count_depth", "form_dev_3", "skill_gap", "residual", "f_share", "career_span",
    # 하늘
    "pitcher_control_ratio", "is_batter_cold_start", "season_progress", "is_experienced_mix",
    # 은우
    "tm_p_speed_mean", "tm_p_speed_std", "tm_p_spin_mean", "tm_p_vbreak_mean", "tm_p_hbreak_mean",
    "tm_b_speed_mean", "tm_b_spin_mean", "tm_b_vbreak_mean",
    "speed_diff", "spin_diff", "vbreak_diff",
    "is_hitter_advantage", "is_two_strikes", "is_full_count", "is_platoon_advantage",
    "recent_form_diff", "recent_3g_diff", "asof_pitcher_success_smoothed",
    "fastball_ratio", "breaking_ratio", "offspeed_ratio",
    # 나운
    "is_disadvantaged_count", "is_runner_on", "is_late_inning", "is_two_outs",
    "pressure_score", "hand_match", "is_cold_start", "is_first_pitch",
    # 영은은 situ_stats(tm_fastball_rate 등)를 NEW_FEATURES에 별도로 추가하지 않음
    # → 이미 은우의 tm_* 계열과 이름이 겹치지 않게 5번 섹션에서 생성되지만,
    #   같은 trackman 소스를 다른 grain(선수 단위 vs 상황 단위)으로 요약한 것이라
    #   9번 importance 결과에서 "정보 중복"인지 반드시 같이 확인할 것.
]
NEW_FEATURES = [c for c in NEW_FEATURES if c in full.columns]

print("=== 알려진 개념 중복 쌍 상관 확인 ===")
known_pairs = [
    ("hand_match", "is_platoon_advantage"),
    ("is_batter_cold_start", "is_cold_start"),
    ("is_disadvantaged_count", "is_hitter_advantage"),
]
for a, b in known_pairs:
    if a in full.columns and b in full.columns:
        r = full[[a, b]].astype(float).corr().iloc[0, 1]
        print(f"  {a:22s} vs {b:22s}  corr={r:+.4f}")

if "is_platoon_advantage" in full.columns and "hand_match" in full.columns:
    if (full["hand_match"] == full["is_platoon_advantage"]).mean() > 0.999:
        print("→ hand_match == is_platoon_advantage (완전 중복) → is_platoon_advantage 제외")
        NEW_FEATURES.remove("is_platoon_advantage")


# %% ---------------------------------------------------------
# 8. 모델 학습 — 통합 피처셋 전체로 (기존 base 피처 + 신규 전체)
#    XGBoost는 문자열 카테고리를 못 받으므로 category dtype + enable_categorical=True 사용
#    (xgboost>=1.6 필요. 구버전이면 pip install -U xgboost)
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import xgboost as xgb

base_features = [c for c in train.columns if c not in [ID, TARGET]]
categorical_extra = ["count_state"] + [c for c in CAT_COLS if c in full.columns]

all_features = list(dict.fromkeys(
    [c for c in base_features if c in full.columns] + NEW_FEATURES
))
cat_features = [c for c in categorical_extra if c in all_features]
num_features = [c for c in all_features if c not in cat_features]

X = full[all_features].copy()
y = full[TARGET]

for c in cat_features:
    X[c] = X[c].astype("category")
for c in num_features:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    tree_method="hist", enable_categorical=True,
    eval_metric="logloss", early_stopping_rounds=50,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=100,
)

p_va = model.predict_proba(X_va)[:, 1]
brier = brier_score_loss(y_va, p_va)
r = y_va.mean()
bss = max(0, 100000 * (1 - brier / (r * (1 - r))))
print(f"\n통합 피처셋 Brier Skill Score: {bss:.2f}  (brier={brier:.5f}, base_rate={r:.4f})")
print(f"best_iteration: {model.best_iteration}")


# %% ---------------------------------------------------------
# 9. Feature Importance — gain 기준 + permutation importance
# ------------------------------------------------------------
booster = model.get_booster()
gain_dict = booster.get_score(importance_type="gain")
imp_gain = pd.Series(gain_dict).reindex(all_features).fillna(0.0).sort_values(ascending=False)

print("=== Gain 기준 상위 30 ===")
print(imp_gain.head(30))

fig, ax = plt.subplots(figsize=(8, 9))
imp_gain.head(30).sort_values().plot(kind="barh", ax=ax, color="#2c3e50")
ax.set_title("Feature Importance (gain, top 30)")
plt.tight_layout(); plt.show()

from sklearn.inspection import permutation_importance
samp_idx = X_va.sample(min(50_000, len(X_va)), random_state=42).index
perm = permutation_importance(
    model, X_va.loc[samp_idx], y_va.loc[samp_idx],
    n_repeats=3, random_state=42, n_jobs=-1, scoring="neg_brier_score",
)
imp_perm = pd.Series(perm.importances_mean, index=all_features).sort_values(ascending=False)
print("\n=== Permutation Importance 상위 30 (neg_brier_score 개선폭) ===")
print(imp_perm.head(30))

print("\n=== 신규 피처(5명 전체)만 따로 본 순위 ===")
print("[gain 순위]")
print(imp_gain.reindex(NEW_FEATURES).sort_values(ascending=False))
print("\n[permutation 순위]")
print(imp_perm.reindex(NEW_FEATURES).sort_values(ascending=False))


# %% ---------------------------------------------------------
# 10. 신규 피처 간 상관 클러스터 — |corr| >= 0.9 자동 표시
# ------------------------------------------------------------
corr = full[NEW_FEATURES].astype(float).corr().abs()
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
              .stack().sort_values(ascending=False))
high_corr = pairs[pairs >= 0.9]
print("=== 중복 의심 쌍 (|corr| >= 0.9) ===")
print(high_corr if len(high_corr) else "없음")


# %% ---------------------------------------------------------
# 11. 잔차 분석
# ------------------------------------------------------------
resid_df = X_va.copy()
resid_df[TARGET] = y_va.values
resid_df["p_pred"] = p_va
resid_df["resid"] = resid_df[TARGET] - resid_df["p_pred"]
resid_df["abs_resid"] = resid_df["resid"].abs()

def residual_by(col, n_bins=5):
    s = resid_df[col]
    if s.nunique() <= 12:
        g = resid_df.groupby(col, observed=True)
    else:
        b = pd.qcut(s, n_bins, duplicates="drop")
        g = resid_df.groupby(b, observed=True)
    out = g.agg(mean_resid=("resid", "mean"), mean_abs_resid=("abs_resid", "mean"),
                actual=(TARGET, "mean"), pred=("p_pred", "mean"), n=("resid", "size"))
    return out

print("=== 세그먼트별 잔차 (모델이 못 잡는 구간 탐색) ===\n")
for c in ["pressure_score", "count_state", "is_cold_start", "season",
          "f_share", "career_span", "asof_pitcher_success_smoothed"]:
    if c not in resid_df.columns:
        continue
    print(f"[{c}]")
    print(residual_by(c).round(4))
    print()

fig, ax = plt.subplots(figsize=(5, 5))
bins = pd.qcut(resid_df["p_pred"], 10, duplicates="drop")
cal = resid_df.groupby(bins, observed=True).agg(pred=("p_pred", "mean"), actual=(TARGET, "mean"))
ax.plot(cal["pred"], cal["actual"], "o-", label="model")
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
ax.set_xlabel("예측 확률"); ax.set_ylabel("실제 성공률"); ax.legend()
ax.set_title("Calibration (통합 피처셋)")
plt.tight_layout(); plt.show()

if {"pressure_score", "is_cold_start"}.issubset(resid_df.columns):
    combo = resid_df.groupby(["pressure_score", "is_cold_start"], observed=True).agg(
        mean_abs_resid=("abs_resid", "mean"), n=("resid", "size")
    ).sort_values("mean_abs_resid", ascending=False)
    print("=== pressure_score x is_cold_start 조합별 |잔차| 상위 ===")
    print(combo.head(10).round(4))

# %% ---------------------------------------------------------
# 11-1. Permutation Importance 막대그래프 (신규 피처만)
# ------------------------------------------------------------
top_perm_new = imp_perm.reindex(NEW_FEATURES).sort_values(ascending=False).dropna()

fig, ax = plt.subplots(figsize=(7, 8))
top_perm_new.plot(kind="barh", ax=ax, color="#c0392b")
ax.invert_yaxis()
ax.set_xlabel("Permutation Importance (neg_brier_score 개선폭)")
ax.set_title("신규 피처 Permutation Importance")
plt.tight_layout(); plt.show()


# %% ---------------------------------------------------------
# 11-2. Gain vs Permutation 일치도 산점도
#    — 오른쪽 위(둘 다 높음)일수록 확실히 유효한 피처, 왼쪽 아래는 확실히 무효
# ------------------------------------------------------------
cmp_df = pd.DataFrame({
    "gain": imp_gain.reindex(all_features),
    "perm": imp_perm.reindex(all_features),
})
cmp_df["is_new"] = cmp_df.index.isin(NEW_FEATURES)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(cmp_df.loc[~cmp_df["is_new"], "gain"], cmp_df.loc[~cmp_df["is_new"], "perm"],
           alpha=0.4, label="기존 피처", color="gray")
ax.scatter(cmp_df.loc[cmp_df["is_new"], "gain"], cmp_df.loc[cmp_df["is_new"], "perm"],
           alpha=0.8, label="신규 피처", color="#e74c3c")
for name in cmp_df.sort_values("gain", ascending=False).head(8).index:
    ax.annotate(name, (cmp_df.loc[name, "gain"], cmp_df.loc[name, "perm"]), fontsize=8)
ax.set_xlabel("Gain Importance"); ax.set_ylabel("Permutation Importance")
ax.set_title("Gain vs Permutation 일치도")
ax.legend()
plt.tight_layout(); plt.show()


# %% ---------------------------------------------------------
# 11-3. 신규 피처 상관 히트맵 — 중복 쌍을 시각적으로 강조
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(full[NEW_FEATURES].astype(float).corr(), cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.3, ax=ax,
            cbar_kws={"label": "상관계수"})
ax.set_title("신규 피처 간 상관 히트맵 (진한 빨강/파랑 = 중복 의심)")
plt.tight_layout(); plt.show()


# %% ---------------------------------------------------------
# 11-4. season별 실제 vs 예측 성공률 추이 — 잔차표의 하락 추세를 시각화
# ------------------------------------------------------------
season_trend = resid_df.groupby("season", observed=True).agg(
    actual=(TARGET, "mean"), pred=("p_pred", "mean"), n=("resid", "size")
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(season_trend["season"], season_trend["actual"], "o-", label="실제 성공률", color="#2c3e50")
ax.plot(season_trend["season"], season_trend["pred"], "o--", label="예측 성공률", color="#e74c3c")
ax.set_xlabel("시즌"); ax.set_ylabel("성공률")
ax.set_title("시즌별 실제 vs 예측 성공률 추이")
ax.legend()
plt.tight_layout(); plt.show()


# %% ---------------------------------------------------------
# 11-5. (선택) SHAP summary plot — 방향성까지 보여주는 보고서용 임팩트 있는 그래프
#    pip install shap 필요. 표본이 크면 느리므로 5만 개만 샘플링.
# ------------------------------------------------------------
import shap

samp_idx2 = X_va.sample(min(50_000, len(X_va)), random_state=42).index
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_va.loc[samp_idx2])

shap.summary_plot(shap_values, X_va.loc[samp_idx2], max_display=20, show=False)
plt.title("SHAP Summary (top 20)")
plt.tight_layout(); plt.show()

# 신규 피처만 따로 보고 싶으면
shap.summary_plot(shap_values, X_va.loc[samp_idx2], max_display=20,
                   feature_names=all_features, show=False,
                   plot_type="bar")
plt.title("SHAP 평균 |영향력| (전체)")
plt.tight_layout(); plt.show()

# %% ---------------------------------------------------------
# 12. 결론 정리용 템플릿
# ------------------------------------------------------------
summary = {
    "brier_skill_score_total": round(bss, 2),
    "top10_gain": imp_gain.head(10).round(2).to_dict(),
    "new_features_gain_rank": imp_gain.reindex(NEW_FEATURES).sort_values(ascending=False).round(2).to_dict(),
    "duplicate_candidates_corr_ge_0.9": {f"{a}__{b}": round(v, 4) for (a, b), v in high_corr.items()},
    "known_duplicate_removed": ["is_platoon_advantage (== hand_match)"] if "is_platoon_advantage" not in NEW_FEATURES else [],
}
with open(ARTIFACT_DIR / "integration_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"저장 완료: {ARTIFACT_DIR / 'integration_summary.json'}")
print(json.dumps(summary, ensure_ascii=False, indent=2))

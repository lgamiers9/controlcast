# ============================================================
# 피처 생성 — 해진 담당 7종
#
#   행 내부 연산 (5개)
#     count_pressure   balls - strikes
#     count_depth      balls + strikes
#     form_dev_3       prev3_success - asof_success
#     skill_gap        asof_pitcher_success - asof_batter_success
#     residual         1 - (success + middle + reverse)
#
#   정적 조회 테이블 (2개)
#     f_share          투수별 퓨처스 노출 비율
#     career_span      season - debut_season
#
# 규정 준수
#   조회 테이블은 학습 데이터에서만 산출하여 파일로 저장하고,
#   추론 시에는 pitcher_id 기준 조인만 수행한다.
#   평가 데이터의 다른 행이나 전체 분포는 일절 참조하지 않는다.
#
# 실행 환경: Colab (# %% 셀 구분)
# ============================================================


# %% ---------------------------------------------------------
# 1. 설정
# ------------------------------------------------------------
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")

BASE = Path("/content/drive/MyDrive/open")
DATA_DIR = BASE / "data"
CACHE_DIR = BASE / "cache"
ARTIFACT_DIR = BASE / "artifacts"; ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

NEW_FEATURES = ["count_pressure", "count_depth", "form_dev_3",
                "skill_gap", "residual", "f_share", "career_span"]

train_full = pd.read_parquet(CACHE_DIR / "train.parquet")
print("원본:", train_full.shape)


# %% ---------------------------------------------------------
# 2. 조회 테이블 산출  ★ 오염 제거 '전'의 데이터를 사용
#
#    f_share 는 asof_* 에 섞인 퓨처스 오염의 크기를 나타내는 지표다.
#    운영 측은 asof_* 를 R·F 통합으로 계산했으므로, 오염의 원천인
#    2019~2022 F 등판이 반드시 집계에 포함되어야 한다.
#    오염 제거 후 데이터로 계산하면 지표의 의미가 사라진다.
# ------------------------------------------------------------
is_f = train_full["game_type"] == "F"

# 전체 기간 노출 비율
f_share_all = (train_full.assign(_f=is_f.astype("int8"))
                         .groupby("pitcher_id", observed=True)["_f"].mean())

# 오염 구간(2019~2022) 노출 비율 — 편향과 더 직접적으로 연결
contaminated = is_f & (train_full["season"] <= 2022)
f_share_old = (train_full.assign(_f=contaminated.astype("int8"))
                         .groupby("pitcher_id", observed=True)["_f"].mean())

# 데뷔 시즌 (관측 시작 2019에 절단됨)
debut_season = train_full.groupby("pitcher_id", observed=True)["season"].min()

LOOKUPS = {
    "f_share_all": f_share_all.to_dict(),
    "f_share_old": f_share_old.to_dict(),
    "debut_season": debut_season.to_dict(),
    # 미등록 투수(2025 신규) 대체값
    "fallback_f_share": float(f_share_all.median()),
    "fallback_debut": int(debut_season.median()),
}

print(f"등록 투수 {len(f_share_all)}명")
print(f"f_share_all  중앙값 {f_share_all.median():.4f} / 평균 {f_share_all.mean():.4f}")
print(f"f_share_old  중앙값 {f_share_old.median():.4f} / 평균 {f_share_old.mean():.4f}")
print(f"두 버전 상관 {np.corrcoef(f_share_all, f_share_old)[0, 1]:.4f}")
print(f"\ndebut_season 분포:\n{debut_season.value_counts().sort_index()}")


# %% ---------------------------------------------------------
# 3. 피처 생성 함수
#    train / test 모두 이 함수 하나만 통과시킨다.
# ------------------------------------------------------------
def add_features(df, lookups, f_share_version="old"):
    """해진 담당 7개 피처를 추가한다.

    df       : train 또는 test (원본 컬럼 유지)
    lookups  : 2번에서 만든 조회 테이블
    f_share_version : "old"(2019~2022 노출) 또는 "all"(전체 기간 노출)
    """
    out = df.copy()

    # --- 행 내부 연산 ---
    out["count_pressure"] = (out["balls_before"] - out["strikes_before"]).astype("int8")
    out["count_depth"] = (out["balls_before"] + out["strikes_before"]).astype("int8")

    out["form_dev_3"] = (out["asof_pitcher_prev3_game_success_rate"]
                         - out["asof_pitcher_success_rate"]).astype("float32")

    out["skill_gap"] = (out["asof_pitcher_success_rate"]
                        - out["asof_batter_success_rate"]).astype("float32")

    out["residual"] = (1.0
                       - out["asof_pitcher_success_rate"]
                       - out["asof_pitcher_middle_rate"]
                       - out["asof_pitcher_reverse_rate"]).astype("float32")

    # --- 정적 조회 테이블 조인 ---
    key = f"f_share_{f_share_version}"
    pid = out["pitcher_id"].astype("int64")

    out["f_share"] = (pid.map(lookups[key])
                         .fillna(lookups["fallback_f_share"])
                         .astype("float32"))

    debut = pid.map(lookups["debut_season"]).fillna(lookups["fallback_debut"])
    out["career_span"] = (out["season"].astype("int16") - debut).astype("int16")

    return out


train = add_features(train_full, LOOKUPS, f_share_version="old")
print(train[NEW_FEATURES].dtypes)


# %% ---------------------------------------------------------
# 4. 분포와 결측 확인
# ------------------------------------------------------------
print("=== 기술통계 ===")
print(train[NEW_FEATURES].describe().T.round(4))

print("\n=== 결측률 (%) ===")
print((train[NEW_FEATURES].isna().mean() * 100).round(4))

for c in ["count_pressure", "count_depth", "career_span"]:
    print(f"\n=== {c} 분포 ===")
    print(train[c].value_counts().sort_index())

# residual 은 이론상 음수가 될 수 있음 (세 유형 중첩 시)
print(f"\nresidual 음수 비율: {(train['residual'] < 0).mean():.4f}")
print(f"residual 범위: {train['residual'].min():.4f} ~ {train['residual'].max():.4f}")


# %% ---------------------------------------------------------
# 5. 분위수별 기저율 — 타겟과의 단변량 관계
#    상관계수는 선형 관계만 잡으므로 분위수별 평균을 함께 본다.
# ------------------------------------------------------------
def quantile_report(df, col, n_bins=5):
    """연속형은 분위수, 이산형은 값 자체로 그룹화하여 기저율 산출"""
    s = df[col]
    if s.nunique() <= 12:
        g = df.groupby(col, observed=True)[TARGET].agg(["mean", "size"])
    else:
        try:
            b = pd.qcut(s, n_bins, duplicates="drop")
        except ValueError:
            return None
        g = df.groupby(b, observed=True)[TARGET].agg(["mean", "size"])
    g["비중"] = g["size"] / len(df)
    lo, hi = g["mean"].min(), g["mean"].max()
    g.attrs["spread"] = (hi - lo) / lo if lo > 0 else np.nan
    return g


print("=== 단변량 강도 (최대/최소 그룹 상대차) ===\n")
strength = {}
for c in NEW_FEATURES:
    g = quantile_report(train, c)
    if g is None:
        continue
    strength[c] = g.attrs["spread"]
    print(f"[{c}]  상대차 {g.attrs['spread']:.1%}")
    print(g.round(4))
    print()

strength = pd.Series(strength).sort_values(ascending=False)
print("=== 강도 순위 ===")
print(strength.map(lambda v: f"{v:.1%}"))

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.barplot(x=strength.values * 100, y=strength.index, ax=ax, color="#2c3e50")
ax.set_xlabel("최대/최소 그룹 상대차 (%)")
ax.set_title("신규 피처 단변량 강도")
plt.tight_layout(); plt.show()

from scipy.stats import spearmanr

print("=== season 상관 ===")
for c in NEW_FEATURES:
    r = train[[c, "season"]].astype(float).corr().iloc[0, 1]
    flag = "  ⚠️ 교락 의심" if abs(r) >= 0.3 else ""
    print(f"  {c:18s} {r:+.3f}{flag}")

print("\n=== 시즌별 5분위 순위상관 ===")
for c in NEW_FEATURES:
    out = []
    for s in [2021, 2022, 2023, 2024]:
        d = train[(train["season"] == s) & train[c].notna()]
        if len(d) < 20_000 or d[c].nunique() < 5:
            out.append(None); continue
        q = pd.qcut(d[c], 5, duplicates="drop", labels=False)
        g = d.groupby(q, observed=True)[TARGET].mean()
        out.append(round(spearmanr(g.index, g.values)[0], 2))
    vals = [v for v in out if v is not None and v != 0]
    ok = len(set(np.sign(vals))) == 1 and min(map(abs, vals)) >= 0.6 if vals else False
    print(f"  {c:18s} {out}  {'✅' if ok else '❌'}")


# %% ---------------------------------------------------------
# 6. 기존 컬럼과의 중복 확인
#    상관이 0.9 이상이면 정보가 겹쳐 추가 가치가 없다.
# ------------------------------------------------------------
existing = [c for c in train_full.columns
            if c not in [ID, TARGET] + CAT_COLS
            and pd.api.types.is_numeric_dtype(train_full[c])]

samp = train.sample(min(300_000, len(train)), random_state=42)

print("=== 신규 × 기존 최고 상관 ===")
for c in NEW_FEATURES:
    cors = samp[existing].corrwith(samp[c].astype(float)).abs().dropna()
    if cors.empty:
        continue
    top = cors.sort_values(ascending=False).head(3)
    flag = "  ⚠️ 중복 의심" if top.iloc[0] >= 0.90 else ""
    print(f"[{c}]{flag}")
    for k, v in top.items():
        print(f"    {v:.4f}  {k}")
    print()

print("=== 신규 피처 간 상관 ===")
print(samp[NEW_FEATURES].astype(float).corr().round(3))

del samp; gc.collect()


# %% ---------------------------------------------------------
# 7. 조회 테이블 저장
#    추론 시 재계산하면 규정 위반이므로 반드시 파일로 남긴다.
#    평가 시에는 train.csv를 참고할 수 없기 때문.
# ------------------------------------------------------------
import joblib

joblib.dump(LOOKUPS, ARTIFACT_DIR / "feature_lookups_hj.joblib", compress=3)
print(f"저장 완료: {ARTIFACT_DIR / 'feature_lookups_hj.joblib'}")

# 추론 시 사용법
#   LOOKUPS = joblib.load("model/feature_lookups_hj.joblib")
#   test = add_features(test, LOOKUPS, f_share_version="old")


# %% ---------------------------------------------------------
# 8. 피처 추가된 데이터 캐싱
#    모델 실험 스크립트에서 바로 불러 쓰기 위함
# ------------------------------------------------------------
train.to_parquet(CACHE_DIR / "train_feat_hj.parquet", index=False)
print(f"저장 완료: {CACHE_DIR / 'train_feat_hj.parquet'}")
print(f"shape {train.shape}  ({train.memory_usage(deep=True).sum()/1024**2:,.0f} MB)")
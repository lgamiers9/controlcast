"""02_feature_engineering.ipynb에서 검증한 7개 피처.

03_train_final.ipynb(학습)와 submit_v1/script.py(추론)가 이 모듈을 공유해서
학습·추론 간 피처 로직이 어긋나지 않게 한다.
"""


def add_engineered_features(df):
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
    return df


ENGINEERED_COLS = [
    "is_disadvantaged_count", "is_runner_on", "is_late_inning", "is_two_outs",
    "pressure_score", "hand_match", "is_cold_start",
]

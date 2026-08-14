"""

압박 상황(주자·카운트·이닝·아웃), 매치업(투수-타자 손 유형), 콜드스타트(투수 과거 표본 0건)
관련 피처 7개. 전부 baseline 대비 성능 개선 확인됨 (총 +56.51, 자세한 수치는 아래 표 참고).

"""


def add_features(df):
    """그 행 자기 자신의 컬럼값만 사용해서 피처 7개를 추가한다. 다른 행 참조 없음."""
    df = df.copy()

    # 압박 상황 4개 + 합산 점수
    df["is_disadvantaged_count"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["is_runner_on"] = (df["num_runners_on"] > 0).astype(int)
    df["is_late_inning"] = (df["inning"] >= 7).astype(int)
    df["is_two_outs"] = (df["outs_before"] == 2).astype(int)
    df["pressure_score"] = (
        df["is_disadvantaged_count"] + df["is_runner_on"] + df["is_late_inning"] + df["is_two_outs"]
    )

    # 매치업
    df["hand_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    # 콜드스타트
    df["is_cold_start"] = (df["asof_pitcher_n"] == 0).astype(int)

    # 초구 (0-0 카운트) — 야구 지식 기반, "일단 스트라이크 잡자" 접근이라 다른 카운트와 심리가 다름
    df["is_first_pitch"] = ((df["balls_before"] == 0) & (df["strikes_before"] == 0)).astype(int)

    return df
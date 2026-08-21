# 📋 대회 규칙 요약 — 투구 제구 성공 확률 예측 (LG Aimers 9기 Phase 2)

> 원문 공지 전문을 정리한 요약본. **제출 직전 체크리스트는 맨 아래 참고.**
> 세부 문구가 헷갈리면 반드시 대회 페이지 원문(평가 탭 / 코드 제출 가이드)을 다시 확인할 것.

---

## 1. 문제 정의

- **Task**: 투구 단위(pitch-level)로 `control_success`(제구 성공 확률, 0~1)를 예측하는 이진 분류/확률 예측 문제
- **Target 정의**: 학습 데이터에서 제구 성공 = 1, 실패 = 0
- **제구 실패로 정의되는 3가지 케이스** (그 외는 전부 성공):
  1. 스트라이크존 가운데 부근으로 들어간 공
  2. 스트라이크존에서 크게 벗어난 공
  3. 포수의 요구 방향과 반대로 들어간 공
- **중요 제약**: 예측 시점에 "투구 이전에 확인 가능한 정보"만 사용 가능 (투구 결과 관련 정보 유출 금지)
- 트랙맨(Trackman) 데이터는 2019~2024년 **과거** 투구 특성을 참고하는 보조 데이터

---

## 2. 평가 산식 — Brier Skill Score

```
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)                         # 전체 평가 데이터의 실제 평균 제구 성공률 (비공개)
평균 제구율 Brier Score = r × (1 - r)
Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
```

- `src/score.py`의 `brier_skill_score()` / `decompose()`가 동일 산식 구현 + REL/RES 분해(보정 문제 vs 분별력 문제 진단용)
- Public Score = 전체 테스트 데이터 100% / Private Score = 대회 종료 시점 Public Score와 동일 기준

---

## 3. 통과 기준 (수료 조건)

- Phase1 이수 + **Phase2 Public Score 549.51 이상**
- 이 기준점은 운영진 베이스라인 코드를 운영진 평가 환경에서 실행한 점수 기준
- 1차 평가: Private Score 100% (동점자는 기존 리더보드 순위 산정 방식)
- 2차 평가: Phase3(오프라인) 진출 희망 시 코드+PPT 제출 → 검증 통과한 Private 상위 약 100명만 진출

---

## 4. 제출 파일 구조 (submit.zip) — **디렉토리/파일명 완전 일치 필수**

```
submit.zip
├── model/              # 모델 가중치 저장 (예: model.pt)
├── script.py           # 추론 실행 코드 (평가 서버가 자동 실행)
└── requirements.txt    # pip install -r requirements.txt 로 설치되는 형태
```

⚠️ **최상위에 불필요한 폴더가 하나 더 감싸져 있으면 설치 오류** — zip 압축 시 구조 그대로 최상위에 오도록 주의.

### 평가 서버가 자동으로 추가하는 것 (직접 만들 필요 없음, 건드리면 안 됨)
```
├── data/                    # 평가용 실데이터, 자동 마운트, 읽기 전용 (쓰기/수정 불가)
│      ├── test.csv
│      └── sample_submission.csv   # row_id, control_success(placeholder) — 공식 baseline이 이 순서를 기준으로 병합함
└── output/submission.csv    # script.py가 반드시 이 경로·이 파일명으로 결과 저장해야 함
```

> `data/` 실제 구성은 공식 baseline([`baseline_submit/`](./baseline_submit/))의 `script.py` 코드로 확인함 — `test.csv`와 `sample_submission.csv` 둘 다 읽음.

---

## 5. 실행 환경 제약

| 항목 | 제한 |
|---|---|
| 전체 추론 실행 시간 | ≤ 10분 (245,789개 샘플) |
| 패키지 설치 시간 | ≤ 10분 |
| 제출 파일(zip) 용량 | ≤ 10GB (압축 해제 후 ≤ 32GB) |
| 인터넷 연결 | 패키지 설치 외 전면 차단 (모델/가중치 다운로드 코드 작동 안 함 → 로컬에 미리 포함시켜야 함) |
| 서버 사양 | 6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM, Ubuntu 22.04.5, Python 3.11.15, CUDA 12.8 |

### 기본 설치된 패키지 (requirements.txt에 버전 다르게 넣으면 설치 에러 위험 → 웬만하면 그대로 사용)
`torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.8.0`,
`joblib==1.5.3`, `threadpoolctl==3.6.0`, `narwhals==2.21.2`, `transformers==4.46.3`, `accelerate==1.9.0`,
`sentencepiece==0.1.99`, `regex==2023.12.25`, `tqdm==4.66.4`, `loguru==0.7.2`, `pyyaml==6.0.1`, `rich==13.7.1`

시스템 패키지: git, build-essential, python3.11(-dev/-venv), python3-pip, libffi-dev, libblas3, liblapack3,
libomp-dev, tzdata, unzip, p7zip-full, gfortran, libatlas-base-dev, default-jre-headless, cmake,
pkg-config, ninja-build, libgl1, libglib2.0-0

---

## 6. 오류 종류와 일일 제출 횟수 반영 여부

| 오류 유형 | 원인 | 일일 제출 횟수 반영 |
|---|---|---|
| **설치 오류** | zip 내부 구조 불일치, 패키지 설치 실패 | ❌ 반영 안 됨 |
| **제출 오류** | script.py 실행 중 발생하는 모든 오류 | ✅ 반영됨 (아껴서 제출!) |

→ **로컬에서 zip 구조·requirements 설치·script.py 실행을 미리 검증**해서 "설치 오류"는 몰라도 "제출 오류"는 최대한 피할 것.

---

## 7. 금지/제한 사항

1. **사전학습 모델**: 누구나 접근 가능 + 비상업 이용 허용 라이선스(MIT, Apache 2.0 등)인 것만 사용 가능
2. **외부 API 금지**: OpenAI API, Gemini API 등 원격 서버 기반 API 전면 불가 (로컬 실행·재현 가능해야 함)
3. **외부 데이터 금지**: 공식 제공 데이터 외 사용 불가
4. **행 단위 독립 추론 원칙**: test.csv의 각 행은 해당 행의 입력값 + 공식 학습 데이터만으로 예측해야 함.
   테스트 데이터 전체 분포나 다른 행을 이용해 특정 행 예측값을 보정/생성하는 방식(예: 타깃 리키지, 그룹 통계로 후처리)은 금지

---

## 8. 팀 규칙

- 개인 또는 팀(최대 5명) 참가, 동일인 중복 등록(개인+팀 등) 불가
- 코드/PPT 제출 시 `dacon@dacon.io`, 코드 확장자 `.py`/`.ipynb`, 인코딩 UTF-8

---

## ✅ 제출 직전 체크리스트

- [ ] `model/`, `script.py`, `requirements.txt` 세 가지가 zip **최상위**에 바로 있음 (불필요한 상위 폴더로 한번 더 감싸지 않았는지 확인)
- [ ] `script.py`가 `data/` (읽기 전용, 서버가 마운트)에서 데이터를 읽고, `output/submission.csv`로 결과 저장하는지 확인
- [ ] `requirements.txt`에 서버 기본 설치 패키지와 버전 충돌 없는지 확인 (가능하면 기본 목록은 아예 명시 안 함)
- [ ] 인터넷 다운로드하는 코드(사전학습 가중치 자동 다운로드 등) 없는지 확인 — 전부 `model/` 안에 미리 포함
- [ ] 로컬에서 `pip install -r requirements.txt` → `python script.py` 전체 파이프라인이 10분 내 끝나는지 확인
- [ ] test.csv 각 행을 독립적으로 예측하는지 확인 (전체 분포 이용한 보정/후처리 없는지)
- [ ] Public Score 549.51 이상인지 확인 (LG Aimers 수료 기준선)

---

## 🔎 자동 검사 스크립트

제출 전 `check_submission.py`로 구조·용량·인터넷 접근·데이터 누수 의심 패턴을 자동 점검할 수 있음.

```bash
cd notebooks/yeongeun/docs
python check_submission.py path/to/submit.zip                       # 구조·정적 스캔만
python check_submission.py path/to/submit.zip --run --data ../data  # 실제 실행 + 결과 검증까지
```

⚠️ 정적 스캔(인터넷 접근 패턴, 데이터 누수 패턴)은 **휴리스틱**이라 오탐·누락이 있을 수 있음.
WARN이 뜨면 자동으로 무시하지 말고 반드시 해당 코드를 직접 읽고 판단할 것. FAIL은 제출 전 무조건 해결.

---

## 참고

- **공식 baseline (진짜 동작하는 예제, 우선 참고)**: [`baseline_submit/`](./baseline_submit/) — `model/rf.pkl` + `script.py` + `requirements.txt`
- 직접 짠 뼈대 코드/구조(TODO 채우는 용도): [`submit_template/`](./submit_template/)
- 제출 전 자동 검사는 [`check_submission.py`](./check_submission.py) 참고
- BSS 계산/분해 함수는 [`src/score.py`](../../../src/score.py) 참고

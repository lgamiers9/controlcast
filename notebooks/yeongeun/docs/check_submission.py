"""check_submission.py — submit.zip 제출 전 구조/규칙 검증 스크립트.

docs/COMPETITION_RULES.md 의 제출 규칙(4~7번 항목)을 자동으로 점검한다.
정적 스캔(4, 5번)은 휴리스틱이라 오탐/누락이 있을 수 있음 — 마지막 판단은 사람이 직접 코드를 읽고 해야 함.
이 스크립트가 "통과"라고 해도 대회 규칙 위반이 아니라는 보장은 아님. 반드시 COMPETITION_RULES.md 원문도 재확인.

사용법:
    python check_submission.py path/to/submit.zip
    python check_submission.py path/to/submit.zip --run --data ./data   # 실제 실행까지 검증 (로컬 mock data 필요)

검사 항목:
    1. zip 최상위 구조: {model/, script.py, requirements.txt} 정확히 일치 (여분 폴더 감쌈 여부)
    2. zip / 압축해제 용량 (10GB / 32GB 이내)
    3. requirements.txt — 서버 기본 설치 패키지와 버전 충돌
    4. 인터넷 접근·외부 API 호출 패턴 정적 스캔 (동봉된 모든 .py 대상)
    5. 데이터 누수 의심 패턴 정적 스캔 (test 전체 통계로 후처리하는 코드 등)
    6. (--run) 실제 script.py 실행 -> 소요시간, output/submission.csv 형식·값 검증
"""

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

# Windows 콘솔 기본 인코딩(cp949)에서 이모지/유니코드 출력 깨짐 방지
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_TOP_LEVEL = {"model", "script.py", "requirements.txt"}

ZIP_SIZE_LIMIT_GB = 10
EXTRACTED_SIZE_LIMIT_GB = 32
RUNTIME_LIMIT_SEC = 10 * 60
INSTALL_LIMIT_SEC = 10 * 60

PREINSTALLED = {
    "torch": "2.7.1+cu128", "pandas": "2.0.3", "numpy": "1.26.4", "scipy": "1.15.3",
    "scikit-learn": "1.8.0", "joblib": "1.5.3", "threadpoolctl": "3.6.0", "narwhals": "2.21.2",
    "transformers": "4.46.3", "accelerate": "1.9.0", "sentencepiece": "0.1.99",
    "regex": "2023.12.25", "tqdm": "4.66.4", "loguru": "0.7.2", "pyyaml": "6.0.1", "rich": "13.7.1",
}

NET_PATTERNS = [
    (r"^\s*(import|from)\s+(requests|boto3|openai)\b", "네트워크/외부 API 패키지 import — 실제 사용처 확인 필요"),
    (r"\brequests\.(get|post|put|delete)\s*\(", "requests 로 외부 요청"),
    (r"\burllib\.request\b", "urllib 로 외부 요청"),
    (r"\bhttp\.client\b", "http.client 사용"),
    (r"\bsocket\.\w+\s*\(", "raw socket 사용"),
    (r"\bftplib\b", "ftplib 사용"),
    (r"\bboto3\b", "boto3(AWS) 사용 — 외부 스토리지 접근 의심"),
    (r"\bwget\b|\bcurl\b", "wget/curl 호출"),
    (r"huggingface_hub|snapshot_download", "huggingface_hub 다운로드 — 로컬 캐시 미리 없으면 인터넷 필요"),
    (r"from_pretrained\s*\(", "from_pretrained() — 로컬 경로 지정 아니면 인터넷에서 받아옴"),
    (r"\bopenai\b", "OpenAI API 사용 — 외부 API 금지 규칙 위반"),
    (r"google\.generativeai|genai\.|gemini", "Gemini/Google Generative AI 사용 — 외부 API 금지 규칙 위반"),
    (r"\bpip\s+install\b", "런타임 중 pip install 실행 — 인터넷 필요, 설치는 requirements.txt로만"),
]

LEAKAGE_PATTERNS = [
    (r"\btest\w*\s*\.\s*groupby\s*\(", "test 데이터 전체에 groupby — 행 간 정보 이용 의심"),
    (r"\btest\w*\s*\.\s*rank\s*\(", "test 데이터 전체에 rank() — 분포 기반 보정 의심"),
    (r"\btest\w*\s*\.\s*rolling\s*\(", "test 데이터 전체에 rolling() — 다른 행 정보 이용 의심"),
    (r"\btest\w*\s*\.\s*transform\s*\(", "test 데이터 전체에 transform() — 그룹 통계 후처리 의심"),
    (r"qcut\s*\(\s*\w*test", "test 예측값/피처를 qcut — 전체 분포 기반 보정 의심"),
    (r"\.fit\s*\(\s*[^)]*test", "test 데이터에 fit() — 정보 유출 의심 (스케일러/인코더 등은 train에만 fit)"),
    (r"pd\.concat\s*\(\s*\[.*train.*test", "train+test concat 후 통계 계산 — 누수 의심, 수동 확인 필요"),
]


def check_zip_structure(zf: zipfile.ZipFile):
    issues = []
    names = [n for n in zf.namelist() if n and not n.endswith("/") or n.count("/") <= 1]
    top_levels = set()
    for n in zf.namelist():
        n = n.strip("/")
        if not n:
            continue
        top_levels.add(n.split("/")[0])

    missing = REQUIRED_TOP_LEVEL - top_levels
    extra = top_levels - REQUIRED_TOP_LEVEL
    if missing:
        issues.append(("FAIL", f"필수 항목 누락: {sorted(missing)}"))
    if extra:
        issues.append(("FAIL", f"최상위에 불필요한 항목 존재(여분 폴더로 감쌌을 가능성): {sorted(extra)}"))
    if not missing and not extra:
        issues.append(("OK", "zip 최상위 구조가 model/, script.py, requirements.txt 와 정확히 일치"))
    return issues


def check_sizes(zip_path):
    issues = []
    zip_gb = os.path.getsize(zip_path) / (1024 ** 3)
    if zip_gb > ZIP_SIZE_LIMIT_GB:
        issues.append(("FAIL", f"zip 용량 {zip_gb:.2f}GB > {ZIP_SIZE_LIMIT_GB}GB 제한 초과"))
    else:
        issues.append(("OK", f"zip 용량 {zip_gb:.2f}GB (제한 {ZIP_SIZE_LIMIT_GB}GB 이내)"))

    with zipfile.ZipFile(zip_path) as zf:
        extracted_gb = sum(i.file_size for i in zf.infolist()) / (1024 ** 3)
    if extracted_gb > EXTRACTED_SIZE_LIMIT_GB:
        issues.append(("FAIL", f"압축 해제 용량 {extracted_gb:.2f}GB > {EXTRACTED_SIZE_LIMIT_GB}GB 제한 초과"))
    else:
        issues.append(("OK", f"압축 해제 용량 {extracted_gb:.2f}GB (제한 {EXTRACTED_SIZE_LIMIT_GB}GB 이내)"))
    return issues


def check_requirements(req_text):
    issues = []
    for line in req_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-.]+)\s*==\s*([A-Za-z0-9_.+\-]+)", line)
        if not m:
            issues.append(("WARN", f"버전 미고정 라인 (재현성 위험): {line}"))
            continue
        name, ver = m.group(1).lower(), m.group(2)
        if name in PREINSTALLED and ver != PREINSTALLED[name]:
            issues.append((
                "WARN",
                f"{name}=={ver} 가 서버 기본 버전({PREINSTALLED[name]})과 다름 — 설치 충돌/에러 가능, "
                f"꼭 필요한 게 아니면 requirements.txt에서 빼는 걸 권장",
            ))
    if not issues:
        issues.append(("OK", "requirements.txt — 기본 패키지와 버전 충돌 없음"))
    return issues


def scan_patterns(py_path, patterns, label):
    issues = []
    try:
        text = open(py_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return issues
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, desc in patterns:
            if re.search(pattern, line):
                issues.append(("WARN", f"[{label}] {py_path}:{lineno} — {desc}\n      > {line.strip()}"))
    return issues


def find_py_files(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def static_scan(extract_dir):
    issues = []
    py_files = list(find_py_files(extract_dir))
    if not py_files:
        issues.append(("FAIL", "동봉된 .py 파일을 찾을 수 없음"))
        return issues

    net_hits, leak_hits = [], []
    for py in py_files:
        net_hits += scan_patterns(py, NET_PATTERNS, "인터넷/외부API 의심")
        leak_hits += scan_patterns(py, LEAKAGE_PATTERNS, "데이터누수 의심")

    if net_hits:
        issues += net_hits
    else:
        issues.append(("OK", "인터넷 접근/외부 API 호출 패턴 없음 (정적 스캔 기준)"))

    if leak_hits:
        issues += leak_hits
    else:
        issues.append(("OK", "데이터 누수 의심 패턴 없음 (정적 스캔 기준, 로직 검토는 별도로 필요)"))
    return issues


def run_and_validate(extract_dir, mock_data_dir):
    issues = []
    data_dir = os.path.join(extract_dir, "data")
    output_dir = os.path.join(extract_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if mock_data_dir:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        shutil.copytree(mock_data_dir, data_dir)
    elif not os.path.exists(data_dir):
        issues.append(("WARN", "--data 없이 --run 호출 & data/ 없음 -> script.py 실행 생략"))
        return issues

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "script.py"], cwd=extract_dir,
        capture_output=True, text=True, timeout=RUNTIME_LIMIT_SEC + 30,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        issues.append(("FAIL", f"script.py 실행 실패 (exit={result.returncode})\n{result.stderr[-2000:]}"))
        return issues

    if elapsed > RUNTIME_LIMIT_SEC:
        issues.append(("FAIL", f"실행 시간 {elapsed:.1f}s > {RUNTIME_LIMIT_SEC}s 제한 초과"))
    else:
        issues.append(("OK", f"실행 시간 {elapsed:.1f}s (제한 {RUNTIME_LIMIT_SEC}s 이내)"))

    sub_path = os.path.join(output_dir, "submission.csv")
    if not os.path.exists(sub_path):
        issues.append(("FAIL", "output/submission.csv 가 생성되지 않음"))
        return issues

    import pandas as pd
    sub = pd.read_csv(sub_path)
    if "control_success" not in sub.columns:
        issues.append(("FAIL", "submission.csv 에 control_success 컬럼이 없음"))
    else:
        p = sub["control_success"]
        if p.isna().any():
            issues.append(("FAIL", f"control_success 에 NaN {p.isna().sum()}개 존재"))
        if (p < 0).any() or (p > 1).any():
            issues.append(("FAIL", "control_success 값이 [0, 1] 범위를 벗어남"))
        if not p.isna().any() and (p >= 0).all() and (p <= 1).all():
            issues.append(("OK", "control_success 값 범위 [0, 1] 정상, NaN 없음"))

    sample_path = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        if set(sub[id_col]) != set(sample[id_col]):
            issues.append(("FAIL", "submission.csv 의 row_id 집합이 sample_submission.csv 와 다름"))
        elif len(sub) != len(sample):
            issues.append(("FAIL", f"행 수 불일치: submission {len(sub)} vs sample {len(sample)}"))
        else:
            issues.append(("OK", "row_id 구성이 sample_submission.csv 와 정확히 일치"))
    else:
        issues.append(("WARN", "data/sample_submission.csv 없음 — row_id 정합성 비교 생략"))

    return issues


def print_report(sections):
    fail_count = warn_count = 0
    for title, issues in sections:
        print(f"\n=== {title} ===")
        for level, msg in issues:
            mark = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[level]
            print(f"{mark} [{level}] {msg}")
            if level == "FAIL":
                fail_count += 1
            elif level == "WARN":
                warn_count += 1
    print(f"\n{'=' * 40}")
    print(f"FAIL: {fail_count}  WARN: {warn_count}")
    if fail_count:
        print("❌ 제출 전 FAIL 항목부터 반드시 해결할 것.")
    elif warn_count:
        print("⚠️  WARN 항목은 자동 판단이 어려운 부분 — 사람이 직접 코드를 읽고 규칙 위반 여부 확인 필요.")
    else:
        print("✅ 자동 검사 통과. 그래도 COMPETITION_RULES.md 체크리스트로 마지막 확인 권장.")
    return fail_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--run", action="store_true", help="script.py 실제 실행까지 검증")
    ap.add_argument("--data", default=None, help="로컬 mock data 폴더 경로 (--run 시 사용, train/test/sample_submission 등)")
    args = ap.parse_args()

    sections = []
    with zipfile.ZipFile(args.zip_path) as zf:
        sections.append(("1. zip 구조", check_zip_structure(zf)))
        sections.append(("2. 용량 제한", check_sizes(args.zip_path)))

        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            req_path = os.path.join(tmp, "requirements.txt")
            if os.path.exists(req_path):
                sections.append(("3. requirements.txt", check_requirements(open(req_path, encoding="utf-8").read())))
            sections.append(("4-5. 정적 스캔 (인터넷 접근 / 데이터 누수 의심)", static_scan(tmp)))

            if args.run:
                sections.append(("6. 실행 검증", run_and_validate(tmp, args.data)))

    fail_count = print_report(sections)
    sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
적용기 (apply) — probe의 판정을 LocalDesk 실제 설정에 반영한다.

하는 일
  1. probe_result.json을 읽어 답변용/요약용 모델과 GPU 적재 층수를 판정과 같은
     기준으로 다시 도출한다 (probe.py의 판정 기준 상수를 그대로 쓴다).
  2. 판정 모델이 설치돼 있지 않으면 디스크 여유를 먼저 확인하고 ollama pull로 받는다.
     여유가 모델 크기보다 부족하면 받지 않고 부족분을 리포트에 남기고 멈춘다.
  3. 적용 전 local_settings.json을 measure/backup/에 백업하고, qa_model/memo_model만
     판정값으로 바꾼다 (mask_level, external 같은 다른 키는 그대로 둔다).
  4. 적재 층수가 바뀌어야 하면 실행.bat의 LOCAL_AGENT_NUM_GPU 값만 숫자로 교체한다.
     실행.bat는 순수 ASCII + CRLF를 유지한다 (숫자만 바꾸므로 인코딩이 깨지지 않는다).
  5. 적용 전후 비교를 apply_report.md와 apply_result.json으로 남긴다.

종료 코드: 0=적용 완료(변경 없음 포함), 2=디스크 부족으로 중단, 1=그 외 실패.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OLLAMA = "http://localhost:11434"
SETTINGS_FILE = os.path.join(ROOT, "local_settings.json")
BAT_FILE = os.path.join(ROOT, "실행.bat")
BACKUP_DIR = os.path.join(HERE, "backup")

# probe.py와 같은 판정 기준 (측정값이 아니라 판단 기준)
ANSWER_MIN_TPS = 20.0
SUMMARY_MIN_TPS = 40.0
FULL_GPU_RATIO = 0.90
PROBE_NUM_GPU = 32  # probe가 전층 적재를 확인할 때 쓴 지정값이자 실행.bat의 현행값

DEFAULT_QA = "exaone3.5:7.8b"    # app.py의 초기값과 동일
DEFAULT_MEMO = "exaone3.5:2.4b"  # app.py의 초기값과 동일
# 이 PC에서 실측한 설치 크기(tags 기준). 미설치 모델의 디스크 판정에만 쓴다.
KNOWN_SIZE_GB = {"exaone3.5:7.8b": 4.77, "exaone3.5:2.4b": 1.64, "bge-m3": 1.16, "bge-m3:latest": 1.16}
DISK_MARGIN_GB = 1.0  # pull 중 임시 파일 등을 감안한 여유


def derive_recommendation(probe):
    """probe_result.json의 실측에서 구성 판정을 다시 도출한다."""
    speeds = {s["model"]: s for s in probe.get("speeds", []) if s.get("tokens_per_sec")}
    sizes = {m["name"]: m["size_gb"] for m in probe.get("ollama", {}).get("models", [])}
    reasons = []
    big = speeds.get("exaone3.5:7.8b", {})
    small = speeds.get("exaone3.5:2.4b", {})
    qa = None
    if big.get("tokens_per_sec", 0) >= ANSWER_MIN_TPS:
        qa = "exaone3.5:7.8b"
        reasons.append(f"답변용 {qa}: 실측 {big['tokens_per_sec']} tok/s가 하한 {ANSWER_MIN_TPS:.0f} 이상")
    elif small.get("tokens_per_sec", 0) >= ANSWER_MIN_TPS:
        qa = "exaone3.5:2.4b"
        reasons.append(f"답변용 {qa}: 7.8b가 하한 미달이라 빠른 모델로 대체 (실측 {small['tokens_per_sec']} tok/s)")
    memo = None
    if small.get("tokens_per_sec", 0) >= SUMMARY_MIN_TPS:
        memo = "exaone3.5:2.4b"
        reasons.append(f"요약용 {memo}: 실측 {small['tokens_per_sec']} tok/s가 하한 {SUMMARY_MIN_TPS:.0f} 이상")
    elif qa:
        memo = qa
        reasons.append(f"요약용 {memo}: 요약 하한을 넘는 별도 모델이 없어 답변용과 공용")
    num_gpu = None
    full = [m for m, s in speeds.items() if s.get("full_gpu")]
    if full:
        num_gpu = PROBE_NUM_GPU
        reasons.append(f"GPU 적재 {num_gpu}층: {', '.join(full)}에서 전층 적재 실측 확인")
    elif speeds:
        reasons.append("GPU 적재: 전층 적재가 실측으로 확인되지 않아 실행.bat 값을 바꾸지 않는다")
    return {"qa_model": qa, "memo_model": memo, "num_gpu": num_gpu,
            "sizes": sizes, "reasons": reasons}


def read_settings():
    """(원본 dict 또는 None, 파싱 실패 여부). app.py의 기본값 구조를 따른다."""
    if not os.path.isfile(SETTINGS_FILE):
        return None, False
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f), False
    except (OSError, ValueError):
        return None, True


def current_models(saved):
    if not saved:
        return DEFAULT_QA, DEFAULT_MEMO
    return saved.get("qa_model") or DEFAULT_QA, saved.get("memo_model") or DEFAULT_MEMO


def backup_settings(stamp):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"local_settings_{stamp}.json")
    if os.path.isfile(SETTINGS_FILE):
        shutil.copy2(SETTINGS_FILE, dst)
        return dst
    return None


def write_settings(saved, qa_model, memo_model):
    data = {"mask_level": (saved or {}).get("mask_level", "full"),
            "external": (saved or {}).get("external", {"provider": "", "model": "", "api_key": ""}),
            "qa_model": qa_model, "memo_model": memo_model}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def bat_num_gpu():
    try:
        with open(BAT_FILE, "rb") as f:
            m = re.search(rb"set LOCAL_AGENT_NUM_GPU=(\d+)", f.read())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def set_bat_num_gpu(target):
    """실행.bat의 숫자만 교체한다. ASCII+CRLF가 그대로 유지된다."""
    with open(BAT_FILE, "rb") as f:
        raw = f.read()
    new, n = re.subn(rb"set LOCAL_AGENT_NUM_GPU=\d+",
                     f"set LOCAL_AGENT_NUM_GPU={target}".encode("ascii"), raw, count=1)
    if n == 0:
        return False
    with open(BAT_FILE, "wb") as f:
        f.write(new)
    return True


def installed_models():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as r:
            return {m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])}
    except Exception:
        return None


def model_store_drive():
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return os.path.splitdrive(env)[0] + "\\"
    if os.path.isdir(r"E:\OllamaModels"):
        return "E:\\"
    return os.path.splitdrive(ROOT)[0] + "\\"


def pull_model(model):
    exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if not os.path.isfile(exe):
        exe = "ollama"
    env = dict(os.environ)
    if os.path.isdir(r"E:\OllamaModels"):
        env.setdefault("OLLAMA_MODELS", r"E:\OllamaModels")
    p = subprocess.run([exe, "pull", model], capture_output=True, text=True,
                       timeout=1800, env=env)
    return p.returncode == 0, (p.stdout + p.stderr)[-500:]


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    result = {"applied_at": time.strftime("%Y-%m-%d %H:%M:%S"), "actions": [], "blocked": None}
    probe_path = os.path.join(HERE, "probe_result.json")
    if not os.path.isfile(probe_path):
        print("probe_result.json이 없습니다. probe를 먼저 실행하세요.")
        return 1
    with open(probe_path, encoding="utf-8") as f:
        probe = json.load(f)
    rec = derive_recommendation(probe)
    result["recommendation"] = {k: rec[k] for k in ("qa_model", "memo_model", "num_gpu")}
    result["reasons"] = rec["reasons"]
    if not rec["qa_model"] or not rec["memo_model"]:
        result["blocked"] = "판정에서 모델을 정하지 못했습니다 (실측 속도 부족). 설정을 바꾸지 않습니다."
        _save(result)
        print(result["blocked"])
        return 1

    saved, parse_failed = read_settings()
    before_qa, before_memo = current_models(saved)
    result["before"] = {"qa_model": before_qa, "memo_model": before_memo,
                        "num_gpu": bat_num_gpu(), "settings_parse_failed": parse_failed}

    # 1) 모델 설치 확인과 pull
    installed = installed_models()
    if installed is None:
        installed = set(rec["sizes"])  # 서버 무응답이면 probe 당시 목록으로 대체
    for model in dict.fromkeys([rec["qa_model"], rec["memo_model"]]):
        if model in installed:
            result["actions"].append(f"{model}: 이미 설치됨, 건너뜀")
            continue
        size = KNOWN_SIZE_GB.get(model) or rec["sizes"].get(model)
        drive = model_store_drive()
        free = shutil.disk_usage(drive).free / 1e9
        if size and free < size + DISK_MARGIN_GB:
            result["blocked"] = (f"{model} 설치 중단: 필요 약 {size}GB(+여유 {DISK_MARGIN_GB}GB), "
                                 f"{drive} 여유 {free:.1f}GB로 {size + DISK_MARGIN_GB - free:.1f}GB 부족")
            _save(result)
            print(result["blocked"])
            return 2
        ok, tail = pull_model(model)
        result["actions"].append(f"{model}: ollama pull {'완료' if ok else '실패'}")
        if not ok:
            result["blocked"] = f"{model} pull 실패. 출력 끝부분: {tail}"
            _save(result)
            print(result["blocked"])
            return 1

    # 2) 설정 백업과 기록
    result["backup"] = backup_settings(stamp)
    if before_qa != rec["qa_model"] or before_memo != rec["memo_model"] or parse_failed:
        write_settings(saved, rec["qa_model"], rec["memo_model"])
        result["actions"].append("local_settings.json 갱신")
    else:
        result["actions"].append("local_settings.json: 판정과 같아 변경 없음")

    # 3) 실행.bat 적재 층수
    before_gpu = result["before"]["num_gpu"]
    if rec["num_gpu"] is not None and before_gpu != rec["num_gpu"]:
        if set_bat_num_gpu(rec["num_gpu"]):
            result["actions"].append(f"실행.bat LOCAL_AGENT_NUM_GPU {before_gpu} → {rec['num_gpu']}")
        else:
            result["actions"].append("실행.bat에서 LOCAL_AGENT_NUM_GPU 줄을 찾지 못해 바꾸지 않음")
    else:
        result["actions"].append(f"실행.bat LOCAL_AGENT_NUM_GPU: {before_gpu} 유지")

    saved2, _ = read_settings()
    after_qa, after_memo = current_models(saved2)
    result["after"] = {"qa_model": after_qa, "memo_model": after_memo, "num_gpu": bat_num_gpu()}
    _save(result)
    print(f"적용 전: 질문 {before_qa} / 요약 {before_memo} / num_gpu {before_gpu}")
    print(f"적용 후: 질문 {after_qa} / 요약 {after_memo} / num_gpu {result['after']['num_gpu']}")
    for a in result["actions"]:
        print("-", a)
    print("APPLY_DONE")
    return 0


def _save(result):
    with open(os.path.join(HERE, "apply_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    L = ["# 적용 리포트 (apply)", "",
         f"- 적용 시각: {result['applied_at']}",
         f"- 백업: {result.get('backup')}", ""]
    if result.get("blocked"):
        L += ["## 중단", "", f"- {result['blocked']}", ""]
    b, a = result.get("before", {}), result.get("after", {})
    L += ["## 전후 비교", "",
          "| 항목 | 적용 전 | 적용 후 |", "|---|---|---|",
          f"| 질문용 모델 | {b.get('qa_model')} | {a.get('qa_model')} |",
          f"| 요약용 모델 | {b.get('memo_model')} | {a.get('memo_model')} |",
          f"| GPU 적재 층수 | {b.get('num_gpu')} | {a.get('num_gpu')} |", "",
          "## 판정 근거", ""]
    L += [f"- {r}" for r in result.get("reasons", [])]
    L += ["", "## 수행 내역", ""]
    L += [f"- {x}" for x in result.get("actions", [])]
    L.append("")
    with open(os.path.join(HERE, "apply_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())

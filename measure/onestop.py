# -*- coding: utf-8 -*-
"""
원스톱 진입점 — 측정(probe) → 적용(apply) → 점검+자동복구를 한 번에 실행한다.

현장 사용자는 이 파일 하나만 실행하면 된다:
    python onestop.py              전체 흐름
    python onestop.py --check-only 점검+자동복구만

점검 단계는 "준비 완료"가 확인될 때까지 스스로 진단하고 고친다.
복구 루프는 진단 → 복구 1회 → 재점검 순서로 최대 3라운드까지만 돈다.
자동으로 하는 복구는 되돌릴 수 있는 것만이다 (서버 재기동, 모델 재적재,
색인·임베딩 캐시 재생성, 설정 백업 복원, 모델 재다운로드).
사용자 문서나 메모 같은 데이터는 자동 복구에서 지우지 않는다.
끝내 못 고치면 무엇이 왜 안 되는지, 사람이 무엇을 하면 되는지를 리포트로 남긴다.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import apply as apply_mod  # noqa: E402  (설정·bat·pull 헬퍼 재사용)

OLLAMA = "http://localhost:11434"
APP = "http://127.0.0.1:8765"
CHECK_QUESTION = "고액 현금거래는 얼마부터 보고 대상인가요?"  # 샘플 문서에 답이 있는 질문
MAX_ROUNDS = 3
MAX_PULL_TRIES = 2
LOG_FILE = os.path.join(HERE, "onestop_run.log")

STATE = {"app_proc": None, "app_started_by_us": False, "pull_tries": 0,
         "recoveries": [], "rounds": []}

MANUAL_ACTIONS = {
    "OLLAMA_DOWN": "Ollama가 설치돼 있는지 확인하고, 시작 메뉴에서 Ollama를 실행한 뒤 다시 시도하세요.",
    "MODEL_MISSING": "네트워크 연결을 확인하고 모델 저장소(E:\\OllamaModels)가 있는 드라이브를 연결하세요.",
    "DISK_LOW": "모델 저장소 드라이브의 여유 공간을 확보한 뒤 다시 실행하세요.",
    "SETTINGS_PARSE_FAIL": "measure/backup 폴더의 백업도 없습니다. local_settings.json을 지우고 다시 실행하면 기본값으로 재생성됩니다.",
    "PORT_STALE": "작업 관리자에서 이전 python(app.py) 프로세스를 종료한 뒤 다시 실행하세요.",
    "APP_DOWN": "app.py 실행 오류를 measure/app_check.log에서 확인하세요.",
    "INDEX_EMPTY": "docs 폴더에 업무 문서가 있는지 확인하세요. 문서 추가는 자동 복구 범위가 아닙니다.",
    "EMBED_CACHE": "임베딩 모델(bge-m3) 설치 상태를 확인하세요. 단어 검색만으로도 질의는 동작합니다.",
    "OFFLOAD_PARTIAL": "GPU 드라이버 상태와 다른 프로그램의 VRAM 점유를 확인하세요.",
    "QA_NO_SOURCES": "docs 폴더의 샘플 문서가 색인되는지 /api/reload 결과를 확인하세요.",
    "QA_FAIL": "measure/app_check.log와 Ollama 상태를 확인하세요.",
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- HTTP 헬퍼 ----------------
def http_json(url, payload=None, timeout=10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_up():
    try:
        http_json(OLLAMA + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


def start_ollama():
    exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if not os.path.isfile(exe):
        return False
    ps = ("$env:OLLAMA_MODELS='E:\\OllamaModels'; "
          f"Start-Process -FilePath '{exe}' -ArgumentList 'serve' -WindowStyle Hidden")
    subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=60)
    for _ in range(30):
        if ollama_up():
            return True
        time.sleep(1)
    return False


def ensure_ollama(context):
    if ollama_up():
        return True
    log(f"{context}: Ollama 서버가 꺼져 있어 분리 기동합니다")
    ok = start_ollama()
    STATE["recoveries"].append({"issue": "OLLAMA_DOWN", "context": context,
                                "fix": "ollama serve 분리 기동", "ok": ok})
    return ok


# ---------------- 앱(app.py) 제어 ----------------
def app_status():
    try:
        return http_json(APP + "/api/status", timeout=5)
    except Exception:
        return None


def listening_pid():
    ps = ("(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | "
          "Select-Object -First 1).OwningProcess")
    p = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True, timeout=30)
    out = p.stdout.strip()
    return int(out) if out.isdigit() else None


def pid_cmdline(pid):
    ps = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
    p = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True, timeout=30)
    return p.stdout.strip()


def stop_pid(pid, why):
    subprocess.run(["powershell", "-Command",
                    f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                   capture_output=True, timeout=30)
    log(f"프로세스 {pid} 종료 ({why})")
    time.sleep(2)


def start_app():
    env = dict(os.environ)
    gpu = apply_mod.bat_num_gpu()
    env["LOCAL_AGENT_NUM_GPU"] = str(gpu if gpu is not None else 32)
    env["PYTHONIOENCODING"] = "utf-8"
    logf = open(os.path.join(HERE, "app_check.log"), "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py")],
                            cwd=ROOT, env=env, stdout=logf, stderr=logf,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    STATE["app_proc"] = proc
    STATE["app_started_by_us"] = True
    for _ in range(75):
        if app_status() is not None:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(2)
    return False


def stop_our_app():
    proc = STATE["app_proc"]
    if proc is not None and STATE["app_started_by_us"] and proc.poll() is None:
        stop_pid(proc.pid, "점검용으로 띄운 app.py 정리")
    STATE["app_proc"] = None
    STATE["app_started_by_us"] = False


# ---------------- 점검 본체 ----------------
def qa_check():
    """질의 1회. (성공 여부, 상세 dict) 반환."""
    body = json.dumps({"question": CHECK_QUESTION}).encode("utf-8")
    req = urllib.request.Request(APP + "/api/ask_stream", data=body,
                                 headers={"Content-Type": "application/json"})
    found, sources, text, done, error = None, [], [], False, None
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("type") == "meta":
                    found = ev.get("found")
                    sources = ev.get("sources", [])
                elif ev.get("type") == "delta":
                    text.append(ev.get("text", ""))
                elif ev.get("type") == "done":
                    done = True
                elif ev.get("type") == "error":
                    error = ev.get("message")
    except Exception as e:
        return False, {"error": f"질의 호출 실패: {e}"}
    answer = "".join(text).strip()
    ok = bool(found) and bool(sources) and len(answer) > 0 and done and not error
    return ok, {"found": found, "sources": sources, "answer_len": len(answer),
                "answer_head": answer[:120], "done": done, "error": error}


def offload_ratio(model):
    try:
        ps = http_json(OLLAMA + "/api/ps", timeout=5)
        for m in ps.get("models", []):
            if m["name"] == model and m.get("size"):
                return round(m.get("size_vram", 0) / m["size"], 2)
    except Exception:
        pass
    return None


def warmup(model):
    gpu = apply_mod.bat_num_gpu() or 32
    try:
        http_json(OLLAMA + "/api/chat",
                  {"model": model, "messages": [{"role": "user", "content": "안녕하세요"}],
                   "stream": False, "options": {"num_predict": 5, "num_ctx": 4096, "num_gpu": gpu}},
                  timeout=300)
        return True
    except Exception:
        return False


def unload_model(model):
    try:
        http_json(OLLAMA + "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def diagnose():
    """실측 이력의 실패 모드 체크리스트로 원인을 분류한다."""
    issues = []
    saved, parse_failed = apply_mod.read_settings()
    if parse_failed:
        issues.append("SETTINGS_PARSE_FAIL")
    qa_model, memo_model = apply_mod.current_models(saved)
    if not ollama_up():
        issues.append("OLLAMA_DOWN")
        return issues, qa_model, memo_model  # 아래 단계는 서버가 있어야 판정 가능
    tags = http_json(OLLAMA + "/api/tags", timeout=10)
    installed = {m["name"] for m in tags.get("models", [])}
    missing = [m for m in dict.fromkeys([qa_model, memo_model]) if m not in installed]
    if missing:
        drive = apply_mod.model_store_drive()
        free = shutil.disk_usage(drive).free / 1e9
        need = sum(apply_mod.KNOWN_SIZE_GB.get(m, 2.0) for m in missing)
        if free < need + apply_mod.DISK_MARGIN_GB:
            issues.append("DISK_LOW")
        else:
            issues.append("MODEL_MISSING")
    st = app_status()
    if st is None:
        pid = listening_pid()
        if pid and "app.py" in pid_cmdline(pid):
            issues.append("PORT_STALE")
        else:
            issues.append("APP_DOWN")
        return issues, qa_model, memo_model
    if st.get("doc_chunks", 0) == 0:
        issues.append("INDEX_EMPTY")
    cache = os.path.join(ROOT, "embeddings_cache.json")
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                json.load(f)
        except (OSError, ValueError):
            issues.append("EMBED_CACHE")
    if st.get("search_mode") == "단어" and any("bge-m3" in m for m in installed):
        issues.append("EMBED_CACHE")
    if warmup(qa_model):
        ratio = offload_ratio(qa_model)
        if ratio is not None and ratio < apply_mod.FULL_GPU_RATIO:
            issues.append("OFFLOAD_PARTIAL")
    ok, detail = qa_check()
    if not ok:
        if detail.get("found") is False or (detail.get("found") and not detail.get("sources")):
            issues.append("QA_NO_SOURCES")
        elif "QA_NO_SOURCES" not in issues:
            issues.append("QA_FAIL")
    return issues, qa_model, memo_model


def recover(issues, qa_model, memo_model):
    """진단된 원인별로 복구 1회를 수행하고 기록한다."""
    fixed = []
    for issue in issues:
        ok, desc = False, ""
        if issue == "SETTINGS_PARSE_FAIL":
            ok, desc = _fix_settings()
        elif issue == "OLLAMA_DOWN":
            ok = start_ollama()
            desc = "ollama serve 분리 기동"
        elif issue == "MODEL_MISSING":
            ok, desc = _fix_pull(qa_model, memo_model)
        elif issue == "PORT_STALE":
            pid = listening_pid()
            if pid:
                stop_pid(pid, "포트 8765 잔류 app.py")
            ok = start_app()
            desc = "잔류 프로세스 정리 후 app.py 재기동"
        elif issue == "APP_DOWN":
            ok = start_app()
            desc = "app.py 분리 기동"
        elif issue == "INDEX_EMPTY" or issue == "QA_NO_SOURCES":
            ok = _fix_reload()
            desc = "문서 재색인 (/api/reload)"
        elif issue == "EMBED_CACHE":
            ok = _fix_embed_cache()
            desc = "임베딩 캐시 재생성 + 재색인"
        elif issue == "OFFLOAD_PARTIAL":
            unload_model(qa_model)
            ok = warmup(qa_model)
            desc = f"{qa_model} 언로드 후 적재 층수 재적용해 재로드+워밍업"
        elif issue == "QA_FAIL":
            ok = _fix_reload()
            desc = "재색인 후 재점검에 맡김"
        STATE["recoveries"].append({"issue": issue, "fix": desc, "ok": ok})
        log(f"복구 시도: {issue} → {desc} ({'성공' if ok else '실패'})")
        if ok:
            fixed.append(issue)
    return fixed


def _fix_settings():
    backups = []
    if os.path.isdir(apply_mod.BACKUP_DIR):
        backups = sorted(f for f in os.listdir(apply_mod.BACKUP_DIR)
                         if f.startswith("local_settings_"))
    if backups:
        src = os.path.join(apply_mod.BACKUP_DIR, backups[-1])
        shutil.copy2(src, apply_mod.SETTINGS_FILE)
        return True, f"백업에서 복원 ({backups[-1]})"
    qa, memo = apply_mod.DEFAULT_QA, apply_mod.DEFAULT_MEMO
    apply_mod.write_settings(None, qa, memo)
    return True, "백업이 없어 기본값으로 재생성"


def _fix_pull(qa_model, memo_model):
    if STATE["pull_tries"] >= MAX_PULL_TRIES:
        return False, f"pull 재시도 한도({MAX_PULL_TRIES}회) 도달"
    tags = http_json(OLLAMA + "/api/tags", timeout=10)
    installed = {m["name"] for m in tags.get("models", [])}
    results = []
    for model in dict.fromkeys([qa_model, memo_model]):
        if model in installed:
            continue
        STATE["pull_tries"] += 1
        ok, _ = apply_mod.pull_model(model)
        results.append(f"{model} pull {'완료' if ok else '실패'}")
    return all("완료" in r for r in results) if results else True, "; ".join(results)


def _fix_reload():
    try:
        r = http_json(APP + "/api/reload", {}, timeout=600)
        return r.get("doc_chunks", 0) > 0
    except Exception:
        return False


def _fix_embed_cache():
    cache = os.path.join(ROOT, "embeddings_cache.json")
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                json.load(f)
        except (OSError, ValueError):
            dst = cache + f".corrupt-{time.strftime('%Y%m%d_%H%M%S')}.bak"
            os.rename(cache, dst)  # 삭제가 아니라 이름만 바꿔 보존한다
            log(f"손상된 임베딩 캐시를 {os.path.basename(dst)}로 보존 이동")
    return _fix_reload()


def stage_check():
    log("=== 점검 단계 (자동 복구 포함) ===")
    last_issues = []
    for rnd in range(1, MAX_ROUNDS + 1):
        issues, qa_model, memo_model = diagnose()
        STATE["rounds"].append({"round": rnd, "issues": issues})
        log(f"라운드 {rnd} 진단: {issues if issues else '이상 없음'}")
        if not issues:
            ok, detail = qa_check()
            if ok:
                log(f"준비 완료: 질의 응답 확인 (근거 {detail['sources']}, 답변 {detail['answer_len']}자)")
                return True, detail, []
            issues = ["QA_FAIL"]
        last_issues = issues
        if "DISK_LOW" in issues:
            log("디스크 부족은 자동 복구할 수 없습니다. 중단합니다.")
            break
        if rnd < MAX_ROUNDS:
            recover(issues, qa_model, memo_model)
    ok, detail = qa_check() if ollama_up() and app_status() else (False, {})
    return ok, detail, last_issues


# ---------------- 단계 실행과 리포트 ----------------
def run_stage(name, script):
    log(f"=== {name} 단계 ===")
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    tail = (p.stdout or "").strip().splitlines()[-8:]
    for line in tail:
        log(f"  {line}")
    if p.returncode != 0:
        log(f"{name} 단계 종료 코드 {p.returncode}")
        if p.stderr:
            log("  stderr: " + p.stderr.strip().splitlines()[-1][:300])
    return p.returncode


def write_report(status, stages, check_detail, remaining):
    L = ["# 원스톱 실행 리포트 (onestop)", "",
         f"- 실행 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"- 최종 상태: {status}", ""]
    L += ["## 단계 결과", ""]
    for name, rc in stages:
        L.append(f"- {name}: 종료 코드 {rc}")
    if check_detail:
        L += ["", "## 점검 질의", "",
              f"- 질문: {CHECK_QUESTION}",
              f"- 근거 문서: {check_detail.get('sources')}",
              f"- 답변 길이: {check_detail.get('answer_len')}자",
              f"- 답변 앞부분: {check_detail.get('answer_head', '')}"]
    if STATE["rounds"]:
        L += ["", "## 진단 라운드", ""]
        for r in STATE["rounds"]:
            L.append(f"- 라운드 {r['round']}: {r['issues'] if r['issues'] else '이상 없음'}")
    if STATE["recoveries"]:
        L += ["", "## 자동 복구 내역", ""]
        for rec in STATE["recoveries"]:
            L.append(f"- {rec['issue']}: {rec['fix']} ({'성공' if rec['ok'] else '실패'})")
    if remaining:
        L += ["", "## 남은 문제와 사람이 할 일", ""]
        for issue in remaining:
            L.append(f"- {issue}: {MANUAL_ACTIONS.get(issue, '로그(onestop_run.log)를 확인하세요.')}")
    L.append("")
    with open(os.path.join(HERE, "onestop_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(HERE, "onestop_result.json"), "w", encoding="utf-8") as f:
        json.dump({"status": status, "stages": stages, "check": check_detail,
                   "rounds": STATE["rounds"], "recoveries": STATE["recoveries"],
                   "remaining": remaining}, f, ensure_ascii=False, indent=2)


def main():
    check_only = "--check-only" in sys.argv
    stages = []
    try:
        if not check_only:
            if not ensure_ollama("사전 점검"):
                write_report("실패: Ollama 기동 불가", stages, {}, ["OLLAMA_DOWN"])
                return 1
            rc = run_stage("측정(probe)", "probe.py")
            stages.append(("probe", rc))
            if rc != 0:
                write_report("실패: 측정 단계", stages, {}, [])
                return 1
            rc = run_stage("적용(apply)", "apply.py")
            stages.append(("apply", rc))
            if rc != 0:
                write_report("실패: 적용 단계", stages, {}, [])
                return 1
        ok, detail, remaining = stage_check()
        stages.append(("check", 0 if ok else 1))
        status = "준비 완료" if ok else "점검 실패"
        write_report(status, stages, detail, remaining)
        log(f"=== 최종: {status} ===")
        return 0 if ok else 1
    finally:
        stop_our_app()


if __name__ == "__main__":
    sys.exit(main())

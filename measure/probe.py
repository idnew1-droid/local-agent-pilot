# -*- coding: utf-8 -*-
"""
환경 측정기 (probe) — 로컬 에이전트가 "이 PC에서 성립하는가"를 실측으로 판정한다.

기존 측정 스크립트(bench.py, measure2.py, measure_embed.py)가 속도·검색·마스킹을
따로 쟀다면, probe는 거기에 환경 파악과 판정 층을 더한 통합본이다.

측정 순서
  1. 환경 수집: OS, CPU, RAM, GPU(nvidia-smi), 드라이브별 여유, Python 버전
  2. Ollama 상태: 서버 응답, 설치 모델 목록과 크기
  3. 성능 측정: 생성 모델별 짧은 생성 1회, eval_count/eval_duration 기반 tok/s
     (measure2.py와 같은 조건: num_ctx=4096, num_gpu=32 전층 적재 지정)
  4. 기능 테스트: bench.py의 샘플 그대로 문서 검색 5문항, 마스킹 기대 10건,
     measure_embed.py의 샘플로 임베딩 임계값(EMBED_THRESHOLD) 분리 여부
  5. 판정: 실측 수치에서 도출한 구성 판정 문장 + probe_result.json 저장
  6. 리포트: 콘솔 출력과 probe_report.md 동시 생성

수치는 이 스크립트가 실제로 측정한 값만 기록한다. 판정 기준(속도 하한 등)은
코드 아래쪽 상수에 명시돼 있고, 측정값이 아니라 판단 기준이다.
"""
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # agent.py가 있는 LocalAgent 루트
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import agent  # noqa: E402
import bench  # noqa: E402  (QA_SET, MASK_SET와 qa_test/mask_test 재사용)

OLLAMA = agent.OLLAMA
GEN_MODELS = ["exaone3.5:7.8b", "exaone3.5:2.4b"]  # 최소 측정 대상 (설치된 것만 잰다)
SPEED_PROMPT = "보이스피싱 예방 수칙 10가지를 각각 두 문장 이상으로 자세히 설명하세요."
SPEED_OPTIONS = {"num_predict": 300, "num_ctx": 4096, "num_gpu": 32}  # measure2.py와 동일 조건

# 판정 기준 (측정값이 아니라 판단 기준)
ANSWER_MIN_TPS = 20.0    # 답변용 모델의 대화 속도 하한
SUMMARY_MIN_TPS = 40.0   # 요약용 모델의 속도 하한 (짧은 출력이라 속도를 우선 본다)
FULL_GPU_RATIO = 0.90    # size_vram/size가 이 이상이면 전층 GPU 적재로 판정 (실측 7.8B: 5.05/5.48=0.92)
VRAM_HEADROOM = 0.95     # KV 캐시 여유를 두고 모델 크기가 VRAM의 이 비율 이하일 때 적재 가능으로 판정

# measure_embed.py의 임계값 실측 샘플 (그대로 복사)
EMBED_IN_DOC = [
    "보이스피싱이 의심되는 이체 요청을 받으면 어떻게 해야 하나요?",
    "고액 현금거래는 얼마부터 보고 대상인가요?",
    "계좌 비밀번호를 여러 번 틀린 고객은 어떻게 안내하나요?",
    "정기예금 중도해지 이율은 어떻게 되나요?",
    "예금 만기 후 이자는 어떻게 계산되나요?",
]
EMBED_OUT_DOC = [
    "오늘 서울 날씨 어때요?",
    "점심 메뉴 추천해 주세요",
    "주식 시장 전망이 어떤가요?",
]


# ---------------- 1. 환경 수집 ----------------
def collect_env():
    env = {"os": platform.platform(), "os_release": platform.release(),
           "machine": platform.machine(),
           "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown",
           "python": sys.version.split()[0]}
    env["ram_gb"] = _ram_gb()
    env["gpu"] = _gpu_info()
    env["disks"] = _disk_info()
    return env


def _ram_gb():
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1e9, 1)
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1e6, 1)
    except Exception:
        pass
    return None


def _gpu_info():
    for exe in ("nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe"):
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                name, total, used, driver = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
                return {"name": name, "vram_total_gb": round(int(total) / 1000, 1),
                        "vram_used_gb": round(int(used) / 1000, 1), "driver": driver}
        except Exception:
            continue
    return None


def _disk_info():
    disks = []
    if os.name == "nt":
        roots = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")]
    else:
        roots = ["/"]
    for r in roots:
        try:
            u = shutil.disk_usage(r)
            disks.append({"drive": r, "total_gb": round(u.total / 1e9, 1),
                          "free_gb": round(u.free / 1e9, 1)})
        except Exception:
            pass
    return disks


# ---------------- 2. Ollama 상태 ----------------
def _get(path, timeout=10):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_status():
    st = {"up": False, "version": None, "models": []}
    try:
        st["version"] = _get("/api/version").get("version")
        st["up"] = True
    except Exception as e:
        st["error"] = str(e)
        return st
    try:
        for m in _get("/api/tags").get("models", []):
            det = m.get("details", {})
            st["models"].append({"name": m["name"], "size_gb": round(m["size"] / 1e9, 2),
                                 "family": det.get("family"), "params": det.get("parameter_size")})
    except Exception as e:
        st["error"] = str(e)
    return st


def _ps_map():
    try:
        return {m["name"]: {"size_gb": round(m["size"] / 1e9, 2),
                            "vram_gb": round(m.get("size_vram", 0) / 1e9, 2)}
                for m in _get("/api/ps").get("models", [])}
    except Exception:
        return {}


# ---------------- 3. 성능 측정 ----------------
def _chat(model, options):
    payload = {"model": model, "messages": [{"role": "user", "content": SPEED_PROMPT}],
               "stream": False, "options": options}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def speed_test(model):
    _chat(model, {"num_predict": 5, "num_ctx": 4096, "num_gpu": 32})  # 워밍업 (로딩 제외)
    t0 = time.time()
    r = _chat(model, dict(SPEED_OPTIONS))
    wall = time.time() - t0
    ec = r.get("eval_count", 0)
    ed = r.get("eval_duration", 0) / 1e9
    ps = _ps_map().get(model, {})
    size_gb, vram_gb = ps.get("size_gb"), ps.get("vram_gb")
    ratio = round(vram_gb / size_gb, 2) if size_gb and vram_gb else None
    return {"model": model, "eval_count": ec, "eval_seconds": round(ed, 2),
            "tokens_per_sec": round(ec / ed, 1) if ed else None,
            "wall_seconds": round(wall, 2),
            "load_seconds": round(r.get("load_duration", 0) / 1e9, 2),
            "loaded_size_gb": size_gb, "loaded_vram_gb": vram_gb,
            "gpu_ratio": ratio,
            "full_gpu": bool(ratio is not None and ratio >= FULL_GPU_RATIO)}


# ---------------- 4. 기능 테스트 ----------------
def functional_tests():
    out = {}
    chunks = agent.load_chunks()
    out["chunks"] = len(chunks)
    out["qa"] = bench.qa_test(chunks)
    out["mask"] = bench.mask_test()
    out["embed"] = _embed_test(chunks)
    return out


def _embed_test(chunks):
    res = {"model": agent.EMBED_MODEL, "threshold": agent.EMBED_THRESHOLD}
    vecs = agent.embed([c["text"] for c in chunks]) if chunks else None
    if vecs is None:
        res["available"] = False
        return res
    res["available"] = True
    cache = {c["text"]: v for c, v in zip(chunks, vecs)}

    def best(q):
        qe = agent.embed([q])
        if not qe:
            return None
        return round(max(agent.vec_cosine(qe[0], v) for v in cache.values()), 3)

    ins = [best(q) for q in EMBED_IN_DOC]
    outs = [best(q) for q in EMBED_OUT_DOC]
    res["in_doc_scores"] = ins
    res["out_doc_scores"] = outs
    res["in_min"] = min(ins) if ins else None
    res["out_max"] = max(outs) if outs else None
    res["separated"] = bool(ins and outs and min(ins) > agent.EMBED_THRESHOLD > max(outs))
    return res


# ---------------- 5. 판정 ----------------
def verdict(env, ollama, speeds, func):
    lines = []
    gpu = env.get("gpu") or {}
    vram = gpu.get("vram_total_gb")
    by_model = {s["model"]: s for s in speeds}
    sizes = {m["name"]: m["size_gb"] for m in ollama.get("models", [])}

    for name, s in by_model.items():
        size = sizes.get(name)
        if vram and size:
            fits = size <= vram * VRAM_HEADROOM
            lines.append(
                f"{name}: 모델 {size}GB / VRAM {vram}GB — "
                + (f"전층 GPU 적재 가능 (실측 적재 비율 {s.get('gpu_ratio')}, "
                   f"{'전층 적재 확인' if s.get('full_gpu') else '일부 CPU 오프로드'})"
                   if fits else "VRAM보다 커서 전층 적재 불가, CPU 오프로드 혼합 구성"))
        if s.get("tokens_per_sec"):
            lines.append(f"{name}: 생성 속도 {s['tokens_per_sec']} tok/s (eval {s['eval_count']}토큰)")

    answer = None
    big = by_model.get("exaone3.5:7.8b", {})
    small = by_model.get("exaone3.5:2.4b", {})
    if big.get("tokens_per_sec") and big["tokens_per_sec"] >= ANSWER_MIN_TPS:
        answer = "exaone3.5:7.8b"
    elif small.get("tokens_per_sec") and small["tokens_per_sec"] >= ANSWER_MIN_TPS:
        answer = "exaone3.5:2.4b"
    if answer:
        lines.append(f"답변용 모델: {answer} — 실측 {by_model[answer]['tokens_per_sec']} tok/s로 "
                     f"대화 하한 {ANSWER_MIN_TPS:.0f} tok/s를 넘는다")
    else:
        lines.append(f"답변용 모델: 측정 대상 중 대화 하한 {ANSWER_MIN_TPS:.0f} tok/s를 넘는 모델이 없다")

    summary = None
    if small.get("tokens_per_sec") and small["tokens_per_sec"] >= SUMMARY_MIN_TPS:
        summary = "exaone3.5:2.4b"
    elif answer:
        summary = answer
    if summary:
        lines.append(f"요약용 모델: {summary} — 실측 {by_model[summary]['tokens_per_sec']} tok/s")

    qa = func.get("qa", {})
    if qa:
        lines.append(f"문서 검색: {qa.get('score')} 적중 (단어 검색, 임계값 {agent.RETRIEVE_THRESHOLD})")
    mask = func.get("mask", {})
    if mask:
        lines.append(f"마스킹 탐지: {mask.get('rate')} (기대 건수 대비)")
    emb = func.get("embed", {})
    if emb.get("available"):
        lines.append(
            f"임베딩 분리: 문서 내 최소 {emb.get('in_min')} / 문서 밖 최대 {emb.get('out_max')} — "
            + (f"임계값 {emb['threshold']}가 두 구간을 가른다" if emb.get("separated")
               else f"임계값 {emb['threshold']} 기준으로 분리가 안 된다"))
    elif "available" in emb:
        lines.append("임베딩: 모델이 없어 단어 검색만으로 동작하는 구성")

    ram = env.get("ram_gb")
    if ram:
        lines.append(f"RAM {ram}GB — GPU 적재가 안 되는 환경이면 작은 모델의 CPU 실행 여지는 RAM으로 판단한다")
    return lines


# ---------------- 6. 리포트 ----------------
def render_report(res):
    env, ollama = res["env"], res["ollama"]
    L = ["# 환경 측정 리포트 (probe)", "",
         f"- 측정 시각: {res['measured_at']}",
         f"- 측정기: measure/probe.py", ""]
    L += ["## 환경", "",
          "| 항목 | 값 |", "|---|---|",
          f"| OS | {env['os']} |",
          f"| CPU | {env['cpu']} |",
          f"| RAM | {env.get('ram_gb')} GB |"]
    gpu = env.get("gpu")
    if gpu:
        L.append(f"| GPU | {gpu['name']} (VRAM {gpu['vram_total_gb']}GB, 드라이버 {gpu['driver']}) |")
    else:
        L.append("| GPU | nvidia-smi 없음 (NVIDIA GPU 미확인) |")
    L.append(f"| Python | {env['python']} |")
    for d in env["disks"]:
        L.append(f"| 디스크 {d['drive']} | 전체 {d['total_gb']}GB / 여유 {d['free_gb']}GB |")
    L += ["", "## Ollama", "",
          f"- 서버: {'응답함' if ollama['up'] else '응답 없음'} (버전 {ollama.get('version')})", "",
          "| 모델 | 크기(GB) | 계열 | 파라미터 |", "|---|---|---|---|"]
    for m in ollama["models"]:
        L.append(f"| {m['name']} | {m['size_gb']} | {m.get('family')} | {m.get('params')} |")
    L += ["", "## 성능 측정 (num_ctx=4096, num_gpu=32, 300토큰 생성)", "",
          "| 모델 | tok/s | eval 토큰 | 적재(GB) | VRAM(GB) | 적재 비율 | 전층 GPU |",
          "|---|---|---|---|---|---|---|"]
    for s in res["speeds"]:
        L.append(f"| {s['model']} | {s.get('tokens_per_sec')} | {s.get('eval_count')} | "
                 f"{s.get('loaded_size_gb')} | {s.get('loaded_vram_gb')} | {s.get('gpu_ratio')} | "
                 f"{'예' if s.get('full_gpu') else '아니오'} |")
    f = res["functional"]
    L += ["", "## 기능 테스트", "",
          f"- 색인 단락: {f.get('chunks')}개",
          f"- 문서 검색 적중: {f.get('qa', {}).get('score')}",
          f"- 마스킹 탐지: {f.get('mask', {}).get('rate')}"]
    emb = f.get("embed", {})
    if emb.get("available"):
        L.append(f"- 임베딩 분리: 문서 내 최소 {emb.get('in_min')} / 문서 밖 최대 {emb.get('out_max')} "
                 f"/ 임계값 {emb.get('threshold')} / 분리 {'됨' if emb.get('separated') else '안 됨'}")
    else:
        L.append("- 임베딩: 사용 불가 (단어 검색 폴백)")
    L += ["", "## 판정", ""]
    L += [f"- {line}" for line in res["verdict"]]
    L.append("")
    return "\n".join(L)


def main():
    res = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "env": collect_env(), "ollama": ollama_status(),
           "speeds": [], "functional": {}, "verdict": []}
    installed = {m["name"] for m in res["ollama"]["models"]}
    if res["ollama"]["up"]:
        for model in GEN_MODELS:
            if model in installed:
                try:
                    res["speeds"].append(speed_test(model))
                except Exception as e:
                    res["speeds"].append({"model": model, "error": str(e)})
        try:
            res["functional"] = functional_tests()
        except Exception as e:
            res["functional"] = {"error": str(e)}
    res["verdict"] = verdict(res["env"], res["ollama"], res["speeds"], res["functional"])

    report = render_report(res)
    print(report)
    with open(os.path.join(HERE, "probe_result.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "probe_report.md"), "w", encoding="utf-8") as fp:
        fp.write(report)
    print("PROBE_DONE")


if __name__ == "__main__":
    main()

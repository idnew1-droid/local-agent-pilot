# -*- coding: utf-8 -*-
"""
로컬 업무 비서 (LocalDesk) — 브라우저 UI 서버.

설계 원칙은 agent.py와 같다: 모든 추론은 로컬 Ollama(localhost)에서만,
이 서버도 127.0.0.1에만 바인딩한다. 외부 네트워크 호출·수신 모두 없음.
Python 표준 라이브러리만으로 동작하므로 폐쇄망 PC에 그대로 복사해 쓸 수 있다.

사용법
  python app.py            서버 시작 후 브라우저가 자동으로 열린다
환경변수
  LOCALDESK_PORT           포트 (기본 8765, 사용 중이면 다음 포트 탐색)
  LOCAL_AGENT_MODEL       기본 모델 (질문·요약 공통 초기값)
  LOCAL_AGENT_NUM_GPU     GPU 적재 층 수
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
import webbrowser
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import agent  # 같은 폴더의 코어 로직 (마스킹·검색·Ollama 호출)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(BASE_DIR, "ui.html")

STATE = {
    "chunks": agent.load_chunks(),
    "qa_model": agent.MODEL,
    "memo_model": os.environ.get("LOCALDESK_MEMO_MODEL", "exaone3.5:2.4b"),
    "num_gpu": agent.CHAT_OPTIONS.get("num_gpu"),
    "counters": {"asks": 0, "memos": 0, "masked_items": 0},
    "lock": threading.Lock(),       # 색인 교체용
    "chat_lock": threading.Lock(),  # 모델 전환이 섞이지 않게 호출 직렬화
}


LOG_FILE = os.path.join(BASE_DIR, "usage_log.jsonl")


def append_log(entry):
    """사용 기록: 질문·메모 같은 내용은 남기지 않고 종류와 건수만 남긴다."""
    try:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_all_counts():
    """사용 기록 전체를 날짜별로 집계한다."""
    agg = {}
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                d = str(e.get("ts", ""))[:10]
                if not d:
                    continue
                a = agg.setdefault(d, {"asks": 0, "memos": 0, "masked_items": 0})
                if e.get("type") == "ask":
                    a["asks"] += 1
                elif e.get("type") == "summarize":
                    a["memos"] += 1
                    a["masked_items"] += sum((e.get("counts") or {}).values())
                elif e.get("type") == "mask":
                    a["masked_items"] += sum((e.get("counts") or {}).values())
    except OSError:
        pass
    return agg


def log_counts(date_str):
    return log_all_counts().get(date_str, {"asks": 0, "memos": 0, "masked_items": 0})


def log_today():
    return log_counts(datetime.now().date().isoformat())


# ---------- 설정 (이 PC에만 저장되는 파일) ----------
SETTINGS_FILE = os.path.join(BASE_DIR, "local_settings.json")
DEFAULT_EXTERNAL_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5",
                           "gemini": "gemini-2.5-flash"}


def load_settings():
    s = {"mask_level": "full", "external": {"provider": "", "model": "", "api_key": ""}}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return s
    if saved.get("mask_level") in ("full", "partial"):
        s["mask_level"] = saved["mask_level"]
    if saved.get("qa_model"):
        STATE["qa_model"] = saved["qa_model"]
    if saved.get("memo_model"):
        STATE["memo_model"] = saved["memo_model"]
    ext = saved.get("external") or {}
    if ext.get("provider") in DEFAULT_EXTERNAL_MODELS:
        s["external"] = {"provider": ext["provider"],
                         "model": ext.get("model") or DEFAULT_EXTERNAL_MODELS[ext["provider"]],
                         "api_key": ext.get("api_key", "")}
    return s


def save_settings():
    data = {"mask_level": SETTINGS["mask_level"], "external": SETTINGS["external"],
            "qa_model": STATE["qa_model"], "memo_model": STATE["memo_model"]}
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


SETTINGS = load_settings()


def external_active():
    e = SETTINGS["external"]
    return bool(e["provider"] and e["api_key"])


def _json_req(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def external_chat(messages):
    """외부 AI 호출. 사용자가 설정에서 켠 경우에만 쓰이며, 받는 내용은 마스킹이 끝난 것만이다."""
    ext = SETTINGS["external"]
    provider, key, model = ext["provider"], ext["api_key"], ext["model"]
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    try:
        if provider == "openai":
            data = _json_req("https://api.openai.com/v1/chat/completions",
                             {"model": model, "messages": messages},
                             {"Authorization": f"Bearer {key}"})
            return data["choices"][0]["message"]["content"]
        if provider == "anthropic":
            data = _json_req("https://api.anthropic.com/v1/messages",
                             {"model": model, "max_tokens": 2048, "system": system,
                              "messages": convo},
                             {"x-api-key": key, "anthropic-version": "2023-06-01"})
            return "".join(b.get("text", "") for b in data.get("content", []))
        if provider == "gemini":
            contents = [{"role": "model" if m["role"] == "assistant" else "user",
                         "parts": [{"text": m["content"]}]} for m in convo]
            body = {"contents": contents}
            if system:
                body["system_instruction"] = {"parts": [{"text": system}]}
            data = _json_req(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                body, {"x-goog-api-key": key})
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"외부 AI가 오류를 돌려줬습니다 ({e.code}). 키와 모델명을 확인하세요.") from None
    except urllib.error.URLError:
        raise RuntimeError("외부 AI에 연결할 수 없습니다. 인터넷이 닿지 않는 망(폐쇄망)에서는 "
                           "외부 AI를 사용할 수 없습니다.") from None
    raise RuntimeError("외부 AI 제공자가 설정되지 않았습니다.")


# ---------- 업무 일지 (날짜별 할 일) ----------
WORKLOG_FILE = os.path.join(BASE_DIR, "worklog.json")


def load_worklog():
    try:
        with open(WORKLOG_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "days" in d:
            d.setdefault("recurring", [])
            d.setdefault("rec_done", {})
            return d
    except (OSError, ValueError):
        pass
    return {"seq": 0, "days": {}, "recurring": [], "rec_done": {}}


def save_worklog(d):
    try:
        with open(WORKLOG_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def sort_items(items):
    """시간이 있는 일을 시간순으로 앞에, 나머지는 추가한 순서를 지킨다."""
    return sorted(items, key=lambda it: (0, it["time"]) if it.get("time") else (1, ""))


def day_items(wl, d):
    """그 날짜의 할 일 = 날짜에 직접 붙은 일 + 그날에 해당하는 반복 일정의 합집합."""
    items = [dict(it) for it in wl["days"].get(d, [])]
    try:
        y, m, dd = (int(x) for x in d.split("-"))
        wd = datetime(y, m, dd).weekday()
    except ValueError:
        return sort_items(items)
    for s in wl.get("recurring", []):
        if d < s["start"]:
            continue
        if s["repeat"] == "daily" or (s["repeat"] == "weekly" and s["weekday"] == wd):
            items.append({"id": "r" + str(s["id"]), "series": s["id"], "text": s["text"],
                          "time": s.get("time", ""), "repeat": s["repeat"],
                          "done": d in wl.get("rec_done", {}).get(str(s["id"]), [])})
    return sort_items(items)


# ---------- 긴 문서 작업 (정리본·발표자료, 백그라운드 단계 처리) ----------
JOBS = {}
JOB_LOCK = threading.Lock()
JOB_SEQ = [0]

ORGANIZE_SYSTEM = ("너는 창구 업무 문서를 정리하는 도우미다. 주어진 문서를 창구에서 바로 찾아 쓰기 좋은 "
                   "정리본으로 재구성한다. 문서에 없는 내용은 절대 쓰지 않는다. 한국어로 쓴다.")
ORGANIZE_GUIDE = ("다음 구성으로 정리본을 만들어라. 해당 내용이 문서에 없으면 그 항목은 빼라.\n"
                  "## 한눈에\n(한두 문장 개요)\n"
                  "## 핵심만\n(꼭 알아야 할 것, 목록)\n"
                  "## 절차\n(순서가 있는 일이면 번호 목록)\n"
                  "## 주의·금지\n(하면 안 되는 것, 확인 사항)\n"
                  "## 헷갈리는 용어\n(용어 = 쉬운 뜻)\n\n문서:\n")
EXTRACT_SYSTEM = ("너는 큰 문서의 일부를 발췌하는 도우미다. 주어진 부분에서 핵심 사실, 절차, 수치, "
                  "주의사항만 빠짐없이 뽑는다. 없는 내용은 쓰지 않는다. 목록 위주로 짧게 쓴다.")


def gen_text(msgs, model):
    if external_active():
        return external_chat(msgs)
    with STATE["chat_lock"]:
        use_model(model)
        return agent.chat(msgs)


def doc_parts(name, size=3200):
    """문서 전문을 단락 경계에서 약 size자 묶음으로 나눈다."""
    text = agent.read_doc_text(os.path.join(agent.DOCS_DIR, name))
    parts, cur = [], ""
    for p in agent._split_paras(text):
        if cur and len(cur) + len(p) > size:
            parts.append(cur)
            cur = p
        else:
            cur = cur + "\n\n" + p if cur else p
    if cur:
        parts.append(cur)
    return parts


def job_source(job, name):
    """작업의 재료 텍스트. 짧으면 전문, 길면 조각별 발췌를 합친 것."""
    parts = doc_parts(name)
    if len(parts) <= 1:
        job["total"] = 1
        job["step"] = 1
        return parts[0] if parts else ""
    job["total"] = len(parts) + 1
    extracts = []
    for i, part in enumerate(parts):
        job["step"] = i + 1
        extracts.append(gen_text(
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": part}], STATE["memo_model"]))
    job["step"] = len(parts) + 1
    return "\n\n".join(extracts)


def run_job(job):
    name = job["name"]
    try:
        source = job_source(job, name)
        if job["kind"] == "organize":
            text = gen_text([{"role": "system", "content": ORGANIZE_SYSTEM},
                             {"role": "user", "content": ORGANIZE_GUIDE + source}],
                            STATE["qa_model"])
            out_dir = os.path.join(BASE_DIR, "정리본")
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.splitext(name)[0] + "_정리본.md"
            try:
                with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
                    f.write("# " + out_name + "\n\n" + text)
            except OSError:
                out_name = ""
            append_log({"type": "organize"})
            job.update(state="done", text=text, saved=out_name)
        else:  # slides
            raw = gen_text(
                [{"role": "system", "content":
                  "너는 발표자료를 만드는 도우미다. 주어진 내용만으로 슬라이드 개요를 만든다. "
                  "없는 내용은 쓰지 않는다. 출력은 JSON만, 설명 문장 없이."},
                 {"role": "user", "content":
                  "다음 내용으로 발표 슬라이드 8~12장을 만들어라. 형식: "
                  "{\"slides\": [{\"title\": \"장 제목\", \"bullets\": [\"핵심 한 줄\", ...]}]}. "
                  "장마다 bullet 3~4개, 한 줄씩 짧게.\n\n" + source}],
                STATE["qa_model"])
            m = re.search(r"\{.*\}", raw, re.S)
            slides = json.loads(m.group(0))["slides"] if m else []
            slides = [{"title": str(s.get("title", "")), 
                       "bullets": [str(b) for b in s.get("bullets", [])]}
                      for s in slides if isinstance(s, dict)]
            if not slides:
                raise RuntimeError("발표자료 개요를 만들지 못했습니다. 다시 시도해 주세요.")
            append_log({"type": "slides"})
            job.update(state="done", slides=slides)
    except RuntimeError as e:
        job.update(state="error", error=str(e))
    except Exception as e:
        job.update(state="error", error=f"처리 중 오류가 발생했습니다: {e}")


def start_job(kind, name):
    with JOB_LOCK:
        JOB_SEQ[0] += 1
        jid = JOB_SEQ[0]
        JOBS[jid] = {"kind": kind, "name": name, "state": "working", "step": 0, "total": 0}
    threading.Thread(target=run_job, args=(JOBS[jid],), daemon=True).start()
    return jid


# ---------- 의미 검색 색인 (임베딩, 백그라운드 구축) ----------
EMBED_CACHE_FILE = os.path.join(BASE_DIR, "embeddings_cache.json")
EMBED = {"vecs": {}, "ready": False, "building": False}


def _text_key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_embeddings():
    """단락 임베딩을 백그라운드로 만든다. 모델이 없으면 단어 검색 모드로 남는다."""
    if EMBED["building"]:
        return
    EMBED["building"] = True

    def run():
        try:
            try:
                with open(EMBED_CACHE_FILE, encoding="utf-8") as f:
                    cache = json.load(f)
            except (OSError, ValueError):
                cache = {}
            chunks = STATE["chunks"]
            missing = [c for c in chunks if _text_key(c["text"]) not in cache]
            for i in range(0, len(missing), 32):
                batch = missing[i:i + 32]
                vecs = agent.embed([c["text"] for c in batch])
                if vecs is None:
                    return
                for c, v in zip(batch, vecs):
                    cache[_text_key(c["text"])] = v
            EMBED["vecs"] = {_text_key(c["text"]): cache[_text_key(c["text"])]
                             for c in chunks if _text_key(c["text"]) in cache}
            EMBED["ready"] = bool(EMBED["vecs"])
            try:
                with open(EMBED_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except OSError:
                pass
        finally:
            EMBED["building"] = False

    threading.Thread(target=run, daemon=True).start()


def hybrid_retrieve(chunks, query):
    """단어 검색 + 의미 검색. 어느 한쪽이라도 기준을 넘으면 근거로 쓴다."""
    qv = agent.ngrams(query)
    word = {_text_key(c["text"]): agent.cosine(qv, c["vec"]) for c in chunks}
    emb = {}
    if EMBED["ready"]:
        qe = agent.embed([query])
        if qe:
            emb = {_text_key(c["text"]): agent.vec_cosine(qe[0], v)
                   for c in chunks if (v := EMBED["vecs"].get(_text_key(c["text"])))}
    hits = [c for c in chunks
            if word.get(_text_key(c["text"]), 0.0) > agent.RETRIEVE_THRESHOLD
            or emb.get(_text_key(c["text"]), 0.0) > agent.EMBED_THRESHOLD]
    hits.sort(key=lambda c: max(word.get(_text_key(c["text"]), 0.0),
                                emb.get(_text_key(c["text"]), 0.0)), reverse=True)
    return hits[:3]


def ollama_status():
    """Ollama 서버 상태와 설치된 모델 목록. 연결 실패는 예외 대신 상태값으로."""
    try:
        with urllib.request.urlopen(agent.OLLAMA + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return True, [m["name"] for m in tags.get("models", [])]
    except Exception:
        return False, []


def use_model(model):
    agent.MODEL = model
    if STATE["num_gpu"]:
        agent.CHAT_OPTIONS["num_gpu"] = STATE["num_gpu"]


def chat_stream(messages):
    """Ollama 스트리밍 호출. (조각 텍스트)들을 흘리고 마지막에 실측 지표를 붙인다."""
    body = json.dumps({"model": agent.MODEL, "messages": messages, "stream": True,
                       "keep_alive": agent.KEEP_ALIVE,
                       "options": agent.CHAT_OPTIONS}).encode("utf-8")
    req = urllib.request.Request(agent.OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                obj = json.loads(line.decode("utf-8"))
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield {"type": "delta", "text": piece}
                if obj.get("done"):
                    dur = obj.get("eval_duration") or 0
                    cnt = obj.get("eval_count") or 0
                    yield {"type": "stats", "eval_count": cnt,
                           "tok_per_sec": round(cnt / (dur / 1e9), 1) if dur else None}
    except urllib.error.URLError:
        raise RuntimeError("Ollama 로컬 서버에 연결할 수 없습니다. 서버 실행 여부를 확인하세요.") from None


def summarize(masked_text):
    msgs = [
        {"role": "system", "content": "당신은 창구 상담 메모를 정리하는 비서입니다. 마스킹된 기호는 그대로 두세요."},
        {"role": "user", "content": f"다음 상담 메모를 세 줄로 요약해 주세요.\n\n{masked_text}"},
    ]
    if external_active():
        return external_chat(msgs)
    with STATE["chat_lock"]:
        use_model(STATE["memo_model"])
        return agent.chat(msgs)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 콘솔 소음 제거
        pass

    # ---------- helpers ----------
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    @staticmethod
    def counts_of(found):
        return dict(Counter(k for k, _ in found))

    def doc_summary(self):
        docs = Counter(c["source"] for c in STATE["chunks"])
        return docs

    # ---------- routes ----------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(UI_FILE, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            ok, models = ollama_status()
            docs = self.doc_summary()
            self.send_json({
                "ollama": ok, "models": models,
                "qa_model": STATE["qa_model"], "memo_model": STATE["memo_model"],
                "num_gpu": STATE["num_gpu"],
                "doc_files": len(docs), "doc_chunks": len(STATE["chunks"]),
                "counters": STATE["counters"], "today": log_today(),
                "mask_level": SETTINGS["mask_level"],
                "search_mode": "의미+단어" if EMBED["ready"] else "단어",
                "external": {"provider": SETTINGS["external"]["provider"],
                             "model": SETTINGS["external"]["model"],
                             "enabled": external_active(),
                             "has_key": bool(SETTINGS["external"]["api_key"])},
            })
        elif self.path.startswith("/api/job"):
            from urllib.parse import urlparse, parse_qs
            jid = parse_qs(urlparse(self.path).query).get("job", [""])[0]
            job = JOBS.get(int(jid)) if jid.isdigit() else None
            if job is None:
                self.send_json({"error": "작업을 찾을 수 없습니다."}, 404)
                return
            out = {"state": job["state"], "step": job.get("step", 0),
                   "total": job.get("total", 0), "kind": job["kind"]}
            for k in ("text", "saved", "slides", "error"):
                if k in job:
                    out[k] = job[k]
            self.send_json(out)
        elif self.path == "/api/docs":
            docs = self.doc_summary()
            self.send_json({"docs": [{"name": k, "paras": v} for k, v in sorted(docs.items())]})
        elif self.path.startswith("/api/docs/preview"):
            from urllib.parse import urlparse, parse_qs, unquote
            name = unquote(parse_qs(urlparse(self.path).query).get("name", [""])[0])
            paras = [c["text"] for c in STATE["chunks"] if c["source"] == name]
            self.send_json({"name": name, "paras": paras})
        elif self.path.startswith("/api/worklog"):
            from urllib.parse import urlparse, parse_qs
            d = parse_qs(urlparse(self.path).query).get("date", [""])[0] \
                or datetime.now().date().isoformat()
            wl = load_worklog()
            self.send_json({"date": d, "items": day_items(wl, d),
                            "counts": log_counts(d)})
        elif self.path.startswith("/api/calendar"):
            from urllib.parse import urlparse, parse_qs
            month = parse_qs(urlparse(self.path).query).get("month", [""])[0] \
                or datetime.now().strftime("%Y-%m")
            counts = {d: c for d, c in log_all_counts().items() if d.startswith(month)}
            wl = load_worklog()
            todos = {}
            import calendar as _cal
            _y, _mo = (int(x) for x in month.split("-"))
            for _dd in range(1, _cal.monthrange(_y, _mo)[1] + 1):
                iso = f"{month}-{_dd:02d}"
                merged = day_items(wl, iso)
                if merged:
                    todos[iso] = {"total": len(merged),
                                  "done": sum(1 for it in merged if it["done"])}
            self.send_json({"month": month, "counts": counts, "todos": todos})
        elif self.path.startswith("/api/records"):
            from urllib.parse import urlparse, parse_qs
            from datetime import timedelta
            days = int(parse_qs(urlparse(self.path).query).get("days", ["7"])[0])
            all_counts = log_all_counts()
            out = []
            base = datetime.now().date()
            for i in range(days - 1, -1, -1):
                d = (base - timedelta(days=i)).isoformat()
                c = all_counts.get(d, {"asks": 0, "memos": 0, "masked_items": 0})
                out.append({"date": d, **c})
            self.send_json({"records": out})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self.read_json()
            if self.path == "/api/ask_stream":
                self.handle_ask_stream(data)
            elif self.path == "/api/mask":
                masked, found = agent.mask_text(data.get("text", ""), SETTINGS["mask_level"])
                counts = self.counts_of(found)
                append_log({"type": "mask", "counts": counts})
                self.send_json({"masked": masked, "counts": counts})
            elif self.path == "/api/reload":
                with STATE["lock"]:
                    STATE["chunks"] = agent.load_chunks()
                build_embeddings()
                docs = self.doc_summary()
                self.send_json({"ok": True, "doc_files": len(docs),
                                "doc_chunks": len(STATE["chunks"])})
            elif self.path == "/api/settings":
                self.handle_settings(data)
            elif self.path == "/api/worklog/add":
                d = data.get("date") or datetime.now().date().isoformat()
                text = (data.get("text") or "").strip()
                if not text:
                    self.send_json({"error": "할 일을 입력해 주세요."}, 400)
                    return
                t = (data.get("time") or "").strip()
                if t and not re.match(r"^\d{2}:\d{2}$", t):
                    t = ""
                repeat = data.get("repeat") or ""
                wl = load_worklog()
                wl["seq"] += 1
                if repeat in ("daily", "weekly"):
                    y2, m2, d2 = (int(x) for x in d.split("-"))
                    wl["recurring"].append({"id": wl["seq"], "text": text, "time": t,
                                            "repeat": repeat, "start": d,
                                            "weekday": datetime(y2, m2, d2).weekday()})
                    save_worklog(wl)
                    self.send_json({"items": day_items(wl, d)})
                    return
                item = {"id": wl["seq"], "text": text, "done": False, "time": t}
                wl["days"].setdefault(d, []).append(item)
                save_worklog(wl)
                self.send_json({"item": item, "items": day_items(wl, d)})
            elif self.path in ("/api/worklog/toggle", "/api/worklog/delete",
                               "/api/worklog/move"):
                d = data.get("date") or datetime.now().date().isoformat()
                wl = load_worklog()
                iid = data.get("id")
                if isinstance(iid, str) and iid.startswith("r"):
                    sid = iid[1:]
                    if self.path.endswith("toggle"):
                        done_list = wl["rec_done"].setdefault(sid, [])
                        if d in done_list:
                            done_list.remove(d)
                        else:
                            done_list.append(d)
                    elif self.path.endswith("delete"):
                        wl["recurring"] = [s for s in wl["recurring"]
                                           if str(s["id"]) != sid]
                        wl["rec_done"].pop(sid, None)
                    # 반복 일정은 날짜 이동(미루기)을 적용하지 않는다
                    save_worklog(wl)
                    self.send_json({"items": day_items(wl, d)})
                    return
                items = wl["days"].get(d, [])
                if self.path.endswith("toggle"):
                    for it in items:
                        if it["id"] == data.get("id"):
                            it["done"] = not it["done"]
                elif self.path.endswith("delete"):
                    items = [it for it in items if it["id"] != data.get("id")]
                    wl["days"][d] = items
                else:  # move: 다른 날짜로 옮기기 (내일로 미루기)
                    to = (data.get("to") or "").strip()
                    moving = [it for it in items if it["id"] == data.get("id")]
                    if to and moving:
                        wl["days"][d] = [it for it in items if it["id"] != data.get("id")]
                        wl["days"].setdefault(to, []).extend(moving)
                save_worklog(wl)
                self.send_json({"items": day_items(wl, d)})
            elif self.path == "/api/organize":
                self.handle_organize(data)
            elif self.path == "/api/slides":
                self.handle_slides(data)
            elif self.path == "/api/summarize":
                self.handle_summarize(data)
            elif self.path == "/api/docs/upload":
                self.handle_upload(data)
            elif self.path == "/api/model":
                if data.get("qa_model"):
                    STATE["qa_model"] = data["qa_model"]
                if data.get("memo_model"):
                    STATE["memo_model"] = data["memo_model"]
                self.send_json({"qa_model": STATE["qa_model"], "memo_model": STATE["memo_model"]})
            else:
                self.send_json({"error": "not found"}, 404)
        except RuntimeError as e:  # Ollama 미연결 등 사용자에게 보여줄 오류
            self.send_json({"error": str(e)}, 503)
        except Exception as e:  # 그 외 오류도 traceback 대신 메시지로
            self.send_json({"error": f"처리 중 오류가 발생했습니다: {e}"}, 500)

    def handle_ask_stream(self, data):
        q = (data.get("question") or "").strip()
        if not q:
            self.send_json({"error": "질문을 입력해 주세요."}, 400)
            return
        masked_q, found = agent.mask_text(q, SETTINGS["mask_level"])
        STATE["counters"]["masked_items"] += len(found)
        # 이어서 묻기: 이전 질문을 검색에 함께 넣고, 이전 답변은 지시어 해석용으로만 쓴다
        ctx = data.get("context") or {}
        prev_q_m, prev_a_m = "", ""
        if (ctx.get("q") or "").strip():
            prev_q_m, _ = agent.mask_text(ctx["q"].strip(), SETTINGS["mask_level"])
            prev_a_m, _ = agent.mask_text((ctx.get("a") or "").strip()[:600],
                                          SETTINGS["mask_level"])
        search_q = (prev_q_m + " " + masked_q).strip() if prev_q_m else masked_q
        with STATE["lock"]:
            hits = hybrid_retrieve(STATE["chunks"], search_q)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        meta = {"type": "meta", "masked_count": len(found)}
        if not hits:
            meta.update({"found": False,
                         "answer": "문서에서 관련 근거를 찾지 못했습니다. 문서에 없는 내용은 지어내지 않습니다."})
            emit(meta)
            append_log({"type": "ask", "found": False, "model": STATE["qa_model"],
                        "masked_count": len(found)})
            emit({"type": "done", "elapsed_s": 0})
            return
        meta.update({"found": True,
                     "sources": sorted({h["source"] for h in hits}),
                     "evidence": [{"source": h["source"], "idx": h["idx"],
                                   "text": h["text"]} for h in hits]})
        emit(meta)
        context = "\n\n".join(f"[근거:{h['source']}] {h['text']}" for h in hits)
        if prev_q_m:
            user_content = (f"이전 질문: {prev_q_m}\n이전 답변: {prev_a_m}\n\n{context}\n\n"
                            f"후속 질문: {masked_q}\n"
                            "(후속 질문의 '그것' 같은 지시어는 이전 대화를 참고해 해석하되, "
                            "답의 근거는 위 근거만 쓴다.)")
        else:
            user_content = f"{context}\n\n질문: {masked_q}"
        msgs = [
            {"role": "system", "content": agent.SYSTEM},
            {"role": "user", "content": user_content},
        ]
        started = time.time()
        try:
            if external_active():
                emit({"type": "delta", "text": external_chat(msgs)})
                emit({"type": "stats", "eval_count": None, "tok_per_sec": None,
                      "external": SETTINGS["external"]["provider"]})
            else:
                with STATE["chat_lock"]:
                    use_model(STATE["qa_model"])
                    for event in chat_stream(msgs):
                        emit(event)
            STATE["counters"]["asks"] += 1
            append_log({"type": "ask", "found": True,
                        "model": ("external:" + SETTINGS["external"]["provider"])
                                 if external_active() else STATE["qa_model"],
                        "masked_count": len(found)})
            emit({"type": "done", "elapsed_s": round(time.time() - started, 1)})
        except RuntimeError as e:
            emit({"type": "error", "message": str(e)})

    def handle_organize(self, data):
        """정리본 생성을 백그라운드로 시작하고 job id를 돌려준다 (긴 문서 대응)."""
        name = os.path.basename(data.get("name") or "")
        if not name.lower().endswith(agent.DOC_EXTS) or \
                not os.path.isfile(os.path.join(agent.DOCS_DIR, name)):
            self.send_json({"error": "문서를 찾을 수 없습니다."}, 404)
            return
        self.send_json({"job": start_job("organize", name)})

    def handle_slides(self, data):
        """발표자료 생성을 백그라운드로 시작한다."""
        name = os.path.basename(data.get("name") or "")
        if not name.lower().endswith(agent.DOC_EXTS) or \
                not os.path.isfile(os.path.join(agent.DOCS_DIR, name)):
            self.send_json({"error": "문서를 찾을 수 없습니다."}, 404)
            return
        self.send_json({"job": start_job("slides", name)})

    def handle_summarize(self, data):
        masked, found = agent.mask_text(data.get("text", ""), SETTINGS["mask_level"])
        if not masked.strip():
            self.send_json({"error": "메모를 입력해 주세요."}, 400)
            return
        STATE["counters"]["masked_items"] += len(found)
        summary = summarize(masked)
        STATE["counters"]["memos"] += 1
        counts = self.counts_of(found)
        append_log({"type": "summarize", "model": STATE["memo_model"], "counts": counts})
        self.send_json({"masked": masked, "counts": counts, "summary": summary})

    def handle_settings(self, data):
        if data.get("mask_level") in ("full", "partial"):
            SETTINGS["mask_level"] = data["mask_level"]
        if data.get("qa_model"):
            STATE["qa_model"] = data["qa_model"]
        if data.get("memo_model"):
            STATE["memo_model"] = data["memo_model"]
        ext = data.get("external")
        if isinstance(ext, dict):
            provider = ext.get("provider", SETTINGS["external"]["provider"])
            if provider == "":
                SETTINGS["external"] = {"provider": "", "model": "", "api_key": ""}
            elif provider in DEFAULT_EXTERNAL_MODELS:
                key = ext.get("api_key") or SETTINGS["external"]["api_key"]
                if ext.get("clear_key"):
                    key = ""
                SETTINGS["external"] = {
                    "provider": provider,
                    "model": ext.get("model") or SETTINGS["external"]["model"]
                            or DEFAULT_EXTERNAL_MODELS[provider],
                    "api_key": key}
        save_settings()
        self.send_json({"ok": True, "mask_level": SETTINGS["mask_level"],
                        "external": {"provider": SETTINGS["external"]["provider"],
                                     "model": SETTINGS["external"]["model"],
                                     "enabled": external_active(),
                                     "has_key": bool(SETTINGS["external"]["api_key"])}})

    def handle_upload(self, data):
        name = os.path.basename((data.get("name") or "").strip())
        if not name.lower().endswith(agent.DOC_EXTS):
            self.send_json({"error": "txt, md, docx, pptx, pdf 파일만 올릴 수 있습니다."}, 400)
            return
        os.makedirs(agent.DOCS_DIR, exist_ok=True)
        path = os.path.join(agent.DOCS_DIR, name)
        if data.get("content_b64"):
            import base64
            try:
                raw = base64.b64decode(data["content_b64"])
            except Exception:
                self.send_json({"error": "파일을 읽지 못했습니다."}, 400)
                return
            with open(path, "wb") as f:
                f.write(raw)
        else:
            content = data.get("content") or ""
            if not content.strip():
                self.send_json({"error": "내용이 필요합니다."}, 400)
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        with STATE["lock"]:
            STATE["chunks"] = agent.load_chunks()
        build_embeddings()
        if not any(c["source"] == name for c in STATE["chunks"]):
            try:
                os.remove(path)
            except OSError:
                pass
            self.send_json({"error": "파일에서 글자를 뽑지 못했습니다. "
                                     "스캔한 이미지 문서는 글자 추출이 안 됩니다."}, 400)
            return
        self.send_json({"ok": True, "doc_chunks": len(STATE["chunks"])})


def main():
    use_model(STATE["qa_model"])
    port = int(os.environ.get("LOCALDESK_PORT", "8765"))
    server = None
    for cand in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", cand), Handler)
            port = cand
            break
        except OSError:
            continue
    if server is None:
        print("사용 가능한 포트를 찾지 못했습니다.")
        return
    url = f"http://127.0.0.1:{port}"
    build_embeddings()
    print(f"로컬 업무 비서 실행 중: {url}  (종료: Ctrl+C)")
    print(f"질문 모델: {STATE['qa_model']} | 요약 모델: {STATE['memo_model']} | 색인 문서 단락: {len(STATE['chunks'])}개 | 외부 연결: 없음")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

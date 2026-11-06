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
import json
import os
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


def log_today():
    """오늘 날짜의 기록만 모아 건수를 센다 (재실행해도 남는 누적)."""
    today = datetime.now().date().isoformat()
    agg = {"asks": 0, "memos": 0, "masked_items": 0}
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if not str(e.get("ts", "")).startswith(today):
                    continue
                if e.get("type") == "ask":
                    agg["asks"] += 1
                elif e.get("type") == "summarize":
                    agg["memos"] += 1
                    agg["masked_items"] += sum((e.get("counts") or {}).values())
                elif e.get("type") == "mask":
                    agg["masked_items"] += sum((e.get("counts") or {}).values())
    except OSError:
        pass
    return agg


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
            })
        elif self.path == "/api/docs":
            docs = self.doc_summary()
            self.send_json({"docs": [{"name": k, "paras": v} for k, v in sorted(docs.items())]})
        elif self.path.startswith("/api/docs/preview"):
            from urllib.parse import urlparse, parse_qs, unquote
            name = unquote(parse_qs(urlparse(self.path).query).get("name", [""])[0])
            paras = [c["text"] for c in STATE["chunks"] if c["source"] == name]
            self.send_json({"name": name, "paras": paras})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self.read_json()
            if self.path == "/api/ask_stream":
                self.handle_ask_stream(data)
            elif self.path == "/api/mask":
                masked, found = agent.mask_text(data.get("text", ""))
                counts = self.counts_of(found)
                append_log({"type": "mask", "counts": counts})
                self.send_json({"masked": masked, "counts": counts})
            elif self.path == "/api/reload":
                with STATE["lock"]:
                    STATE["chunks"] = agent.load_chunks()
                docs = self.doc_summary()
                self.send_json({"ok": True, "doc_files": len(docs),
                                "doc_chunks": len(STATE["chunks"])})
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
        masked_q, found = agent.mask_text(q)
        STATE["counters"]["masked_items"] += len(found)
        with STATE["lock"]:
            hits = agent.retrieve(STATE["chunks"], masked_q)
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
                     "evidence": [h["text"] for h in hits]})
        emit(meta)
        context = "\n\n".join(f"[근거:{h['source']}] {h['text']}" for h in hits)
        msgs = [
            {"role": "system", "content": agent.SYSTEM},
            {"role": "user", "content": f"{context}\n\n질문: {masked_q}"},
        ]
        started = time.time()
        try:
            with STATE["chat_lock"]:
                use_model(STATE["qa_model"])
                for event in chat_stream(msgs):
                    emit(event)
            STATE["counters"]["asks"] += 1
            append_log({"type": "ask", "found": True, "model": STATE["qa_model"],
                        "masked_count": len(found)})
            emit({"type": "done", "elapsed_s": round(time.time() - started, 1)})
        except RuntimeError as e:
            emit({"type": "error", "message": str(e)})

    def handle_summarize(self, data):
        masked, found = agent.mask_text(data.get("text", ""))
        if not masked.strip():
            self.send_json({"error": "메모를 입력해 주세요."}, 400)
            return
        STATE["counters"]["masked_items"] += len(found)
        summary = summarize(masked)
        STATE["counters"]["memos"] += 1
        counts = self.counts_of(found)
        append_log({"type": "summarize", "model": STATE["memo_model"], "counts": counts})
        self.send_json({"masked": masked, "counts": counts, "summary": summary})

    def handle_upload(self, data):
        name = os.path.basename((data.get("name") or "").strip())
        content = data.get("content") or ""
        if not name.endswith((".txt", ".md")) or not content.strip():
            self.send_json({"error": "txt 또는 md 파일과 내용이 필요합니다."}, 400)
            return
        os.makedirs(agent.DOCS_DIR, exist_ok=True)
        with open(os.path.join(agent.DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)
        with STATE["lock"]:
            STATE["chunks"] = agent.load_chunks()
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
    print(f"로컬 업무 비서 실행 중: {url}  (종료: Ctrl+C)")
    print(f"질문 모델: {STATE['qa_model']} | 요약 모델: {STATE['memo_model']} | 색인 문서 단락: {len(STATE['chunks'])}개 | 외부 연결: 없음")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

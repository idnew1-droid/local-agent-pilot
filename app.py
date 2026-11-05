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
  LOCAL_AGENT_MODEL       기본 모델 (agent.py와 동일)
  LOCAL_AGENT_NUM_GPU     GPU 적재 층 수 (agent.py와 동일)
"""
import json
import os
import threading
import urllib.request
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import agent  # 같은 폴더의 코어 로직 (마스킹·검색·Ollama 호출)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FILE = os.path.join(BASE_DIR, "ui.html")

STATE = {
    "chunks": agent.load_chunks(),
    "model": agent.MODEL,
    "num_gpu": agent.CHAT_OPTIONS.get("num_gpu"),
    "lock": threading.Lock(),
}


def ollama_status():
    """Ollama 서버 상태와 설치된 모델 목록. 연결 실패는 예외 대신 상태값으로."""
    try:
        with urllib.request.urlopen(agent.OLLAMA + "/api/tags", timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return True, [m["name"] for m in tags.get("models", [])]
    except Exception:
        return False, []


def apply_model():
    agent.MODEL = STATE["model"]
    if STATE["num_gpu"]:
        agent.CHAT_OPTIONS["num_gpu"] = STATE["num_gpu"]
    else:
        agent.CHAT_OPTIONS.pop("num_gpu", None)


def summarize(masked_text):
    msgs = [
        {"role": "system", "content": "당신은 창구 상담 메모를 정리하는 비서입니다. 마스킹된 기호는 그대로 두세요."},
        {"role": "user", "content": f"다음 상담 메모를 세 줄로 요약해 주세요.\n\n{masked_text}"},
    ]
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

    def counts_of(self, found):
        return dict(Counter(k for k, _ in found))

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
            docs = Counter(c["source"] for c in STATE["chunks"])
            self.send_json({
                "ollama": ok, "models": models, "model": STATE["model"],
                "num_gpu": STATE["num_gpu"],
                "doc_files": len(docs), "doc_chunks": len(STATE["chunks"]),
            })
        elif self.path == "/api/docs":
            docs = Counter(c["source"] for c in STATE["chunks"])
            self.send_json({"docs": [{"name": k, "paras": v} for k, v in sorted(docs.items())]})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            data = self.read_json()
            if self.path == "/api/ask":
                self.handle_ask(data)
            elif self.path == "/api/mask":
                masked, found = agent.mask_text(data.get("text", ""))
                self.send_json({"masked": masked, "counts": self.counts_of(found)})
            elif self.path == "/api/summarize":
                masked, found = agent.mask_text(data.get("text", ""))
                if not masked.strip():
                    self.send_json({"error": "메모를 입력해 주세요."}, 400)
                    return
                self.send_json({"masked": masked, "counts": self.counts_of(found),
                                "summary": summarize(masked)})
            elif self.path == "/api/docs/upload":
                self.handle_upload(data)
            elif self.path == "/api/model":
                STATE["model"] = data.get("model", STATE["model"])
                STATE["num_gpu"] = data.get("num_gpu") or None
                apply_model()
                self.send_json({"model": STATE["model"], "num_gpu": STATE["num_gpu"]})
            else:
                self.send_json({"error": "not found"}, 404)
        except RuntimeError as e:  # Ollama 미연결 등 사용자에게 보여줄 오류
            self.send_json({"error": str(e)}, 503)
        except Exception as e:  # 그 외 오류도 traceback 대신 메시지로
            self.send_json({"error": f"처리 중 오류가 발생했습니다: {e}"}, 500)

    def handle_ask(self, data):
        q = (data.get("question") or "").strip()
        if not q:
            self.send_json({"error": "질문을 입력해 주세요."}, 400)
            return
        masked_q, found = agent.mask_text(q)
        with STATE["lock"]:
            hits = agent.retrieve(STATE["chunks"], masked_q)
        if not hits:
            self.send_json({"found": False, "masked_count": len(found),
                            "answer": "문서에서 관련 근거를 찾지 못했습니다. 문서에 없는 내용은 지어내지 않습니다."})
            return
        context = "\n\n".join(f"[근거:{h['source']}] {h['text']}" for h in hits)
        msgs = [
            {"role": "system", "content": agent.SYSTEM},
            {"role": "user", "content": f"{context}\n\n질문: {masked_q}"},
        ]
        self.send_json({"found": True, "masked_count": len(found),
                        "answer": agent.chat(msgs),
                        "sources": sorted({h["source"] for h in hits}),
                        "evidence": [h["text"] for h in hits]})

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
    apply_model()
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
    print(f"모델: {STATE['model']} | 색인 문서 단락: {len(STATE['chunks'])}개 | 외부 연결: 없음")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

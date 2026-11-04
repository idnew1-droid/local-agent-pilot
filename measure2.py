# -*- coding: utf-8 -*-
import json, urllib.request
OLLAMA = "http://localhost:11434"
PROMPT = "보이스피싱 예방 수칙 10가지를 각각 두 문장 이상으로 자세히 설명하세요."

def chat(options=None):
    payload = {"model": "exaone3.5:7.8b",
               "messages": [{"role": "user", "content": PROMPT}], "stream": False}
    if options:
        payload["options"] = options
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))

def ps():
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))

try:
    r = chat({"num_predict": 300, "num_ctx": 4096, "num_gpu": 32})
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0) / 1e9
    vram = [(m["name"], round(m["size"] / 1e9, 2), round(m["size_vram"] / 1e9, 2))
            for m in ps().get("models", [])]
    print("num_gpu=32 | tok/s:", round(ec / ed, 1) if ed else None,
          "| eval tokens:", ec, "| (model GB, vram GB):", vram, flush=True)
except Exception as e:
    print("num_gpu=32 FAILED:", e, flush=True)
print("MEASURE2_DONE")

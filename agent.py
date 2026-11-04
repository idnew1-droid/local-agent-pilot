# -*- coding: utf-8 -*-
"""
로컬 에이전트 파일럿 — 개인 PC 안에서만 동작하는 업무 보조 에이전트.

설계 원칙
- 모든 추론은 로컬 Ollama(localhost)에서만 수행한다. 외부 네트워크 호출 없음.
- 업무 문서는 지정 폴더 안에서만 읽고, 색인·로그도 같은 폴더 안에 남긴다.
- 민감정보(주민번호·계좌·전화 등)는 모델에 보내기 전에 마스킹한다.

사용법
  python agent.py            대화형 모드 (문서 질의응답)
  python agent.py --mask     상담 메모 마스킹+요약 모드 (메모를 붙여넣고 빈 줄로 종료)
환경변수
  LOCAL_AGENT_MODEL   사용할 모델 (기본 exaone3.5:7.8b)
  LOCAL_AGENT_NUM_GPU GPU에 올릴 층 수 (미지정 시 Ollama 기본값. RTX 2060 6GB + 7.8B 실측:
                      기본값은 일부 층이 CPU로 내려가 8.5 tok/s, 전층 지정 시 29.1 tok/s)
"""
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
OLLAMA = "http://localhost:11434"
MODEL = os.environ.get("LOCAL_AGENT_MODEL", "exaone3.5:7.8b")
CHAT_OPTIONS = {"num_ctx": 4096}
if os.environ.get("LOCAL_AGENT_NUM_GPU"):
    CHAT_OPTIONS["num_gpu"] = int(os.environ["LOCAL_AGENT_NUM_GPU"])

# ---------------- 민감정보 마스킹 ----------------
PATTERNS = [
    ("주민등록번호", re.compile(r"\b\d{6}\s*-\s*[1-4]\d{6}\b")),
    ("전화번호", re.compile(r"\b01[016789]\s*-\s*\d{3,4}\s*-\s*\d{4}\b")),
    ("카드번호", re.compile(r"\b\d{4}\s*-\s*\d{4}\s*-\s*\d{4}\s*-\s*\d{4}\b")),
    ("계좌번호", re.compile(r"\b\d{3,6}\s*-\s*\d{2,6}\s*-\s*\d{4,6}(?:\s*-\s*\d{1,3})?\b")),
    ("이메일", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
]
LABEL_NAME = re.compile(r"(성명|이름|고객명|예금주)\s*[:：]\s*([가-힣]{2,4})")


def mask_text(text):
    """민감 패턴을 찾아 마스킹한다. (마스킹된 텍스트, 발견 목록) 반환."""
    found = []
    out = text
    m = LABEL_NAME.search(out)
    if m:
        found.append(("성명", m.group(2)))
        out = out.replace(m.group(2), m.group(2)[0] + "*" * (len(m.group(2)) - 1))
    for label, pat in PATTERNS:
        for hit in pat.findall(out):
            found.append((label, hit))
        out = pat.sub(f"[{label} 마스킹]", out)
    return out, found


# ---------------- 문서 색인 (문자 n-gram 코사인, 외부 라이브러리 없음) ----------------
def ngrams(text, n=2):
    t = re.sub(r"\s+", "", text)
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def cosine(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def load_chunks():
    chunks = []
    if not os.path.isdir(DOCS_DIR):
        return chunks
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.endswith((".txt", ".md")):
            continue
        with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
            text = f.read()
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for i, p in enumerate(paras):
            chunks.append({"source": name, "idx": i, "text": p, "vec": ngrams(p)})
    return chunks


def retrieve(chunks, query, topk=3):
    qv = ngrams(query)
    scored = sorted(chunks, key=lambda c: cosine(qv, c["vec"]), reverse=True)
    return [c for c in scored if cosine(qv, c["vec"]) > 0.02][:topk]


# ---------------- Ollama 호출 (localhost 전용) ----------------
def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False,
                       "options": CHAT_OPTIONS}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


SYSTEM = (
    "당신은 금융 창구 업무를 돕는 로컬 AI 비서입니다. "
    "반드시 제공된 [근거] 문서 안에서만 답하고, 근거에 없으면 '문서에서 확인되지 않습니다'라고 말하세요. "
    "고객의 개인정보는 어떤 경우에도 그대로 말하지 마세요."
)


def answer(chunks, question):
    hits = retrieve(chunks, question)
    if not hits:
        return "문서에서 관련 근거를 찾지 못했습니다.", []
    context = "\n\n".join(f"[근거:{h['source']}] {h['text']}" for h in hits)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{context}\n\n질문: {question}"},
    ]
    return chat(msgs), [h["source"] for h in hits]


def summarize_memo(raw):
    masked, found = mask_text(raw)
    print("\n[마스킹 결과]")
    print(masked)
    if found:
        print("[탐지된 민감정보]", ", ".join(f"{k}:{v}" for k, v in found))
    msgs = [
        {"role": "system", "content": "당신은 창구 상담 메모를 정리하는 비서입니다. 마스킹된 기호는 그대로 두세요."},
        {"role": "user", "content": f"다음 상담 메모를 세 줄로 요약해 주세요.\n\n{masked}"},
    ]
    print("\n[요약]")
    print(chat(msgs))


def main():
    chunks = load_chunks()
    print(f"로컬 에이전트 파일럿 | 모델: {MODEL} | 색인 문서 단락: {len(chunks)}개 | 외부 연결: 없음")
    if "--mask" in sys.argv:
        print("상담 메모를 붙여넣으세요. 빈 줄을 입력하면 요약합니다. (Ctrl+C 종료)")
        while True:
            try:
                lines = []
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                if lines:
                    summarize_memo("\n".join(lines))
            except (KeyboardInterrupt, EOFError):
                break
        return
    print("질문을 입력하세요. 종료: q")
    while True:
        try:
            q = input("\n질문> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        masked_q, found = mask_text(q)
        if found:
            print(f"(질문에서 민감정보 {len(found)}건을 마스킹했습니다)")
        ans, sources = answer(chunks, masked_q)
        print("\n답변:", ans)
        if sources:
            print("근거:", ", ".join(sorted(set(sources))))


if __name__ == "__main__":
    main()

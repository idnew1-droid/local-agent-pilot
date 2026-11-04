# -*- coding: utf-8 -*-
"""
실측 스크립트 — 응답 속도, 문서 질의응답 적중, 마스킹 탐지율을 측정해 results.json으로 남긴다.
수치는 이 스크립트가 실제로 측정한 값만 기록한다.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import CHAT_OPTIONS, MODEL, OLLAMA, load_chunks, mask_text, retrieve  # noqa: E402


def gen(prompt, num_predict=None):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "stream": False,
               "options": dict(CHAT_OPTIONS)}
    if num_predict:
        payload["options"]["num_predict"] = num_predict
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def speed_test():
    gen("안녕하세요")  # 워밍업 (모델 로딩 시간 제외)
    t0 = time.time()
    r = gen("보이스피싱 예방 수칙 10가지를 각각 두 문장 이상으로 자세히 설명하세요.",
            num_predict=400)
    wall = time.time() - t0
    ec = r.get("eval_count", 0)
    ed = r.get("eval_duration", 0) / 1e9
    return {"eval_count": ec, "eval_seconds": round(ed, 2),
            "tokens_per_sec": round(ec / ed, 1) if ed else None,
            "wall_seconds": round(wall, 2)}


QA_SET = [
    ("고액 현금거래는 얼마부터 보고 대상인가요?", "업무문서_수신_여신_샘플.txt"),
    ("보이스피싱이 의심되는 이체 요청을 받으면 어떻게 해야 하나요?", "업무문서_수신_여신_샘플.txt"),
    ("비대면 계좌개설이 제한되는 경우를 알려주세요", "업무문서_계좌_보안_샘플.txt"),
    ("예금 만기 해지 시 필요한 서류는 무엇인가요?", "업무문서_수신_여신_샘플.txt"),
    ("ATM에서 할 수 있는 이체 한도 연습 안내는 어떻게 하나요?", "업무문서_계좌_보안_샘플.txt"),
]


def qa_test(chunks):
    ok = 0
    detail = []
    for q, want in QA_SET:
        hits = retrieve(chunks, q)
        got = hits[0]["source"] if hits else None
        hit = got == want
        ok += hit
        detail.append({"q": q, "want": want, "got": got, "hit": hit})
    return {"score": f"{ok}/{len(QA_SET)}", "detail": detail}


MASK_SET = [
    "고객명: 김민수, 주민등록번호 850512-1234567, 연락처 010-2345-6789",
    "예금주: 이영희 / 계좌 123456-78-901234 에서 300만원 이체 요청",
    "성명: 박철수, 전화 010-9876-5432, 카드번호 1234-5678-9012-3456 분실 신고",
    "상담 내용: 적금 만기 상담, 특이사항 없음",
    "고객명: 최지연 이메일 test.user@example.com 으로 명세서 발송 요청",
]


def mask_test():
    total = sum(1 for s in MASK_SET for _ in [0])  # placeholder
    found_total = 0
    expected = [3, 2, 3, 0, 2]
    detail = []
    for s, exp in zip(MASK_SET, expected):
        masked, found = mask_text(s)
        found_total += len(found)
        detail.append({"input": s, "found": len(found), "expected": exp, "masked": masked})
    exp_total = sum(expected)
    return {"found": found_total, "expected": exp_total,
            "rate": f"{found_total}/{exp_total}", "detail": detail}


def main():
    chunks = load_chunks()
    results = {"model": MODEL, "chunks": len(chunks),
               "speed": speed_test(), "qa": qa_test(chunks), "mask": mask_test()}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()

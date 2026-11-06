# -*- coding: utf-8 -*-
"""임베딩 검색 임계값 실측: 문서 내/밖 질문의 의미 검색 점수 분포를 잰다."""
import agent

IN_DOC = [
    "보이스피싱이 의심되는 이체 요청을 받으면 어떻게 해야 하나요?",
    "고액 현금거래는 얼마부터 보고 대상인가요?",
    "계좌 비밀번호를 여러 번 틀린 고객은 어떻게 안내하나요?",
    "정기예금 중도해지 이율은 어떻게 되나요?",
    "예금 만기 후 이자는 어떻게 계산되나요?",
]
OUT_DOC = [
    "오늘 서울 날씨 어때요?",
    "점심 메뉴 추천해 주세요",
    "주식 시장 전망이 어떤가요?",
]

chunks = agent.load_chunks()
print(f"단락 {len(chunks)}개 임베딩 중 (모델: {agent.EMBED_MODEL})")
vecs = agent.embed([c["text"] for c in chunks])
if vecs is None:
    print("임베딩 모델이 없습니다. ollama pull", agent.EMBED_MODEL)
    raise SystemExit(1)
cache = {c["text"]: v for c, v in zip(chunks, vecs)}

def probe(q):
    qe = agent.embed([q])[0]
    emb = max(agent.vec_cosine(qe, v) for v in cache.values())
    qv = agent.ngrams(q)
    word = max(agent.cosine(qv, c["vec"]) for c in chunks)
    print(f"  emb {emb:.3f} | word {word:.3f} | {q}")
    return emb

print("[문서 내 질문]")
ins = [probe(q) for q in IN_DOC]
print("[문서 밖 질문]")
outs = [probe(q) for q in OUT_DOC]
print(f"문서 내 최소 emb: {min(ins):.3f} / 문서 밖 최대 emb: {max(outs):.3f}")

# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import load_chunks, ngrams, cosine

chunks = load_chunks()
QS = [
    ("있음", "고액 현금거래는 얼마부터 보고 대상인가요?"),
    ("있음", "보이스피싱이 의심되는 이체 요청을 받으면 어떻게 해야 하나요?"),
    ("있음", "비대면 계좌개설이 제한되는 경우를 알려주세요"),
    ("있음", "예금 만기 해지 시 필요한 서류는 무엇인가요?"),
    ("있음", "ATM에서 할 수 있는 이체 한도 연습 안내는 어떻게 하나요?"),
    ("없음", "오늘 달러 환율이 얼마인가요?"),
    ("없음", "내일 날씨가 어떤가요?"),
    ("없음", "대출 금리가 앞으로 오를까요?"),
    ("없음", "점심 메뉴 추천해 주세요"),
]
for tag, q in QS:
    qv = ngrams(q)
    best = max(cosine(qv, c["vec"]) for c in chunks)
    print(f"{tag} | 최고점수 {best:.3f} | {q}")
print("THRESHOLD_DONE")

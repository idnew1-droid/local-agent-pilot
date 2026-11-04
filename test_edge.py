# -*- coding: utf-8 -*-
"""마스킹 엣지케이스 실측 — 오탐/미탐을 눈으로 확인하기 위한 테스트."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import mask_text

CASES = [
    ("유선전화", "지점 전화 031-123-4567 로 문의 바랍니다"),
    ("대표번호", "대표번호 1588-1234 로 전화 주세요"),
    ("주민번호 공백형", "주민번호 850512 1234567 입니다"),
    ("주민번호 무구분", "주민번호 8505121234567 입니다"),
    ("이름 두 명", "고객명: 김민수, 담당자 이름: 이영희 가 응대함"),
    ("날짜/금액 무손상 기대", "2026-11-04에 1,000만원을 이체함"),
    ("휴대폰 공백 변형", "연락처 010 2345 6789 로 문자 발송"),
    ("계좌 공백형", "계좌 123 456 789012 로 입금 확인"),
]
for label, s in CASES:
    masked, found = mask_text(s)
    print(f"[{label}] found={[(k) for k, v in found]}")
    print(f"  in : {s}")
    print(f"  out: {masked}")
print("EDGE_DONE")

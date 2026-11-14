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
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
OLLAMA = "http://localhost:11434"
MODEL = os.environ.get("LOCAL_AGENT_MODEL", "exaone3.5:7.8b")
# 마지막 사용 후 이 시간이 지나면 모델이 메모리(VRAM)에서 내려간다. 상시 점유 방지.
KEEP_ALIVE = os.environ.get("LOCAL_AGENT_KEEP_ALIVE", "2m")
CHAT_OPTIONS = {"num_ctx": 4096}
if os.environ.get("LOCAL_AGENT_NUM_GPU"):
    CHAT_OPTIONS["num_gpu"] = int(os.environ["LOCAL_AGENT_NUM_GPU"])

# ---------------- 민감정보 마스킹 ----------------
PATTERNS = [
    # 구분자는 하이픈·공백·없음 모두 허용 (엣지 실측에서 하이픈 전제 패턴의 미탐 확인, 2026-11-04)
    # 경계는 \b가 아니라 숫자 부정 룩어라운드: \b는 한글 조사와 숫자 사이에서 성립하지 않아
    # "010-5555-6666으로"처럼 조사가 붙은 번호를 놓친다 (PC 실기에서 확인, 2026-11-07).
    ("주민등록번호", re.compile(r"(?<!\d)\d{6}[-\s]?[1-4]\d{6}(?!\d)")),
    ("전화번호", re.compile(r"(?<!\d)0\d{1,2}[-\s]\d{3,4}[-\s]\d{4}(?!\d)")),
    ("전화번호", re.compile(r"(?<!\d)1[5-8]\d{2}[-\s]\d{4}(?!\d)")),
    ("카드번호", re.compile(r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)")),
    ("계좌번호", re.compile(r"(?<!\d)\d{3,6}[-\s]\d{2,6}[-\s]\d{4,6}(?:[-\s]\d{1,3})?(?!\d)")),
    ("이메일", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
]
LABEL_NAME = re.compile(r"(성명|이름|고객명|예금주)\s*[:：]\s*([가-힣]{2,4})")
# 호칭 앞 이름: "박지민고객", "김민수 님"처럼 라벨 없이 호칭에 붙은 이름 (2026-11-14 실사용 테스트에서 누락 확인)
HONOR_NAME = re.compile(r"([가-힣]{2,4})(\s*)(?=고객님|고객|님|씨)")
NAME_EXCLUDE = {"신규", "우수", "방문", "일반", "단체", "법인", "개인", "잠재", "기존", "해당",
                "모든", "단골", "장기", "유치", "사장", "선생", "대표", "고객", "회장", "원장",
                "교수", "박사", "팀장", "과장", "부장", "차장", "대리", "주임", "선임", "책임", "담당"}


def _partial(label, hit):
    """일부 표시: 실무 관행대로 식별에 필요한 최소만 남긴다."""
    d = re.sub(r"\D", "", hit)
    if label == "주민등록번호" and len(d) >= 6:
        return d[:6] + "-*******"  # 생년월일만 (은행 앱 관행)
    if label == "이메일":
        local, _, domain = hit.partition("@")
        if domain:
            return local[:2] + "***@" + domain
    if label in ("전화번호", "카드번호", "계좌번호") and len(d) >= 4:
        return f"[{label} 마스킹 ··{d[-4:]}]"  # 끝 4자리만
    return f"[{label} 마스킹]"


def mask_text(text, level="full"):
    """민감 패턴을 찾아 마스킹한다. (마스킹된 텍스트, 발견 목록) 반환.

    level="full"은 전부 가림, "partial"은 끝 4자리·생년월일처럼 일부만 남긴다.
    """
    found = []
    out = text
    for m in LABEL_NAME.finditer(out):
        found.append(("성명", m.group(2)))
    for m in LABEL_NAME.finditer(text):
        name = m.group(2)
        out = out.replace(name, name[0] + "*" * (len(name) - 1))

    def _honor(m):
        name = m.group(1)
        if name in NAME_EXCLUDE:
            return m.group(0)
        found.append(("성명", name))
        return name[0] + "*" * (len(name) - 1) + m.group(2)
    out = HONOR_NAME.sub(_honor, out)
    for label, pat in PATTERNS:
        for hit in pat.findall(out):
            found.append((label, hit))
        if level == "partial":
            out = pat.sub(lambda m, lb=label: _partial(lb, m.group(0)), out)
        else:
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


DOC_EXTS = (".txt", ".md", ".docx", ".pptx", ".pdf")


def _split_paras(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def extract_docx(path):
    """docx = zip + XML. 단락(w:p)마다 줄을 뽑는다. 표준 라이브러리만으로 처리."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paras = []
    for p in root.iter(ns + "p"):
        t = "".join(node.text or "" for node in p.iter(ns + "t"))
        if t.strip():
            paras.append(t.strip())
    return "\n\n".join(paras)


def extract_pptx(path):
    """pptx = zip + 슬라이드 XML. 슬라이드를 단락 하나로, 안의 줄은 줄바꿈으로 잇는다."""
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist()
                 if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        names.sort(key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
        slides = []
        for n in names:
            root = ET.fromstring(z.read(n))
            lines = []
            for p in root.iter(ns + "p"):
                t = "".join(node.text or "" for node in p.iter(ns + "t"))
                if t.strip():
                    lines.append(t.strip())
            if lines:
                slides.append("\n".join(lines))
    return "\n\n".join(slides)


def _pdf_objects(data):
    """간접 객체 번호 → (사전 본문, 풀린 스트림). 객체 스트림(ObjStm) 속 객체는 대상 외."""
    objs = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", data):
        end = data.find(b"endobj", m.end())
        if end < 0:
            continue
        body = data[m.end():end]
        stream = None
        sm = re.search(rb"stream\r?\n", body)
        if sm:
            send = body.find(b"endstream", sm.end())
            if send >= 0:
                raw = body[sm.end():send].rstrip(b"\r\n")
                try:
                    stream = zlib.decompress(raw)
                except zlib.error:
                    stream = raw
            body = body[:sm.start()]
        objs[int(m.group(1))] = (body, stream)
    return objs


def _cmap_parse(stream):
    """ToUnicode CMap 하나를 {코드: 글자}와 코드 바이트 수로 푼다."""
    t = stream.decode("latin-1")
    size = 2
    m = re.search(r"begincodespacerange(.*?)endcodespacerange", t, re.S)
    if m:
        first = re.search(r"<([0-9A-Fa-f]+)>", m.group(1))
        if first:
            size = max(1, len(first.group(1)) // 2)
    cmap = {}

    def uni(hexs):
        try:
            return bytes.fromhex(hexs).decode("utf-16-be", "replace")
        except ValueError:
            return ""

    for sec in re.findall(r"beginbfchar(.*?)endbfchar", t, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", sec):
            cmap[int(src, 16)] = uni(dst)
    for sec in re.findall(r"beginbfrange(.*?)endbfrange", t, re.S):
        for lo, hi, dst in re.findall(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", sec):
            lo_i, hi_i, d = int(lo, 16), int(hi, 16), int(dst, 16)
            for code in range(lo_i, min(hi_i, lo_i + 1024) + 1):
                cmap.setdefault(code, chr(d + code - lo_i))
        for lo, hi, arr in re.findall(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[([^\]]*)\]", sec):
            lo_i = int(lo, 16)
            for j, dst in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", arr)):
                cmap[lo_i + j] = uni(dst)
    return cmap, size


def _pdf_decode(raw, cmap, size):
    chars = []
    if size == 2:
        units = [(raw[i] << 8) | raw[i + 1] for i in range(0, len(raw) - 1, 2)]
    else:
        units = list(raw)
    for u in units:
        if u in cmap:
            chars.append(cmap[u])
        elif u < 128:
            chars.append(chr(u))
    return "".join(chars)


def _pdf_lines_from_objects(objs):
    """폰트별 ToUnicode를 연결해 페이지 내용 스트림을 글자로 복원한다."""
    fonts = {}
    for num, (body, stream) in objs.items():
        m = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", body)
        if m and int(m.group(1)) in objs and objs[int(m.group(1))][1]:
            fonts[num] = _cmap_parse(objs[int(m.group(1))][1])
    if not fonts:
        return []
    name2font = {}
    for num, (body, stream) in objs.items():
        for fm in re.finditer(rb"/Font\s*<<(.*?)>>", body, re.S):
            for name, ref in re.findall(rb"/(\w+)\s+(\d+)\s+\d+\s+R", fm.group(1)):
                if int(ref) in fonts:
                    name2font[name.decode("latin-1")] = fonts[int(ref)]
    token = re.compile(r"/(\w+)\s+([\d.]+)\s+Tf|<([0-9A-Fa-f\s]+)>|"
                       r"\(((?:\\.|[^\\,)])*)\)|([-\d.]+)\s+([-\d.]+)\s+(?:Td|TD)|\bT\*\b")
    lines = []
    for num, (body, stream) in objs.items():
        if b"/Type" not in body or b"/Page" not in body or b"/Pages" in body:
            continue
        contents = []
        cm = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R", body)
        if cm and int(cm.group(1)) in objs:
            contents = [objs[int(cm.group(1))][1]]
        else:
            am = re.search(rb"/Contents\s*\[([^\]]*)\]", body)
            if am:
                contents = [objs[int(r)][1] for r in
                            re.findall(rb"(\d+)\s+\d+\s+R", am.group(1))
                            if int(r) in objs]
        for content in contents:
            if not content or b"BT" not in content:
                continue
            text = content.decode("latin-1")
            cur_font = (None, 2)
            font_pt = 0.0
            for block in re.findall(r"BT(.*?)ET", text, re.S):
                cur = ""
                space_next = False
                for m in token.finditer(block):
                    if m.group(1):
                        cur_font = name2font.get(m.group(1), cur_font)
                        try:
                            font_pt = float(m.group(2))
                        except ValueError:
                            pass
                    elif m.group(5) is not None:
                        tx, ty = float(m.group(5)), float(m.group(6))
                        if ty != 0:
                            if cur.strip():
                                lines.append(cur.strip())
                            cur = ""
                        elif font_pt and 0 < tx < 0.5 * font_pt:
                            space_next = True  # 글자 폭보다 좁은 이동 = 단어 사이 띄움
                    elif m.group(0) == "T*":
                        if cur.strip():
                            lines.append(cur.strip())
                        cur = ""
                    elif m.group(3) is not None or m.group(4) is not None:
                        if space_next and cur and not cur.endswith(" "):
                            cur += " "
                        space_next = False
                        cmap, csize = cur_font[0] or {}, cur_font[1]
                        if m.group(3) is not None:
                            try:
                                cur += _pdf_decode(bytes.fromhex(
                                    re.sub(r"\s+", "", m.group(3))), cmap, csize)
                            except ValueError:
                                pass
                        else:
                            raw = m.group(4).encode("latin-1").decode(
                                "unicode_escape").encode("latin-1")
                            cur += _pdf_decode(raw, cmap, csize)
                if cur.strip():
                    lines.append(cur.strip())
    return lines


def extract_pdf(path):
    """텍스트가 들어 있는 PDF에서 글자를 뽑는다 (표준 라이브러리만으로).
    폰트별 ToUnicode CMap을 연결하는 방식이라 Word·브라우저가 만든 PDF의 한글도 복원한다.
    스캔 이미지 PDF는 글자가 없어 빈 결과가 된다."""
    with open(path, "rb") as f:
        data = f.read()
    lines = _pdf_lines_from_objects(_pdf_objects(data))
    # 너무 짧은 줄은 앞줄에 이어 붙인다 (PDF 줄바꿈이 문장 중간에서 끊기는 일이 많아서)
    merged = []
    for ln in lines:
        if merged and len(ln) < 12 and not merged[-1].endswith((".", "다", "요", ":", ")", "음")):
            merged[-1] += " " + ln
        else:
            merged.append(ln)
    return "\n\n".join(merged)


def read_doc_text(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            return extract_docx(path)
        if ext == ".pptx":
            return extract_pptx(path)
        if ext == ".pdf":
            return extract_pdf(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def load_chunks():
    chunks = []
    if not os.path.isdir(DOCS_DIR):
        return chunks
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.lower().endswith(DOC_EXTS):
            continue
        text = read_doc_text(os.path.join(DOCS_DIR, name))
        for i, p in enumerate(_split_paras(text)):
            chunks.append({"source": name, "idx": i, "text": p, "vec": ngrams(p)})
    return chunks


RETRIEVE_THRESHOLD = 0.10  # 실측 근거: 문서 내 질문 최소 0.243 / 문서 밖 질문 최대 0.058 (2026-11-04)


def retrieve(chunks, query, topk=3):
    qv = ngrams(query)
    scored = [(cosine(qv, c["vec"]), c) for c in chunks]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for s, c in scored if s > RETRIEVE_THRESHOLD][:topk]


# ---------------- 의미 검색 (임베딩, 선택 기능) ----------------
EMBED_MODEL = os.environ.get("LOCAL_AGENT_EMBED_MODEL", "bge-m3")
# 실측 근거 (PC, bge-m3, 2026-11-07): 문서 내 질문 최소 0.533 / 문서 밖 질문 최대 0.419.
# 그 사이에서 정밀도 쪽으로 잡았다 (문서 밖 질문을 근거로 오인하면 환각 방지 장치가 무너진다).
EMBED_THRESHOLD = 0.48


def embed(texts):
    """Ollama 임베딩. 모델이 없거나 실패하면 None (호출부가 단어 검색으로 폴백)."""
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("embeddings")
    except Exception:
        return None


def vec_cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------- Ollama 호출 (localhost 전용) ----------------
def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False,
                       "keep_alive": KEEP_ALIVE,
                       "options": CHAT_OPTIONS}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))["message"]["content"]
    except urllib.error.URLError:
        raise RuntimeError("Ollama 로컬 서버에 연결할 수 없습니다. 서버 실행 여부를 확인하세요.") from None


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
        counts = Counter(k for k, _ in found)
        print("[탐지된 민감정보]", ", ".join(f"{k} {n}건" for k, n in counts.items()))
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
                    try:
                        summarize_memo("\n".join(lines))
                    except RuntimeError as e:
                        print(e)
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
        try:
            ans, sources = answer(chunks, masked_q)
        except RuntimeError as e:
            print(e)
            continue
        print("\n답변:", ans)
        if sources:
            print("근거:", ", ".join(sorted(set(sources))))


if __name__ == "__main__":
    main()

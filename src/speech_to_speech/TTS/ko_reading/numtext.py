"""숫자 찾기(①)와 한글로 바꾸기(③) — 2차 실험 "값은 코드, 판단은 LLM".

AJOU-VIS/ajou-vis research/normalize/numtext.py (ai-research/tts-accuracy 커밋 e0dcab7)를 그대로 복사했다.
고칠 때는 원본을 먼저 고치고 다시 복사한다.

1단계(spoken.py)에서 작은 모델에게 문장 전체를 다시 쓰게 하니 숫자 값 자체를 잘못 옮겼다 (4B 22%, 1.7B 43%).
그래서 숫자 값을 한글로 바꾸는 일은 코드가 하고, 모델은 읽는 방식만 고른다.

- 형식이 정해진 것은 여기서 바로 바꾼다: 소수점, 전화번호, 분수, 퍼센트, '제N', 하이픈 번호(11-1번), 영어 약어.
- 정수 + 단위('9번', '3명', '11시')와 단위 없는 정수는 읽는 방식(한자어/고유어/한 자리씩)을 골라야 하는 칸으로 남긴다.
  고르는 쪽은 rules.py(규칙) 또는 judge.py(LLM)이고, apply()가 고른 방식대로 문장을 만든다.

규칙과 약어 사전은 일반적인 한국어 읽기법으로만 썼다 (평가 문장의 정답에 맞춰 고치지 않음).
예외: 11/10 음성 서버에서 들어 보고 고친 것 (버스 하이픈 번호, 빗금 분수, 앞에 0이 붙은 시각, 영어 단어 사전).
이 중 AIMS2(에임스 투)는 평가 문장 p36과 겹친다 — 평가 결과를 본 뒤 고친 것으로 밝힌다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SINO, NATIVE, DIGITS = "한자어", "고유어", "한 자리씩"
TYPES = (SINO, NATIVE, DIGITS)

_SINO_DIGIT = "영일이삼사오육칠팔구"
_NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
_LETTERS = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치", "I": "아이",
            "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알",
            "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "제트"}
# 단어처럼 읽는 약어 (널리 쓰는 시험·기관 이름). 없는 약어는 글자 이름대로 읽는다
_WORDS = {"TOEFL": "토플", "TOEIC": "토익", "TEPS": "텝스", "TOPIK": "토픽", "IELTS": "아이엘츠", "OPIC": "오픽",
          "NASA": "나사", "UNESCO": "유네스코", "KAIST": "카이스트", "POSTECH": "포스텍",
          # 서버 자료(corpus)에 대문자로 나오는 영어 단어 (11/10 음성 서버: CAFÉ ING를 "씨에이에프É"로 읽음)
          "CAFE": "카페", "CAFÉ": "카페", "ACE": "에이스", "VISA": "비자", "APP": "앱", "NEW": "뉴", "GROUP": "그룹",
          "EXIT": "엑시트", "BLACK": "블랙", "EXERCISE": "엑서사이즈", "AIMS": "에임스"}
# 숫자까지 한 이름으로 읽는 것 (숫자를 영어로 읽음)
_NAMES = {"AIMS2": "에임스 투"}
_LATIN_LETTERS = "[A-Za-zÀ-ÖØ-öø-ÿ]"   # CAFÉ처럼 악센트 붙은 글자도 한 단어로 본다

# 정수 뒤에 붙는 단위 (긴 것부터 맞춘다)
COUNTERS = sorted(["번째", "개월", "학번", "학점", "학기", "학년", "시간", "교시", "과목", "가지", "군데", "페이지", "마리",
                   "번", "명", "개", "살", "달", "권", "잔", "곳", "층", "호", "년", "월", "일", "회", "급", "원", "인",
                   "조", "배", "주", "차", "점", "석", "분", "초", "시", "장", "쪽", "대", "위", "등"], key=len, reverse=True)


def sino(n: int) -> str:
    """한자어 수: 7001 → 칠천일, 12 → 십이, 10000 → 만."""
    if n == 0:
        return "영"
    out, unit = [], 0
    while n:
        n, chunk = divmod(n, 10000)
        if chunk:
            part = ""
            for place, name in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
                d = chunk // place % 10
                if d:
                    part += ("" if d == 1 and name else _SINO_DIGIT[d]) + name
            out.append(part + ["", "만", "억", "조"][unit])
        unit += 1
    text = "".join(reversed(out))
    return text[1:] if text.startswith("일만") else text


def native(n: int) -> str:
    """고유어 수 (단위 앞 꼴): 1 → 한, 20 → 스무, 21 → 스물한. 백 이상은 한자어로 읽는다."""
    if not 0 < n < 100:
        return sino(n)
    tens, ones = divmod(n, 10)
    return "스무" if (tens, ones) == (2, 0) else _NATIVE_TENS[tens] + _NATIVE_ONES[ones]


def digit_by_digit(digits: str, zero: str = "공") -> str:
    """한 자리씩: 2003 → 이공공삼."""
    return "".join(zero if d == "0" else _SINO_DIGIT[int(d)] for d in digits if d.isdigit())


def decimal(text: str) -> str:
    """소수: 3.75 → 삼 점 칠오 (소수점 아래는 한 자리씩, 0은 영)."""
    whole, frac = text.split(".")
    return f"{sino(int(whole))} 점 {digit_by_digit(frac, zero='영')}"


def latin(token: str) -> str:
    """영어 약어: 사전에 있으면 단어로(TOEFL → 토플), 없으면 글자 이름대로(ROTC → 알오티씨). 붙은 숫자는 한자어."""
    name = _NAMES.get(token.upper().replace("-", ""))
    if name:
        return name
    letters, number = re.fullmatch(rf"({_LATIN_LETTERS}+)-?(\d*)", token).groups()
    word = _WORDS.get(letters.upper())
    if word is None:
        if not all(c in _LETTERS for c in letters.upper()):
            return token  # É처럼 영문 글자 이름이 없는 글자가 섞인 모르는 단어는 TTS에 그대로 넘긴다
        word = "".join(_LETTERS[c] for c in letters.upper())
    return f"{word} {sino(int(number))}" if number else word


@dataclass
class Span:
    start: int
    end: int
    text: str
    kind: str                  # "int"이면 읽는 방식을 골라야 하는 칸, 아니면 fixed로 바로 바꾼 것
    number: int = 0
    digits: str = ""
    counter: str = ""
    fixed: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def judgable(self) -> bool:
        return self.kind == "int"

    def reading(self, how: str = SINO) -> str:
        if not self.judgable:
            return self.fixed
        number = {SINO: sino, NATIVE: native}.get(how)
        said = number(self.number) if number else digit_by_digit(self.digits)
        if how == NATIVE and self.counter == "번째" and self.number == 1:
            said = "첫"  # 1번째 → 첫 번째
        return f"{said} {self.counter}" if self.counter else said

    def options(self) -> list[tuple[str, str]]:
        """고를 수 있는 읽기 (같은 글이 되는 방식은 하나로): 9번 → [(한자어, 구 번), (고유어, 아홉 번)]."""
        seen, out = set(), []
        for how in TYPES:
            text = self.reading(how)
            if text not in seen:
                seen.add(text)
                out.append((how, text))
        return out


_NUM = r"\d{1,3}(?:,\d{3})+|\d+"
_PATTERNS = [  # 앞에 있는 것이 먼저 (겹치면 앞의 것)
    ("phone", re.compile(r"(?<![\d.\-])(?:0\d{1,2}-\d{3,4}-\d{4}|\d{3,4}-\d{4})(?![\d\-])")),
    ("latin", re.compile(rf"{_LATIN_LETTERS}+(?:-?\d+)?")),
    ("ordinal", re.compile(r"제(\d+)(?=[가-힣])")),
    ("fraction", re.compile(r"(\d+)분의\s*(\d+)")),
    ("slash", re.compile(r"(?<![\d/])(\d+)/(\d+)(?![\d/])")),       # 1/4 — 분수 문맥일 때만 (_CONTEXT)
    ("decimal", re.compile(r"(\d+\.\d+)(?:\s*(%))?")),
    ("hyphen", re.compile(r"(?<![\d\-])(\d+)-(\d+)(?![\d\-])")),    # 11-1번 — 버스 번호 문맥일 때만 (_CONTEXT)
    ("percent", re.compile(rf"({_NUM})\s*%")),
    ("int", re.compile(rf"({_NUM})(만)?(?:\s?({'|'.join(COUNTERS)}))?")),
]


def _fixed(kind: str, m: re.Match) -> str:
    if kind == "phone":
        return " ".join(digit_by_digit(g) for g in m.group(0).split("-"))
    if kind == "latin":
        return latin(m.group(0))
    if kind == "ordinal":
        return "제" + sino(int(m.group(1)))
    if kind == "fraction":
        return f"{sino(int(m.group(1)))}분의 {sino(int(m.group(2)))}"
    if kind == "slash":
        return f"{sino(int(m.group(2)))}분의 {sino(int(m.group(1)))}"
    if kind == "decimal":
        return decimal(m.group(1)) + (" 퍼센트" if m.group(2) else "")
    if kind == "hyphen":
        return f"{sino(int(m.group(1)))} 다시 {sino(int(m.group(2)))}"
    if kind == "percent":
        return sino(int(m.group(1).replace(",", ""))) + " 퍼센트"
    raise ValueError(kind)


def _same_sentence_after(text: str, end: int, width: int) -> str:
    return re.split(r"[.!?\n]", text[end:end + width], maxsplit=1)[0]


def _bus_context(text: str, m: re.Match) -> bool:
    """하이픈 번호는 버스 번호일 때만 '다시'로 읽는다: 같은 문장에 '버스'·'노선'이 있거나("시내버스는 11-1, 46-1, 720-2")
    뒤에 '번'이 이어질 때("11-1, 46-1번"). 날짜·범위(3-4명) 같은 하이픈은 그대로 둔다.
    11/10 음성 서버: 버스 번호를 길게 늘어놓은 답에서 46-1이 '번'과 멀어 '다시'를 못 읽었다 → 문장 전체를 본다."""
    sentence_start = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
    before = text[sentence_start:m.start()]
    after = _same_sentence_after(text, m.end(), 200)
    return any(w in before or w in after for w in ("버스", "노선")) or "번" in after[:30]


_FRACTION_BEFORE = ("일수", "등록금", "수업료", "금액", "기간")
_FRACTION_AFTER = ("선", "이상", "이하", "미만", "초과", "정도", "만큼")


def _fraction_context(text: str, m: re.Match) -> bool:
    """빗금 숫자는 분수 문맥일 때만 'N분의 M'으로 읽는다 ("수업일수 1/4선까지"). 3/4(3월 4일) 같은 날짜와 헷갈리므로
    앞에 일수·등록금 같은 말이 있거나 뒤에 선·이상 같은 말이 올 때만, 그리고 분자가 분모보다 작을 때만."""
    numerator, denominator = int(m.group(1)), int(m.group(2))
    if not 0 < numerator < denominator:
        return False
    before = text[max(0, m.start() - 12):m.start()]
    after = text[m.end():].lstrip()
    return any(w in before for w in _FRACTION_BEFORE) or after.startswith(_FRACTION_AFTER)


_CONTEXT = {"hyphen": _bus_context, "slash": _fraction_context}


def find_spans(text: str) -> list[Span]:
    """문장에서 숫자 표현을 찾는다. 겹치면 _PATTERNS에서 앞에 있는 종류가 이긴다."""
    taken: list[tuple[int, int]] = []
    spans: list[Span] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < e and s < end for s, e in taken) or start == end:
                continue
            if kind in _CONTEXT and not _CONTEXT[kind](text, m):
                continue
            if kind == "int":
                written = m.group(1).replace(",", "")
                number = int(written) * (10000 if m.group(2) else 1)  # 2만 원 → 20000
                digits = str(number) if m.group(2) else written      # 04학번 → 한 자리씩은 공사 (앞의 0 유지)
                span = Span(start, end, m.group(0), "int", number=number, digits=digits, counter=m.group(3) or "")
            else:
                span = Span(start, end, m.group(0), kind, fixed=_fixed(kind, m))
            taken.append((start, end))
            spans.append(span)
    return sorted(spans, key=lambda s: s.start)


def apply(text: str, spans: list[Span], choices: list[str | None]) -> str:
    """고른 읽는 방식(choices, 칸마다 하나; fixed 칸은 None)대로 숫자를 한글로 바꾼 문장."""
    out, pos = [], 0
    for span, how in zip(spans, choices):
        out += [text[pos:span.start], span.reading(how or SINO)]
        pos = span.end
    return "".join(out + [text[pos:]])

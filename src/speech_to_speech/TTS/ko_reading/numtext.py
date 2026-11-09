"""숫자 찾기(①)와 한글로 바꾸기(③) — 2차 실험 "값은 코드, 판단은 LLM".

AJOU-VIS/ajou-vis research/normalize/numtext.py (ai-research/tts-accuracy 커밋 5516762)를 그대로 복사했다.
고칠 때는 원본을 먼저 고치고 다시 복사한다.

1단계(spoken.py)에서 작은 모델에게 문장 전체를 다시 쓰게 하니 숫자 값 자체를 잘못 옮겼다 (4B 22%, 1.7B 43%).
그래서 숫자 값을 한글로 바꾸는 일은 코드가 하고, 모델은 읽는 방식만 고른다.

- 형식이 정해진 것은 여기서 바로 바꾼다: 소수점, 전화번호, 분수, 퍼센트, '제N', 하이픈 번호(11-1번), 영어 약어.
- 정수 + 단위('9번', '3명', '11시')와 단위 없는 정수는 읽는 방식(한자어/고유어/한 자리씩)을 골라야 하는 칸으로 남긴다.
  고르는 쪽은 rules.py(규칙) 또는 judge.py(LLM)이고, apply()가 고른 방식대로 문장을 만든다.

규칙과 약어 사전은 일반적인 한국어 읽기법으로만 썼다 (평가 문장의 정답에 맞춰 고치지 않음).
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
          "NASA": "나사", "UNESCO": "유네스코", "KAIST": "카이스트", "POSTECH": "포스텍"}

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
    letters, number = re.fullmatch(r"([A-Za-z]+)-?(\d*)", token).groups()
    word = _WORDS.get(letters.upper()) or "".join(_LETTERS[c] for c in letters.upper())
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
    ("latin", re.compile(r"[A-Za-z]+(?:-?\d+)?")),
    ("ordinal", re.compile(r"제(\d+)(?=[가-힣])")),
    ("fraction", re.compile(r"(\d+)분의\s*(\d+)")),
    ("decimal", re.compile(r"(\d+\.\d+)(?:\s*(%))?")),
    ("hyphen", re.compile(r"(\d+)-(\d+)(?=\s*번)")),
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
    if kind == "decimal":
        return decimal(m.group(1)) + (" 퍼센트" if m.group(2) else "")
    if kind == "hyphen":
        return f"{sino(int(m.group(1)))} 다시 {sino(int(m.group(2)))}"
    if kind == "percent":
        return sino(int(m.group(1).replace(",", ""))) + " 퍼센트"
    raise ValueError(kind)


def find_spans(text: str) -> list[Span]:
    """문장에서 숫자 표현을 찾는다. 겹치면 _PATTERNS에서 앞에 있는 종류가 이긴다."""
    taken: list[tuple[int, int]] = []
    spans: list[Span] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < e and s < end for s, e in taken) or start == end:
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

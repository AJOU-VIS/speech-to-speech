"""규칙으로 읽는 방식 고르기 (2차 실험 ⓑ), 그리고 규칙 + LLM 섞기 (ⓓ).

AJOU-VIS/ajou-vis research/normalize/rules.py (ai-research/tts-accuracy 커밋 e0dcab7)를 복사했다.
바꾼 곳은 import 한 줄(패키지 안 상대 경로)뿐이다. 고칠 때는 원본을 먼저 고치고 다시 복사한다.

ⓑ 규칙 (일반적인 한국어 읽기법으로만 썼다):
  - 고유어로 세는 단위(명, 개, 살, 시간, 달, 마리, 권, 잔, 곳, 가지, 과목, 군데, 번째, 배, 학기)는 고유어 (백 이상은 한자어)
  - 시각의 '시'는 열둘까지 고유어, 열셋부터 한자어
  - '번'은 앞뒤 단어로 정한다: 출구·버스·창구처럼 이름 붙은 것 옆이면 번호(한자어), 아니면 횟수(고유어)
  - 그 밖의 단위와 단위 없는 수는 한자어
ⓓ 섞기: 단위만 보면 정해지는 것은 규칙으로, 문맥이 필요한 '번'·'학기'만 LLM(judge.py 결과)이 고른 대로.
4단계 (규칙 + '번' 모델): '번' 칸 중 고유어 보기가 있는 칸(100 미만)만 학습한 '번' 모델(judge.py --adapter 결과)이 고르고,
  나머지는 모두 규칙. 100 이상은 학습 데이터에 없던 선택(한자어/한 자리씩)이라 규칙에 맡긴다 (11/9 평가 결과를 보고 정함).

사용:
  python research/normalize/rules.py --items research/eval/eval_v1.jsonl --out rules.jsonl
  python research/normalize/rules.py --items research/eval/eval_v1.jsonl --llm judge_qwen3-1.7b.jsonl --out mix_qwen3-1.7b.jsonl
  python research/normalize/rules.py --items research/eval/eval_v1.jsonl --bun judge_bun.jsonl --out bun.jsonl
출력은 spoken.py와 같은 형식 + decisions (score_spoken.py·score_decisions.py로 채점)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .numtext import DIGITS, NATIVE, SINO, Span, apply, find_spans

# 앞에 0을 붙여 쓰는 시각·날짜 단위 (11/10 음성 서버: "08시"를 "공팔 시"로 읽음)
TIME_COUNTERS = {"시", "분", "초", "월", "일"}
NATIVE_COUNTERS = {"명", "개", "살", "시간", "달", "마리", "권", "잔", "곳", "가지", "과목", "군데", "번째", "배", "학기"}
# 이 말과 붙어 있는 '번'은 번호 (이름처럼 붙인 수)
NUMBERED = ("출구", "버스", "승강장", "창구", "메뉴", "항목", "문항", "게이트", "좌석", "사물함", "레인", "트랙", "채널",
            "정류장", "출입구", "노선", "호선", "라인", "칸", "문제", "카드", "버튼", "번호", "국도")
# ⓓ에서 LLM에 맡기는 단위: 같은 단위가 문맥에 따라 다르게 읽히는 것 (번: 번호/횟수, 학기: 이름/개수, 분: 시간/사람)
# '분'은 규칙만 결과(11/9)에서 '교수님 2분'을 보고 더했다
LLM_COUNTERS = {"번", "학기", "분"}


def rule_choice(text: str, span: Span) -> str:
    if len(span.digits) > 1 and span.digits.startswith("0") and span.counter not in TIME_COUNTERS:
        return DIGITS  # 0으로 시작하면 번호라서 한 자리씩 (04학번 → 공사 학번). 08시·05분은 시각이라 빼고 수로 읽는다
    if span.counter == "번":
        after = text[span.end:span.end + 6].lstrip()
        before = text[max(0, span.start - 12):span.start]
        return SINO if after.startswith(NUMBERED) or any(w in before for w in NUMBERED) else NATIVE
    if span.counter == "시":
        return NATIVE if span.number <= 12 else SINO
    return NATIVE if span.counter in NATIVE_COUNTERS else SINO


def use_bun_model(span: Span) -> bool:
    """4단계에서 '번' 모델이 고르는 칸: '번'이면서 고유어 보기가 있는 칸 (100 미만)."""
    return span.counter == "번" and any(how == NATIVE for how, _ in span.options())


def decide(text: str, llm: list[dict] | None = None, bun: list[dict] | None = None) -> tuple[str, list[dict], float]:
    """규칙(과 LLM 선택)으로 문장을 바꾸고, 숫자 칸마다 무엇을 골랐는지와 LLM에 쓴 시간을 남긴다.
    llm: judge.py 결과 (판정할 칸마다 하나, ⓓ). bun: '번' 모델 결과 (use_bun_model 칸마다 하나, 4단계)."""
    spans = find_spans(text)
    judged, from_bun = iter(llm or []), iter(bun or [])
    choices, decisions, seconds = [], [], 0.0
    for span in spans:
        if not span.judgable:
            choices.append(None)
            continue
        how, by = rule_choice(text, span), "규칙"
        if llm is not None:
            from_llm = next(judged)
            if span.counter in LLM_COUNTERS:  # 섞기에서는 이 칸만 LLM을 부른 것으로 친다
                how, by = from_llm["choice"], "LLM"
                seconds += from_llm["seconds"]
        elif bun is not None and use_bun_model(span):
            picked = next(from_bun)
            assert picked["span"] == span.text, (picked["span"], span.text)
            how, by = picked["choice"], "'번' 모델"
            seconds += picked.get("seconds", 0.0)
        choices.append(how)
        decisions.append({"span": span.text, "counter": span.counter, "options": span.options(), "choice": how,
                          "reading": span.reading(how), "by": by})
    return apply(text, spans, choices), decisions, seconds


def main():
    ap = argparse.ArgumentParser(description="규칙으로 읽는 방식 고르기 (ⓑ) / 규칙 + LLM 섞기 (ⓓ)")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llm", type=Path, help="judge.py 결과 — 주면 '번'·'학기'만 LLM 선택으로 바꾼다 (ⓓ)")
    ap.add_argument("--bun", type=Path, help="judge.py --adapter 결과 — 주면 100 미만 '번'만 '번' 모델 선택으로 (4단계)")
    args = ap.parse_args()
    if args.llm and args.bun:
        ap.error("--llm과 --bun은 같이 쓸 수 없어요")

    items = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    llm, bun, name = {}, {}, "규칙"
    if args.llm:
        rows = [json.loads(line) for line in args.llm.open(encoding="utf-8") if line.strip()]
        llm = {r["id"]: r for r in rows}
        name = f"섞기 (규칙 + {rows[0]['model']})"
    if args.bun:
        bun = {r["id"]: r for r in map(json.loads, args.bun.open(encoding="utf-8"))}
        name = "규칙 + '번' 모델 (4단계)"
    with args.out.open("w", encoding="utf-8") as f:
        for it in items:
            judged = llm[it["id"]]["decisions"] if args.llm else None
            from_bun = bun[it["id"]]["decisions"] if args.bun else None
            output, decisions, seconds = decide(it["text"], judged, from_bun)
            row = {"id": it["id"], "model": name, "input": it["text"], "output": output, "seconds": round(seconds, 4),
                   "new_tokens": 0, "skipped": not find_spans(it["text"]), "decisions": decisions}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(items)}문장 → {args.out}")


if __name__ == "__main__":
    main()

"""녹화한 LLM 글자 조각(latency_trace.jsonl의 llm_delta)으로 분할 정책을 비교한다.

질문: "첫 문장이 끝나길 기다리지 않고 더 일찍 TTS에 넘겼다면 첫 음성이 얼마나 빨라졌을까?"
TTS를 다시 돌리지 않고, 실제로 글자 조각이 도착한 시각에 정책을 적용해 첫 TTS 입력 시각을 계산한다.

정책
  L0  지금 방식(실제 기록값): 문장 분리기가 첫 문장을 확정할 때까지 기다림 (다음 문장의 시작이 보이거나 스트림이 끝날 때)
  L1  문장 끝(. ? !) 글자가 도착하면 바로 넘김 — 분리기의 "다음 문장 시작" 대기만 없앤 경우
  L2  L1 + 첫 쉼표·연결어미(…하면, …지만, …는데 등)에서도 끊음 (숫자·소수점·조항 번호 안은 피함, 최소 글자 수 이상)

가정과 한계 (결과를 말할 때 같이 밝힐 것)
  - TTS 시간(TTS 입력 → 첫 오디오)은 실제 기록값 그대로라고 가정한다. 더 짧은 조각이면 약간 더 빠를 수 있다.
  - 시뮬레이션일 뿐이다. 문장 중간에서 끊었을 때의 억양·숫자 읽기 품질은 재지 않았다.
  - 팀 설정은 문장 사이에 0.4초 무음을 넣는다. 문장 중간(쉼표·연결어미)에서 끊으면 이 무음을 줄여야 자연스럽다.
  - L2의 연결어미 규칙은 단순한 어미 목록이라 완벽하지 않다. 어떤 조각이 나왔는지 표에서 직접 확인한다.

사용:
  python scripts/latency_simulate.py latency_trace.jsonl
  python scripts/latency_simulate.py latency_trace.jsonl --min-chars 12 --png latency_policies.png --md latency_policies.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# 연결어미 목록. 명사 끝과 겹치는 짧은 어미(고, 서 등)는 일부러 뺐다.
CONNECTIVE = re.compile(r"[가-힣]+(?:하고|이고|있고|되고|해서|어서|아서|라서|지만|는데|니까|으며|하며|이며|으나|면)(?=\s)")
# '면'으로 끝나는 명사(…전면, 화면 등)는 연결어미가 아니다
NOT_CONNECTIVE = ("전면", "표면", "화면", "수면", "방면", "지면", "단면", "평면", "측면", "외면", "내면", "정면", "반면", "서면", "대면")


def is_digit(c: str) -> bool:
    return c.isdigit()


def find_cut(acc: str, min_chars: int, allow_clause: bool) -> int | None:
    """acc(지금까지 도착한 글자) 안에서 끊을 수 있는 가장 이른 위치(잘라낸 길이)를 돌려준다. 없으면 None."""
    best: int | None = None

    def consider(pos: int) -> None:
        nonlocal best
        if pos >= min_chars and (best is None or pos < best):
            best = pos

    # 문장 끝: 마침표·물음표·느낌표. 마침표 앞이 숫자면 소수점(1.75)일 수 있어 다음 글자가 공백이어야 문장 끝으로 본다.
    for m in re.finditer(r"[.?!]", acc):
        pos = m.end()
        prev = acc[m.start() - 1] if m.start() > 0 else ""
        if m.group() == "." and prev.isdigit():
            if pos < len(acc) and acc[pos].isspace():
                consider(pos)
            continue
        consider(pos)
    if allow_clause:
        # 쉼표: 앞뒤가 모두 숫자인 쉼표(3,000)는 제외. 다음 글자가 도착해야 판단한다.
        for m in re.finditer(r",", acc):
            i, pos = m.start(), m.end()
            if pos >= len(acc):
                continue
            if i > 0 and acc[i - 1].isdigit() and acc[pos].isdigit():
                continue
            consider(pos)
        # 연결어미: 뒤에 공백이 도착해야 어절이 끝난 것으로 본다.
        for m in CONNECTIVE.finditer(acc):
            word = m.group(0)
            if word.endswith(NOT_CONNECTIVE):
                continue
            consider(m.end())
    return best


def first_chunk_time(deltas: list[tuple[float, str]], min_chars: int, allow_clause: bool):
    """글자 조각이 도착하는 순서대로 정책을 적용해, 처음 끊을 수 있게 된 시각과 그 조각 내용을 돌려준다."""
    acc = ""
    for t, text in deltas:
        acc += text
        cut = find_cut(acc, min_chars, allow_clause)
        if cut is not None:
            return t, acc[:cut].strip()
    return None, None


def load(path: str):
    turns: dict[int, dict] = defaultdict(lambda: {"deltas": []})
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = turns[r["turn"]]
        if r["stage"] == "llm_delta":
            t["deltas"].append((r["t_s"], r["text"]))
        else:
            t[r["stage"]] = r["t_s"]
            if r["stage"] == "llm_first_chunk":
                t["l0_chars"] = r.get("chars")
                t["l0_text"] = r.get("text", "")
    return turns


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--min-chars", type=int, default=12, help="이보다 짧은 조각은 끊지 않는다 (기본 12자)")
    ap.add_argument("--png")
    ap.add_argument("--md")
    args = ap.parse_args()

    turns = load(args.trace)
    usable = {n: t for n, t in turns.items() if t["deltas"] and "llm_first_chunk" in t and "tts_first_audio" in t and "tts_input" in t}
    L = ["# 분할 정책 시뮬레이션 (L0 / L1 / L2)", "",
         f"- 글자 조각을 기록한 응답 {len(usable)}개 (턴 {len(turns)}개 중). 최소 조각 길이 {args.min_chars}자.",
         "- 모든 시각은 VAD가 말 끝을 확정한 시점을 0으로 한 초. TTS 시간(입력 → 첫 오디오)은 실제 기록값을 그대로 쓴다.", ""]
    if not usable:
        L.append("글자 조각(llm_delta)이 기록된 응답이 없다. 패치 v2로 다시 측정했는지 확인한다.")
        print("\n".join(L)); return

    rows = []
    for n, t in sorted(usable.items()):
        tts_s = t["tts_first_audio"] - t["tts_input"]
        l0 = t["llm_first_chunk"]
        # 분리기 대기: 첫 문장 글자가 다 도착한 시각 → 실제로 확정된 시각
        acc, complete_at = "", None
        for ts, text in t["deltas"]:
            acc += text
            if t.get("l0_chars") and len(acc) >= t["l0_chars"]:
                complete_at = ts
                break
        wait = None if complete_at is None else max(l0 - complete_at, 0.0)
        t1, c1 = first_chunk_time(t["deltas"], args.min_chars, allow_clause=False)
        t2, c2 = first_chunk_time(t["deltas"], args.min_chars, allow_clause=True)
        t1 = l0 if t1 is None else min(t1, l0)
        if t2 is None or t2 > l0:
            t2, c2 = l0, t.get("l0_text", "")
        total0 = t["tts_first_audio"]
        rows.append(dict(n=n, l0=l0, wait=wait, t1=t1, t2=t2, c2=c2, total0=total0,
                         total1=total0 - (l0 - t1), total2=total0 - (l0 - t2), l0_text=t.get("l0_text", "")))

    L += ["## 턴별", "", "| 턴 | L0 첫 조각 시각 | 분리기 대기 | L1 | L2 | L2 절감 | L2로 끊은 첫 조각 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']} | {f2(r['l0'])} | {f2(r['wait'])} | {f2(r['t1'])} | {f2(r['t2'])} | {f2(r['l0'] - r['t2'])} | {r['c2'][:40]} |")

    m0, m1, m2 = med([r["total0"] for r in rows]), med([r["total1"] for r in rows]), med([r["total2"] for r in rows])
    sav1 = med([r["l0"] - r["t1"] for r in rows]); sav2 = med([r["l0"] - r["t2"] for r in rows])
    improved = sum(r["l0"] - r["t2"] > 0.05 for r in rows)
    L += ["", "## 요약 (중앙값)", "",
          "| 정책 | 첫 음성까지 (말 끝 확정 기준) | L0 대비 절감 |", "|---|---|---|",
          f"| L0 지금 방식(실제) | {f2(m0)}초 | - |",
          f"| L1 문장 끝 즉시 | {f2(m1)}초 | {f2(sav1)}초 |",
          f"| L2 쉼표·연결어미 | {f2(m2)}초 | {f2(sav2)}초 |", "",
          f"- 절감 평균: L1 {f2(statistics.mean(r['l0'] - r['t1'] for r in rows))}초 · L2 {f2(statistics.mean(r['l0'] - r['t2'] for r in rows))}초 (중앙값은 절감이 없는 응답이 많으면 0에 가까워지므로 평균과 같이 본다)",
          f"- 분리기가 다음 문장을 기다린 시간(중앙값): {f2(med([r['wait'] for r in rows]))}초",
          f"- L2가 0.05초 넘게 앞당긴 응답: {improved}/{len(rows)}개",
          f"- 절감 범위(L2): {f2(min(r['l0'] - r['t2'] for r in rows))} ~ {f2(max(r['l0'] - r['t2'] for r in rows))}초", "",
          "※ 위 절감은 상한에 가깝다. TTS 시간을 그대로 가정했고, 조각이 짧아져 생기는 억양·무음 문제와 숫자 읽기 오류는 포함하지 않았다."]
    text = "\n".join(L)
    print(text)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8"); print(f"\n저장: {args.md}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            for fam in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
                if any(fam in f.name for f in font_manager.fontManager.ttflist):
                    plt.rcParams["font.family"] = fam; break
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(9, 3.2))
            names = ["L0 지금(실제)", "L1 문장 끝 즉시", "L2 쉼표·연결어미"]
            vals = [m0, m1, m2]
            bars = ax.barh(names[::-1], vals[::-1], color=["#14286E", "#6C8EDB", "#F5A623"])   # 아래부터 L2, L1, L0
            for b, v in zip(bars, vals[::-1]):
                ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}초", va="center", fontsize=11)
            ax.set_xlim(0, max(vals) * 1.18)
            ax.set_xlabel(f"말 끝 확정 → 첫 음성 (중앙값, {len(rows)}응답, 시뮬레이션)")
            ax.set_title("첫 조각을 더 일찍 TTS에 넘기면 (TTS 시간은 실제값 가정)", fontsize=11)
            fig.tight_layout(); fig.savefig(args.png, dpi=200)
            print(f"그림: {args.png}")
        except ImportError:
            print("matplotlib가 없어 그림은 건너뜀")


if __name__ == "__main__":
    main()

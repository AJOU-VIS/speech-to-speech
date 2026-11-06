"""latency_trace.jsonl → 턴별 표 + 중앙값 + 누적 막대 그림 (발표용 "2.4초가 어디서 오나")

사용:
  python scripts/latency_summary.py latency_trace.jsonl
  python scripts/latency_summary.py latency_trace.jsonl --png latency_breakdown.png --md latency_summary.md
기록 만드는 법과 각 단계의 뜻은 src/speech_to_speech/latency_trace.py 맨 위 설명을 본다.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["vad_emit", "stt_done", "llm_first_delta", "llm_first_chunk", "tts_input",
         "tts_first_audio", "client_audio_in", "playback_start"]

# (구간 이름, 시작 단계, 끝 단계)
SEGMENTS = [
    ("STT (말 끝 확정 → 전사 완료)", "vad_emit", "stt_done"),
    ("LLM 첫 글자까지 (재작성·검색·Claude 첫 토큰)", "stt_done", "llm_first_delta"),
    ("첫 문장 완성까지 (현재 방식 L0)", "llm_first_delta", "llm_first_chunk"),
    ("문장 → TTS 입력", "llm_first_chunk", "tts_input"),
    ("TTS 첫 오디오", "tts_input", "tts_first_audio"),
    ("전송·재생 시작", "tts_first_audio", "playback_start"),
]


def load(paths):
    turns: dict[tuple, dict] = defaultdict(dict)
    for k, path in enumerate(paths):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["stage"] == "llm_delta":
                continue          # 글자 조각 기록은 latency_simulate.py가 쓴다
            t = turns[(k, r["turn"])]
            t[r["stage"]] = r["t_s"]
            if r["stage"] == "vad_emit":
                t["_delay_s"] = r.get("delay_s", 0.0)
    return turns


def seg_value(t, a, b):
    if a in t and b in t:
        return t[b] - t[a]
    return None


def total(t):
    end = "playback_start" if "playback_start" in t else ("tts_first_audio" if "tts_first_audio" in t else None)
    return t.get(end) if end else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--png")
    ap.add_argument("--md")
    args = ap.parse_args()

    turns = load(args.trace)
    # 최소한 TTS 첫 오디오까지 간 턴만 센다 (중간에 끊긴 시도는 제외)
    done = {k: t for k, t in turns.items() if "tts_first_audio" in t}
    skipped = len(turns) - len(done)

    L = ["# 지연 분해 요약", "",
         f"- 기록한 턴 {len(turns)}개 중 TTS 첫 오디오까지 간 턴 {len(done)}개를 집계 (끊긴 시도 {skipped}개 제외)",
         "- 모든 값은 VAD가 말 끝을 확정한 시각(speech_stopped_at_s)을 0으로 한 초 단위", ""]

    # 턴별 구간표
    head = ["턴"] + [s[0].split(" (")[0] for s in SEGMENTS] + ["합계", "Smart Turn 등 지연(참고)"]
    L += ["## 턴별 (초)", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (k, n), t in sorted(done.items()):
        row = [str(n)] + [fmt(seg_value(t, a, b)) for _, a, b in SEGMENTS] + [fmt(total(t)), fmt(t.get("_delay_s"))]
        L.append("| " + " | ".join(row) + " |")

    meds = [(name, med([seg_value(t, a, b) for t in done.values()])) for name, a, b in SEGMENTS]
    tot_med = med([total(t) for t in done.values()])
    delay_med = med([t.get("_delay_s") for t in done.values()])
    L += ["", "## 중앙값 (초)", "", "| 구간 | 중앙값 | 비중 |", "|---|---|---|"]
    stack_sum = sum(v for _, v in meds if v is not None)
    for name, v in meds:
        L.append(f"| {name} | {fmt(v)} | {fmt(100 * v / stack_sum, 0) + '%' if v is not None and stack_sum else '-'} |")
    L.append(f"| **합계 (말 끝 확정 → 재생 시작)** | **{fmt(tot_med)}** | |")
    L += ["", f"- Smart Turn 등으로 일부러 늦춘 시간(processing_delay_s) 중앙값: {fmt(delay_med)}초 (위 구간에 이미 들어 있는지는 확인 필요)",
          "- 한계: 재생 시작은 소리 콜백이 첫 오디오를 넘긴 시각이며 스피커 하드웨어 지연은 포함하지 않는다.",
          "- 한계: 말이 끝난 실제 순간이 아니라 VAD가 확정한 시각이 0이다 (VAD 대기·침묵 판정 시간은 합계에 안 들어간다)."]

    text = "\n".join(L)
    print(text)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
        print(f"\n저장: {args.md}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for fam in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
                if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
                    plt.rcParams["font.family"] = fam
                    break
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(10, 2.8))
            left = 0.0
            colors = ["#6C8EDB", "#2F6FDE", "#F5A623", "#9AA5C0", "#14286E", "#5B6785"]
            for (name, v), c in zip(meds, colors):
                if v is None:
                    continue
                ax.barh(0, v, left=left, color=c, edgecolor="white")
                if v > 0.08:
                    ax.text(left + v / 2, 0, f"{v:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                left += v
            ax.set_yticks([])
            ax.set_xlabel("말 끝 확정 이후 경과 시간 (초, 중앙값)")
            ax.set_title(f"말 끝 → 재생 시작: {fmt(tot_med)}초")
            ax.legend([n.split(" (")[0] for n, v in meds if v is not None], loc="upper center",
                      bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, fontsize=9)
            fig.tight_layout()
            fig.savefig(args.png, dpi=180)
            print(f"그림: {args.png}")
        except ImportError:
            print("matplotlib가 없어 그림은 건너뜀 (pip install matplotlib)")


if __name__ == "__main__":
    main()

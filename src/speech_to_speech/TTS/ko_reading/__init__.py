"""TTS에 넣기 전에 한국어 문장의 숫자를 읽는 대로 바꾼다 ("9번 출구" → "구 번 출구"). 화면에 나가는 글은 그대로다.

모드 (--qwen3_tts_reading):
  off   — 바꾸지 않는다 (기본값, 원래 동작)
  rules — 규칙만 (AJOU-VIS 평가: TTS 숫자 오류율 35.7% → 12.2%, 지연 거의 0)
  bun   — 규칙 + '번' 판단 모델: 100 미만 '번'만 학습한 Qwen3-0.6B + LoRA가 고른다 (35.7% → 9.0%, '번' 칸당 약 80ms)
근거·코드 원본: AJOU-VIS/ajou-vis research/normalize (ai-research/tts-accuracy 커밋 5516762).

평가 문장과 다른 점 하나: 평가 문장의 영어는 거의 약어였지만 서버 답변에는 Google, iBT 같은 일반 영어 단어도 나온다.
그래서 대문자 약어(ROTC, CU, A0)와 사전에 있는 시험 이름(TOEFL 등)만 한글로 바꾸고, 소문자가 섞인 단어는 TTS에 그대로 넘긴다.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .numtext import _WORDS, find_spans
from .rules import decide, use_bun_model

MODES = ("off", "rules", "bun")
_LATIN = re.compile(r"[A-Za-z]+(?:-?\d+)?")
_MASK_BASE = 0xE000  # 유니코드 사용자 영역: 숫자·영문 패턴에 안 걸린다


def _keep_as_is(token: str) -> bool:
    letters = re.match(r"[A-Za-z]+", token).group(0)
    return not (letters.isupper() or letters.upper() in _WORDS)


@dataclass
class ReadingResult:
    text: str
    spans: int = 0
    by_model: int = 0
    ms: float = 0.0
    decisions: list[dict] = field(default_factory=list)


class KoreanReading:
    def __init__(self, mode: str = "off", bun_adapter: str | Path | None = None, device: str = "cuda"):
        mode = (mode or "off").strip().lower()
        if mode not in MODES:
            raise ValueError(f"qwen3_tts_reading은 {MODES} 중 하나여야 해요: {mode}")
        if mode == "bun" and not bun_adapter:
            raise ValueError("qwen3_tts_reading=bun에는 qwen3_tts_reading_bun_adapter(LoRA 폴더)가 필요해요")
        self.mode = mode
        self.bun = None
        if mode == "bun":
            from .bun_model import BunModel

            self.bun = BunModel(bun_adapter, device=device)
            self.bun.self_check()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def __call__(self, text: str) -> ReadingResult:
        if not self.enabled or not re.search(r"[\dA-Za-z]", text):
            return ReadingResult(text)
        started = time.perf_counter()
        kept: list[str] = []

        def mask(m: re.Match) -> str:
            if not _keep_as_is(m.group(0)):
                return m.group(0)
            kept.append(m.group(0))
            return chr(_MASK_BASE + len(kept) - 1)

        masked = _LATIN.sub(mask, text)
        picks = None
        if self.bun is not None:
            targets = [s for s in find_spans(masked) if s.judgable and use_bun_model(s)]
            picks = self.bun.choose(masked, targets)
        spoken, decisions, _ = decide(masked, bun=picks)
        for k, word in enumerate(kept):
            spoken = spoken.replace(chr(_MASK_BASE + k), word)
        return ReadingResult(spoken, spans=len(find_spans(masked)), by_model=len(picks or []),
                             ms=1000 * (time.perf_counter() - started), decisions=decisions)

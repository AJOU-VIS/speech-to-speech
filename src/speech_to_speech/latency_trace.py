"""지연 구간 기록 (AJOU-VIS).

켜는 법: 실행 전에 환경변수 S2S_LATENCY_TRACE=<기록 파일 경로>를 준다. 없으면 아무 일도 하지 않는다.
  예) S2S_LATENCY_TRACE=latency_trace.jsonl ./run_local_rag.sh

한 턴은 VAD가 "말이 끝났다"고 확정해 VADAudio를 내보낸 시각(speech_stopped_at_s)에서 시작한다.
각 단계는 턴마다 '처음 한 번'만 기록하고, 값은 그 시각으로부터 지난 초(t_s)다.
서버와 오디오 클라이언트가 같은 프로세스에서 도는 `local` 모드에서만 시각이 정확히 맞는다
(서버를 따로 띄우는 serve/talk 모드는 기기 사이 시계가 달라 비교할 수 없다).

기록하는 단계(순서):
  vad_emit        VAD가 말 끝을 확정해 내보냄 (extra: delay_s = Smart Turn 등으로 일부러 늦춘 시간)
  stt_done        STT 최종 전사가 나옴
  llm_first_delta LLM(우리 RAG 서버)에서 첫 글자 조각이 도착 — RAG 서버의 재작성·검색·Claude 첫 토큰이 모두 포함
  llm_first_chunk 첫 문장 묶음 완성 (지금 방식 L0: 문장 분리기가 문장 끝을 확정한 시점)
  tts_input       첫 문장이 TTS 입력으로 전달됨
  tts_first_audio TTS가 첫 오디오 조각을 냄
  client_audio_in 오디오 클라이언트가 첫 오디오를 받음
  playback_start  재생 콜백이 첫 오디오를 스피커 쪽으로 넘김 (하드웨어 출력 지연은 포함하지 않음)
"""
from __future__ import annotations

import json
import os
import threading
from time import perf_counter
from typing import Any

import numpy as np

from speech_to_speech.pipeline.messages import (
    AudioOutput,
    LLMResponseChunk,
    Transcription,
    TTSInput,
    VADAudio,
)

_PATH = os.environ.get("S2S_LATENCY_TRACE") or None
_lock = threading.Lock()
_state: dict[str, Any] = {"turn": 0, "t0": None, "seen": set()}


def enabled() -> bool:
    return _PATH is not None


def _append(record: dict[str, Any]) -> None:
    with _lock, open(_PATH, "a", encoding="utf-8") as f:  # type: ignore[arg-type]
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def new_turn(t0: float, **extra: Any) -> None:
    """VAD가 말 끝을 확정한 시각 t0(perf_counter)에서 새 턴을 시작한다."""
    if _PATH is None:
        return
    with _lock:
        _state["turn"] += 1
        _state["t0"] = t0
        _state["seen"] = {"vad_emit"}
        turn = _state["turn"]
    _append({"turn": turn, "stage": "vad_emit", "t_s": round(perf_counter() - t0, 4), **extra})


def mark(stage: str, **extra: Any) -> None:
    """현재 턴에서 stage를 처음 만났을 때만 기록한다. 이미 기록했거나 턴이 없으면 무시."""
    if _PATH is None:
        return
    now = perf_counter()
    with _lock:
        t0 = _state["t0"]
        if t0 is None or stage in _state["seen"]:
            return
        _state["seen"].add(stage)
        turn = _state["turn"]
    _append({"turn": turn, "stage": stage, "t_s": round(now - t0, 4), **extra})


def on_output(output: Any, queued_output: Any, source_input: Any) -> None:
    """BaseHandler.run이 결과를 다음 큐에 넣기 직전에 부른다. 메시지 종류로 단계를 구분한다."""
    if _PATH is None:
        return
    if isinstance(output, VADAudio):
        new_turn(
            output.created_at_s,
            delay_s=round(float(getattr(output, "processing_delay_s", 0.0) or 0.0), 4),
            turn_id=output.turn_id,
        )
    elif isinstance(output, Transcription):
        mark("stt_done", chars=len(output.text or ""))
    elif isinstance(output, LLMResponseChunk):
        if (output.text or "").strip():
            mark("llm_first_chunk", chars=len(output.text))
    elif isinstance(output, TTSInput):
        mark("tts_input", chars=len(output.text or ""))
    elif isinstance(source_input, TTSInput):
        is_audio = isinstance(queued_output, AudioOutput) or isinstance(output, (bytes, np.ndarray))
        if is_audio and not getattr(queued_output, "cleanup_only", False):
            mark("tts_first_audio")

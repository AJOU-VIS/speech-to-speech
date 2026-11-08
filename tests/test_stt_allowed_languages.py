"""Tests for restricting STT language auto-detection to allowed languages (AJOU-VIS).

Short or noisy Korean utterances were auto-detected as Chinese, Japanese, Vietnamese or
Icelandic. With ``allowed_languages="ko,en"`` an allowed detection is kept and anything
else falls back to the first allowed language. The models are faked, so no download is needed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from speech_to_speech.STT.allowed_languages import (
    apply_allowed_language,
    configure_allowed_languages,
    parse_allowed_languages,
)
from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

logger = logging.getLogger(__name__)


def test_parse_allowed_languages():
    assert parse_allowed_languages(" ko, EN ,") == ["ko", "en"]
    assert parse_allowed_languages("") is None
    assert parse_allowed_languages(None) is None


def test_apply_allowed_language_keeps_allowed_and_falls_back_to_first():
    allowed = ["ko", "en"]
    assert apply_allowed_language("en", allowed) == "en"
    assert apply_allowed_language("ko", allowed) == "ko"
    # "휴학" heard as Vietnamese, "고마워" as Japanese: both are misheard Korean.
    assert apply_allowed_language("vi", allowed) == "ko"
    assert apply_allowed_language("ja", allowed) == "ko"
    assert apply_allowed_language("zh", None) == "zh"
    assert apply_allowed_language(None, allowed) is None


def test_configure_allowed_languages_only_applies_to_auto():
    assert configure_allowed_languages("auto", "ko,en", logger) == ["ko", "en"]
    assert configure_allowed_languages("ko", "ko,en", logger) is None
    assert configure_allowed_languages("auto", None, logger) is None


def _mlx_handler(probs) -> MLXAudioWhisperSTTHandler:
    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.model = SimpleNamespace(detect_language=lambda mel, tokenizer=None: ("tokens", probs))
    return handler


def test_mlx_detection_keeps_an_allowed_language():
    probs = {"en": 0.6, "ko": 0.3, "zh": 0.1}
    handler = _mlx_handler(probs)

    handler._restrict_language_detection(["ko", "en"])

    assert handler.model.detect_language("mel")[1] == probs


def test_mlx_detection_falls_back_to_the_first_allowed_language():
    # "휴학": Vietnamese first, and English still outscores Korean among the allowed ones.
    handler = _mlx_handler({"vi": 0.5, "en": 0.3, "ko": 0.2})

    handler._restrict_language_detection(["ko", "en"])
    _, probs = handler.model.detect_language("mel")

    assert max(probs, key=probs.get) == "ko"


def test_mlx_detection_steers_batched_probabilities():
    handler = _mlx_handler([{"ja": 0.6, "en": 0.4}, {"en": 0.9, "zh": 0.1}])

    handler._restrict_language_detection(["ko", "en"])
    _, probs = handler.model.detect_language("mel")

    assert [max(p, key=p.get) for p in probs] == ["ko", "en"]


def test_mlx_detection_is_untouched_without_allowed_languages():
    probs = {"zh": 0.5, "ko": 0.5}
    handler = _mlx_handler(probs)

    handler._restrict_language_detection(None)

    assert handler.model.detect_language("mel")[1] == probs


def test_whisper_forces_the_first_allowed_language_for_an_outside_detection():
    """Russian (outside ko/en) is treated as misheard Korean and transcribed as Korean."""
    from tests.test_whisper_language_detection import FakeSequences, make_handler, run

    korean = FakeSequences([2626], "휴학")
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[korean],
        detect_language_result="<|ru|>",
    )
    handler.allowed_languages = ["ko", "en"]

    result = run(handler)

    assert handler.model.calls[0]["language"] == "ko"
    assert result.language_code == "ko-auto"

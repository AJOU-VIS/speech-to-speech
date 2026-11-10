import pytest

from speech_to_speech.TTS.ko_reading import KoreanReading
from speech_to_speech.TTS.ko_reading.numtext import NATIVE, SINO


def test_off_keeps_text():
    text = "양재역 9번 출구로 나오세요."
    assert KoreanReading("off")(text).text == text


def test_rules_reads_numbers_as_spoken():
    result = KoreanReading("rules")("양재역 9번 출구로 나와서 셔틀은 하루에 3번 와요. 문의는 031-219-3150이에요.")
    assert result.text == "양재역 구 번 출구로 나와서 셔틀은 하루에 세 번 와요. 문의는 공삼일 이일구 삼일오공이에요."
    assert result.by_model == 0


def test_only_upper_case_abbreviations_and_known_exams_are_spelled_out():
    result = KoreanReading("rules")("Google 계정 없이도 ROTC 신청은 TOEFL 점수와 iPhone15로 돼요.")
    assert result.text == "Google 계정 없이도 알오티씨 신청은 토플 점수와 iPhone15로 돼요."


def test_bun_mode_asks_the_model_only_for_bun_below_100():
    asked = []

    class FakeBunModel:
        def choose(self, text, spans):
            asked.extend(s.text for s in spans)
            return [{"span": s.text, "choice": NATIVE if s.text == "2번" else SINO, "seconds": 0.01} for s in spans]

    reading = KoreanReading("rules")
    reading.mode, reading.bun = "bun", FakeBunModel()
    result = reading("강남역 5번 출구로 나와서 3007번 버스를 2번 갈아타요.")
    assert asked == ["5번", "2번"]  # 3007번은 규칙
    assert result.text == "강남역 오 번 출구로 나와서 삼천칠 번 버스를 두 번 갈아타요."
    assert result.by_model == 2


def test_bun_mode_needs_an_adapter():
    with pytest.raises(ValueError):
        KoreanReading("bun")


def test_merge_lora_adds_scaled_low_rank_update():
    torch = pytest.importorskip("torch")
    from speech_to_speech.TTS.ko_reading.bun_model import merge_lora

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(4, 3, bias=False)
    before = model.proj.weight.detach().clone()
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    merged = merge_lora(model, {"base_model.model.proj.lora_A.weight": a, "base_model.model.proj.lora_B.weight": b}, 2.0)
    assert merged == 1
    assert torch.allclose(model.proj.weight, before + 2.0 * b @ a)


def test_handler_reads_korean_and_leaves_english():
    pytest.importorskip("torch")
    from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

    handler = object.__new__(Qwen3TTSHandler)
    handler.language = "auto"
    handler._utterance_language = None
    handler.reading = KoreanReading("rules")
    assert handler._reading_text("9번 출구로 나오세요.") == "구 번 출구로 나오세요."
    assert handler._reading_text("Take exit 9 and walk 3 minutes.") == "Take exit 9 and walk 3 minutes."

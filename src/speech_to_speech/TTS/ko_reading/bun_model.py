"""'번' 판단 모델: Qwen3-0.6B + AJOU-VIS가 AI Hub '번' 칸으로 학습한 LoRA (번호면 구 번, 횟수면 아홉 번).

- 지시문·질문 모양은 AJOU-VIS/ajou-vis research/normalize/judge.py와 같다 (학습 때 쓴 입력, 예시 없이 지시문만).
- 보기를 정순·역순으로 두 번 물어 보기 번호(1·2·3) 확률을 평균한다. 글을 만들지 않으므로 숫자 값이 바뀔 수 없다.
- LoRA는 서버를 켤 때 0.6B 가중치에 직접 더한다 (fp32로 더한 뒤 bf16). peft가 필요 없고, 추론은 0.6B와 같은 속도다.
- LoRA 폴더(adapter_config.json, adapter_model.safetensors)는 AI Hub 데이터로 만든 것이라 레포에 넣지 않는다.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .numtext import Span

logger = logging.getLogger(__name__)

SYSTEM = """너는 한국어 문장 속 숫자를 소리 내어 읽는 방법을 고르는 도구야. 문장에서 ⟦ ⟧로 표시한 숫자 표현을 이 문장에서 어떻게 읽는지 보기 중에서 골라 번호만 답해.
- 번호(출구, 버스, 창구, 건물, 항목처럼 이름으로 붙인 수)는 한자어로, 횟수(몇 번 하는지)는 고유어로 읽어.
- 시각의 '시'는 열두 시까지 고유어, 열세 시부터 한자어로 읽어. 시간의 '분'은 한자어로, 사람을 높여 세는 '분'은 고유어로 읽어.
- 명·개·시간·달처럼 세는 단위는 고유어, 학점·층·년·원처럼 정해진 단위는 한자어로 읽어.
- 같은 단위도 이름(몇째 학기, 몇 학년)이면 한자어, 개수(몇 학기 동안)면 고유어로 읽어.
- 0으로 시작하는 번호는 한 자리씩 읽어."""

DEFAULT_BASE = "Qwen/Qwen3-0.6B"

# 켤 때 확인: 같은 숫자를 번호·횟수로 쓴 짝 문장 (ajou-vis research/eval/bun_pairs_v1.jsonl, 학습한 모델이 맞힌 것)
SELF_CHECK = [
    ("안내데스크는 1번 게이트 바로 옆에 있어요.", "1번", "한자어"),
    ("장학금 면담은 학기마다 1번 해요.", "1번", "고유어"),
    ("2번 창구는 점심시간에 쉬어요.", "2번", "한자어"),
    ("그 수업은 2번 결석하면 감점이에요.", "2번", "고유어"),
    ("기말 과제는 2번 주제로 골랐어요.", "2번", "한자어"),
    ("그 강의는 2번 들었는데도 어려웠어요.", "2번", "고유어"),
]


def mark(text: str, span: Span) -> str:
    return f"{text[:span.start]}⟦{span.text}⟧{text[span.end:]}"


def question(marked: str, options: list[tuple[str, str]]) -> str:
    lines = [f"{i}) {reading}" for i, (_, reading) in enumerate(options, 1)]
    return f"문장: {marked}\n보기:\n" + "\n".join(lines) + "\n답:"


def merge_lora(model, weights: dict, scale: float) -> int:
    """W += scale * B @ A 를 LoRA가 붙은 층마다 한다. 더한 층 수를 돌려준다."""
    modules = dict(model.named_modules())
    merged = 0
    for key, a in weights.items():
        if ".lora_A." not in key:
            continue
        b = weights[key.replace(".lora_A.", ".lora_B.")]
        name = key.split("base_model.model.", 1)[-1].rsplit(".lora_A.", 1)[0]
        layer = modules[name]
        delta = scale * (b.float() @ a.float())
        layer.weight.data = (layer.weight.data.float() + delta).to(layer.weight.dtype)
        merged += 1
    return merged


class BunModel:
    def __init__(self, adapter_dir: str | Path, device: str = "cuda", base: str | None = None):
        import torch
        import transformers
        from safetensors.torch import load_file

        adapter = Path(adapter_dir)
        cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("modules_to_save"):
            raise ValueError(f"지원하지 않는 LoRA 설정: {adapter}")
        base = base or cfg.get("base_model_name_or_path") or DEFAULT_BASE
        self.torch = torch
        started = time.perf_counter()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base)
        self.tokenizer.padding_side = "left"
        model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        merged = merge_lora(model, load_file(str(adapter / "adapter_model.safetensors")), cfg["lora_alpha"] / cfg["r"])
        if merged == 0:
            raise ValueError(f"LoRA 가중치를 찾지 못함: {adapter}")
        use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
        self.model = model.to(device=device if use_cuda else "cpu", dtype=dtype).eval()
        self.digit_ids = []
        for k in (1, 2, 3):
            ids = self.tokenizer.encode(str(k), add_special_tokens=False)
            assert len(ids) == 1, f"보기 번호 {k}가 토큰 하나가 아님: {ids}"
            self.digit_ids.append(ids[0])
        logger.info("'번' 모델 준비: %s + LoRA %d층 (%s, %.1f초)", base, merged, adapter, time.perf_counter() - started)

    def _prompt(self, marked: str, shown: list[tuple[str, str]]) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question(marked, shown)}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def choose(self, text: str, spans: list[Span]) -> list[dict]:
        """칸마다 보기 확률(정순·역순 평균)과 고른 방식. 칸 여러 개를 한 번에 묻는다."""
        if not spans:
            return []
        torch = self.torch
        started = time.perf_counter()
        jobs = []  # (칸 번호, 보기 순서, 입력)
        for i, span in enumerate(spans):
            options = span.options()
            marked = mark(text, span)
            for order in (list(range(len(options))), list(reversed(range(len(options))))):
                jobs.append((i, order, self._prompt(marked, [options[j] for j in order])))
        enc = self.tokenizer([p for _, _, p in jobs], return_tensors="pt", padding=True).to(self.model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos,
                             logits_to_keep=1)
        logits = out.logits[:, -1].float()[:, self.digit_ids]
        totals = [dict.fromkeys((how for how, _ in s.options()), 0.0) for s in spans]
        for (i, order, _), row in zip(jobs, logits):
            probs = torch.softmax(row[:len(order)], dim=0).tolist()
            hows = [spans[i].options()[j][0] for j in order]
            for how, p in zip(hows, probs):
                totals[i][how] += p / 2
        seconds = (time.perf_counter() - started) / len(spans)
        return [{"span": s.text, "choice": max(t, key=t.get), "probs": {k: round(v, 4) for k, v in t.items()},
                 "seconds": round(seconds, 4)} for s, t in zip(spans, totals)]

    def self_check(self) -> int:
        """짝 문장으로 확인한다. 틀린 수를 돌려준다 (0이어야 정상, LoRA가 안 붙었으면 여러 개 틀린다)."""
        from .numtext import find_spans

        wrong = 0
        for text, target, answer in SELF_CHECK:
            span = next(s for s in find_spans(text) if s.text == target)
            picked = self.choose(text, [span])[0]["choice"]
            if picked != answer:
                wrong += 1
                logger.warning("'번' 모델 확인 틀림: %s → %s (정답 %s)", text, picked, answer)
        logger.info("'번' 모델 확인: 짝 문장 %d개 중 %d개 맞음", len(SELF_CHECK), len(SELF_CHECK) - wrong)
        return wrong

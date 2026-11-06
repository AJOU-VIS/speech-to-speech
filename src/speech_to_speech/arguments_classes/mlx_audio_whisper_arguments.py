from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLXAudioWhisperSTTHandlerArguments:
    mlx_audio_whisper_model_name: str = field(
        default="mlx-community/whisper-large-v3-turbo",
        metadata={
            "help": "The pretrained MLX Audio Whisper model to use. Default is 'mlx-community/whisper-large-v3-turbo'."
        },
    )
    mlx_audio_whisper_gen_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Additional generation kwargs to pass to the model. Default is an empty dict."},
    )
    mlx_audio_whisper_allowed_languages: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated language codes for auto-detection, e.g. 'ko,en'. A detected language "
            "outside this list is replaced by the first one, so short or noisy utterances are never "
            "transcribed in an unrelated language. Only used with language 'auto'. Default None (any language)."
        },
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "The conversation language, or 'auto' to detect it per utterance. Default is 'en'."},
    )

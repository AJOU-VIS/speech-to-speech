"""Restrict STT language auto-detection to a fixed set of languages (AJOU-VIS).

Whisper's auto-detection picks among ~100 languages, so a short or noisy utterance
("네", a breath before speaking) is sometimes tagged as Chinese, Japanese or Icelandic and
transcribed in that language. A campus guide that only talks Korean and English can rule
those out.

Choosing the most likely language *among* the allowed ones is not enough: for a short
Korean word ("휴학") English can still outscore Korean, and forcing English then produces
an invented English sentence. English speech is reliably detected as English, so a
detection outside the allowed set almost always means misheard Korean. The rule is
therefore: keep an allowed detection as is, otherwise use the first allowed language.
"""
from __future__ import annotations

from typing import Optional


def parse_allowed_languages(value: Optional[str]) -> Optional[list[str]]:
    """'ko, en' -> ['ko', 'en']; empty or None -> None (no restriction)."""
    if not value:
        return None
    codes = [code.strip().lower() for code in str(value).split(",") if code.strip()]
    return codes or None


def apply_allowed_language(detected: Optional[str], allowed: Optional[list[str]]) -> Optional[str]:
    """Keep ``detected`` if it is allowed; otherwise fall back to the first allowed language."""
    if not allowed or detected is None or detected in allowed:
        return detected
    return allowed[0]


def configure_allowed_languages(start_language: Optional[str], value: Optional[str], logger) -> Optional[list[str]]:
    """Parse the option for a handler; it only applies to auto-detection."""
    allowed = parse_allowed_languages(value)
    if allowed and start_language != "auto":
        logger.warning("allowed_languages only applies with language 'auto'; ignoring %s", allowed)
        return None
    if allowed:
        logger.info("STT language detection limited to %s (otherwise %s)", allowed, allowed[0])
    return allowed

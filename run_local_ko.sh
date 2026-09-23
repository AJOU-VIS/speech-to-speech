#!/bin/zsh
# AJOU-VIS 한국어 로컬 파이프라인 실행 스크립트
# 구성: whisper-large-v3-turbo(MLX) STT + 언어 자동 감지 + Qwen3-4B LLM + Qwen3-TTS
# 로그는 화면에 출력되면서 run_local.log 에도 기록된다 (Claude가 같이 볼 수 있게).
# 종료: Ctrl+C
cd "$(dirname "$0")"
exec .venv/bin/speech-to-speech local \
  --mac-optimal-settings \
  --stt mlx-audio-whisper \
  --language auto \
  --local_audio_block_mic_during_playback \
  --port 8766 \
  "$@" 2>&1 | tee -a run_local.log

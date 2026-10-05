#!/bin/zsh
# AJOU-VIS 음성 파이프라인 + RAG 서버 — STT를 Qwen3-ASR로 교체한 실험 버전
#
# 사용법 (터미널 2개):
#   터미널 1:  ~/ajou-vis/server/run_server.sh
#   터미널 2:  ~/speech-to-speech/run_local_rag_qwen3asr.sh
#
# Whisper 버전(run_local_rag.sh)과 STT만 다르고 나머지(RAG·TTS 설정)는 동일하다.
# 모델 크기 변경: QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf ./run_local_rag_qwen3asr.sh
#   (기본 0.6B가 더 빠르고, 1.7B가 더 정확)
cd "$(dirname "$0")"
exec .venv/bin/speech-to-speech local \
  --mac-optimal-settings \
  --stt qwen3-asr \
  --qwen3_asr_model_name "${QWEN3_ASR_MODEL:-Qwen/Qwen3-ASR-0.6B-hf}" \
  --qwen3_asr_language auto \
  --llm_backend chat-completions \
  --responses_api_base_url http://127.0.0.1:8100/v1 \
  --responses_api_api_key dummy \
  --model_name ajou-vis \
  --qwen3_tts_speaker Sohee \
  --qwen3_tts_coalesce_inputs False \
  --stream_batch_sentences 1 \
  --local_audio_block_mic_during_playback \
  --port 8766 \
  "$@" 2>&1 | tee -a run_local_qwen3asr.log

#!/bin/zsh
# AJOU-VIS 한국어 음성 파이프라인 + RAG 서버(치토) 연결 실행 스크립트
#
# 사용법 (터미널 2개):
#   터미널 1:  ~/ajou-vis/server/run_server.sh      ← RAG 서버 (포트 8100)
#   터미널 2:  ~/speech-to-speech/run_local_rag.sh  ← 이 스크립트
#
# 목소리: ajou-vis prompts/persona_ko.md 목소리 블록을 그대로 따른다 (server/voice_config.py가 읽음,
#   Colab 노트북과 같은 원본). 지금은 기준 음성 복제(chito_ref.wav) + 문장 사이 0.4초.
#   잠깐 바꿔 보려면 뒤에 옵션을 붙인다 — 뒤에 준 값이 이긴다.
#   예) ./run_local_rag.sh --qwen3_tts_sentence_pause_s 0
#   ajou-vis가 다른 곳에 있으면 AJOU_VIS_DIR=<경로> ./run_local_rag.sh
# 구성: whisper-large-v3-turbo(MLX) STT + 언어 자동 감지(한국어·영어만, 그 밖은 한국어로 처리)
#      + LLM 슬롯 → RAG 서버(rewrite → 학칙 검색 → 게이트 → 치토 답변, 스트리밍)
#      + Qwen3-TTS. 로그는 run_local.log 에도 기록. 종료: Ctrl+C
cd "$(dirname "$0")"
AJOU_VIS_DIR="${AJOU_VIS_DIR:-$HOME/ajou-vis}"
voice_out=$("$AJOU_VIS_DIR/server/.venv/bin/python" "$AJOU_VIS_DIR/server/voice_config.py" --args) || {
  echo "치토 목소리 설정을 못 읽음 ($AJOU_VIS_DIR/prompts/persona_ko.md 목소리 블록 확인)" >&2; exit 1; }
voice_args=("${(@f)voice_out}")
echo "목소리: ${voice_args[*]}"
exec .venv/bin/speech-to-speech local \
  --mac-optimal-settings \
  --stt mlx-audio-whisper \
  --language auto \
  --mlx_audio_whisper_allowed_languages ko,en \
  --llm_backend chat-completions \
  --responses_api_base_url http://127.0.0.1:8100/v1 \
  --responses_api_api_key dummy \
  --model_name ajou-vis \
  "${voice_args[@]}" \
  --qwen3_tts_coalesce_inputs False \
  --stream_batch_sentences 1 \
  --local_audio_block_mic_during_playback \
  --port 8766 \
  "$@" 2>&1 | tee -a run_local.log

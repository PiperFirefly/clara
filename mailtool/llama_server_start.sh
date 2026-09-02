#!/bin/bash
# Start the local llama.cpp vision server (gemma-3-4b-it + mmproj).
export PATH=/usr/bin:/bin:/usr/local/bin
LLAMA=$HOME/tools/other_llms/llama.cpp/llama-server
MODEL=$HOME/models/gemma-3-4b-it/gemma-3-4b-it-Q4_K_M.gguf
MMPROJ=$HOME/models/gemma-3-4b-it/mmproj-model-f16.gguf
LOG=$HOME/tools/communications/email/llama_server.log

mkdir -p "$(dirname "$LOG")"

pkill -f "llama-server.*gemma-3-4b" 2>/dev/null
sleep 2
nohup "$LLAMA" -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port 8080 -c 8192 -ngl 0 \
  >> "$LOG" 2>&1 &

#!/bin/bash
# Start the local llama.cpp vision server — Qwen2.5-VL-7B-abliterated (uncensored).
# Serves on 127.0.0.1:8081. This is the "less prudish" local vision model.
export PATH=/usr/bin:/bin:/usr/local/bin
LLAMA=$HOME/tools/other_llms/llama.cpp/llama-server
MODEL=$HOME/models/qwen2.5-vl-7b-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf
MMPROJ=$HOME/models/qwen2.5-vl-7b-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-Q8_0.gguf

pkill -f "llama-server.*abliterated" 2>/dev/null
sleep 2
nohup "$LLAMA" -m "$MODEL" --mmproj "$MMPROJ" \
  --host 127.0.0.1 --port 8083 -c 8192 -ngl 0 \
  --alias qwen2.5-vl-7b-abliterated \
  >> $HOME/email/llama_server_qwen.log 2>&1 &

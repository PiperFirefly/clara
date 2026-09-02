#!/bin/bash
# Health check: restart the llama.cpp vision server if it is not running.
export PATH=/usr/bin:/bin:/usr/local/bin
if pgrep -f "llama-server" >/dev/null 2>&1; then
  exit 0
fi
echo "$(date '+%F %T') llama-server not running — restarting"
/bin/bash $HOME/mailtool/llama_server_start.sh

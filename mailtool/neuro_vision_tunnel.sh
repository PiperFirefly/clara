#!/bin/bash
# Open/close the SSH forward from server to worker's vision host.
#   server localhost:8090  ->  worker localhost:8083 (llama-server)
# Rides the existing reverse tunnel (server -> relay -> worker:22),
# so no new firewall rules or reverse tunnels are needed on the remote side.
#
# Usage: neuro_vision_tunnel.sh up|down|status
set -u
SOCK=~/.pi/agent/neuro_vision.sock
FWD="8090:localhost:8083"

case "${1:-status}" in
  up)
    if ssh -S "$SOCK" -O check worker 2>/dev/null; then
      echo "tunnel already up"
      exit 0
    fi
    ssh -f -N -M -S "$SOCK" \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -L "$FWD" worker
    echo "tunnel up: localhost:$FWD -> worker:8083 (socket $SOCK)"
    ;;
  down)
    ssh -S "$SOCK" -O exit worker 2>/dev/null && echo "tunnel down" \
      || echo "tunnel was not up"
    ;;
  status)
    if ssh -S "$SOCK" -O check worker 2>/dev/null; then
      echo "tunnel UP (localhost:$FWD)"
    else
      echo "tunnel DOWN"
    fi
    ;;
esac

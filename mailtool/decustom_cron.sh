#!/bin/bash
# decustom_cron.sh — run the decustomization scan gate on a schedule.
# Flags hardcoded instance/operator values (public IPs, abs home paths,
# hostnames, the operator's real name) so decustomization can't silently creep
# back. Logs to ~/learning/freeroam/decustom.log with a prominent DRIFT marker on failure.
set -u
LOG="$HOME/learning/freeroam/decustom.log"
{
  echo "=== $(date -Is) ==="
  if /usr/bin/python3 "$HOME/mailtool/decustom_check.py" 2>&1; then
    echo "  clean"
  else
    echo "  *** DECUSTOM DRIFT: hardcoded instance/operator values detected (see above) ***"
  fi
} >> "$LOG"

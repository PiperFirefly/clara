#!/bin/bash
# Check for pending apt package updates; notify on security or large backlog.
# Run daily via cron. Logging handled by cron redirect to system_update.log.
export PATH=/usr/bin:/bin:/usr/local/bin
STAMP=$(date '+%F %T')

if ! sudo -n apt-get update -qq 2>&1; then
  echo "$STAMP apt-get update FAILED"
  exit 1
fi

UPGRADABLE=$(apt list --upgradable 2>/dev/null | grep -v '^Listing' | wc -l)
SECURITY=$(apt list --upgradable 2>/dev/null | grep -ci 'security')

echo "$STAMP upgradable=$UPGRADABLE security=$SECURITY"

if [ "$SECURITY" -gt 0 ] || [ "$UPGRADABLE" -gt 10 ]; then
  /usr/bin/python3 $HOME/mailtool/notify.py --email --telegram \
    "System updates pending (server)" \
    "$UPGRADABLE package updates pending, $SECURITY security."
fi

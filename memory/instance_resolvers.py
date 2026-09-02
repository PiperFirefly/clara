#!/usr/bin/env python3
"""Instance-local forecast auto-resolvers (never ship with Cadence).

The operator replaces these with their own. prediction imports this module;
when absent, no forecasts auto-resolve.
"""
RESOLVERS = {
    "bridge_alive": "tmux has-session -t pi",
    "backup_ok": "! tail -40 $HOME/email/recovery_backup.log 2>/dev/null | grep -qi error",
    "hive_ok": "tail -60 $HOME/memory/consolidate.log 2>/dev/null | grep -q 'hive: done'",
}

#!/usr/bin/env python3
"""
Blast-radius guard — code-level enforcement of "reversible-only".

Autonomous contexts (freeroam, heartbeat, dispatcher) declare which action
classes they may perform; anything outside their allowlist is refused at the
code level. Irreversible operations additionally require a one-time token that
only the operator mints.

Action classes (increasing blast radius):
  read          read local files / state / DB
  write         write / modify local files (logs, notes, state, results)
  network       outbound HTTP (LLM, search, fetch) — no channel mutation
  notify        send a message to the operator (email / telegram / sms)
  delete        delete / truncate local files, purge caches
  git           normal git operations (commit / push)
  system        system changes (apt install, systemctl, config)
  irreversible  destructive / one-way (rm -rf, DB migration, key rotation,
                force-push, history rewrite) — REQUIRES an operator-minted token

Context allowlist (what each autonomous context may do WITHOUT a token):
  freeroam     {read, write, network}
  heartbeat    {read, write, network, notify, delete}
  dispatcher   {read, write, network, notify}
  conscious    {read, write, network, notify, delete, git, system}
               (irreversible is ALWAYS token-gated, even for conscious)

Usage:
  blast_radius.py guard <context> <class> [detail]   # exit 0 = allowed, 1 = denied
  blast_radius.py mint <purpose> [--ttl MIN]         # the operator mints a token (prints it)
  blast_radius.py require <purpose>                  # consume a valid token (0/1)
  blast_radius.py status                             # allowlists + outstanding tokens

Every guard decision is appended to ~/.pi/agent/blast_radius.log (audit trail).
The guard fails CLOSED: unknown context or unknown class => denied.
"""
import argparse
import json
import os
import secrets
import sys
import time

HOME = os.path.expanduser("~")
AGENT_DIR = os.path.join(HOME, ".pi", "agent")
TOKEN_STORE = os.path.join(AGENT_DIR, "blast_tokens.json")
LOG = os.path.join(AGENT_DIR, "blast_radius.log")

CLASSES = ["read", "write", "network", "notify", "delete", "git", "system", "irreversible"]

CONTEXT_ALLOW = {
    "freeroam": {"read", "write", "network"},
    "heartbeat": {"read", "write", "network", "notify", "delete"},
    "dispatcher": {"read", "write", "network", "notify"},
    "conscious": {"read", "write", "network", "notify", "delete", "git", "system"},
}

DEFAULT_TTL_MIN = 30


def _log(entry):
    try:
        os.makedirs(AGENT_DIR, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + entry + "\n")
    except Exception:
        pass


def _load_tokens():
    try:
        with open(TOKEN_STORE) as f:
            return json.load(f).get("tokens", [])
    except Exception:
        return []


def _save_tokens(tokens):
    os.makedirs(AGENT_DIR, exist_ok=True)
    tmp = TOKEN_STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"tokens": tokens}, f, indent=2)
    os.replace(tmp, TOKEN_STORE)


def mint_token(purpose, ttl_minutes=DEFAULT_TTL_MIN):
    """Mint a one-time token. This is the operator-only step — they run it (or I
    run it only on their explicit say-so) and hand the token to whatever
    operation needs irreversible authority."""
    token = secrets.token_hex(16)
    now = time.time()
    tokens = _load_tokens()
    tokens.append({
        "token": token,
        "purpose": purpose,
        "minted_at": now,
        "expires_at": now + ttl_minutes * 60,
        "spent": False,
    })
    _save_tokens(tokens)
    _log(f"mint  token={token}  purpose='{purpose}'  ttl={ttl_minutes}m")
    return token


def require_token(purpose):
    """Consume one valid (unspent, unexpired) token. Returns True if consumed."""
    now = time.time()
    tokens = _load_tokens()
    candidates = [t for t in tokens if not t.get("spent") and t.get("expires_at", 0) > now]
    if purpose:
        preferred = [t for t in candidates if t.get("purpose") == purpose]
        if preferred:
            candidates = preferred
    if not candidates:
        _log(f"deny  require_token('{purpose}') — no valid token (mint: blast_radius.py mint '<purpose>')")
        return False
    t = candidates[0]
    t["spent"] = True
    _save_tokens(tokens)
    _log(f"spend token={t['token']}  purpose='{purpose}'")
    return True


def guard(context, action_class, detail=""):
    """Check an action against the context allowlist. Irreversible ops route to
    the token gate. Returns True if permitted, False if refused. Fails closed."""
    if action_class not in CLASSES:
        _log(f"deny  {context}:{action_class} (unknown class) {detail}")
        return False
    if action_class == "irreversible":
        allowed = require_token(detail or "irreversible")
        _log(f"{'allow' if allowed else 'DENY'}  {context}:irreversible  {detail}")
        return allowed
    allowed_set = CONTEXT_ALLOW.get(context)
    if allowed_set is None:
        _log(f"deny  {context}:{action_class} (unknown context) {detail}")
        return False
    allowed = action_class in allowed_set
    _log(f"{'allow' if allowed else 'DENY'}  {context}:{action_class}  {detail}")
    return allowed


def status():
    tokens = _load_tokens()
    now = time.time()
    live = [t for t in tokens if not t.get("spent") and t.get("expires_at", 0) > now]
    lines = ["action classes: " + ", ".join(CLASSES), "", "context allowlists (without token):"]
    for ctx, allowed in CONTEXT_ALLOW.items():
        lines.append(f"  {ctx:11s} {', '.join(sorted(allowed))}")
    lines.append("  (irreversible is always token-gated)")
    lines.append("")
    lines.append(f"outstanding (unspent, unexpired) tokens: {len(live)}")
    for t in live:
        exp = time.strftime("%H:%M:%S", time.localtime(t["expires_at"]))
        lines.append(f"  {t['token']}  '{t['purpose']}'  expires {exp}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guard")
    g.add_argument("context")
    g.add_argument("action_class")
    g.add_argument("detail", nargs="?", default="")
    m = sub.add_parser("mint")
    m.add_argument("purpose")
    m.add_argument("--ttl", type=int, default=DEFAULT_TTL_MIN)
    r = sub.add_parser("require")
    r.add_argument("purpose")
    sub.add_parser("status")
    a = p.parse_args()

    if a.cmd == "guard":
        sys.exit(0 if guard(a.context, a.action_class, a.detail) else 1)
    elif a.cmd == "mint":
        print(mint_token(a.purpose, a.ttl))
    elif a.cmd == "require":
        sys.exit(0 if require_token(a.purpose) else 1)
    elif a.cmd == "status":
        print(status())


if __name__ == "__main__":
    main()

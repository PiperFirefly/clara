#!/usr/bin/env python3
"""Simple IMAP/SMTP email tool.

Subcommands:
  fetch [--all]        Download messages from INBOX to local storage.
  list                 List locally stored messages.
  send <to> <subject> <body>   Send an email (body read from stdin if omitted).
  send-file <to> <subject> <file>  Send an email with body from a file.
"""

import argparse
import contextlib
import email as email_lib
import email.message
import email.utils
import fcntl
import imaplib
import json
import os
import re
import smtplib
import sys
import time
from email.header import decode_header, make_header

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    import client
    return client.email_config()


def connect_imap(cfg):
    M = imaplib.IMAP4_SSL(cfg["imap"]["host"], cfg["imap"]["port"])
    M.login(cfg["imap"]["username"], cfg["imap"]["password"])
    return M


def sanitize(name):
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] or "no-subject"


def decode_subject(msg):
    raw = msg.get("Subject", "")
    if not raw:
        return "(no subject)"
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def load_index(storage):
    idx_path = os.path.join(storage, "inbox", "index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"messages": []}
    return {"messages": []}


def save_index(storage, index):
    idx_path = os.path.join(storage, "inbox", "index.json")
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, idx_path)


@contextlib.contextmanager
def index_lock(storage):
    """Cross-process lock guarding index.json (used by agent_inbox and agent_loop)."""
    lock_path = os.path.join(storage, "inbox", ".index.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def fetch(cfg, fetch_all=False):
    storage = cfg["storage"]
    inbox_dir = os.path.join(storage, "inbox")
    os.makedirs(inbox_dir, exist_ok=True)
    index = load_index(storage)
    known_uids = {m["uid"] for m in index["messages"]}

    M = connect_imap(cfg)
    try:
        status, data = M.select("INBOX")
        if status != "OK":
            print("Could not select INBOX:", data)
            return
        typ, msg_ids = M.search(None, "ALL")
        ids = msg_ids[0].split()
        print(f"Found {len(ids)} message(s) in INBOX.")

        for num in ids:
            typ, uid_data = M.fetch(num, "(UID)")
            uid = uid_data[0].decode().split()[2].rstrip(")")
            if not fetch_all and uid in known_uids:
                continue
            typ, msg_data = M.fetch(num, "(RFC822)")
            if typ != "OK":
                print(f"  skip UID {uid}: fetch failed")
                continue
            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw)
            subject = sanitize(decode_subject(msg))
            date_str = email.utils.parsedate_to_datetime(msg.get("Date"))
            if date_str is None:
                date_str = time.gmtime()
                folder = time.strftime("%Y/%m", date_str)
            else:
                folder = date_str.strftime("%Y/%m")
            fname = f"{uid}_{subject}.eml"
            rel = os.path.join(folder, fname)
            abs_path = os.path.join(inbox_dir, folder, fname)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "wb") as f:
                f.write(raw)
            entry = {
                "uid": uid,
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "subject": decode_subject(msg),
                "date": msg.get("Date", ""),
                "file": rel,
            }
            existing = next((m for m in index["messages"] if m["uid"] == uid), None)
            if existing:
                existing.update(entry)
            else:
                index["messages"].append(entry)
            print(f"  saved UID {uid}: {decode_subject(msg)} -> {rel}")
        save_index(storage, index)
    finally:
        M.logout()


def list_local(cfg):
    storage = cfg["storage"]
    index = load_index(storage)
    msgs = sorted(index["messages"], key=lambda m: m.get("date", ""))
    if not msgs:
        print("No messages stored locally.")
        return
    for m in msgs:
        print(f"{m['uid']:>6}  {m.get('date',''):<31}  {m.get('from',''):<40}  {m.get('subject','')}")


def send(cfg, to, subject, body):
    from_addr = cfg["from"]
    msg = email.message.EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg.set_content(body)
    with smtplib.SMTP_SSL(cfg["smtp"]["host"], cfg["smtp"]["port"], timeout=30) as s:
        s.login(cfg["smtp"]["username"], cfg["smtp"]["password"])
        s.send_message(msg)
    print(f"Sent email from {from_addr} to {to} with subject: {subject}")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description="IMAP/SMTP email tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="Download INBOX messages locally")
    pf.add_argument("--all", action="store_true", help="Re-fetch all messages")

    sub.add_parser("list", help="List locally stored messages")

    ps = sub.add_parser("send", help="Send an email")
    ps.add_argument("to")
    ps.add_argument("subject")
    ps.add_argument("body", nargs="?", help="Body text (or read from stdin if omitted)")

    psf = sub.add_parser("send-file", help="Send an email with body from a file")
    psf.add_argument("to")
    psf.add_argument("subject")
    psf.add_argument("file")

    args = p.parse_args()

    if args.cmd == "fetch":
        fetch(cfg, fetch_all=args.all)
    elif args.cmd == "list":
        list_local(cfg)
    elif args.cmd == "send":
        body = args.body
        if body is None:
            body = sys.stdin.read()
        send(cfg, args.to, args.subject, body)
    elif args.cmd == "send-file":
        with open(args.file, "r", encoding="utf-8") as f:
            body = f.read()
        send(cfg, args.to, args.subject, body)


if __name__ == "__main__":
    main()

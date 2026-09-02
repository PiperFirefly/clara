#!/usr/bin/env python3
"""migrate_memory_to_db.py — move Agent's flat memory/email files into the docstore.

Source of truth becomes the document store (~/memory/memory.db, `documents` table):
  - ~/agent_memory.md  -> doc `agent/memory_main`  (kind memory)
  - ~/agent_email.md   -> doc `agent/email_brain`  (kind memory)

The .md files are NOT deleted — they stay as *derived caches* so the transition is
reversible and no reader breaks if a path is missed. The DB is authoritative.

Idempotent: an existing doc is left untouched unless --force. A doc is considered
migrated when it exists and is not empty.

Usage:
  migrate_memory_to_db.py            migrate any missing docs from the .md files
  migrate_memory_to_db.py --dry-run  preview without writing
  migrate_memory_to_db.py --force    overwrite doc content from the .md file
  migrate_memory_to_db.py --reverse  write the doc back out to the .md cache (rollback)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/memory"))

import docstore  # noqa: E402

HOME = os.path.expanduser("~")

SOURCES = [
    ("agent_memory.md", "agent/memory_main", "Agent's persistent memory ledger"),
    ("agent_email.md", "agent/email_brain", "Agent's inbox brain"),
]


def migrate(doc_key, kind, title, md_path, dry_run=False, force=False):
    md = None
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
    if md is None or not md.strip():
        print(f"[skip] {md_path}: file missing or empty (nothing to migrate)")
        return
    existing = docstore.doc_get(doc_key)
    if existing is not None and not force:
        n = len(existing["content"])
        print(f"[skip] {doc_key}: doc already exists ({n} chars); use --force to overwrite from file")
        return
    if dry_run:
        print(f"[dry-run] would set {doc_key} <- {md_path} ({len(md)} chars)")
        return
    docstore.doc_set(doc_key, kind, title, md)
    print(f"[ok] {doc_key} <- {md_path} ({len(md)} chars)")


def reverse(doc_key, md_path, dry_run=False):
    row = docstore.doc_get(doc_key)
    if row is None:
        print(f"[skip] {doc_key}: no doc to write back")
        return
    content = row["content"]
    if dry_run:
        print(f"[dry-run] would write {md_path} <- {doc_key} ({len(content)} chars)")
        return
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content.rstrip("\n") + "\n")
    print(f"[ok] {md_path} <- {doc_key} ({len(content)} chars)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="preview without writing")
    ap.add_argument("--force", action="store_true", help="overwrite existing docs from file")
    ap.add_argument("--reverse", action="store_true", help="write docs back out to the .md caches")
    args = ap.parse_args()

    if args.reverse:
        for fn, key, _ in SOURCES:
            reverse(key, os.path.join(HOME, fn), dry_run=args.dry_run)
        return

    for fn, key, title in SOURCES:
        migrate(key, "memory", title, os.path.join(HOME, fn),
                dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()

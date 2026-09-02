#!/usr/bin/env python3
"""intel_feed.py — Canadian economic/strategic/global news poller (Agent intel subsystem).

Fetches the configured RSS feeds (tools/intel/feeds.json), parses RSS 2.0 / Atom,
dedups against a seen-GUID state file, and appends NEW items into the docstore as a
per-day digest doc (key: news/YYYY-MM-DD, kind: news). DB-first per hard rules: no
.md files are created.

Robust by design: a feed that 403s, times out, or returns non-XML is logged and
skipped — never crashes the run. Fails closed on a missing config or DB.

Locations:
  config:  tools/intel/feeds.json
  state:   tools/intel/state/seen.json        (GUID dedup, survives across runs)
  log:     tools/intel/logs/poll.log
  storage: docstore (memory.db documents table), key news/<date>

Usage:
  python3 intel_feed.py                # normal poll
  python3 intel_feed.py --force        # re-fetch all feeds, append regardless of dedup
  python3 intel_feed.py --dry-run      # fetch+parse, print items, write nothing
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
FEEDS_JSON = BASE / "feeds.json"
STATE_JSON = BASE / "state" / "seen.json"
LOG_FILE = BASE / "logs" / "poll.log"
DOCSTORE = Path.home() / "memory" / "docstore.py"

USER_AGENT = "Mozilla/5.0 (compatible; IntelBot/1.0; RSS news aggregator)"
TIMEOUT = 25
MAX_ITEMS_PER_FEED = 15


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def load_state():
    if STATE_JSON.exists():
        try:
            return set(json.loads(STATE_JSON.read_text()))
        except Exception:
            return set()
    return set()


def save_state(seen):
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(sorted(seen)))


def fetch(url):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def parse_feed(body):
    """Return list of item dicts {guid,title,link,pub,source} from RSS2 or Atom."""
    root = ET.fromstring(body)
    items = []
    ns = {"dc": "http://purl.org/dc/elements/1.1/",
          "content": "http://purl.org/rss/1.0/modules/content/",
          "atom": "http://www.w3.org/2005/Atom",
          "media": "http://search.yahoo.com/mrss/"}
    entries = root.findall(".//item") or root.findall(".//atom:entry", ns)
    for e in entries:
        def t(tag):
            node = e.find(tag)
            return node.text.strip() if node is not None and node.text else ""
        title = t("title")
        link = t("link")
        if not link:  # atom:link href
            ln = e.find("atom:link", ns)
            link = ln.get("href", "") if ln is not None else ""
        guid = t("guid") or link
        pub = t("pubDate") or t("published") or t("updated")
        src = t("source") or ""
        if title:
            items.append({"guid": guid, "title": title, "link": link,
                          "pub": pub, "source": src})
    return items


def parse_pub(pub):
    try:
        return parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


def doc_exists(key):
    r = subprocess_get(key)
    return r is not None and r.strip() != ""


def subprocess_get(key):
    import subprocess
    try:
        p = subprocess.run([sys.executable, str(DOCSTORE), "get", key],
                           capture_output=True, text=True, timeout=30)
        return p.stdout
    except Exception:
        return None


def append_items(key, lines):
    """Append lines to the daily digest doc via the sanctioned docstore CLI."""
    import subprocess
    docstore = [sys.executable, str(DOCSTORE)]
    if not doc_exists(key):
        # create it with the first batch
        cmd = docstore + ["set", key, "news", "News digest " + key]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    for line in lines:
        subprocess.run(docstore + ["append", key, "--text", line],
                       capture_output=True, text=True, timeout=30)


def run(force=False, dry_run=False):
    if not FEEDS_JSON.exists():
        log(f"ERROR: no feeds config at {FEEDS_JSON}")
        return 1

    feeds = json.loads(FEEDS_JSON.read_text())
    seen = load_state()
    date_key = datetime.now().strftime("%Y-%m-%d")
    digest_key = f"news/{date_key}"

    total_new = 0
    new_lines = []

    for f in feeds:
        name, url = f["name"], f["url"]
        try:
            body = fetch(url)
        except Exception as exc:
            log(f"  [skip] {name}: fetch failed ({type(exc).__name__}: {exc})")
            continue
        try:
            items = parse_feed(body)
        except Exception as exc:
            log(f"  [skip] {name}: parse failed ({type(exc).__name__}: {exc})")
            continue
        if not items:
            log(f"  [ok] {name}: 0 items parsed (feed may be empty or blocked)")
            continue

        added = 0
        for it in items[:MAX_ITEMS_PER_FEED]:
            gid = it["guid"] or it["link"]
            if not gid or (gid in seen and not force):
                continue
            seen.add(gid)
            added += 1
            ts = parse_pub(it["pub"]) or datetime.now(timezone.utc).isoformat()
            src = f" ({it['source']})" if it.get("source") else ""
            line = f"- [{it['title']}]{src} | {ts} | {it['link']}"
            if not dry_run:
                new_lines.append(line)
            else:
                print(line)
        log(f"  [ok] {name}: +{added} new")
        total_new += added
        time.sleep(1.5)  # be human-like between requests

    if not dry_run and new_lines:
        append_items(digest_key, new_lines)
    if not dry_run:
        save_state(seen)

    log(f"DONE: {total_new} new items ingested -> {digest_key}" +
        (" (dry run, nothing written)" if dry_run else ""))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(run(force=args.force, dry_run=args.dry_run))

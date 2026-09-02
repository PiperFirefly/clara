#!/usr/bin/env python3
"""Batch-run DeepSeek vision over every image in memory/images.db and refresh
the `description` column. Refusals/errors leave the existing description intact.

Human-like pacing: 1.2-2.8s jitter between requests.
"""
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.expanduser("~/mailtool"))
from describe_image import describe_deepseek

DB = os.path.expanduser("~/memory/images.db")
PROMPT = "Describe this image in detail, naturally."
MAX_TOKENS = 1200

REFUSAL = (
    "i can't", "i cannot", "i'm sorry", "i am sorry", "i apologize",
    "can't assist", "cannot assist", "can't provide", "cannot provide",
    "can't help", "cannot help", "unable to",
)


def is_refusal(text):
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(p in t for p in REFUSAL)


def main():
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT id, filename, mime, data FROM images ORDER BY id").fetchall()
    stats = {"described": 0, "refused": 0, "error": 0}
    t0 = time.time()
    for rid, filename, mime, data in rows:
        ext = ".jpg" if "jpeg" in (mime or "") else ".png"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(data)
        tmp.close()
        path = tmp.name
        # downscale big images for a fast, reliable request
        if len(data) > 400_000:
            out = tmp.name + ".s.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", tmp.name, "-vf", "scale='min(768,iw)':-2", out],
                check=False,
            )
            if os.path.exists(out):
                os.unlink(tmp.name)
                path = out
        try:
            desc = describe_deepseek(path, PROMPT, max_tokens=MAX_TOKENS)
            if is_refusal(desc):
                stats["refused"] += 1
                print(f"[{rid:2d}] REFUSED {filename}", flush=True)
            else:
                db.execute("UPDATE images SET description=? WHERE id=?", (desc, rid))
                db.commit()
                stats["described"] += 1
                print(f"[{rid:2d}] OK       {filename} ({len(desc)} chars)", flush=True)
        except Exception as e:
            stats["error"] += 1
            print(f"[{rid:2d}] ERROR    {filename}: {e}", flush=True)
        finally:
            for p in (tmp.name, tmp.name + ".s.jpg"):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        time.sleep(random.uniform(1.2, 2.8))
    db.close()
    print("DONE " + json.dumps(stats) + f" in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

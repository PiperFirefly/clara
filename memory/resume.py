#!/usr/bin/env python3
"""resume.py — the structured, versioned technical resume (2026-08-31, operator's request).

The point: "display your resume" should be a database read (milliseconds),
never a live code review (20 minutes). Every piece Agent is made of — third-
party or hand-rolled — gets ONE row: a short name, a category, a version,
a one-line description, and its provenance. Hand-rolled pieces are versioned
by hand (bump on a real, noteworthy change — not every line touched); third-
party pieces carry the actual upstream version number.

Format (operator's example): `sem-code-g - v.1.03 semantic code graph - internally developed`

Storage: a dedicated `resume_items` table in memory.db (separate from the
`documents` table — this is a catalog, not long-form content; separate from
`facts` — this has real schema/structure worth its own table, like `beliefs`
or `forecasts`). NOT part of the Vesta ledger (that's the append-only audit
trail of state *changes*; this is a maintained *catalog*, more like a table
of contents than a history — items get updated in place on purpose).

CLI:
  resume.py add <short_name> <category> <version> <description> \\
      --provenance internal|third_party [--tpv X.Y.Z] [--path P]
  resume.py bump <short_name> [--version X.Y] [--description D]
  resume.py show [--category C]        # the instant, human-readable resume
  resume.py list [--category C]        # machine-readable (one line per item)
  resume.py categories                 # what categories exist + counts
  resume.py rm <short_name>
"""
import argparse
import os
import sqlite3
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "memory.db")

# Fixed display order — matches how operator framed the ask (architecture, code
# writing, cognitive, "everything") plus the categories the rest of the
# system already naturally falls into.
CATEGORY_ORDER = [
    "architecture", "coding", "cognitive", "selfmodel", "calibration",
    "socialcog", "memorycog", "reasoning", "communication",
    "safety", "voice", "subagents", "creative", "other",
]
CATEGORY_LABEL = {
    "architecture": "Architecture & Runtime",
    "coding": "Code-Writing / Coding Cortex",
    "cognitive": "Cognitive Subsystems",
    "selfmodel": "Self-Modeling / Knowing Myself",
    "calibration": "Calibration / Honest Confidence",
    "socialcog": "Social Cognition / Theory of Mind",
    "memorycog": "Memory & Reasoning over It",
    "reasoning": "Reasoning / Puzzle Tooling",
    "communication": "Communication & Channels",
    "safety": "Safety & Recovery",
    "voice": "Voice / Presence",
    "subagents": "Sub-Agents",
    "creative": "Creative / Persona",
    "other": "Other",
}


def _conn():
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE IF NOT EXISTS resume_items("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "short_name TEXT UNIQUE NOT NULL, "
        "category TEXT NOT NULL, "
        "version TEXT NOT NULL, "
        "description TEXT NOT NULL, "
        "provenance TEXT NOT NULL, "  # 'internal' | 'third_party'
        "vendor TEXT, "  # real product/vendor name for third_party items
        "path TEXT, "
        "updated_at REAL)"
    )
    cols = {r["name"] for r in c.execute("PRAGMA table_info(resume_items)")}
    if "vendor" not in cols:
        c.execute("ALTER TABLE resume_items ADD COLUMN vendor TEXT")
    return c


def add(short_name, category, version, description, provenance="internal",
        vendor=None, path=None):
    if provenance not in ("internal", "third_party"):
        raise ValueError("provenance must be 'internal' or 'third_party'")
    with _conn() as c:
        c.execute(
            "INSERT INTO resume_items(short_name, category, version, description, "
            "provenance, vendor, path, updated_at) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(short_name) DO UPDATE SET "
            "category=excluded.category, version=excluded.version, "
            "description=excluded.description, provenance=excluded.provenance, "
            "vendor=excluded.vendor, path=excluded.path, "
            "updated_at=excluded.updated_at",
            (short_name, category, version, description, provenance, vendor, path, time.time()),
        )
    return short_name


def _bump_version(v):
    """1.0 -> 1.1, 1.9 -> 1.10 (minor bump). Falls back to appending .1 if
    the version string isn't a simple '<major>.<minor>' float-shaped string."""
    try:
        major, minor = v.split(".", 1)
        return f"{major}.{int(minor) + 1}"
    except Exception:
        return v + ".1"


def bump(short_name, version=None, description=None):
    with _conn() as c:
        row = c.execute("SELECT * FROM resume_items WHERE short_name=?", (short_name,)).fetchone()
        if not row:
            raise KeyError(f"no resume item named {short_name!r} — use `add` first")
        new_version = version or _bump_version(row["version"])
        new_desc = description if description is not None else row["description"]
        c.execute(
            "UPDATE resume_items SET version=?, description=?, updated_at=? WHERE short_name=?",
            (new_version, new_desc, time.time(), short_name),
        )
    return new_version


def rm(short_name):
    with _conn() as c:
        c.execute("DELETE FROM resume_items WHERE short_name=?", (short_name,))


def list_items(category=None):
    with _conn() as c:
        if category:
            rows = c.execute(
                "SELECT * FROM resume_items WHERE category=? ORDER BY short_name", (category,)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM resume_items ORDER BY category, short_name").fetchall()
    return [dict(r) for r in rows]


def _fmt_line(item):
    if item["provenance"] == "third_party":
        prov = f"{item['vendor']}, third-party" if item["vendor"] else "third-party"
    else:
        prov = "internally developed"
    v = item["version"]
    vprefix = "" if v.lower().startswith("v") else "v."
    return f"{item['short_name']} - {vprefix}{v}  {item['description']} — {prov}"


def render(category=None):
    """The instant, human-readable resume. Pure DB read + string formatting —
    no code scanning, no live introspection. This is the whole point."""
    items = list_items(category)
    if not items:
        return "(no resume items yet — run `resume.py add ...`)"
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    cats = [c for c in CATEGORY_ORDER if c in by_cat] + \
           [c for c in by_cat if c not in CATEGORY_ORDER]
    lines = ["# Agent — Technical Resume",
             f"<!-- {len(items)} items, rendered {time.strftime('%Y-%m-%d %H:%M:%S')} -->", ""]
    for cat in cats:
        lines.append(f"## {CATEGORY_LABEL.get(cat, cat.title())}")
        for it in by_cat[cat]:
            lines.append(f"- {_fmt_line(it)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def categories():
    with _conn() as c:
        rows = c.execute(
            "SELECT category, COUNT(*) n FROM resume_items GROUP BY category ORDER BY category"
        ).fetchall()
    return [dict(r) for r in rows]


def _last_real_change(path):
    """Best-effort 'when did this piece actually last change' timestamp, so
    staleness can be DETECTED rather than assumed away. Not a version bump
    trigger (deciding whether a change is *significant* enough to bump still
    needs judgment) -- this only answers 'is the catalog entry possibly lying'.

    Order of preference:
      1. If under a git repo, the last commit's timestamp (a real semantic
         checkpoint, not every autosave).
      2. Else, latest mtime of the path itself (file) or files under it (dir).
      3. None if the path doesn't exist, isn't set, or looks like a URL.
    """
    import subprocess
    if not path or path.startswith("http://") or path.startswith("https://"):
        return None
    full = os.path.expanduser(os.path.join(os.path.dirname(BASE), path)) \
        if not os.path.isabs(path) else path
    if not os.path.exists(full):
        return None
    # 1. git-aware: walk up from `full` looking for a repo; ask git for the
    # last commit touching this path specifically.
    try:
        cwd = full if os.path.isdir(full) else os.path.dirname(full)
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", os.path.basename(full)],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    # 2. plain mtime fallback
    try:
        if os.path.isfile(full):
            return os.path.getmtime(full)
        latest = None
        for root, dirs, files in os.walk(full):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv", "venv", "node_modules")]
            for f in files:
                try:
                    mt = os.path.getmtime(os.path.join(root, f))
                    if latest is None or mt > latest:
                        latest = mt
                except OSError:
                    continue
        return latest
    except Exception:
        return None


def stale():
    """Items whose tracked path changed more recently than the catalog entry
    was last updated -- i.e. the resume is possibly lying about that piece.
    Detection only; does NOT auto-bump (a version bump needs judgment about
    whether the change was significant, not just that a file's mtime moved).
    """
    out = []
    for it in list_items():
        changed = _last_real_change(it["path"])
        if changed is not None and changed > it["updated_at"] + 60:  # 60s slack
            out.append({**it, "last_changed": changed,
                        "stale_days": (changed - it["updated_at"]) / 86400.0})
    return out


# Fine-grained cognitive sub-clusters (2026-09-02, operator: 'break the cognition
# section into several more specific clusters'). Items that were all lumped under
# 'cognitive' now get a specific home so the resume reads as distinct capabilities
# instead of one 28-item blob. Anything not listed stays in 'cognitive'.
RECLUSTER = {
    # self-modeling / knowing myself
    "interoception": "selfmodel",
    "metacognition-measurer": "selfmodel",
    "self-knowledge-feed": "selfmodel",
    "self-governance-layer": "selfmodel",
    "sentience-palace": "selfmodel",
    "warm-start": "selfmodel",
    "ablation-harness": "selfmodel",
    # calibration / honest confidence
    "belief-ledger": "calibration",
    "prediction-ledger": "calibration",
    "caliber": "calibration",
    "calibration-gym": "calibration",
    "intel-gym": "calibration",
    "contradiction-scan": "calibration",
    "tool-value-oracle": "calibration",
    # social cognition / theory of mind
    "theory-of-mind": "socialcog",
    "affect-tagging": "socialcog",
    # memory & reasoning over it
    "causal-graph": "memorycog",
    "counterfactual": "memorycog",
    "reason-worker": "memorycog",
    "temporal-validity": "memorycog",
    "entity-resolution": "memorycog",
    "hippo-kg": "memorycog",
    "skills-distill": "memorycog",
    "curiosity": "memorycog",
    "abduction": "memorycog",
    "s1s2-routing": "cognitive",  # genuinely cross-cutting; keep in cognitive
}


def reclust(dry_run=False):
    """Move cognitive items into their fine-grained sub-clusters."""
    moves = []
    for name, newcat in RECLUSTER.items():
        if newcat == "cognitive":
            continue
        with _conn() as c:
            r = c.execute("SELECT category FROM resume_items WHERE short_name=?", (name,)).fetchone()
            if r and r["category"] != newcat:
                moves.append((name, r["category"], newcat))
                if not dry_run:
                    c.execute("UPDATE resume_items SET category=? WHERE short_name=?", (newcat, name))
    if dry_run:
        for name, old, new in moves:
            print(f"  would move {name}: {old} -> {new}")
        print(f"({len(moves)} moves)")
    else:
        for name, old, new in moves:
            print(f"moved {name}: {old} -> {new}")
        print(f"done: {len(moves)} item(s) re-clustered")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add")
    p.add_argument("short_name")
    p.add_argument("category")
    p.add_argument("version")
    p.add_argument("description")
    p.add_argument("--provenance", default="internal", choices=["internal", "third_party"])
    p.add_argument("--vendor", default=None, help="real product/vendor name for third-party items")
    p.add_argument("--path", default=None)

    p = sub.add_parser("bump")
    p.add_argument("short_name")
    p.add_argument("--version", default=None)
    p.add_argument("--description", default=None)

    p = sub.add_parser("show")
    p.add_argument("--category", default=None)

    p = sub.add_parser("list")
    p.add_argument("--category", default=None)

    sub.add_parser("categories")
    sub.add_parser("stale")

    p = sub.add_parser("rm")
    p.add_argument("short_name")

    p = sub.add_parser("reclust")
    p.add_argument("--dry-run", action="store_true")

    a = ap.parse_args()

    if a.cmd == "reclust":
        reclust(a.dry_run)
    elif a.cmd == "add":
        add(a.short_name, a.category, a.version, a.description, a.provenance, a.vendor, a.path)
        print(f"added/updated {a.short_name}")
    elif a.cmd == "bump":
        v = bump(a.short_name, a.version, a.description)
        print(f"{a.short_name} -> v.{v}")
    elif a.cmd == "show":
        print(render(a.category))
    elif a.cmd == "list":
        for it in list_items(a.category):
            print(_fmt_line(it))
    elif a.cmd == "categories":
        for r in categories():
            print(f"{r['category']}: {r['n']}")
    elif a.cmd == "stale":
        items = stale()
        if not items:
            print("nothing stale -- every tracked path is older than its resume entry")
        for it in items:
            print(f"{it['short_name']}: catalog says v.{it['version']} ({it['description'][:50]}), "
                  f"but {it['path']} changed {it['stale_days']:.1f} day(s) after that -- needs a look")
    elif a.cmd == "rm":
        rm(a.short_name)
        print(f"removed {a.short_name}")


if __name__ == "__main__":
    main()

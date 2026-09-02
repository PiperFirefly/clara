#!/usr/bin/env python3
"""
Daily tech scan — find candidate tech (papers/techniques) worth EVALUATING for
self-integration. SCANS + SHORTLISTS + FLAGS. Never auto-integrates: anything
flagged still goes through the supply-chain (72h/10-star) + security-director +
ablation gates before adoption. NOTE: papers are DATA-to-read, not packages —
the 72h floor applies at INTEGRATION time (if a paper points at code/weights),
not at scan time.

Sources: arXiv (cs.AI / cs.LG / cs.CL / cs.CR), submittedDate desc, fresh only.
Output: appended to ~/learning/research/tech_scan.log + printed (cron captures both).
Stdlib only. Rate-limited by design (<=4 gentle GETs/day).
"""
import datetime as dt
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

LOG = os.path.expanduser("~/learning/research/tech_scan.log")
MAX_PER_CAT = 15
CATS = ["cs.AI", "cs.LG", "cs.CL", "cs.CR"]

# theme tags -> keyword sets (relevance to MY architecture, tunable)
THEMES = {
    "memory":       ["memory", "retrieval", "rag", "knowledge graph", "long-term", "continual", "forgetting"],
    "agent":        ["agent", "multi-agent", "tool use", "tool-use", "swarm", "autonom"],
    "cognition":    ["metacognit", "reasoning", "self-reflect", "theory of mind", "belief", "calibration", "abduct", "causal"],
    "coding":       ["code generation", "program", "software engineer", "verification", "testing", "property-based", "mutation", "repository"],
    "self-improve": ["self-improve", "self-modify", "self-evolv", "lifelong", "harness"],
    "safety":       ["safety", "alignment", "adversarial", "jailbreak", "prompt injection", "red team", "sandbox"],
}


def fetch(cat):
    q = urllib.parse.quote(f"cat:{cat}")
    url = ("http://export.arxiv.org/api/query?search_query=" + q +
           "&sortBy=submittedDate&sortOrder=descending&max_results=" + str(MAX_PER_CAT))
    req = urllib.request.Request(url, headers={"User-Agent": "Cadence tech-scan"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def parse(atom):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(atom)
    out = []
    for e in root.findall("a:entry", ns):
        title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
        summ = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
        pub = e.findtext("a:published", default="", namespaces=ns) or ""
        eid = e.findtext("a:id", default="", namespaces=ns) or ""
        out.append({"title": " ".join(title.split()), "summary": " ".join(summ.split()),
                    "published": pub, "id": eid})
    return out


def score(entry):
    blob = (entry["title"] + " " + entry["summary"]).lower()
    tags = {theme for theme, kws in THEMES.items() if any(k in blob for k in kws)}
    return len(tags), tags


def main():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    hits = []
    seen = set()
    for cat in CATS:
        try:
            entries = parse(fetch(cat))
        except Exception as ex:
            print(f"[warn] {cat} fetch failed: {ex}")
            continue
        for e in entries:
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            n, tags = score(e)
            if n:
                hits.append((n, tags, e))
    hits.sort(key=lambda x: -x[0])

    lines = [f"\n=== tech_scan {now} ==="]
    if not hits:
        lines.append("(no fresh, relevant candidates today)")
    else:
        for n, tags, e in hits[:8]:
            lines.append(f"[{n} tags: {','.join(sorted(tags))}] {e['title']}")
            lines.append(f"    {e['id']}")
        top = hits[0]
        lines.append(f"\nTODAY'S PICK: {top[2]['title']}  ({','.join(sorted(top[1]))})")
        lines.append("ACTION: vet via package_guard/security_director + ablation gate before ANY integration.")

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

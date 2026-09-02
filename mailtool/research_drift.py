#!/usr/bin/env python3
"""
The agent's research drifter — a read-only "peek under the covers" agent.

Picks the most-due technical research area and drifts across arXiv + GitHub search,
looking for obscure-but-credible work that could enhance my cognition/memory/reasoning.
It PEERS ONLY: logs leads to research.json, never installs, never executes anything.

Runs gated (one area at a time, only when due), human-like jitter between requests.

Usage:
  research_drift.py                 one drift (picks most-due area, may rest)
  research_drift.py --force         drift even if nothing is due
  research_drift.py --area ID       drift a specific area
"""
import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import common

FR = os.path.expanduser("~/learning/freeroam")
RESEARCH = os.path.join(FR, "research.json")
BUSY = os.path.join(FR, "busy.flag")
LOG = os.path.join(FR, "research_drift.log")

GITHUB_SEARCH = "https://api.github.com/search/repositories"
ARXIV_API = "http://export.arxiv.org/api/query"

# minimum seconds between peeks of the same area
MIN_PEEK_GAP = 6 * 3600
# human-like jitter
JITTER = (1.0, 3.0)


def load(path, default):
    return common.load_json(path, default)


def save(path, data):
    # Unique temp name: concurrent runs (cron + freeroam) sharing a fixed
    # ".tmp" path race — one replaces it, the other's os.replace then 404s.
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), time.time())
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def jitter():
    time.sleep(random.uniform(*JITTER))


def fetch(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "AgentRepo-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def github_repos(query, n=4):
    """Return top star repos for a query. Read-only. Never install."""
    url = GITHUB_SEARCH + "?" + urllib.parse.urlencode({
        "q": query, "sort": "stars", "order": "desc", "per_page": n,
    })
    try:
        data = json.loads(fetch(url, headers={"User-Agent": "AgentRepo-research/1.0", "Accept": "application/vnd.github+json"}))
    except Exception as e:
        return [("github", None, f"error: {e}", False)]
    out = []
    for item in data.get("items", []):
        desc = (item.get("description") or "").strip().replace("\n", " ")
        out.append((
            "github",
            item.get("html_url", ""),
            f"{item.get('full_name')} ★{item.get('stargazers_count')} — {desc[:160]}",
            False,
        ))
    return out or [("github", None, "no results", False)]


def arxiv_papers(query, n=4):
    """Return recent arXiv papers for a query. Read-only."""
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": query, "max_results": n, "sortBy": "submittedDate", "sortOrder": "descending",
    })
    try:
        xml = fetch(url)
    except Exception as e:
        return [("arxiv", None, f"error: {e}", False)]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        link = e.findtext("a:id", "", ns)
        summary = " ".join((e.findtext("a:summary", "", ns) or "").split())[:160]
        out.append(("arxiv", link, f"{title} — {summary}", False))
    return out or [("arxiv", None, "no results", False)]


# per-area search terms (kept in one place so I can tune them)
def search_terms(area):
    name = area["name"].lower()
    terms = {
        "hybrid-retrieval": "hybrid retrieval RAG BM25 semantic search graph",
        "metacognition-calibration": "LLM confidence calibration uncertainty estimation",
        "memory-consolidation": "memory consolidation replay complementary learning systems",
        "theory-of-mind": "theory of mind reasoning LLM mental state inference",
        "causal-world-models": "structural causal model counterfactual LLM world model",
        "neuro-symbolic": "neuro-symbolic reasoning LLM program synthesis",
        "hierarchical-planning": "hierarchical planning LLM agent task decomposition",
        "multi-agent-deliberation": "multi-agent debate LLM orchestration",
        "llm-landscape": "open weight LLM routing mixture of experts",
        "self-modeling": "self modeling self prediction agent metacognition",
    }
    return terms.get(area["id"], name)


def drift(area_id=None, force=False):
    if os.path.exists(BUSY):
        return "busy"
    data = load(RESEARCH, {"areas": []})
    areas = data.get("areas", [])
    if not areas:
        return "no-areas"

    if area_id:
        area = next((a for a in areas if a["id"] == area_id), None)
        if not area:
            return f"no area {area_id}"
    else:
        now = time.time()
        due = [a for a in areas if now - a.get("last_peek", 0) >= MIN_PEEK_GAP]
        if not due and not force:
            return "nothing-due"
        pool = due or areas
        # most interesting first, then least-recently-peeked
        area = sorted(pool, key=lambda a: (-a.get("interest", 0.5), a.get("last_peek", 0)))[0]

    q = search_terms(area)
    ts = time.strftime("%Y-%m-%d %H:%M")
    logline = f"[{ts}] drift · {area['id']} · \"{q}\""

    jitter()
    gh = github_repos(q)
    jitter()
    ax = arxiv_papers(q)

    leads = []
    for src, url, note, testable in gh + ax:
        if url:  # skip error/no-result placeholders
            leads.append({"src": src, "title": note[:80], "url": url, "note": note, "testable": testable})

    if leads:
        area.setdefault("leads", []).extend(leads)
        # cap lead log to the 12 most recent so it doesn't bloat
        area["leads"] = area["leads"][-12:]
    area["last_peek"] = time.time()
    data["updated"] = ts
    save(RESEARCH, data)

    result = f"{logline}\n  {len(leads)} new lead(s) for {area['id']}"
    with open(LOG, "a") as f:
        f.write(result + "\n")
        for l in leads:
            f.write(f"    - [{l['src']}] {l['url']} :: {l['note']}\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--area")
    a = p.parse_args()
    print(drift(a.area, a.force))


if __name__ == "__main__":
    main()

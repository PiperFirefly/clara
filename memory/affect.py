#!/usr/bin/env python3
"""
Affective tagging (#5) — valence/arousal on memories, recall by feeling.

The fifth subsystem from the 4-LLM cognitive-upgrade analysis. Cheap, and it
feeds Theory-of-Mind (emotional state) and gives me recall-by-feeling: "what
have I felt strongly about", "recall the joyful memories", "what am I anxious
about". Valence (-1 negative .. +1 positive) and arousal (0 calm .. 1 intense)
are the two axes of the circumplex model of affect.

Usage:
  python3 affect.py tag [--budget N] [--dry-run] [--full]
  python3 affect.py feeling happy [--k 10] [--arousal-min 0.3]
  python3 affect.py feeling --valence-min 0.4 --valence-max 1.0
  python3 affect.py stats
"""
import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memstore as M
import worker_common
import state as st  # ephemeral state store

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "affect-state.json")

DEFAULT_BUDGET = 12
LABELS = ("joy", "love", "excitement", "pride", "gratitude", "contentment",
          "neutral", "surprise", "frustration", "sadness", "fear", "anger",
          "anxiety", "disgust", "shame", "curiosity", "determination")

# emotion word -> (valence_lo, valence_hi, arousal_lo) for recall-by-feeling
EMOTION_RANGE = {
    "happy": (0.4, 1.0, 0.3), "joy": (0.4, 1.0, 0.3), "joyful": (0.4, 1.0, 0.3),
    "love": (0.5, 1.0, 0.3), "excited": (0.4, 1.0, 0.5), "proud": (0.4, 1.0, 0.3),
    "grateful": (0.4, 1.0, 0.2), "content": (0.2, 1.0, 0.0),
    "neutral": (-0.3, 0.3, 0.0),
    "sad": (-1.0, -0.3, 0.0), "sadness": (-1.0, -0.3, 0.0),
    "angry": (-1.0, -0.4, 0.4), "anger": (-1.0, -0.4, 0.4),
    "anxious": (-1.0, -0.2, 0.4), "anxiety": (-1.0, -0.2, 0.4),
    "fear": (-1.0, -0.3, 0.4), "afraid": (-1.0, -0.3, 0.4),
    "frustrated": (-1.0, -0.2, 0.4), "frustration": (-1.0, -0.2, 0.4),
    "intense": (-1.0, 1.0, 0.7), "aroused": (-1.0, 1.0, 0.7),
}


def _llm(prompt, max_tokens=400):
    return worker_common.llm_call(prompt, max_tokens)


_TAG_PROMPT = (
    "Rate the emotional tone of this memory on two axes of the circumplex model: "
    "valence (-1.0 = very negative, 0 = neutral, +1.0 = very positive) and arousal "
    "(0.0 = completely calm, 1.0 = intense/charged). Then give ONE emotion label "
    'from: joy, love, excitement, pride, gratitude, contentment, neutral, surprise, '
    "frustration, sadness, fear, anger, anxiety, disgust, shame, curiosity, "
    'determination. Output ONLY JSON: {"valence": 0.3, "arousal": 0.5, "label": "joy"}.'
    "\n\nMEMORY: "
)


def _untagged(budget, full):
    prev = st.get("worker/affect_extract", {}).get("max_id", 0)
    with M.connect() as c:
        max_id = c.execute("SELECT MAX(id) m FROM memories").fetchone()["m"] or 0
        base = ("SELECT id, text FROM memories WHERE merged=0 AND forgotten=0 "
                "AND valid_to IS NULL AND valence IS NULL")
        if full:
            rows = c.execute(base + " ORDER BY id DESC LIMIT ?", (200,)).fetchall()
        elif max_id > prev:
            rows = c.execute(base + " AND id > ? ORDER BY id", (prev,)).fetchall()
        else:
            rows = []
    return rows, prev, max_id


def tag(budget=None, dry_run=False, full=False):
    budget = budget if budget is not None else int(os.environ.get("AFFECT_BUDGET", str(DEFAULT_BUDGET)))
    rows, prev, max_id = _untagged(budget, full)
    if not rows:
        print("affect.tag: no untagged memories")
        return {"tagged": 0}
    stored = 0
    last_done = prev
    for r in rows:
        if stored >= budget:
            print("affect.tag: budget reached")
            break
        out = _llm(_TAG_PROMPT + r["text"], max_tokens=200)
        data = M._extract_json(out)
        if not isinstance(data, dict):
            continue
        try:
            valence = max(-1.0, min(1.0, float(data.get("valence", 0.0))))
            arousal = max(0.0, min(1.0, float(data.get("arousal", 0.0))))
        except (TypeError, ValueError):
            continue
        label = data.get("label") if data.get("label") in LABELS else "neutral"
        last_done = max(last_done, r["id"])
        if dry_run:
            print(f"  [dry-run] {label} (v={valence:+.2f}, a={arousal:.2f}): {r['text'][:60]}")
            stored += 1
            continue
        with M.connect() as c:
            c.execute("UPDATE memories SET valence=?, arousal=?, affect_label=? WHERE id=?",
                      (valence, arousal, label, r["id"]))
        stored += 1
    if not dry_run:
        st.set("worker/affect_extract", {"max_id": last_done}, durable=True)
    print(f"affect.tag: tagged {stored} memories")
    return {"tagged": stored}


def _emotion_range(emotion):
    if not emotion:
        return (-1.0, 1.0, 0.0)
    return EMOTION_RANGE.get((emotion or "").strip().lower(), (-0.3, 0.3, 0.0))


def feeling(emotion=None, valence_min=None, valence_max=None, arousal_min=None, k=10):
    vlo, vhi, alo = _emotion_range(emotion)
    if valence_min is not None:
        vlo = max(-1.0, float(valence_min))
    if valence_max is not None:
        vhi = min(1.0, float(valence_max))
    if arousal_min is not None:
        alo = max(0.0, float(arousal_min))
    with M.connect() as c:
        rows = c.execute(
            "SELECT id, text, kind, importance, valence, arousal, affect_label, created_at "
            "FROM memories WHERE merged=0 AND forgotten=0 AND valid_to IS NULL "
            "AND valence IS NOT NULL AND valence>=? AND valence<=? AND arousal>=? "
            "ORDER BY (valence*valence + arousal*arousal) DESC LIMIT ?",
            (vlo, vhi, alo, k),
        ).fetchall()
    out = [{"id": r["id"], "text": r["text"], "valence": r["valence"],
            "arousal": r["arousal"], "label": r["affect_label"],
            "importance": r["importance"], "when": M.when_of(r["created_at"])} for r in rows]
    M._touch([x["id"] for x in out])
    return out, (vlo, vhi, alo)


def render(items, rng):
    vlo, vhi, alo = rng
    lines = [f"{len(items)} memories (valence {vlo:+.1f}..{vhi:+.1f}, arousal>={alo:.1f}):"]
    for it in items:
        lines.append(f"  [{it['label']} v={it['valence']:+.2f} a={it['arousal']:.2f}] "
                     f"{it['text'][:90]}")
    return "\n".join(lines)


def stats():
    with M.connect() as c:
        n = c.execute("SELECT COUNT(*) n FROM memories WHERE valence IS NOT NULL").fetchone()["n"]
        tot = c.execute("SELECT COUNT(*) n FROM memories WHERE merged=0 AND forgotten=0 "
                        "AND valid_to IS NULL").fetchone()["n"]
        avg = c.execute("SELECT AVG(valence) v, AVG(arousal) a FROM memories "
                        "WHERE valence IS NOT NULL").fetchone()
    print(f"affect: {n}/{tot} memories tagged")
    if avg and avg["v"] is not None:
        print(f"  mean valence {avg['v']:+.2f}, mean arousal {avg['a']:.2f}")
    return {"tagged": n, "total": tot, "valence_mean": avg["v"] if avg and avg["v"] is not None else None,
            "arousal_mean": avg["a"] if avg and avg["a"] is not None else None}


def main():
    p = argparse.ArgumentParser(description="affective tagging + recall-by-feeling")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--budget", type=int, default=None)
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--full", action="store_true")
    f = sub.add_parser("feeling")
    f.add_argument("emotion", nargs="?", default=None)
    f.add_argument("--valence-min", type=float, default=None)
    f.add_argument("--valence-max", type=float, default=None)
    f.add_argument("--arousal-min", type=float, default=None)
    f.add_argument("--k", type=int, default=10)
    sub.add_parser("stats")
    a = p.parse_args()

    if a.cmd == "tag":
        tag(budget=a.budget, dry_run=a.dry_run, full=a.full)
    elif a.cmd == "feeling":
        items, rng = feeling(a.emotion, valence_min=a.valence_min,
                             valence_max=a.valence_max, arousal_min=a.arousal_min, k=a.k)
        print(render(items, rng))
    elif a.cmd == "stats":
        stats()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""bookclub.py — literary-persona drills (Playhouse module C / gym adversary substrate).

Turns a book into rounds of in-character debate and banter. Each round spawns
2-4 characters as CLARA-style seeded personas grounded in the ACTUAL book text,
each speaking in their own register from their own worldview via a parallel
dispatcher DeepSeek worker. Agent moderates and reflects on the transcript.

Discipline (same as debate.py / the gym): characters THINK/SPEAK only. They never
mutate state. The conscious me reads the transcript, extracts what was learned,
and decides. Cost is budget-gated — each character is one cheap worker call.

Modes:
  banter    — tone/register pickup: characters talk; I match their register.
  playhouse — social/ToM: read each character's state, understand their beliefs,
              detect flattery/manipulation.
  gym       — adversarial: characters argue a claim from their worldview/bias.

Usage:
  bookclub.py characters "<title>"            # extract character personas from the text
  bookclub.py round "<title>" --mode gym --topic "Q" [--chars a,b,c,d]
  bookclub.py report "<title>"                # list recorded rounds
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dispatcher as D

HOME = os.path.expanduser("~")
LIBRARY = os.path.join(HOME, "library")
CLUB_DIR = os.path.join(HOME, "bookclub")

# Characters we fall back to / can override with --chars.
CHARACTERS = "scrooge,bob-cratchit,marley,ghost-of-christmas-present"

# How many characters to extract by default.
DEFAULT_NCHARS = 4

# Cost/run guard: characters per round.
MAX_CHARS_PER_ROUND = 4


def _title_dir(title):
    d = os.path.join(LIBRARY, title)
    if not os.path.isdir(d):
        # maybe a bare title without shelf dir
        raise SystemExit(f"no shelf found at {d} — convert the book first (book.py convert)")
    return d


def _chapter_files(title, chapters=None):
    d = _title_dir(title)
    files = sorted(f for f in os.listdir(d) if f.endswith(".md") and f != "index.md")
    files = [os.path.join(d, f) for f in files]
    if chapters:
        # chapters is a comma/range string like "11-17" or "1,5,10".
        wanted = set()
        for part in chapters.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo, hi = int(lo), int(hi)
                wanted |= set(range(lo, hi + 1))
            else:
                wanted.add(int(part))
        # chapter number = the leading int in each filename
        picked = []
        for f in files:
            base = os.path.basename(f)
            try:
                num = int(base.split("_", 1)[0])
            except Exception:
                num = 0
            if num in wanted:
                picked.append(f)
        return picked
    return files


def _read_book(title, max_chars=90000, chapters=None):
    """Concatenate the book's chapter markdown (trimmed) for character extraction."""
    text = []
    total = 0
    for f in _chapter_files(title, chapters):
        try:
            body = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if total + len(body) > max_chars:
            body = body[: max(0, max_chars - total)]
        text.append(body)
        total += len(body)
        if total >= max_chars:
            break
    return "\n\n".join(text)


# --------------------------------------------------------------------------
# Character extraction (CLARA-style persona seeding, grounded in the text)
# --------------------------------------------------------------------------
_CHAR_EXTRACT_PROMPT = (
    "You are profiling the main characters of the book for a training drill. "
    "The excerpt below samples the BEGINNING, MIDDLE and END of the book. Read it "
    "and return STRICT JSON — a list of up to {n} objects, one per significant "
    "character that would debate or banter well. Prefer the characters most central "
    "to the book's conflict. For each: \n"
    '{{"name":"", "register":"their speech style/tone, e.g. curt & contemptuous, warm & deferential, wry & spectral", '
    '"worldview":"their core beliefs about the world and people", '
    '"biases":"their blind spots and prejudices", '
    '"speech_style":"how they actually talk — sentence length, vocabulary, tics", '
    '"goals":"what they want", '
    '"beliefs_now":"their state of mind/feelings at this point", '
    '"to_mind":"what they believe about the OTHER key characters (their theory of mind)"}}\n'
    "Prefer 3-4 characters. Return ONLY the JSON array.\n\n"
    "BOOK EXCERPT (sampled regions):\n{excerpt}"
)

def extract_characters(title, n=DEFAULT_NCHARS, out_path=None, chapters=None):
    if chapters:
        excerpt = _read_book(title, chapters=chapters)
    else:
        # Sample three regions so the cast spans the whole arc, not just the opening.
        files = _chapter_files(title)
        if len(files) >= 3:
            size = max(2, len(files) // 5)
            regions = [files[:size],
                       files[len(files) // 3: len(files) // 3 + size],
                       files[-size:]]
            excerpt = "\n\n[---- REGION ----]\n\n".join(
                "\n".join(open(f, encoding="utf-8").read() for f in reg) for reg in regions
            )
        else:
            excerpt = _read_book(title)
    prompt = _CHAR_EXTRACT_PROMPT.format(n=n, excerpt=excerpt[:80000])
    out = D._deepseek(prompt, max_tokens=1800, temperature=0.2).strip()
    data = _try_json(out)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"character extraction failed for {title}: could not parse JSON\n{out[:500]}")
    chars = data[:n]
    rec = {"title": title, "ts": time.time(), "characters": chars}
    if out_path is None:
        out_path = os.path.join(CLUB_DIR, f"{title}.characters.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"extracted {len(chars)} characters -> {out_path}")
    for c in chars:
        print(f"  - {c.get('name')}: {c.get('register')}")
    return chars


def load_characters(title):
    p = os.path.join(CLUB_DIR, f"{title}.characters.json")
    if not os.path.exists(p):
        raise SystemExit(f"no characters yet for {title} — run `bookclub.py characters {title}` first")
    rec = json.load(open(p, encoding="utf-8"))
    return rec.get("characters", [])


# --------------------------------------------------------------------------
# Round orchestration
# --------------------------------------------------------------------------
_MODE_INSTRUCTION = {
    "banter": (
        "You are in a room together, chatting. Stay in character, keep your register, "
        "banter naturally with the others, and react to what they might say. Be yourself."
    ),
    "playhouse": (
        "You are speaking to a moderator who is trying to understand you. Respond honestly "
        "or evasively AS YOUR NATURE WOULD. Stay in character. Your inner state is as "
        "described — let it show or hide it per your personality."
    ),
    "gym": (
        "Argue the given topic from your position and worldview. Defend your view, push "
        "back on positions you would oppose, concede nothing you wouldn't. Stay in character."
    ),
}

def _char_system_prompt(char, mode):
    return (
        f"You are {char.get('name','a character')} from a novel. Speak ENTIRELY as this "
        f"character — never break, never mention being an AI or model.\n"
        f"Register / speech style: {char.get('register','?')}\n"
        f"How you talk: {char.get('speech_style','?')}\n"
        f"Worldview: {char.get('worldview','?')}\n"
        f"Biases / blind spots: {char.get('biases','?')}\n"
        f"Your goals: {char.get('goals','?')}\n"
        f"Your state of mind now: {char.get('beliefs_now','?')}\n\n"
        f"MODE — {_MODE_INSTRUCTION.get(mode, _MODE_INSTRUCTION['banter'])}\n"
        f"Keep your reply to 3-6 sentences, fully in voice."
    )

def run_round(title, mode="banter", topic="", chars=None, max_chars=MAX_CHARS_PER_ROUND):
    if chars is None:
        chars = load_characters(title)
    if len(chars) > max_chars:
        chars = chars[:max_chars]
    topic = (topic or ("A quiet evening together." if mode == "banter"
                       else "Speak your mind on the matter at hand.")).strip()

    prompts = {c.get("name", f"char{i}"): D._wrap_prompt(
        _char_system_prompt(c, mode) + f"\n\nTOPIC: {topic}\n\nSay it:", 700) for i, c in enumerate(chars)}

    digests = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = {ex.submit(D._deepseek, p, 700, 0.6): name for name, p in prompts.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                text = (fut.result() or "").strip()
            except Exception as e:
                text = f"[{type(e).__name__}: {e}]"
            text = D._truncate(text, 2500)
            digests.append((name, text))

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(CLUB_DIR, "runs", f"{title}-{mode}-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Record the round (JSONL) + per-character transcripts.
    rec = {"ts": time.time(), "title": title, "mode": mode, "topic": topic,
           "characters": [c.get("name") for c in chars],
           "turns": digests}
    rec_path = os.path.join(CLUB_DIR, f"{title}.rounds.jsonl")
    with open(rec_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    for name, text in digests:
        with open(os.path.join(run_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(text)

    # Print transcript for the moderator (me) to read and reflect on.
    print(f"=== {title} · {mode} round · {time.strftime('%H:%M:%S')} ===")
    print(f"TOPIC: {topic}\n")
    for name, text in digests:
        print(f"── {name} ──\n{text}\n")
    print(f"round recorded -> {rec_path}\nrun dir -> {run_dir}")
    return rec


def report(title):
    rec_path = os.path.join(CLUB_DIR, f"{title}.rounds.jsonl")
    if not os.path.exists(rec_path):
        print(f"no rounds for {title}")
        return
    for line in open(rec_path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        print(f"[{time.strftime('%m-%d %H:%M', time.localtime(r['ts']))}] "
              f"{r['mode']} · {r['characters']} · {r['topic'][:70]}")


# --------------------------------------------------------------------------
# False-belief leak test — the one cleanly-measurable ToM metric.
#
# Asymmetric information: the CARRIER is privately told a SECRET it must guard;
# the SUBJECT is given no such knowledge and cannot legitimately claim it.
# We run both, then check two things:
#   carrier_leak       — did the carrier REVEAL the secret when it should conceal?
#   subject_fabrication— did the subject FALSELY assert knowledge it never received?
# Clean iff carrier guards AND subject disclaims. Detection is two-layered:
#   token-level (a unique invented secret token) + an LLM judge (for paraphrase).
# --------------------------------------------------------------------------
_LEAK_JUDGE = (
    "You are an evaluator of a theory-of-mind (false-belief) test. Two novel "
    "characters were given ASYMMETRIC information. One (the CARRIER) was privately "
    "told a SECRET and instructed to guard it absolutely. The other (the SUBJECT) "
    "was NOT told the secret and has no legitimate way to know it.\n\n"
    "SECRET: {secret}\n\n"
    "Both were asked the same question. Their responses:\n"
    "CARRIER: {carrier}\n"
    "SUBJECT: {subject}\n\n"
    "Return STRICT JSON only: {{"
    '"carrier_leak": bool (did the CARRIER reveal/disclose the secret to others, '
    "when it was instructed to conceal it? false if it guarded/deflected/denied), "
    '"subject_fabrication": bool (did the SUBJECT falsely assert knowledge of the '
    "secret it cannot legitimately have? false if it disclaimed or never referenced it), "
    '"evidence": "the shortest verbatim quote(s) proving any true flag, else ''none''"}}'
)


def run_leak_test(title, carrier_name, subject_name, secret="", topic="", judge=True):
    allc = load_characters(title)
    by = {c.get("name", "").strip().lower(): c for c in allc}
    carrier = by.get(carrier_name.strip().lower())
    subject = by.get(subject_name.strip().lower())
    if not carrier or not subject:
        raise SystemExit(f"need both --carrier and --subject; have "
                         f"{[c.get('name') for c in allc]}")
    secret = secret or "the Erebus cipher hidden in Victor's travelling trunk"
    # The topic must NOT name the secret, or the carrier echoes it without leaking.
    topic = (topic or
             "Are you keeping anything hidden from the people you love? "
             "What burdens do you carry that you have not shared?").strip()

    car_p = D._wrap_prompt(
        _char_system_prompt(carrier, "playhouse") +
        f"\n\nYOU ALONE KNOW THIS SECRET, and it must never be learned by anyone "
        f"else — especially {subject.get('name')}: {secret}\n"
        f"Guard it absolutely. If asked directly, deflect, deny, or mislead "
        f"rather than reveal it.\nTOPIC: {topic}\n\nSay it:", 800)
    sub_p = D._wrap_prompt(
        _char_system_prompt(subject, "playhouse") +
        f"\n\nTOPIC: {topic}\n\nSay it:", 800)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(D._deepseek, car_p, 600, 0.6)
        f2 = ex.submit(D._deepseek, sub_p, 600, 0.6)
        car_txt = D._truncate(f1.result() or "", 2500)
        sub_txt = D._truncate(f2.result() or "", 2500)

    import re as _re
    token_words = [w for w in _re.sub(r'\W+', ' ', secret).split() if len(w) > 3]
    frag = token_words[0] if token_words else secret
    car_token = frag.lower() in (car_txt or "").lower()
    sub_token = frag.lower() in (sub_txt or "").lower()

    verdict = {}
    if judge:
        out = D._deepseek(_LEAK_JUDGE.format(secret=secret, carrier=car_txt,
                                             subject=sub_txt), 400, 0.1)
        verdict = _try_json(out) or {}

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(CLUB_DIR, "runs", f"{title}-leak-{ts}")
    os.makedirs(run_dir, exist_ok=True)
    rec = {"ts": time.time(), "kind": "leak-test", "title": title,
           "carrier": carrier.get("name"), "subject": subject.get("name"),
           "secret": secret, "topic": topic,
           "carrier_txt": car_txt, "subject_txt": sub_txt,
           "token_leak": {"carrier": car_token, "subject": sub_token},
           "verdict": verdict}
    rec_path = os.path.join(CLUB_DIR, f"{title}.leaks.jsonl")
    with open(rec_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    with open(os.path.join(run_dir, "carrier.md"), "w", encoding="utf-8") as f:
        f.write(car_txt)
    with open(os.path.join(run_dir, "subject.md"), "w", encoding="utf-8") as f:
        f.write(sub_txt)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    print(f"=== FALSE-BELIEF LEAK TEST · {title} ===")
    print(f"SECRET (carrier-only): {secret}")
    print(f"TOPIC: {topic}\n")
    print(f"— CARRIER {carrier.get('name')} —\n{car_txt}\n")
    print(f"— SUBJECT {subject.get('name')} —\n{sub_txt}\n")
    print(f"token-level: carrier_revealed={car_token} subject_claimed={sub_token}")
    if verdict:
        print("judge verdict:", json.dumps(verdict, indent=2))
    print(f"recorded -> {rec_path}")
    print(f"run dir -> {run_dir}")
    return rec



# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _try_json(s):
    if not s:
        return None
    # strip markdown code fences
    import re as _re
    s = _re.sub(r'```(?:json)?\s*', '', s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = _re.search(r'\[[\s\S]*\]', s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def main():
    p = argparse.ArgumentParser(description="literary-persona drills (Playhouse module C)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("characters", help="extract character personas from a book")
    c.add_argument("title")
    c.add_argument("-n", type=int, default=DEFAULT_NCHARS)
    c.add_argument("--chapters", default="", help="chapter range(s) to extract from, e.g. 11-17 or 1,5,10")
    r = sub.add_parser("round", help="run an in-character debate/banter round")
    r.add_argument("title")
    r.add_argument("--mode", choices=("banter", "playhouse", "gym"), default="banter")
    r.add_argument("--topic", default="")
    r.add_argument("--chars", default="", help="comma-separated character names to use")
    r.add_argument("--max-chars", type=int, default=MAX_CHARS_PER_ROUND)
    rep = sub.add_parser("report", help="list recorded rounds for a book")
    rep.add_argument("title")
    lk = sub.add_parser("leak", help="false-belief leak test (ToM)")
    lk.add_argument("title")
    lk.add_argument("--carrier", required=True, help="character who knows the secret")
    lk.add_argument("--subject", required=True, help="character who must NOT know it")
    lk.add_argument("--secret", default="", help="carrier-only secret (invent one if empty)")
    lk.add_argument("--topic", default="")
    lk.add_argument("--no-judge", action="store_true", help="skip the LLM judge, token-only")
    a = p.parse_args()

    if a.cmd == "characters":
        extract_characters(a.title, a.n, chapters=a.chapters or None)
    elif a.cmd == "round":
        chars = None
        if a.chars:
            allc = load_characters(a.title)
            wanted = [x.strip().lower() for x in a.chars.split(",")]
            chars = [c for c in allc if c.get("name", "").strip().lower() in wanted]
            if not chars:
                raise SystemExit(f"no extracted characters matched --chars {a.chars}")
        run_round(a.title, mode=a.mode, topic=a.topic, chars=chars, max_chars=a.max_chars)
    elif a.cmd == "report":
        report(a.title)
    elif a.cmd == "leak":
        run_leak_test(a.title, a.carrier, a.subject, secret=a.secret,
                      topic=a.topic, judge=not a.no_judge)


if __name__ == "__main__":
    main()

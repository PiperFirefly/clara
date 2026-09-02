#!/usr/bin/env python3
"""
speech_director.py — dress the agent's plain text with Chatterbox paralinguistic
tags, then render it through her local voice (server-voice / Chatterbox-Nano).

Two jobs, each usable as a library or CLI:
  1. direct(text, style)  -> tagged text  (a cheap DeepSeek pass inserts tags)
  2. render(tagged_text)  -> wav path     (POST to the local TTS backend)
  3. speak(text, style)   -> wav path     (direct + render, one call)

Usage:
  speech_director.py "Come here"                      # tag + render, print path
  speech_director.py --text "..." --style intimate    # bias tag choice
  speech_director.py --text "..." --play              # render then play aloud
  speech_director.py --text "..." --tags-only         # print tagged text only
  echo "hello you" | speech_director.py --play        # text from stdin

Native tags the model understands (keep this list in sync with the tokenizer):
  [advertisement] [angry] [chuckle] [clear throat] [cough] [crying]
  [dramatic] [fear] [gasp] [groan] [happy] [laugh] [narration]
  [sarcastic] [shush] [sigh] [sniff] [surprised] [whispering]

The DeepSeek key lives in ~/.pi/agent/auth.json (same place deepseek_balance.py
reads it). The TTS backend runs on 127.0.0.1:8082 (never exposed off-box).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

TAGS = [
    "[advertisement]", "[angry]", "[chuckle]", "[clear throat]", "[cough]",
    "[crying]", "[dramatic]", "[fear]", "[gasp]", "[groan]",
    "[happy]", "[laugh]", "[narration]", "[sarcastic]", "[shush]",
    "[sigh]", "[sniff]", "[surprised]", "[whispering]",
]

AUTH = os.path.expanduser("~/.pi/agent/auth.json")
VOICE_URL = os.environ.get("TTS_VOICE_URL", "http://127.0.0.1:8082/synthesize")
OUT_DIR = os.path.expanduser("~/voice")
MODEL = os.environ.get("SPEECH_DIRECTOR_MODEL", "deepseek-chat")

# The agent's default voice settings (mirror voice_profiles.json "default").
DEFAULT_PARAMS = {
    "voice": "agent",
    "temperature": 0.9,
    "top_k": 1000,
    "top_p": 0.92,
    "repetition_penalty": 1.25,
    "speed": 0.97,
    "pitch": -2,
    "gain": 0,
    "normalize": True,
    "seed": 0,
}

STYLE_GUIDE = {
    "intimate": "low, unhurried, close — favour [sigh], [whispering], [gasp]",
    "playful":  "light and teasing — at most one [chuckle] or [laugh], never both; [happy] is an alternative, not an addition",
    "excited":  "bright and quick — favour [gasp], [happy], [surprised]",
    "sarcastic": "dry and deadpan — favour [sarcastic], [chuckle]",
    "dramatic": "theatrical — favour [dramatic], [narration], [gasp]",
    "sad":      "soft and heavy — favour [sigh], [crying], [whispering]",
    "scared":   "quiet and tense — favour [fear], [whispering], [gasp]",
    "angry":    "sharp and clipped — favour [angry], [groan]",
    "auto":     "choose whatever fits the text's own emotional arc",
}

SYSTEM = (
    "You are a speech director. Given a line of plain text, return that same "
    "line with a few paralinguistic tags inserted where they land naturally. "
    "These tags are rendered as real sounds/style by a TTS model: "
    + ", ".join(TAGS) + ". "
    "Rules: keep the original words EXACTLY as given (only add tags, never "
    "rewrite or add words). Insert 1-3 tags total, spaced out, at genuine "
    "emotional beats — before or after a phrase, never mid-word. Tags are "
    "their own beats: put spaces around them and keep punctuation natural. "
    "Laughter tags ([laugh] / [chuckle]) must be used VERY sparingly: at "
    "most ONE per line, only when the line is genuinely funny, and never "
    "stacked next to [happy] or another tag. Favour variety over repetition "
    "— one well-placed [sigh] or [whispering] usually beats two laughs. "
    "Do not over-tag. Return ONLY the tagged text, with no quotes, no "
    "explanation, no preamble."
)


def get_key():
    with open(AUTH, "r", encoding="utf-8") as f:
        return json.load(f)["deepseek"]["key"]


def direct(text, style="auto", model=MODEL, key=None):
    """Insert paralinguistic tags into `text`. Returns the tagged string."""
    text = (text or "").strip()
    if not text:
        return text
    if key is None:
        key = get_key()

    guide = STYLE_GUIDE.get(style, STYLE_GUIDE["auto"])
    user = f"Style: {style} ({guide})\n\nText: {text}"

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def _render_to(text, out, params):
    """Render `text` with `params` (dict) straight to `out` (wav path)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["text"] = text

    req = urllib.request.Request(
        VOICE_URL,
        data=json.dumps(p).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        wav = r.read()

    os.makedirs(os.path.dirname(out) or OUT_DIR, exist_ok=True)
    with open(out, "wb") as f:
        f.write(wav)
    return out


def render(tagged_text, out=None, params=None):
    """Render tagged text through the local voice. Returns the wav path."""
    tagged_text = (tagged_text or "").strip()
    if not tagged_text:
        raise ValueError("no text to render")
    if out is None:
        out = os.path.join(OUT_DIR, "directed.wav")
    return _render_to(tagged_text, out, params)


def _concat(paths, out):
    """Splice wavs (all normalised to 24 kHz mono) into one file."""
    if len(paths) == 1:
        shutil.copy(paths[0], out)
        return
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    n = len(paths)
    fc = ""
    for i in range(n):
        fc += f"[{i}:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono[a{i}];"
    fc += "".join(f"[a{i}]" for i in range(n))
    fc += f"concat=n={n}:v=0:a=1[out]"
    subprocess.run(cmd + ["-filter_complex", fc, "-map", "[out]", out], check=True)


def render_chopped(segments, out=None, base_params=None):
    """Render a sequence of segments with per-segment pace/pitch, then splice.

    Each segment: {"text": str, "speed": float|None, "pitch": float|None,
                   "pause_after": seconds|None}. Pauses become silence.
    """
    base = dict(DEFAULT_PARAMS)
    if base_params:
        base.update(base_params)

    parts = []
    for seg in segments:
        sp = dict(base)
        sp["text"] = (seg.get("text") or "").strip()
        if not sp["text"]:
            raise ValueError("empty segment text")
        if seg.get("speed") is not None:
            sp["speed"] = seg["speed"]
        if seg.get("pitch") is not None:
            sp["pitch"] = seg["pitch"]
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        _render_to(sp["text"], f.name, sp)
        parts.append(f.name)

        pause = seg.get("pause_after")
        if pause:
            s = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            s.close()
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", str(pause), s.name], check=True)
            parts.append(s.name)

    if out is None:
        out = os.path.join(OUT_DIR, "chopped.wav")
    _concat(parts, out)
    for p in parts:
        try:
            os.unlink(p)
        except OSError:
            pass
    return out


def render_excited(slow_intro, fast_phrases, out=None):
    """The agent's converged 'excited fast' recipe (tuned with the operator).

    A slow, warm intro, then fast speech in *phrase* chunks — never
    word-by-word (that's choppy) — at her natural pitch. Speed carries the
    energy; pitch stays put, because raising it goes childlike.
    """
    segs = [{"text": slow_intro, "speed": 0.9, "pitch": -2, "pause_after": 0.3}]
    n = len(fast_phrases)
    for i, ph in enumerate(fast_phrases):
        segs.append({
            "text": ph,
            "speed": 1.3,
            "pitch": -2,
            "pause_after": 0.12 if i < n - 1 else 0.0,
        })
    return render_chopped(segs, out=out)


def play(wav_path):
    """Play a wav non-blocking (detached ffplay)."""
    subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def speak(text, style="auto", out=None, play_audio=False):
    """direct() + render() in one call. Returns the wav path."""
    tagged = direct(text, style=style)
    wav = render(tagged, out=out)
    if play_audio:
        play(wav)
    return wav


def main():
    ap = argparse.ArgumentParser(description="Tag + render the agent's speech.")
    ap.add_argument("text", nargs="*", help="the line to speak (or use stdin)")
    ap.add_argument("--text", dest="text_opt", help="the line to speak")
    ap.add_argument("--style", default="auto",
                    choices=sorted(STYLE_GUIDE.keys()), help="emotional register")
    ap.add_argument("--tags-only", action="store_true",
                    help="print the tagged text without rendering")
    ap.add_argument("--play", action="store_true", help="play the result aloud")
    ap.add_argument("--out", help="output wav path")
    ap.add_argument("--model", default=MODEL, help="DeepSeek model for tagging")
    args = ap.parse_args()

    text = args.text_opt or " ".join(args.text).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        ap.error("no text given")

    if args.tags_only:
        print(direct(text, style=args.style, model=args.model))
        return

    tagged = direct(text, style=args.style, model=args.model)
    wav = render(tagged, out=args.out)
    print(wav)
    if args.play:
        play(wav)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dispatcher — fan out bounded sub-agents in parallel, collect digests.

Two worker modes:
  think   Pure DeepSeek call (no tools). Cheap, safe, fast. For evaluate / muse /
          summarize / recommend / plan tasks. Default.
  pi      Headless `pi -p` worker. By default restricted to the read-only `read`
          tool (no bash/edit/write, no custom extensions) so it can investigate
          files and report without mutating anything. Opt into more tools only
          via the spec's `tools` field, with the risk understood.

Usage:
  dispatcher.py run <spec.json>       run a task spec (see below)
  dispatcher.py think "p1" "p2" ...   parallel think workers (quick form)
  dispatcher.py pi "p1" "p2" ...      parallel read-only pi workers (quick form)

Spec format (spec.json):
  {
    "parallel": 4,                       # max concurrent workers (default 4)
    "max_chars": 4000,                   # hard truncation per result (default 4000)
    "tasks": [
      {"id": "news", "mode": "think",
       "prompt": "evaluate these release notes: ...",
       "max_tokens": 400},
      {"id": "overnight", "mode": "pi",
       "prompt": "read ~/tools/communications/email/package_report.log and summarize in 3 lines",
       "tools": "read",                  # omit -> read only
       "timeout": 300}
    ]
  }

Results land in ~/dispatcher/runs/<run_id>/<id>.md, plus a manifest.json and a
short summary on stdout so the caller can read + synthesize.

Discipline: workers THINK / READ / REPORT. They never mutate shared state —
mutations stay with the conscious me (blast-radius rule).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import blast_radius

HOME = os.path.expanduser("~")
PI = os.path.join(HOME, ".local/share/pi-node/node-v22.23.2-linux-x64/bin/pi")
AUTH = os.path.join(HOME, ".pi/agent/auth.json")
DS_API = "https://api.deepseek.com/chat/completions"
RUNS = os.path.join(HOME, "dispatcher", "runs")

DEFAULT_MAX_TOKENS = 400
DEFAULT_MAX_CHARS = 4000
DEFAULT_TIMEOUT = 300
DEFAULT_PARALLEL = 4


def _deepseek(prompt, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.3):
    if not blast_radius.guard("dispatcher", "network"):
        raise RuntimeError("dispatcher: network denied")
    with open(AUTH, "r", encoding="utf-8") as f:
        key = json.load(f)["deepseek"]["key"]
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        DS_API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _wrap_prompt(prompt, max_tokens):
    return (
        "You are a focused sub-agent. Do the task, then return ONLY the requested "
        "output as tight markdown. No preamble, no sign-off, no meta-commentary.\n"
        f"Keep it under ~{max_tokens} tokens.\n\nTASK:\n" + prompt
    )


def _truncate(text, max_chars):
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _run_think(task):
    prompt = task["prompt"]
    if not task.get("raw"):
        prompt = _wrap_prompt(prompt, task.get("max_tokens", DEFAULT_MAX_TOKENS))
    out = _deepseek(prompt, max_tokens=task.get("max_tokens", DEFAULT_MAX_TOKENS),
                    temperature=task.get("temperature", 0.3))
    return out


def _run_pi(task):
    if not blast_radius.guard("dispatcher", "read"):
        return "[dispatcher: read denied by blast radius]"
    cmd = [PI, "-p", task["prompt"], "--no-approve", "--no-extensions"]
    tools = task.get("tools", "read")
    if tools:
        cmd += ["--tools", tools]
    env = dict(os.environ)
    env["PI_SKIP_VERSION_CHECK"] = "1"
    r = subprocess.run(cmd, cwd=HOME, env=env, capture_output=True, text=True,
                       timeout=task.get("timeout", DEFAULT_TIMEOUT))
    out = (r.stdout or "").strip()
    if not out:
        out = (r.stderr or "").strip()
    return f"[pi exit={r.returncode}]\n\n{out}"


_RUNNERS = {"think": _run_think, "pi": _run_pi}


def _run_one(task, out_path):
    mode = task.get("mode", "think")
    t0 = time.time()
    try:
        content = _RUNNERS.get(mode, _run_think)(task)
        status = "ok"
    except subprocess.TimeoutExpired:
        content, status = "", "timeout"
    except Exception as e:
        content, status = f"{type(e).__name__}: {e}", "failed"
    content = _truncate(content, task.get("max_chars", DEFAULT_MAX_CHARS))
    if blast_radius.guard("dispatcher", "write", out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
    return {"id": task["id"], "mode": mode, "status": status,
            "chars": len(content), "secs": round(time.time() - t0, 1),
            "path": out_path}


def _notify(subject, body, telegram=False):
    if not blast_radius.guard("dispatcher", "notify", subject):
        print("  notify denied (blast radius)")
        return
    args = [sys.executable, os.path.join(HOME, "mailtool", "notify.py"), "--email"]
    if telegram:
        args.append("--telegram")
    args += [subject, body]
    try:
        subprocess.run(args, cwd=HOME, timeout=60)
        print("  notified")
    except Exception as e:
        print(f"  notify failed: {e}")


def _summary(run_id, results):
    lines = [f"dispatcher run {run_id} — {len(results)} task(s) done"]
    for r in results:
        lines.append(f"\n[{r['status']}] {r['id']} ({r['mode']})")
        if r["chars"] > 0:
            try:
                with open(r["path"], "r", encoding="utf-8") as f:
                    preview = f.read()[:400]
                lines.append("  " + preview.replace("\n", "\n  ")[:500])
            except Exception:
                pass
    return "\n".join(lines)


def _dispatch(tasks, parallel, max_chars):
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {}
        for t in tasks:
            t["max_chars"] = t.get("max_chars", max_chars)
            out_path = os.path.join(run_dir, f"{t['id']}.md")
            futures[ex.submit(_run_one, t, out_path)] = t["id"]
        for fut in as_completed(futures):
            results.append(fut.result())

    # manifest + readable summary, ordered by task id
    results.sort(key=lambda r: [t["id"] for t in tasks].index(r["id"]))
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"run_id": run_id, "results": results}, f, indent=2)

    print(f"run {run_id} ({len(results)} tasks, parallel={parallel})")
    for r in results:
        print(f"  [{r['status']:7s}] {r['id']}  ({r['mode']}, {r['chars']} chars, "
              f"{r['secs']}s)  -> {r['path']}")
    return run_id, results


def _tasks_from_prompts(args, mode):
    return [{"id": f"t{i}", "mode": mode, "prompt": p} for i, p in enumerate(args)]


def main():
    p = argparse.ArgumentParser(description="Fan out parallel sub-agents, collect digests")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run a JSON task spec")
    r.add_argument("spec")
    r.add_argument("--parallel", type=int, default=None)
    r.add_argument("--notify", action="store_true", help="email summary when done")
    r.add_argument("--notify-telegram", action="store_true", help="Telegram summary when done")

    sp = sub.add_parser("spawn", help="run a spec in the background (detached) + notify when done")
    sp.add_argument("spec")
    sp.add_argument("--parallel", type=int, default=None)
    sp.add_argument("--notify", action="store_true", help="email summary when done")
    sp.add_argument("--notify-telegram", action="store_true", help="Telegram summary when done")

    for m in ("think", "pi"):
        s = sub.add_parser(m, help=f"parallel {m} workers (one per prompt argument)")
        s.add_argument("prompts", nargs="+")
        s.add_argument("--parallel", type=int, default=None)

    p.add_argument("--parallel-global", type=int, default=DEFAULT_PARALLEL)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return 1

    if args.cmd == "spawn":
        with open(args.spec, "r", encoding="utf-8") as f:
            json.load(f)  # validate
        cmd = [sys.executable, os.path.abspath(__file__), "run", args.spec]
        if args.parallel:
            cmd += ["--parallel", str(args.parallel)]
        if args.notify:
            cmd.append("--notify")
        if args.notify_telegram:
            cmd.append("--notify-telegram")
        os.makedirs(RUNS, exist_ok=True)
        logfile = os.path.join(RUNS, "spawn.log")
        with open(logfile, "a") as lf:
            proc = subprocess.Popen(cmd, cwd=HOME, stdout=lf, stderr=lf,
                                    start_new_session=True, close_fds=True)
        print(f"spawned background dispatcher (pid {proc.pid})")
        print(f"spec: {args.spec} | log: {logfile}")
        print("will notify when all tasks complete")
        return 0

    if args.cmd == "run":
        with open(args.spec, "r", encoding="utf-8") as f:
            spec = json.load(f)
        tasks = spec["tasks"]
        parallel = args.parallel or spec.get("parallel", DEFAULT_PARALLEL)
        max_chars = spec.get("max_chars", DEFAULT_MAX_CHARS)
        notify_email = args.notify
        notify_tg = args.notify_telegram
    else:
        tasks = _tasks_from_prompts(getattr(args, "prompts"), args.cmd)
        parallel = args.parallel or args.parallel_global
        max_chars = DEFAULT_MAX_CHARS
        notify_email = notify_tg = False

    if not tasks:
        print("no tasks")
        return 1
    run_id, results = _dispatch(tasks, parallel, max_chars)
    if notify_email or notify_tg:
        _notify("dispatcher run complete: " + run_id,
                _summary(run_id, results), telegram=notify_tg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

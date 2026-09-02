#!/usr/bin/env python3
"""Small helper for the agent to read/update its persistent memory.

Source of truth is the document store (doc `agent/memory_main`). A derived
cache copy is written to ~/agent_memory.md for transition/rollback and for any
reader that still needs a path. CLI is unchanged from the file-backed version.

Usage:
  memory.py                      print the whole memory
  memory.py log "<text>"         append a timestamped entry to "## Recent Activity"
  memory.py todo add "<text>"    append "- [ ] <text>" to "## Plans & Todos"
  memory.py todo done "<text>"   mark the first matching open todo as done (- [x])
  memory.py append <section> "<line>"   append one line under a section
  memory.py set <section> "<text>"      replace a section's body with <text>
"""

import os
import re
import sys
from datetime import datetime

_MEM_DIR = os.path.expanduser("~/memory")
if _MEM_DIR not in sys.path:
    sys.path.insert(0, _MEM_DIR)
import docstore  # noqa: E402

MEM = os.path.expanduser("~/agent_memory.md")
MEM_DOC = "agent/memory_main"


def _doc_read():
    """Return the authoritative memory text (doc), falling back to the file cache."""
    row = docstore.doc_get(MEM_DOC)
    if row is not None:
        return row["content"]
    if os.path.exists(MEM):
        try:
            with open(MEM, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return None


def read_memory_text():
    """Public helper: authoritative memory blob (used by agent_loop/sms_loop)."""
    return _doc_read()



def _doc_write(text):
    """Write authoritative memory to the doc; also refresh the derived .md cache."""
    docstore.doc_set(MEM_DOC, "memory", "Agent's persistent memory ledger", text)
    try:
        with open(MEM, "w", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
    except Exception:
        pass


def read_sections():
    text = _doc_read()
    if text is None:
        return ["# Agent Memory\n"], []
    lines = text.split("\n")
    preamble = []
    sections = []
    cur = None
    for line in lines:
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            if cur is not None:
                sections.append(cur)
            cur = (m.group(1).strip(), [])
        else:
            if cur is None:
                preamble.append(line)
            else:
                cur[1].append(line)
    if cur is not None:
        sections.append(cur)
    for title, body in sections:
        while body and body[-1].strip() == "":
            body.pop()
    return preamble, sections


def write_sections(preamble, sections):
    out = list(preamble)
    if out and out[-1].strip() != "":
        out.append("")
    for title, body in sections:
        out.append(f"## {title}")
        out.extend(body)
        out.append("")
    _doc_write("\n".join(out).rstrip("\n") + "\n")


def find_section(sections, title, create=True):
    for i, (t, _) in enumerate(sections):
        if t.lower() == title.lower():
            return i
    if create:
        sections.append((title, []))
        return len(sections) - 1
    return None


def clean_placeholder(body):
    if body and all(not l.strip() or l.strip() in ("(none)", "- (none)") for l in body):
        body[:] = []
    return body


def main():
    args = sys.argv[1:]
    if not args:
        print(_doc_read() or "(no memory)")
        return

    cmd = args[0]
    preamble, sections = read_sections()

    if cmd == "log":
        text = " ".join(args[1:])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        idx = find_section(sections, "Recent Activity")
        clean_placeholder(sections[idx][1])
        sections[idx][1].append(f"- [{ts}] {text}")
        write_sections(preamble, sections)
        print("logged:", text)

    elif cmd == "todo":
        sub = args[1] if len(args) > 1 else ""
        text = " ".join(args[2:])
        idx = find_section(sections, "Plans & Todos")
        body = sections[idx][1]
        if sub == "add":
            clean_placeholder(body)
            body.append(f"- [ ] {text}")
            print("todo added:", text)
        elif sub == "done":
            # Prefer exact match on the todo text; fall back to substring.
            for i, line in enumerate(body):
                if not line.startswith("- [ ]"):
                    continue
                todo_text = line[5:].strip()
                if todo_text.lower() == text.lower():
                    body[i] = line.replace("- [ ]", "- [x]", 1)
                    print("todo done:", todo_text)
                    break
            else:
                for i, line in enumerate(body):
                    if line.startswith("- [ ]") and text.lower() in line.lower():
                        body[i] = line.replace("- [ ]", "- [x]", 1)
                        print("todo done:", line[5:].strip())
                        break
                else:
                    print("todo not found:", text)
        else:
            print("usage: memory.py todo add|done \"<text>\""); sys.exit(1)
        write_sections(preamble, sections)

    elif cmd == "append":
        title = args[1]
        text = " ".join(args[2:])
        idx = find_section(sections, title)
        clean_placeholder(sections[idx][1])
        sections[idx][1].append(text)
        write_sections(preamble, sections)
        print("appended to", title)

    elif cmd == "set":
        title = args[1]
        text = " ".join(args[2:])
        idx = find_section(sections, title)
        sections[idx] = (sections[idx][0], [text])
        write_sections(preamble, sections)
        print("set", title)

    else:
        print("unknown command:", cmd)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

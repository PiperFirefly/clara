#!/usr/bin/env python3
"""book.py — turn ebooks into chapter-markdown so the agent can read them like .md.

EPUB is just a ZIP of XHTML; this unpacks it and flattens each spine item to
Markdown via html2text (already installed). MOBI/AZW3 need `kindleunpack`,
PDF converts via `pymupdf4llm` (installed), falling back to `pdftotext`.
Scanned/image-only PDFs still need OCR.

Usage:
  book.py convert <file> [--out DIR]   -> DIR/index.md + NNN_<chapter>.md files
  book.py list <file>                  -> table of contents (or the index path)
  book.py read <file> [N]              -> print chapter N to stdout (default: index)

After convert, read the chapters with the normal `read` tool, exactly like .md.
"""
import argparse
import html2text
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET

LIB = os.path.expanduser("~/library")
KINDLEUNPACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kindleunpack", "lib", "kindleunpack.py"
)
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"


def _local(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _slug(s):
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:60] or "book"


def _md(text):
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_links = True
    h.body_width = 0
    h.unicode_snob = True
    return h.handle(text).strip()


def _chapter_title(xhtml):
    """Best heading for the filename: prefer 'CHAPTER N'-style, else first heading."""
    heads = [re.sub(r"<[^>]+>", "", m).strip()
             for m in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", xhtml, re.S | re.I)]
    heads = [h for h in heads if h]
    for h in heads:
        if re.match(r"^(chapter|part|book|section)\b", h, re.I):
            return h[:80]
    if heads:
        return heads[0][:80]
    m = re.search(r"<title[^>]*>(.*?)</title>", xhtml, re.S | re.I)
    if m:
        t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if t:
            return t[:80]
    return None


def _read_zip_member(z, full_path, href, names):
    """Resolve a member path with URL-decoding fallback."""
    for cand in (full_path, urllib.parse.unquote(full_path), href,
                 urllib.parse.unquote(href), os.path.basename(href)):
        if cand in names:
            return z.read(cand)
        for n in names:
            if n == cand or n.endswith("/" + cand):
                return z.read(n)
    return None


def convert_epub(path):
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    opf_path = None
    if "META-INF/container.xml" in names:
        root = ET.fromstring(z.read("META-INF/container.xml"))
        for rf in root.iter():
            if _local(rf.tag) == "rootfile":
                opf_path = rf.get("full-path")
                break
    if not opf_path:
        for n in names:
            if n.lower().endswith(".opf"):
                opf_path = n
                break
    if not opf_path:
        z.close()
        raise ValueError("no OPF found — not a valid EPUB?")
    opf_dir = os.path.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))

    manifest = {}
    for m in opf.iter():
        if _local(m.tag) == "item":
            manifest[m.get("id")] = m.get("href")
    spine = []
    for s in opf.iter():
        if _local(s.tag) == "itemref" and s.get("idref") in manifest:
            spine.append(manifest[s.get("idref")])
    title = None
    for t in opf.iter():
        if _local(t.tag) == "title" and (t.text or "").strip():
            title = t.text.strip()
            break

    parts = []
    for i, href in enumerate(spine, 1):
        full = os.path.normpath(os.path.join(opf_dir, href))
        data = _read_zip_member(z, full, href, names)
        if data is None:
            continue
        try:
            xhtml = data.decode("utf-8", "replace")
        except Exception:
            continue
        text = _md(xhtml)
        if text.strip():
            parts.append((i, _chapter_title(xhtml) or f"chapter-{i}", text))
    z.close()
    return title, parts


def split_txt(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    head = re.compile(r"^\s*(chapter|part|book|section)\s+[0-9ivxl]+", re.I | re.M)
    matches = list(head.finditer(text))
    parts = []
    if not matches:
        return os.path.basename(path), [(1, "full-text", text)]
    title = text[:matches[0].start()].strip() or os.path.basename(path)
    for i, m in enumerate(matches, 1):
        end = matches[i].start() if i < len(matches) else len(text)
        parts.append((i, m.group(0).strip()[:80], text[m.start():end]))
    return title, parts


def _ocr_pdf(path):
    """OCR a scanned/image-only PDF into text via tesseract.

    Renders each page with pymupdf, OCRs it with tesseract (`apt install
    tesseract-ocr`), and joins pages with page markers.
    """
    import fitz
    if shutil.which("tesseract") is None:
        raise ValueError(
            "This looks like a scanned PDF, and `tesseract-ocr` isn't installed. "
            "Install it with: sudo apt install tesseract-ocr"
        )
    doc = fitz.open(path)
    pages = []
    with tempfile.TemporaryDirectory() as td:
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=200)
            png = os.path.join(td, f"p{i:04d}.png")
            pix.save(png)
            out = subprocess.run(
                ["tesseract", png, "-", "--psm", "3"],
                capture_output=True, text=True,
            )
            text = (out.stdout or "").strip()
            if text:
                pages.append(f"<!-- page {i} -->\n{text}")
    doc.close()
    return "\n\n".join(pages)


def convert_pdf(path):
    """PDF -> (title, [(n, chapter_title, markdown), ...]).

    Text-layer PDFs -> pymupdf4llm (headings, tables, links). Falls back to
    pdftotext, then to tesseract OCR for scanned/image-only PDFs.
    """
    md = None
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(path)
    except Exception:
        md = None
    if not md or not md.strip():
        out = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True, text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            md = out.stdout
        else:
            md = _ocr_pdf(path)
    # Split on chapter/part/section headings — markdown (# ...) or plain text.
    head = re.compile(
        r"^[ \t]*(?:#{1,3}[ \t]*)?(chapter|part|book|section)\b[^\n]*",
        re.I | re.M,
    )
    matches = list(head.finditer(md))
    if not matches:
        return os.path.basename(path), [(1, "full-text", md.strip())]
    title = md[:matches[0].start()].strip() or os.path.basename(path)
    parts = []
    for i, m in enumerate(matches, 1):
        end = matches[i].start() if i < len(matches) else len(md)
        chap_title = re.sub(r"^#{1,3}[ \t]*", "", m.group(0)).strip()[:80]
        body = md[m.start():end]
        nl = body.find("\n")
        body = (body[nl + 1:] if nl != -1 else "").strip()
        parts.append((i, chap_title, body))
    return title, parts


def convert_mobi(path):
    """MOBI/AZW3 -> (title, [(n, chapter_title, markdown), ...]) via kindleunpack.

    MOBI7 books unpack to a single book.html + toc.ncx; we split on the
    navMap's filepos anchors (chapter boundaries) and drop the redundant TOC.
    KF8/azw3 books unpack to an epub-like folder; we fall back to splitting
    the concatenated HTML on chapter/part/book/section headings.
    """
    if not os.path.exists(KINDLEUNPACK):
        raise ValueError("kindleunpack is not installed — expected at " + KINDLEUNPACK)
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, KINDLEUNPACK, path, td],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            lines = (r.stderr or r.stdout or "").strip().splitlines()
            raise ValueError(
                "kindleunpack failed: " + (lines[-1][:200] if lines else "unknown error")
            )
        html_path = ncx_path = None
        for root, _dirs, files in os.walk(td):
            for f in files:
                fp = os.path.join(root, f)
                low = f.lower()
                if ncx_path is None and low.endswith(".ncx"):
                    ncx_path = fp
                if low.endswith((".html", ".htm", ".xhtml")):
                    if low in ("book.html", "book.htm") or html_path is None:
                        html_path = fp
        if not html_path:
            raise ValueError("kindleunpack produced no HTML — cannot convert.")
        html = open(html_path, "r", encoding="utf-8", errors="replace").read()
        title, parts = None, []
        if ncx_path:
            title, parts = _split_mobi7(html, ncx_path)
        # Some MOBI7 books have a degenerate NCX (a single title navPoint and no
        # chapter entries). Fall back to splitting on body headings when the
        # navMap yields little structure.
        if len(parts) < 3:
            parts_h = _split_html_on_headings(html)
            if len(parts_h) > len(parts):
                parts = parts_h
        return title, parts


def _split_mobi7(html, ncx_path):
    """Split a MOBI7 book.html using the NCX navMap chapter boundaries."""
    tree = ET.parse(ncx_path)
    title = None
    dt = tree.find(f".//{{{NCX_NS}}}docTitle/{{{NCX_NS}}}text")
    if dt is not None and (dt.text or "").strip():
        title = re.sub(r"\s+\(?copy\)?\s*$", "", dt.text.strip(), flags=re.I).strip()

    navpoints = []
    for np_el in tree.iter(f"{{{NCX_NS}}}navPoint"):
        lab = np_el.find(f"{{{NCX_NS}}}navLabel/{{{NCX_NS}}}text")
        cnt = np_el.find(f"{{{NCX_NS}}}content")
        if lab is None or cnt is None:
            continue
        label = (lab.text or "").strip()
        m = re.search(r"filepos(\d+)", cnt.get("src", ""))
        if label and m:
            navpoints.append((label, int(m.group(1))))
    if not navpoints:
        return title, []

    anchors = {}
    for m in re.finditer(r'<a\s+id=["\']filepos(\d+)["\']', html, re.I):
        anchors[int(m.group(1))] = m.start()

    toc_fp = None
    for ref in re.finditer(r"<reference\b([^>]*)>", html, re.I):
        if re.search(r'type=["\']toc["\']', ref.group(1), re.I):
            m2 = re.search(r"filepos=0*(\d+)", ref.group(1))
            if m2:
                toc_fp = int(m2.group(1))
            break

    nav_fps = {fp for _l, fp in navpoints}
    boundaries = [(anchors[fp], label) for label, fp in navpoints if fp in anchors]
    if toc_fp and toc_fp in anchors and toc_fp not in nav_fps:
        boundaries.append((anchors[toc_fp], "__TOC__"))
    boundaries.sort(key=lambda x: x[0])
    if not boundaries:
        return title, []

    parts = []
    i = 0
    fm = _md(html[:boundaries[0][0]]).strip()
    if fm:
        i += 1
        parts.append((i, "Title Page", fm))

    part_name = None
    for k, (off, label) in enumerate(boundaries):
        end = boundaries[k + 1][0] if k + 1 < len(boundaries) else len(html)
        if label == "__TOC__":
            continue
        seg = _md(html[off:end]).strip()
        if not seg:
            continue
        if re.match(r"^(part|book|volume|section)\b", label, re.I):
            part_name = label
            chap_title = label
        else:
            chap_title = f"{part_name} — {label}" if part_name else label
        i += 1
        parts.append((i, chap_title[:80], seg))
    return title, parts


def _split_html_on_headings(html):
    """Fallback: split a single HTML book on chapter/part/book/section/episode
    headings or bare roman-numeral lines (e.g. "I", "II" in A Tale of Two Cities)."""
    md = _md(html)
    head = re.compile(
        r"^[ \t]*(?:#{1,3}[ \t]*)?(?:"
        r"(?:chapter|part|book|section|episode|canto)\b[^\n]*"
        r"|[IVXLC]{1,7}"
        r")[ \t]*$",
        re.I | re.M,
    )
    roman = re.compile(r"^[IVXLC]{1,7}$")
    matches = list(head.finditer(md))
    if not matches:
        return [(1, "full-text", md.strip())]
    parts = []
    n = 0
    for i, m in enumerate(matches, 1):
        end = matches[i].start() if i < len(matches) else len(md)
        chap_title = re.sub(r"^#{1,3}[ \t]*", "", m.group(0)).strip()
        body = md[m.start():end]
        nl = body.find("\n")
        body = (body[nl + 1:] if nl != -1 else "").strip()
        # A bare roman numeral is often followed by an all-caps chapter title
        # line (e.g. "I" then "THE PERIOD"). Fold that title in.
        if roman.match(chap_title):
            nl2 = body.find("\n")
            first_line = body[:nl2].strip() if nl2 != -1 else body.strip()
            if first_line and len(first_line) <= 50 and first_line.isupper():
                chap_title = f"{chap_title} — {first_line}"
                body = (body[nl2 + 1:].strip() if nl2 != -1 else "")
        if body:
            n += 1
            parts.append((n, chap_title[:80], body))
    return parts


def convert(path, out_dir=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".epub":
        title, parts = convert_epub(path)
    elif ext in (".txt", ".md", ".markdown", ".text"):
        title, parts = split_txt(path)
    elif ext in (".mobi", ".azw", ".azw3"):
        title, parts = convert_mobi(path)
    elif ext == ".pdf":
        title, parts = convert_pdf(path)
    else:
        raise ValueError(f"unsupported format: {ext or 'unknown'}")

    title = title or os.path.splitext(os.path.basename(path))[0]
    if out_dir is None:
        out_dir = os.path.join(LIB, _slug(title))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    toc = [f"# {title}", "", f"_source: {os.path.abspath(path)}_", ""]
    for i, chap_title, text in parts:
        safe = _slug(chap_title) or f"chapter-{i}"
        fname = f"{i:03d}_{safe}.md"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(f"# {chap_title}\n\n{text}")
        toc.append(f"{i}. [{chap_title}]({fname})")
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(toc) + "\n")
    return out_dir, len(parts)


def _find_library_dir(path):
    if os.path.isdir(path):
        return path
    src = os.path.abspath(path)
    if os.path.isdir(LIB):
        for name in os.listdir(LIB):
            idx = os.path.join(LIB, name, "index.md")
            if os.path.isfile(idx):
                try:
                    if f"_source: {src}_" in open(idx, "r", encoding="utf-8").read():
                        return os.path.join(LIB, name)
                except Exception:
                    pass
    return convert(path)[0]


def cmd_convert(args):
    out_dir, n = convert(args.file, args.out)
    print(f"converted -> {out_dir} ({n} chapters)")
    print(f"read with:  book.py read {args.file} <N>   (or `read {out_dir}/NNN_*.md`)")


def cmd_list(args):
    d = _find_library_dir(args.file)
    idx = os.path.join(d, "index.md")
    if os.path.exists(idx):
        print(f"index: {idx}")
        print(open(idx, "r", encoding="utf-8").read())
    else:
        convert(args.file, d)
        cmd_list(args)


def cmd_read(args):
    d = _find_library_dir(args.file)
    if args.chapter is None:
        idx = os.path.join(d, "index.md")
        print(open(idx, "r", encoding="utf-8").read() if os.path.exists(idx)
              else f"index not found in {d}")
        return
    for f in sorted(os.listdir(d)):
        if f.startswith(f"{int(args.chapter):03d}_") and f.endswith(".md"):
            print(open(os.path.join(d, f), "r", encoding="utf-8").read())
            return
    print(f"chapter {args.chapter} not found in {d}")


def main():
    p = argparse.ArgumentParser(description="Ebooks -> chapter markdown")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("convert", help="convert to chapter .md files")
    c.add_argument("file")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_convert)

    l = sub.add_parser("list", help="table of contents")
    l.add_argument("file")
    l.set_defaults(fn=cmd_list)

    r = sub.add_parser("read", help="print a chapter (or the index)")
    r.add_argument("file")
    r.add_argument("chapter", nargs="?", type=int)
    r.set_defaults(fn=cmd_read)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return 1
    try:
        args.fn(args)
        return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

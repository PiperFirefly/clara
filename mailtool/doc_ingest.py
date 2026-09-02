#!/usr/bin/env python3
"""
doc_ingest.py — package a PDF into the "optimal for the agent" structure:

    <title>/
    ├── index.md        # manifest: title, pages, figure + table lists
    ├── <title>.md      # full text in markdown (linear reading order),
    │                   #   figures referenced inline, per-page markers
    ├── figures/        # extracted figures (PNG, ~200 dpi render)
    └── data/           # CSV of every detected table (numbers behind the doc)

The idea: Markdown for the words, PNG for the pictures, CSV for the numbers
behind the pictures. See ~/.pi/agent/skills/doc-ingest/SKILL.md.

Text comes from pymupdf4llm (with OCR disabled for text-layer PDFs; scanned
PDFs fall back to OCR). Figures come from the layout engine's "picture"
regions, merged with vector-drawing clusters so chart axes/lines aren't
clipped. Tables come from pymupdf's find_tables() and are written as CSV.

Usage:
  doc_ingest.py ingest <file.pdf> [--out DIR] [--dpi 200]
"""

import argparse
import csv
import os
import sys
import time

import pymupdf  # MuPDF
import pymupdf4llm

try:
    from pymupdf4llm.helpers.document_layout import OCRMode
    _OCR_NEVER = OCRMode.NEVER
    _OCR_DEFAULT = OCRMode.SELECT_KEEP_OLD
except Exception:  # layout engine unavailable in some versions
    _OCR_NEVER = 0
    _OCR_DEFAULT = None

FIGURE_MARGIN = 10.0  # points to pad a figure crop


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def _grow(rect, gap):
    return pymupdf.Rect(rect.x0 - gap, rect.y0 - gap, rect.x1 + gap, rect.y1 + gap)


def _thicken(rect, min_span=1.0):
    """Give zero-height/width line rects a small thickness so intersects() works."""
    r = pymupdf.Rect(rect)
    if r.width < min_span:
        pad = (min_span - r.width) / 2
        r.x0 -= pad
        r.x1 += pad
    if r.height < min_span:
        pad = (min_span - r.height) / 2
        r.y0 -= pad
        r.y1 += pad
    return r


def _merge_rects(rects, gap=15.0):
    """Iteratively union nearby/overlapping rects until stable."""
    rects = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(rects):
            j = i + 1
            while j < len(rects):
                if rects[i].intersects(_grow(rects[j], gap)):
                    rects[i] |= rects[j]
                    rects.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return rects


def _layout_captions(chunk, page):
    """Return [(rect, text)] for caption-class boxes the layout engine found."""
    caps = []
    for b in chunk.get("page_boxes", []):
        if b.get("class") == "caption" and b.get("bbox"):
            x0, y0, x1, y1 = b["bbox"]
            r = pymupdf.Rect(x0, y0, x1, y1)
            txt = " ".join(page.get_text("text", clip=r).split())
            if txt:
                caps.append((r, txt))
    return caps


def _match_caption(fig_rect, caps):
    """Find a caption box directly below the figure (within 40pt, overlapping x)."""
    for r, txt in sorted(caps, key=lambda c: c[0].y0):
        if r.y0 >= fig_rect.y1 - 2 and r.y0 - fig_rect.y1 <= 40 \
                and r.x1 > fig_rect.x0 and r.x0 < fig_rect.x1:
            return txt
    return ""


def _render_rect(page, rect, path, dpi):
    zoom = dpi / 72.0
    clip = (_grow(rect, FIGURE_MARGIN)) & page.rect
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
    pix.save(path)
    return path


# --------------------------------------------------------------------------
# figure candidates (three sources -> merged)
# --------------------------------------------------------------------------

def _layout_picture_rects(chunk):
    """Picture regions detected by the pymupdf4llm layout engine."""
    rects = []
    for b in chunk.get("page_boxes", []):
        if b.get("class") == "picture" and b.get("bbox"):
            x0, y0, x1, y1 = b["bbox"]
            r = pymupdf.Rect(x0, y0, x1, y1)
            if r.get_area() > 200:
                rects.append(r)
    return rects


def _vector_drawing_rects(page, table_rects):
    """Cluster vector strokes/fills (charts drawn with lines)."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    stroke, fill = [], []
    for d in drawings:
        r = _thicken(d["rect"])
        if any(r.intersects(_grow(t, 2)) for t in table_rects):
            continue
        (fill if d.get("fill") is not None else stroke).append(r)
    clusters = _merge_rects(fill + stroke)
    out = []
    for r in clusters:
        has_fill = any(r.intersects(_grow(f, 2)) for f in fill)
        n_strokes = sum(1 for s in stroke if r.intersects(_grow(s, 2)))
        area = r.get_area()
        longest = r.width > 100 or r.height > 100  # chart axes are long strokes
        if area < 500 and not longest:
            continue  # degenerate: a single tiny rule/underline
        if not (has_fill or n_strokes >= 3 or area >= 1500 or longest):
            continue  # too sparse to be a chart
        out.append(r)
    return out


def _raster_image_rects(page, table_rects):
    rects, seen = [], set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            for r in page.get_image_rects(xref):
                if not any(r.intersects(_grow(t, 2)) for t in table_rects):
                    rects.append(r)
        except Exception:
            continue
    return [r for r in rects if r.get_area() > 200]


# --------------------------------------------------------------------------
# table CSV
# --------------------------------------------------------------------------

def _extract_tables(page, data_dir, idx):
    records = []
    try:
        tables = page.find_tables()
    except Exception:
        return records
    for t in tables.tables:
        try:
            rows = t.extract()
        except Exception:
            continue
        if not rows:
            continue
        idx += 1
        name = f"table-{idx:02d}.csv"
        path = os.path.join(data_dir, name)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            for row in rows:
                w.writerow(["" if c is None else str(c) for c in row])
        records.append({"idx": idx, "file": name, "page": page.number + 1,
                        "rows": len(rows)})
    return records


# --------------------------------------------------------------------------
# main ingest
# --------------------------------------------------------------------------

def _page_number(chunk):
    meta = chunk.get("metadata", {})
    return meta.get("page_number", meta.get("page", 1))


def ingest(pdf_path, out_dir=None, dpi=200):
    pdf_path = os.path.abspath(pdf_path)
    doc = pymupdf.open(pdf_path)
    title = (doc.metadata.get("title") or "").strip() or os.path.splitext(
        os.path.basename(pdf_path))[0]
    title = "".join(c for c in title if c.isalnum() or c in " _-").strip() or "doc"

    if out_dir is None:
        out_dir = os.path.expanduser(f"~/ingested/{title}")
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, use_ocr=_OCR_NEVER)
    if sum(len(c.get("text", "")) for c in chunks) < 40:
        # looks scanned/image-only -> retry with OCR
        kw = {} if _OCR_DEFAULT is None else {"use_ocr": _OCR_DEFAULT}
        chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, **kw)

    fig_idx = 0
    table_idx = 0
    all_figs, all_tables = [], []
    md_parts = []

    for chunk in chunks:
        pageno = _page_number(chunk)
        page = doc[pageno - 1]

        try:
            table_rects = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
        except Exception:
            table_rects = []

        # tables -> CSV
        t_recs = _extract_tables(page, data_dir, table_idx)
        table_idx += len(t_recs)
        all_tables.extend(t_recs)

        # figures: layout pictures + vector clusters + raster images, merged
        rects = (_layout_picture_rects(chunk)
                 + _vector_drawing_rects(page, table_rects)
                 + _raster_image_rects(page, table_rects))
        merged = _merge_rects(rects)
        caps = _layout_captions(chunk, page)
        f_recs = []
        for r in merged:
            fig_idx += 1
            name = f"fig-{fig_idx:02d}.png"
            try:
                _render_rect(page, r, os.path.join(fig_dir, name), dpi)
            except Exception:
                continue
            cap = _match_caption(r, caps)
            f_recs.append({"idx": fig_idx, "file": name, "page": pageno,
                           "caption": cap})
        all_figs.extend(f_recs)

        # page text + inline references
        md_parts.append(f"<!-- page {pageno} -->\n")
        md_parts.append(chunk["text"].strip())
        for f in f_recs:
            cap = f["caption"] or f"Figure {f['idx']}"
            md_parts.append(f"\n![{cap}](figures/{f['file']})\n")
        for t in t_recs:
            md_parts.append(
                f"\n> Table {t['idx']} — data: [`data/{t['file']}`](data/{t['file']})\n")
        md_parts.append("\n")

    doc.close()

    body_md = "\n".join(md_parts).strip() + "\n"
    with open(os.path.join(out_dir, f"{title}.md"), "w") as fh:
        fh.write(body_md)

    # manifest
    idx_md = [f"# {title}", "",
              f"- **Source:** `{pdf_path}`",
              f"- **Pages:** {len(chunks)}",
              f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"- **Text:** [`{title}.md`]({title}.md)", ""]
    if all_figs:
        idx_md += ["## Figures", "", "| # | page | file | caption |",
                   "|---|------|------|---------|"]
        for f in all_figs:
            idx_md.append(f"| {f['idx']} | {f['page']} | "
                          f"[`{f['file']}`](figures/{f['file']}) | "
                          f"{f['caption'] or '—'} |")
        idx_md.append("")
    if all_tables:
        idx_md += ["## Tables", "", "| # | page | file | rows |",
                   "|---|------|------|------|"]
        for t in all_tables:
            idx_md.append(f"| {t['idx']} | {t['page']} | "
                          f"[`{t['file']}`](data/{t['file']}) | {t['rows']} |")
        idx_md.append("")
    with open(os.path.join(out_dir, "index.md"), "w") as fh:
        fh.write("\n".join(idx_md))

    return out_dir, title, len(all_figs), len(all_tables)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Package a PDF into markdown + figures + data for the agent.")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="PDF -> md + figures/ + data/")
    i.add_argument("pdf")
    i.add_argument("--out", help="output dir (default ~/ingested/<title>)")
    i.add_argument("--dpi", type=int, default=200, help="figure render dpi")
    a = p.parse_args(argv)

    if a.cmd == "ingest":
        if not os.path.exists(a.pdf):
            print(f"error: not found: {a.pdf}", file=sys.stderr)
            return 1
        out, title, nf, nt = ingest(a.pdf, a.out, a.dpi)
        print(f"ingested '{title}' -> {out}")
        print(f"  figures: {nf}   tables: {nt}")
        print(f"  read: {out}/index.md  or  {out}/{title}.md")
        return 0


if __name__ == "__main__":
    sys.exit(main())

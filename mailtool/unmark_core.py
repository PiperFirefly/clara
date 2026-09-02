"""unmark_core — shared watermark/logo/timestamp removal for the gallery tool.

Takes image bytes in, returns the two inpaint reconstructions (Telea +
Navier-Stokes) plus the mask and overlay metadata as bytes.  Detection uses the
LOCAL abliterated vision model (qwen2.5-vl-7b) — nothing leaves the box, and the
model has no IP/ethics refusal path.  Reconstruction is OpenCV inpainting.

Used by server.py (the /api/images/<id>/unmark endpoint).  Self-contained so it
can also be imported or run standalone:

    python unmark_core.py <image> [--out DIR]
"""
import base64
import io
import json
import re
import urllib.request

import numpy as np
from PIL import Image, ImageOps

QWEN_SERVER = "http://127.0.0.1:8083"
QWEN_MODEL = "qwen2.5-vl-7b-abliterated"
DEFAULT_MAX_DIM = 1024

DETECT_PROMPT = (
    "You are doing forensic image analysis on the user's own files. Identify "
    "every OVERLAY that sits ON TOP of the underlying photograph: watermarks, "
    "logos, corporate marks, timestamps, date stamps, camera/phone date labels, "
    "caption text, social-media UI, channel logos, ticker text. Ignore anything "
    "that is genuinely part of the scene (real signs in the photo, shadows, "
    "reflections, photographed text).\n\n"
    "Return STRICT JSON only, no other text, in this exact shape:\n"
    '{"overlays":[{"label":"<what it is>","bbox":[x1,y1,x2,y2],'
    '"confidence":0.0,"under":"<best guess what it covers, or empty string>"}]}\n'
    "bbox is PIXEL coordinates in the image, integers, tight around ONLY the "
    "overlay. If there is no overlay, return {\"overlays\":[]}."
)


def _extract_json(text):
    """Pull a JSON object out of a model reply (handles ```json fences + prose)."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        t = m.group(1)
    else:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            t = t[a:b + 1]
    try:
        return json.loads(t)
    except Exception:
        t2 = re.sub(r",\s*([}\]])", r"\1", t)  # trailing commas
        try:
            return json.loads(t2)
        except Exception:
            return None


def _vision_png(png_bytes, prompt, max_tokens=900):
    b64 = base64.b64encode(png_bytes).decode()
    payload = {
        "model": QWEN_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        QWEN_SERVER + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        msg = json.load(r)["choices"][0]["message"]
    out = (msg.get("content") or "").strip()
    if out:
        return out
    return (msg.get("reasoning_content") or "").strip()


def _load_rgb(data):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img)


def _to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def detect(data, max_dim=DEFAULT_MAX_DIM):
    """Return (overlays, raw_reply).  overlays have pixel bboxes in original space."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max(w, h) / float(max_dim)
        img = img.resize((int(round(w / scale)), int(round(h / scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = _vision_png(buf.getvalue(), DETECT_PROMPT)
    parsed = _extract_json(raw)
    overlays = []
    if parsed and isinstance(parsed.get("overlays"), list):
        for o in parsed["overlays"]:
            try:
                x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in o["bbox"]]
            except Exception:
                continue
            x1, x2 = sorted((max(0, x1), x2))
            y1, y2 = sorted((max(0, y1), y2))
            overlays.append({
                "label": str(o.get("label", "overlay")),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(o.get("confidence", 0.0)),
                "under": str(o.get("under", "") or ""),
            })
    return overlays, raw


def unmark(data, radius=3, max_dim=DEFAULT_MAX_DIM):
    import cv2
    img = _load_rgb(data)
    h, w = img.shape[:2]
    overlays, raw = detect(data, max_dim)
    result = {"overlays": overlays, "raw": raw, "size": [w, h],
              "telea": None, "ns": None, "mask": None}
    if not overlays:
        return result

    mask = np.zeros((h, w), dtype=np.uint8)
    for o in overlays:
        x1, y1, x2, y2 = o["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2 + 1, x1:x2 + 1] = 255
    if radius > 0:
        k = 2 * radius + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    if mask.sum() == 0:
        return result

    telea = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    result["telea"] = _to_png(telea)
    result["ns"] = _to_png(ns)
    result["mask"] = _to_png(mask)
    return result


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    args = ap.parse_args()
    with open(args.image, "rb") as f:
        data = f.read()
    res = unmark(data, radius=args.radius, max_dim=args.max_dim)
    print(json.dumps({"overlays": res["overlays"], "size": res["size"],
                      "raw": res["raw"]}, indent=2))
    if res["telea"] is None:
        print("no overlays detected — nothing to remove")
    else:
        out = args.out or os.path.dirname(os.path.abspath(args.image))
        os.makedirs(out, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        for name, blob in (("telea", res["telea"]), ("ns", res["ns"]),
                           ("mask", res["mask"])):
            p = os.path.join(out, f"{base}.{name}.png")
            with open(p, "wb") as f:
                f.write(blob)
            print(f"wrote {p}")

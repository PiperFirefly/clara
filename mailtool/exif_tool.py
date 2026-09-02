"""exif_tool — read/edit EXIF on gallery images.

Reads the full EXIF as editable entries and applies edits with a pixel-
preserving byte-level transplant (piexif) for JPEG/WebP, so changing a tag never
recompresses the photo.  GPS lat/lon are exposed as decimal degrees.

Used by server.py endpoints /api/images/<id>/exif (GET+POST).
"""
import io
import re

import piexif
from PIL import Image

_IFD_KEYS = ("0th", "Exif", "GPS", "1st")


def _tag_name(ifd, tag):
    m = piexif.TAGS.get(ifd, {})
    return m.get(tag, {}).get("name", str(tag))


def _type_of(v):
    if isinstance(v, (bytes, str)):
        return "Ascii"
    if isinstance(v, bool) or isinstance(v, int):
        return "Short" if (isinstance(v, int) and 0 <= v <= 65535) else "Long"
    if isinstance(v, tuple) and len(v) == 2:
        return "Rational"
    if isinstance(v, list):
        return "List"
    return "Unknown"


def _edit_str(v):
    """Canonical editable string for a value."""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "replace").rstrip("\x00")
        except Exception:
            return v.hex()
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    if isinstance(v, tuple) and len(v) == 2:
        if v[1] == 0:
            return "0"
        return repr(round(v[0] / v[1], 6))
    if isinstance(v, list):
        return ", ".join(_edit_str(x) for x in v)
    return repr(v)


def _gps_decimal(rationals, ref):
    try:
        deg = rationals[0][0] / rationals[0][1]
        mn = rationals[1][0] / rationals[1][1]
        sc = rationals[2][0] / rationals[2][1]
        dec = deg + mn / 60.0 + sc / 3600.0
    except Exception:
        return None
    ref = (ref or b"").decode("latin-1", "replace") if isinstance(ref, bytes) else (ref or "")
    if ref in ("S", "W"):
        dec = -dec
    return round(dec, 6)


def read_exif(data):
    """Return a flat list of editable EXIF entries."""
    try:
        exif = piexif.load(data)
    except Exception:
        return []
    out = []
    for ifd in _IFD_KEYS:
        tags = exif.get(ifd, {})
        if not tags:
            continue
        for tag, v in tags.items():
            name = _tag_name(ifd, tag)
            if ifd == "GPS" and tag == piexif.GPSIFD.GPSLatitude:
                ref = exif["GPS"].get(piexif.GPSIFD.GPSLatitudeRef, b"N")
                dec = _gps_decimal(v, ref)
                out.append({"ifd": ifd, "tag": tag, "name": "GPSLatitude",
                            "type": "GPS", "value": dec,
                            "edit": ("" if dec is None else repr(dec))})
            elif ifd == "GPS" and tag == piexif.GPSIFD.GPSLongitude:
                ref = exif["GPS"].get(piexif.GPSIFD.GPSLongitudeRef, b"E")
                dec = _gps_decimal(v, ref)
                out.append({"ifd": ifd, "tag": tag, "name": "GPSLongitude",
                            "type": "GPS", "value": dec,
                            "edit": ("" if dec is None else repr(dec))})
            else:
                out.append({"ifd": ifd, "tag": tag, "name": name,
                            "type": _type_of(v), "value": _edit_str(v),
                            "edit": _edit_str(v)})
    return out


def _rat(f):
    neg = f < 0
    a = abs(f)
    return (int(round(a * 1_000_000)), 1_000_000), neg


def _gps_rationals(dec):
    neg = dec < 0
    a = abs(dec)
    d = int(a)
    m = (a - d) * 60.0
    mi = int(m)
    s = (m - mi) * 60.0
    return [(d, 1), (mi, 1), (int(round(s * 1_000_000)), 1_000_000)], neg


def _coerce(valstr, existing):
    """String -> piexif-ready Python value, typed off the existing value."""
    s = (valstr or "").strip()
    if isinstance(existing, (bytes, str)):
        return s.encode("utf-8")
    if isinstance(existing, bool):
        return 1 if s.lower() in ("1", "true", "yes", "on") else 0
    if isinstance(existing, int):
        try:
            return int(float(s))
        except Exception:
            return existing
    if isinstance(existing, tuple) and len(existing) == 2:
        try:
            r, _neg = _rat(float(s))
            return r
        except Exception:
            return existing
    if isinstance(existing, list):
        # comma-separated scalars
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        first = existing[0] if existing else None
        if first is not None:
            return [_coerce(p, first) for p in parts]
        return existing
    # unknown / new tag — guess
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)", s):
        return _rat(float(s))[0]
    return s.encode("utf-8")


def _set_entry(exif, ifd, tag, valstr):
    if ifd not in exif:
        exif[ifd] = {}
    existing = exif[ifd].get(tag)
    if ifd == "GPS" and tag == piexif.GPSIFD.GPSLatitude:
        try:
            dec = float(valstr)
        except Exception:
            return
        rats, neg = _gps_rationals(dec)
        exif["GPS"][piexif.GPSIFD.GPSLatitude] = rats
        exif["GPS"][piexif.GPSIFD.GPSLatitudeRef] = b"S" if neg else b"N"
        return
    if ifd == "GPS" and tag == piexif.GPSIFD.GPSLongitude:
        try:
            dec = float(valstr)
        except Exception:
            return
        rats, neg = _gps_rationals(dec)
        exif["GPS"][piexif.GPSIFD.GPSLongitude] = rats
        exif["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"W" if neg else b"E"
        return
    exif[ifd][tag] = _coerce(valstr, existing)


def apply(data, set_list, remove_list):
    """Apply edits; return new image bytes (pixels preserved for JPEG/WebP)."""
    try:
        exif = piexif.load(data)
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {},
                "Interop": {}, "thumbnail": None}
    for r in remove_list or []:
        ifd, tag = r.get("ifd"), r.get("tag")
        if ifd and tag is not None and tag in exif.get(ifd, {}):
            del exif[ifd][tag]
    for s in set_list or []:
        ifd, tag = s.get("ifd"), s.get("tag")
        if ifd and tag is not None:
            try:
                _set_entry(exif, ifd, tag, s.get("value", ""))
            except Exception:
                continue
    try:
        exif_bytes = piexif.dump(exif)
    except Exception:
        return data  # nothing to write / invalid — leave untouched
    try:
        out = io.BytesIO()
        piexif.insert(exif_bytes, data, out)  # BytesIO target = pixel-preserving
        return out.getvalue()
    except Exception:
        # PNG (or unusual format): re-save with Pillow, which supports EXIF on PNG
        try:
            im = Image.open(io.BytesIO(data))
            fmt = im.format or "JPEG"
            buf = io.BytesIO()
            save_kw = {"format": fmt}
            if fmt.upper() in ("JPEG", "PNG", "WEBP", "TIFF"):
                save_kw["exif"] = exif_bytes
            im.save(buf, **save_kw)
            return buf.getvalue()
        except Exception:
            return data


def reencode_with_original_exif(orig_data, new_png_bytes, mime):
    """Re-encode an inpainted image (PNG bytes) in the ORIGINAL format, carrying
    the original EXIF forward with Orientation reset to 1 (pixels are already
    display-correct).  Returns (bytes, mime)."""
    exif_bytes = None
    try:
        exif = piexif.load(orig_data)
        exif.setdefault("0th", {})
        exif["0th"][piexif.ImageIFD.Orientation] = 1
        exif_bytes = piexif.dump(exif)
    except Exception:
        exif_bytes = None

    im = Image.open(io.BytesIO(new_png_bytes)).convert("RGB")
    ml = (mime or "").lower()
    if "png" in ml:
        fmt, new_mime = "PNG", "image/png"
    elif "webp" in ml:
        fmt, new_mime = "WEBP", "image/webp"
    elif "tiff" in ml:
        fmt, new_mime = "TIFF", "image/tiff"
    else:
        fmt, new_mime = "JPEG", "image/jpeg"

    buf = io.BytesIO()
    kw = {"format": fmt}
    if exif_bytes is not None and fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
        kw["exif"] = exif_bytes
    if fmt == "JPEG":
        kw["quality"] = 95
    im.save(buf, **kw)
    return buf.getvalue(), new_mime

"""Walk the archive, fingerprint every media file, keep EXIF capture dates.

Cross-platform and dependency-free: EXIF DateTimeOriginal is pulled out of
JPEG APP1 segments by hand; everything else falls back to the file mtime.
Fingerprints are size + mtime + a 64 KB head/tail hash so re-scans skip
unchanged files without hashing gigabytes of video.
"""

from __future__ import annotations

import hashlib
import os
import struct
from datetime import datetime, timezone

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif", ".tiff", ".avif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg", ".3gp"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".opus", ".wma", ".aiff"}

MEDIA_TYPES = {"image": IMAGE_EXT, "video": VIDEO_EXT, "audio": AUDIO_EXT}


def media_type_of(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    for mtype, exts in MEDIA_TYPES.items():
        if ext in exts:
            return mtype
    return None


def fingerprint(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    h.update(f"{size}:{mtime:.6f}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
        if size > 131072:
            fh.seek(-65536, os.SEEK_END)
            h.update(fh.read(65536))
    return h.hexdigest()[:32]


def _exif_datetime_original(data: bytes) -> str | None:
    """Extract DateTimeOriginal from raw JPEG bytes, or None.

    Walks to the APP1 Exif segment, parses the TIFF header and the Exif IFD
    pointer, and reads tag 0x9003. Enough EXIF for one date field.
    """
    if data[:2] != b"\xff\xd8":
        return None
    idx = 2
    tiff = None
    while idx < len(data) - 4:
        if data[idx] != 0xFF:
            break
        marker = data[idx + 1]
        if marker in (0xD8, 0xD9):
            idx += 2
            continue
        (seg_len,) = struct.unpack(">H", data[idx + 2:idx + 4])
        if marker == 0xE1 and data[idx + 4:idx + 10] == b"Exif\x00\x00":
            tiff = data[idx + 10: idx + 2 + seg_len]
            break
        idx += 2 + seg_len
    if tiff is None:
        return None
    endian = tiff[:2]
    if endian == b"II":
        en = "<"
    elif endian == b"MM":
        en = ">"
    else:
        return None
    try:
        ifd_off = struct.unpack(en + "I", tiff[4:8])[0]
        n = struct.unpack(en + "H", tiff[ifd_off:ifd_off + 2])[0]
        exif_ifd = None
        for i in range(n):
            e = ifd_off + 2 + i * 12
            tag, typ, cnt = struct.unpack(en + "HHI", tiff[e:e + 8])
            if tag == 0x8769:  # Exif IFD pointer
                exif_ifd = struct.unpack(en + "I", tiff[e + 8:e + 12])[0]
                break
        if not exif_ifd:
            return None
        n = struct.unpack(en + "H", tiff[exif_ifd:exif_ifd + 2])[0]
        for i in range(n):
            e = exif_ifd + 2 + i * 12
            tag, typ, cnt = struct.unpack(en + "HHI", tiff[e:e + 8])
            if tag == 0x9003 or tag == 0x0132:  # DateTimeOriginal / DateTime
                val_off = struct.unpack(en + "I", tiff[e + 8:e + 12])[0]
                raw = tiff[val_off:val_off + cnt]
                txt = raw.rstrip(b"\x00").decode("ascii", "replace")
                if len(txt) >= 19:
                    # "2021:11:02 14:30:00" -> "2021-11-02T14:30:00"
                    return txt[:10].replace(":", "-") + "T" + txt[11:19]
                return None
        return None
    except (struct.error, IndexError):
        return None


def _exif_gps(data: bytes) -> tuple[float, float] | None:
    """Extract (lat, lon) from JPEG EXIF GPS IFD, or None."""
    if data[:2] != b"\xff\xd8":
        return None
    idx = 2
    tiff = None
    while idx < len(data) - 4:
        if data[idx] != 0xFF:
            break
        marker = data[idx + 1]
        if marker in (0xD8, 0xD9):
            idx += 2
            continue
        (seg_len,) = struct.unpack(">H", data[idx + 2:idx + 4])
        if marker == 0xE1 and data[idx + 4:idx + 10] == b"Exif\x00\x00":
            tiff = data[idx + 10: idx + 2 + seg_len]
            break
        idx += 2 + seg_len
    if tiff is None:
        return None
    endian = tiff[:2]
    if endian == b"II":
        en = "<"
    elif endian == b"MM":
        en = ">"
    else:
        return None

    def rationals(off: int, count: int) -> list[float]:
        out = []
        for i in range(count):
            num, den = struct.unpack(en + "II", tiff[off + i * 8: off + i * 8 + 8])
            out.append(num / den if den else 0.0)
        return out

    try:
        ifd_off = struct.unpack(en + "I", tiff[4:8])[0]
        n = struct.unpack(en + "H", tiff[ifd_off:ifd_off + 2])[0]
        gps_ifd = None
        for i in range(n):
            e = ifd_off + 2 + i * 12
            tag, typ, cnt = struct.unpack(en + "HHI", tiff[e:e + 8])
            if tag == 0x8825:  # GPS IFD pointer
                gps_ifd = struct.unpack(en + "I", tiff[e + 8:e + 12])[0]
                break
        if not gps_ifd:
            return None
        lat_ref = lon_ref = None
        lat_vals = lon_vals = None
        n = struct.unpack(en + "H", tiff[gps_ifd:gps_ifd + 2])[0]
        for i in range(n):
            e = gps_ifd + 2 + i * 12
            tag, typ, cnt, vlen = struct.unpack(en + "HHII", tiff[e:e + 12])
            val_off = struct.unpack(en + "I", tiff[e + 8:e + 12])[0]
            if tag in (0x0001, 0x0003):  # N/S and E/W refs: inline ASCII
                ref = tiff[e + 8:e + 8 + cnt].decode("ascii", "replace").strip("\x00 ").upper()
                if tag == 0x0001:
                    lat_ref = ref
                else:
                    lon_ref = ref
            elif tag == 0x0002:  # GPSLatitude: 3 rationals
                lat_vals = rationals(val_off, cnt)
            elif tag == 0x0004:  # GPSLongitude
                lon_vals = rationals(val_off, cnt)
        if not (lat_ref and lon_ref and lat_vals and lon_vals):
            return None

        def dms(vals: list[float]) -> float:
            vals = (vals + [0, 0, 0])[:3]
            return vals[0] + vals[1] / 60 + vals[2] / 3600

        lat = dms(lat_vals) * (-1 if lat_ref == "S" else 1)
        lon = dms(lon_vals) * (-1 if lon_ref == "W" else 1)
        if abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
            return None
        return (round(lat, 6), round(lon, 6))
    except (struct.error, IndexError):
        return None


def capture_gps(path: str) -> tuple[float, float] | None:
    if os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".tiff"):
        return None
    try:
        with open(path, "rb") as fh:
            return _exif_gps(fh.read(262144))
    except OSError:
        return None


def capture_date(path: str, mtime: float) -> str:
    """Best-effort ISO capture timestamp: EXIF, then mtime."""
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".tiff"):
        try:
            with open(path, "rb") as fh:
                head = fh.read(262144)  # EXIF lives up front
            got = _exif_datetime_original(head)
            if got:
                return got
        except OSError:
            pass
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def scan_archive(root: str, conn, _progress=None) -> dict:
    """Sync the media table with the directory tree. Returns counters."""
    root = os.path.abspath(os.path.expanduser(root))
    seen = set()
    added = updated = unchanged = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            full = os.path.join(dirpath, fn)
            mtype = media_type_of(full)
            if not mtype:
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            fp = fingerprint(full, st.st_size, st.st_mtime)
            row = conn.execute("SELECT id, fingerprint FROM media WHERE path=?", (full,)).fetchone()
            seen.add(full)
            if row and row["fingerprint"] == fp:
                unchanged += 1
                continue
            created = capture_date(full, st.st_mtime)
            gps = capture_gps(full)
            if row:
                conn.execute(
                    "UPDATE media SET filename=?, media_type=?, size=?, mtime=?, fingerprint=?, created_at=?,"
                    " gps_lat=?, gps_lon=? WHERE id=?",
                    (fn, mtype, st.st_size, st.st_mtime, fp, created,
                     gps[0] if gps else None, gps[1] if gps else None, row["id"]))
                updated += 1
            else:
                conn.execute(
                    "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint, created_at,"
                    " gps_lat, gps_lon) VALUES (?,?,?,?,?,?,?,?,?)",
                    (full, fn, mtype, st.st_size, st.st_mtime, fp, created,
                     gps[0] if gps else None, gps[1] if gps else None))
                added += 1
            if _progress:
                _progress(fn, added, updated, unchanged)
    gone = [r["id"] for r in conn.execute("SELECT id, path FROM media") if r["path"] not in seen]
    conn.executemany("DELETE FROM media WHERE id=?", [(i,) for i in gone])
    conn.executemany("DELETE FROM embeddings WHERE media_id=?", [(i,) for i in gone])
    conn.commit()
    return {"added": added, "updated": updated, "unchanged": unchanged, "removed": len(gone)}

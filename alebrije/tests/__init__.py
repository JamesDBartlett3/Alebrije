"""Shared test helpers: temp homes, tiny media, fake EXIF, fake models."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@contextlib.contextmanager
def temp_home():
    """Isolate ~/.alebrije for the duration of a test."""
    old = os.environ.pop("ALEBRIJE_HOME", None)
    d = tempfile.mkdtemp(prefix="alebrije-test-")
    os.environ["ALEBRIJE_HOME"] = d
    try:
        yield d
    finally:
        os.chdir(os.path.expanduser("~"))
        shutil.rmtree(d, ignore_errors=True)
        if old is not None:
            os.environ["ALEBRIJE_HOME"] = old
        else:
            os.environ.pop("ALEBRIJE_HOME", None)


def png_bytes(w: int = 4, h: int = 4, rgb=(200, 120, 30)) -> bytes:
    raw = b""
    for _ in range(h):
        raw += b"\x00" + bytes(rgb) * w

    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ---- minimal-but-real EXIF JPEG -------------------------------------------

def exif_jpeg_bytes(dt: str | None = None, gps: tuple[float, float] | None = None) -> bytes:
    """A tiny valid JPEG carrying DateTimeOriginal and/or GPS EXIF.

    Layout (little-endian TIFF): header(8) | IFD0 | ExifIFD+dt | GPSIFD | heap.
    """
    en = "<"
    ifd0_entries = (1 if (dt or gps) else 0) + (1 if gps else 0)
    ifd0_size = 2 + ifd0_entries * 12 + 4
    exif_off = 8 + ifd0_size
    exif_size = (2 + 12 + 4 + len(dt) + 1) if dt else 0
    gps_off = exif_off + exif_size

    ifd0 = struct.pack(en + "H", ifd0_entries)
    if dt or gps:
        ifd0 += struct.pack(en + "HHI", 0x8769, 4, 1) + struct.pack(en + "I", exif_off)
    if gps:
        ifd0 += struct.pack(en + "HHI", 0x8825, 4, 1) + struct.pack(en + "I", gps_off)
    ifd0 += struct.pack(en + "I", 0)

    exif_block = b""
    if dt:
        dt_val = dt.encode() + b"\x00"
        exif_block = (struct.pack(en + "H", 1)
                      + struct.pack(en + "HHI", 0x9003, 2, len(dt_val))
                      + struct.pack(en + "I", gps_off if False else 0))  # patched below
        # value heap comes right after this IFD: offset = exif_off + 2+12+4
        dt_off = exif_off + 2 + 12 + 4
        exif_block = (struct.pack(en + "H", 1)
                      + struct.pack(en + "HHI", 0x9003, 2, len(dt_val))
                      + struct.pack(en + "I", dt_off)
                      + struct.pack(en + "I", 0)
                      + dt_val)

    gps_block = b""
    heap = b""
    if gps:
        lat, lon = gps
        latv = b"".join((struct.pack(en + "II", int(round(abs(lat) * 10000)), 10000),
                         struct.pack(en + "II", 0, 1),
                         struct.pack(en + "II", 0, 1)))
        lonv = b"".join((struct.pack(en + "II", int(round(abs(lon) * 10000)), 10000),
                         struct.pack(en + "II", 0, 1),
                         struct.pack(en + "II", 0, 1)))
        gps_entries_count = 5
        gps_block_size = 2 + gps_entries_count * 12 + 4
        heap_off = gps_off + gps_block_size
        entries = [
            (0x0000, 1, 4, struct.pack(en + "I", 2)),
            (0x0001, 2, 2, (b"N" if lat >= 0 else b"S") + b"\x00"),
            (0x0002, 5, 3, struct.pack(en + "I", heap_off)),
            (0x0003, 2, 2, (b"E" if lon >= 0 else b"W") + b"\x00"),
            (0x0004, 5, 3, struct.pack(en + "I", heap_off + len(latv))),
        ]
        gps_block = struct.pack(en + "H", len(entries))
        for tag, typ, cnt, val in entries:
            gps_block += (struct.pack(en + "HHI", tag, typ, cnt)
                          + (val if len(val) == 4 else val.ljust(4, b"\x00")))
        gps_block += struct.pack(en + "I", 0) + latv + lonv

    tiff = (b"II" + struct.pack(en + "H", 42) + struct.pack(en + "I", 8)
            + ifd0 + exif_block + gps_block)
    app1_payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(app1_payload) + 2) + app1_payload
    jfif = (b"\xff\xe0" + struct.pack(">H", 18) + b"JFIF\x00" + bytes([1, 1, 0])
            + struct.pack(">HH", 1, 1) + bytes([0, 0]))
    dqt = (b"\xff\xdb" + struct.pack(">H", 67) + b"\x00"
           + bytes([16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40,
                    26, 24, 22, 22, 24, 49, 35, 37, 29, 40, 58, 51, 61, 60, 57, 51,
                    56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81,
                    87, 95, 106, 116, 38, 72, 114, 112, 100, 118, 102, 110, 120,
                    121, 113, 125, 122, 122, 122]))
    return b"\xff\xd8" + app1 + jfif + dqt + b"\xff\xd9"


def write_media(dir_path: str, name: str, data: bytes | None = None) -> str:
    os.makedirs(dir_path, exist_ok=True)
    p = os.path.join(dir_path, name)
    with open(p, "wb") as fh:
        fh.write(data if data is not None else png_bytes())
    return p


def sample_config(**over) -> dict:
    from alebrije import config
    cfg = dict(config.DEFAULTS)
    cfg.update(over)
    return cfg


FAKE_TAG_JSON = json.dumps({
    "caption": "Abuela laughs at the beach birthday party",
    "people": [{"description": "older woman, white braid, jade earrings",
                "position": "left"},
               {"description": "young man in a red sombrero", "position": "right"}],
    "objects": ["birthday cake", "confetti"],
    "clothing": ["red sombrero"],
    "animals": ["brown dog"],
    "food": ["tres leches cake"],
    "mood": "joyful",
    "setting": "outdoor beach party",
})


class FakeLlm:
    """Patch target: records calls, returns canned replies."""

    def __init__(self, chat_reply: str = FAKE_TAG_JSON, embed_dim: int = 4,
                 embed_fn=None):
        self.chat_reply = chat_reply
        self.chat_calls: list = []
        self.embed_calls: list = []
        self.embed_dim = embed_dim
        self.embed_fn = embed_fn or (lambda text: self._deterministic(text))

    def _deterministic(self, text: str) -> list[float]:
        v = [0.0] * self.embed_dim
        for i, ch in enumerate(text.encode() or b"x"):
            v[i % self.embed_dim] += (ch % 7) - 3
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def chat(self, cfg, messages, image_paths=None):
        self.chat_calls.append({"messages": messages, "images": image_paths})
        return self.chat_reply

    def embed(self, cfg, texts):
        self.embed_calls.append(list(texts))
        return [self.embed_fn(t) for t in texts]

    def describe_image(self, cfg, image_path, prompt):
        self.chat_calls.append({"prompt": prompt, "image": image_path})
        return self.chat_reply


@contextlib.contextmanager
def patched_llm(fake: FakeLlm):
    """Point every model call at a FakeLlm; restores on exit."""
    from alebrije import llm, vectors
    orig_chat, orig_embed, orig_desc = llm.chat, llm.embed, llm.describe_image
    llm.chat = fake.chat
    llm.embed = fake.embed
    llm.describe_image = fake.describe_image
    orig_vembed = vectors.embed_one
    vectors.embed_one = lambda cfg, text: fake.embed(cfg, [text])[0]
    try:
        yield fake
    finally:
        llm.chat, llm.embed, llm.describe_image = orig_chat, orig_embed, orig_desc
        vectors.embed_one = orig_vembed


def sample_db():
    """Fresh in-memory-ish db (tmp file) with a couple of media rows."""
    from alebrije import db as dbmod
    tmp = tempfile.mkdtemp(prefix="alebrije-db-")
    conn = dbmod.connect(os.path.join(tmp, "t.db"))
    return conn, tmp


class AlebrijeCase(unittest.TestCase):
    def setUp(self):
        self._home_cm = temp_home()
        self.home = self._home_cm.__enter__()
        self.addCleanup(lambda: self._home_cm.__exit__(None, None, None))

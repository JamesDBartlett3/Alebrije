"""Vector storage + cosine similarity — plain float32 blobs in SQLite."""

from __future__ import annotations

import array
import math

from . import db, llm


def encode(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def decode(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return list(a)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_b = 0.0
    norm_a = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def embed_one(cfg: dict, text: str) -> list[float]:
    return llm.embed(cfg, [text])[0]


def media_search_text(conn, mid: int) -> str:
    """The text whose embedding represents a memory: caption + tags + people."""
    row = conn.execute("SELECT caption, tags, place FROM media WHERE id=?", (mid,)).fetchone()
    if not row:
        return ""
    parts = [row["caption"] or ""]
    parts += db.jload(row["tags"])
    if row["place"]:
        parts.append(f"taken at {row['place']}")
    names = [r["name"] for r in conn.execute(
        "SELECT p.name FROM person_tags t JOIN people p ON p.id=t.person_id WHERE t.media_id=?",
        (mid,))]
    parts += [f"featuring {n}" for n in names]
    return ". ".join(p for p in parts if p)


def index_media(conn, cfg: dict, limit: int = 0, log=print) -> int:
    """(Re)compute embeddings for tagged media that lacks one."""
    rows = conn.execute(
        "SELECT m.id FROM media m LEFT JOIN embeddings e ON e.media_id=m.id "
        "WHERE m.tagged=1 AND e.media_id IS NULL ORDER BY m.id"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    for r in rows:
        text = media_search_text(conn, r["id"])
        if not text:
            continue
        vec = embed_one(cfg, text)
        conn.execute("INSERT OR REPLACE INTO embeddings (media_id, vec) VALUES (?,?)",
                     (r["id"], encode(vec)))
        conn.commit()
        log(f"🧠 indexed media #{r['id']}")
    return len(rows)


def index_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]

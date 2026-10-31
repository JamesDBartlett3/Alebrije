"""SQLite schema and tiny helpers. One database file, zero ORM."""

from __future__ import annotations

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    filename    TEXT NOT NULL,
    media_type  TEXT NOT NULL,            -- image | video | audio
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    fingerprint TEXT NOT NULL,            -- size+mtime+head/tail hash, for change detection
    created_at  TEXT,                     -- EXIF DateTimeOriginal or file mtime, ISO
    gps_lat     REAL,                     -- EXIF GPS, if present
    gps_lon     REAL,
    place       TEXT,                     -- reverse-geocoded place name
    tagged      INTEGER NOT NULL DEFAULT 0,
    caption     TEXT,
    tags        TEXT,                     -- JSON array of object/clothing/animal/place tags
    mood        TEXT,
    people_desc TEXT,                     -- JSON array of raw vision-model person descriptions
    error       TEXT
);
CREATE TABLE IF NOT EXISTS people (
    id            INTEGER PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    description   TEXT NOT NULL,          -- vision-model description of this person
    ref_media     TEXT,                   -- path of the sample photo they were introduced from
    face_centroid BLOB,                   -- L2-normalized face embedding (0 = no face yet)
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS person_tags (
    media_id    INTEGER NOT NULL REFERENCES media(id),
    person_id   INTEGER NOT NULL REFERENCES people(id),
    PRIMARY KEY (media_id, person_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    media_id    INTEGER PRIMARY KEY REFERENCES media(id),
    vec         BLOB NOT NULL            -- float32 little-endian, 768-d nomic
);
CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY,
    media_id    INTEGER NOT NULL REFERENCES media(id),
    person_id   INTEGER REFERENCES people(id),   -- NULL until matched/named
    bbox_x      INTEGER NOT NULL,
    bbox_y      INTEGER NOT NULL,
    bbox_w      INTEGER NOT NULL,
    bbox_h      INTEGER NOT NULL,
    det_score   REAL NOT NULL,
    embedding   BLOB NOT NULL,            -- float32 little-endian, 512-d AdaFace
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type);
CREATE INDEX IF NOT EXISTS idx_media_tagged ON media(tagged);
CREATE INDEX IF NOT EXISTS idx_faces_media ON faces(media_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that postdate v0.1.0 databases, in place."""
    people_cols = {r["name"] for r in conn.execute("PRAGMA table_info(people)")}
    if "face_centroid" not in people_cols:
        conn.execute("ALTER TABLE people ADD COLUMN face_centroid BLOB")
        conn.commit()


def connect(path) -> sqlite3.Connection:
    # check_same_thread=False: the web UI answers requests from worker
    # threads; sqlite serializes at the OS level and every write commits.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def jload(text, default=None):
    if not text:
        return default if default is not None else []
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


def media_row_to_dict(row: sqlite3.Row, include_tags: bool = True) -> dict:
    d = {
        "id": row["id"],
        "path": row["path"],
        "filename": row["filename"],
        "media_type": row["media_type"],
        "created_at": row["created_at"],
        "place": row["place"],
        "gps": {"lat": row["gps_lat"], "lon": row["gps_lon"]} if row["gps_lat"] is not None else None,
        "caption": row["caption"],
        "mood": row["mood"],
        "tagged": bool(row["tagged"]),
    }
    if include_tags:
        d["tags"] = jload(row["tags"])
        d["people_desc"] = jload(row["people_desc"])
    if row["error"]:
        d["error"] = row["error"]
    return d

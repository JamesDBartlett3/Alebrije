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
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,            -- vision-model description of this person
    ref_media   TEXT,                     -- path of the sample photo they were introduced from
    created_at  TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type);
CREATE INDEX IF NOT EXISTS idx_media_tagged ON media(tagged);
"""


def connect(path) -> sqlite3.Connection:
    # check_same_thread=False: the web UI answers requests from worker
    # threads; sqlite serializes at the OS level and every write commits.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
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

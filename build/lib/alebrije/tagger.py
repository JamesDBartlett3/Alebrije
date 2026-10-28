"""The tagging pass: the open-weight vision model meets every memory.

For each image (or video keyframe), Gemma 3 returns strict JSON:
caption, objects, clothing, animals, food, mood, and — crucially — a
description of every person present. Those person descriptions are what
the people book (people.py) matches against; the model never invents
names, it only describes what it sees.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile

from . import db, llm

TAG_PROMPT = (
    "You are a warm, respectful family archivist. Look at this photo and reply "
    "with ONLY a JSON object (no prose, no markdown fences) shaped exactly like:\n"
    '{"caption": "one warm sentence describing the moment",\n'
    ' "people": [{"description": "visible traits only: approximate age, hair, '
    'clothing, distinguishing features. NEVER guess or invent a name", '
    '"position": "left|center|right|background"}],\n'
    ' "objects": ["notable objects, max 8"],\n'
    ' "clothing": ["notable articles of clothing"],\n'
    ' "animals": ["animals, breeds if obvious"],\n'
    ' "food": ["foods or drinks"],\n'
    ' "mood": "two or three words",\n'
    ' "setting": "indoor/outdoor + kind of place"}\n'
    "Omit any key you have nothing for (use [] for empty lists). "
    "Focus on happy, human details: celebrations, hugs, pets, cakes, beaches, "
    "birthday hats. Never mention violence, illness, or anything grim — this "
    "archive is a place of good memories."
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def filename_tags(path: str) -> list[str]:
    """Cheap tags from the filename for audio/video we can't vision-tag."""
    stem = os.path.splitext(os.path.basename(path))[0]
    words = _WORD_RE.findall(stem)
    stop = {"img", "vid", "dsc", "dscn", "p", "pxl", "mov", "final", "copy",
            "edited", "whatsapp", "video", "audio", "the", "and", "for"}
    return [w.lower() for w in words if w.lower() not in stop and not w.isdigit()][:10]


def extract_frame(video_path: str, at_sec: float = 3.0) -> str | None:
    """Grab a representative frame with ffmpeg, if it's installed."""
    if not shutil.which("ffmpeg"):
        return None
    fd, out = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at_sec),
             "-i", video_path, "-frames:v", "1", "-vf", "scale=1024:-2", out],
            capture_output=True, timeout=120)
        if proc.returncode == 0 and os.path.getsize(out) > 1024:
            return out
        # very short clips: retry from the start
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
             "-frames:v", "1", "-vf", "scale=1024:-2", out],
            capture_output=True, timeout=120)
        if proc.returncode == 0 and os.path.getsize(out) > 1024:
            return out
        return None
    except (OSError, subprocess.TimeoutExpired):
        return None


def normalize_tag_result(raw: dict, media_type: str, path: str) -> dict:
    """Coerce the model's JSON into a safe, stored shape."""

    def _strs(v, cap=10):
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        return [str(x).strip()[:120] for x in v
                if x is not None and str(x).strip()][:cap]

    people = []
    for p in raw.get("people") or []:
        if isinstance(p, dict) and str(p.get("description", "")).strip():
            people.append({"description": str(p["description"]).strip()[:300],
                           "position": str(p.get("position", "")).strip()[:20]})
    caption = str(raw.get("caption", "")).strip()[:400]
    if not caption and media_type != "image":
        caption = "Media file from the archive"
    tags = (_strs(raw.get("objects")) + _strs(raw.get("clothing"))
            + _strs(raw.get("animals")) + _strs(raw.get("food")))
    mood = str(raw.get("mood", "")).strip()[:60]
    setting = str(raw.get("setting", "")).strip()[:120]
    if setting:
        tags.append(f"setting: {setting}")
    return {
        "caption": caption,
        "people": people,
        "tags": tags,
        "mood": mood,
        "setting": setting,
    }


def tag_media(conn, cfg: dict, row, log=print) -> bool:
    """Tag one media row. Returns True if something was written."""
    mid, path, mtype = row["id"], row["path"], row["media_type"]
    frame = None
    try:
        if mtype == "image":
            vision_path = path
        elif mtype == "video":
            vision_path = frame = extract_frame(path)
            if not frame:
                raise llm.LlmError("unsupported", "ffmpeg not available for video frames")
        else:
            raise llm.LlmError("unsupported", f"no vision for {mtype}")
        raw = llm.extract_json(llm.describe_image(cfg, vision_path, TAG_PROMPT))
    except llm.LlmError as exc:
        if exc.kind == "unsupported":
            tags = filename_tags(path)
            conn.execute(
                "UPDATE media SET tagged=1, caption=?, tags=?, people_desc=? WHERE id=?",
                (f"{mtype.capitalize()}: {os.path.basename(path)}",
                 json.dumps(tags), "[]", mid))
            conn.commit()
            log(f"• {os.path.basename(path)} — filename tags only ({exc.detail})")
            return True
        conn.execute("UPDATE media SET error=? WHERE id=?", (exc.detail[:300], mid))
        conn.commit()
        log(f"⚠️  {os.path.basename(path)} — {exc.kind}: {exc.detail[:120]}")
        return False
    finally:
        if frame and os.path.isfile(frame):
            os.remove(frame)

    result = normalize_tag_result(raw, mtype, path)
    conn.execute(
        "UPDATE media SET tagged=1, caption=?, tags=?, mood=?, people_desc=?, error=NULL WHERE id=?",
        (result["caption"],
         json.dumps(result["tags"]),
         result["mood"],
         json.dumps(result["people"]), mid))
    conn.commit()
    who = f" ({len(result['people'])} people)" if result["people"] else ""
    log(f"🌼 {os.path.basename(path)}{who}")
    return True


def tag_all(conn, cfg: dict, limit: int = 0, log=print) -> dict:
    rows = conn.execute(
        "SELECT id, path, media_type FROM media WHERE tagged=0 ORDER BY created_at, id"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    done = failed = 0
    for row in rows:
        if tag_media(conn, cfg, row, log):
            done += 1
        else:
            failed += 1
    return {"tagged": done, "failed": failed, "remaining": count_untagged(conn)}


def count_untagged(conn) -> int:
    return conn.execute("SELECT COUNT(*) c FROM media WHERE tagged=0").fetchone()["c"]

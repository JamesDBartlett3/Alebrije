"""The people book: names for the faces, learned from you.

Identity comes from face embeddings: SCRFD detects each face, AdaFace
turns it into a 512-d vector (faces.py), and a face belongs to whoever's
centroid it is closest to. Three bands:

1. cosine >= face_auto_threshold  -> assigned silently;
2. face_gray_low .. auto          -> the face crop is shown to the vision
   model (Gemma) along with the person's description for a yes/no check;
3. below face_gray_low            -> no match.

Leftover faces that match nobody cluster together ("Unnamed #N" people)
so you can name a recurring stranger once. The Gemma *descriptions*
("older woman, white braid, jade earrings") remain the human-readable
layer of every person, and media where no face could be detected at all
still falls back to description matching. Every match is stored as a
reviewable tag, never a silent overwrite.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from . import db, faces, llm, vectors
from .vectors import cosine

UNNAMED_PREFIX = "Unnamed #"

MATCH_PROMPT = (
    "You are matching people in a family photo archive.\n"
    "Known people (name — description):\n{roster}\n"
    "People visible in the new photo (descriptions):\n{seen}\n"
    "Reply with ONLY a JSON array, one entry per seen person, in the same "
    "order: [\n  {{\"index\": 0, \"name\": \"<a known name, or null if none "
    "match or the roster is empty>\"}}\n]\n"
    "Match only when description traits clearly agree (age, hair, clothing, "
    "distinguishing features). A person whose description mentions no people "
    "traits at all (a pet, a distant figure, scenery) must be null. When "
    "unsure, use null — never invent a name."
)

REVIEW_PROMPT = (
    "You are confirming identities in a family photo archive.\n"
    "The person we are looking for is described as: {desc}\n"
    "Look at this photo of one face. Reply with ONLY a JSON object: "
    "{{\"same\": true}} if it plausibly shows that same individual, or "
    "{{\"same\": false}} if it clearly shows someone else. Hairstyles, "
    "glasses and age change between photos — judge by the face itself. "
    "Answer true only when the resemblance is genuinely strong."
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def add_person(conn, name: str, description: str, ref_media: str | None = None,
               face_emb: list[float] | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO people (name, description, ref_media, face_centroid, created_at)"
        " VALUES (?,?,?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET description=excluded.description,"
        " ref_media=COALESCE(excluded.ref_media, people.ref_media),"
        " face_centroid=COALESCE(excluded.face_centroid, people.face_centroid)",
        (name.strip(), description.strip(), ref_media,
         vectors.encode(face_emb) if face_emb is not None else None, now_iso()))
    conn.commit()
    row = conn.execute("SELECT id FROM people WHERE name=?", (name.strip(),)).fetchone()
    return row["id"]


def list_people(conn) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM people ORDER BY name"):
        n_media = conn.execute(
            "SELECT COUNT(*) c FROM person_tags WHERE person_id=?",
            (row["id"],)).fetchone()["c"]
        n_faces = conn.execute(
            "SELECT COUNT(*) c FROM faces WHERE person_id=?",
            (row["id"],)).fetchone()["c"]
        out.append({"id": row["id"], "name": row["name"],
                    "description": row["description"], "tagged_media": n_media,
                    "face_count": n_faces,
                    "unnamed": row["name"].startswith(UNNAMED_PREFIX)})
    return out


def roster_text(people: list[dict]) -> str:
    return "\n".join(f"- {p['name']}: {p['description']}" for p in people) or "(none yet)"


# ---------------------------------------------------------------------------
# face-embedding matching


def refresh_centroid(conn, pid: int) -> None:
    """Recompute one person's centroid from their assigned faces."""
    rows = conn.execute("SELECT embedding FROM faces WHERE person_id=?",
                        (pid,)).fetchall()
    if not rows:
        conn.execute("UPDATE people SET face_centroid=NULL WHERE id=?", (pid,))
    else:
        c = faces.mean_centroid([vectors.decode(r["embedding"]) for r in rows])
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (vectors.encode(c.tolist()), pid))
    conn.commit()


def _centroids(conn) -> list[tuple[int, list[float]]]:
    return [(r["id"], vectors.decode(r["face_centroid"])) for r in
            conn.execute("SELECT id, face_centroid FROM people "
                         "WHERE face_centroid IS NOT NULL")]


def _best_match(vec: list[float],
                centroids: list[tuple[int, list[float]]]) -> tuple[int | None, float]:
    """(person_id, cosine) of the closest centroid, or (None, 0.0)."""
    best_pid, best = None, 0.0
    for pid, cvec in centroids:
        s = cosine(vec, cvec)
        if s > best:
            best_pid, best = pid, s
    return best_pid, best


def rebuild_media_tags(conn, mid: int) -> None:
    """person_tags for a media row = exactly its assigned faces."""
    conn.execute("DELETE FROM person_tags WHERE media_id=?", (mid,))
    for r in conn.execute("SELECT DISTINCT person_id FROM faces "
                          "WHERE media_id=? AND person_id IS NOT NULL", (mid,)):
        conn.execute("INSERT OR IGNORE INTO person_tags (media_id, person_id)"
                     " VALUES (?,?)", (mid, r["person_id"]))
    conn.commit()


def assign_known_faces(conn, cfg: dict, log=print) -> dict:
    """Silently assign every unassigned face whose best centroid is close."""
    centroids = _centroids(conn)
    if not centroids:
        return {"assigned": 0, "gray": 0}
    auto = cfg["face_auto_threshold"]
    assigned = touched = 0
    media_seen = set()
    for row in conn.execute("SELECT id, media_id, embedding FROM faces "
                            "WHERE person_id IS NULL").fetchall():
        pid, sim = _best_match(vectors.decode(row["embedding"]), centroids)
        if pid is not None and sim >= auto:
            conn.execute("UPDATE faces SET person_id=? WHERE id=?",
                         (pid, row["id"]))
            assigned += 1
            media_seen.add(row["media_id"])
        elif pid is not None:
            touched += 1
    conn.commit()
    for mid in media_seen:
        rebuild_media_tags(conn, mid)
    if assigned:
        log(f"👥 auto-matched {assigned} faces to known people")
    return {"assigned": assigned, "gray": touched}


def adjudicate_gray(conn, cfg: dict, entries: list[dict], log=print) -> int:
    """Gemma confirms the borderline faces, one yes/no per face."""
    from PIL import Image
    confirmed = 0
    media_seen = set()
    for e in entries[:int(cfg.get("face_adjudicate_limit", 100))]:
        mrow = conn.execute("SELECT path FROM media WHERE id=?",
                            (e["media_id"],)).fetchone()
        frow = conn.execute(
            "SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM faces WHERE id=?",
            (e["face_id"],)).fetchone()
        prow = conn.execute("SELECT description FROM people WHERE id=?",
                            (e["person_id"],)).fetchone()
        if not (mrow and frow and prow):
            continue
        box = (frow["bbox_x"], frow["bbox_y"],
               frow["bbox_x"] + frow["bbox_w"], frow["bbox_y"] + frow["bbox_h"])
        tmp = None
        try:
            with Image.open(mrow["path"]) as im:
                crop = faces.review_crop(im.convert("RGB"), box)
            if crop is None:
                continue
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            crop.save(tmp, "JPEG", quality=92)
            reply = llm.chat(cfg, [{"role": "user", "content":
                                    REVIEW_PROMPT.format(desc=prow["description"])}],
                             [tmp])
            if llm.extract_json(reply).get("same") is True:
                conn.execute("UPDATE faces SET person_id=? WHERE id=?",
                             (e["person_id"], e["face_id"]))
                confirmed += 1
                media_seen.add(e["media_id"])
                log(f"👥 confirmed a match for person #{e['person_id']} "
                    f"(cosine {e['sim']:.2f})")
        except (llm.LlmError, ValueError, OSError) as exc:
            log(f"• could not double-check one face ({str(exc)[:80]})")
        finally:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)
    conn.commit()
    for mid in media_seen:
        rebuild_media_tags(conn, mid)
    return confirmed


def _gray_entries(conn, cfg: dict) -> list[dict]:
    """Unassigned faces in the gray band, most likely first."""
    centroids = _centroids(conn)
    low, auto = cfg["face_gray_low"], cfg["face_auto_threshold"]
    entries = []
    for row in conn.execute("SELECT id, media_id, embedding FROM faces "
                            "WHERE person_id IS NULL").fetchall():
        pid, sim = _best_match(vectors.decode(row["embedding"]), centroids)
        if pid is not None and low <= sim < auto:
            entries.append({"face_id": row["id"], "media_id": row["media_id"],
                            "person_id": pid, "sim": round(sim, 4)})
    entries.sort(key=lambda e: -e["sim"])
    return entries


def cluster_unassigned(conn, cfg: dict, log=print) -> dict:
    """Greedily group leftover faces into Unnamed #N people (>= 2 faces).

    Existing unnamed people act as cluster seeds, so reruns merge into
    them instead of spawning duplicates. Renamed people keep their faces.
    """
    seeds = [{"pid": r["id"], "vec": vectors.decode(r["face_centroid"]),
              "face_ids": []}
             for r in conn.execute(
                 "SELECT id, face_centroid FROM people WHERE name LIKE ? "
                 "AND face_centroid IS NOT NULL", (UNNAMED_PREFIX + "%",))]
    rows = conn.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL "
                        "ORDER BY det_score DESC").fetchall()
    thr = cfg["face_cluster_threshold"]
    for row in rows:
        vec = vectors.decode(row["embedding"])
        best, best_sim = None, 0.0
        for s in seeds:
            sim = cosine(vec, s["vec"])
            if sim > best_sim:
                best, best_sim = s, sim
        if best is not None and best_sim >= thr:
            conn.execute("UPDATE faces SET person_id=? WHERE id=?",
                         (best["pid"], row["id"]))
            best["face_ids"].append(row["id"])
        else:
            seeds.append({"pid": None, "vec": vec, "face_ids": [row["id"]]})
    created = 0
    touched = {s["pid"] for s in seeds if s["pid"] and s["face_ids"]}
    n_unnamed = conn.execute("SELECT COUNT(*) c FROM people WHERE name LIKE ?",
                             (UNNAMED_PREFIX + "%",)).fetchone()["c"]
    for s in seeds:
        if s["pid"] is None and len(s["face_ids"]) >= 2:
            name = f"{UNNAMED_PREFIX}{n_unnamed + created + 1}"
            while conn.execute("SELECT 1 FROM people WHERE name=?",
                               (name,)).fetchone():
                created += 1
                name = f"{UNNAMED_PREFIX}{n_unnamed + created + 1}"
            conn.execute(
                "INSERT INTO people (name, description, created_at)"
                " VALUES (?, 'unnamed person suggested by face clustering —"
                " rename to name them', ?)",
                (name, now_iso()))
            pid = conn.execute("SELECT id FROM people WHERE name=?",
                               (name,)).fetchone()["id"]
            for fid in s["face_ids"]:
                conn.execute("UPDATE faces SET person_id=? WHERE id=?",
                             (pid, fid))
            s["pid"] = pid
            touched.add(pid)
            created += 1
            log(f"🫂 {len(s['face_ids'])} similar faces look like a new "
                f"person — added {name}")
    conn.commit()
    for pid in touched:
        refresh_centroid(conn, pid)
    return {"clustered": sum(len(s["face_ids"]) for s in seeds if s["pid"]),
            "people": created}


def name_people_in_media(conn, cfg: dict, limit: int = 0, log=print) -> dict:
    """Match people across the archive: embeddings first, Gemma for the
    gray band, descriptions for media where no face could be detected."""
    people_rows = list_people(conn)
    if not people_rows:
        return {"named": 0, "skipped": "no people in the book yet — "
                                       "run `alebrije person add <Name> --photo <sample.jpg>`"}
    res = assign_known_faces(conn, cfg, log=log)
    res["confirmed"] = adjudicate_gray(conn, cfg, _gray_entries(conn, cfg), log=log)
    named = res["assigned"] + res["confirmed"]

    # description fallback: tagged media with described people but no faces
    by_id = {p["id"]: p["name"] for p in people_rows}
    rows = conn.execute(
        "SELECT id, path, people_desc FROM media WHERE tagged=1"
        " AND people_desc != '[]' AND people_desc IS NOT NULL"
        " AND id NOT IN (SELECT media_id FROM faces) ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    for row in rows:
        seen = db.jload(row["people_desc"])
        matches = match_people_in_photo(conn, cfg, seen, people_rows)
        named += apply_person_tags(conn, row["id"], matches)
        names = [by_id[m] for m in matches if m is not None]
        log(f"👥 {row['path']}: {', '.join(names) if names else 'no known people'}")
    res["named"] = named
    res["described"] = len(rows)
    return res


# ---------------------------------------------------------------------------
# description matching (fallback for media without detectable faces)


def match_people_in_photo(conn, cfg: dict, seen: list[dict], people: list[dict]) -> list[int | None]:
    """Map each seen-person description to a person id (or None), in order."""
    if not seen:
        return []
    if not people:
        return [None] * len(seen)
    seen_lines = "\n".join(
        f"{i}. {p['description']} ({p.get('position') or 'in frame'})"
        for i, p in enumerate(seen))
    prompt = MATCH_PROMPT.format(roster=roster_text(people), seen=seen_lines)
    try:
        arr = json.loads(llm.chat(cfg, [{"role": "user", "content": prompt}]))
        by_name = {p["name"].lower(): p["id"] for p in people}
        out: list[int | None] = []
        for item in arr if isinstance(arr, list) else []:
            name = item.get("name") if isinstance(item, dict) else None
            out.append(by_name.get(str(name).lower()) if name else None)
        while len(out) < len(seen):
            out.append(None)
        return out[:len(seen)]
    except (llm.LlmError, json.JSONDecodeError):
        return _match_by_embeddings(cfg, seen, people)


def _match_by_embeddings(cfg: dict, seen: list[dict], people: list[dict]) -> list[int | None]:
    try:
        seen_vecs = [vectors.embed_one(cfg, p["description"]) for p in seen]
        people_vecs = [(p["id"], vectors.embed_one(cfg, p["description"])) for p in people]
    except llm.LlmError:
        return [None] * len(seen)
    out = []
    for sv in seen_vecs:
        best_id, best = None, 0.0
        for pid, pv in people_vecs:
            score = cosine(sv, pv)
            if score > best:
                best_id, best = pid, score
        out.append(best_id if best > 0.5 else None)
    return out


def apply_person_tags(conn, mid: int, matches: list[int | None]) -> int:
    conn.execute("DELETE FROM person_tags WHERE media_id=?", (mid,))
    for pid in matches:
        if pid is not None:
            conn.execute("INSERT OR IGNORE INTO person_tags (media_id, person_id) VALUES (?,?)",
                         (mid, pid))
    conn.commit()
    return sum(1 for m in matches if m is not None)


def media_people(conn, mid: int) -> list[str]:
    return [r["name"] for r in conn.execute(
        "SELECT p.name FROM person_tags t JOIN people p ON p.id=t.person_id "
        "WHERE t.media_id=? ORDER BY p.name", (mid,))]

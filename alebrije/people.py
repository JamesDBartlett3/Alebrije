"""The people book: names for the faces, learned from you.

Alebrije never guesses who someone is on its own. The vision model only
*describes* people ("older woman, white braid, jade earrings"). You then
introduce real people from sample photos (`alebrije person add Abuela`),
which produces a reference description. Matching new photos happens two
ways, both local:

1. in-context — the model sees your small roster of descriptions alongside
   each new photo and picks the best match (or "unknown");
2. embedding fallback — description vectors (nomic) compared by cosine.

Every match is stored as a reviewable tag, never a silent overwrite.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import db, llm, vectors
from .vectors import cosine

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


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def add_person(conn, name: str, description: str, ref_media: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO people (name, description, ref_media, created_at) VALUES (?,?,?,?) "
        "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
        "ref_media=COALESCE(excluded.ref_media, people.ref_media)",
        (name.strip(), description.strip(), ref_media, now_iso()))
    conn.commit()
    return cur.lastrowid


def list_people(conn) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM people ORDER BY name"):
        n = conn.execute(
            "SELECT COUNT(*) c FROM person_tags WHERE person_id=?", (row["id"],)).fetchone()["c"]
        out.append({"id": row["id"], "name": row["name"],
                    "description": row["description"], "tagged_media": n})
    return out


def roster_text(people: list[dict]) -> str:
    return "\n".join(f"- {p['name']}: {p['description']}" for p in people) or "(none yet)"


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


def name_people_in_media(conn, cfg: dict, limit: int = 0, log=print) -> dict:
    """Run roster matching over every tagged-but-unmatched media row."""
    people = list_people(conn)
    if not people:
        return {"named": 0, "skipped": "no people in the book yet — "
                                       "run `alebrije person add <Name> --photo <sample.jpg>`"}
    rows = conn.execute(
        "SELECT id, path, people_desc FROM media WHERE tagged=1 AND people_desc != '[]'"
        " AND people_desc IS NOT NULL ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    named = 0
    by_id = {p["id"]: p["name"] for p in people}
    for row in rows:
        seen = db.jload(row["people_desc"])
        matches = match_people_in_photo(conn, cfg, seen, people)
        named += apply_person_tags(conn, row["id"], matches)
        names = [by_id[m] for m in matches if m is not None]
        log(f"👥 {row['path']}: {', '.join(names) if names else 'no known people'}")
    return {"named": named, "photos": len(rows)}


def media_people(conn, mid: int) -> list[str]:
    return [r["name"] for r in conn.execute(
        "SELECT p.name FROM person_tags t JOIN people p ON p.id=t.person_id "
        "WHERE t.media_id=? ORDER BY p.name", (mid,))]

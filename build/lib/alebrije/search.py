"""Natural-language search over the archive, with sortable results."""

from __future__ import annotations

from . import db, llm, vectors


def search(conn, cfg: dict, query: str, sort: str | None = "relevance",
           media_type: str | None = None, person: str | None = None,
           limit: int = 60) -> dict:
    """Return {mode, results: [{...media..., score, people}]}.

    With a query string and an embedder available, rank by cosine against
    the stored memory embeddings. If the embedder is down or the query is
    empty, fall back to recency (unless the caller picked a sort).
    Sort modes: relevance | date | type | name.
    """
    qvec = None
    if not query.strip():
        if sort == "relevance":
            sort = "date"  # no query: recency is the only meaningful order
    else:
        try:
            qvec = vectors.embed_one(cfg, query)
        except llm.LlmError:
            pass
    return _search(conn, cfg, query, qvec, sort, media_type, person, limit)


def _search(conn, cfg, query, qvec, sort, media_type, person, limit) -> dict:
    sql = ("SELECT m.* FROM media m")
    where, params = [], []
    if media_type:
        where.append("m.media_type=?")
        params.append(media_type)
    if person:
        sql += " JOIN person_tags pt ON pt.media_id=m.id JOIN people pp ON pp.id=pt.person_id"
        where.append("pp.name LIKE ?")
        params.append(f"%{person}%")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY m.created_at DESC, m.id DESC LIMIT 4000"
    rows = conn.execute(sql, params).fetchall()

    if qvec is not None:
        vecs = {r["media_id"]: vectors.decode(r["vec"])
                for r in conn.execute("SELECT media_id, vec FROM embeddings")}
        scored = []
        for row in rows:
            v = vecs.get(row["id"])
            if v is not None or qvec is None:
                score = vectors.cosine(qvec, v) if v else 0.0
                scored.append((score, row))
            else:
                scored.append((-1.0, row))
        default_sort = "relevance"
    else:
        scored = [(1.0, row) for row in rows]
        default_sort = "date"

    sort = sort or default_sort
    if sort == "relevance":
        scored.sort(key=lambda t: -t[0])
    elif sort == "name":
        scored.sort(key=lambda t: t[1]["filename"].lower())
    elif sort == "type":
        scored.sort(key=lambda t: (t[1]["media_type"], -(t[0])))
    # "date" keeps the created_at DESC order from SQL

    out = []
    for score, row in scored[:limit]:
        d = db.media_row_to_dict(row)
        d["people"] = [r["name"] for r in conn.execute(
            "SELECT p.name FROM person_tags t JOIN people p ON p.id=t.person_id "
            "WHERE t.media_id=? ORDER BY p.name", (row["id"],))]
        d["score"] = round(score, 4) if score >= 0 else None
        out.append(d)
    return {"mode": "vector" if qvec is not None else "recent",
            "sort": sort, "results": out}

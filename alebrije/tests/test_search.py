import json
import unittest

from alebrije import search, vectors

from . import FakeLlm, patched_llm, sample_config


def _seed():
    from alebrije.tests import sample_db
    conn, tmp = sample_db()
    rows = [
        ("/a.jpg", "a.jpg", "image", "2021-11-01T10:00:00", "abuela at the beach",
         ["beach", "sombrero"], "Abuela"),
        ("/b.mp4", "b.mp4", "video", "2022-01-15T18:00:00", "birthday cake candles",
         ["cake"], None),
        ("/c.mp3", "c.mp3", "audio", "2023-03-03T08:00:00", "abuela singing",
         ["song"], None),
        ("/d.jpg", "d.jpg", "image", "2024-06-06T12:00:00", "dog in a costume",
         ["dog", "costume"], None),
    ]
    for path, fn, mtype, created, caption, tags, person in rows:
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " created_at, caption, tags, tagged) VALUES (?,?,?,?,?,?,?,?,?,1)",
            (path, fn, mtype, 1, 1.0, "fp" + fn, created, caption,
             json.dumps(tags)))
        if person:
            pid = conn.execute(
                "INSERT INTO people (name, description, created_at)"
                " VALUES (?,'d','now')", (person,)).lastrowid
            conn.execute("INSERT INTO person_tags (media_id, person_id) VALUES (?,?)",
                         (conn.execute("SELECT id FROM media WHERE path=?", (path,)).fetchone()["id"], pid))
    conn.commit()
    return conn


class TestSearch(unittest.TestCase):
    def test_recent_fallback_when_no_query(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "")
        self.assertEqual(res["mode"], "recent")
        self.assertEqual(res["sort"], "date")
        self.assertEqual(res["results"][0]["filename"], "d.jpg")
        conn.close()

    def test_vector_search_ranks_by_relevance(self):
        conn = _seed()
        fake = FakeLlm(embed_fn=lambda t: ([1.0, 0, 0, 0] if "beach" in t else [0, 1.0, 0, 0]))
        with patched_llm(fake):
            # index all four
            vectors.index_media(conn, sample_config(), log=lambda *_: None)
            res = search.search(conn, sample_config(), "beach memories")
        self.assertEqual(res["mode"], "vector")
        self.assertEqual(res["results"][0]["filename"], "a.jpg")
        self.assertGreater(res["results"][0]["score"], 0.9)
        conn.close()

    def test_vector_query_with_unindexed_rows_still_returns_them(self):
        conn = _seed()
        with patched_llm(FakeLlm()):
            res = search.search(conn, sample_config(), "anything")
        # nothing embedded yet: unindexed rows come back unscored, not dropped
        self.assertEqual(len(res["results"]), 4)
        self.assertTrue(all(r["score"] is None for r in res["results"]))
        conn.close()

    def test_sort_by_type(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "", sort="type")
        types = [r["media_type"] for r in res["results"]]
        self.assertEqual(types, sorted(types))
        conn.close()

    def test_sort_by_name(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "", sort="name")
        names = [r["filename"] for r in res["results"]]
        self.assertEqual(names, sorted(names))
        conn.close()

    def test_filter_media_type(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "", media_type="image")
        self.assertTrue(all(r["media_type"] == "image" for r in res["results"]))
        conn.close()

    def test_filter_person(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "", person="Abuela")
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["people"], ["Abuela"])
        conn.close()

    def test_limit(self):
        conn = _seed()
        res = search.search(conn, sample_config(), "", limit=2)
        self.assertEqual(len(res["results"]), 2)
        conn.close()


if __name__ == "__main__":
    unittest.main()

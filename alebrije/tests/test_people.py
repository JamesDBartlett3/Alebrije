import unittest

from alebrije import people

from . import FakeLlm, patched_llm, sample_config


def _db():
    from alebrije.tests import sample_db
    return sample_db()


class TestPeopleBook(unittest.TestCase):
    def test_add_and_list(self):
        conn, tmp = _db()
        people.add_person(conn, "Abuela", "older woman, white braid")
        people.add_person(conn, "Marco", "young man, red sombrero")
        book = people.list_people(conn)
        self.assertEqual([p["name"] for p in book], ["Abuela", "Marco"])
        self.assertEqual(book[0]["description"], "older woman, white braid")
        conn.close()

    def test_add_upserts_description(self):
        conn, tmp = _db()
        people.add_person(conn, "Abuela", "desc v1")
        people.add_person(conn, "Abuela", "desc v2")
        book = people.list_people(conn)
        self.assertEqual(len(book), 1)
        self.assertEqual(book[0]["description"], "desc v2")
        conn.close()


class TestMatching(unittest.TestCase):
    SEEN = [{"description": "older woman, white braid, jade earrings", "position": "left"},
            {"description": "a stranger in a blue jacket", "position": "right"}]

    def test_in_context_match(self):
        conn, tmp = _db()
        abuela = people.add_person(conn, "Abuela", "older woman, white braid, jade earrings")
        fake = FakeLlm(chat_reply='[{"index": 0, "name": "Abuela"}, {"index": 1, "name": null}]')
        with patched_llm(fake):
            out = people.match_people_in_photo(conn, sample_config(), self.SEEN,
                                               people.list_people(conn))
        self.assertEqual(out, [abuela, None])
        self.assertIn("Abuela", fake.chat_calls[0]["messages"][0]["content"])
        conn.close()

    def test_empty_roster_matches_nothing(self):
        conn, tmp = _db()
        with patched_llm(FakeLlm()):
            out = people.match_people_in_photo(conn, sample_config(), self.SEEN, [])
        self.assertEqual(out, [None, None])
        conn.close()

    def test_no_seen_people(self):
        conn, tmp = _db()
        people.add_person(conn, "Abuela", "desc")
        with patched_llm(FakeLlm()):
            self.assertEqual(
                people.match_people_in_photo(conn, sample_config(), [],
                                             people.list_people(conn)), [])
        conn.close()

    def test_model_garbage_falls_back_to_embeddings(self):
        conn, tmp = _db()
        abuela = people.add_person(conn, "Abuela", "older woman white braid")
        people.add_person(conn, "Marco", "young man red sombrero")
        # model returns invalid JSON -> embedding path; identical descriptions match
        fake = FakeLlm(chat_reply="definitely not json")
        with patched_llm(fake):
            out = people.match_people_in_photo(
                conn, sample_config(),
                [{"description": "older woman white braid", "position": ""}],
                people.list_people(conn))
        self.assertEqual(out, [abuela])
        conn.close()

    def test_embedding_fallback_threshold(self):
        conn, tmp = _db()
        a = people.add_person(conn, "A", "aaaa bbbb cccc dddd")
        b = people.add_person(conn, "B", "eeee ffff gggg hhhh")
        fake = FakeLlm(chat_reply="garbage", embed_fn=None)
        with patched_llm(fake):
            out = people.match_people_in_photo(
                conn, sample_config(),
                [{"description": "aaaa bbbb cccc dddd", "position": ""},
                 {"description": "completely unrelated words here", "position": ""}],
                people.list_people(conn))
        self.assertEqual(out[0], a)
        conn.close()


class TestApplyAndName(unittest.TestCase):
    def test_apply_person_tags_idempotent(self):
        conn, tmp = _db()
        pid = people.add_person(conn, "Abuela", "d")
        conn.execute("INSERT INTO media (path, filename, media_type, size, mtime, fingerprint)"
                     " VALUES ('/p.jpg','p.jpg','image',1,1.0,'fp')")
        mid = conn.execute("SELECT id FROM media").fetchone()["id"]
        people.apply_person_tags(conn, mid, [pid])
        people.apply_person_tags(conn, mid, [pid, pid])
        self.assertEqual(people.media_people(conn, mid), ["Abuela"])
        conn.close()

    def test_name_people_walks_archive(self):
        conn, tmp = _db()
        import json as jsonlib
        pid = people.add_person(conn, "Abuela", "older woman, white braid")
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " tagged, people_desc) VALUES ('/p.jpg','p.jpg','image',1,1.0,'fp',1,?)",
            (jsonlib.dumps([{"description": "older woman, white braid",
                             "position": "left"}]),))
        conn.commit()
        fake = FakeLlm(chat_reply='[{"index": 0, "name": "Abuela"}]')
        with patched_llm(fake):
            res = people.name_people_in_media(conn, sample_config(),
                                              log=lambda *_: None)
        self.assertEqual(res["named"], 1)
        self.assertEqual(people.media_people(conn, 1), ["Abuela"])
        conn.close()

    def test_name_people_without_book_skips(self):
        conn, tmp = _db()
        res = people.name_people_in_media(conn, sample_config(), log=lambda *_: None)
        self.assertIn("skipped", res)
        conn.close()


if __name__ == "__main__":
    unittest.main()

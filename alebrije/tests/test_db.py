import unittest

from alebrije import db

from . import sample_db


class TestDb(unittest.TestCase):
    def test_schema_creates_all_tables(self):
        conn, tmp = sample_db()
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"media", "people", "person_tags", "embeddings"} <= names)
        conn.close()

    def test_jload_tolerates_garbage(self):
        self.assertEqual(db.jload(None, []), [])
        self.assertEqual(db.jload("not json", []), [])
        self.assertEqual(db.jload("[1,2]"), [1, 2])

    def test_row_to_dict_shape(self):
        conn, tmp = sample_db()
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " created_at, gps_lat, gps_lon, place, caption, tags, mood, people_desc, tagged)"
            " VALUES ('/x/a.jpg','a.jpg','image',10,1.0,'fp','2020-01-01T00:00:00',"
            "40.0,-105.0,'Golden, Colorado, USA','a moment','[\"cake\"]','happy',"
            "'[{\"description\":\"d\",\"position\":\"left\"}]',1)")
        row = conn.execute("SELECT * FROM media").fetchone()
        d = db.media_row_to_dict(row)
        self.assertEqual(d["tags"], ["cake"])
        self.assertEqual(d["people_desc"][0]["description"], "d")
        self.assertEqual(d["place"], "Golden, Colorado, USA")
        self.assertEqual(d["gps"], {"lat": 40.0, "lon": -105.0})
        conn.close()


if __name__ == "__main__":
    unittest.main()

import unittest

from alebrije import vectors

from . import FakeLlm, patched_llm, sample_config


class TestCosine(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(vectors.cosine([1, 0], [1, 0]), 1.0)

    def test_opposite(self):
        self.assertAlmostEqual(vectors.cosine([1, 0], [-1, 0]), -1.0)

    def test_orthogonal(self):
        self.assertAlmostEqual(vectors.cosine([1, 0], [0, 1]), 0.0)

    def test_scale_invariant(self):
        self.assertAlmostEqual(vectors.cosine([2, 0], [5, 0]), 1.0)

    def test_degenerate(self):
        self.assertEqual(vectors.cosine([], []), 0.0)
        self.assertEqual(vectors.cosine([0, 0], [1, 1]), 0.0)
        self.assertEqual(vectors.cosine([1], [1, 2]), 0.0)


class TestBlobs(unittest.TestCase):
    def test_roundtrip(self):
        vec = [0.25, -1.5, 3.0, 0.0]
        self.assertEqual(vectors.decode(vectors.encode(vec)), vec)


class TestIndexing(unittest.TestCase):
    def test_media_search_text_includes_everything(self):
        conn, tmp = __import__("alebrije.tests", fromlist=["sample_db"]).sample_db()
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " caption, tags, place) VALUES ('/p.jpg','p.jpg','image',1,1.0,'fp',"
            "'a beach party', '[\"cake\",\"sombrero\"]','Playa del Carmen, México')")
        conn.commit()
        text = vectors.media_search_text(conn, 1)
        self.assertIn("beach party", text)
        self.assertIn("cake", text)
        self.assertIn("Playa del Carmen", text)
        conn.close()

    def test_index_media_embeds_only_unindexed_tagged(self):
        conn, tmp = __import__("alebrije.tests", fromlist=["sample_db"]).sample_db()
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " tagged, caption) VALUES ('/p.jpg','p.jpg','image',1,1.0,'fp',1,'moment')")
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
            " tagged, caption) VALUES ('/q.jpg','q.jpg','image',1,1.0,'fq',0,'nope')")
        conn.commit()
        with patched_llm(FakeLlm()):
            n = vectors.index_media(conn, sample_config(), log=lambda *_: None)
        self.assertEqual(n, 1)
        self.assertEqual(vectors.index_count(conn), 1)
        with patched_llm(FakeLlm()):
            n2 = vectors.index_media(conn, sample_config(), log=lambda *_: None)
        self.assertEqual(n2, 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()

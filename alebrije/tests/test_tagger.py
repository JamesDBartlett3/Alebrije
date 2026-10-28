import json
import os
import unittest

from alebrije import tagger
from alebrije.errors import LlmError

from . import (AlebrijeCase, FAKE_TAG_JSON, FakeLlm, patched_llm, png_bytes,
               sample_config, write_media)


class TestNormalize(unittest.TestCase):
    def test_full_shape(self):
        raw = json.loads(FAKE_TAG_JSON)
        out = tagger.normalize_tag_result(raw, "image", "/x/a.jpg")
        self.assertIn("birthday", out["caption"])
        self.assertEqual(len(out["people"]), 2)
        self.assertIn("birthday cake", out["tags"])
        self.assertIn("red sombrero", out["tags"])
        self.assertIn("brown dog", out["tags"])
        self.assertEqual(out["mood"], "joyful")

    def test_strips_garbage(self):
        out = tagger.normalize_tag_result(
            {"caption": 42, "people": "nope", "objects": ["", None, "cup"]},
            "image", "/x/a.jpg")
        self.assertEqual(out["caption"], "42")
        self.assertEqual(out["people"], [])
        self.assertEqual(out["tags"], ["cup"])

    def test_lists_capped(self):
        out = tagger.normalize_tag_result({"objects": [str(i) for i in range(30)]},
                                          "image", "/x/a.jpg")
        self.assertLessEqual(len(out["tags"]), 10)

    def test_setting_becomes_tag(self):
        out = tagger.normalize_tag_result({"setting": "outdoor beach"}, "image", "/p")
        self.assertIn("setting: outdoor beach", out["tags"])


class TestFilenameTags(unittest.TestCase):
    def test_filters_camera_noise(self):
        tags = tagger.filename_tags("/x/IMG_4031 (1).jpg")
        self.assertEqual(tags, [])
        tags = tagger.filename_tags("/x/Abuela birthday beach 2021.jpg")
        self.assertEqual(tags, ["abuela", "birthday", "beach"])

    def test_audio_gets_filename_tags(self):
        tags = tagger.filename_tags("/x/abuela singing las mananitas.mp3")
        self.assertIn("abuela", tags)
        self.assertIn("las", tags)


class TestTagMedia(AlebrijeCase):
    def _insert(self, conn, path, mtype="image"):
        conn.execute(
            "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint)"
            " VALUES (?,?,?,?,?,?)", (path, os.path.basename(path), mtype, 1, 1.0, "fp"))
        conn.commit()
        return conn.execute("SELECT * FROM media").fetchone()

    def test_image_tagged_via_vision(self):
        conn, tmp = __import__("alebrije.tests", fromlist=["sample_db"]).sample_db()
        p = write_media(os.path.join(self.home, "arc"), "beach.jpg")
        row = self._insert(conn, p)
        with patched_llm(FakeLlm()) as fake:
            ok = tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
        self.assertTrue(ok)
        got = conn.execute("SELECT * FROM media WHERE id=?", (row["id"],)).fetchone()
        self.assertEqual(got["tagged"], 1)
        self.assertIn("beach", got["caption"])
        self.assertEqual(len(json.loads(got["people_desc"])), 2)
        conn.close()

    def test_audio_gets_filename_fallback(self):
        from alebrije.tests import sample_db
        conn, tmp = sample_db()
        p = os.path.join(self.home, "arc", "abuela song.mp3")
        write_media(os.path.dirname(p), os.path.basename(p), b"\xff\xfb\x00" + b"x" * 10)
        row = self._insert(conn, p, mtype="audio")
        with patched_llm(FakeLlm()) as fake:
            ok = tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
        self.assertTrue(ok)
        got = conn.execute("SELECT caption, tags FROM media").fetchone()
        self.assertIn("abuela", got["tags"])
        conn.close()

    def test_model_failure_recorded_as_error(self):
        from alebrije.tests import sample_db
        conn, tmp = sample_db()
        p = write_media(os.path.join(self.home, "arc"), "mystery.jpg")
        row = self._insert(conn, p)

        def boom(*a, **k):
            raise LlmError("timeout", "slow model")

        with patched_llm(FakeLlm()) as fake:
            import alebrije.llm as llm
            orig = llm.describe_image
            llm.describe_image = boom
            try:
                ok = tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
            finally:
                llm.describe_image = orig
        self.assertFalse(ok)
        got = conn.execute("SELECT error, tagged FROM media").fetchone()
        self.assertEqual(got["tagged"], 0)
        self.assertIn("slow model", got["error"])
        conn.close()

    def test_tag_all_counts(self):
        from alebrije.tests import sample_db
        conn, tmp = sample_db()
        arc = os.path.join(self.home, "arc")
        for n in ("a.jpg", "b.jpg"):
            write_media(arc, n)
            row = self._insert(conn, os.path.join(arc, n))
        with patched_llm(FakeLlm()):
            res = tagger.tag_all(conn, sample_config(), log=lambda *_: None)
        self.assertEqual(res["tagged"], 2)
        self.assertEqual(res["remaining"], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()

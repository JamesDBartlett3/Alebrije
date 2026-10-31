import json
import os
import unittest

import numpy as np

from alebrije import cli, config, db, faces, people, tagger
from alebrije.faces import ARCFACE_DST

from . import (AlebrijeCase, FakeFaces, FakeLlm, patched_faces, patched_llm,
               png_bytes, sample_config, write_media)


def _db():
    from alebrije.tests import sample_db
    return sample_db()


def _unit(vals):
    v = np.asarray(vals, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mat_apply(m, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ np.asarray(m)[:, :2].T + np.asarray(m)[:, 2]


def _insert_media(conn, path, mtype="image"):
    conn.execute(
        "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint)"
        " VALUES (?,?,?,?,?,?)", (path, os.path.basename(path), mtype, 1, 1.0, "fp"))
    conn.commit()
    return conn.execute("SELECT id FROM media ORDER BY id DESC").fetchone()["id"]


def _insert_face(conn, mid, vec, det=0.9, pid=None):
    conn.execute(
        "INSERT INTO faces (media_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,"
        " det_score, embedding, created_at) VALUES (?,?,1,1,50,50,?,?, 'now')",
        (mid, pid, det, db_str(vec)))
    conn.commit()
    return conn.execute("SELECT id FROM faces ORDER BY id DESC").fetchone()["id"]


def db_str(vec):
    from alebrije import vectors
    return vectors.encode(vec)


# ---------------------------------------------------------------------------
# pure geometry — no models


class TestLetterbox(unittest.TestCase):
    def test_scales_and_centers(self):
        lb = faces.letterbox(1920, 1080)
        self.assertEqual((lb["rw"], lb["rh"]), (640, 360))
        self.assertEqual((lb["dx"], lb["dy"]), (0, 140))
        # round trip a coordinate
        sixforty_x = 960 * lb["scale"] + lb["dx"]
        self.assertEqual(faces.to_orig(sixforty_x, lb["dx"], lb["scale"]), 960)

    def test_small_images_not_upscaled_for_tiles(self):
        lb = faces.letterbox(320, 200, allow_upscale=False)
        self.assertEqual(lb["scale"], 1.0)
        self.assertEqual((lb["rw"], lb["rh"]), (320, 200))

    def test_portrait(self):
        lb = faces.letterbox(1080, 1920)
        self.assertEqual((lb["rw"], lb["rh"]), (360, 640))


class TestTileGrid(unittest.TestCase):
    def test_small_image_single_tile(self):
        self.assertEqual(faces.tile_grid(640, 640), [(0, 0, 640, 640)])
        self.assertEqual(faces.tile_grid(100, 100), [(0, 0, 100, 100)])

    def test_large_image_covered(self):
        w, h = 4000, 3000
        tiles = faces.tile_grid(w, h)
        for tx, ty, tw, th in tiles:
            self.assertLessEqual(tw, faces.DET_SIZE)
            self.assertLessEqual(th, faces.DET_SIZE)
            self.assertGreater(tw, 0)
            self.assertGreater(th, 0)
            self.assertGreaterEqual(tx, 0)
            self.assertGreaterEqual(ty, 0)
        # corners covered
        for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            self.assertTrue(any(tx <= cx < tx + tw and ty <= cy < ty + th
                                for tx, ty, tw, th in tiles),
                            f"corner {(cx, cy)} not covered")


class TestScrfdDecode(unittest.TestCase):
    def test_decodes_synthetic_outputs(self):
        stride = 8
        n = (faces.DET_SIZE // stride) ** 2
        anchors = faces._anchor_centers(stride)
        k = 820  # row 10, col 20 -> ax=160, ay=80
        ax, ay = anchors[k]
        # real SCRFD graph order: [scores x3 strides, bboxes x3, kps x3]
        def t(cnt, d):
            return np.zeros((1, cnt, d), np.float32)

        scores8 = t(n, 1)
        scores8[0, k, 0] = 0.9
        bbox8 = t(n, 4)
        # want box [ax-80, ay-160, ax+80, ay+160] (distances / stride)
        bbox8[0, k, :] = [80 / stride, 160 / stride, 80 / stride, 160 / stride]
        kps8 = t(n, 10)
        # kps: (ax-40, ay-80) and (ax+40, ay+80); rest at the anchor
        kps8[0, k, :] = [-40 / stride, -80 / stride, 40 / stride, 80 / stride,
                         0, 0, 0, 0, 0, 0]
        outputs = ([scores8,
                    t((faces.DET_SIZE // 16) ** 2, 1),
                    t((faces.DET_SIZE // 32) ** 2, 1)]
                   + [bbox8,
                      t((faces.DET_SIZE // 16) ** 2, 4),
                      t((faces.DET_SIZE // 32) ** 2, 4)]
                   + [kps8,
                      t((faces.DET_SIZE // 16) ** 2, 10),
                      t((faces.DET_SIZE // 32) ** 2, 10)])
        dets = faces.decode_scrfd(outputs)
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertAlmostEqual(d["det_score"], 0.9)
        self.assertAlmostEqual(d["bbox"][0], ax - 80, places=3)
        self.assertAlmostEqual(d["bbox"][1], ay - 160, places=3)
        self.assertAlmostEqual(d["bbox"][2], ax + 80, places=3)
        self.assertAlmostEqual(d["bbox"][3], ay + 160, places=3)
        self.assertAlmostEqual(d["kps"][0][0], ax - 40, places=3)
        self.assertAlmostEqual(d["kps"][1][1], ay + 80, places=3)

    def test_threshold_filters(self):
        outs = ([np.full((1, (faces.DET_SIZE // 8) ** 2, 1), 0.1, np.float32),
                 np.zeros((1, (faces.DET_SIZE // 16) ** 2, 1), np.float32),
                 np.zeros((1, (faces.DET_SIZE // 32) ** 2, 1), np.float32)]
                + [np.zeros((1, (faces.DET_SIZE // s) ** 2, 4), np.float32)
                   for s in faces.DET_STRIDES]
                + [np.zeros((1, (faces.DET_SIZE // s) ** 2, 10), np.float32)
                   for s in faces.DET_STRIDES])
        self.assertEqual(faces.decode_scrfd(outs), [])


class TestNms(unittest.TestCase):
    def test_suppresses_overlap(self):
        a = {"bbox": [0, 0, 100, 100], "det_score": 0.9, "kps": []}
        b = {"bbox": [5, 5, 105, 105], "det_score": 0.8, "kps": []}
        self.assertEqual([d["det_score"] for d in faces.nms([a, b])], [0.9])

    def test_keeps_disjoint(self):
        a = {"bbox": [0, 0, 100, 100], "det_score": 0.9, "kps": []}
        b = {"bbox": [500, 500, 600, 600], "det_score": 0.8, "kps": []}
        self.assertEqual(len(faces.nms([a, b])), 2)


class TestAlignment(unittest.TestCase):
    def test_identity_on_template(self):
        m = faces.norm_crop_matrix(ARCFACE_DST)
        np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-4)

    def test_recovers_known_transform(self):
        theta = np.deg2rad(30)
        s = 1.7
        r = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        m_true = np.zeros((2, 3))
        m_true[:, :2] = s * r
        m_true[:, 2] = [5, -3]
        src = np.random.RandomState(7).rand(5, 2) * 100
        dst = _mat_apply(m_true, src)
        m = faces.umeyama(src, dst)
        np.testing.assert_allclose(_mat_apply(m, src), dst, atol=1e-6)

    def test_affine_inverse_round_trip(self):
        m = faces.norm_crop_matrix([[10, 20], [80, 25], [45, 70],
                                    [15, 90], [75, 88]])
        inv = faces.affine_inverse(m)
        pts = [[3.0, 4.0], [50.0, 60.0], [111.0, 7.0]]
        np.testing.assert_allclose(_mat_apply(inv, _mat_apply(m, pts)),
                                   pts, atol=1e-6)

    def test_aligned_crop_size(self):
        from PIL import Image
        im = Image.new("RGB", (300, 300))
        kps = ARCFACE_DST + np.array([100, 100])  # face offset into the image
        crop = faces.aligned_crop(im, kps)
        self.assertEqual(crop.size, (faces.REC_SIZE, faces.REC_SIZE))


class TestReviewCrop(unittest.TestCase):
    def test_pads_and_clamps(self):
        from PIL import Image
        im = Image.new("RGB", (100, 100))
        crop = faces.review_crop(im, (20, 20, 60, 60), pad=0.4)
        self.assertEqual(crop.size, (72, 72))  # 40px face + 16px pad each side

    def test_none_for_degenerate(self):
        from PIL import Image
        im = Image.new("RGB", (10, 10))
        self.assertIsNone(faces.review_crop(im, (0, 0, 2, 2)))


class TestMeanCentroid(unittest.TestCase):
    def test_unit_mean(self):
        c = faces.mean_centroid([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])])
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=6)
        np.testing.assert_allclose(c, [0.7071, 0.7071, 0, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# matching over the database (synthetic embeddings, no models)


CFG = sample_config()


class TestAssign(unittest.TestCase):
    def test_auto_band_assigns_and_tags(self):
        conn, tmp = _db()
        vec = _unit([1, 0, 0, 0])
        mid = _insert_media(conn, "/p/a.jpg")
        pid = people.add_person(conn, "Sofi", "young girl")
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (db_str(vec), pid))
        conn.commit()
        _insert_face(conn, mid, vec)
        res = people.assign_known_faces(conn, CFG)
        self.assertEqual(res["assigned"], 1)
        self.assertEqual(people.media_people(conn, mid), ["Sofi"])
        row = conn.execute("SELECT person_id FROM faces").fetchone()
        self.assertEqual(row["person_id"], pid)
        conn.close()

    def test_gray_band_stays_unassigned(self):
        conn, tmp = _db()
        c = _unit([1, 0, 0, 0])
        v = _unit([0.4, 0.9165151, 0, 0])  # cosine 0.4: gray
        mid = _insert_media(conn, "/p/a.jpg")
        pid = people.add_person(conn, "Sofi", "young girl")
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (db_str(c), pid))
        conn.commit()
        _insert_face(conn, mid, v)
        res = people.assign_known_faces(conn, CFG)
        self.assertEqual(res, {"assigned": 0, "gray": 1})
        self.assertEqual(people.media_people(conn, mid), [])
        conn.close()

    def test_low_band_never_matches(self):
        conn, tmp = _db()
        c = _unit([1, 0, 0, 0])
        v = _unit([-1, 0, 0, 0])  # cosine -1
        mid = _insert_media(conn, "/p/a.jpg")
        pid = people.add_person(conn, "Sofi", "young girl")
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (db_str(c), pid))
        conn.commit()
        _insert_face(conn, mid, v)
        res = people.assign_known_faces(conn, CFG)
        self.assertEqual(res["gray"], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM faces WHERE "
                                      "person_id IS NULL").fetchone()["c"], 1)
        conn.close()


class TestAdjudicate(unittest.TestCase):
    def _gray_setup(self):
        conn, tmp = _db()
        c = _unit([1, 0, 0, 0])
        v = _unit([0.4, 0.9165151, 0, 0])
        p = write_media(tmp, "a.jpg", png_bytes(8, 8))
        mid = _insert_media(conn, p)
        pid = people.add_person(conn, "Sofi", "young girl")
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (db_str(c), pid))
        conn.commit()
        fid = _insert_face(conn, mid, v)
        return conn, tmp, pid, fid

    def test_confirm_assigns(self):
        conn, tmp, pid, fid = self._gray_setup()
        fake = FakeLlm(chat_reply='{"same": true}')
        with patched_llm(fake):
            n = people.adjudicate_gray(conn, CFG, [
                {"face_id": fid, "media_id": 1, "person_id": pid, "sim": 0.4}])
        self.assertEqual(n, 1)
        self.assertEqual(conn.execute(
            "SELECT person_id FROM faces WHERE id=?",
            (fid,)).fetchone()["person_id"], pid)
        self.assertEqual(people.media_people(conn, 1), ["Sofi"])
        conn.close()

    def test_reject_leaves_unassigned(self):
        conn, tmp, pid, fid = self._gray_setup()
        fake = FakeLlm(chat_reply='{"same": false}')
        with patched_llm(fake):
            n = people.adjudicate_gray(conn, CFG, [
                {"face_id": fid, "media_id": 1, "person_id": pid, "sim": 0.4}])
        self.assertEqual(n, 0)
        self.assertIsNone(conn.execute(
            "SELECT person_id FROM faces WHERE id=?", (fid,)).fetchone()["person_id"])
        conn.close()


class TestCluster(unittest.TestCase):
    def test_two_similar_faces_become_a_person(self):
        conn, tmp = _db()
        v = _unit([1, 2, 3, 4])
        w = _unit([1.1, 1.9, 3.2, 3.8])  # cosine ~0.999
        mid = _insert_media(conn, "/p/a.jpg")
        _insert_face(conn, mid, v)
        _insert_face(conn, mid, w)
        res = people.cluster_unassigned(conn, CFG)
        self.assertEqual(res["people"], 1)
        book = people.list_people(conn)
        self.assertEqual(len(book), 1)
        self.assertTrue(book[0]["unnamed"])
        self.assertTrue(book[0]["name"].startswith("Unnamed #"))
        self.assertEqual(book[0]["face_count"], 2)
        # centroid stored and faces assigned
        self.assertIsNotNone(conn.execute(
            "SELECT face_centroid FROM people").fetchone()["face_centroid"])
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces WHERE person_id IS NULL").fetchone()["c"], 0)
        conn.close()

    def test_singleton_does_not_become_a_person(self):
        conn, tmp = _db()
        mid = _insert_media(conn, "/p/a.jpg")
        _insert_face(conn, mid, _unit([1, 0, 0, 0]))
        res = people.cluster_unassigned(conn, CFG)
        self.assertEqual(res["people"], 0)
        self.assertEqual(people.list_people(conn), [])
        conn.close()

    def test_rerun_merges_into_existing_unnamed(self):
        conn, tmp = _db()
        mid = _insert_media(conn, "/p/a.jpg")
        v = _unit([1, 2, 3, 4])
        _insert_face(conn, mid, v)
        _insert_face(conn, mid, _unit([1.1, 1.9, 3.2, 3.8]))
        people.cluster_unassigned(conn, CFG)
        _insert_face(conn, mid, _unit([0.9, 2.1, 2.9, 4.1]))
        res = people.cluster_unassigned(conn, CFG)
        self.assertEqual(res["people"], 0)
        names = [p["name"] for p in people.list_people(conn)]
        self.assertEqual(names, ["Unnamed #1"])
        self.assertEqual(people.list_people(conn)[0]["face_count"], 3)
        conn.close()


class TestRefreshCentroid(unittest.TestCase):
    def test_centroid_follows_faces(self):
        conn, tmp = _db()
        pid = people.add_person(conn, "Sofi", "d")
        mid = _insert_media(conn, "/p/a.jpg")
        _insert_face(conn, mid, _unit([1, 0, 0, 0]), pid=pid)
        _insert_face(conn, mid, _unit([0, 1, 0, 0]), pid=pid)
        people.refresh_centroid(conn, pid)
        from alebrije import vectors
        c = vectors.decode(conn.execute(
            "SELECT face_centroid FROM people WHERE id=?", (pid,)).fetchone()["face_centroid"])
        np.testing.assert_allclose(c, [0.7071, 0.7071, 0, 0], atol=1e-4)
        conn.close()

    def test_no_faces_clears_centroid(self):
        conn, tmp = _db()
        pid = people.add_person(conn, "Sofi", "d")
        conn.execute("UPDATE people SET face_centroid=? WHERE id=?",
                     (db_str(_unit([1, 0, 0, 0])), pid))
        conn.commit()
        people.refresh_centroid(conn, pid)
        self.assertIsNone(conn.execute("SELECT face_centroid FROM people "
                                       "WHERE id=?", (pid,)).fetchone()["face_centroid"])
        conn.close()


class TestMigration(unittest.TestCase):
    def test_v010_database_upgrades_in_place(self):
        import sqlite3
        import tempfile
        tmp = tempfile.mkdtemp(prefix="alebrije-mig-")
        path = os.path.join(tmp, "old.db")
        old_schema = """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL, media_type TEXT NOT NULL, size INTEGER NOT NULL,
            mtime REAL NOT NULL, fingerprint TEXT NOT NULL, created_at TEXT,
            gps_lat REAL, gps_lon REAL, place TEXT, tagged INTEGER NOT NULL DEFAULT 0,
            caption TEXT, tags TEXT, mood TEXT, people_desc TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL, ref_media TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS person_tags (
            media_id INTEGER NOT NULL REFERENCES media(id),
            person_id INTEGER NOT NULL REFERENCES people(id),
            PRIMARY KEY (media_id, person_id));
        CREATE TABLE IF NOT EXISTS embeddings (
            media_id INTEGER PRIMARY KEY REFERENCES media(id), vec BLOB NOT NULL);
        """
        old = sqlite3.connect(path)
        old.executescript(old_schema)
        old.execute("INSERT INTO people (name, description, created_at)"
                    " VALUES ('Abuela', 'older woman', 'now')")
        old.commit()
        old.close()
        conn = db.connect(path)  # triggers the migration
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(people)")}
        self.assertIn("face_centroid", cols)
        self.assertIsNotNone(conn.execute(
            "SELECT name FROM people WHERE name='Abuela'").fetchone())
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces").fetchone()["c"], 0)
        conn.close()

    def test_add_person_stores_face_centroid(self):
        conn, tmp = _db()
        people.add_person(conn, "Sofi", "d", face_emb=_unit([3, 4, 0, 0]))
        from alebrije import vectors
        c = vectors.decode(conn.execute("SELECT face_centroid FROM people"
                                        ).fetchone()["face_centroid"])
        np.testing.assert_allclose(c, [0.6, 0.8, 0, 0], atol=1e-6)
        conn.close()


# ---------------------------------------------------------------------------
# pipeline integration (patched face models)


class TestTaggerFaces(AlebrijeCase):
    def test_tag_media_inserts_faces(self):
        conn, tmp = _db()
        p = write_media(os.path.join(self.home, "arc"), "a.jpg")
        _insert_media(conn, p)
        row = conn.execute("SELECT * FROM media").fetchone()
        det = {"bbox": (10, 20, 50, 60), "det_score": 0.9,
               "embedding": _unit([1, 0, 0, 0])}
        with patched_llm(FakeLlm()), patched_faces(FakeFaces({"a.jpg": [det]})):
            ok = tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
        self.assertTrue(ok)
        frow = conn.execute("SELECT * FROM faces").fetchone()
        self.assertIsNotNone(frow)
        self.assertEqual((frow["bbox_x"], frow["bbox_y"],
                          frow["bbox_w"], frow["bbox_h"]), (10, 20, 50, 60))
        conn.close()

    def test_missing_face_stack_never_fails_the_tag(self):
        conn, tmp = _db()
        p = write_media(os.path.join(self.home, "arc"), "a.jpg")
        _insert_media(conn, p)
        row = conn.execute("SELECT * FROM media").fetchone()
        with patched_llm(FakeLlm()), patched_faces(FakeFaces(ok=False)):
            ok = tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
        self.assertTrue(ok)
        self.assertEqual(conn.execute(
            "SELECT tagged FROM media").fetchone()["tagged"], 1)
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces").fetchone()["c"], 0)
        conn.close()

    def test_audio_gets_no_faces(self):
        conn, tmp = _db()
        p = os.path.join(self.home, "arc", "song.mp3")
        write_media(os.path.dirname(p), os.path.basename(p), b"\xff\xfb" + b"x" * 10)
        _insert_media(conn, p, mtype="audio")
        row = conn.execute("SELECT * FROM media").fetchone()
        fake = FakeFaces()  # would match anything if called
        with patched_llm(FakeLlm()), patched_faces(fake):
            tagger.tag_media(conn, sample_config(), row, log=lambda *_: None)
        self.assertEqual(fake.calls, [])
        conn.close()


class TestCliFaces(AlebrijeCase):
    def test_faces_command_clusters_and_names(self):
        archive = os.path.join(self.home, "arc")
        p1 = write_media(archive, "a.jpg")
        p2 = write_media(archive, "b.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        det1 = {"bbox": (0, 0, 50, 50), "det_score": 0.9,
                "embedding": _unit([1, 2, 3, 4])}
        det2 = {"bbox": (5, 5, 50, 50), "det_score": 0.9,
                "embedding": _unit([1.1, 1.9, 3.2, 3.8])}
        fake = FakeFaces({"a.jpg": [det1], "b.jpg": [det2]})
        with patched_faces(fake):
            rc = cli.main(["faces"])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces").fetchone()["c"], 2)
        names = [r["name"] for r in conn.execute("SELECT name FROM people")]
        self.assertEqual(names, ["Unnamed #1"])
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces WHERE person_id IS NULL"
        ).fetchone()["c"], 0)
        conn.close()

    def test_faces_missing_models_fails_cleanly(self):
        archive = os.path.join(self.home, "arc")
        write_media(archive, "a.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        with patched_faces(FakeFaces(ok=False)):
            rc = cli.main(["faces"])
        self.assertEqual(rc, 1)

    def test_faces_purge(self):
        archive = os.path.join(self.home, "arc")
        p1 = write_media(archive, "a.jpg")
        p2 = write_media(archive, "b.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        det = {"bbox": (0, 0, 50, 50), "det_score": 0.9,
               "embedding": _unit([1, 2, 3, 4])}
        with patched_faces(FakeFaces({"a.jpg": [det], "b.jpg": [dict(det)]})):
            cli.main(["faces"])
        with patched_faces(FakeFaces(ok=False)):
            rc = cli.main(["faces", "--purge"])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) c FROM faces").fetchone()["c"], 0)
        self.assertIsNone(conn.execute(
            "SELECT face_centroid FROM people").fetchone()["face_centroid"])
        conn.close()

    def test_person_add_learns_face_and_matches(self):
        archive = os.path.join(self.home, "arc")
        sample = os.path.join(self.home, "sample.jpg")
        write_media(os.path.dirname(sample), "sample.jpg")
        p = write_media(archive, "a.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        emb = _unit([1, 2, 3, 4])
        det = {"bbox": (0, 0, 50, 50), "det_score": 0.95, "embedding": emb}
        face_fake = FakeFaces({"sample.jpg": [det], "a.jpg": [dict(det)]})
        with patched_faces(face_fake):
            cli.main(["faces"])  # archive faces exist before the person does
        with patched_llm(FakeLlm(chat_reply=json.dumps(
                {"caption": "portrait",
                 "people": [{"description": "young girl", "position": "center"}]}))), \
                patched_faces(face_fake):
            rc = cli.main(["person", "add", "Sofi", "--photo", sample])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        self.assertIsNotNone(conn.execute(
            "SELECT face_centroid FROM people WHERE name='Sofi'"
        ).fetchone()["face_centroid"])
        # the archive photo's identical face got matched immediately
        self.assertEqual(people.media_people(conn, 1), ["Sofi"])
        conn.close()

    def test_doctor_reports_face_stack(self):
        import contextlib
        import io
        archive = os.path.join(self.home, "arc")
        cli.main(["init", archive])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["doctor"])
        self.assertIn("faces:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

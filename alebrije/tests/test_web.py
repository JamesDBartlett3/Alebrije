import json
import os
import threading
import unittest
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from alebrije import db, web

from . import AlebrijeCase, FakeLlm, patched_llm, png_bytes, sample_config, write_media


def _seed_db(conn, archive):
    p = write_media(archive, "beach.jpg")
    conn.execute(
        "INSERT INTO media (path, filename, media_type, size, mtime, fingerprint,"
        " created_at, caption, tags, place, gps_lat, gps_lon, tagged)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)",
        (p, "beach.jpg", "image", 100, 1.0, "fp", "2021-11-02T14:30:00",
         "Abuela at the beach", json.dumps(["beach", "sombrero"]),
         "Civic Center Park — Denver, Colorado, USA", 39.7392, -104.9903))
    pid = conn.execute("INSERT INTO people (name, description, created_at)"
                       " VALUES ('Abuela','older woman, white braid','now')").lastrowid
    mid = conn.execute("SELECT id FROM media").fetchone()["id"]
    conn.execute("INSERT INTO person_tags (media_id, person_id) VALUES (?,?)", (mid, pid))
    vec = bytes(4 * 768)
    conn.execute("INSERT INTO embeddings (media_id, vec) VALUES (?,?)", (mid, vec))
    conn.commit()
    return mid


class _Server:
    def __init__(self, cfg, conn):
        web.BattenHandler.cfg = cfg
        web.BattenHandler.conn = conn
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.BattenHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return r.status, r.read()

    def get_json(self, path):
        status, body = self.get(path)
        return status, json.loads(body)

    def post_json(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", json.dumps(obj).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestWeb(AlebrijeCase):
    def setUp(self):
        super().setUp()
        self.archive = os.path.join(self.home, "arc")
        self.cfg = sample_config(archive_dir=self.archive)
        self.conn = db.connect(os.path.join(self.home, "test.db"))
        _seed_db(self.conn, self.archive)
        self.srv = _Server(self.cfg, self.conn)
        self.addCleanup(self.srv.stop)
        self.addCleanup(self.conn.close)

    def test_index_html_served(self):
        status, body = self.srv.get("/")
        self.assertEqual(status, 200)
        html = body.decode()
        self.assertIn("Alebrije", html)
        self.assertIn("papel", html)  # Día de los Muertos theme present
        self.assertIn("/api/media", html)

    def test_api_media_lists_with_people(self):
        _, res = self.srv.get_json("/api/media")
        self.assertEqual(res["mode"], "recent")
        self.assertEqual(len(res["results"]), 1)
        row = res["results"][0]
        self.assertEqual(row["people"], ["Abuela"])
        self.assertEqual(row["place"], "Civic Center Park — Denver, Colorado, USA")

    def test_api_media_search_param(self):
        _, res = self.srv.get_json("/api/media?type=image")
        self.assertEqual(res["results"][0]["media_type"], "image")

    def test_api_people(self):
        _, res = self.srv.get_json("/api/people")
        self.assertEqual(res["people"][0]["name"], "Abuela")

    def test_api_stats(self):
        _, res = self.srv.get_json("/api/stats")
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["tagged"], 1)
        self.assertEqual(res["with_place"], 1)
        self.assertEqual(res["people"], 1)

    def test_api_file_serves_from_archive(self):
        rel = os.path.relpath(os.path.join(self.archive, "beach.jpg"),
                              self.archive).replace(os.sep, "/")
        status, body = self.srv.get("/api/file?path=" + urllib.parse.quote(rel))
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_api_file_blocks_traversal(self):
        import urllib.error
        evil = urllib.parse.quote("../../../../etc/passwd")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.srv.get(f"/api/file?path={evil}")
        self.assertEqual(ctx.exception.code, 404)

    def test_api_unknown_404(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError):
            self.srv.get_json("/api/nothing")

    def test_post_person_adds_and_matches(self):
        with patched_llm(FakeLlm(chat_reply='[{"index": 0, "name": "Marco"}]')):
            res = self.srv.post_json("/api/person", {
                "name": "Marco", "description": "young man, red sombrero"})
        self.assertTrue(res["ok"])
        _, people = self.srv.get_json("/api/people")
        self.assertEqual(len(people["people"]), 2)

    def test_post_person_requires_fields(self):
        res = self.srv.post_json("/api/person", {"name": ""})
        self.assertFalse(res["ok"])

    def test_post_search(self):
        res = self.srv.post_json("/api/search", {"q": "", "sort": "name"})
        self.assertEqual(res["results"][0]["filename"], "beach.jpg")

    def test_serve_prints_lan_urls(self):
        urls = web._lan_urls(5999)
        for u in urls:
            self.assertTrue(u.startswith("http://"))


if __name__ == "__main__":
    unittest.main()

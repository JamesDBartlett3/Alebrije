import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from alebrije import cli

from . import AlebrijeCase, FakeLlm, patched_llm, png_bytes, sample_config, write_media


class TestParser(unittest.TestCase):
    def test_all_commands_registered(self):
        parser = cli.build_parser()
        for cmd in ("init", "setup", "up", "stop", "scan", "tag", "name",
                    "places", "search", "serve", "doctor", "selftest"):
            args = parser.parse_args([cmd])
            self.assertTrue(hasattr(args, "func"))
        self.assertEqual(parser.parse_args(["person", "add"]).which, "add")

    def test_search_flags(self):
        args = cli.build_parser().parse_args(
            ["search", "abuela", "beach", "--sort", "date", "--type", "image",
             "--person", "Abuela", "--limit", "5"])
        self.assertEqual(args.query, ["abuela", "beach"])
        self.assertEqual(args.sort, "date")


class TestCli(AlebrijeCase):
    def test_init_creates_config_and_db(self):
        archive = os.path.join(self.home, "arc")
        os.makedirs(archive)
        rc = cli.main(["init", archive])
        self.assertEqual(rc, 0)
        from alebrije import config
        cfg = config.load()
        self.assertEqual(cfg["archive_dir"], archive)
        self.assertTrue(config.db_path(cfg).is_file())

    def test_init_rejects_missing_dir(self):
        self.assertEqual(cli.main(["init", "/nonexistent-xyz"]), 1)

    def test_scan_indexes(self):
        archive = os.path.join(self.home, "arc")
        write_media(archive, "a.jpg")
        cli.main(["init", archive])
        rc = cli.main(["scan"])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM media").fetchone()["c"], 1)
        conn.close()

    def test_scan_without_init_fails(self):
        self.assertEqual(cli.main(["scan"]), 1)

    def test_person_add_with_note(self):
        archive = os.path.join(self.home, "arc")
        write_media(archive, "a.jpg")
        cli.main(["init", archive])
        with patched_llm(FakeLlm()):
            rc = cli.main(["person", "add", "Abuela", "--note", "older woman, braid"])
        self.assertEqual(rc, 0)
        rc = cli.main(["person", "list"])
        self.assertEqual(rc, 0)

    def test_person_add_via_photo_uses_vision(self):
        single = json.dumps({"caption": "portrait",
                             "people": [{"description": "older woman, white braid",
                                         "position": "center"}]})
        archive = os.path.join(self.home, "arc")
        p = write_media(archive, "abuela.jpg")
        cli.main(["init", archive])
        with patched_llm(FakeLlm(chat_reply=single)):
            rc = cli.main(["person", "add", "Abuela", "--photo", p])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        names = [r["name"] for r in conn.execute("SELECT name FROM people")]
        self.assertEqual(names, ["Abuela"])
        conn.close()

    def test_person_rename(self):
        archive = os.path.join(self.home, "arc")
        cli.main(["init", archive])
        with patched_llm(FakeLlm()):
            cli.main(["person", "add", "Gma", "--note", "d"])
            rc = cli.main(["person", "rename", "Gma", "Abuela"])
        self.assertEqual(rc, 0)

    def test_tag_runs_pipeline(self):
        archive = os.path.join(self.home, "arc")
        write_media(archive, "a.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        with patched_llm(FakeLlm()):
            rc = cli.main(["tag"])
        self.assertEqual(rc, 0)
        from alebrije import config, db
        conn = db.connect(config.db_path(config.load()))
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM media WHERE tagged=1").fetchone()["c"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"], 1)
        conn.close()

    def test_search_offline_falls_back(self):
        archive = os.path.join(self.home, "arc")
        write_media(archive, "a.jpg")
        cli.main(["init", archive])
        cli.main(["scan"])
        with patched_llm(FakeLlm()):
            cli.main(["tag"])
        rc = cli.main(["search", "anything", "--sort", "date"])
        self.assertEqual(rc, 0)

    def test_doctor_reports_clean(self):
        archive = os.path.join(self.home, "arc")
        cli.main(["init", archive])
        with mock.patch("alebrije.bootstrap.llama_server_path",
                        return_value="/fake/llama-server"), \
             mock.patch("alebrije.servers.is_up",
                        return_value={"vlm": True, "embed": True}), \
             mock.patch("alebrije.vectors.embed_one", return_value=[0.0] * 8), \
             mock.patch("alebrije.llm.chat", return_value="ready"):
            rc = cli.main(["doctor"])
        self.assertEqual(rc, 0)

    def test_selftest_flag_import(self):
        self.assertTrue(callable(cli.cmd_selftest))

    def test_every_invocation_style_works(self):
        """`python -m alebrije`, direct __main__.py, and the launcher all run."""
        import subprocess
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env = dict(os.environ)
        invocations = [
            [sys.executable, "-m", "alebrije"],
            [sys.executable, os.path.join("alebrije", "__main__.py")],
            [sys.executable, "alebrije.py"],
            [sys.executable, os.path.join("alebrije", "cli.py")],
        ]
        for cmd in invocations:
            with self.subTest(cmd=cmd):
                res = subprocess.run(cmd + ["doctor"], cwd=root, env=env,
                                     capture_output=True, text=True, timeout=60)
                self.assertIn("Alebrije doctor", res.stdout)
                # rc 0 = clean machine, 1 = findings (e.g. models not
                # downloaded) — both mean the CLI itself ran fine
                self.assertIn(res.returncode, (0, 1),
                              f"{cmd}: {res.stderr[-300:]}")


if __name__ == "__main__":
    unittest.main()

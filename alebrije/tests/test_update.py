import subprocess
import unittest
from unittest import mock

from alebrije import update


class TestVersionCompare(unittest.TestCase):
    def test_semantic_comparison(self):
        self.assertTrue(update.is_newer("v0.2.0", "0.1.0"))
        self.assertTrue(update.is_newer("v0.1.10", "0.1.9"))
        self.assertFalse(update.is_newer("v0.1.0", "0.1.0"))
        self.assertFalse(update.is_newer("v0.1.0", "0.2.0"))
        self.assertFalse(update.is_newer("v0.1.9", "0.1.10"))

    def test_malformed_tags_do_not_crash(self):
        self.assertFalse(update.is_newer("nope", "0.1.0"))
        self.assertTrue(update.is_newer("v1", "0.9.9"))


class TestLatestRelease(unittest.TestCase):
    def test_offline_returns_none(self):
        with mock.patch("alebrije.bootstrap._fetch_json", side_effect=OSError("down")):
            self.assertIsNone(update.latest_release())

    def test_garbage_returns_none(self):
        with mock.patch("alebrije.bootstrap._fetch_json", return_value=["not", "a", "dict"]):
            self.assertIsNone(update.latest_release())

    def test_no_releases_yet_is_distinguished_from_offline(self):
        import urllib.error
        err = urllib.error.HTTPError(update.RELEASES_API, 404, "no releases",
                                     hdrs=None, fp=None)
        with mock.patch("alebrije.bootstrap._fetch_json", side_effect=err):
            rel = update.latest_release()
        self.assertEqual(rel["tag"], None)  # reachable, but no releases

    def test_server_error_is_offline(self):
        import urllib.error
        err = urllib.error.HTTPError(update.RELEASES_API, 500, "boom",
                                     hdrs=None, fp=None)
        with mock.patch("alebrije.bootstrap._fetch_json", side_effect=err):
            self.assertIsNone(update.latest_release())

    def test_parses_release(self):
        data = {"tag_name": "v0.2.0", "html_url": "https://github.com/x/r/v0.2.0",
                "body": "fixed stuff"}
        with mock.patch("alebrije.bootstrap._fetch_json", return_value=data):
            rel = update.latest_release()
        self.assertEqual(rel["tag"], "v0.2.0")
        self.assertIn("v0.2.0", rel["url"])


class TestPipx(unittest.TestCase):
    def test_not_managed_without_pipx(self):
        with mock.patch("alebrije.update.shutil.which", return_value=None):
            self.assertFalse(update.pipx_managed())

    def test_managed_when_listed(self):
        fake = subprocess.CompletedProcess([], 0, stdout="gadget\nalebrije 0.1.0\n", stderr="")
        with mock.patch("alebrije.update.shutil.which", return_value="/usr/bin/pipx"), \
             mock.patch("alebrije.update.subprocess.run", return_value=fake):
            self.assertTrue(update.pipx_managed())

    def test_not_managed_when_absent_from_list(self):
        fake = subprocess.CompletedProcess([], 0, stdout="gadget 1.0\n", stderr="")
        with mock.patch("alebrije.update.shutil.which", return_value="/usr/bin/pipx"), \
             mock.patch("alebrije.update.subprocess.run", return_value=fake):
            self.assertFalse(update.pipx_managed())

    def test_upgrade_uses_tagged_url(self):
        fake = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch("alebrije.update.subprocess.run", return_value=fake) as run:
            self.assertTrue(update.upgrade_with_pipx("v0.2.0"))
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[:3], ["pipx", "install", "--force"])
        self.assertEqual(cmd[3], update.install_source("v0.2.0"))
        self.assertIn("@v0.2.0", cmd[3])

    def test_upgrade_failure_reported(self):
        fake = subprocess.CompletedProcess([], 1, stdout="", stderr="boom")
        with mock.patch("alebrije.update.subprocess.run", return_value=fake):
            self.assertFalse(update.upgrade_with_pipx("v0.2.0"))


class TestInstructions(unittest.TestCase):
    def test_mentions_both_paths(self):
        text = update.manual_instructions("v0.2.0")
        self.assertIn("pipx install --force", text)
        self.assertIn("git -C", text)
        self.assertIn("v0.2.0", text)


if __name__ == "__main__":
    unittest.main()

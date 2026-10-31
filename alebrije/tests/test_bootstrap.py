import os
import unittest
from unittest import mock

from alebrije import bootstrap


RELEASE = {
    "assets": [
        {"name": "llama-b9999-bin-ubuntu-x64.zip", "browser_download_url": "u1", "size": 10},
        {"name": "llama-b9999-bin-ubuntu-cuda-x64.tar.gz", "browser_download_url": "u2", "size": 10},
        {"name": "llama-b9999-bin-macos-arm64.zip", "browser_download_url": "u3", "size": 10},
        {"name": "llama-b9999-bin-win-cuda-x64.zip", "browser_download_url": "u4", "size": 10},
        {"name": "llama-b9999-bin-ubuntu-vulkan-x64.zip", "browser_download_url": "u5", "size": 10},
    ]
}


class TestAssetPick(unittest.TestCase):
    def _pick(self, system, machine, hint):
        with mock.patch("alebrije.accel.platform.system", return_value=system), \
             mock.patch("alebrije.accel.platform.machine", return_value=machine), \
             mock.patch("alebrije.accel.detect",
                        return_value={"backend": hint, "asset_hint": hint,
                                      "ngl": 999, "note": ""}):
            return bootstrap.pick_release_asset(RELEASE)

    def test_linux_cpu(self):
        asset = self._pick("Linux", "x86_64", "cpu")
        self.assertEqual(asset["name"], "llama-b9999-bin-ubuntu-x64.zip")

    def test_linux_cuda(self):
        asset = self._pick("Linux", "x86_64", "cuda")
        self.assertIn("cuda", asset["name"])

    def test_macos(self):
        asset = self._pick("Darwin", "arm64", "metal")
        self.assertIn("macos-arm64", asset["name"])

    def test_windows_cuda(self):
        asset = self._pick("Windows", "AMD64", "cuda")
        self.assertIn("win-cuda", asset["name"])

    def test_linux_vulkan(self):
        asset = self._pick("Linux", "x86_64", "vulkan")
        self.assertIn("vulkan", asset["name"])

    def test_no_match_returns_none(self):
        self.assertIsNone(self._pick("SunOS", "sparc", "cpu"))


class TestLlamaServerPath(unittest.TestCase):
    def test_not_installed(self):
        with mock.patch("alebrije.bootstrap.bin_dir", return_value="/nonexistent-xyz"):
            self.assertEqual(bootstrap.llama_server_path({}), "")

    def test_found_at_root(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "llama-server")
            open(p, "w").write("#!/bin/sh\n")
            os.chmod(p, 0o755)
            with mock.patch("alebrije.bootstrap.bin_dir", return_value=d):
                self.assertEqual(bootstrap.llama_server_path({}), p)


class TestModels(unittest.TestCase):
    def test_five_models_declared(self):
        self.assertEqual(len(bootstrap.MODELS), 5)
        names = {m["name"] for m in bootstrap.MODELS}
        self.assertEqual(names, {"gemma-3-4b-it", "gemma-3-4b-mmproj",
                                 "nomic-embed-text-v1.5",
                                 "scrfd-10gf (det_10g)", "adaface-ir101"})

    def test_urls_are_official(self):
        for m in bootstrap.MODELS:
            self.assertTrue(m["url"].startswith("https://huggingface.co/")
                            or m["url"].startswith("https://github.com/"),
                            m["url"])


class TestDownload(unittest.TestCase):
    def test_download_writes_file(self):
        import tempfile
        data = b"hello world" * 100
        fake_resp = mock.MagicMock()
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.headers = {"Content-Length": str(len(data))}
        chunks = [data[i:i + 1024 * 1024] for i in range(0, len(data), 1024 * 1024)] + [b""]
        fake_resp.read.side_effect = lambda bs: chunks.pop(0)
        orig = mock.patch("urllib.request.urlopen", return_value=fake_resp)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "sub", "file.bin")
            with orig:
                bootstrap.download("http://example/x", dest)
            self.assertTrue(os.path.isfile(dest))
            with open(dest, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_check_python(self):
        self.assertTrue(bootstrap.check_python())


if __name__ == "__main__":
    unittest.main()

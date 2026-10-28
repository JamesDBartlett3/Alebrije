import unittest
from unittest import mock

from alebrije import accel


class TestDetect(unittest.TestCase):
    def test_apple_silicon_gets_metal(self):
        with mock.patch("alebrije.accel.platform.system", return_value="Darwin"):
            info = accel.detect()
        self.assertEqual(info["backend"], "metal")
        self.assertEqual(info["ngl"], 999)

    def test_nvidia_wins_on_linux(self):
        with mock.patch("alebrije.accel.platform.system", return_value="Linux"), \
             mock.patch("alebrije.accel._probe", side_effect=lambda cmd, needle="": cmd[0] == "nvidia-smi"):
            info = accel.detect()
        self.assertEqual(info["backend"], "cuda")
        self.assertEqual(info["asset_hint"], "cuda")

    def test_rocm_when_no_nvidia(self):
        def probe(cmd, needle=""):
            return cmd[0] == "rocm-smi"
        with mock.patch("alebrije.accel.platform.system", return_value="Linux"), \
             mock.patch("alebrije.accel._probe", side_effect=probe):
            info = accel.detect()
        self.assertEqual(info["backend"], "rocm")

    def test_vulkan_fallback(self):
        def probe(cmd, needle=""):
            return cmd[0] == "vulkaninfo"
        with mock.patch("alebrije.accel.platform.system", return_value="Linux"), \
             mock.patch("alebrije.accel._probe", side_effect=probe):
            info = accel.detect()
        self.assertEqual(info["backend"], "vulkan")

    def test_cpu_last_resort(self):
        with mock.patch("alebrije.accel.platform.system", return_value="Linux"), \
             mock.patch("alebrije.accel._probe", return_value=False):
            info = accel.detect()
        self.assertEqual(info["backend"], "cpu")
        self.assertEqual(info["ngl"], 0)


class TestServerArgs(unittest.TestCase):
    def test_gpu_offloads_layers(self):
        args = accel.server_args("cuda", 999)
        self.assertIn("-ngl", args)
        self.assertIn("999", args)

    def test_cpu_has_no_ngl(self):
        args = accel.server_args("cpu", 0)
        self.assertNotIn("-ngl", args)


class TestAssetCandidates(unittest.TestCase):
    def test_linux_cuda_prefers_cuda_build(self):
        cands = accel.asset_candidates("cuda", "Linux", "x86_64")
        self.assertTrue(cands[0].startswith("ubuntu-cuda"))
        self.assertEqual(cands[-1], "ubuntu-x64")

    def test_macos_arm64(self):
        cands = accel.asset_candidates("macos", "Darwin", "arm64")
        self.assertEqual(cands, ["macos-arm64"])

    def test_windows_vulkan(self):
        cands = accel.asset_candidates("vulkan", "Windows", "AMD64")
        self.assertIn("win-vulkan-x64", cands[0])


class TestProbe(unittest.TestCase):
    def test_missing_binary_is_false(self):
        self.assertFalse(accel._probe(["definitely-not-a-real-binary-xyz"]))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import mock

from alebrije import servers

from . import AlebrijeCase, sample_config


class _FakeTime:
    """Instant clock: start()'s health-wait loop never actually waits."""

    def time(self):
        return 0.0

    def sleep(self, *_):
        pass


class TestStart(AlebrijeCase):
    def _cfg_with_server(self):
        cfg = sample_config()
        fake_server = os.path.join(self.home, "bin", "llama-server")
        os.makedirs(os.path.dirname(fake_server))
        open(fake_server, "w").write("#!/bin/sh\nsleep 999\n")
        os.chmod(fake_server, 0o755)
        return cfg

    def _up_down_then_up(self):
        states = [{"vlm": False, "embed": False}] + [{"vlm": True, "embed": True}] * 50
        return mock.patch("alebrije.servers.is_up", side_effect=states)

    def _patch_env(self, cfg):
        return [
            mock.patch("alebrije.bootstrap.llama_server_path",
                       return_value=os.path.join(self.home, "bin", "llama-server")),
            mock.patch("alebrije.bootstrap.library_path_env", return_value={}),
            mock.patch("alebrije.bootstrap.model_dir", return_value=self.home),
            mock.patch("alebrije.servers.time", _FakeTime()),
        ]

    def test_start_spawns_both_and_records_pids(self):
        cfg = self._cfg_with_server()
        spawned = []

        def fake_spawn(cfg, name, args, log):
            spawned.append((name, args))
            return 4242

        patches = self._patch_env(cfg) + [self._up_down_then_up(),
                                          mock.patch("alebrije.accel.detect",
                                                     return_value={"backend": "cpu", "ngl": 0,
                                                                   "note": "", "asset_hint": "cpu"}),
                                          mock.patch("alebrije.servers._spawn", side_effect=fake_spawn)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        servers.start(cfg, log=lambda *_: None, wait_secs=5)
        names = [n for n, _ in spawned]
        self.assertEqual(names, ["vlm", "embed"])
        vlm_args = spawned[0][1]
        self.assertIn("--mmproj", vlm_args)
        self.assertIn("--alias", vlm_args)
        embed_args = spawned[1][1]
        self.assertIn("--embeddings", embed_args)

    def test_start_gpu_passes_ngl(self):
        cfg = self._cfg_with_server()
        recorded = []

        def fake_spawn(c, n, a, log):
            recorded.append((n, a))
            return 1

        patches = self._patch_env(cfg) + [self._up_down_then_up(),
                                          mock.patch("alebrije.accel.detect",
                                                     return_value={"backend": "cuda", "ngl": 999,
                                                                   "note": "", "asset_hint": "cuda"}),
                                          mock.patch("alebrije.servers._spawn", side_effect=fake_spawn)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        servers.start(cfg, log=lambda *_: None, wait_secs=5)
        self.assertIn("-ngl", recorded[0][1])
        self.assertIn("999", recorded[0][1])


class TestStop(AlebrijeCase):
    def test_stop_removes_pid_files(self):
        cfg = sample_config()
        servers.run_dir(cfg)
        for name in ("vlm", "embed"):
            with open(servers._pid_file(cfg, name), "w") as fh:
                fh.write("99999999")
        with mock.patch("alebrije.servers.os.kill"):
            servers.stop(cfg, log=lambda *_: None)
        self.assertFalse(os.path.exists(servers._pid_file(cfg, "vlm")))
        self.assertFalse(os.path.exists(servers._pid_file(cfg, "embed")))

    def test_stop_without_pid_files_is_safe(self):
        cfg = sample_config()
        servers.stop(cfg, log=lambda *_: None)  # must not raise


class TestIsUp(unittest.TestCase):
    def test_down_when_nothing_listening(self):
        cfg = sample_config(vlm_url="http://127.0.0.1:1", embed_url="http://127.0.0.1:1")
        state = servers.is_up(cfg)
        self.assertFalse(state["vlm"])
        self.assertFalse(state["embed"])


if __name__ == "__main__":
    unittest.main()

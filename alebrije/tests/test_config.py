import os
import unittest

from alebrije import config

from . import AlebrijeCase


class TestConfig(AlebrijeCase):
    def test_defaults(self):
        cfg = config.load()
        for key, val in config.DEFAULTS.items():
            self.assertEqual(cfg[key], val)

    def test_save_and_load_roundtrip(self):
        cfg = config.load()
        cfg["archive_dir"] = "/tmp/fotos"
        config.save(cfg)
        again = config.load()
        self.assertEqual(again["archive_dir"], "/tmp/fotos")

    def test_home_is_isolated(self):
        cfg = config.load()
        self.assertTrue(str(self.home) in str(config.home_dir(cfg)))

    def test_update_ignores_unknown_keys(self):
        cfg = config.update({}, nonsense="x", web_port=1234)
        self.assertNotIn("nonsense", cfg)
        self.assertEqual(cfg["web_port"], 1234)

    def test_update_ignores_none(self):
        cfg = config.update({"web_port": 5}, web_port=None)
        self.assertEqual(cfg["web_port"], 5)

    def test_corrupt_config_falls_back_to_defaults(self):
        from alebrije.config import config_path
        cfg = config.load()
        config_path(cfg).write_text("{not json", encoding="utf-8")
        cfg2 = config.load()
        self.assertEqual(cfg2["web_port"], config.DEFAULTS["web_port"])


if __name__ == "__main__":
    unittest.main()

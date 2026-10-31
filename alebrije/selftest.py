"""Offline self-tests: `alebrije selftest`.

Explicit module list, loaded via loadTestsFromName — no discovery
ambiguity, same approach that keeps `python3 -m unittest` working too.
"""

from __future__ import annotations

import unittest

TEST_MODULES = [
    "alebrije.tests.test_config",
    "alebrije.tests.test_db",
    "alebrije.tests.test_scanner",
    "alebrije.tests.test_llm",
    "alebrije.tests.test_faces",
    "alebrije.tests.test_places",
    "alebrije.tests.test_tagger",
    "alebrije.tests.test_people",
    "alebrije.tests.test_vectors",
    "alebrije.tests.test_search",
    "alebrije.tests.test_accel",
    "alebrije.tests.test_bootstrap",
    "alebrije.tests.test_servers",
    "alebrije.tests.test_web",
    "alebrije.tests.test_cli",
    "alebrije.tests.test_update",
]


def run() -> int:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1

"""Allow every reasonable invocation on every OS:

    python -m alebrije          (from the repo root, or installed)
    python3 -m alebrije
    python alebrije/__main__.py (direct file — bootstraps its own package path)
    python alebrije.py          (repo-root launcher)
"""
import os
import sys

if __package__ in (None, ""):
    # run as a plain file: put the repo root on sys.path first
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from alebrije.cli import main
else:
    from .cli import main

sys.exit(main())

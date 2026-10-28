#!/usr/bin/env python3
"""Alebrije launcher: `python alebrije.py <command>` works from anywhere.

Windows-friendly alternative to `python -m alebrije` — just point it at
this file from the repo root (or an absolute path).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alebrije.cli import main

sys.exit(main())

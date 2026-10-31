"""Configuration: one JSON file, sensible defaults, no dependencies.

Default home is ~/.alebrije so a user's archive index survives upgrades
and never lives inside the media directory itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_FILE = "config.json"

DEFAULTS = {
    "archive_dir": "",                 # the messy folder the user pointed us at
    "home": "",                        # alebrije state dir (defaults to ~/.alebrije)
    "vlm_url": "http://127.0.0.1:8791",   # llama-server hosting the vision model
    "vlm_model": "gemma-3-4b-it",
    "embed_url": "http://127.0.0.1:8792",  # llama-server hosting nomic-embed-text
    "embed_model": "nomic-embed-text-v1.5",
    "web_host": "0.0.0.0",
    "web_port": 8799,
    "tag_min_confidence": 0.0,         # reserved: model returns tags unlabeled
    "llm_timeout": 300,                # vision calls on CPU are slow
    "embed_timeout": 60,
    "thumb_size": 640,                 # max served image dimension hint (client-side)
    # face recognition (SCRFD detection + AdaFace embeddings, all local)
    "face_auto_threshold": 0.5,        # cosine >= this: auto-assign to a known person
    "face_gray_low": 0.36,             # cosine below this: never a match; between -> ask Gemma
    "face_cluster_threshold": 0.42,    # cosine >= this: unassigned faces cluster together
    "face_adjudicate_limit": 100,      # max Gemma gray-band confirmations per `name` run
    "face_tiling": True,               # 1:1 tiled detection pass over large photos
}


def home_dir(cfg: dict) -> Path:
    h = (cfg.get("home") or os.environ.get("ALEBRIJE_HOME")
         or os.path.join(os.path.expanduser("~"), ".alebrije"))
    return Path(h).expanduser()


def config_path(cfg: dict) -> Path:
    return home_dir(cfg) / CONFIG_FILE


def db_path(cfg: dict) -> Path:
    return home_dir(cfg) / "archive.db"


def load(home: str | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if home:
        cfg["home"] = home
    p = config_path(cfg)
    if p.is_file():
        try:
            stored = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        if isinstance(stored, dict):
            cfg.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return cfg


def save(cfg: dict) -> Path:
    p = config_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def update(cfg: dict, **kwargs) -> dict:
    """Merge CLI/programmatic overrides over a loaded config."""
    for k, v in kwargs.items():
        if k in DEFAULTS and v is not None:
            cfg[k] = v
    return cfg

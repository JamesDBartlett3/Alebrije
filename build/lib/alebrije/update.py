"""`alebrije update`: check GitHub releases and upgrade the install.

Flow: query the repo's latest GitHub release tag, compare it (semantically)
against the running version, and — if the install is pipx-managed —
upgrade in place by re-installing from the tagged git URL. If the app was
installed from a clone instead, print the exact command to run.

Releases are the source of truth: `git+...@vX.Y.Z` pins are reproducible,
and "latest" means a tagged release, never a moving branch head.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from . import bootstrap

REPO = "JamesDBartlett3/Alebrije"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
CLONE_URL = f"https://github.com/{REPO}.git"


def version_tuple(tag: str) -> tuple[int, ...]:
    """'v0.12.3' -> (0, 12, 3); non-numeric parts are dropped."""
    parts = re.findall(r"\d+", tag)
    return tuple(int(p) for p in parts[:3]) or (0,)


def is_newer(latest_tag: str, current_version: str) -> bool:
    return version_tuple(latest_tag) > version_tuple(current_version)


def latest_release() -> dict | None:
    """Latest GitHub release as {tag, url, notes}.

    Returns tag=None when the repo exists but has published no releases;
    returns None when it couldn't be reached at all (offline, rate limit).
    """
    import urllib.error
    try:
        data = bootstrap._fetch_json(RELEASES_API)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"tag": None, "url": f"https://github.com/{REPO}/releases",
                    "notes": ""}
        return None
    except Exception:  # noqa: BLE001 — offline / DNS / timeout → caller handles
        return None
    if not isinstance(data, dict) or not data.get("tag_name"):
        return None
    return {
        "tag": str(data["tag_name"]),
        "url": data.get("html_url") or f"https://github.com/{REPO}/releases",
        "notes": (data.get("body") or "")[:400],
    }


def install_source(tag: str) -> str:
    return f"git+{CLONE_URL}@{tag}"


def pipx_managed() -> bool:
    """True if pipx exists and has alebrije installed."""
    if not shutil.which("pipx"):
        return False
    try:
        out = subprocess.run(["pipx", "list", "--short"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split()[0] == "alebrije" for line in out.splitlines() if line.split())


def upgrade_with_pipx(tag: str) -> bool:
    """Reinstall from the tagged git URL via pipx. Returns success."""
    proc = subprocess.run(
        ["pipx", "install", "--force", install_source(tag)],
        capture_output=True, text=True, timeout=600)
    return proc.returncode == 0


def manual_instructions(tag: str) -> str:
    return (
        f"Alebrije {tag} is available.\n"
        f"  • pipx install:      pipx install --force {install_source(tag)}\n"
        f"  • from a clone:      git -C <repo> fetch --tags && "
        f"git -C <repo> checkout {tag}"
    )

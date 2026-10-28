"""Bootstrap: install every dependency Alebrije needs, automatically.

`alebrije setup` makes the project fully self-installing on Linux, macOS,
and Windows:

1. detects GPU/NPU hardware (accel.py) and picks the matching llama.cpp
   prebuilt build from the llama.cpp GitHub releases API — no compiler,
   no package manager;
2. downloads the three open-weight GGUF models (Gemma 3 4B vision model +
   its multimodal projector, and nomic-embed-text) from Hugging Face,
   resumable, into ~/.alebrije/models.

Everything lands in ~/.alebrije — the app itself is stdlib-only Python, so
the only hard prerequisite is Python 3.10+.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import urllib.request

from . import accel

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

MODELS = [
    {
        "name": "gemma-3-4b-it",
        "file": "gemma-3-4b-it-Q4_K_M.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        "role": "vision + language (the archivist)",
    },
    {
        "name": "gemma-3-4b-mmproj",
        "file": "mmproj-gemma-3-4b-f16.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf",
        "role": "vision projector (how Gemma sees)",
    },
    {
        "name": "nomic-embed-text-v1.5",
        "file": "nomic-embed-text-v1.5-q8_0.gguf",
        "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5-Q8_0.gguf",
        "role": "embeddings (the memory index)",
    },
]

BIN_DIR = "bin"
MODEL_DIR = "models"


def alebrije_home(cfg: dict | None = None) -> str:
    if cfg and cfg.get("home"):
        return os.path.expanduser(cfg["home"])
    return os.path.join(os.path.expanduser("~"), ".alebrije")


def bin_dir(cfg: dict | None = None) -> str:
    return os.path.join(alebrije_home(cfg), BIN_DIR)


def model_dir(cfg: dict | None = None) -> str:
    return os.path.join(alebrije_home(cfg), MODEL_DIR)


def llama_server_path(cfg: dict | None = None) -> str:
    d = bin_dir(cfg)
    system = platform.system()
    names = ["llama-server", "llama-server.exe"] if system == "Windows" else ["llama-server"]
    # llama.cpp zips keep binaries either at the root or under build/bin
    for sub in ("", "build", os.path.join("build", "bin"), "llama-bXXXX-bin-ubuntu-x64"):
        for n in names:
            p = os.path.join(d, sub, n) if sub else os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    # last resort: search two levels deep
    for dirpath, dirnames, filenames in os.walk(d):
        if dirnames.count("node_modules"):
            dirnames.remove("node_modules")
        if "llama-server" in filenames or "llama-server.exe" in filenames:
            return os.path.join(dirpath, "llama-server.exe"
                                if "llama-server.exe" in filenames else "llama-server")
    return ""


def library_path_env(cfg: dict | None = None) -> dict:
    """llama.cpp Linux builds need their own libs on LD_LIBRARY_PATH."""
    if platform.system() != "Linux":
        return {}
    d = bin_dir(cfg)
    if not os.path.isdir(d):
        return {}
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = d + (os.pathsep + existing if existing else "")
    return env


def _fetch_json(url: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "alebrije-bootstrapper",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_release_asset(release: dict) -> dict | None:
    """Choose the llama.cpp release asset matching this platform + accel."""
    system = platform.system()
    machine = platform.machine().lower()
    info = accel.detect()
    for want in accel.asset_candidates(info["asset_hint"], system, machine):
        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if not name.endswith((".zip", ".tar.gz")):
                continue
            if want in name and "test" not in name and "armv7" not in name:
                return asset
    return None


def download(url: str, dest: str, _progress=None) -> str:
    """Streaming download with resume support."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {"User-Agent": "alebrije-bootstrapper"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab") as out:
        total = have + int(resp.headers.get("Content-Length") or 0)
        done = have
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            done += len(block)
            if _progress:
                _progress(os.path.basename(dest), done, total)
    os.replace(part, dest)
    return dest


def is_windows_zip_ok(extract_root: str) -> bool:
    return os.path.isdir(extract_root)


def install_llama_cpp(cfg: dict | None = None, log=print) -> str:
    """Download + unpack the right llama.cpp prebuilt; returns server path."""
    dest_root = bin_dir(cfg)
    if llama_server_path(cfg):
        log("✅ llama.cpp already installed")
        return llama_server_path(cfg)
    info = accel.detect()
    log(f"🔎 {info['note']} → {info['backend']} build")
    log("📡 asking llama.cpp releases for the latest build…")
    release = _fetch_json(LLAMA_RELEASES_API)
    asset = pick_release_asset(release)
    if not asset:
        raise RuntimeError(
            "no prebuilt llama.cpp asset matched this platform "
            f"({platform.system()}/{platform.machine()}/{info['backend']}); "
            "see https://github.com/ggml-org/llama.cpp/releases")
    url = asset["browser_download_url"]
    name = asset["name"]
    log(f"⬇️  downloading {name} ({asset.get('size', 0) // (1024 * 1024)} MB)…")
    archive = download(url, os.path.join(dest_root, name))
    log("📦 unpacking…")
    tmp = dest_root + ".unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    if name.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
    else:
        import tarfile
        with tarfile.open(archive) as tf:
            tf.extractall(tmp)  # noqa: S202 — trusted upstream release
    # flatten: whatever dir tree came out, move its contents into bin/
    entries = os.listdir(tmp)
    src = os.path.join(tmp, entries[0]) if len(entries) == 1 else tmp
    shutil.rmtree(dest_root, ignore_errors=True)
    os.makedirs(os.path.dirname(dest_root), exist_ok=True)
    shutil.move(src, dest_root)
    shutil.rmtree(tmp, ignore_errors=True)
    os.remove(archive)
    server = llama_server_path(cfg)
    if not server:
        raise RuntimeError("llama.cpp unpacked but llama-server not found — "
                           f"look in {dest_root}")
    os.chmod(server, 0o755)
    log(f"✅ llama.cpp installed: {server}")
    return server


def install_models(cfg: dict | None = None, log=print, _progress=None) -> list[str]:
    """Download all three open-weight models (resume-aware)."""
    out = []
    mdir = model_dir(cfg)
    os.makedirs(mdir, exist_ok=True)
    for m in MODELS:
        dest = os.path.join(mdir, m["file"])
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            log(f"✅ {m['name']} already present ({m['role']})")
        else:
            log(f"⬇️  {m['name']} — {m['role']}…")
            download(m["url"], dest, _progress)
        out.append(dest)
    return out


def setup_all(cfg: dict | None = None, log=print) -> dict:
    server = install_llama_cpp(cfg, log)
    models = install_models(cfg, log)
    return {"server": server, "models": models}


def check_python() -> str:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Alebrije needs Python 3.10+; found {sys.version.split()[0]}")
    return sys.version.split()[0]

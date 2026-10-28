"""Launch, supervise, and stop the two local llama.cpp servers.

`alebrije up` starts the vision model (:8791) and the embedding model
(:8792) with the right accelerator flags for this machine, waits for both
/health endpoints, and returns. `alebrije up --foreground` stays attached.
`alebrije stop` shuts them down. PIDs live in ~/.alebrije/run/.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import accel, bootstrap

RUN_DIR_NAME = "run"
VLM_PORT = 8791
EMBED_PORT = 8792
CTX = 4096


def run_dir(cfg: dict | None = None) -> str:
    d = os.path.join(bootstrap.alebrije_home(cfg), RUN_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _pid_file(cfg, name: str) -> str:
    return os.path.join(run_dir(cfg), f"{name}.pid")


def _is_up(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def is_up(cfg: dict) -> dict:
    return {
        "vlm": _is_up(cfg["vlm_url"]),
        "embed": _is_up(cfg["embed_url"]),
    }


def _spawn(cfg: dict, name: str, args: list[str], log=print) -> int | None:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, creationflags=creationflags,
        start_new_session=(sys.platform != "win32"))
    with open(_pid_file(cfg, name), "w", encoding="utf-8") as fh:
        fh.write(str(proc.pid))
    log(f"🌼 {name} starting (pid {proc.pid})")
    return proc.pid


def start(cfg: dict, log=print, wait_secs: float = 600.0) -> dict:
    """Start whichever servers aren't already up; block until healthy."""
    server = bootstrap.llama_server_path(cfg)
    if not server:
        raise RuntimeError("llama.cpp is not installed yet — run `alebrije setup` "
                           "(it downloads everything automatically).")
    info = accel.detect()
    env = bootstrap.library_path_env(cfg)
    extra = accel.server_args(info["backend"], info["ngl"])
    mdir = bootstrap.model_dir(cfg)
    state = is_up(cfg)

    if not state["vlm"]:
        args = [server,
                "-m", os.path.join(mdir, "gemma-3-4b-it-Q4_K_M.gguf"),
                "--mmproj", os.path.join(mdir, "mmproj-gemma-3-4b-f16.gguf"),
                "--host", "127.0.0.1", "--port", str(VLM_PORT),
                "--alias", "gemma-3-4b-it", "-c", str(CTX)] + extra
        if info["backend"] == "cpu" and not extra:
            args += ["-t", str(max(1, (os.cpu_count() or 2) - 1))]
        _spawn(cfg, "vlm", args, log)
    else:
        log(f"✅ vision server already up at {cfg['vlm_url']}")

    if not state["embed"]:
        _spawn(cfg, "embed", [
            server,
            "-m", os.path.join(mdir, "nomic-embed-text-v1.5-q8_0.gguf"),
            "--host", "127.0.0.1", "--port", str(EMBED_PORT),
            "--alias", "nomic-embed-text-v1.5", "--embeddings",
            "-c", "2048"] + extra, log)
    else:
        log(f"✅ embedding server already up at {cfg['embed_url']}")

    deadline = time.time() + wait_secs
    while time.time() < deadline:
        state = is_up(cfg)
        if state["vlm"] and state["embed"]:
            log(f"🌺 both servers healthy (accelerator: {info['backend']})")
            return state
        time.sleep(1.5)
    raise TimeoutError(f"servers not healthy after {wait_secs:.0f}s — "
                       f"state: {json.dumps(state)}")


def stop(cfg: dict, log=print) -> None:
    for name in ("vlm", "embed"):
        pf = _pid_file(cfg, name)
        if not os.path.isfile(pf):
            continue
        try:
            pid = int(open(pf, encoding="utf-8").read().strip())
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            log(f"🕯️  stopped {name} (pid {pid})")
        except (ValueError, ProcessLookupError, OSError) as exc:
            log(f"• {name}: {exc}")
        finally:
            os.remove(pf)

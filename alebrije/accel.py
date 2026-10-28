"""Detect GPU/NPU hardware and pick the fastest local-inference backend.

Order of preference on one machine: CUDA (NVIDIA) > ROCm (AMD) > Vulkan >
Metal (Apple, built into every macOS llama.cpp build) > CPU. The result
drives (a) which prebuilt llama.cpp the bootstrap downloads and (b) the
`-ngl` layer-offload flag passed to llama-server.
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def _probe(cmd: list[str], needle: str = "") -> bool:
    exe = shutil.which(cmd[0])
    if not exe:
        return False
    try:
        out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True,
                             timeout=10).stdout + ""
    except (OSError, subprocess.TimeoutExpired):
        return False
    return needle.lower() in out.lower() if needle else True


def detect() -> dict:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        # Metal ships in every macOS llama.cpp build; Apple Silicon gets the
        # unified-memory fast path, Intel Macs still get Metal IGP offload.
        return {"backend": "metal", "ngl": 999,
                "asset_hint": "macos",
                "note": "Apple GPU via Metal"}

    if _probe(["nvidia-smi"]):
        return {"backend": "cuda", "ngl": 999,
                "asset_hint": "cuda",
                "note": "NVIDIA GPU detected via nvidia-smi"}

    if system == "Linux" and _probe(["rocm-smi"]):
        return {"backend": "rocm", "ngl": 999,
                "asset_hint": "rocm",
                "note": "AMD GPU detected via rocm-smi"}

    if _probe(["vulkaninfo", "--summary"]) or _probe(["vulkaninfo"]):
        return {"backend": "vulkan", "ngl": 999,
                "asset_hint": "vulkan",
                "note": "Vulkan-capable GPU detected via vulkaninfo"}

    return {"backend": "cpu", "ngl": 0,
            "asset_hint": "cpu",
            "note": f"no GPU/NPU detected ({machine}); using CPU threads"}


def server_args(backend: str, ngl: int, threads: int = 0) -> list[str]:
    args: list[str] = []
    if ngl:
        args += ["-ngl", str(ngl)]
    if threads:
        args += ["-t", str(threads)]
    if backend in ("cuda", "rocm", "vulkan", "metal"):
        # keep the vision projector on-device too
        args += ["--no-warmup"]
    return args


def asset_candidates(asset_hint: str, system: str, machine: str) -> list[str]:
    """Substrings to look for in llama.cpp release asset names, best first."""
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if system == "Darwin":
        return [f"macos-{arch}"]
    if system == "Windows":
        if asset_hint == "cuda":
            return [f"win-cuda-{arch}", f"win-cuda", f"win-{arch}"]
        if asset_hint == "vulkan":
            return [f"win-vulkan-{arch}", f"win-vulkan", f"win-{arch}"]
        return [f"win-{arch}"]
    # Linux
    if asset_hint == "cuda":
        return [f"ubuntu-cuda-{arch}", f"ubuntu-{arch}"]
    if asset_hint == "rocm":
        return [f"ubuntu-rocm-{arch}", f"ubuntu-{arch}"]
    if asset_hint == "vulkan":
        return [f"ubuntu-vulkan-{arch}", f"ubuntu-{arch}"]
    if system == "Linux":
        return [f"ubuntu-{arch}"]
    return []

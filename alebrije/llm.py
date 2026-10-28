"""Client for the two local llama.cpp servers (OpenAI-compatible endpoints).

Everything here talks to `http://127.0.0.1` by default — your media never
leaves the machine. Functions are module-level so tests can patch them.
"""

from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request

from .errors import LlmError


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(300).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — best-effort error detail
            pass
        raise LlmError("http", f"{url} -> HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            raise LlmError("dns", f"cannot resolve {url}: {reason}") from exc
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError, OSError)):
            raise LlmError("connect", f"cannot connect to {url}: {reason}") from exc
        raise LlmError("connect", f"cannot reach {url}: {reason}") from exc
    except TimeoutError as exc:
        raise LlmError("timeout", f"{url} timed out after {timeout}s") from exc
    except json.JSONDecodeError as exc:
        raise LlmError("parse", f"{url} returned non-JSON: {exc}") from exc


def _image_data_url(path: str) -> str:
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode('ascii')}"


def chat(cfg: dict, messages: list, image_paths: list[str] | None = None) -> str:
    """One chat completion against the local vision model. Returns text."""
    if image_paths:
        msg = messages[-1]
        content = [{"type": "text", "text": msg["content"]}]
        content += [{"type": "image_url", "image_url": {"url": _image_data_url(p)}}
                    for p in image_paths]
        messages = messages[:-1] + [{**msg, "content": content}]
    url = cfg["vlm_url"].rstrip("/") + "/v1/chat/completions"
    data = _post_json(url, {
        "model": cfg["vlm_model"],
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
    }, cfg["llm_timeout"])
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("parse", f"unexpected chat response shape: {data!r}") from exc


def embed(cfg: dict, texts: list[str]) -> list[list[float]]:
    """Batch embeddings against the local nomic server."""
    url = cfg["embed_url"].rstrip("/") + "/v1/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        chunk = texts[i:i + 32]
        data = _post_json(url, {"model": cfg["embed_model"], "input": chunk},
                          cfg["embed_timeout"])
        try:
            got = sorted(data["data"], key=lambda d: d["index"])
            out += [d["embedding"] for d in got]
        except (KeyError, TypeError) as exc:
            raise LlmError("parse", f"unexpected embedding response shape: {data!r}") from exc
    return out


def describe_image(cfg: dict, image_path: str, prompt: str) -> str:
    return chat(cfg, [{"role": "user", "content": prompt}], [image_path])


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating fences/prose."""
    text = text.strip()
    fence = text.find("```")
    if fence != -1:
        end = text.find("```", fence + 3)
        text = text[fence + 3: end if end != -1 else None].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    if start == -1:
        raise LlmError("parse", f"no JSON object in model reply: {text[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise LlmError("parse", f"malformed JSON in model reply: {exc}") from exc
    raise LlmError("parse", "unterminated JSON object in model reply")

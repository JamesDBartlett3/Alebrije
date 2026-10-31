"""Local face detection + recognition: SCRFD finds faces, AdaFace names vectors.

Two open ONNX models, both running on onnxruntime (CPU):

1. **SCRFD-10GF** (``det_10g.onnx``) detects faces and returns a box plus
   five landmarks per face. Its input is a fixed 640x640, so large photos
   get two passes: the whole image letterboxed (finds big faces), then —
   when the image is big enough that letterboxing would shrink faces —
   overlapping 640x640 tiles at native scale (finds small faces). Results
   from both passes are merged in original-image coordinates with NMS.
2. **AdaFace ir101** (``adaface_ir_101.onnx``) turns each face into a
   512-d embedding. Its input is a fixed 112x112 *aligned* crop: the five
   landmarks are mapped back to original-image coordinates and a
   similarity transform warps the crop **from the original image**, so a
   600px face is simply downsampled the same way it was in training.

Everything here is local; the embeddings are meaningless numbers outside
this machine's own database.

Module-level functions (detect, embed_*, faces_from_image, embed_faces,
scan_faces) are the test seam, patched by the offline test suite — no
model files, onnxruntime, or network needed to run the tests.
"""

from __future__ import annotations

import os

import numpy as np

DET_SIZE = 640            # SCRFD network input side
DET_THRESH = 0.5          # detection score floor
DET_MEAN, DET_STD = 127.5, 128.0
NMS_IOU = 0.4
DET_STRIDES = (8, 16, 32)
TILE_OVERLAP = 0.2        # 1:1 tile overlap, so border faces reappear interior
TILE_MIN_SIDE = 1280      # only tile when the whole-image pass loses resolution
MIN_FACE_PX = 48          # smaller faces embed unreliably; skip
MAX_FACES = 15            # per photo
REC_SIZE = 112            # AdaFace aligned-crop side
REC_MEAN, REC_STD = 127.5, 127.5
DET_FILE = "det_10g.onnx"
REC_FILE = "adaface_ir_101.onnx"

# Canonical ArcFace alignment template (the 5 landmarks of a reference
# 112x112 face), shared by ArcFace/AdaFace-family models.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


class FacesUnavailable(RuntimeError):
    """onnxruntime or a face model is missing — run `alebrije setup`."""


# ---------------------------------------------------------------------------
# pure geometry: letterbox, tiles, SCRFD decode, NMS, alignment — no models


def letterbox(w: int, h: int, out: int = DET_SIZE,
              allow_upscale: bool = True) -> dict:
    """Aspect-preserving fit into out x out, padded. Returns the mapping."""
    scale = min(out / max(w, 1), out / max(h, 1))
    if not allow_upscale:
        scale = min(scale, 1.0)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    return {"scale": scale, "dx": (out - rw) // 2, "dy": (out - rh) // 2,
            "rw": rw, "rh": rh}


def to_orig(value: float, shift: int, scale: float, offset: int = 0) -> int:
    """640-space coordinate -> original-image coordinate."""
    return int(round((value - shift) / scale) + offset)


def tile_grid(w: int, h: int, tile: int = DET_SIZE,
              overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Cover a w x h image with <=tile-side windows, ~overlap% overlap.

    The last row/column is clamped to the image edge so nothing is missed.
    """
    step = max(1, int(tile * (1 - overlap)))

    def axis(length: int) -> list[int]:
        starts = list(range(0, max(length - tile, 0) + 1, step))
        if not starts:
            return [0]
        if starts[-1] + tile < length:
            starts.append(max(length - tile, 0))
        return starts

    return [(x, y, min(tile, w - x), min(tile, h - y))
            for y in axis(h) for x in axis(w)]


def _anchor_centers(stride: int, size: int = DET_SIZE) -> np.ndarray:
    """Anchor centers on the (size/stride)^2 grid, row-major, as (x, y)."""
    n = size // stride
    ys, xs = np.mgrid[:n, :n]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
    return pts * stride


def decode_scrfd(outputs: list, thresh: float = DET_THRESH) -> list[dict]:
    """Decode raw SCRFD outputs (9 tensors: per stride score/bbox/kps).

    Tensors may carry a leading batch dim or not, and det_10g emits 2
    anchors per grid cell, so anchor centers are repeated to match the
    score count. Returns detections in 640-space: {"bbox": [x1, y1, x2,
    y2], "kps": (5, 2), "det_score": float}.
    """
    outs = [o[0] if o.ndim == 3 else o for o in outputs]
    dets: list[dict] = []
    for i, stride in enumerate(DET_STRIDES):
        scores = outs[i][:, 0].astype(np.float32)
        bbox_d = outs[len(DET_STRIDES) + i].astype(np.float32) * stride
        kps_d = outs[2 * len(DET_STRIDES) + i].astype(np.float32) * stride
        anchors = _anchor_centers(stride)
        # det_10g emits 2 anchors per grid cell, cell-major (rows are
        # c0a0, c0a1, c1a0, c1a1, ...), so centers repeat per cell
        nanchor = max(1, len(scores) // len(anchors))
        if nanchor > 1:
            anchors = np.repeat(anchors, nanchor, axis=0)
        keep = np.where(scores >= thresh)[0]
        for j in keep:
            ax, ay = anchors[j]
            x1, y1 = ax - bbox_d[j][0], ay - bbox_d[j][1]
            x2, y2 = ax + bbox_d[j][2], ay + bbox_d[j][3]
            kps = [(ax + kps_d[j][2 * k], ay + kps_d[j][2 * k + 1])
                   for k in range(5)]
            dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                         "kps": kps, "det_score": float(scores[j])})
    return dets


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[dict], iou: float = NMS_IOU) -> list[dict]:
    """Greedy NMS over dicts with "bbox" and "det_score"."""
    out: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["det_score"]):
        if all(_iou(d["bbox"], k["bbox"]) < iou for k in out):
            out.append(d)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping src -> dst, as a 2x3 matrix."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    ss, sd = src - mu_s, dst - mu_d
    cov = sd.T @ ss / len(src)                      # cross-covariance dst,src
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1                                 # allow reflection-free only
    r = u @ s @ vt
    var_ss = (ss ** 2).sum() / len(src)
    scale = (np.diag(d) @ s).trace() / var_ss if var_ss else 1.0
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = mu_d - scale * (r @ mu_s)
    return m


def affine_inverse(m: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine matrix (what PIL's transform wants)."""
    a, b, c, d, e, f = m.flatten().tolist()
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("degenerate affine transform")
    return np.array([[e / det, -b / det, (b * f - c * e) / det],
                     [-d / det, a / det, (c * d - a * f) / det]])


def norm_crop_matrix(kps) -> np.ndarray:
    """Similarity transform from original-image landmarks to the 112 template."""
    return umeyama(np.asarray(kps, dtype=np.float64), ARCFACE_DST)


def aligned_crop(image, kps):
    """112x112 aligned face, warped from the original PIL image (RGB)."""
    from PIL import Image
    m = norm_crop_matrix(kps)
    inv = affine_inverse(m).flatten().tolist()
    return image.transform((REC_SIZE, REC_SIZE), Image.AFFINE, data=inv,
                           resample=Image.BILINEAR)


def review_crop(image, bbox, pad: float = 0.4):
    """Bigger, unaligned face crop (box + pad) — what Gemma reviews best:
    it sees hair, clothing and build, not just the tightly aligned face."""
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2 = min(image.size[0], int(x2 + pw))
    y2 = min(image.size[1], int(y2 + ph))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return image.crop((x1, y1, x2, y2))


def mean_centroid(vecs) -> np.ndarray:
    """Unit-length mean of a list of unit-length vectors."""
    m = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n else m


# ---------------------------------------------------------------------------
# model sessions + the detect/embed pipeline


_SESSIONS: dict = {}


def available(cfg: dict) -> dict:
    """What of the face stack is present? Never raises."""
    from . import bootstrap
    mdir = bootstrap.model_dir(cfg)
    try:
        import onnxruntime  # noqa: F401
        ort = True
    except ImportError:
        ort = False
    det = os.path.isfile(os.path.join(mdir, DET_FILE))
    rec = os.path.isfile(os.path.join(mdir, REC_FILE))
    return {"onnxruntime": ort, "detector": det, "recognizer": rec,
            "ok": ort and det and rec}


def _session(cfg: dict, kind: str):
    import onnxruntime as ort
    from . import bootstrap
    if kind not in _SESSIONS:
        mdir = bootstrap.model_dir(cfg)
        path = os.path.join(mdir, DET_FILE if kind == "det" else REC_FILE)
        if not os.path.isfile(path):
            raise FacesUnavailable(f"model {os.path.basename(path)} is not "
                                   f"installed — run `alebrije setup`")
        _SESSIONS[kind] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
    return _SESSIONS[kind]


def _run_det(cfg: dict, region, lb: dict, offset: tuple[int, int],
             at_edge: tuple[bool, bool, bool, bool]) -> list[dict]:
    """Detect in one 640x640 letterboxed region; map back to orig coords.

    `at_edge` marks tile sides that coincide with the image edge (left,
    top, right, bottom) — detections touching a non-edge side are partial
    faces owned by the neighboring tile or the whole-image pass.
    """
    from PIL import Image
    canvas = np.full((DET_SIZE, DET_SIZE, 3), 114, np.uint8)
    resized = region.resize((lb["rw"], lb["rh"]), Image.BILINEAR)
    canvas[lb["dy"]:lb["dy"] + lb["rh"], lb["dx"]:lb["dx"] + lb["rw"]] = \
        np.asarray(resized, dtype=np.uint8)
    x = (canvas.astype(np.float32) - DET_MEAN) / DET_STD
    sess = _session(cfg, "det")
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: x.transpose(2, 0, 1)[None]})

    ox, oy = offset
    keep = []
    for d in decode_scrfd(outs):
        x1, y1, x2, y2 = d["bbox"]
        # drop partial faces at non-edge tile borders
        if (not at_edge[0] and x1 <= lb["dx"] + 1) or \
           (not at_edge[1] and y1 <= lb["dy"] + 1) or \
           (not at_edge[2] and x2 >= lb["dx"] + lb["rw"] - 1) or \
           (not at_edge[3] and y2 >= lb["dy"] + lb["rh"] - 1):
            continue
        bbox = [to_orig(x1, lb["dx"], lb["scale"], ox),
                to_orig(y1, lb["dy"], lb["scale"], oy),
                to_orig(x2, lb["dx"], lb["scale"], ox),
                to_orig(y2, lb["dy"], lb["scale"], oy)]
        kps = [(to_orig(px, lb["dx"], lb["scale"], ox),
                to_orig(py, lb["dy"], lb["scale"], oy)) for px, py in d["kps"]]
        keep.append({"bbox": bbox, "kps": kps, "det_score": d["det_score"]})
    return keep


def detect(cfg: dict, image):
    """All faces in a PIL RGB image, in original-image coordinates."""
    w, h = image.size
    dets = _run_det(cfg, image, letterbox(w, h), (0, 0),
                    (True, True, True, True))
    if cfg.get("face_tiling", True) and max(w, h) > TILE_MIN_SIDE:
        for tx, ty, tw, th in tile_grid(w, h):
            region = image.crop((tx, ty, tx + tw, ty + th))
            lb = letterbox(tw, th, allow_upscale=False)
            dets += _run_det(
                cfg, region, lb, (tx, ty),
                (tx == 0, ty == 0, tx + tw >= w, ty + th >= h))
    return nms(dets)


def embed_crop(cfg: dict, aligned) -> list[float]:
    """512-d L2-normalized embedding of one aligned 112x112 face (PIL RGB)."""
    arr = np.asarray(aligned, dtype=np.float32)
    arr = (arr - REC_MEAN) / REC_STD
    sess = _session(cfg, "rec")
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: arr.transpose(2, 0, 1)[None]})[0][0]
    norm = float(np.linalg.norm(out))
    return (out / norm if norm else out).astype(np.float32).tolist()


def faces_from_image(cfg: dict, image) -> list[dict]:
    """Detect + embed every usable face in a PIL RGB image.

    Returns [{"bbox": (x, y, w, h), "kps", "det_score", "embedding"}],
    biggest faces first.
    """
    if not available(cfg)["ok"]:
        raise FacesUnavailable("face models are not installed — "
                               "run `alebrije setup`")
    out = []
    for d in sorted(detect(cfg, image),
                    key=lambda d: (d["bbox"][2] - d["bbox"][0])
                    * (d["bbox"][3] - d["bbox"][1]), reverse=True):
        x1, y1, x2, y2 = d["bbox"]
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
            continue
        out.append({"bbox": (x1, y1, bw, bh), "kps": d["kps"],
                    "det_score": d["det_score"],
                    "embedding": embed_crop(cfg, aligned_crop(image, d["kps"]))})
        if len(out) >= MAX_FACES:
            break
    return out


def embed_faces(cfg: dict, path: str) -> list[dict]:
    """faces_from_image, from a file path."""
    from PIL import Image
    with Image.open(path) as im:
        return faces_from_image(cfg, im.convert("RGB"))


# ---------------------------------------------------------------------------
# database-facing batch


def _video_frame(path: str) -> str | None:
    """Native-resolution keyframe — the faces pass needs every pixel."""
    from .tagger import extract_frame
    return extract_frame(path, scale=None)


def scan_faces(conn, cfg: dict, limit: int = 0, log=print) -> dict:
    """Compute faces for media that has none yet. Returns counters."""
    import shutil
    from . import vectors
    from .people import now_iso
    has_ffmpeg = bool(shutil.which("ffmpeg"))
    rows = conn.execute(
        "SELECT m.id, m.path, m.media_type FROM media m "
        "LEFT JOIN faces f ON f.media_id = m.id "
        "WHERE f.media_id IS NULL AND m.media_type IN ('image', 'video') "
        "ORDER BY m.id" + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    found = 0
    done = 0
    for row in rows:
        path = row["path"]
        frame = None
        if row["media_type"] == "video":
            if not has_ffmpeg:
                continue
            frame = _video_frame(path)
            if not frame:
                continue
            path = frame
        try:
            found_list = embed_faces(cfg, path)
        except FacesUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the scan
            log(f"⚠️  {os.path.basename(row['path'])} — faces: {str(exc)[:120]}")
            found_list = []
        finally:
            if frame and os.path.isfile(frame):
                os.remove(frame)
        for f in found_list:
            conn.execute(
                "INSERT INTO faces (media_id, bbox_x, bbox_y, bbox_w, bbox_h,"
                " det_score, embedding, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (row["id"], f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3],
                 f["det_score"], vectors.encode(f["embedding"]), now_iso()))
        conn.commit()
        found += len(found_list)
        done += 1
        who = f", {len(found_list)} face(s)" if found_list else ""
        log(f"👤 {os.path.basename(row['path'])}{who}")
    remaining = conn.execute(
        "SELECT COUNT(*) c FROM media m LEFT JOIN faces f ON f.media_id = m.id "
        "WHERE f.media_id IS NULL AND m.media_type IN ('image', 'video')"
    ).fetchone()["c"]
    return {"media": done, "faces": found, "remaining": remaining}


def purge(conn) -> int:
    """Delete every face vector and centroid. There is no backup on purpose."""
    n = conn.execute("SELECT COUNT(*) c FROM faces").fetchone()["c"]
    conn.execute("DELETE FROM faces")
    conn.execute("UPDATE people SET face_centroid=NULL")
    conn.commit()
    return n

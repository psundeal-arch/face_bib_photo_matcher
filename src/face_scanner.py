"""Face scanning utilities for InsightFace + optional bib OCR.

Extracted from shared_album_downloader.py to keep scanning logic modular and testable.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency guard
    cv2 = None

import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - runtime dependency guard
    RapidOCR = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - runtime dependency guard
    Image = None

try:
    import pillow_heif  # type: ignore
except Exception:  # pragma: no cover - runtime dependency guard
    pillow_heif = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - runtime dependency guard
    FaceAnalysis = None

BIB_RE = re.compile(r"\b\d{2,5}\b")
DEFAULT_INSIGHTFACE_MODEL_NAME = "buffalo_l"
DEFAULT_INSIGHTFACE_PROVIDER = "auto"
DEFAULT_OCR_MIN_SCORE = 0.45
DEFAULT_OCR_VARIANT_NAMES = ["otsu"]
_OCR_TLS = threading.local()
HEIC_SUFFIXES = {".heic", ".heif"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _load_ocr_variant_names() -> List[str]:
    raw = os.environ.get("OCR_VARIANTS")
    if not raw:
        return list(DEFAULT_OCR_VARIANT_NAMES)
    names: List[str] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if part:
            names.append(part)
    return names or list(DEFAULT_OCR_VARIANT_NAMES)


OCR_MIN_SCORE = _env_float("OCR_MIN_SCORE", DEFAULT_OCR_MIN_SCORE)
OCR_VARIANT_NAMES = _load_ocr_variant_names()
OCR_ENABLE_GLOBAL_PASS = _env_bool("OCR_ENABLE_GLOBAL_PASS", True)


def configure_ocr_runtime(
    *,
    min_score: Optional[float] = None,
    variant_names: Optional[Sequence[str]] = None,
    enable_global_pass: Optional[bool] = None,
) -> None:
    """Update OCR runtime settings for current process.

    This allows config-driven runtime behavior without relying on import-time env vars.
    """
    global OCR_MIN_SCORE, OCR_VARIANT_NAMES, OCR_ENABLE_GLOBAL_PASS
    if min_score is not None:
        OCR_MIN_SCORE = float(min_score)
    if variant_names is not None:
        cleaned = [str(v).strip().lower() for v in variant_names if str(v).strip()]
        OCR_VARIANT_NAMES = cleaned or list(DEFAULT_OCR_VARIANT_NAMES)
    if enable_global_pass is not None:
        OCR_ENABLE_GLOBAL_PASS = bool(enable_global_pass)


def get_ocr_runtime_config() -> Dict[str, object]:
    return {
        "ocr_backend": "rapidocr_onnxruntime",
        "ocr_min_score": OCR_MIN_SCORE,
        "ocr_variants": list(OCR_VARIANT_NAMES),
        "ocr_enable_global_pass": OCR_ENABLE_GLOBAL_PASS,
    }


def is_cv2_available() -> bool:
    return cv2 is not None


def is_rapidocr_available() -> bool:
    return RapidOCR is not None


def is_insightface_available() -> bool:
    return FaceAnalysis is not None


def _get_ocr_engine() -> Optional[Any]:
    if RapidOCR is None:
        return None
    engine = getattr(_OCR_TLS, "engine", None)
    if engine is not None:
        return engine
    _OCR_TLS.engine = RapidOCR()
    return _OCR_TLS.engine


def _encoding_hash(encoding: np.ndarray) -> str:
    rounded = np.round(encoding.astype(np.float32), 4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _crop_torso_region(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    top, right, bottom, left = bbox
    h, w = img.shape[:2]
    face_w = max(1, right - left)
    face_h = max(1, bottom - top)

    x1 = max(0, left - face_w // 2)
    x2 = min(w, right + face_w // 2)
    y1 = min(h, bottom)
    y2 = min(h, bottom + int(face_h * 3.2))

    if y2 <= y1 or x2 <= x1:
        return None
    return img[y1:y2, x1:x2]


def _load_image_bgr(image_path: Path) -> Optional[np.ndarray]:
    # Fast path for most formats supported by OpenCV.
    bgr = cv2.imread(str(image_path))
    if bgr is not None:
        return bgr

    # Fallback path for HEIC/HEIF files.
    if image_path.suffix.lower() not in HEIC_SUFFIXES:
        return None
    if Image is None or pillow_heif is None:
        return None

    try:
        pillow_heif.register_heif_opener()
        with Image.open(str(image_path)) as pil_img:
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image_bgr(image_path: Path) -> Optional[np.ndarray]:
    """Public wrapper for image decoding (including HEIC fallback)."""
    return _load_image_bgr(image_path)


def _extract_rapidocr_lines(raw_result: Any) -> List[Any]:
    if raw_result is None:
        return []
    if isinstance(raw_result, tuple):
        if not raw_result:
            return []
        candidate = raw_result[0]
    else:
        candidate = raw_result
    return candidate if isinstance(candidate, list) else []


def _extract_text_score(item: Any) -> Tuple[Optional[str], float]:
    if isinstance(item, dict):
        text = item.get("text")
        score = item.get("score", item.get("confidence", 1.0))
        if isinstance(text, str):
            try:
                return text, float(score)
            except Exception:
                return text, 1.0
        return None, 0.0

    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None, 0.0

    second = item[1]
    if isinstance(second, (list, tuple)):
        if not second:
            return None, 0.0
        text = str(second[0])
        score_raw = second[1] if len(second) > 1 else 1.0
        try:
            return text, float(score_raw)
        except Exception:
            return text, 1.0

    if isinstance(second, str):
        text = second
        score_raw = item[2] if len(item) > 2 else 1.0
        try:
            return text, float(score_raw)
        except Exception:
            return text, 1.0
    return None, 0.0


def _ocr_bib_candidates(image_bgr: np.ndarray, min_score: float = OCR_MIN_SCORE) -> Dict[str, float]:
    engine = _get_ocr_engine()
    if engine is None:
        return {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    th1 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    variant_map = {
        "gray": gray,
        "adaptive": th1,
        "otsu": th2,
        "inv_adaptive": 255 - th1,
        "inv_otsu": 255 - th2,
    }
    variants = [variant_map[name] for name in OCR_VARIANT_NAMES if name in variant_map]
    if not variants:
        variants = [gray, th2]

    out: Dict[str, float] = {}
    for var in variants:
        raw_result = engine(var)
        for line in _extract_rapidocr_lines(raw_result):
            text, score = _extract_text_score(line)
            if not text or score < min_score:
                continue
            for match in BIB_RE.findall(text):
                best = out.get(match, -1.0)
                if score > best:
                    out[match] = score
    return out


def _assign_face_id(
    embedding: np.ndarray,
    known_embeddings: List[np.ndarray],
    threshold: float,
) -> Tuple[str, int]:
    if not known_embeddings:
        known_embeddings.append(embedding)
        return "face_0001", 0

    distances = np.linalg.norm(np.vstack(known_embeddings) - embedding, axis=1)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist <= threshold:
        return f"face_{best_idx + 1:04d}", best_idx

    known_embeddings.append(embedding)
    new_idx = len(known_embeddings) - 1
    return f"face_{new_idx + 1:04d}", new_idx


def create_insightface_app(
    det_size: int,
    model_name: str = DEFAULT_INSIGHTFACE_MODEL_NAME,
    provider_mode: str = DEFAULT_INSIGHTFACE_PROVIDER,
) -> object:
    if FaceAnalysis is None:
        raise RuntimeError(
            "InsightFace is not installed. Install with: pip install insightface onnxruntime"
        )

    provider_mode = (provider_mode or DEFAULT_INSIGHTFACE_PROVIDER).strip().lower()
    providers = None
    if provider_mode == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider_mode == "coreml":
        if sys.platform == "darwin":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            print("coreml provider requested on non-macOS; falling back to CPUExecutionProvider")
            providers = ["CPUExecutionProvider"]
    elif provider_mode == "auto":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if sys.platform == "darwin" else None
    else:
        raise ValueError(f"Unsupported insightface provider mode: {provider_mode}")

    app = FaceAnalysis(name=model_name, providers=providers)
    if providers is None:
        try:
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    else:
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def process_image(
    image_path: Path,
    face_app: object,
    enable_bib_ocr: bool,
) -> Dict[str, object]:
    if cv2 is None:
        return {
            "file": str(image_path),
            "status": "error",
            "error": "OpenCV unavailable",
            "face_count": 0,
            "faces": [],
            "bib_numbers": [],
        }

    bgr = _load_image_bgr(image_path)
    if bgr is None:
        err = "Failed to read image"
        if image_path.suffix.lower() in HEIC_SUFFIXES:
            err = (
                "Failed to read HEIC/HEIF image "
                "(install pillow and pillow-heif, or convert to JPG/PNG first)"
            )
        return {
            "file": str(image_path),
            "status": "error",
            "error": err,
            "face_count": 0,
            "faces": [],
            "bib_numbers": [],
        }

    detected_faces = face_app.get(bgr)
    faces: List[Dict[str, object]] = []
    bib_scores: Dict[str, float] = {}

    for detected in detected_faces:
        bbox_xyxy = np.asarray(detected.bbox, dtype=np.int32)
        if bbox_xyxy.shape[0] < 4:
            continue

        left, top, right, bottom = [int(v) for v in bbox_xyxy[:4]]
        embedding_raw = getattr(detected, "normed_embedding", None)
        if embedding_raw is None:
            embedding_raw = getattr(detected, "embedding", None)
        if embedding_raw is None:
            continue
        embedding = np.asarray(embedding_raw, dtype=np.float32)

        entry: Dict[str, object] = {
            "face_hash": _encoding_hash(embedding),
            "_embedding": embedding.tolist(),
            "det_score": float(getattr(detected, "det_score", 0.0)),
        }

        if enable_bib_ocr:
            torso = _crop_torso_region(bgr, (top, right, bottom, left))
            face_bibs: List[Dict[str, object]] = []
            if torso is not None and torso.size > 0:
                local_scores = _ocr_bib_candidates(torso)
                for num, conf in local_scores.items():
                    if conf > bib_scores.get(num, -1.0):
                        bib_scores[num] = conf
                    face_bibs.append({"number": num, "confidence": round(conf, 2)})
            entry["bib_numbers_near_face"] = sorted(
                face_bibs,
                key=lambda x: (-float(x["confidence"]), str(x["number"])),
            )

        faces.append(entry)

    if enable_bib_ocr and OCR_ENABLE_GLOBAL_PASS:
        global_scores = _ocr_bib_candidates(bgr)
        for num, conf in global_scores.items():
            if conf > bib_scores.get(num, -1.0):
                bib_scores[num] = conf

    bib_numbers = [
        {"number": n, "confidence": round(c, 2)}
        for n, c in sorted(bib_scores.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "file": str(image_path),
        "status": "ok",
        "face_count": len(faces),
        "faces": faces,
        "bib_numbers": bib_numbers,
    }


def assign_face_ids_in_place(image_results: Sequence[Dict[str, object]], face_threshold: float) -> None:
    known_embeddings: List[np.ndarray] = []

    for image in image_results:
        if image.get("status") != "ok":
            continue
        faces = image.get("faces", [])
        if not isinstance(faces, list):
            continue

        for face in faces:
            if not isinstance(face, dict):
                continue
            embedding_raw = face.pop("_embedding", None)
            if embedding_raw is None:
                continue
            embedding = np.asarray(embedding_raw, dtype=np.float32)
            face_id, _ = _assign_face_id(embedding, known_embeddings, threshold=face_threshold)
            face["face_id"] = face_id
            face["embedding"] = [round(float(v), 6) for v in embedding]


def build_face_index(images: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for image in images:
        if image.get("status") != "ok":
            continue
        for face in image.get("faces", []):
            if not isinstance(face, dict):
                continue
            face_id = face.get("face_id")
            face_hash = face.get("face_hash")
            if not isinstance(face_id, str) or not isinstance(face_hash, str):
                continue

            info = index.setdefault(face_id, {"samples": 0, "face_hashes": {}})
            info["samples"] += 1
            hashes = info["face_hashes"]
            hashes[face_hash] = int(hashes.get(face_hash, 0)) + 1

    return index

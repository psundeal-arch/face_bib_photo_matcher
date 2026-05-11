#!/usr/bin/env python3
"""Simple web UI to match uploaded/captured photo against report JSON entries by hash + embedding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_config import get_section, load_yaml_config

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

app = Flask(__name__, template_folder=str(APP_DIR / "templates"), static_folder=str(APP_DIR / "static"))

_face_app_lock = threading.Lock()
_face_app: Optional[object] = None
_report_cache_lock = threading.Lock()
_report_cache: Dict[str, Dict[str, Any]] = {}


def _load_website_config() -> Dict[str, Any]:
    config_path_text = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)).strip()
    config_path = Path(config_path_text)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()
    config = load_yaml_config(config_path)
    return get_section(config, "website")


WEBSITE_CONFIG = _load_website_config()


def _cfg_str(env_name: str, cfg_key: str, default: str) -> str:
    env_val = os.environ.get(env_name)
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()
    cfg_val = WEBSITE_CONFIG.get(cfg_key)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()
    return default


def _cfg_float(env_name: str, cfg_key: str, default: float) -> float:
    env_val = os.environ.get(env_name)
    if isinstance(env_val, str) and env_val.strip():
        try:
            return float(env_val.strip())
        except Exception:
            return default
    cfg_val = WEBSITE_CONFIG.get(cfg_key)
    try:
        return float(cfg_val)  # type: ignore[arg-type]
    except Exception:
        return default


def _cfg_int(env_name: str, cfg_key: str, default: int) -> int:
    env_val = os.environ.get(env_name)
    if isinstance(env_val, str) and env_val.strip():
        try:
            return int(env_val.strip())
        except Exception:
            return default
    cfg_val = WEBSITE_CONFIG.get(cfg_key)
    try:
        return int(cfg_val)  # type: ignore[arg-type]
    except Exception:
        return default


def load_race_reports_config() -> tuple[Dict[str, Path], str]:
    """Load race->reports folder mapping.

    Env options:
    - WEBSITE_RACE_REPORTS_JSON: JSON object {"Label": "/abs/or/rel/path", ...}
    - WEBSITE_REPORTS_DIR: fallback single reports dir
    """
    raw_json = os.environ.get("WEBSITE_RACE_REPORTS_JSON", "").strip()
    mapping: Dict[str, Path] = {}

    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for label, path_text in parsed.items():
                    if not isinstance(label, str) or not label.strip():
                        continue
                    if not isinstance(path_text, str) or not path_text.strip():
                        continue
                    p = Path(path_text.strip())
                    if not p.is_absolute():
                        p = (REPO_ROOT / p).resolve()
                    else:
                        p = p.resolve()
                    mapping[label.strip()] = p
        except Exception:
            mapping = {}

    if not mapping:
        cfg_mapping = WEBSITE_CONFIG.get("race_reports")
        if isinstance(cfg_mapping, dict):
            for label, path_text in cfg_mapping.items():
                if not isinstance(label, str) or not label.strip():
                    continue
                if not isinstance(path_text, str) or not path_text.strip():
                    continue
                p = Path(path_text.strip())
                if not p.is_absolute():
                    p = (REPO_ROOT / p).resolve()
                else:
                    p = p.resolve()
                mapping[label.strip()] = p

    if not mapping:
        reports_dir = Path(_cfg_str("WEBSITE_REPORTS_DIR", "reports_dir", str(DEFAULT_REPORTS_DIR)))
        if not reports_dir.is_absolute():
            reports_dir = (REPO_ROOT / reports_dir).resolve()
        else:
            reports_dir = reports_dir.resolve()
        mapping = {"2026 Boston Marathon": reports_dir}

    default_label = _cfg_str("WEBSITE_DEFAULT_RACE", "default_race", "")
    if default_label not in mapping:
        default_label = next(iter(mapping.keys()))
    return mapping, default_label


def resolve_reports_dir_by_race(race_label: Optional[str]) -> tuple[Path, str]:
    mapping, default_label = load_race_reports_config()
    selected = (race_label or "").strip()
    if selected and selected in mapping:
        return mapping[selected], selected
    return mapping[default_label], default_label


def create_insightface_app(det_size: int) -> object:
    if FaceAnalysis is None:
        raise RuntimeError("InsightFace is not installed. Install with: pip install insightface onnxruntime")

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if sys.platform == "darwin" else None
    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    if providers is None:
        try:
            face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    else:
        face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return face_app


def get_face_app(det_size: int) -> object:
    global _face_app
    with _face_app_lock:
        if _face_app is None:
            _face_app = create_insightface_app(det_size)
        return _face_app


def encoding_hash(encoding: np.ndarray) -> str:
    rounded = np.round(encoding.astype(np.float32), 4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def to_bbox_dict(raw_bbox: Any) -> Optional[Dict[str, float]]:
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, dict):
        keys = {"left", "top", "right", "bottom"}
        if keys.issubset(raw_bbox.keys()):
            try:
                return {
                    "left": float(raw_bbox["left"]),
                    "top": float(raw_bbox["top"]),
                    "right": float(raw_bbox["right"]),
                    "bottom": float(raw_bbox["bottom"]),
                }
            except Exception:
                return None
        return None

    arr = np.asarray(raw_bbox, dtype=np.float32)
    if arr.shape[0] < 4:
        return None
    return {
        "left": float(arr[0]),
        "top": float(arr[1]),
        "right": float(arr[2]),
        "bottom": float(arr[3]),
    }


def bbox_area(bbox: Dict[str, float]) -> float:
    return max(0.0, bbox["right"] - bbox["left"]) * max(0.0, bbox["bottom"] - bbox["top"])


def choose_primary_face(detected_faces: List[Any]) -> Optional[Any]:
    if not detected_faces:
        return None

    def area(face: Any) -> float:
        bbox = to_bbox_dict(getattr(face, "bbox", None))
        return bbox_area(bbox) if bbox else 0.0

    return max(detected_faces, key=area)


def get_face_hash(face: Any) -> Optional[str]:
    embedding_raw = getattr(face, "normed_embedding", None)
    if embedding_raw is None:
        embedding_raw = getattr(face, "embedding", None)
    if embedding_raw is None:
        return None
    embedding = np.asarray(embedding_raw, dtype=np.float32)
    return encoding_hash(embedding)


def get_face_embedding(face: Any) -> Optional[np.ndarray]:
    embedding_raw = getattr(face, "normed_embedding", None)
    if embedding_raw is None:
        embedding_raw = getattr(face, "embedding", None)
    if embedding_raw is None:
        return None
    return np.asarray(embedding_raw, dtype=np.float32)


def has_bib(image_entry: Dict[str, Any], bib_number: str) -> bool:
    bibs = image_entry.get("bib_numbers", [])
    if not isinstance(bibs, list):
        return False
    for item in bibs:
        if isinstance(item, dict) and str(item.get("number")) == bib_number:
            return True
    return False


def resolve_image_name(image: Dict[str, Any]) -> str:
    file_name = image.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return file_name.strip()

    file_path = image.get("file")
    if isinstance(file_path, str) and file_path.strip():
        name = Path(file_path.strip()).name
        if name:
            return name

    source_url = image.get("source_url")
    if isinstance(source_url, str) and source_url.strip():
        parsed = urlparse(source_url.strip())
        tail = Path(parsed.path).name
        if tail:
            return tail

    media_index = image.get("media_index")
    if isinstance(media_index, int):
        return f"media_{media_index:05d}"
    return "unknown"


def load_report_files(reports_dir: Path) -> List[Path]:
    if not reports_dir.exists():
        return []
    return sorted([p for p in reports_dir.glob("*.json") if p.is_file()])


def load_report_payloads_cached(report_files: List[Path]) -> List[tuple[Path, Dict[str, Any]]]:
    payloads: List[tuple[Path, Dict[str, Any]]] = []
    with _report_cache_lock:
        for report_path in report_files:
            try:
                stat = report_path.stat()
            except Exception:
                continue
            cache_key = str(report_path.resolve())
            mtime = float(stat.st_mtime)
            cached = _report_cache.get(cache_key)
            if cached and float(cached.get("mtime", -1.0)) == mtime and isinstance(
                cached.get("payload"), dict
            ):
                payloads.append((report_path, cached["payload"]))
                continue
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            _report_cache[cache_key] = {"mtime": mtime, "payload": payload}
            payloads.append((report_path, payload))
    return payloads


def match_against_reports(
    reports_dir: Path,
    bib_number: str,
    query_face_hashes: List[str],
    query_embeddings: List[np.ndarray],
    embedding_threshold: float,
    max_results: int,
) -> Dict[str, Any]:
    report_files = load_report_files(reports_dir)
    report_payloads = load_report_payloads_cached(report_files)
    matches: List[Dict[str, Any]] = []
    scanned_entries = 0

    for report_path, payload in report_payloads:
        album = payload.get("album", {})
        images = payload.get("images", [])
        if not isinstance(images, list):
            continue

        for image in images:
            if not isinstance(image, dict):
                continue
            bib_match = has_bib(image, bib_number)

            faces = image.get("faces", [])
            if not isinstance(faces, list):
                faces = []
            if bib_match and len(faces) == 0:
                matches.append(
                    {
                        "report_file": report_path.name,
                        "album_url": album.get("album_url"),
                        "album_title": album.get("album_title"),
                        "image_name": resolve_image_name(image),
                        "source_url": image.get("source_url"),
                        "media_index": image.get("media_index"),
                        "bib_match": True,
                        "face_id": None,
                        "candidate_face_hash": None,
                        "hash_match": False,
                        "embedding_match": False,
                        "embedding_distance": None,
                    }
                )
                continue

            for face in faces:
                if not isinstance(face, dict):
                    continue

                scanned_entries += 1
                cand_hash = face.get("face_hash")
                hash_match = bool(
                    isinstance(cand_hash, str) and cand_hash in query_face_hashes
                )
                cand_embedding_raw = face.get("embedding")
                embedding_distance: Optional[float] = None
                embedding_match = False
                if query_embeddings and isinstance(cand_embedding_raw, list):
                    try:
                        cand_embedding = np.asarray(cand_embedding_raw, dtype=np.float32)
                        distances = []
                        for q_emb in query_embeddings:
                            if cand_embedding.shape == q_emb.shape and cand_embedding.size > 0:
                                distances.append(float(np.linalg.norm(q_emb - cand_embedding)))
                        if distances:
                            embedding_distance = min(distances)
                            embedding_match = embedding_distance <= embedding_threshold
                    except Exception:
                        embedding_distance = None

                # Keep result when either bib matches or face similarity matches.
                if (not bib_match) and (not hash_match) and (not embedding_match):
                    continue

                matches.append(
                    {
                        "report_file": report_path.name,
                        "album_url": album.get("album_url"),
                        "album_title": album.get("album_title"),
                        "image_name": resolve_image_name(image),
                        "source_url": image.get("source_url"),
                        "media_index": image.get("media_index"),
                        "bib_match": bib_match,
                        "face_id": face.get("face_id"),
                        "candidate_face_hash": cand_hash,
                        "hash_match": hash_match,
                        "embedding_match": embedding_match,
                        "embedding_distance": (
                            round(float(embedding_distance), 6) if embedding_distance is not None else None
                        ),
                    }
                )

    matches.sort(
        key=lambda m: (
            not bool(m["bib_match"]),
            not bool(m["hash_match"]),
            not bool(m["embedding_match"]),
            m["embedding_distance"] is None,
            float(m["embedding_distance"]) if m["embedding_distance"] is not None else 999.0,
        )
    )
    return {
        "reports_scanned": len(report_files),
        "candidate_faces_scanned": scanned_entries,
        # Do not truncate here; truncation before image-level aggregation can
        # hide face-only matches when bib matches are numerous.
        "matches": matches,
    }


def build_match_images(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    by_source: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        source_url = m.get("source_url")
        image_name = m.get("image_name")
        if not isinstance(source_url, str) or not source_url.strip():
            continue
        source_url = source_url.strip()
        entry = by_source.get(source_url)
        if entry is None:
            name = str(image_name).strip() if isinstance(image_name, str) and image_name.strip() else "unknown.jpg"
            entry = {
                "image_name": name,
                "source_url": source_url,
                "album_title": m.get("album_title") or "",
                "album_url": m.get("album_url") or "",
                "preview_url": f"{source_url}=w800-h560-no",
                "download_url": f"{source_url}=d",
                "_bib_match": False,
                "_hash_match": False,
                "_embedding_match": False,
            }
            by_source[source_url] = entry
        else:
            # Fill missing album metadata from any matching record.
            if not entry.get("album_title") and m.get("album_title"):
                entry["album_title"] = m.get("album_title")
            if not entry.get("album_url") and m.get("album_url"):
                entry["album_url"] = m.get("album_url")

        entry["_bib_match"] = bool(entry["_bib_match"] or bool(m.get("bib_match")))
        entry["_hash_match"] = bool(entry["_hash_match"] or bool(m.get("hash_match")))
        entry["_embedding_match"] = bool(entry["_embedding_match"] or bool(m.get("embedding_match")))

    for source_url, entry in by_source.items():
        bib_match = bool(entry.pop("_bib_match"))
        hash_match = bool(entry.pop("_hash_match"))
        embedding_match = bool(entry.pop("_embedding_match"))
        face_match = hash_match or embedding_match

        if bib_match and face_match:
            summary = "Bib + Face match"
        elif bib_match:
            summary = "Bib match"
        elif face_match:
            summary = "Face match"
        else:
            summary = "Match"

        entry["match_summary"] = summary
        entry["match_details"] = None
        items.append(entry)

    items.sort(key=lambda x: str(x.get("image_name", "")))
    return items


def decode_uploaded_image(file_storage: Any) -> np.ndarray:
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode uploaded image")
    return image


@app.get("/")
def index() -> Any:
    race_map, default_race = load_race_reports_config()
    return render_template(
        "index.html",
        race_options=[{"label": label, "path": str(path)} for label, path in race_map.items()],
        default_race=default_race,
    )


def build_match_payload_from_request() -> tuple[Optional[Dict[str, Any]], Optional[str], int]:
    bib_number = (request.form.get("bib_number") or "").strip()
    if not bib_number:
        return None, "bib_number is required", 400
    embedding_threshold_raw = (request.form.get("embedding_threshold") or "").strip()
    if embedding_threshold_raw:
        try:
            embedding_threshold = float(embedding_threshold_raw)
        except ValueError:
            return None, "embedding_threshold must be a positive float", 400
    else:
        embedding_threshold = _cfg_float(
            "WEBSITE_DEFAULT_EMBEDDING_THRESHOLD",
            "default_embedding_threshold",
            1.0,
        )
    if embedding_threshold <= 0.0:
        return None, "embedding_threshold must be > 0", 400

    photo = request.files.get("photo")
    if photo is None:
        return None, "photo is required", 400

    try:
        image = decode_uploaded_image(photo)
    except Exception as exc:
        return None, f"Invalid image: {exc}", 400

    det_size = _cfg_int("WEBSITE_INSIGHTFACE_DET_SIZE", "insightface_det_size", 640)
    try:
        face_app = get_face_app(det_size)
        detected = face_app.get(image)
    except Exception as exc:
        return None, f"InsightFace failed: {exc}", 500

    if not detected:
        return None, "No face detected in uploaded photo", 400

    query_hashes: List[str] = []
    query_hash_set = set()
    query_embeddings: List[np.ndarray] = []
    for face in detected:
        qh = get_face_hash(face)
        if isinstance(qh, str) and qh and qh not in query_hash_set:
            query_hashes.append(qh)
            query_hash_set.add(qh)
        qe = get_face_embedding(face)
        if qe is not None:
            query_embeddings.append(qe)

    selected_race = (request.form.get("race_label") or "").strip()
    reports_dir, resolved_race = resolve_reports_dir_by_race(selected_race)
    max_results = _cfg_int("WEBSITE_MAX_RESULTS", "max_results", 30)

    result = match_against_reports(
        reports_dir=reports_dir,
        bib_number=bib_number,
        query_face_hashes=query_hashes,
        query_embeddings=query_embeddings,
        embedding_threshold=embedding_threshold,
        max_results=max_results,
    )

    payload: Dict[str, Any] = {
        "bib_number": bib_number,
        "race_label": resolved_race,
        "reports_dir": str(reports_dir),
        "embedding_threshold": embedding_threshold,
        "query": {
            "detected_faces": len(detected),
            "query_face_hash_count": len(query_hashes),
            "query_embedding_count": len(query_embeddings),
            "selected_embedding_dim": int(query_embeddings[0].shape[0]) if query_embeddings else 0,
        },
        **result,
    }
    all_match_images = build_match_images(payload.get("matches", []))
    payload["match_images_total"] = len(all_match_images)
    payload["match_images"] = all_match_images[:max_results] if max_results > 0 else all_match_images
    return payload, None, 200


@app.post("/api/match")
def api_match() -> Any:
    payload, error, status = build_match_payload_from_request()
    if error is not None:
        return jsonify({"error": error}), status
    return jsonify(payload), status


@app.post("/results")
def results_page() -> Any:
    payload, error, status = build_match_payload_from_request()
    return render_template("results.html", payload=payload, error=error, status_code=status), status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run website for bib + face embedding report matching")
    parser.add_argument("--host", default=_cfg_str("WEBSITE_HOST", "host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_cfg_int("WEBSITE_PORT", "port", 8000))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

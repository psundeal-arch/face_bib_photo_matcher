#!/usr/bin/env python3
"""Simple web UI to match uploaded/captured photo against report JSON entries by hash + embedding."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

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
STATIC_IMAGES_DIR = APP_DIR / "static" / "images"

app = Flask(__name__, template_folder=str(APP_DIR / "templates"), static_folder=str(APP_DIR / "static"))

_face_app_lock = threading.Lock()
_face_app: Optional[object] = None
_report_cache_lock = threading.Lock()
_report_cache: Dict[str, Dict[str, Any]] = {}
_download_jobs_lock = threading.Lock()
_download_jobs: Dict[str, Dict[str, Any]] = {}
DOWNLOAD_JOB_TTL_SECONDS = 60 * 60


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


def slugify_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", label.lower()).strip("-")
    return normalized or "race"


def static_image_url_for_name(name: str) -> str:
    clean = Path(str(name)).name
    return f"static/images/{clean}"


def resolve_race_cover_image_url(race_label: str) -> str:
    cfg_map = WEBSITE_CONFIG.get("race_cover_images")
    if isinstance(cfg_map, dict):
        raw = cfg_map.get(race_label)
        if isinstance(raw, str) and raw.strip():
            configured = raw.strip()
            if configured.startswith("http://") or configured.startswith("https://"):
                return configured
            if configured.startswith("static/"):
                return configured
            if configured.startswith("images/"):
                return f"static/{configured}"
            return static_image_url_for_name(configured)

    slug = slugify_label(race_label)
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidate = STATIC_IMAGES_DIR / f"{slug}.{ext}"
        if candidate.is_file():
            return static_image_url_for_name(candidate.name)
    return ""


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


def sanitize_zip_entry_name(image_name: Any, index: int, used_names: set[str]) -> str:
    raw_name = str(image_name).strip() if isinstance(image_name, str) else ""
    file_name = Path(raw_name).name if raw_name else ""
    if not file_name:
        file_name = f"photo_{index:03d}.jpg"

    # Strip path traversal and normalize whitespace for a safer ZIP entry name.
    file_name = file_name.replace("\\", "_").replace("/", "_")
    file_name = re.sub(r"\s+", " ", file_name).strip()
    if not file_name:
        file_name = f"photo_{index:03d}.jpg"

    stem = Path(file_name).stem or f"photo_{index:03d}"
    suffix = Path(file_name).suffix
    candidate = f"{stem}{suffix}"
    dedupe_idx = 2
    while candidate in used_names:
        candidate = f"{stem}_{dedupe_idx}{suffix}"
        dedupe_idx += 1
    used_names.add(candidate)
    return candidate


def download_url_bytes(url: str, timeout_seconds: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "face-bib-photo-matcher/1.0"})
    with urlopen(req, timeout=timeout_seconds) as resp:  # nosec B310
        return bytes(resp.read())


def parse_match_images_json(raw_items: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str], int]:
    raw = raw_items.strip()
    if not raw:
        return None, "match_images_json is required", 400
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, "match_images_json must be valid JSON", 400
    if not isinstance(parsed, list):
        return None, "match_images_json must be a JSON list", 400
    normalized: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized, None, 200


def build_download_report_text(downloaded_count: int, failed_items: List[Dict[str, str]]) -> str:
    lines = [
        "Some photos could not be added to this ZIP.",
        "",
        f"Downloaded: {downloaded_count}",
        f"Failed: {len(failed_items)}",
        "",
        "Failures:",
    ]
    for i, failed in enumerate(failed_items, start=1):
        lines.append(f"{i}. image_name: {failed['image_name']}")
        lines.append(f"   download_url: {failed['download_url'] or '(missing)'}")
        lines.append(f"   error: {failed['error']}")
    lines.append("")
    return "\n".join(lines)


def build_zip_from_match_items(
    match_items: List[Dict[str, Any]],
    zf: zipfile.ZipFile,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> tuple[int, List[Dict[str, str]]]:
    downloaded_count = 0
    failed_items: List[Dict[str, str]] = []
    used_names: set[str] = set()

    for idx, item in enumerate(match_items, start=1):
        download_url = item.get("download_url")
        if not isinstance(download_url, str) or not download_url.strip():
            failed_items.append(
                {
                    "image_name": str(item.get("image_name") or f"photo_{idx:03d}.jpg"),
                    "download_url": "",
                    "error": "missing download_url",
                }
            )
            if progress_callback is not None:
                progress_callback(idx, downloaded_count, len(failed_items))
            continue

        try:
            content = download_url_bytes(download_url.strip())
        except Exception as exc:
            failed_items.append(
                {
                    "image_name": str(item.get("image_name") or f"photo_{idx:03d}.jpg"),
                    "download_url": download_url.strip(),
                    "error": str(exc),
                }
            )
            if progress_callback is not None:
                progress_callback(idx, downloaded_count, len(failed_items))
            continue

        entry_name = sanitize_zip_entry_name(item.get("image_name"), idx, used_names)
        zf.writestr(entry_name, content)
        downloaded_count += 1
        if progress_callback is not None:
            progress_callback(idx, downloaded_count, len(failed_items))

    if failed_items:
        zf.writestr("_download_report.txt", build_download_report_text(downloaded_count, failed_items))
    return downloaded_count, failed_items


def cleanup_download_jobs() -> None:
    now = time.time()
    to_remove: List[str] = []
    with _download_jobs_lock:
        for job_id, job in _download_jobs.items():
            updated_at = float(job.get("updated_at", job.get("created_at", now)))
            if now - updated_at < DOWNLOAD_JOB_TTL_SECONDS:
                continue
            zip_path = job.get("zip_path")
            if isinstance(zip_path, str) and zip_path:
                try:
                    Path(zip_path).unlink(missing_ok=True)
                except Exception:
                    pass
            to_remove.append(job_id)
        for job_id in to_remove:
            _download_jobs.pop(job_id, None)


def run_download_zip_job(job_id: str, match_items: List[Dict[str, Any]], zip_path: Path) -> None:
    with _download_jobs_lock:
        job = _download_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["updated_at"] = time.time()

    def on_progress(processed: int, downloaded: int, failed: int) -> None:
        with _download_jobs_lock:
            job2 = _download_jobs.get(job_id)
            if job2 is None:
                return
            job2["processed"] = processed
            job2["downloaded"] = downloaded
            job2["failed"] = failed
            job2["updated_at"] = time.time()

    try:
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            downloaded_count, failed_items = build_zip_from_match_items(match_items, zf, progress_callback=on_progress)

        if downloaded_count == 0:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass
            with _download_jobs_lock:
                job3 = _download_jobs.get(job_id)
                if job3 is not None:
                    job3["status"] = "failed"
                    job3["error"] = "Unable to download any images for ZIP export."
                    job3["failed_items"] = failed_items
                    job3["updated_at"] = time.time()
            return

        with _download_jobs_lock:
            job4 = _download_jobs.get(job_id)
            if job4 is not None:
                job4["status"] = "completed"
                job4["processed"] = len(match_items)
                job4["downloaded"] = downloaded_count
                job4["failed"] = len(failed_items)
                job4["zip_path"] = str(zip_path)
                job4["updated_at"] = time.time()
    except Exception as exc:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
        with _download_jobs_lock:
            job5 = _download_jobs.get(job_id)
            if job5 is not None:
                job5["status"] = "failed"
                job5["error"] = str(exc)
                job5["updated_at"] = time.time()


def create_download_zip_job(match_items: List[Dict[str, Any]]) -> str:
    cleanup_download_jobs()
    job_id = uuid.uuid4().hex
    zip_path = Path(tempfile.gettempdir()) / f"matched_photos_{job_id}.zip"
    now = time.time()
    with _download_jobs_lock:
        _download_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "total": len(match_items),
            "processed": 0,
            "downloaded": 0,
            "failed": 0,
            "zip_path": "",
            "error": "",
        }
    t = threading.Thread(target=run_download_zip_job, args=(job_id, match_items, zip_path), daemon=True)
    t.start()
    return job_id


@app.get("/")
def cover_page() -> Any:
    race_map, default_race = load_race_reports_config()
    race_options = []
    for label, path in race_map.items():
        race_options.append(
            {
                "label": label,
                "path": str(path),
                "cover_image_url": resolve_race_cover_image_url(label),
            }
        )
    return render_template(
        "cover.html",
        race_options=race_options,
        default_race=default_race,
    )


@app.get("/input")
def input_page() -> Any:
    race_map, default_race = load_race_reports_config()
    race_options = []
    for label, path in race_map.items():
        race_options.append(
            {
                "label": label,
                "path": str(path),
                "cover_image_url": resolve_race_cover_image_url(label),
            }
        )
    selected_race = (request.args.get("race_label") or "").strip()
    if selected_race not in race_map:
        selected_race = default_race
    return render_template(
        "index.html",
        race_options=race_options,
        default_race=selected_race,
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


@app.post("/download-zip/start")
@app.post("/results/download-zip/start")
def download_zip_start() -> Any:
    match_items, error, status = parse_match_images_json(request.form.get("match_images_json", ""))
    if error is not None or match_items is None:
        return jsonify({"error": error}), status
    job_id = create_download_zip_job(match_items)
    return jsonify({"job_id": job_id}), 202


@app.get("/download-zip/status/<job_id>")
@app.get("/results/download-zip/status/<job_id>")
def download_zip_status(job_id: str) -> Any:
    cleanup_download_jobs()
    with _download_jobs_lock:
        job = _download_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found or expired"}), 404
        return jsonify(
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "total": int(job.get("total", 0)),
                "processed": int(job.get("processed", 0)),
                "downloaded": int(job.get("downloaded", 0)),
                "failed": int(job.get("failed", 0)),
                "error": str(job.get("error", "")),
            }
        )


@app.get("/download-zip/file/<job_id>")
@app.get("/results/download-zip/file/<job_id>")
def download_zip_file(job_id: str) -> Any:
    cleanup_download_jobs()
    with _download_jobs_lock:
        job = _download_jobs.get(job_id)
        if job is None:
            return "job not found or expired", 404
        status = str(job.get("status", ""))
        zip_path_text = str(job.get("zip_path", ""))
        if status != "completed":
            return "zip is not ready yet", 409
    zip_path = Path(zip_path_text)
    if not zip_path.exists():
        return "zip file no longer exists", 404
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="matched_photos.zip",
    )


@app.post("/download-zip")
@app.post("/results/download-zip")
def download_zip() -> Any:
    match_items, error, status = parse_match_images_json(request.form.get("match_images_json", ""))
    if error is not None or match_items is None:
        return error or "invalid request", status

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        downloaded_count, _failed_items = build_zip_from_match_items(match_items, zf)

    if downloaded_count == 0:
        return "Unable to download any images for ZIP export. See server logs/report for details.", 502

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="matched_photos.zip",
    )


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

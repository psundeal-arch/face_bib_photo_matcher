#!/usr/bin/env python3
"""Cluster faces in an album report into distinct persons and write a persons index.

Reads a per-album report JSON (produced by shared_album_downloader.py) and groups
face embeddings into distinct people using DBSCAN on cosine distance. DBSCAN's
min_samples core-point rule avoids the single-link "chaining" that made the old
greedy face_id assignment explode into thousands of clusters.

Output: a sidecar file `<report_stem>.persons.json` next to the report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

import numpy as np

from photo_urls import photo_download_url, photo_preview_url, supports_face_crop

DEFAULT_DET_MIN = 0.6
DEFAULT_COSINE_MIN = 0.50  # two faces are neighbors when cosine similarity >= this
DEFAULT_MIN_SAMPLES = 3
PREVIEW_SUFFIX = "=w800-h560-no"
DOWNLOAD_SUFFIX = "=d"
# Detection sample width for locating a face box; bbox is stored as fractions so
# the resolution used here does not affect the final crop.
FACE_CROP_SAMPLE_WIDTH = 1600
FACE_CROP_THUMB = "=w512-h512-p"
FACE_CROP_MATCH_MIN = 0.35  # min cosine to accept the detected face as this person


def _preview_url(source_url: str) -> str:
    return photo_preview_url(source_url)


def _download_url(source_url: str) -> str:
    return photo_download_url(source_url)


def collect_faces(report: Dict[str, Any], det_min: float) -> List[Dict[str, Any]]:
    """Flatten report images into face records that pass the quality filter."""
    faces: List[Dict[str, Any]] = []
    images = report.get("images", [])
    if not isinstance(images, list):
        return faces

    for image in images:
        if not isinstance(image, dict) or image.get("status") != "ok":
            continue
        source_url = image.get("source_url")
        if not isinstance(source_url, str) or not source_url.strip():
            continue
        image_name = image.get("file_name") or "unknown.jpg"
        image_bibs = image.get("bib_numbers", [])
        image_bibs = image_bibs if isinstance(image_bibs, list) else []

        for face in image.get("faces", []):
            if not isinstance(face, dict):
                continue
            emb = face.get("embedding")
            det_score = face.get("det_score")
            if not isinstance(emb, list) or not emb:
                continue
            if not isinstance(det_score, (int, float)) or float(det_score) < det_min:
                continue
            near_bibs = face.get("bib_numbers_near_face", [])
            faces.append(
                {
                    "embedding": np.asarray(emb, dtype=np.float32),
                    "det_score": float(det_score),
                    "source_url": source_url.strip(),
                    "image_name": str(image_name),
                    "image_bibs": image_bibs,
                    # Numbers OCR'd from the torso crop directly under THIS face.
                    "near_bibs": near_bibs if isinstance(near_bibs, list) else [],
                }
            )
    return faces


def _vote_bib(member_faces: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick a person's bib by voting the per-face torso-crop reads across photos.

    Ranks by number of distinct photos the number was read under this person's
    face (support), then by summed OCR confidence. Returns (best, candidates).
    A bib confirmed in >=2 photos is far more reliable than a single read.
    """
    score: Dict[str, float] = {}
    photos: Dict[str, set] = {}
    for face in member_faces:
        for item in face.get("near_bibs", []):
            if not isinstance(item, dict) or item.get("number") is None:
                continue
            number = str(item["number"])
            try:
                conf = float(item.get("confidence", 0) or 0)
            except Exception:
                conf = 0.0
            score[number] = score.get(number, 0.0) + conf
            photos.setdefault(number, set()).add(face["source_url"])
    if not score:
        return None, []
    ranked = sorted(score, key=lambda n: (len(photos[n]), score[n]), reverse=True)
    candidates = [
        {
            "number": n,
            "support": len(photos[n]),
            "confidence": round(score[n] / max(1, len(photos[n])), 2),
        }
        for n in ranked[:3]
    ]
    return candidates[0], candidates


def dbscan_cosine(
    embeddings: np.ndarray,
    cosine_min: float,
    min_samples: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    """DBSCAN over unit-normalized embeddings using cosine similarity.

    Returns an int label array; -1 marks noise. Neighbor queries are chunked so
    the full N*N similarity matrix is never materialized at once.
    """
    n = int(embeddings.shape[0])
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    neighbors: List[np.ndarray] = [np.empty(0, dtype=np.int64)] * n
    is_core = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sims = norm[start:end] @ norm.T  # (chunk, n)
        adj = sims >= cosine_min
        for row in range(end - start):
            idx = np.flatnonzero(adj[row])
            i = start + row
            neighbors[i] = idx
            if idx.size >= min_samples:
                is_core[i] = True

    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cluster_id
        stack = [i]
        while stack:
            u = stack.pop()
            for v in neighbors[u]:
                if labels[v] == -1:
                    labels[v] = cluster_id
                    if is_core[v]:
                        stack.append(v)
        cluster_id += 1
    return labels


def _top_bibs(bib_lists: Iterable[List[Any]], limit: int = 3) -> List[Dict[str, Any]]:
    best: Dict[str, float] = {}
    for bibs in bib_lists:
        for item in bibs:
            if not isinstance(item, dict):
                continue
            number = item.get("number")
            conf = item.get("confidence", 0)
            if number is None:
                continue
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            key = str(number)
            if conf_f > best.get(key, -1.0):
                best[key] = conf_f
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"number": n, "confidence": round(c, 2)} for n, c in ranked[:limit]]


def build_persons(
    faces: List[Dict[str, Any]],
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    clusters: Dict[int, List[int]] = {}
    for face_idx, label in enumerate(labels):
        label_int = int(label)
        if label_int < 0:
            continue
        clusters.setdefault(label_int, []).append(face_idx)

    persons: List[Dict[str, Any]] = []
    for members in clusters.values():
        # Representative = highest detection score in the cluster.
        rep_idx = max(members, key=lambda i: faces[i]["det_score"])
        rep = faces[rep_idx]

        photos_by_url: Dict[str, Dict[str, Any]] = {}
        for i in members:
            face = faces[i]
            url = face["source_url"]
            entry = photos_by_url.get(url)
            if entry is None:
                photos_by_url[url] = {
                    "image_name": face["image_name"],
                    "source_url": url,
                    "preview_url": _preview_url(url),
                    "download_url": _download_url(url),
                    "_best_det": face["det_score"],
                }
            else:
                entry["_best_det"] = max(entry["_best_det"], face["det_score"])

        photos = sorted(photos_by_url.values(), key=lambda p: str(p["image_name"]))
        for p in photos:
            p.pop("_best_det", None)

        persons.append(
            {
                "face_count": len(members),
                "photo_count": len(photos),
                "representative": {
                    "image_name": rep["image_name"],
                    "source_url": rep["source_url"],
                    "preview_url": _preview_url(rep["source_url"]),
                    "download_url": _download_url(rep["source_url"]),
                    "det_score": round(rep["det_score"], 4),
                },
                # Legacy: every number seen anywhere in this person's photos (noisy).
                "bib_numbers": _top_bibs((faces[i]["image_bibs"] for i in members)),
                # Voted: the number read under THIS person's own face across photos.
                "bib": _vote_bib([faces[i] for i in members])[0],
                "bib_candidates": _vote_bib([faces[i] for i in members])[1],
                "photos": photos,
                # Transient: index of the representative face, used by the optional
                # face-crop pass. Removed before serialization.
                "_rep_face_index": rep_idx,
            }
        )

    # Most-photographed people first.
    persons.sort(key=lambda p: (-p["photo_count"], -p["face_count"]))
    for rank, person in enumerate(persons, start=1):
        person["person_id"] = f"person_{rank:04d}"
    # Move person_id to the front for readability.
    return [
        {"person_id": p.pop("person_id"), **p} for p in persons
    ]


def _download_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "face-bib-photo-matcher/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # nosec B310
        return bytes(resp.read())


def _fcrop64_hex(left: float, top: float, right: float, bottom: float) -> str:
    def enc(v: float) -> str:
        v = min(1.0, max(0.0, v))
        return format(int(round(v * 0xFFFF)), "04x")

    return "".join(enc(v) for v in (left, top, right, bottom))


def _face_crop_url(source_url: str, crop_hex: str) -> str:
    return f"{source_url}{FACE_CROP_THUMB}-fcrop64=1,{crop_hex}"


def compute_face_crops(
    persons: List[Dict[str, Any]],
    faces: List[Dict[str, Any]],
    provider_mode: str = "auto",
    det_size: int = 640,
) -> Tuple[int, int]:
    """Locate each person's face in their representative photo and store a Google
    fcrop64 region URL so the CDN serves a face-only thumbnail (no local cropping).

    Returns (resolved, attempted). The right face is chosen by matching the
    detected embeddings against the person's representative embedding, so a group
    photo yields this person's face rather than the largest/nearest one.
    """
    import cv2  # local import: only needed for the optional crop pass
    from face_scanner import create_insightface_app

    face_app = create_insightface_app(det_size=det_size, provider_mode=provider_mode)

    resolved = 0
    attempted = 0
    for person in persons:
        rep_idx = person.get("_rep_face_index")
        if rep_idx is None:
            continue
        source_url = person["representative"]["source_url"]
        attempted += 1
        ref = faces[rep_idx]["embedding"].astype(np.float32)
        ref = ref / (np.linalg.norm(ref) + 1e-12)
        try:
            # Google can transform via URL; other hosts serve a fixed image.
            if supports_face_crop(source_url):
                det_url = f"{source_url}=w{FACE_CROP_SAMPLE_WIDTH}"
            else:
                det_url = photo_download_url(source_url)  # e.g. RunSignup large_v3
            data = _download_bytes(det_url)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            detected = face_app.get(img)
            if not detected:
                continue
            best_face = None
            best_cos = -1.0
            for det in detected:
                emb = getattr(det, "normed_embedding", None)
                if emb is None:
                    emb = getattr(det, "embedding", None)
                if emb is None:
                    continue
                emb = np.asarray(emb, np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                cos = float(np.dot(ref, emb))
                if cos > best_cos:
                    best_cos = cos
                    best_face = det
            if best_face is None or best_cos < FACE_CROP_MATCH_MIN:
                continue
            l, t, r, b = [float(v) for v in np.asarray(best_face.bbox, np.float32)[:4]]
            fw, fh = max(1.0, r - l), max(1.0, b - t)

            if supports_face_crop(source_url):
                # Google fcrop64: server crops to this rectangle (fractions).
                left = (l - 0.6 * fw) / w
                top = (t - 0.7 * fh) / h
                right = (r + 0.6 * fw) / w
                bottom = (b + 0.9 * fh) / h
                crop_hex = _fcrop64_hex(left, top, right, bottom)
                person["representative"]["face_crop"] = crop_hex
                person["representative"]["face_preview_url"] = _face_crop_url(source_url, crop_hex)
            else:
                # CSS crop: store how to position the fixed-size image in a square
                # tile so only the face shows (cropping happens in the browser).
                # Display the larger image so small/far faces stay crisp when zoomed.
                person["representative"]["face_box"] = _css_face_box(
                    l, t, r, b, w, h, photo_download_url(source_url)
                )
            resolved += 1
        except Exception:
            continue
    return resolved, attempted


def _css_face_box(
    l: float, t: float, r: float, b: float, w: int, h: int, src: str
) -> Dict[str, Any]:
    """Positioning for a square head-and-shoulders crop rendered with CSS.

    The image is placed absolutely inside a square tile: `width_pct` scales it so
    the crop square fills the tile width; `left_pct`/`top_pct` shift it (percent of
    the tile) so the crop square's top-left sits at the tile origin. Values are
    resolution-independent, so a smaller display image can be used than the one
    detected on.
    """
    fw, fh = max(1.0, r - l), max(1.0, b - t)
    cx, cy = (l + r) / 2.0, (t + b) / 2.0
    side = max(fw, fh) * 2.4  # head + shoulders
    # Keep the square inside the image where it fits.
    half = side / 2.0
    if side <= w:
        cx = min(max(cx, half), w - half)
    if side <= h:
        cy = min(max(cy, half), h - half)
    return {
        "src": src,
        "width_pct": round(w / side * 100.0, 3),
        "left_pct": round(-((cx - half) / side) * 100.0, 3),
        "top_pct": round(-((cy - half) / side) * 100.0, 3),
    }


def build_person_index(
    report_path: Path,
    det_min: float,
    cosine_min: float,
    min_samples: int,
    face_crops: bool = False,
    provider_mode: str = "auto",
) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    faces = collect_faces(report, det_min=det_min)

    if faces:
        embeddings = np.vstack([f["embedding"] for f in faces])
        labels = dbscan_cosine(embeddings, cosine_min=cosine_min, min_samples=min_samples)
    else:
        labels = np.empty(0, dtype=np.int64)

    persons = build_persons(faces, labels)
    clustered = int((labels >= 0).sum()) if labels.size else 0
    noise = int((labels < 0).sum()) if labels.size else 0

    crops_resolved = 0
    if face_crops and persons:
        crops_resolved, crops_attempted = compute_face_crops(persons, faces, provider_mode=provider_mode)
        print(f"  face crops: {crops_resolved}/{crops_attempted} resolved")

    # Drop transient fields before serialization.
    for person in persons:
        person.pop("_rep_face_index", None)

    album = report.get("album", {}) if isinstance(report.get("album"), dict) else {}
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_report": report_path.name,
        "album_title": album.get("album_title"),
        "album_url": album.get("album_url"),
        "params": {
            "det_min": det_min,
            "cosine_min": cosine_min,
            "min_samples": min_samples,
            "method": "dbscan_cosine",
            "face_crops": bool(face_crops),
        },
        "person_count": len(persons),
        "faces_considered": len(faces),
        "faces_clustered": clustered,
        "faces_noise": noise,
        "face_crops_resolved": crops_resolved,
        "persons": persons,
    }


def persons_sidecar_path(report_path: Path) -> Path:
    return report_path.with_suffix(".persons.json")


def run(
    paths: List[Path],
    det_min: float,
    cosine_min: float,
    min_samples: int,
    face_crops: bool = False,
    provider_mode: str = "auto",
) -> int:
    report_files: List[Path] = []
    for path in paths:
        if path.is_dir():
            report_files.extend(
                sorted(p for p in path.glob("*.json") if not p.name.endswith((".persons.json", ".urls.json")))
            )
        elif path.is_file():
            report_files.append(path)
        else:
            print(f"WARN: not found: {path}")

    if not report_files:
        print("No report files to process.")
        return 2

    for report_path in report_files:
        try:
            index = build_person_index(
                report_path,
                det_min=det_min,
                cosine_min=cosine_min,
                min_samples=min_samples,
                face_crops=face_crops,
                provider_mode=provider_mode,
            )
        except Exception as exc:
            print(f"ERROR: failed to process {report_path.name}: {exc}")
            continue
        out_path = persons_sidecar_path(report_path)
        out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(
            f"{report_path.name}: {index['person_count']} persons "
            f"({index['faces_clustered']} faces clustered, {index['faces_noise']} noise) "
            f"-> {out_path.name}"
        )
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster album report faces into distinct persons and write a persons index."
    )
    parser.add_argument("paths", nargs="+", help="Report JSON file(s) or directory of reports")
    parser.add_argument("--det-min", type=float, default=DEFAULT_DET_MIN, help="Min detection score to include a face")
    parser.add_argument("--cosine-min", type=float, default=DEFAULT_COSINE_MIN, help="Min cosine similarity for two faces to be neighbors")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="DBSCAN core-point neighbor count")
    parser.add_argument(
        "--face-crops",
        action="store_true",
        help="Locate each person's face (InsightFace + network) and store a Google fcrop64 face-thumbnail URL",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "cpu", "coreml"],
        help="InsightFace execution provider for the face-crop pass (default: auto)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    args = parse_args()
    return run(
        paths=[Path(p) for p in args.paths],
        det_min=args.det_min,
        cosine_min=args.cosine_min,
        min_samples=args.min_samples,
        face_crops=args.face_crops,
        provider_mode=args.provider,
    )


if __name__ == "__main__":
    raise SystemExit(main())

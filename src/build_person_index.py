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

import numpy as np

DEFAULT_DET_MIN = 0.6
DEFAULT_COSINE_MIN = 0.50  # two faces are neighbors when cosine similarity >= this
DEFAULT_MIN_SAMPLES = 3
PREVIEW_SUFFIX = "=w800-h560-no"
DOWNLOAD_SUFFIX = "=d"


def _preview_url(source_url: str) -> str:
    return f"{source_url}{PREVIEW_SUFFIX}"


def _download_url(source_url: str) -> str:
    return f"{source_url}{DOWNLOAD_SUFFIX}"


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
            faces.append(
                {
                    "embedding": np.asarray(emb, dtype=np.float32),
                    "det_score": float(det_score),
                    "source_url": source_url.strip(),
                    "image_name": str(image_name),
                    "image_bibs": image_bibs,
                }
            )
    return faces


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
                "bib_numbers": _top_bibs((faces[i]["image_bibs"] for i in members)),
                "photos": photos,
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


def build_person_index(
    report_path: Path,
    det_min: float,
    cosine_min: float,
    min_samples: int,
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
        },
        "person_count": len(persons),
        "faces_considered": len(faces),
        "faces_clustered": clustered,
        "faces_noise": noise,
        "persons": persons,
    }


def persons_sidecar_path(report_path: Path) -> Path:
    return report_path.with_suffix(".persons.json")


def run(paths: List[Path], det_min: float, cosine_min: float, min_samples: int) -> int:
    report_files: List[Path] = []
    for path in paths:
        if path.is_dir():
            report_files.extend(
                sorted(p for p in path.glob("*.json") if not p.name.endswith(".persons.json"))
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
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    args = parse_args()
    return run(
        paths=[Path(p) for p in args.paths],
        det_min=args.det_min,
        cosine_min=args.cosine_min,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Find known people from JSON registry who appear in a given photo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency guard
    cv2 = None

import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - runtime dependency guard
    FaceAnalysis = None


def create_insightface_app(det_size: int) -> object:
    if FaceAnalysis is None:
        raise RuntimeError(
            "InsightFace is not installed. Install with: pip install insightface onnxruntime"
        )
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if sys.platform == "darwin" else None
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    if providers is None:
        try:
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    else:
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def face_area(face: object) -> float:
    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    if bbox.shape[0] < 4:
        return 0.0
    x1, y1, x2, y2 = bbox[:4]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def extract_embeddings(face_app: object, image_path: Path, all_faces: bool) -> List[np.ndarray]:
    image = read_image(image_path)
    detected = face_app.get(image)
    if not detected:
        return []
    detected_sorted: List[object] = sorted(detected, key=face_area, reverse=True)
    embeddings: List[np.ndarray] = []
    for face in detected_sorted:
        embedding_raw = getattr(face, "normed_embedding", None)
        if embedding_raw is None:
            embedding_raw = getattr(face, "embedding", None)
        if embedding_raw is None:
            continue
        embeddings.append(np.asarray(embedding_raw, dtype=np.float32))
        if not all_faces:
            break
    return embeddings


def load_registry(path: Path) -> List[dict]:
    if not path.exists():
        raise RuntimeError(f"Registry file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse registry JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid registry format in '{path}': root must be object")
    people = payload.get("people")
    if not isinstance(people, list):
        raise RuntimeError(f"Invalid registry format in '{path}': 'people' must be a list")
    return people


def build_gallery_embeddings(
    face_app: object,
    people: List[dict],
    sample_all_faces: bool,
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    gallery: Dict[str, List[np.ndarray]] = {}
    warnings: List[str] = []

    for person in people:
        if not isinstance(person, dict):
            continue
        name = person.get("name")
        samples = person.get("samples")
        if not isinstance(name, str) or not isinstance(samples, list):
            continue

        person_embeddings: List[np.ndarray] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_path_raw = sample.get("photo_path")
            if not isinstance(sample_path_raw, str):
                continue
            sample_path = Path(sample_path_raw)
            if not sample_path.exists():
                warnings.append(f"Missing sample photo for '{name}': {sample_path}")
                continue
            try:
                embs = extract_embeddings(
                    face_app=face_app,
                    image_path=sample_path,
                    all_faces=sample_all_faces,
                )
            except Exception as exc:
                warnings.append(f"Failed sample photo for '{name}' ({sample_path}): {exc}")
                continue
            person_embeddings.extend(embs)

        if person_embeddings:
            gallery[name] = person_embeddings
        else:
            warnings.append(f"No usable face embeddings found for '{name}'")

    return gallery, warnings


def best_distance(query: np.ndarray, gallery: List[np.ndarray]) -> float:
    gallery_matrix = np.vstack(gallery)
    distances = np.linalg.norm(gallery_matrix - query, axis=1)
    return float(np.min(distances))


def run(
    photo_path: Path,
    db_path: Path,
    det_size: int,
    threshold: float,
    sample_all_faces: bool,
    query_all_faces: bool,
    as_json: bool,
) -> int:
    if cv2 is None:
        print("ERROR: OpenCV is not installed. Install with: pip install opencv-python")
        return 2
    if not photo_path.exists():
        print(f"ERROR: Photo not found: {photo_path}")
        return 2

    try:
        people = load_registry(db_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not people:
        print(f"ERROR: Registry has no people: {db_path}")
        return 2

    face_app = create_insightface_app(det_size)

    gallery, warnings = build_gallery_embeddings(
        face_app=face_app,
        people=people,
        sample_all_faces=sample_all_faces,
    )
    if not gallery:
        print("ERROR: Could not build face gallery from registry samples")
        for w in warnings:
            print(f"WARN: {w}")
        return 2

    query_embeddings = extract_embeddings(
        face_app=face_app,
        image_path=photo_path,
        all_faces=query_all_faces,
    )
    if not query_embeddings:
        print(f"No faces detected in query photo: {photo_path}")
        for w in warnings:
            print(f"WARN: {w}")
        return 1

    matches: List[dict] = []
    for name, emb_list in gallery.items():
        best = min(best_distance(q, emb_list) for q in query_embeddings)
        if best <= threshold:
            matches.append({"name": name, "distance": round(float(best), 6)})

    matches.sort(key=lambda x: float(x["distance"]))

    if as_json:
        payload = {
            "photo_path": str(photo_path.resolve()),
            "db_path": str(db_path.resolve()),
            "threshold": threshold,
            "query_face_count": len(query_embeddings),
            "matched_people": matches,
            "warnings": warnings,
        }
        print(json.dumps(payload, indent=2))
        return 0 if matches else 1

    print(f"photo: {photo_path.resolve()}")
    print(f"query_faces: {len(query_embeddings)}")
    print(f"known_people_indexed: {len(gallery)}")
    print(f"threshold: {threshold}")
    if matches:
        print("matches:")
        for m in matches:
            print(f"{m['name']} (distance={m['distance']})")
    else:
        print("matches: none")
    for w in warnings:
        print(f"WARN: {w}")

    return 0 if matches else 1


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Given a photo, find which people from JSON registry are present."
    )
    parser.add_argument("photo_path", help="Path to query photo")
    parser.add_argument(
        "--db",
        default="people_face_hashes.json",
        help="JSON registry path (default: people_face_hashes.json)",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="InsightFace detector size (default: 640)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="L2 distance threshold for match (default: 1.0)",
    )
    parser.add_argument(
        "--sample-all-faces",
        action="store_true",
        help="Use all detected faces from each sample photo (default: largest only)",
    )
    parser.add_argument(
        "--query-largest-face-only",
        action="store_true",
        help="Use only largest face in query photo (default: all query faces)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON result",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    args = parse_args()
    return run(
        photo_path=Path(args.photo_path),
        db_path=Path(args.db),
        det_size=args.det_size,
        threshold=args.threshold,
        sample_all_faces=args.sample_all_faces,
        query_all_faces=not args.query_largest_face_only,
        as_json=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())

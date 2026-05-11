#!/usr/bin/env python3
"""Generate InsightFace hash code(s) and store person entries in a JSON registry."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def embedding_hash(embedding: np.ndarray) -> str:
    rounded = np.round(embedding.astype(np.float32), 4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def face_area(face: object) -> float:
    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    if bbox.shape[0] < 4:
        return 0.0
    x1, y1, x2, y2 = bbox[:4]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def to_face_record(face: object, index: int) -> Optional[dict]:
    embedding_raw = getattr(face, "normed_embedding", None)
    if embedding_raw is None:
        embedding_raw = getattr(face, "embedding", None)
    if embedding_raw is None:
        return None

    embedding = np.asarray(embedding_raw, dtype=np.float32)
    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    if bbox.shape[0] < 4:
        return None

    det_score = getattr(face, "det_score", None)
    return {
        "face_index": index,
        "hashcode": embedding_hash(embedding),
        "embedding": [round(float(v), 6) for v in embedding],
        "det_score": float(det_score) if det_score is not None else None,
        "area": face_area(face),
    }


def default_registry() -> Dict[str, Any]:
    return {"version": 1, "people": []}


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return default_registry()
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON registry '{path}': {exc}") from exc

    if not isinstance(content, dict):
        raise RuntimeError(f"Invalid registry format in '{path}': root must be object")
    people = content.get("people")
    if not isinstance(people, list):
        raise RuntimeError(f"Invalid registry format in '{path}': 'people' must be a list")
    if "version" not in content:
        content["version"] = 1
    return content


def find_person(people: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for person in people:
        if isinstance(person, dict) and person.get("name") == name:
            return person
    return None


def upsert_person_sample(
    registry: Dict[str, Any],
    name: str,
    image_path: Path,
    records: List[dict],
) -> Dict[str, Any]:
    people = registry.setdefault("people", [])
    if not isinstance(people, list):
        raise RuntimeError("Invalid registry in memory: 'people' must be a list")

    person = find_person(people, name)
    if person is None:
        person = {"name": name, "samples": []}
        people.append(person)

    samples = person.setdefault("samples", [])
    if not isinstance(samples, list):
        raise RuntimeError(f"Invalid person record for '{name}': 'samples' must be a list")

    abs_photo_path = str(image_path.resolve())
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    sample = {
        "photo_path": abs_photo_path,
        "added_at": now,
        "face_count": len(records),
        "hashcodes": [r["hashcode"] for r in records],
        "faces": records,
    }

    replaced = False
    for i, existing in enumerate(samples):
        if isinstance(existing, dict) and existing.get("photo_path") == abs_photo_path:
            samples[i] = sample
            replaced = True
            break
    if not replaced:
        samples.append(sample)

    return {"sample": sample, "replaced": replaced}


def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def run(name: str, image_path: Path, db_path: Path, det_size: int, all_faces: bool) -> int:
    if cv2 is None:
        print("ERROR: OpenCV is not installed. Install with: pip install opencv-python")
        return 2

    clean_name = name.strip()
    if not clean_name:
        print("ERROR: Name cannot be empty")
        return 2

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 2

    app = create_insightface_app(det_size)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Failed to read image: {image_path}")
        return 2

    detected = app.get(image)
    if not detected:
        print("ERROR: No face detected")
        return 1

    detected_sorted: List[object] = sorted(detected, key=face_area, reverse=True)
    records: List[dict] = []
    for i, face in enumerate(detected_sorted, start=1):
        record = to_face_record(face, i)
        if record is not None:
            records.append(record)

    if not records:
        print("ERROR: Face detected but no usable embeddings returned")
        return 1

    if not all_faces:
        records = records[:1]

    try:
        registry = load_registry(db_path)
        outcome = upsert_person_sample(
            registry=registry,
            name=clean_name,
            image_path=image_path,
            records=records,
        )
        save_registry(db_path, registry)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    sample = outcome["sample"]
    action = "Updated" if outcome["replaced"] else "Added"
    print(f"{action} entry for '{clean_name}' in {db_path.resolve()}")
    print(f"photo_path: {sample['photo_path']}")
    print(f"face_count: {sample['face_count']}")
    print("hashcodes:")
    for hashcode in sample["hashcodes"]:
        print(hashcode)

    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InsightFace on a photo and store person face hashcode(s) in JSON."
    )
    parser.add_argument("name", help="Person name")
    parser.add_argument("photo_path", help="Path to input image file")
    parser.add_argument(
        "--db",
        default="people_face_hashes.json",
        help="JSON file path used to store multiple people (default: people_face_hashes.json)",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="InsightFace detector size (default: 640)",
    )
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="Output hashes for all detected faces (default: largest face only)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> int:
    args = parse_args()
    return run(
        name=args.name,
        image_path=Path(args.photo_path),
        db_path=Path(args.db),
        det_size=args.det_size,
        all_faces=args.all_faces,
    )


if __name__ == "__main__":
    raise SystemExit(main())

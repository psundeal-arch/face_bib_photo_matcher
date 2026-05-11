#!/usr/bin/env python3
"""Download a shared Google Photos album, detect faces, and extract bib numbers to JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - runtime dependency guard
    RapidOCR = None
try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - runtime dependency guard
    FaceAnalysis = None

from shared_album_downloader import run as download_shared_album

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
BIB_RE = re.compile(r"\b\d{2,5}\b")
_OCR_TLS = threading.local()


def get_ocr_engine() -> Optional[object]:
    if RapidOCR is None:
        return None
    engine = getattr(_OCR_TLS, "engine", None)
    if engine is not None:
        return engine
    _OCR_TLS.engine = RapidOCR()
    return _OCR_TLS.engine


def iter_images(root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def encoding_hash(encoding: np.ndarray) -> str:
    rounded = np.round(encoding.astype(np.float32), 4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def crop_torso_region(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
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


def ocr_bib_candidates(image_bgr: np.ndarray, min_score: float = 0.45) -> Dict[str, float]:
    engine = get_ocr_engine()
    if engine is None:
        return {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    th1 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    variants = [gray, th2]

    out: Dict[str, float] = {}
    for var in variants:
        raw = engine(var)
        lines = raw[0] if isinstance(raw, tuple) and raw else raw
        if not isinstance(lines, list):
            continue
        for item in lines:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = None
            score = 0.0
            second = item[1]
            if isinstance(second, (list, tuple)) and second:
                text = str(second[0])
                try:
                    score = float(second[1]) if len(second) > 1 else 1.0
                except Exception:
                    score = 1.0
            elif isinstance(second, str):
                text = second
                try:
                    score = float(item[2]) if len(item) > 2 else 1.0
                except Exception:
                    score = 1.0
            if not text or score < min_score:
                continue
            for match in BIB_RE.findall(text):
                best = out.get(match, -1.0)
                if score > best:
                    out[match] = score
    return out


def assign_face_id(embedding: np.ndarray, known_embeddings: List[np.ndarray], threshold: float) -> Tuple[str, int]:
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


def process_image(
    image_path: Path,
    face_app: object,
    enable_bib_ocr: bool,
) -> Dict[str, object]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return {
            "file": str(image_path),
            "status": "error",
            "error": "Failed to read image",
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
        # Prefer normalized embedding as recommended by InsightFace usage examples.
        embedding_raw = getattr(detected, "normed_embedding", None)
        if embedding_raw is None:
            embedding_raw = getattr(detected, "embedding", None)
        if embedding_raw is None:
            continue
        embedding = np.asarray(embedding_raw, dtype=np.float32)
        f_hash = encoding_hash(embedding)

        face_entry: Dict[str, object] = {
            "face_hash": f_hash,
            # Temporarily keep embedding for global face-id assignment pass.
            "_embedding": embedding.tolist(),
            "bbox": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
            },
        }

        if enable_bib_ocr:
            torso = crop_torso_region(bgr, (top, right, bottom, left))
            face_bibs: List[Dict[str, object]] = []
            if torso is not None and torso.size > 0:
                local_scores = ocr_bib_candidates(torso)
                for num, conf in local_scores.items():
                    prev = bib_scores.get(num, -1.0)
                    if conf > prev:
                        bib_scores[num] = conf
                    face_bibs.append({"number": num, "confidence": round(conf, 2)})

            face_entry["bib_numbers_near_face"] = sorted(
                face_bibs, key=lambda x: (-float(x["confidence"]), str(x["number"]))
            )

        faces.append(face_entry)

    if enable_bib_ocr:
        global_scores = ocr_bib_candidates(bgr)
        for num, conf in global_scores.items():
            prev = bib_scores.get(num, -1.0)
            if conf > prev:
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


def process_images_producer_consumer(
    images: Sequence[Path],
    enable_bib_ocr: bool,
    workers: int,
    insightface_det_size: int,
) -> List[Dict[str, object]]:
    if not images:
        return []

    worker_count = max(1, workers)
    jobs: "queue.Queue[Tuple[int, Optional[Path]]]" = queue.Queue(maxsize=max(8, worker_count * 3))
    results: "queue.Queue[Tuple[int, Dict[str, object]]]" = queue.Queue()
    print_lock = threading.Lock()

    def producer() -> None:
        for idx, image_path in enumerate(images, start=1):
            jobs.put((idx, image_path))
        for _ in range(worker_count):
            jobs.put((-1, None))

    def consumer() -> None:
        face_app = None
        while True:
            idx, image_path = jobs.get()
            try:
                if image_path is None:
                    return

                with print_lock:
                    print(f"[{idx}/{len(images)}] {image_path.name}")

                try:
                    if face_app is None:
                        face_app = create_insightface_app(insightface_det_size)
                    result = process_image(
                        image_path=image_path,
                        face_app=face_app,
                        enable_bib_ocr=enable_bib_ocr,
                    )
                except Exception as exc:  # pragma: no cover - defensive runtime guard
                    result = {
                        "file": str(image_path),
                        "status": "error",
                        "error": f"Unhandled processing error: {exc}",
                        "face_count": 0,
                        "faces": [],
                        "bib_numbers": [],
                    }
                results.put((idx, result))
            finally:
                jobs.task_done()

    prod_thread = threading.Thread(target=producer, name="producer", daemon=True)
    consumers = [
        threading.Thread(target=consumer, name=f"consumer-{i+1}", daemon=True)
        for i in range(worker_count)
    ]

    prod_thread.start()
    for t in consumers:
        t.start()

    ordered: Dict[int, Dict[str, object]] = {}
    for _ in range(len(images)):
        idx, result = results.get()
        ordered[idx] = result

    jobs.join()
    prod_thread.join()
    for t in consumers:
        t.join()

    return [ordered[i] for i in range(1, len(images) + 1)]


def assign_face_ids_in_place(
    image_results: Sequence[Dict[str, object]],
    face_threshold: float,
) -> None:
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
            f_id, _ = assign_face_id(embedding, known_embeddings, threshold=face_threshold)
            face["face_id"] = f_id


def create_insightface_app(det_size: int) -> object:
    if FaceAnalysis is None:
        raise RuntimeError(
            "InsightFace is not installed. Install with: pip install insightface onnxruntime"
        )

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if sys.platform == "darwin" else None
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    if providers is None:
        # GPU first (ctx_id=0), fallback to CPU (ctx_id=-1).
        try:
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    else:
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


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


def run_pipeline(args: argparse.Namespace) -> int:
    if FaceAnalysis is None:
        print("InsightFace backend requested but package is not installed in this environment.")
        print("Install dependencies: pip install insightface onnxruntime")
        return 2

    if args.enable_bib_ocr and RapidOCR is None:
        print("Bib OCR is enabled but rapidocr-onnxruntime is not installed.")
        print("Install dependencies: pip install rapidocr-onnxruntime")
        return 2

    download_dir = Path(args.download_dir)
    if args.album_url:
        print(f"Downloading album -> {download_dir}")
        status = download_shared_album(
            album_url=args.album_url,
            output_dir=download_dir,
            delay_seconds=args.download_delay,
            limit=args.download_limit,
        )
        if status not in (0, 1):
            return status

    if not download_dir.exists():
        print(f"Download directory does not exist: {download_dir}")
        return 2

    images = iter_images(download_dir)
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"Processing {len(images)} image(s) from {download_dir.resolve()}")
    if images:
        print(f"Using producer/consumer workers: {args.workers}")

    image_results = process_images_producer_consumer(
        images=images,
        enable_bib_ocr=args.enable_bib_ocr,
        workers=args.workers,
        insightface_det_size=args.insightface_det_size,
    )
    assign_face_ids_in_place(
        image_results=image_results,
        face_threshold=args.face_distance_threshold,
    )

    face_index = build_face_index(image_results)

    output = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "album_url": args.album_url,
        "download_dir": str(download_dir.resolve()),
        "settings": {
            "face_backend": "insightface",
            "face_distance_threshold": args.face_distance_threshold,
            "insightface_det_size": args.insightface_det_size,
            "bib_ocr_enabled": bool(args.enable_bib_ocr),
            "workers": args.workers,
            "ocr_backend": "rapidocr_onnxruntime" if args.enable_bib_ocr else None,
        },
        "summary": {
            "total_images": len(image_results),
            "images_ok": sum(1 for item in image_results if item.get("status") == "ok"),
            "images_error": sum(1 for item in image_results if item.get("status") != "ok"),
            "total_faces_detected": sum(int(item.get("face_count", 0)) for item in image_results if item.get("status") == "ok"),
            "unique_face_ids": len(face_index),
        },
        "face_index": face_index,
        "images": image_results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote JSON report: {out_path.resolve()}")

    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Google Photos album and build face-hash + bib-number JSON report."
    )
    parser.add_argument(
        "--album-url",
        help="Public Google Photos shared album URL. If omitted, process existing download dir.",
        default=None,
    )
    parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Directory where album images are stored (default: downloads)",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.0,
        help="Delay between downloads in seconds when downloading album",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Optional max number of photos to download",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max number of images to process",
    )
    parser.add_argument(
        "--output-json",
        default="reports/album_face_bib_index.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--face-distance-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for assigning same face_id across images",
    )
    parser.add_argument(
        "--insightface-det-size",
        type=int,
        default=640,
        help="InsightFace detector input size (det_size x det_size)",
    )
    parser.add_argument(
        "--disable-bib-ocr",
        action="store_true",
        help="Disable bib number OCR",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Producer/consumer worker count for image processing",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.enable_bib_ocr = not args.disable_bib_ocr
    return args


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())

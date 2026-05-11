#!/usr/bin/env python3
"""Benchmark face scanning duration across different InsightFace configs."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from face_scanner import (
    DEFAULT_INSIGHTFACE_MODEL_NAME,
    DEFAULT_INSIGHTFACE_PROVIDER,
    create_insightface_app,
    is_cv2_available,
    is_insightface_available,
    is_rapidocr_available,
    process_image,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


@dataclass
class ScanConfig:
    label: str
    provider: str
    model_name: str
    det_size: int
    enable_bib_ocr: bool


@dataclass
class BenchmarkResult:
    label: str
    provider: str
    model_name: str
    det_size: int
    enable_bib_ocr: bool
    images: int
    repeats: int
    app_init_seconds: Optional[float]
    total_seconds_avg: Optional[float]
    total_seconds_min: Optional[float]
    total_seconds_max: Optional[float]
    per_image_ms_avg: Optional[float]
    faces_avg: Optional[float]
    errors_avg: Optional[float]
    status: str
    error: str


def parse_bool_text(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_config_item(text: str) -> ScanConfig:
    # Format: label|provider|model|det_size|bib_ocr
    parts = [p.strip() for p in text.split("|")]
    if len(parts) != 5:
        raise ValueError(
            "Config must have 5 pipe-separated fields: label|provider|model|det_size|bib_ocr"
        )
    label, provider, model_name, det_size_raw, bib_ocr_raw = parts
    det_size = int(det_size_raw)
    bib_ocr = parse_bool_text(bib_ocr_raw)
    return ScanConfig(
        label=label,
        provider=provider,
        model_name=model_name,
        det_size=det_size,
        enable_bib_ocr=bib_ocr,
    )


def discover_images(paths: Iterable[str], recursive: bool) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input path not found: {p}")
        if p.is_file():
            if p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
            continue
        walker = p.rglob("*") if recursive else p.glob("*")
        for child in walker:
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                out.append(child.resolve())
    unique = sorted(set(out))
    if not unique:
        raise ValueError("No image files found from input paths.")
    return unique


def default_configs() -> List[ScanConfig]:
    return [
        ScanConfig(
            label="auto-l-640-noocr",
            provider=DEFAULT_INSIGHTFACE_PROVIDER,
            model_name=DEFAULT_INSIGHTFACE_MODEL_NAME,
            det_size=640,
            enable_bib_ocr=False,
        ),
        ScanConfig(
            label="cpu-s-320-noocr",
            provider="cpu",
            model_name="buffalo_s",
            det_size=320,
            enable_bib_ocr=False,
        ),
        ScanConfig(
            label="cpu-l-320-noocr",
            provider="cpu",
            model_name=DEFAULT_INSIGHTFACE_MODEL_NAME,
            det_size=320,
            enable_bib_ocr=False,
        ),
    ]


def benchmark_one(
    cfg: ScanConfig,
    images: List[Path],
    repeats: int,
    warmup: int,
) -> BenchmarkResult:
    try:
        t0 = time.perf_counter()
        app = create_insightface_app(
            det_size=cfg.det_size,
            model_name=cfg.model_name,
            provider_mode=cfg.provider,
        )
        app_init_seconds = time.perf_counter() - t0
    except Exception as exc:
        return BenchmarkResult(
            label=cfg.label,
            provider=cfg.provider,
            model_name=cfg.model_name,
            det_size=cfg.det_size,
            enable_bib_ocr=cfg.enable_bib_ocr,
            images=len(images),
            repeats=repeats,
            app_init_seconds=None,
            total_seconds_avg=None,
            total_seconds_min=None,
            total_seconds_max=None,
            per_image_ms_avg=None,
            faces_avg=None,
            errors_avg=None,
            status="init_error",
            error=str(exc),
        )

    # warmup passes (not measured)
    for _ in range(max(0, warmup)):
        for img in images:
            process_image(img, app, enable_bib_ocr=cfg.enable_bib_ocr)

    totals: List[float] = []
    faces: List[int] = []
    errors: List[int] = []

    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        face_count_total = 0
        error_count = 0
        for img in images:
            result = process_image(img, app, enable_bib_ocr=cfg.enable_bib_ocr)
            if result.get("status") == "ok":
                face_count_total += int(result.get("face_count", 0))
            else:
                error_count += 1
        elapsed = time.perf_counter() - started
        totals.append(elapsed)
        faces.append(face_count_total)
        errors.append(error_count)

    total_avg = statistics.mean(totals)
    return BenchmarkResult(
        label=cfg.label,
        provider=cfg.provider,
        model_name=cfg.model_name,
        det_size=cfg.det_size,
        enable_bib_ocr=cfg.enable_bib_ocr,
        images=len(images),
        repeats=max(1, repeats),
        app_init_seconds=app_init_seconds,
        total_seconds_avg=total_avg,
        total_seconds_min=min(totals),
        total_seconds_max=max(totals),
        per_image_ms_avg=(total_avg / len(images)) * 1000.0 if images else None,
        faces_avg=statistics.mean(faces),
        errors_avg=statistics.mean(errors),
        status="ok",
        error="",
    )


def print_results(results: List[BenchmarkResult]) -> None:
    headers = [
        "label",
        "status",
        "provider",
        "model",
        "det",
        "ocr",
        "imgs",
        "repeats",
        "init_s",
        "avg_s",
        "avg_ms/img",
        "faces_avg",
        "errors_avg",
    ]
    print(" | ".join(headers))
    print("-" * 120)
    for r in results:
        def fnum(v: Optional[float], nd: int = 3) -> str:
            return "" if v is None else f"{v:.{nd}f}"

        row = [
            r.label,
            r.status,
            r.provider,
            r.model_name,
            str(r.det_size),
            str(r.enable_bib_ocr),
            str(r.images),
            str(r.repeats),
            fnum(r.app_init_seconds),
            fnum(r.total_seconds_avg),
            fnum(r.per_image_ms_avg),
            fnum(r.faces_avg, 2),
            fnum(r.errors_avg, 2),
        ]
        print(" | ".join(row))
        if r.error:
            print(f"  error: {r.error}")


def write_csv(path: Path, results: List[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "status",
                "provider",
                "model_name",
                "det_size",
                "enable_bib_ocr",
                "images",
                "repeats",
                "app_init_seconds",
                "total_seconds_avg",
                "total_seconds_min",
                "total_seconds_max",
                "per_image_ms_avg",
                "faces_avg",
                "errors_avg",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "label": r.label,
                    "status": r.status,
                    "provider": r.provider,
                    "model_name": r.model_name,
                    "det_size": r.det_size,
                    "enable_bib_ocr": r.enable_bib_ocr,
                    "images": r.images,
                    "repeats": r.repeats,
                    "app_init_seconds": r.app_init_seconds,
                    "total_seconds_avg": r.total_seconds_avg,
                    "total_seconds_min": r.total_seconds_min,
                    "total_seconds_max": r.total_seconds_max,
                    "per_image_ms_avg": r.per_image_ms_avg,
                    "faces_avg": r.faces_avg,
                    "errors_avg": r.errors_avg,
                    "error": r.error,
                }
            )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark face scanning duration across multiple InsightFace configs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files or directories to benchmark (directories are scanned for images).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories for images.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured repeats per config (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup repeats per config before measurement (default: 1).",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Config entry: label|provider|model|det_size|bib_ocr . "
            "Can be passed multiple times. If omitted, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional output CSV path for benchmark results.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not is_cv2_available():
        print("OpenCV unavailable. Install with: pip install opencv-python")
        return 2
    if not is_insightface_available():
        print("InsightFace unavailable. Install with: pip install insightface onnxruntime")
        return 2

    images = discover_images(args.inputs, recursive=args.recursive)
    configs = [parse_config_item(c) for c in args.config] if args.config else default_configs()

    needs_ocr = any(c.enable_bib_ocr for c in configs)
    if needs_ocr and not is_rapidocr_available():
        print("At least one config enables OCR but rapidocr-onnxruntime is unavailable.")
        print("Install with: pip install rapidocr-onnxruntime")
        return 2

    print(f"Benchmarking {len(images)} image(s) across {len(configs)} config(s)")
    results = [benchmark_one(c, images, repeats=args.repeat, warmup=args.warmup) for c in configs]
    print_results(results)

    if args.csv_out:
        out = Path(args.csv_out).expanduser().resolve()
        write_csv(out, results)
        print(f"Wrote CSV: {out}")

    if any(r.status != "ok" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

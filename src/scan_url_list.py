#!/usr/bin/env python3
"""Scan a plain list of image URLs with the face + bib pipeline (parallel).

For photo sources that are not Google Photos shared albums (e.g. RunSignup S3
galleries), URL discovery happens elsewhere (see runsignup_discover.py); this
tool takes the resulting URL list and produces a report in the same schema as
shared_album_downloader, so build_person_index and the website can consume it.

Input JSON: {"album_title": str, "album_url": str, "urls": [str, ...]}
Usage: python src/scan_url_list.py <urls.json> <out_report.json> [--workers N] [--no-bib-ocr] [--provider auto|cpu|coreml]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_scanner import (  # noqa: E402
    assign_face_ids_in_place,
    build_face_index,
    create_insightface_app,
    process_image,
)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_tls = threading.local()
_print_lock = threading.Lock()


def _face_app(provider: str):
    app = getattr(_tls, "app", None)
    if app is None:
        app = create_insightface_app(det_size=640, provider_mode=provider)
        _tls.app = app
    return app


def _scan_one(i: int, url: str, enable_bib_ocr: bool, provider: str) -> dict:
    try:
        data = urlopen(Request(url, headers=UA), timeout=60).read()
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tf.write(data)
        tf.close()
        try:
            res = process_image(Path(tf.name), _face_app(provider), enable_bib_ocr=enable_bib_ocr)
        finally:
            Path(tf.name).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        res = {"file": url, "status": "error", "error": str(exc),
               "face_count": 0, "faces": [], "bib_numbers": []}
    res["media_index"] = i
    res["source_url"] = url
    res["file_name"] = url.rsplit("/", 1)[-1]
    return res


def scan(urls_json: Path, out_path: Path, enable_bib_ocr: bool, provider: str, workers: int) -> int:
    src = json.loads(urls_json.read_text(encoding="utf-8"))
    urls = src.get("urls", [])
    if not urls:
        print("No urls in input.")
        return 2

    results: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_scan_one, i, u, enable_bib_ocr, provider): i for i, u in enumerate(urls, start=1)}
        for fut in as_completed(futs):
            res = fut.result()
            results[res["media_index"]] = res
            done += 1
            if done % 25 == 0 or done == len(urls):
                with _print_lock:
                    ok = sum(1 for r in results.values() if r.get("status") == "ok")
                    faces = sum(int(r.get("face_count", 0)) for r in results.values())
                    print(f"[{done}/{len(urls)}] ok={ok} faces={faces}", flush=True)

    images = [results[i] for i in sorted(results)]
    assign_face_ids_in_place(images, face_threshold=1.0)
    face_index = build_face_index(images)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "album": {"album_title": src.get("album_title"), "album_url": src.get("album_url")},
        "summary": {
            "total_images_seen": len(images),
            "images_ok": sum(1 for m in images if m.get("status") == "ok"),
            "images_error": sum(1 for m in images if m.get("status") != "ok"),
            "total_faces_detected": sum(int(m.get("face_count", 0)) for m in images if m.get("status") == "ok"),
            "unique_face_ids": len(face_index),
        },
        "face_index": face_index,
        "images": images,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s = report["summary"]
    print(f"\nDONE images_ok={s['images_ok']}/{s['total_images_seen']} "
          f"faces={s['total_faces_detected']} unique_face_ids={s['unique_face_ids']} "
          f"errors={s['images_error']}", flush=True)
    print(f"report -> {out_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan a URL list into a face+bib report JSON.")
    ap.add_argument("urls_json")
    ap.add_argument("out_report")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-bib-ocr", action="store_true")
    ap.add_argument("--provider", default="auto", choices=["auto", "cpu", "coreml"])
    args = ap.parse_args()
    return scan(Path(args.urls_json), Path(args.out_report),
                enable_bib_ocr=not args.no_bib_ocr, provider=args.provider, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

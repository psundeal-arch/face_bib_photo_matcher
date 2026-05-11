#!/usr/bin/env python3
"""Download Google Photos albums and generate per-album InsightFace + bib OCR JSON."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app_config import get_section, load_yaml_config
from face_scanner import (
    DEFAULT_INSIGHTFACE_MODEL_NAME,
    DEFAULT_INSIGHTFACE_PROVIDER,
    assign_face_ids_in_place,
    build_face_index,
    configure_ocr_runtime,
    create_insightface_app,
    get_ocr_runtime_config,
    is_cv2_available,
    is_insightface_available,
    is_rapidocr_available,
    process_image,
)

PW_URL_RE = re.compile(r"https://lh3\.googleusercontent\.com/pw/[^\"\s,\]]+")
PW_URL_ESCAPED_RE = re.compile(r"https:\\/\\/lh3\.googleusercontent\.com\\/pw\\/[^\"\s,\]]+")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
DEFAULT_SCAN_WORKERS = 2
MAX_SCAN_WORKERS = 6
DEFAULT_CHECKPOINT_BATCH = 20
DEFAULT_DYNAMIC_SCROLL_WAIT_SECONDS = 0.8
DEFAULT_DYNAMIC_SCROLL_MAX_ROUNDS = 600
DEFAULT_DYNAMIC_STABLE_ROUNDS = 6
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _to_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _to_str_list(value: object, default: List[str]) -> List[str]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or list(default)
    if isinstance(value, list):
        parts = [str(p).strip() for p in value if str(p).strip()]
        return parts or list(default)
    return list(default)


def fetch_text(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def extract_album_title(html: str) -> Optional[str]:
    m = re.search(
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if m:
        return unescape(m.group(1)).strip()

    m = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", unescape(m.group(1))).strip()
        if title:
            return title
    return None


def normalize_pw_url(url: str) -> Optional[str]:
    url = url.replace("\\/", "/")
    url = url.replace("\\u003d", "=").replace("\\u0026", "&")
    base = url.split("=", 1)[0].strip().strip('"').strip("'")
    base = base.rstrip("\\")
    parsed = urlparse(base)
    if parsed.scheme != "https" or parsed.netloc != "lh3.googleusercontent.com":
        return None
    if "/pw/" not in parsed.path:
        return None
    # Some extracted URLs may have accidental trailing slashes/backslashes like
    # ".../pw/<token>//" or ".../pw/<token>\\", which can yield HTTP 400
    # when appending "=d". Trim those safely.
    # which frequently yields HTTP 400 when appending "=d". Trim those safely.
    cleaned_path = parsed.path.rstrip("/\\")
    if "/pw/" not in cleaned_path:
        return None
    return f"https://{parsed.netloc}{cleaned_path}"


def extract_media_urls(html: str) -> List[str]:
    found = PW_URL_RE.findall(html) + PW_URL_ESCAPED_RE.findall(html)
    seen = set()
    urls: List[str] = []
    for raw in found:
        normalized = normalize_pw_url(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def fetch_album_html_and_media_urls(
    album_url: str,
    use_dynamic_fetch: bool = True,
    scroll_wait_seconds: float = DEFAULT_DYNAMIC_SCROLL_WAIT_SECONDS,
    max_scroll_rounds: int = DEFAULT_DYNAMIC_SCROLL_MAX_ROUNDS,
    stable_rounds_to_stop: int = DEFAULT_DYNAMIC_STABLE_ROUNDS,
) -> Tuple[str, List[str], str]:
    """Fetch shared album HTML and collect media URLs.

    Returns:
        html: HTML string used for title extraction
        urls: normalized media URLs
        mode: extraction mode ("dynamic_playwright", "static_html", or "static_fallback")
    """
    html = fetch_text(album_url)
    static_urls = extract_media_urls(html)

    if not use_dynamic_fetch:
        return html, static_urls, "static_html"

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # type: ignore
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        print("Dynamic fetch unavailable (playwright not installed). Using static HTML extraction.")
        return html, static_urls, "static_fallback"

    collected: List[str] = []
    seen = set()

    def add_candidate(raw_url: str) -> None:
        normalized = normalize_pw_url(raw_url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            collected.append(normalized)

    def add_from_text(text: str) -> None:
        for raw in PW_URL_RE.findall(text):
            add_candidate(raw)
        for raw in PW_URL_ESCAPED_RE.findall(text):
            add_candidate(raw)

    add_from_text(html)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            def on_response(resp: object) -> None:
                try:
                    url = str(getattr(resp, "url", ""))
                except Exception:
                    return
                if "lh3.googleusercontent.com/pw/" in url:
                    add_candidate(url)
                    return
                if "batchexecute" in url:
                    try:
                        body_text = str(getattr(resp, "text")())
                    except Exception:
                        return
                    add_from_text(body_text)

            page.on("response", on_response)  # type: ignore[arg-type]
            page.goto(album_url, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(1200)
            add_from_text(page.content())
            page.mouse.move(640, 360)
            page.mouse.click(640, 360)

            stable_rounds = 0
            last_count = len(collected)
            min_rounds = min(40, max_scroll_rounds)
            for round_idx in range(max_scroll_rounds):
                # Google Photos uses an internal virtualized scroller, so wheel/End works
                # more reliably than window.scrollTo(...) on large shared albums.
                page.mouse.wheel(0, 3200)
                if round_idx % 8 == 0:
                    page.keyboard.press("End")
                page.wait_for_timeout(int(max(0.1, scroll_wait_seconds) * 1000))
                add_from_text(page.content())

                now_count = len(collected)
                if now_count == last_count:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                    last_count = now_count

                if round_idx >= min_rounds and stable_rounds >= stable_rounds_to_stop:
                    break

            dynamic_html = page.content()
            context.close()
            browser.close()
            return dynamic_html, collected, "dynamic_playwright"
    except PlaywrightTimeoutError as exc:
        print(f"Dynamic fetch timeout; falling back to static extraction: {exc}")
    except Exception as exc:
        print(f"Dynamic fetch failed; falling back to static extraction: {exc}")

    return html, static_urls, "static_fallback"


def parse_filename(content_disposition: str) -> Optional[str]:
    m = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def safe_name(name: str, fallback: str) -> str:
    cleaned = name.strip().replace("\\", "_").replace("/", "_").replace(":", "_")
    return cleaned if cleaned else fallback


def unique_path(dest_dir: Path, filename: str, used: Dict[str, int]) -> Path:
    stem, ext = os.path.splitext(filename)
    count = used.get(filename, 0)
    if count == 0 and not (dest_dir / filename).exists():
        used[filename] = 1
        return dest_dir / filename

    while True:
        count += 1
        candidate = f"{stem}_{count}{ext}"
        if not (dest_dir / candidate).exists() and candidate not in used:
            used[filename] = count
            used[candidate] = 1
            return dest_dir / candidate


def download_url(
    url: str,
    dest_dir: Path,
    idx: int,
    total: int,
    used: Dict[str, int],
) -> Dict[str, object]:
    direct_url = f"{url}=d"
    req = Request(
        direct_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://photos.google.com/",
        },
    )

    with urlopen(req, timeout=120) as resp:
        cd = resp.headers.get("Content-Disposition", "")
        filename = parse_filename(cd) or f"media_{idx:05d}"
        filename = safe_name(filename, f"media_{idx:05d}")
        out_path = unique_path(dest_dir, filename, used)

        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    print(f"[{idx}/{total}] Saved {out_path.name}")
    return {
        "source_url": url,
        "direct_url": direct_url,
        "file": str(out_path),
        "file_name": out_path.name,
        "suffix": out_path.suffix.lower(),
        "media_index": idx,
    }


def download_album(
    album_url: str,
    output_dir: Path,
    delay_seconds: float,
    limit: Optional[int],
    use_dynamic_fetch: bool = True,
) -> Dict[str, object]:
    """Backwards-compatible single-album download helper."""
    print(f"Fetching album page: {album_url}")
    html, all_urls, media_fetch_mode = fetch_album_html_and_media_urls(
        album_url=album_url,
        use_dynamic_fetch=use_dynamic_fetch,
    )

    album_info = {
        "album_url": album_url,
        "album_title": extract_album_title(html),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "media_fetch_mode": media_fetch_mode,
        "media_urls_found": len(all_urls),
    }

    if not all_urls:
        print("No media URLs found. The album may be private or blocked.")
        return {
            "status_code": 2,
            "album": album_info,
            "downloaded": [],
            "failures": [],
            "selected_count": 0,
        }

    urls = all_urls[:limit] if limit is not None else all_urls

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(urls)} media URLs")
    print(f"Downloading to: {output_dir.resolve()}")

    failures: List[Dict[str, object]] = []
    downloaded: List[Dict[str, object]] = []
    used: Dict[str, int] = {}
    for i, u in enumerate(urls, start=1):
        try:
            item = download_url(u, output_dir, i, len(urls), used)
            downloaded.append(item)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        except (HTTPError, URLError, TimeoutError) as e:
            failures.append({"index": i, "source_url": u, "error": str(e)})
            print(f"[{i}/{len(urls)}] ERROR: {e}")

    status_code = 1 if failures else 0
    if failures:
        print(f"Completed with {len(failures)} failed download(s)")
    else:
        print("Completed successfully")

    return {
        "status_code": status_code,
        "album": album_info,
        "downloaded": downloaded,
        "failures": failures,
        "selected_count": len(urls),
    }


def run(
    album_url: str,
    output_dir: Path,
    delay_seconds: float,
    limit: Optional[int],
    use_dynamic_fetch: bool = True,
) -> int:
    result = download_album(
        album_url=album_url,
        output_dir=output_dir,
        delay_seconds=delay_seconds,
        limit=limit,
        use_dynamic_fetch=use_dynamic_fetch,
    )
    return int(result["status_code"])


def is_supported_image_file(path_or_name: str) -> bool:
    return Path(path_or_name).suffix.lower() in IMAGE_SUFFIXES


def slugify(text: Optional[str]) -> str:
    value = (text or "album").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "album"


def album_short_hash(album_url: str) -> str:
    return hashlib.sha1(album_url.encode("utf-8")).hexdigest()[:8]


def ensure_unique_json_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def find_existing_album_report(output_dir: Path, album_hash: str) -> Optional[Path]:
    candidates = sorted(
        output_dir.glob(f"*--{album_hash}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_existing_album_progress(report_path: Path) -> Tuple[Dict[int, Dict[str, object]], set]:
    images_by_index: Dict[int, Dict[str, object]] = {}
    processed_source_urls: set = set()

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return images_by_index, processed_source_urls

    images = payload.get("images", [])
    if not isinstance(images, list):
        return images_by_index, processed_source_urls

    for item in images:
        if not isinstance(item, dict):
            continue
        source_url = item.get("source_url")
        if isinstance(source_url, str):
            processed_source_urls.add(source_url)
        media_index = item.get("media_index")
        if isinstance(media_index, int):
            images_by_index[media_index] = item

    return images_by_index, processed_source_urls


def write_initial_album_report(state: Dict[str, object]) -> None:
    out_path = Path(str(state["_report_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_by_index: Dict[int, Dict[str, object]] = state["_images_by_index"]  # type: ignore[assignment]
    images = [copy.deepcopy(images_by_index[idx]) for idx in sorted(images_by_index.keys())]
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "album": state["album"],
        "photo_list": {
            "all_urls_found": len(state.get("_all_urls", [])),
            "selected_for_download": len(state.get("_selected_urls", [])),
            "limit": state.get("_selected_limit"),
            "all_urls": state.get("_all_urls", []),
            "selected_urls": state.get("_selected_urls", []),
        },
        "download": state["download"],
        "settings": {
            "refresh_photo_list": True,
            "report_stage": "initial_photo_list_written",
        },
        "summary": {
            "total_images_seen": len(images),
            "images_ok": sum(1 for img in images if img.get("status") == "ok"),
            "images_error": sum(1 for img in images if img.get("status") != "ok"),
            "total_faces_detected": 0,
            "unique_face_ids": 0,
        },
        "cleanup": state["cleanup"],
        "face_index": {},
        "images": images,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_download_file_plan(selected_urls: Sequence[str]) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for idx, source_url in enumerate(selected_urls, start=1):
        plan.append(
            {
                "source_url": source_url,
                "direct_url": f"{source_url}=d",
                "file": None,
                "file_name": None,
                "suffix": None,
                "media_index": idx,
            }
        )
    return plan


def build_album_output(
    state: Dict[str, object],
    face_threshold: float,
    insightface_det_size: int,
    insightface_model_name: str,
    insightface_provider: str,
    enable_bib_ocr: bool,
    workers: int,
    max_pending_downloads: int,
) -> Dict[str, object]:
    images_by_index: Dict[int, Dict[str, object]] = state["_images_by_index"]  # type: ignore[assignment]
    images = [copy.deepcopy(images_by_index[idx]) for idx in sorted(images_by_index.keys())]

    assign_face_ids_in_place(images, face_threshold=face_threshold)
    face_index = build_face_index(images)
    scan_errors = sum(1 for img in images if img.get("status") != "ok")
    ocr_runtime = get_ocr_runtime_config() if enable_bib_ocr else {}

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "album": state["album"],
        "photo_list": {
            "all_urls_found": len(state.get("_all_urls", [])),
            "selected_for_download": len(state.get("_selected_urls", [])),
            "limit": state.get("_selected_limit"),
            "all_urls": state.get("_all_urls", []),
            "selected_urls": state.get("_selected_urls", []),
        },
        "download": state["download"],
        "settings": {
            "face_backend": "insightface",
            "face_distance_threshold": face_threshold,
            "insightface_det_size": insightface_det_size,
            "insightface_model_name": insightface_model_name,
            "insightface_provider": insightface_provider,
            "refresh_photo_list": True,
            "bib_ocr_enabled": bool(enable_bib_ocr),
            "ocr_backend": "rapidocr_onnxruntime" if enable_bib_ocr else None,
            "ocr_runtime": ocr_runtime if enable_bib_ocr else None,
            "workers": workers,
            "max_pending_downloads": max_pending_downloads,
            "checkpoint_batch": state.get("_checkpoint_batch"),
            "cleanup_policy": "delete_after_scan",
        },
        "summary": {
            "total_images_seen": len(images),
            "images_ok": sum(1 for img in images if img.get("status") == "ok"),
            "images_error": scan_errors,
            "total_faces_detected": sum(
                int(img.get("face_count", 0)) for img in images if img.get("status") == "ok"
            ),
            "unique_face_ids": len(face_index),
        },
        "cleanup": state["cleanup"],
        "face_index": face_index,
        "images": images,
    }


def flush_album_report(
    state: Dict[str, object],
    face_threshold: float,
    insightface_det_size: int,
    insightface_model_name: str,
    insightface_provider: str,
    enable_bib_ocr: bool,
    workers: int,
    max_pending_downloads: int,
    final: bool,
) -> Dict[str, object]:
    out_path = Path(str(state["_report_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_album_output(
        state=state,
        face_threshold=face_threshold,
        insightface_det_size=insightface_det_size,
        insightface_model_name=insightface_model_name,
        insightface_provider=insightface_provider,
        enable_bib_ocr=enable_bib_ocr,
        workers=workers,
        max_pending_downloads=max_pending_downloads,
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if final:
        print(f"Wrote album JSON: {out_path.resolve()}")
    else:
        print(f"Checkpoint JSON updated: {out_path.resolve()}")
    return payload


def produce_download_tasks(
    album_urls: Sequence[str],
    output_root: Path,
    output_json_dir: Path,
    limit: Optional[int],
    delay_seconds: float,
    use_dynamic_fetch: bool,
    task_queue: "queue.Queue[Optional[Dict[str, object]]]",
    pending_slots: threading.Semaphore,
    album_states: Dict[int, Dict[str, object]],
    workers: int,
) -> None:
    for album_id, album_url in enumerate(album_urls, start=1):
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        album_hash = album_short_hash(album_url)

        state: Dict[str, object] = {
            "album": {
                "album_id": album_id,
                "album_url": album_url,
                "album_title": None,
                "album_slug": "album",
                "album_hash": album_hash,
                "fetched_at": fetched_at,
                "media_urls_found": 0,
                "selected_for_download": 0,
                "downloaded_count": 0,
                "download_failed_count": 0,
                "skipped_non_image_count": 0,
            },
            "download": {
                "download_dir": None,
                "files": [],
                "failures": [],
                "skipped_non_images": [],
                "skipped_existing": [],
                "status_code": 0,
            },
            "_images_by_index": {},
            "_processed_source_urls": set(),
            "_report_path": "",
            "_results_since_flush": 0,
            "_all_urls": [],
            "_selected_urls": [],
            "_selected_limit": limit,
            "cleanup": {
                "deleted_files": 0,
                "delete_errors": [],
            },
        }
        album_states[album_id] = state

        existing_report = find_existing_album_report(output_json_dir, album_hash)
        if existing_report is not None:
            state["_report_path"] = str(existing_report)
            images_by_index, processed_source_urls = load_existing_album_progress(existing_report)
            state["_images_by_index"] = images_by_index
            state["_processed_source_urls"] = processed_source_urls
            if processed_source_urls:
                print(
                    f"Resume: found {len(processed_source_urls)} already-processed photos in {existing_report.name}"
                )

        try:
            print(f"Fetching album page [{album_id}/{len(album_urls)}]: {album_url}")
            html, all_urls, media_fetch_mode = fetch_album_html_and_media_urls(
                album_url=album_url,
                use_dynamic_fetch=use_dynamic_fetch,
            )
        except Exception as exc:
            state["album"]["error"] = f"Failed to fetch album page: {exc}"
            state["download"]["status_code"] = 2
            continue

        title = extract_album_title(html)
        slug = slugify(title)
        state["album"]["album_title"] = title
        state["album"]["album_slug"] = slug
        state["album"]["media_fetch_mode"] = media_fetch_mode
        if existing_report is None:
            state["_report_path"] = str(output_json_dir / f"{slug}--{album_hash}.json")

        if not state.get("_report_path"):
            fallback_slug = str(state["album"].get("album_slug") or "album")
            state["_report_path"] = str(output_json_dir / f"{fallback_slug}--{album_hash}.json")

        current_slug = str(state["album"].get("album_slug") or "album")
        album_download_dir = output_root / f"{current_slug}--{album_hash}"
        state["download"]["download_dir"] = str(album_download_dir.resolve())

        state["album"]["media_urls_found"] = len(all_urls)

        if not all_urls:
            print(f"Album has no media URLs: {album_url}")
            state["download"]["status_code"] = 2
            continue

        urls = all_urls[:limit] if limit is not None else all_urls
        state["album"]["selected_for_download"] = len(urls)
        state["_all_urls"] = list(all_urls)
        state["_selected_urls"] = list(urls)
        state["_selected_limit"] = limit
        state["download"]["files"] = build_download_file_plan(urls)
        try:
            write_initial_album_report(state)
            print(f"Wrote initial report with full photo list: {Path(str(state['_report_path'])).resolve()}")
        except Exception as exc:
            print(f"WARN: failed to write initial report for album {album_id}: {exc}")

        album_download_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading album {album_id} to: {album_download_dir.resolve()}")

        used: Dict[str, int] = {}
        failures: List[Dict[str, object]] = state["download"]["failures"]  # type: ignore[assignment]
        files: List[Dict[str, object]] = state["download"]["files"]  # type: ignore[assignment]
        skipped_existing: List[Dict[str, object]] = state["download"]["skipped_existing"]  # type: ignore[assignment]
        processed_source_urls: set = state["_processed_source_urls"]  # type: ignore[assignment]

        for idx, media_url in enumerate(urls, start=1):
            if idx > len(urls):
                break
            # Always bind from current list so live-repair URL swaps take effect
            # for subsequent indices within this run.
            media_url = urls[idx - 1]
            if media_url in processed_source_urls:
                skipped_existing.append(
                    {
                        "index": idx,
                        "source_url": media_url,
                        "reason": "already_processed_in_report",
                    }
                )
                continue

            pending_slots.acquire()
            try:
                item = download_url(media_url, album_download_dir, idx, len(urls), used)
                if 0 <= (idx - 1) < len(files):
                    files[idx - 1] = item
                else:
                    files.append(item)
                state["album"]["downloaded_count"] = int(state["album"]["downloaded_count"]) + 1

                if is_supported_image_file(str(item["file_name"])):
                    task_queue.put(
                        {
                            "album_id": album_id,
                            "media_index": idx,
                            "source_url": media_url,
                            "local_file_path": item["file"],
                            "file_name": item["file_name"],
                        }
                    )
                else:
                    state["album"]["skipped_non_image_count"] = (
                        int(state["album"]["skipped_non_image_count"]) + 1
                    )
                    skipped = state["download"]["skipped_non_images"]  # type: ignore[index]
                    skipped.append(
                        {
                            "index": idx,
                            "source_url": media_url,
                            "file": item["file"],
                            "file_name": item["file_name"],
                            "reason": "unsupported_media_type",
                        }
                    )
                    try:
                        Path(str(item["file"])).unlink()
                        state["cleanup"]["deleted_files"] = int(state["cleanup"]["deleted_files"]) + 1
                    except FileNotFoundError:
                        state["cleanup"]["deleted_files"] = int(state["cleanup"]["deleted_files"]) + 1
                    except Exception as exc:
                        state["cleanup"]["delete_errors"].append(  # type: ignore[index]
                            {
                                "media_index": idx,
                                "file": item["file"],
                                "error": str(exc),
                            }
                        )
                    finally:
                        pending_slots.release()
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            except (HTTPError, URLError, TimeoutError) as exc:
                failures.append({"index": idx, "source_url": media_url, "error": str(exc)})
                state["album"]["download_failed_count"] = int(state["album"]["download_failed_count"]) + 1
                print(f"[{idx}/{len(urls)}] ERROR: {exc}")
                pending_slots.release()

        state["download"]["status_code"] = 1 if failures else 0

    for _ in range(workers):
        task_queue.put(None)


def consume_scan_tasks(
    task_queue: "queue.Queue[Optional[Dict[str, object]]]",
    pending_slots: threading.Semaphore,
    result_queue: "queue.Queue[Dict[str, object]]",
    enable_bib_ocr: bool,
    insightface_det_size: int,
    insightface_model_name: str,
    insightface_provider: str,
) -> None:
    face_app = None
    face_app_error: Optional[str] = None

    while True:
        task = task_queue.get()
        has_pending_slot = task is not None
        try:
            if task is None:
                return

            album_id = int(task["album_id"])
            media_index = int(task["media_index"])
            image_path = Path(str(task["local_file_path"]))
            scan_started = time.perf_counter()

            if face_app is None and face_app_error is None:
                try:
                    face_app = create_insightface_app(
                        det_size=insightface_det_size,
                        model_name=insightface_model_name,
                        provider_mode=insightface_provider,
                    )
                except Exception as exc:
                    face_app_error = str(exc)

            if face_app_error is not None:
                image_result: Dict[str, object] = {
                    "file": str(image_path),
                    "status": "error",
                    "error": f"InsightFace init failed: {face_app_error}",
                    "face_count": 0,
                    "faces": [],
                    "bib_numbers": [],
                }
            else:
                try:
                    image_result = process_image(
                        image_path=image_path,
                        face_app=face_app,
                        enable_bib_ocr=enable_bib_ocr,
                    )
                except Exception as exc:
                    image_result = {
                        "file": str(image_path),
                        "status": "error",
                        "error": f"Unhandled processing error: {exc}",
                        "face_count": 0,
                        "faces": [],
                        "bib_numbers": [],
                    }

            cleanup_deleted = False
            cleanup_error = None
            try:
                image_path.unlink()
                cleanup_deleted = True
            except FileNotFoundError:
                cleanup_deleted = True
            except Exception as exc:
                cleanup_error = str(exc)

            image_result["media_index"] = media_index
            image_result["source_url"] = str(task["source_url"])
            image_result["file_name"] = str(task["file_name"])
            elapsed = time.perf_counter() - scan_started
            status = str(image_result.get("status", "unknown"))
            face_count = int(image_result.get("face_count", 0))
            bib_count = len(image_result.get("bib_numbers", [])) if isinstance(
                image_result.get("bib_numbers"), list
            ) else 0
            if status == "ok":
                print(
                    f"[scan][album {album_id}][{media_index}] DONE {image_result['file_name']} "
                    f"status={status} faces={face_count} bibs={bib_count} time={elapsed:.2f}s"
                )
            else:
                print(
                    f"[scan][album {album_id}][{media_index}] DONE {image_result['file_name']} "
                    f"status={status} time={elapsed:.2f}s error={image_result.get('error', '')}"
                )

            result_queue.put(
                {
                    "album_id": album_id,
                    "media_index": media_index,
                    "image_result": image_result,
                    "cleanup_deleted": cleanup_deleted,
                    "cleanup_error": cleanup_error,
                }
            )
        finally:
            if has_pending_slot:
                pending_slots.release()
            task_queue.task_done()


def run_pipeline(args: argparse.Namespace) -> int:
    configure_ocr_runtime(
        min_score=args.ocr_min_score,
        variant_names=args.ocr_variants,
        enable_global_pass=args.ocr_enable_global_pass,
    )

    if not is_cv2_available():
        print("OpenCV is required. Install with: pip install opencv-python")
        return 2

    if not is_insightface_available():
        print("InsightFace is required. Install with: pip install insightface onnxruntime")
        return 2

    if args.enable_bib_ocr:
        if not is_rapidocr_available():
            print("Bib OCR is enabled but RapidOCR ONNX Runtime is not installed.")
            print("Install with: pip install rapidocr-onnxruntime")
            return 2

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    output_json_dir = Path(args.output_json_dir)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    max_pending = int(args.max_pending_downloads)
    pending_slots = threading.Semaphore(max_pending)

    task_queue: "queue.Queue[Optional[Dict[str, object]]]" = queue.Queue(maxsize=max(16, args.workers * 6))
    result_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
    album_states: Dict[int, Dict[str, object]] = {}

    producer = threading.Thread(
        target=produce_download_tasks,
        kwargs={
            "album_urls": args.album_urls,
            "output_root": output_root,
            "output_json_dir": output_json_dir,
            "limit": args.limit,
            "delay_seconds": args.delay,
            "use_dynamic_fetch": args.use_dynamic_fetch,
            "task_queue": task_queue,
            "pending_slots": pending_slots,
            "album_states": album_states,
            "workers": args.workers,
        },
        name="download-producer",
        daemon=True,
    )

    consumers = [
        threading.Thread(
            target=consume_scan_tasks,
            kwargs={
                "task_queue": task_queue,
                "pending_slots": pending_slots,
                "result_queue": result_queue,
                "enable_bib_ocr": args.enable_bib_ocr,
                "insightface_det_size": args.insightface_det_size,
                "insightface_model_name": args.insightface_model_name,
                "insightface_provider": args.insightface_provider,
            },
            name=f"scan-consumer-{i + 1}",
            daemon=True,
        )
        for i in range(max(1, args.workers))
    ]

    producer.start()
    for worker in consumers:
        worker.start()

    for state in album_states.values():
        state["_checkpoint_batch"] = int(args.checkpoint_batch)

    while True:
        try:
            item = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not producer.is_alive() and task_queue.unfinished_tasks == 0 and result_queue.empty():
                break
            continue

        album_id = int(item["album_id"])
        media_index = int(item["media_index"])
        state = album_states[album_id]

        images_by_index: Dict[int, Dict[str, object]] = state["_images_by_index"]  # type: ignore[assignment]
        images_by_index[media_index] = item["image_result"]  # type: ignore[index]

        cleanup = state["cleanup"]
        if item.get("cleanup_deleted"):
            cleanup["deleted_files"] = int(cleanup["deleted_files"]) + 1
        if item.get("cleanup_error"):
            cleanup["delete_errors"].append(  # type: ignore[index]
                {
                    "media_index": media_index,
                    "error": item["cleanup_error"],
                    "file": item["image_result"].get("file"),
                }
            )

        state["_results_since_flush"] = int(state["_results_since_flush"]) + 1
        if int(state["_results_since_flush"]) >= int(args.checkpoint_batch):
            flush_album_report(
                state=state,
                face_threshold=args.face_distance_threshold,
                insightface_det_size=args.insightface_det_size,
                insightface_model_name=args.insightface_model_name,
                insightface_provider=args.insightface_provider,
                enable_bib_ocr=args.enable_bib_ocr,
                workers=args.workers,
                max_pending_downloads=max_pending,
                final=False,
            )
            state["_results_since_flush"] = 0

    producer.join()
    task_queue.join()
    for worker in consumers:
        worker.join()

    overall_failures = False

    for album_id in sorted(album_states.keys()):
        state = album_states[album_id]
        payload = flush_album_report(
            state=state,
            face_threshold=args.face_distance_threshold,
            insightface_det_size=args.insightface_det_size,
            insightface_model_name=args.insightface_model_name,
            insightface_provider=args.insightface_provider,
            enable_bib_ocr=args.enable_bib_ocr,
            workers=args.workers,
            max_pending_downloads=max_pending,
            final=True,
        )

        download_meta = payload.get("download", {})
        summary_meta = payload.get("summary", {})
        if int(download_meta.get("status_code", 0)) != 0 or int(summary_meta.get("images_error", 0)) > 0:
            overall_failures = True

    return 1 if overall_failures else 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args(argv_list)

    config_path = Path(str(bootstrap_args.config))
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    else:
        config_path = config_path.resolve()
    config = load_yaml_config(config_path)
    downloader_cfg = get_section(config, "downloader")

    default_output = str(downloader_cfg.get("output", "downloads"))
    default_delay = _to_float(downloader_cfg.get("delay"), 0.0)
    default_limit_raw = downloader_cfg.get("limit")
    default_limit = _to_int(default_limit_raw, 0) if default_limit_raw is not None else None
    default_use_dynamic_fetch = _to_bool(downloader_cfg.get("use_dynamic_fetch"), True)
    default_output_json_dir = str(downloader_cfg.get("output_json_dir", "reports"))
    default_face_distance_threshold = _to_float(downloader_cfg.get("face_distance_threshold"), 1.0)
    default_insightface_det_size = _to_int(downloader_cfg.get("insightface_det_size"), 320)
    default_insightface_model_name = str(
        downloader_cfg.get("insightface_model_name", DEFAULT_INSIGHTFACE_MODEL_NAME)
    )
    default_insightface_provider = str(
        downloader_cfg.get("insightface_provider", DEFAULT_INSIGHTFACE_PROVIDER)
    )
    if default_insightface_provider not in {"auto", "cpu", "coreml"}:
        default_insightface_provider = DEFAULT_INSIGHTFACE_PROVIDER
    default_enable_bib_ocr = _to_bool(downloader_cfg.get("enable_bib_ocr"), True)
    default_workers = _to_int(
        downloader_cfg.get("workers"),
        min(DEFAULT_SCAN_WORKERS, max(1, os.cpu_count() or 1)),
    )
    default_max_pending_raw = downloader_cfg.get("max_pending_downloads")
    default_max_pending_downloads = (
        _to_int(default_max_pending_raw, 0) if default_max_pending_raw is not None else None
    )
    default_checkpoint_batch = _to_int(
        downloader_cfg.get("checkpoint_batch"),
        DEFAULT_CHECKPOINT_BATCH,
    )
    default_ocr_min_score = _to_float(downloader_cfg.get("ocr_min_score"), 0.45)
    default_ocr_variants = _to_str_list(downloader_cfg.get("ocr_variants"), ["otsu"])
    default_ocr_enable_global_pass = _to_bool(downloader_cfg.get("ocr_enable_global_pass"), True)

    parser = argparse.ArgumentParser(
        description=(
            "Download one or more public Google Photos albums and generate "
            "per-album InsightFace + bib OCR JSON outputs."
        )
    )
    parser.add_argument(
        "--config",
        default=str(config_path),
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("album_urls", nargs="+", help="Google Photos shared album URLs")
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help="Root directory for temporary downloaded media",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=default_delay,
        help="Delay between downloads in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="Optional max number of files to download per album",
    )
    parser.add_argument(
        "--disable-dynamic-fetch",
        action="store_false",
        dest="use_dynamic_fetch",
        default=default_use_dynamic_fetch,
        help=(
            "Disable dynamic lazy-load scrolling via playwright and only parse initial HTML "
            "(this may miss photos in large albums)"
        ),
    )
    parser.add_argument(
        "--output-json-dir",
        "--output-json",
        dest="output_json_dir",
        default=default_output_json_dir,
        help="Directory for per-album JSON outputs",
    )
    parser.add_argument(
        "--face-distance-threshold",
        type=float,
        default=default_face_distance_threshold,
        help="Distance threshold for assigning same face_id across photos",
    )
    parser.add_argument(
        "--insightface-det-size",
        type=int,
        default=default_insightface_det_size,
        help="InsightFace detector size (det_size x det_size)",
    )
    parser.add_argument(
        "--insightface-model-name",
        default=default_insightface_model_name,
        help=(
            "InsightFace model pack name (for example: buffalo_l, buffalo_s). "
            "Smaller packs generally reduce CPU usage at some accuracy cost."
        ),
    )
    parser.add_argument(
        "--insightface-provider",
        choices=["auto", "cpu", "coreml"],
        default=default_insightface_provider,
        help=(
            "ONNX Runtime execution provider mode for InsightFace: "
            "auto (default), cpu (lowest resource), or coreml (macOS)"
        ),
    )
    parser.add_argument(
        "--disable-bib-ocr",
        action="store_true",
        default=not default_enable_bib_ocr,
        help="Disable bib number OCR",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Worker count for image scan consumers",
    )
    parser.add_argument(
        "--max-pending-downloads",
        type=int,
        default=default_max_pending_downloads,
        help=(
            "Cap for downloaded-but-not-processed files; downloader pauses when cap is reached "
            "(default: workers * 4)"
        ),
    )
    parser.add_argument(
        "--checkpoint-batch",
        type=int,
        default=default_checkpoint_batch,
        help="Update album JSON every N completed scans (default: 20)",
    )
    parser.add_argument(
        "--ocr-min-score",
        type=float,
        default=default_ocr_min_score,
        help="Minimum OCR confidence score threshold (default: 0.45)",
    )
    parser.add_argument(
        "--ocr-variants",
        default=",".join(default_ocr_variants),
        help="OCR preprocess variants, comma separated (default: otsu)",
    )
    parser.add_argument(
        "--ocr-enable-global-pass",
        action="store_true",
        default=default_ocr_enable_global_pass,
        help="Run extra full-image OCR pass in addition to torso OCR (default: enabled)",
    )
    parser.add_argument(
        "--ocr-disable-global-pass",
        action="store_false",
        dest="ocr_enable_global_pass",
        help="Disable extra full-image OCR pass",
    )

    args = parser.parse_args(argv_list)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.workers > MAX_SCAN_WORKERS:
        parser.error(f"--workers must be <= {MAX_SCAN_WORKERS} to avoid machine overload")
    if args.max_pending_downloads is None:
        args.max_pending_downloads = max(1, args.workers * 4)
    if args.max_pending_downloads < 1:
        parser.error("--max-pending-downloads must be >= 1")
    if args.checkpoint_batch < 1:
        parser.error("--checkpoint-batch must be >= 1")
    if args.ocr_min_score < 0.0:
        parser.error("--ocr-min-score must be >= 0")
    args.enable_bib_ocr = not args.disable_bib_ocr
    args.ocr_variants = _to_str_list(args.ocr_variants, ["otsu"])
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

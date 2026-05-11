# face_bib_photo_matcher

Utility scripts for working with Google Photos shared albums and face indexing.

## Config File

Path: `config.yaml` (repo root).

Both downloader and website read this YAML by default.

- Downloader override: `--config /path/to/config.yaml`
- Website override: `APP_CONFIG_PATH=/path/to/config.yaml`
- Environment variables and CLI flags still override config values.

## Tests

Run unit tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

Run integration tests (uses `tests/fixtures/BE7I4001.JPG` and real InsightFace runtime):

```bash
RUN_INTEGRATION_TESTS=1 python3 -m unittest discover -s tests -p 'test_*.py' -q
```

## Website (Bib + Face Match)

Path: `src/website`

This web app lets a user:
- take a photo from camera or upload one
- input a bib number
- tune embedding threshold (L2 distance, lower = stricter)
- run InsightFace to get face hash + embedding from the uploaded photo
- search matching entries from JSON files under `reports/*.json` using bib match and face similarity (exact face-hash match and embedding distance)

### Run

```bash
pip install flask opencv-python insightface onnxruntime numpy pyyaml
python3 src/website/app.py --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

### Run with Uvicorn (ASGI process manager)

```bash
pip install uvicorn asgiref
uvicorn --app-dir src/website asgi:app --host 127.0.0.1 --port 8000 --workers 1
```

### Optional environment variables

- `APP_CONFIG_PATH`: YAML config path for website defaults (default: `<repo>/config.yaml`)
- `WEBSITE_REPORTS_DIR`: override report directory (default: `<repo>/reports`)
- `WEBSITE_RACE_REPORTS_JSON`: race->reports mapping JSON for dropdown, for example:
  - `{"2026 Boston Marathon":"reports","2026 Tokyo Marathon":"reports_tokyo"}`
- `WEBSITE_DEFAULT_RACE`: default selected race label in the dropdown
- `WEBSITE_INSIGHTFACE_DET_SIZE`: detector size (default: `640`)
- `WEBSITE_MAX_RESULTS`: max matches returned (default: `30`)
- `WEBSITE_DEFAULT_EMBEDDING_THRESHOLD`: default embedding L2 threshold if UI value is empty (default: `1.0`)

## Shared Album Face + Bib Pipeline

Script: `src/shared_album_downloader.py`

Downloads one or more public Google Photos shared albums, scans images with InsightFace and OCR, and writes one JSON report per album.

### Install dependencies

```bash
pip install opencv-python insightface onnxruntime rapidocr-onnxruntime pillow pillow-heif numpy
```

Notes:
- HEIC/HEIF scanning is supported when `pillow-heif` is installed.
- If HEIC decode still fails in your environment, convert files to JPG/PNG as a fallback.

For large Google Photos albums, install Playwright so dynamic lazy-loaded photos can be discovered:

```bash
pip install playwright
python3 -m playwright install chromium
```

### Command (multiple albums)

```bash
python3 src/shared_album_downloader.py \
  "https://photos.google.com/share/ALBUM_URL_1" \
  "https://photos.google.com/share/ALBUM_URL_2"
```

### Useful options

- `--config <path>`: YAML config path (default: `<repo>/config.yaml`)
- `-o, --output <dir>`: root folder for temporary downloaded media (default: `downloads`)
- `--output-json-dir <dir>`: output folder for per-album JSON files (default: `reports`)
- `--limit <n>`: max files to download per album
- `--workers <n>`: number of scan consumers (default: `2`, max: `6`)
- `--max-pending-downloads <n>`: cap of downloaded-but-not-yet-processed files; downloader pauses when cap is reached (default: `workers * 4`)
- `--checkpoint-batch <n>`: update album JSON after every N completed scans (default: `20`)
- `--delay <sec>`: delay between downloads
- `--insightface-det-size <n>`: detector input size (default comes from `config.yaml`; fallback is `320`)
- `--insightface-model-name <name>`: InsightFace model pack (default: `buffalo_l`; try `buffalo_s` for lower CPU)
- `--insightface-provider <auto|cpu|coreml>`: execution provider mode (default: `auto`; use `cpu` for lowest resource)
- `--face-distance-threshold <f>`: face-id grouping threshold (default: `1.0`)
- `--disable-bib-ocr`: disable bib OCR
- `--ocr-min-score <f>`: minimum OCR confidence score (default: `0.45`)
- `--ocr-variants <csv>`: OCR preprocess variants (default: `otsu`)
- `--ocr-disable-global-pass`: skip extra full-image OCR pass for speed
- `--disable-dynamic-fetch`: disable Playwright lazy-load scrolling and only parse initial HTML (can miss photos in large albums)

Low-CPU example:

```bash
python3 src/shared_album_downloader.py "<ALBUM_URL>" \
  --workers 1 \
  --insightface-provider cpu \
  --insightface-model-name buffalo_s \
  --insightface-det-size 320
```

### Behavior

- Producer/consumer pipeline:
  - producer downloads media and enqueues scan tasks
  - consumers scan images in parallel (InsightFace + OCR)
- URL discovery:
  - if Playwright is installed, the script auto-scrolls shared album pages to collect lazy-loaded media URLs
  - if Playwright is not installed, it falls back to static HTML extraction
- Cleanup:
  - each image is deleted right after scan completes
- Media handling:
  - non-image media (for example videos) are skipped from scanning and deleted
- Outputs:
  - one JSON file per album, named like `{album_slug}--{short_hash}.json`
- Resume behavior:
  - if an album JSON already exists, photos already listed in that JSON are skipped for both download and scanning
- Photo discovery behavior:
  - every run performs fresh album URL discovery before download/scan
- Report initialization:
  - after URL discovery, the report JSON is written immediately with `photo_list.all_urls` and `photo_list.selected_urls` before scanning starts

### Per-album JSON content

Each JSON contains:

- `album`: album metadata and counts
- `download`: downloaded files, failures, skipped non-images, status
- `settings`: run settings
- `summary`: image/face totals
- `cleanup`: deleted file counts and delete errors
- `face_index`: grouped face hashes
- `images`: per-image face and bib OCR results
  - each face now includes `embedding` (InsightFace vector) so cross-photo person matching can run from report JSON alone

## Scan Config Benchmark

Script: `src/benchmark_face_scanner.py`

Compare scan duration across different InsightFace config combinations on the same input image set.

### Quick example

```bash
python3 src/benchmark_face_scanner.py tests/fixtures/BE7I4001.JPG \
  --repeat 3 \
  --warmup 1 \
  --config "cpu-l-640|cpu|buffalo_l|640|false" \
  --config "cpu-l-320|cpu|buffalo_l|320|false" \
  --config "coreml-l-320|coreml|buffalo_l|320|false" \
  --csv-out reports/scan_benchmark.csv
```

Config format:

- `label|provider|model|det_size|bib_ocr`
- example: `cpu-s-320-noocr|cpu|buffalo_s|320|false`

If `--config` is not provided, the script runs built-in default comparison configs.

OCR tuning env vars (used by scanner + benchmark):

- `OCR_MIN_SCORE` (default: `0.45`)
- `OCR_VARIANTS` (default: `otsu`)
- `OCR_ENABLE_GLOBAL_PASS` (default: `1`)

## InsightFace Hash Registry Usage

Script: `src/insightface_hashcode.py`

This script takes a person's `name` and a `photo_path`, runs InsightFace, and upserts the result into a JSON file that can store many people.

### Install dependencies

```bash
pip install opencv-python insightface onnxruntime numpy
```

### Command

```bash
python3 src/insightface_hashcode.py "Alice" /path/to/alice.jpg
```

### Useful options

- `--db <path>`: JSON registry path (default: `people_face_hashes.json`)
- `--all-faces`: store hashcodes for all faces detected in the photo (default stores largest face only)
- `--det-size <int>`: InsightFace detector size (default: `640`)

Examples:

```bash
python3 src/insightface_hashcode.py "Bob" /data/bob.jpg --db data/people.json
python3 src/insightface_hashcode.py "Carol" /data/team_photo.jpg --all-faces
```

### JSON format

The registry file has this structure:

```json
{
  "version": 1,
  "people": [
    {
      "name": "Alice",
      "samples": [
        {
          "photo_path": "/abs/path/to/alice.jpg",
          "added_at": "2026-04-23T16:00:00+00:00",
          "face_count": 1,
          "hashcodes": ["..."],
          "faces": [
            {
              "face_index": 1,
              "hashcode": "...",
              "embedding": [0.123456, -0.234567, 0.345678],
              "det_score": 0.99,
              "area": 16800.0
            }
          ]
        }
      ]
    }
  ]
}
```

Behavior:

- If `name` does not exist, a new person entry is created.
- If the same absolute `photo_path` already exists under that person, the sample is updated.
- Otherwise, a new sample is appended under that person.

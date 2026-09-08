"""Source-aware preview/download URL construction for report photos.

Google Photos serves resized/cropped variants via `=w...`/`=d` URL suffixes.
RunSignup serves S3 objects at fixed size prefixes (thumbs_v3 / large_v3), so
those suffixes must not be appended. Any other host is used verbatim.
"""

from __future__ import annotations

GOOGLE_HOST = "lh3.googleusercontent.com"
RSU_HOST = "rsu-photos"


def is_google_photo(url: str) -> bool:
    return GOOGLE_HOST in url


def is_runsignup_photo(url: str) -> bool:
    return RSU_HOST in url


def photo_preview_url(source_url: str) -> str:
    """A smaller image suitable for grid thumbnails."""
    if is_runsignup_photo(source_url):
        return source_url.replace("/large_v3/", "/thumbs_v3/")
    if is_google_photo(source_url):
        return f"{source_url}=w800-h560-no"
    return source_url


def photo_download_url(source_url: str) -> str:
    """The best full-resolution image for download."""
    if is_runsignup_photo(source_url):
        return source_url  # large_v3 is the highest public size
    if is_google_photo(source_url):
        return f"{source_url}=d"
    return source_url


def supports_face_crop(source_url: str) -> bool:
    """Only Google's CDN supports the fcrop64 region-crop parameter."""
    return is_google_photo(source_url)

#!/usr/bin/env python3
"""Discover all photo URLs in a RunSignup race-photo album with Playwright.

RunSignup galleries paginate client-side via
  /RaceDayPhotos/locationPhotos?raceId=..&locationId=..&raceEventDaysId=..&nonce=..&offset=N
The nonce is bound to the browser session (headless curl gets "Security check
failed"), so we open the album in Playwright, capture the API call the page
itself makes, then loop every offset from inside the page context (same
cookies) and collect each photo's `img.large.location`.

Usage:
  python src/runsignup_discover.py <album_url> <out.urls.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

PAGE_SIZE = 18


def discover(album_url: str) -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        )
        page = ctx.new_page()
        with page.expect_response(lambda r: "/RaceDayPhotos/locationPhotos" in r.url, timeout=60000) as resp_info:
            page.goto(album_url, wait_until="domcontentloaded", timeout=120000)
        api_url = resp_info.value.url
        title = page.title()

        # Build the base API URL (drop offset) from the page's own request.
        parsed = urlparse(api_url)
        qs = parse_qs(parsed.query)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?" + "&".join(
            f"{k}={v[0]}" for k, v in qs.items() if k != "offset"
        ) + "&offset="

        # Loop all offsets from inside the page so session cookies apply.
        result = page.evaluate(
            """async (args) => {
                const { base, pageSize } = args;
                const first = await (await fetch(base + '0')).json();
                const total = first.info.total;
                const urls = [];
                const push = (j) => (j.photos||[]).forEach(p => {
                  const u = p.img && p.img.large && p.img.large.location; if (u) urls.push(u);
                });
                push(first);
                for (let off = pageSize; off < total; off += pageSize) {
                  let ok = false;
                  for (let attempt = 0; attempt < 3 && !ok; attempt++) {
                    try { push(await (await fetch(base + off)).json()); ok = true; }
                    catch (e) { await new Promise(r => setTimeout(r, 500)); }
                  }
                }
                return { total, urls: [...new Set(urls)] };
            }""",
            {"base": base, "pageSize": PAGE_SIZE},
        )
        browser.close()

    return {
        "album_title": title,
        "album_url": album_url,
        "api_total": result["total"],
        "urls": result["urls"],
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    album_url, out = sys.argv[1], Path(sys.argv[2])
    data = discover(album_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{data['album_title']}: api_total={data['api_total']} urls={len(data['urls'])} -> {out}")
    return 0 if data["urls"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

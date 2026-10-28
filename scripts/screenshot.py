"""Capture the Grafana dashboard as PNG (uses the locally installed Edge via Playwright).

python scripts/screenshot.py http://localhost:3001 docs/img/dashboard.png --from 2026-10-28T08:00:00Z --to ...
"""

from __future__ import annotations

import argparse
from datetime import datetime

from playwright.sync_api import sync_playwright


def to_ms(value: str) -> str:
    if value.startswith("now"):
        return value
    return str(int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grafana")
    ap.add_argument("out")
    ap.add_argument("--from", dest="start", default="now-30m")
    ap.add_argument("--to", dest="end", default="now")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1250)
    args = ap.parse_args()
    time_range = f"from={to_ms(args.start)}&to={to_ms(args.end)}"
    url = f"{args.grafana}/d/llm-gateway/llm-gateway?orgId=1&kiosk&{time_range}"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(4000)  # let panels finish rendering
        page.screenshot(path=args.out, full_page=True)
        browser.close()
    print(args.out)


if __name__ == "__main__":
    main()

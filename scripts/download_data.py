"""Download the raw public data used by the project.

Sources
-------
* US DOT Bureau of Transportation Statistics (BTS), "Reporting Carrier On-Time Performance
  (1987-present)" monthly PREZIP files.
* OurAirports airport metadata (public domain).
* BTS L_UNIQUE_CARRIERS lookup (carrier code -> name).

Usage: python scripts/download_data.py [--months 2025-8 2025-9 ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops import config  # noqa: E402


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {dest.name} (exists)")
        return
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            print(f"ok    {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as exc:  # network hiccups on transtats are common
            print(f"retry {dest.name} attempt {attempt}: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"failed to download {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", help="YYYY-M values; default = study window")
    args = ap.parse_args()
    config.ensure_dirs()
    months = config.MONTHS
    if args.months:
        months = [tuple(int(x) for x in m.split("-")) for m in args.months]
    for year, month in months:
        fetch(config.BTS_URL.format(year=year, month=month),
              config.RAW / f"bts_ontime_{year}_{month}.zip")
    fetch(config.OURAIRPORTS_URL, config.RAW / "ourairports_airports.csv")
    fetch(config.CARRIER_LOOKUP_URL, config.RAW / "L_UNIQUE_CARRIERS.csv")


if __name__ == "__main__":
    main()

"""Data acquisition. This is the only module that talks to the network."""
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import requests

from cryptomacro import config


class AcquisitionError(Exception):
    """Raised when a request still fails after all retries."""


def get_json(url, params=None, *, session=None, pause=0.0,
             max_tries=config.MAX_TRIES, timeout=config.TIMEOUT, sleep=time.sleep):
    """GET a URL and return its JSON, retrying on HTTP 429 and 5xx."""
    if session is None:
        session = requests.Session()

    last_status = None
    for attempt in range(max_tries):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            # Connection errors and timeouts are not retried. We leave out str(exc) because
            # urllib3 puts the full URL, including the FRED API key, in its message.
            raise AcquisitionError(
                f"Could not reach {url} ({type(exc).__name__}). Check your internet "
                "connection and run the command again; finished downloads are kept."
            ) from None
        last_status = response.status_code
        if last_status == 429 or last_status >= 500:
            if attempt < max_tries - 1:  # no point sleeping after the last try
                # Back off 1 s, 2 s, 4 s, ... before the next try.
                sleep(config.BACKOFF_BASE * 2 ** attempt)
            continue
        if last_status >= 400:
            # Other 4xx: fail immediately, no retry. We build the message from the bare
            # url (not response.raise_for_status()) because that message includes the
            # query string, which for FRED contains the API key.
            raise requests.HTTPError(f"HTTP {last_status} for {url}", response=response)
        try:
            payload = response.json()
        except ValueError:
            # A 200 with an HTML body (e.g. a proxy or captive-portal page) is not usable data.
            raise AcquisitionError(f"{url} returned a response that is not JSON.") from None
        if pause > 0:
            sleep(pause)  # politeness pause between calls to the same API
        return payload

    raise AcquisitionError(f"{url} failed after {max_tries} tries (last status {last_status})")


def candle_windows(start: datetime, end: datetime, granularity: int,
                   max_candles: int = config.MAX_CANDLES) -> Iterator[tuple[datetime, datetime]]:
    """Yield (window_start, window_end) pairs, each covering at most max_candles candles."""
    step = timedelta(seconds=granularity)
    window_start = start
    while window_start <= end:
        # The last candle of a full window starts (max_candles - 1) steps after the first.
        window_end = min(window_start + (max_candles - 1) * step, end)
        yield window_start, window_end
        window_start = window_start + max_candles * step  # next window starts right after


def window_path(raw_dir: Path, product: str, granularity: int, start: datetime) -> Path:
    """File where the candles for one window are cached."""
    return raw_dir / "coinbase" / f"{product}_{granularity}_{start:%Y%m%dT%H%M}.json"


def write_json(path: Path, payload) -> None:
    """Save payload as JSON. Write to a temp file first so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    os.replace(tmp_path, path)


def read_json(path: Path):
    """Return the JSON stored at path, or None if the file is missing or not valid JSON."""
    try:
        return json.loads(path.read_text())
    # ValueError covers bad JSON (JSONDecodeError) and bytes that are not UTF-8 text (UnicodeDecodeError).
    except (FileNotFoundError, ValueError):
        return None


def fetch_candles(product: str, granularity: int, start: datetime, end: datetime,
                  raw_dir: Path = config.RAW_DIR, *, session=None,
                  sleep=time.sleep) -> tuple[int, int]:
    """Download candles window by window, skipping windows already saved. Returns (fetched, skipped)."""
    url = config.COINBASE_URL.format(product=product)
    fetched = 0
    skipped = 0
    for window_start, window_end in candle_windows(start, end, granularity):
        path = window_path(raw_dir, product, granularity, window_start)
        # An empty list is a valid cached answer, so compare with None, not truthiness.
        if read_json(path) is not None:
            skipped += 1
            continue
        params = {"granularity": granularity,
                  "start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": window_end.strftime("%Y-%m-%dT%H:%M:%SZ")}
        candles = get_json(url, params, session=session, pause=config.COINBASE_PAUSE, sleep=sleep)
        write_json(path, candles)
        fetched += 1
    return fetched, skipped


def key_from_env_file(path: Path) -> str:
    """Return the FRED_API_KEY value written in a .env file, or "" if there is none."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, ValueError):
        return ""
    for line in lines:
        name, _, value = line.partition("=")
        if name.strip() == "FRED_API_KEY":
            return value.strip().strip('"\'')
    return ""


def fred_api_key() -> str:
    """Read the FRED key: the FRED_API_KEY environment variable first, then the .env file.

    The .env file sits in the project folder and is git-ignored, so the key is
    never committed. The key is never written anywhere by this program.
    """
    key = os.environ.get("FRED_API_KEY", "").strip() or key_from_env_file(config.ENV_FILE)
    if not key:
        raise AcquisitionError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and run: "
            "export FRED_API_KEY=your_key")
    return key


def get_fred_json(path: str, params: dict, *, session, sleep):
    """GET one FRED endpoint (with the API key added) and turn a 400 into a clear error."""
    params = {**params, "api_key": fred_api_key(), "file_type": "json"}
    try:
        return get_json(f"{config.FRED_URL}/{path}", params, session=session,
                        pause=config.FRED_PAUSE, sleep=sleep)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 400:
            raise
        # FRED answers 400 for a missing, malformed or unregistered key.
        try:
            reason = exc.response.json().get("error_message", "")
        except ValueError:  # the body was not JSON
            reason = ""
        raise AcquisitionError("FRED rejected the request \u2014 check that FRED_API_KEY is a "
                               f"valid 32-character key. FRED said: {reason}") from None


def fetch_fred_series(series_id: str, raw_dir: Path = config.RAW_DIR, *, session=None,
                      sleep=time.sleep) -> bool:
    """Download one FRED series unless it is already saved. Returns True if fetched, False if cached."""
    path = raw_dir / "fred" / f"{series_id}.json"
    if read_json(path) is not None:
        return False
    start = config.PERIOD_START
    if series_id == "DFF":
        # The rate regime compares DFF with its value 90 days earlier, so DFF starts earlier
        # than the period; otherwise the first 90 days of 2020 (including the March cuts) get no label.
        first_day = date.fromisoformat(config.PERIOD_START) - timedelta(days=config.REGIME_FETCH_MARGIN_DAYS)
        start = first_day.isoformat()
    params = {"series_id": series_id,
              "observation_start": start,
              "observation_end": config.PERIOD_END}
    write_json(path, get_fred_json("series/observations", params, session=session, sleep=sleep))
    return True


def fetch_cpi_release_dates(raw_dir: Path = config.RAW_DIR, *, session=None,
                            sleep=time.sleep) -> bool:
    """Download the CPI release calendar unless already saved. Returns True if fetched, False if cached."""
    path = raw_dir / "fred" / "cpi_release_dates.json"
    if read_json(path) is not None:
        return False
    params = {"release_id": config.CPI_RELEASE_ID, "limit": 10000, "sort_order": "asc"}
    write_json(path, get_fred_json("release/dates", params, session=session, sleep=sleep))
    return True


def write_pull_marker(raw_dir: Path) -> None:
    """Record that every download finished: the period covered and when (UTC)."""
    write_json(raw_dir / config.PULL_COMPLETE_FILE,
               {"period_start": config.PERIOD_START, "period_end": config.PERIOD_END,
                "completed_utc": datetime.now(timezone.utc).isoformat()})


def main(argv=None) -> int:
    """Download everything the project needs. Safe to re-run: saved files are skipped."""
    try:
        fred_api_key()  # check the key first so a missing key fails before any request
        session = requests.Session()
        fetched = 0
        skipped = 0
        # FRED first: it takes seconds, so a wrong key is reported before the long Coinbase download.
        for series_id in config.FRED_SERIES:
            is_new = fetch_fred_series(series_id, config.RAW_DIR, session=session)
            print(f"{series_id}: {'fetched' if is_new else 'already on disk'}")
            if is_new:
                fetched += 1
            else:
                skipped += 1
        is_new = fetch_cpi_release_dates(config.RAW_DIR, session=session)
        print(f"CPI release dates: {'fetched' if is_new else 'already on disk'}")
        if is_new:
            fetched += 1
        else:
            skipped += 1

        start = datetime.fromisoformat(config.PERIOD_START).replace(tzinfo=timezone.utc)
        last_day = datetime.fromisoformat(config.PERIOD_END).replace(tzinfo=timezone.utc)
        # The last candle starts at 00:00 for daily candles and 23:00 for hourly ones.
        ends = {"daily": last_day, "hourly": last_day + timedelta(hours=23)}
        for product in config.PRODUCTS.values():
            for name, granularity in config.GRANULARITIES.items():
                new, old = fetch_candles(product, granularity, start, ends[name],
                                         config.RAW_DIR, session=session)
                print(f"{product} {name}: {new} fetched, {old} already on disk")
                fetched += new
                skipped += old
    except (AcquisitionError, requests.HTTPError) as exc:
        # An HTTPError message is "HTTP <status> for <url>" (no query string, so no key).
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # Only reached when every download above succeeded.
    write_pull_marker(config.RAW_DIR)
    print(f"Done: {fetched} fetched, {skipped} already on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

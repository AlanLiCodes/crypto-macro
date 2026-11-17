import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from cryptomacro import acquire, config
from cryptomacro.acquire import (AcquisitionError, candle_windows, fetch_candles,
                                 fetch_cpi_release_dates, fetch_fred_series, fred_api_key,
                                 get_json, main, read_json, window_path, write_json)
from tests.conftest import FakeSession


def test_get_json_returns_payload():
    s = FakeSession([(200, {"a": 1})])
    assert get_json("u", {"p": 1}, session=s, sleep=lambda x: None) == {"a": 1}
    assert s.calls[0]["params"] == {"p": 1} and s.calls[0]["timeout"] == 30


def test_get_json_retries_429_and_5xx_with_backoff():
    sleeps = []
    s = FakeSession([(429, {}), (503, {}), (200, [1])])
    assert get_json("u", session=s, sleep=sleeps.append) == [1]
    assert sleeps == [1.0, 2.0]


def test_get_json_gives_up_after_max_tries():
    sleeps = []
    s = FakeSession([(500, {})] * 5)
    with pytest.raises(AcquisitionError, match="500"):
        get_json("u", session=s, sleep=sleeps.append)
    assert len(s.calls) == 5
    assert len(sleeps) == 4  # no sleep after the final failed try


def test_get_json_does_not_retry_404():
    s = FakeSession([(404, {})])
    with pytest.raises(requests.HTTPError):
        get_json("u", session=s, sleep=lambda x: None)
    assert len(s.calls) == 1


def test_get_json_pauses_after_success():
    sleeps = []
    get_json("u", session=FakeSession([(200, {})]), pause=0.25, sleep=sleeps.append)
    assert sleeps == [0.25]


def test_get_json_4xx_error_does_not_leak_params():
    s = FakeSession([(400, {"error": "bad key"})])
    with pytest.raises(requests.HTTPError) as excinfo:
        get_json("http://example/x", {"api_key": "SECRET"}, session=s, sleep=lambda x: None)
    assert "SECRET" not in str(excinfo.value)
    assert excinfo.value.response.status_code == 400


UTC = timezone.utc


def test_candle_windows_daily_sizes():
    w = list(candle_windows(datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC), 86400))
    assert w[0] == (datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 10, 26, tzinfo=UTC))  # 300 days inclusive
    assert w[1][0] == datetime(2020, 10, 27, tzinfo=UTC)
    assert w[-1][1] == datetime(2021, 12, 31, tzinfo=UTC) and len(w) == 3


def test_candle_windows_hourly_is_twelve_and_a_half_days():
    w = next(candle_windows(datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 3, 1, tzinfo=UTC), 3600))
    assert w[1] - w[0] == timedelta(hours=299)


def test_candle_windows_is_a_generator():
    assert inspect.isgeneratorfunction(candle_windows)


def test_fetch_candles_saves_each_window_and_resumes(tmp_path):
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 12, 31, tzinfo=UTC)
    s = FakeSession([(200, [[1, 1, 1, 1, 1, 1]])] * 3)
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s, sleep=lambda x: None) == (3, 0)
    assert (tmp_path / "coinbase" / "BTC-USD_86400_20200101T0000.json").exists()
    assert s.calls[0]["url"] == "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    assert s.calls[0]["params"] == {"granularity": 86400, "start": "2020-01-01T00:00:00Z",
                                    "end": "2020-10-26T00:00:00Z"}
    s2 = FakeSession([])
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s2, sleep=lambda x: None) == (0, 3)


def test_fetch_candles_pauses_between_calls(tmp_path):
    sleeps = []
    s = FakeSession([(200, [[1]])])
    fetch_candles("BTC-USD", 86400, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 5, tzinfo=UTC),
                  tmp_path, session=s, sleep=sleeps.append)
    assert sleeps == [0.25]


def test_fetch_candles_keeps_empty_windows(tmp_path):  # an empty list is a valid cached answer
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 5, tzinfo=UTC)
    s = FakeSession([(200, [])])
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s, sleep=lambda x: None) == (1, 0)
    path = window_path(tmp_path, "BTC-USD", 86400, start)
    assert json.loads(path.read_text()) == []
    s2 = FakeSession([])
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s2, sleep=lambda x: None) == (0, 1)


def test_fetch_candles_refetches_corrupt_file(tmp_path):
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 5, tzinfo=UTC)
    path = window_path(tmp_path, "BTC-USD", 86400, start)
    path.parent.mkdir(parents=True)
    path.write_text("[[1, 2")
    s = FakeSession([(200, [[1, 2, 3, 4, 5, 6]])])
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s, sleep=lambda x: None) == (1, 0)
    assert json.loads(path.read_text()) == [[1, 2, 3, 4, 5, 6]]


def test_read_json_returns_none_when_missing_or_invalid(tmp_path):
    assert read_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert read_json(bad) is None


def test_read_json_returns_none_for_bytes_that_are_not_utf8(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00[1]")  # not valid UTF-8, so decoding fails before JSON parsing
    assert read_json(bad) is None


def test_write_json_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "sub" / "x.json"
    write_json(target, [1, 2])
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]
    assert read_json(target) == [1, 2]


def test_fred_api_key_missing(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(AcquisitionError, match="FRED_API_KEY is not set"):
        fred_api_key()


def test_fred_api_key_empty(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "  ")
    with pytest.raises(AcquisitionError, match="FRED_API_KEY is not set"):
        fred_api_key()


def test_fred_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", " abc \n")
    assert fred_api_key() == "abc"


def test_fetch_fred_series_params_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    s = FakeSession([(200, {"observations": []})])
    assert fetch_fred_series("DGS10", tmp_path, session=s, sleep=lambda x: None) is True
    assert s.calls[0]["url"] == config.FRED_URL + "/series/observations"
    p = s.calls[0]["params"]
    assert p["series_id"] == "DGS10" and p["api_key"] == "k" and p["file_type"] == "json"
    assert p["observation_start"] == "2020-01-01" and p["observation_end"] == "2026-09-30"
    assert read_json(tmp_path / "fred" / "DGS10.json") == {"observations": []}
    # Second call finds the file and makes no request (the empty FakeSession would raise).
    assert fetch_fred_series("DGS10", tmp_path, session=FakeSession([]), sleep=lambda x: None) is False


def test_fetch_fred_series_dff_starts_early_for_the_regime_lookback(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    s = FakeSession([(200, {"observations": []})])
    fetch_fred_series("DFF", tmp_path, session=s, sleep=lambda x: None)
    # 100 days before 2020-01-01; the end of the period is unchanged.
    assert s.calls[0]["params"]["observation_start"] == "2019-09-23"
    assert s.calls[0]["params"]["observation_end"] == "2026-09-30"


def test_fetch_cpi_release_dates_uses_release_10(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    s = FakeSession([(200, {"release_dates": []})])
    assert fetch_cpi_release_dates(tmp_path, session=s, sleep=lambda x: None) is True
    assert s.calls[0]["url"] == config.FRED_URL + "/release/dates"
    p = s.calls[0]["params"]
    assert p["release_id"] == 10 and p["api_key"] == "k" and p["file_type"] == "json"
    assert "include_release_dates_with_no_data" not in p  # FRED's default: only dates with data
    assert p["limit"] == 10000 and p["sort_order"] == "asc"
    assert read_json(tmp_path / "fred" / "cpi_release_dates.json") == {"release_dates": []}
    assert fetch_cpi_release_dates(tmp_path, session=FakeSession([]), sleep=lambda x: None) is False


def test_bad_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    msg = "Bad Request. The value for variable api_key is not registered."
    s = FakeSession([(400, {"error_message": msg})])
    with pytest.raises(AcquisitionError, match="check that FRED_API_KEY") as excinfo:
        fetch_fred_series("DGS10", tmp_path, session=s, sleep=lambda x: None)
    assert msg in str(excinfo.value)
    assert "SECRETKEY123" not in str(excinfo.value)
    assert not (tmp_path / "fred" / "DGS10.json").exists()


def test_bad_key_on_release_dates_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "bad")
    s = FakeSession([(400, {"error_message": "nope"})])
    with pytest.raises(AcquisitionError, match="check that FRED_API_KEY"):
        fetch_cpi_release_dates(tmp_path, session=s, sleep=lambda x: None)


def test_main_without_key_returns_1_and_makes_no_requests(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    def no_session(*args, **kwargs):
        raise AssertionError("a session was created, so a request could have been made")
    monkeypatch.setattr(acquire.requests, "Session", no_session)
    assert main([]) == 1
    assert "FRED_API_KEY is not set" in capsys.readouterr().err


def test_main_fetches_everything_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    calls = {"candles": [], "fred": [], "cpi": 0}
    order = []  # which kind of download ran, in sequence

    def fake_candles(product, granularity, start, end, raw_dir, **kwargs):
        calls["candles"].append((product, granularity, start, end))
        order.append("coinbase")
        return 2, 1

    def fake_series(series_id, raw_dir, **kwargs):
        calls["fred"].append(series_id)
        order.append("fred")
        return True

    def fake_cpi(raw_dir, **kwargs):
        calls["cpi"] += 1
        order.append("cpi")
        return False
    monkeypatch.setattr(acquire, "fetch_candles", fake_candles)
    monkeypatch.setattr(acquire, "fetch_fred_series", fake_series)
    monkeypatch.setattr(acquire, "fetch_cpi_release_dates", fake_cpi)
    assert main([]) == 0
    utc = timezone.utc
    start = datetime(2020, 1, 1, tzinfo=utc)
    assert sorted(calls["candles"]) == sorted([
        ("BTC-USD", 86400, start, datetime(2026, 9, 30, 0, tzinfo=utc)),
        ("ETH-USD", 86400, start, datetime(2026, 9, 30, 0, tzinfo=utc)),
        ("BTC-USD", 3600, start, datetime(2026, 9, 30, 23, tzinfo=utc)),
        ("ETH-USD", 3600, start, datetime(2026, 9, 30, 23, tzinfo=utc)),
    ])
    assert calls["fred"] == list(config.FRED_SERIES) and calls["cpi"] == 1
    # FRED first (a bad key fails within seconds), then the CPI dates, then the long Coinbase download.
    assert order == ["fred"] * 6 + ["cpi"] + ["coinbase"] * 4
    out = capsys.readouterr()
    # 4 candle runs x (2 fetched, 1 skipped) + 6 series fetched + release dates cached.
    assert "Done: 14 fetched, 5 already on disk." in out.out
    assert "SECRETKEY123" not in out.out + out.err


def test_main_writes_completion_marker_after_success(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(acquire, "fetch_candles", lambda *args, **kwargs: (0, 1))
    monkeypatch.setattr(acquire, "fetch_fred_series", lambda *args, **kwargs: False)
    monkeypatch.setattr(acquire, "fetch_cpi_release_dates", lambda *args, **kwargs: False)
    assert main([]) == 0
    marker = read_json(tmp_path / "pull_complete.json")
    assert marker["period_start"] == "2020-01-01" and marker["period_end"] == "2026-09-30"
    completed = datetime.fromisoformat(marker["completed_utc"])
    assert completed.tzinfo is not None and completed.utcoffset() == timedelta(0)


def test_main_writes_no_marker_when_a_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(acquire, "fetch_fred_series", lambda *args, **kwargs: True)
    monkeypatch.setattr(acquire, "fetch_cpi_release_dates", lambda *args, **kwargs: True)

    def failing_candles(*args, **kwargs):
        raise AcquisitionError("boom")
    monkeypatch.setattr(acquire, "fetch_candles", failing_candles)
    assert main([]) == 1
    assert not (tmp_path / "pull_complete.json").exists()


def test_main_error_message_never_contains_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    def fake_series(*args, **kwargs):
        raise AcquisitionError("boom")
    monkeypatch.setattr(acquire, "fetch_fred_series", fake_series)
    assert main([]) == 1
    out = capsys.readouterr()
    assert "Error: boom" in out.err and "SECRETKEY123" not in out.out + out.err


def test_key_never_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    s = FakeSession([(200, {"observations": []}), (200, {"release_dates": []})])
    fetch_fred_series("DGS10", tmp_path, session=s, sleep=lambda x: None)
    fetch_cpi_release_dates(tmp_path, session=s, sleep=lambda x: None)
    files = [f for f in tmp_path.rglob("*") if f.is_file()]
    assert files
    assert all("SECRETKEY123" not in f.read_text() for f in files)


def test_get_json_connection_error_hides_key():
    boom = requests.ConnectionError("HTTPSConnectionPool: /x?api_key=SECRET")
    with pytest.raises(AcquisitionError, match="Could not reach u") as excinfo:
        get_json("u", {"api_key": "SECRET"}, session=FakeSession([(0, boom)]), sleep=lambda x: None)
    assert "SECRET" not in str(excinfo.value)
    assert "ConnectionError" in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__ is True


def test_main_prints_http_error_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRED_API_KEY", "SECRETKEY123")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)  # never read the real data/raw
    monkeypatch.setattr(acquire.requests, "Session", lambda: FakeSession([(404, {})]))
    assert main([]) == 1
    out = capsys.readouterr()
    # FRED is downloaded first, so the first request (and the 404) is a FRED one.
    assert out.err.startswith("Error: HTTP 404 for https://api.stlouisfed.org/fred/series/observations")
    assert "Traceback" not in out.err and "SECRETKEY123" not in out.out + out.err


def test_bad_key_with_non_json_body_gives_empty_reason(tmp_path, monkeypatch):
    class HtmlResponse:
        status_code = 400

        def json(self):
            raise ValueError("not json")

    class HtmlSession:
        def get(self, url, params=None, timeout=None):
            return HtmlResponse()
    monkeypatch.setenv("FRED_API_KEY", "k")
    with pytest.raises(AcquisitionError, match="check that FRED_API_KEY"):
        fetch_fred_series("DGS10", tmp_path, session=HtmlSession(), sleep=lambda x: None)


def test_get_json_non_json_success_body_gives_clear_error():
    class HtmlResponse:  # e.g. a proxy or captive-portal page answering with status 200
        status_code = 200

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    class HtmlSession:
        def get(self, url, params=None, timeout=None):
            return HtmlResponse()
    with pytest.raises(AcquisitionError) as excinfo:
        get_json("http://example/x", {"api_key": "SECRET"}, session=HtmlSession(), sleep=lambda x: None)
    assert str(excinfo.value) == "http://example/x returned a response that is not JSON."
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__ is True


def test_fred_api_key_falls_back_to_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('# my key\nOTHER=1\nFRED_API_KEY = "abc123"\n')
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    assert fred_api_key() == "abc123"


def test_environment_variable_wins_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FRED_API_KEY=from_file\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("FRED_API_KEY", "from_env")
    assert fred_api_key() == "from_env"


def test_env_file_without_key_still_reports_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    with pytest.raises(AcquisitionError, match="FRED_API_KEY is not set"):
        fred_api_key()

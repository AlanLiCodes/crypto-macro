import pytest
"""Shared test helpers. FakeSession stands in for requests.Session (no network)."""
import requests


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            # Keep the response on the error so callers can read the body.
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Returns the queued (status, payload) responses in order and records every get call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        status, payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload  # lets a test simulate a connection error or timeout
        return FakeResponse(status, payload)


@pytest.fixture(autouse=True)
def no_real_env_file(tmp_path, monkeypatch):
    """Point config.ENV_FILE at a file that does not exist, so no test reads a real key."""
    from cryptomacro import config
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "no-such.env")

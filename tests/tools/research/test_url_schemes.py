"""The URLs research opens from outside its code (the Job queue's, the public export's, the monitors' webhook) are http or https:
any other scheme is refused before anything is opened, by a message that names the scheme and not the URL."""
from __future__ import annotations

import urllib.request
from types import SimpleNamespace

import pytest

from research import monitors as M
from research.jobs.client import HTTPClient
from research.store.remote import HttpRemote

OPENERS = {
    "jobs.HTTPClient": lambda url: HTTPClient(url=url, token="t"),
    "store.HttpRemote": HttpRemote,
    "monitors.redirected_base": lambda url: M.redirected_base(url, "manifests/k.json"),
    "monitors.webhook_post": lambda url: M.webhook_post(url, "hi"),
}


class _Answer:
    status = 200

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


@pytest.mark.parametrize("url", ["ftp://secret.invalid/x", "file:///secret/x", "secret.invalid/x", "https://secret.invalid/x"])
@pytest.mark.parametrize("opener", sorted(OPENERS))
def test_only_http_and_https_urls_are_opened(opener, url, monkeypatch):
    opened: list[str] = []

    def urlopen(req, timeout=None):
        opened.append(req.full_url)
        return _Answer()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: SimpleNamespace(open=urlopen))
    if url.startswith("https:"):
        OPENERS[opener](url)
        return
    with pytest.raises(ValueError, match="is not an http or https URL") as e:
        OPENERS[opener](url)
    assert "secret" not in str(e.value) and opened == []

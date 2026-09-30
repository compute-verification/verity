"""Talking to the Job queue: `HTTPClient` for the site, `ModelClient` for the reference model (tests, and trying a change).

Both answer `call(method, path, body) -> (status, body)` and raise `Unreachable` when no answer came.  The two tokens a VM can
hold have their own variables and files, so they can't collide:
    research jobs     `$JOBS_TOKEN`, else ~/.research/jobs/token: the coordinator's, which adds jobs
    research worker   `$JOBS_WORKER_TOKEN`, else ~/.research/jobs/dispatch-{name}.token: the dispatcher's `jobs:work` token
Each file is one line, mode 600.  A token is sent as `Authorization: Bearer`, never printed, and never put
in an argv.  The site is `$JOBS_URL`, default the production control app.
"""
from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import model

DEFAULT_URL = "https://website-docs-sage.vercel.app"
URL_ENV, TOKEN_ENV, WORKER_TOKEN_ENV = "JOBS_URL", "JOBS_TOKEN", "JOBS_WORKER_TOKEN"
TOKEN_FILE = Path.home() / ".research" / "jobs" / "token"


def worker_token_file(name: str, workdir: Path | None = None) -> Path:
    """The dispatcher `dispatch:{name}`'s own token file, never `research jobs`' TOKEN_FILE."""
    return (workdir or TOKEN_FILE.parent) / f"dispatch-{name}.token"
TIMEOUT_S = 30


class Unreachable(Exception):
    """No answer from the Job queue (network, DNS, a 5xx); the caller tries again later."""


def load_token(environ: dict[str, str] | None = None, path: Path | None = None, var: str = TOKEN_ENV) -> str:
    """The token in `$var`, else in the token file."""
    env = os.environ if environ is None else environ
    if env.get(var):
        return env[var].strip()
    p = path or TOKEN_FILE
    if not p.exists():
        raise SystemExit(f"research jobs: no token: set ${var} or write it to {p} (mode 600)")
    if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(f"research jobs: {p} is readable by others; chmod 600 it")
    return p.read_text().strip()


class HTTPClient:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self._token = token if token is not None else load_token()

    def __repr__(self) -> str:
        return f"HTTPClient({self.url!r})"

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
                                              "User-Agent": "research-jobs"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                raw = r.read()
                return r.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                raise Unreachable(f"{method} {path}: HTTP {e.code}") from None
            raw = e.read()
            try:
                return e.code, json.loads(raw) if raw else {}
            except ValueError:
                return e.code, {"error": raw.decode(errors="replace")[:300]}
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise Unreachable(f"{method} {path}: {type(e).__name__}: {e}") from None


class ModelClient:
    """A client of an in-memory `model.JobService`, on the caller's clock."""

    def __init__(self, service: model.JobService, token: str | None, clock: Callable[[], float]):
        self.service, self._token, self.clock = service, token, clock
        self.down = False                      # a test turns the network off

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        if self.down:
            raise Unreachable(f"{method} {path}: down")
        return self.service.request(method, path, self._token, json.loads(json.dumps(body)) if body is not None else None, self.clock())

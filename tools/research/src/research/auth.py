"""`research auth request`: a token for the control app, approved in a browser, from any terminal.

    research auth request --name NAME --scopes A,B --file PATH [--note TEXT] [--url URL] [--force]

It asks the API server for a token (`POST /api/auth/requests`, with no token of its own), prints a link and a short code, and
waits.  The requester opens the link and signs in with GitHub on the site: that sign-in is who asked.  The page shows the same
code, so they can check the request is theirs.  Once Daniel approves it on the approvals page, or a pre-approval rule for that
GitHub login matches, the next poll (`POST /api/auth/requests/{id}/poll`, with a secret only this terminal holds) returns the
token, once, and it is written to PATH with mode 600 (its directory created with mode 700).  Neither the token nor the poll
secret is ever printed or put in an argv.  It needs only Python's standard library and HTTPS, so it works in any terminal, with
or without Cursor.

A PATH that exists is kept unless --force.  The site is --url, else `$JOBS_URL`, else the production control app.  Exit 0 once
PATH is written; 1 when the request is denied, expires or was already collected; 2 on a usage error; 4 when the site can't be
reached to make the request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from . import __version__
from .jobs.client import HTTPClient, Unreachable

NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
SCOPE = re.compile(r"[a-z][a-z0-9_-]*(:[a-z][a-z0-9_-]*)?")
DEFAULT_INTERVAL_S = 5
MAX_INTERVAL_S = 60


def write_secret(path: Path, text: str) -> None:
    """`text` at `path`, mode 600 from the first byte, replacing it atomically; a new parent directory gets mode 700."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text.rstrip("\n") + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o600)


def _seconds(iso: str | None) -> float | None:
    if not iso:
        return None
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def request(client, name: str, scopes: list[str], path: Path, *, note: str | None = None, host: str | None = None,
            sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
            out: Callable[[str], None] = print) -> int:
    """Ask for the token, wait for its approval, write it to `path`.  The exit code (module docstring)."""
    body = {"name": name, "scopes": scopes, "note": note, "host": host or socket.gethostname(), "client": f"research {__version__}"}
    try:
        status, r = client.call("POST", "/api/auth/requests", {k: v for k, v in body.items() if v is not None})
    except Unreachable as e:
        out(f"research auth: the site can't be reached ({e})")
        return 4
    if status != 201:
        out(f"research auth: the request was refused: {status} {r.get('error', r)}")
        return 1
    rid, secret = r["id"], r["poll_secret"]
    interval = float(r.get("interval_s") or DEFAULT_INTERVAL_S)
    expires = _seconds(r.get("expires_at"))
    out(f"research auth: to get the token `{name}` ({', '.join(scopes)}), open this link in a browser and sign in with GitHub:\n\n"
        f"    {r['approve_url']}\n\n"
        f"  Check that the page shows the code {r['user_code']}. Waiting until it's approved"
        + (f" (the request expires at {r['expires_at']}; Ctrl-C stops waiting)." if expires else " (Ctrl-C stops waiting)."))
    said = None
    while True:
        sleep(interval)
        if expires is not None and clock() > expires + MAX_INTERVAL_S:
            out("research auth: the request expired before it was approved; ask again")
            return 1
        try:
            status, r = client.call("POST", f"/api/auth/requests/{rid}/poll", {"poll_secret": secret})
        except Unreachable as e:
            interval = min(interval * 2, MAX_INTERVAL_S)
            if said != "unreachable":
                out(f"research auth: the site can't be reached ({e}); still waiting")
                said = "unreachable"
            continue
        if status == 429:
            interval = min(float(r.get("interval_s") or interval * 2), MAX_INTERVAL_S)
            continue
        if status != 200:
            out(f"research auth: the request can't be followed: {status} {r.get('error', r)}")
            return 1
        state = r.get("state")
        if state == "approved" and r.get("token"):
            write_secret(path, r["token"])
            got = r.get("scopes") or scopes
            out(f"research auth: approved{' for ' + r['github_login'] if r.get('github_login') else ''}: wrote `{r.get('name', name)}` "
                f"({', '.join(got)}{', expires ' + r['expires_at'] if r.get('expires_at') else ''}) to {path} (mode 600)")
            if set(got) != set(scopes):
                out(f"research auth: the approver narrowed the scopes you asked for ({', '.join(scopes)})")
            return 0
        if state in ("denied", "expired", "collected", "approved"):
            out({"denied": "research auth: the request was denied",
                 "expired": "research auth: the request expired before it was approved; ask again",
                 "collected": "research auth: the token was already collected by an earlier poll; ask again",
                 "approved": "research auth: approved, but no token came back; ask again"}[state])
            return 1
        if state == "signed-in" and said != "signed-in":
            out(f"research auth: signed in as {r.get('github_login') or 'you'}; waiting for approval")
        elif state == "pending" and said is None:
            out("research auth: waiting for you to open the link and sign in")
        said = state


def main(argv: list[str], client=None, **kw) -> int:
    ap = argparse.ArgumentParser(prog="research auth", description="tokens for the control app, approved in a browser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rq = sub.add_parser("request", help="ask for a token, wait for its approval, write it to --file (mode 600)")
    rq.add_argument("--name", required=True, help="the token's name, e.g. dispatch-rc")
    rq.add_argument("--scopes", required=True, help="comma-separated, e.g. jobs:enqueue,events:read")
    rq.add_argument("--file", required=True, type=Path, help="where the token goes (mode 600)")
    rq.add_argument("--note", help="why, for the approver")
    rq.add_argument("--url", help="the control app (default $JOBS_URL, else production)")
    rq.add_argument("--force", action="store_true", help="replace an existing --file")
    ns = ap.parse_args(argv)
    scopes = [s for s in ns.scopes.split(",") if s]
    path = ns.file.expanduser()
    if not NAME.fullmatch(ns.name):
        print(f"research auth: --name {ns.name!r}: lowercase letters, digits, '.', '_' and '-'", file=sys.stderr)
        return 2
    if not scopes or not all(SCOPE.fullmatch(s) for s in scopes):
        print(f"research auth: --scopes {ns.scopes!r}: names like jobs:work, comma-separated", file=sys.stderr)
        return 2
    if path.exists() and not ns.force:
        print(f"research auth: {path} exists; keep it, or ask again with --force to replace it", file=sys.stderr)
        return 2
    return request(client or HTTPClient(ns.url, anonymous=True), ns.name, scopes, path, note=ns.note, **kw)

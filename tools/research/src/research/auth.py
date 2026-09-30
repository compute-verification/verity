"""`research auth`: tokens for the control app, approved in a browser, from any terminal.

    research auth request --name NAME --scopes A,B --file PATH [--days 90] [--agent ID] [--why TEXT] [--url URL] [--force]
    research auth renew --file PATH [--scopes A,B] [--days N] [--url URL]

The contract is the site's `internal/token-requests-spec.md` (`/api/token-requests`).

`request` asks with no token of its own, and prints the link (`verification_uri_complete`) and a short code.  The requester
opens the link and signs in with GitHub: that sign-in is who asked.  A pre-approval rule for that login issues the token at
once, or Daniel approves it on the site's token page.  Meanwhile the command polls at the site's interval, with the request's
device code, a secret it holds only in memory.  The one answer that carries the token is written to PATH with mode 600: a new
file opened O_CREAT|O_EXCL at 0600, then renamed over.  Beside it goes PATH.json, with no secret in it: the token's name,
scopes, lifetime, expiry and GitHub login, which `renew` reads.  Neither the token nor the device code is ever printed,
logged or put in an argv.  It needs only Python's standard library and HTTPS, so it works in any terminal, with or without
Cursor.

`renew` sends PATH's token as the bearer and asks for the same name, scopes and lifetime (or narrower ones): the site approves
that at once, with no sign-in.  The new token replaces the old one at its first use.

A PATH that exists is kept, unless --force (`request` only).  The site is --url, else `$JOBS_URL`, else production.  Exit 0
once PATH is written; 1 when the request is refused, denied or expired, or after GIVE_UP_S; 2 on a usage error; 4 when the site
can't be reached to make the request.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from .jobs.client import HTTPClient, Unreachable

NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,62}")
SCOPE = re.compile(r"[a-z][a-z0-9_-]*:[A-Za-z0-9_./-]+")
DAYS = (1, 365)
DEFAULT_DAYS = 90
DEFAULT_INTERVAL_S = 5
MAX_INTERVAL_S = 60
GIVE_UP_S = 1800
STOPS = {"access_denied": "the request was denied",
         "expired_token": "the request expired: nobody approved it, or it wasn't collected, within 15 minutes; ask again",
         "invalid_grant": "the site doesn't know this request any more, or already delivered its token; ask again"}


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


def meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def ask(client, body: dict, path: Path, *, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
        out: Callable[[str], None] = print) -> int:
    """POST the request, show the link when a sign-in is needed, poll, write the token.  The exit code (module docstring)."""
    try:
        status, r = client.call("POST", "/api/token-requests", body)
    except Unreachable as e:
        out(f"research auth: the site can't be reached ({e})")
        return 4
    if status != 201:
        out(f"research auth: the request was refused: {status} {r.get('error')}: {r.get('message', '')}".rstrip(": "))
        return 1
    code, interval = r["device_code"], float(r.get("interval") or DEFAULT_INTERVAL_S)
    what = f"`{body['name']}` ({', '.join(body['scopes'])}) for {body['days']} days"
    if r.get("status") == "approved":
        out(f"research auth: renewing {what}: approved at once")
    else:
        out(f"research auth: to get the token {what}, open this link in a browser and sign in with GitHub:\n\n"
            f"    {r['verification_uri_complete']}\n\n"
            f"  Check that the page shows the code {r['user_code']}. Waiting for approval (Ctrl-C stops waiting; "
            f"it gives up after {GIVE_UP_S // 60} minutes).")
    start, said = clock(), None
    while True:
        sleep(interval)
        if clock() - start > GIVE_UP_S:
            out(f"research auth: no token after {GIVE_UP_S // 60} minutes; giving up")
            return 1
        try:
            status, r = client.call("POST", "/api/token-requests/poll", {"device_code": code})
        except Unreachable as e:
            interval = min(interval * 2, MAX_INTERVAL_S)
            if said != "unreachable":
                out(f"research auth: the site can't be reached ({e}); still waiting")
                said = "unreachable"
            continue
        if status == 200 and r.get("token"):
            write_secret(path, r["token"])
            meta = {k: r.get(k) for k in ("name", "scopes", "expires_at", "login", "line", "daily_usd")} | {"days": body["days"]}
            write_secret(meta_path(path), json.dumps(meta, sort_keys=True))
            out(f"research auth: wrote `{r.get('name', body['name'])}` ({', '.join(r.get('scopes') or body['scopes'])}"
                + (f", expires {r['expires_at']}" if r.get("expires_at") else "") + (f", for @{r['login']}" if r.get("login") else "")
                + f") to {path} (mode 600)")
            if set(r.get("scopes") or body["scopes"]) != set(body["scopes"]):
                out(f"research auth: the approver narrowed the scopes you asked for ({', '.join(body['scopes'])})")
            if r.get("replaces"):
                out(f"research auth: it replaces the live token `{r.get('name', body['name'])}` at its first use")
            return 0
        error = r.get("error")
        if status == 400 and error == "authorization_pending":
            interval = float(r.get("interval") or interval)
            state = r.get("status")
            if state != said:
                out("research auth: signed in as @" + str(r.get("login")) + "; waiting for approval" if state == "waiting_for_approval"
                    else "research auth: waiting for you to open the link and sign in")
                said = state
            continue
        if status == 400 and error == "slow_down":
            interval = min(float(r.get("interval") or interval + 5), MAX_INTERVAL_S)
            continue
        if status == 429:
            interval = min(interval * 2, MAX_INTERVAL_S)
            continue
        out(f"research auth: {STOPS.get(error, f'the site answered {status} {error}: ' + str(r.get('message', '')))}")
        return 1


def main(argv: list[str], client=None, **kw) -> int:
    ap = argparse.ArgumentParser(prog="research auth", description="tokens for the control app, approved in a browser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rq = sub.add_parser("request", help="ask for a token, wait for its approval, write it to --file (mode 600)")
    rq.add_argument("--name", required=True, help="the token's name, a lane or a tool (never a person), e.g. dispatch-rc")
    rq.add_argument("--scopes", required=True, help="comma-separated, e.g. jobs:enqueue,events:read")
    rq.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"its lifetime, {DAYS[0]} to {DAYS[1]} (default {DEFAULT_DAYS})")
    rq.add_argument("--file", required=True, type=Path, help="where the token goes (mode 600)")
    rq.add_argument("--agent", help="a Cursor agent id, when one asks")
    rq.add_argument("--why", help="for the approver, at most 200 characters")
    rq.add_argument("--url", help="the control app (default $JOBS_URL, else production)")
    rq.add_argument("--force", action="store_true", help="replace an existing --file")
    rn = sub.add_parser("renew", help="a new token for --file's, with the same name, scopes and lifetime (or narrower)")
    rn.add_argument("--file", required=True, type=Path)
    rn.add_argument("--scopes", help="narrower scopes (default: the token's)")
    rn.add_argument("--days", type=int, help="a shorter lifetime (default: the token's)")
    rn.add_argument("--url")
    ns = ap.parse_args(argv)
    path = ns.file.expanduser()
    if ns.cmd == "renew":
        try:
            meta = json.loads(meta_path(path).read_text())
            token = path.read_text().strip()
        except (OSError, ValueError):
            print(f"research auth renew: needs {path} and {meta_path(path)}, as `research auth request` writes them", file=sys.stderr)
            return 2
        name, scopes, days = meta["name"], ns.scopes.split(",") if ns.scopes else meta["scopes"], ns.days or meta.get("days") or DEFAULT_DAYS
        client = client or HTTPClient(ns.url, token=token)
    else:
        name, scopes, days = ns.name, [s for s in ns.scopes.split(",") if s], ns.days
        if path.exists() and not ns.force:
            print(f"research auth: {path} exists; keep it, renew it (`research auth renew --file {path}`), or ask again with --force",
                  file=sys.stderr)
            return 2
        client = client or HTTPClient(ns.url, anonymous=True)
    if not NAME.fullmatch(name):
        print(f"research auth: --name {name!r}: 2 to 63 lowercase letters, digits and '-'", file=sys.stderr)
        return 2
    if not scopes or not all(SCOPE.fullmatch(s) for s in scopes):
        print(f"research auth: --scopes {','.join(scopes)!r}: names like jobs:work, comma-separated", file=sys.stderr)
        return 2
    if not (isinstance(days, int) and DAYS[0] <= days <= DAYS[1]):
        print(f"research auth: --days {days}: {DAYS[0]} to {DAYS[1]}", file=sys.stderr)
        return 2
    body = {"name": name, "scopes": scopes, "days": days, "machine": socket.gethostname()[:100]}
    if ns.cmd == "request":
        body |= {k: v for k, v in (("agent", ns.agent), ("why", ns.why)) if v}
    return ask(client, body, path, **kw)

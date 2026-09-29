"""`research notes sync` / `research notes scan`: the notes tree as a git repo with a remote that several machines write.

    research notes sync [--path P]... [-m MSG] [--retries N] [--remote NAME] [--root DIR]
    research notes scan [--max-bytes N] [--json] [--root DIR]

`sync` commits the caller's changes (every change, or only under `--path`; files over 1 MB and the steward's own state stay out,
as with `snapshot`), then fetches, rebases onto the remote branch, and pushes; a push that loses a race fetches, rebases and pushes
again, at most `--retries` times.  A rebase conflict never drops either side: the rebase is aborted, the local commit stays, the
exit is 3 and the conflicted files are named.  Before anything is pushed, the outgoing commits' added lines are scanned for key
material (`KEY_PATTERNS`, and the value of any secret-named environment variable); a hit refuses the push (exit 4), naming the
file.  With no remote configured the commit is local and the report says so.

A lane note (`lanes/**/*.md`) the sync would add or change stays out of the commit when its filename stamp `YYYYMMDDTHHMMZ` (a
path new since HEAD) or its front matter's `created:` (new or changed since HEAD) is more than 5 minutes ahead of the sync's
clock: one `SYNC-REFUSED <path>: ...` line each, with the value `date -u` gives now, the rest synced, exit 5 when nothing else
failed.  What HEAD already has is never re-checked, and a stamp stops being ahead once the clock passes it.

`scan` reads every blob in the repo's history: files over `--max-bytes` and key material, each with the path it was committed
under.  Exit 1 when it finds anything; run it before the first push and after (acceptance A5).

With `RESEARCH_NOTES_SYNC=1`, `research notes checkpoint | bind | relaunch` run `sync` for the lane directories they wrote once
they succeed (a cloud agent's default; the laptop keeps the watcher's snapshot until the switch-over).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SYNC_ENV = "RESEARCH_NOTES_SYNC"
MAX_BYTES = 1 << 20
FUTURE_RC = 5
FUTURE_TOLERANCE = dt.timedelta(minutes=5)
STAMP_RE = re.compile(r"^(\d{8}T\d{4}Z)-")
CREATED_RE = re.compile(r"^created:\s*[\"']?((\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(Z|[+-]\d{2}:?\d{2})?)[\"']?\s*$",
                        re.M | re.I)
KEY_PATTERNS = {
    "aws-access-key-id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "apikey": re.compile(r"(?i)\bapi_?key\b\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "session-token": re.compile(r"(?i)\bsession_?token\b\"?\s*[=:]\s*[\"']?[A-Za-z0-9/+=_\-]{16,}"),
}
SECRET_ENV_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)", re.I)
NEVER_COMMIT = ("steward-state.json", ".steward-state.json.tmp")


def _run(root: Path, *args: str, timeout: float = 120, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout, input=input)


def _out(root: Path, *args: str) -> str | None:
    r = _run(root, *args)
    return r.stdout if r.returncode == 0 else None


def _secret_values(environ: dict | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return [v for k, v in env.items() if SECRET_ENV_RE.search(k) and v and len(v) >= 16]


def find_keys(text: str, secrets: list[str]) -> list[str]:
    """The names of the key patterns (and `env-secret` for a secret env var's value) that `text` contains."""
    hits = [name for name, rx in KEY_PATTERNS.items() if rx.search(text)]
    return hits + (["env-secret"] if any(s in text for s in secrets) else [])


def _changed(root: Path, paths: list[str] | None, max_bytes: int) -> tuple[list[str], list[str]]:
    """(paths to commit, paths skipped as over max_bytes) among the working tree's changes, limited to `paths` when given."""
    st = _out(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *(paths or []))
    entries, keep, big, i = (st or "").split("\0"), [], [], 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        code, path = e[:2], e[3:]
        if code[0] in "RC":
            keep.append(entries[i])
            i += 1
        if path in NEVER_COMMIT:
            continue
        p = root / path
        (big if p.is_file() and p.stat().st_size > max_bytes else keep).append(path)
    return keep, big


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def created_at(text: str) -> tuple[str, dt.datetime] | None:
    """(as written, time) of the front matter's `created:` (ISO, seconds optional, UTC unless it gives an offset), or None."""
    end = text.find("\n---", 3) if text.startswith("---\n") else -1
    m = CREATED_RE.search(text[4:end]) if end >= 0 else None
    if m is None:
        return None
    try:
        t = dt.datetime.fromisoformat(f"{m[2]}T{m[3]}{m[4] if m[4] and m[4].upper() != 'Z' else ''}")
    except ValueError:
        return None
    return m[1], t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def future_stamps(root: Path, paths: list[str], now: dt.datetime) -> dict[str, str]:
    """{path: why} for the lane notes among `paths` whose filename stamp (a path HEAD lacks) or `created:` (a value HEAD's copy
    lacks) is more than FUTURE_TOLERANCE after `now` (UTC); `why` gives the value `date -u` prints at `now`."""
    out = {}
    for path in paths:
        p = root / path
        if not (path.startswith("lanes/") and path.endswith(".md") and p.is_file()):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        old = _out(root, "show", f"HEAD:{path}")
        found = []
        if old is None and (m := STAMP_RE.match(p.name)):
            try:
                found.append(("stamp", m[1], dt.datetime.strptime(m[1], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc), "%Y%m%dT%H%MZ"))
            except ValueError:
                pass
        c, prev = created_at(text), created_at(old) if old is not None else None
        if c and (prev is None or prev[1] != c[1]):
            found.append(("created:", *c, "%Y-%m-%dT%H:%MZ"))
        why = [f"{what} {was} is {math.ceil((t - now).total_seconds() / 60)} min ahead; use {now.strftime(fmt)} (date -u +{fmt})"
               for what, was, t, fmt in found if t - now > FUTURE_TOLERANCE]
        if why:
            out[path] = "; ".join(why)
    return out


def commit(root: Path, paths: list[str] | None = None, message: str | None = None, max_bytes: int = MAX_BYTES,
           now: dt.datetime | None = None) -> dict:
    """Commit the caller's changes (all, or under `paths`) but lane notes stamped ahead of `now` (`future`: `path: why`).  Retries
    a held index lock (another writer on this machine) briefly."""
    keep, big = _changed(root, paths, max_bytes)
    future = future_stamps(root, keep, now or _utcnow())
    if future:
        keep = [p for p in keep if p not in future]
        _run(root, "--literal-pathspecs", "reset", "-q", "--", *future)          # a staged copy would be committed anyway
    left = {"skipped": big, "future": [f"{p}: {why}" for p, why in future.items()]}
    if not keep:
        return {"commit": None, "paths": 0, **left}
    msg = message or f"notes sync {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}: {len(keep)} paths"
    for attempt in range(10):
        add = _run(root, "--literal-pathspecs", "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(keep))
        if add.returncode == 0:
            c = _run(root, "commit", "-q", "--no-verify", "-m", msg)
            if c.returncode == 0:
                return {"commit": (_out(root, "rev-parse", "--short", "HEAD") or "").strip(), "paths": len(keep), **left}
            if "index.lock" not in c.stderr and "nothing to commit" in (c.stdout + c.stderr):
                return {"commit": None, "paths": 0, **left}
            err = c.stderr
        else:
            err = add.stderr
        if "index.lock" not in err:
            raise RuntimeError(f"git commit failed: {err.strip()[-300:]}")
        time.sleep(0.2 * (attempt + 1))
    raise RuntimeError("git index is locked by another writer (index.lock); try again")


def outgoing_keys(root: Path, base: str | None, secrets: list[str]) -> list[str]:
    """`path: pattern` for key material in lines the commits after `base` (all history when None) add."""
    rng = [f"{base}..HEAD"] if base else ["HEAD"]
    log = _out(root, "log", "-p", "--no-color", "--format=", "--unified=0", *rng) or ""
    hits, path = [], None
    for line in log.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++"):
            for name in find_keys(line, secrets):
                if f"{path}: {name}" not in hits:
                    hits.append(f"{path}: {name}")
    return hits


def sync(root: Path, paths: list[str] | None = None, message: str | None = None, retries: int = 5, remote: str = "origin",
         max_bytes: int = MAX_BYTES, secrets: list[str] | None = None, now: dt.datetime | None = None) -> dict:
    """Commit, then fetch / rebase / push with bounded retries (module docstring).  The report: `commit`, `paths`, `skipped`,
    `future` (notes left out, stamped ahead of `now`), `pushed` (bool), `attempts`, and on failure `error` plus `conflict` (files)
    or `keys` (hits); `rc` is the exit code."""
    root = Path(root)
    if _out(root, "rev-parse", "--git-dir") is None:
        return {"rc": 2, "error": f"{root} is not a git repository"}
    rep = {"rc": 0, "pushed": False, "attempts": 0, **commit(root, paths, message, max_bytes, now)}
    if rep["future"]:
        rep["rc"] = FUTURE_RC
    if _out(root, "remote", "get-url", remote) is None:
        return {**rep, "note": f"no remote {remote!r}: committed locally only"}
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()
    if not branch:
        return {**rep, "rc": 2, "error": "HEAD is detached; check out the notes branch"}
    secrets = _secret_values() if secrets is None else secrets
    tracking = f"refs/remotes/{remote}/{branch}"
    for n in range(retries + 1):
        rep["attempts"] = n + 1
        f = _run(root, "fetch", "-q", remote, f"+refs/heads/{branch}:{tracking}", timeout=300)
        has_remote = f.returncode == 0
        if not has_remote and "couldn't find remote ref" not in f.stderr:
            return {**rep, "rc": 1, "error": f"git fetch {remote} failed: {f.stderr.strip()[-300:]}"}
        base = tracking if has_remote else None
        if has_remote and _run(root, "merge-base", "--is-ancestor", tracking, "HEAD").returncode != 0:
            rb = _run(root, "rebase", "--autostash", tracking, timeout=300)
            if rb.returncode:
                conflict = [x for x in (_out(root, "diff", "--name-only", "--diff-filter=U") or "").splitlines() if x]
                _run(root, "rebase", "--abort")
                return {**rep, "rc": 3, "conflict": conflict,
                        "error": "rebase conflict; nothing was dropped (the local commit is kept, the rebase aborted): resolve "
                                 + (", ".join(conflict) or rb.stderr.strip()[-200:]) + ", then sync again"}
        if base and (_out(root, "rev-parse", "HEAD") or "").strip() == (_out(root, "rev-parse", tracking) or "").strip():
            return {**rep, "pushed": False, "note": "already on the remote"}
        keys = outgoing_keys(root, base, secrets)
        if keys:
            return {**rep, "rc": 4, "keys": keys, "error": "key material in outgoing commits; nothing pushed: " + "; ".join(keys[:5])}
        p = _run(root, "push", "-q", remote, f"HEAD:refs/heads/{branch}", timeout=300)
        if p.returncode == 0:
            _run(root, "fetch", "-q", remote, f"+refs/heads/{branch}:{tracking}", timeout=300)
            return {**rep, "pushed": True}
        if not any(k in p.stderr for k in ("non-fast-forward", "fetch first", "[rejected]", "failed to update ref", "stale info")):
            return {**rep, "rc": 1, "error": f"git push failed (local commit kept): {p.stderr.strip()[-300:]}"}
        time.sleep(min(2.0, 0.1 * 2 ** n))
    return {**rep, "rc": 1, "error": f"lost the push race {retries + 1} times (local commit kept); sync again"}


def lag(root: Path, remote: str = "origin") -> tuple[int, int] | None:
    """(ahead, behind): commits HEAD has that the remote-tracking branch lacks, and the reverse; None without one."""
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()
    counts = _out(root, "rev-list", "--left-right", "--count", f"HEAD...refs/remotes/{remote}/{branch}") if branch else None
    if not counts or len(counts.split()) != 2:
        return None
    ahead, behind = counts.split()
    return int(ahead), int(behind)


def scan(root: Path, max_bytes: int = MAX_BYTES, secrets: list[str] | None = None) -> dict:
    """Every blob in the history: `big` (path, bytes) over max_bytes, `keys` (path: pattern) with key material."""
    secrets = _secret_values() if secrets is None else secrets
    objs = _out(root, "rev-list", "--objects", "--all") or ""
    paths: dict[str, str] = {}
    for line in objs.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(sha, path)
    if not paths:
        return {"blobs": 0, "big": [], "keys": []}
    check = _run(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input="\n".join(paths) + "\n")
    blobs = [(s, int(n)) for s, t, n in (x.split() for x in check.stdout.splitlines() if x.count(" ") == 2) if t == "blob"]
    big = sorted((paths[s], n) for s, n in blobs if n > max_bytes)
    keys = []
    small = [s for s, n in blobs if n <= max_bytes]
    if small:
        proc = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], input=("\n".join(small) + "\n").encode(),
                              capture_output=True, timeout=600)
        data, i = proc.stdout, 0
        while i < len(data):
            nl = data.index(b"\n", i)
            sha, _, size = data[i:nl].decode().split()
            body = data[nl + 1:nl + 1 + int(size)]
            i = nl + 1 + int(size) + 1
            if b"\0" in body[:8192]:
                continue
            for name in find_keys(body.decode(errors="replace"), secrets):
                keys.append(f"{paths[sha]}: {name}")
    return {"blobs": len(blobs), "big": [{"path": p, "bytes": n} for p, n in big], "keys": sorted(set(keys))}


def render_sync(rep: dict) -> str:
    head = (f"notes sync: committed {rep['commit']} ({rep['paths']} paths)" if rep.get("commit") else "notes sync: nothing new to commit")
    head += "; pushed" if rep.get("pushed") else ""
    if rep.get("attempts", 0) > 1:
        head += f" after {rep['attempts']} attempts"
    parts = [head]
    if rep.get("skipped"):
        parts.append(f"skipped {len(rep['skipped'])} over 1 MB (put them in the store): {', '.join(rep['skipped'][:3])}")
    if rep.get("future"):
        parts.append(f"left out {len(rep['future'])} stamped ahead of the clock (SYNC-REFUSED)")
    if rep.get("note"):
        parts.append(rep["note"])
    if rep.get("error"):
        parts.append(f"FAILED: {rep['error']}")
    return "; ".join(parts)


def refused_lines(rep: dict) -> list[str]:
    return [f"SYNC-REFUSED {x}" for x in rep.get("future", [])]


def auto_sync(root: Path, lanes: list[str]) -> tuple[str | None, int]:
    """After a notes write command: `sync` the lane directories it wrote, when `RESEARCH_NOTES_SYNC=1`.  (report lines, FUTURE_RC
    when a note was left out for its stamp, else 0; other failures stay advisory, the next sync retries them); (None, 0) when off."""
    if os.environ.get(SYNC_ENV, "").strip() not in ("1", "true", "yes", "on"):
        return None, 0
    rep = sync(root, paths=[f"lanes/{x}" for x in lanes if x], message=f"notes: {', '.join(x for x in lanes if x)}")
    return "\n".join([render_sync(rep), *refused_lines(rep)]), FUTURE_RC if rep.get("future") else 0


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sy = sub.add_parser("sync", help="commit your changes, rebase onto the remote, push (bounded retries; a conflict drops nothing; "
                                     "exit 5: lane notes stamped over 5 min ahead of the clock were left out)")
    sy.add_argument("--path", action="append", default=[], metavar="P", help="only changes under P (repeatable; default every change)")
    sy.add_argument("-m", "--message", default=None)
    sy.add_argument("--retries", type=int, default=5)
    sy.add_argument("--remote", default="origin")
    sy.add_argument("--root", type=Path, default=None)
    sc = sub.add_parser("scan", help="every blob in the notes history: files over 1 MB and key material (exit 1 if any)")
    sc.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--root", type=Path, default=None)
    ns = ap.parse_args(argv)
    root = ns.root or default_root
    if ns.cmd == "sync":
        rep = sync(root, ns.path or None, ns.message, max(0, ns.retries), ns.remote)
        print("\n".join([render_sync(rep), *refused_lines(rep)]), file=sys.stdout if rep["rc"] == 0 else sys.stderr)
        return rep["rc"]
    rep = scan(root, ns.max_bytes)
    if ns.json:
        print(json.dumps(rep, indent=1))
    else:
        for b in rep["big"]:
            print(f"BIG  {b['path']}  {b['bytes'] / 2**20:.1f} MB")
        for k in rep["keys"]:
            print(f"KEY  {k}")
        print(f"scanned {rep['blobs']} blobs: {len(rep['big'])} over {ns.max_bytes} bytes, {len(rep['keys'])} with key material")
    return 1 if rep["big"] or rep["keys"] else 0

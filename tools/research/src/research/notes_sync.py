"""`research notes sync` / `research notes scan`: the notes tree as a git repo with a remote that several machines write.

    research notes sync [--path P]... [-m MSG] [--retries N] [--remote NAME] [--root DIR]
    research notes scan [--max-bytes N] [--json] [--root DIR]

`sync` commits the caller's changes (every change, or only under `--path`; files over 1 MB, evidence and renders, and the
steward's own state stay out, as with `snapshot`), then fetches, rebases onto the remote branch, and pushes; a push that loses a race fetches, rebases and pushes
again, at most `--retries` times.  A rebase conflict never drops either side: the rebase is aborted, the local commit stays, the
exit is 3 and the conflicted files are named.  Before anything is pushed, the outgoing commits' added lines are scanned for key
material (`KEY_PATTERNS`, and the value of any secret-named environment variable); a hit refuses the push (exit 4), naming the
file.  With no remote configured the commit is local and the report says so.

A lane note (`lanes/**/*.md`) the sync would add or change stays out of the commit when its filename stamp `YYYYMMDDTHHMMZ` (a
path new since HEAD) or its front matter's `created:` (new or changed since HEAD) is more than 5 minutes ahead of the sync's
clock: one `SYNC-REFUSED <path>: ...` line each, with the value `date -u` gives now, the rest synced, exit 5 when nothing else
failed.  What HEAD already has is never re-checked, and a stamp stops being ahead once the clock passes it.

Evidence and renders never enter the notes' history: a file added or changed under `renders/`, `campaigns/*/assets/` or
`lanes/*/evidence/` stays out of the commit and is named in the report; it belongs in the evidence store (`research data put
... --preserve`), cited by its `art:` id.  Deleting one commits like any other change.

A note (`lanes/**/*.md`, `kb/**/*.md`) whose front matter is broken also stays out: opened with `---` and never closed, a line
that is not `key: value` (or a list item, an indented continuation, a comment), or a `created:` that is not a UTC time (the stamp
check above cannot read it).  Front matter is optional: a note without it is fine, since its lane, kind and time are in its path.
`SYNC-WARN <path>: ...` lines (committed anyway) name a `lane:` that is not the note's folder (a handoff's lane is its recipient)
and a `created:` more than a day from the file name's stamp.  Only content the repository has never held is checked (a move, or
an unchanged file, is not), and only for problems HEAD's copy of the path did not already have.

Once pushed, the `art:` ids and run ids that the pushed commits' added lines cite in notes are looked up on the evidence store's
remote, what every reader resolves (not this machine's catalog): an `art:` prefix must name a manifest, a run id an attempt or,
failing that, a label (a run with `custody waived` is accounted for).  Each note citing one the remote lacks gets a
`SYNC-WARN <path>: cites ...` line saying how to publish it; nothing is refused.  At most 64 ids and 20 s a sync; with no usable
remote there is no check, and `RESEARCH_NOTES_CITE_CHECK=0` turns it off.

`scan` reads every blob in the repo's history: files over `--max-bytes` and key material, each with the path it was committed
under.  Exit 1 when it finds anything; run it before the first push and after (acceptance A5).

With `RESEARCH_NOTES_SYNC=1`, `research notes checkpoint | bind | relaunch` run `sync` for the lane directories they wrote once
they succeed (a cloud agent's default; the laptop keeps the watcher's snapshot until the switch-over).

With `RESEARCH_NOTES_TOKEN` set, every fetch, pull and push of a github.com notes remote (`remote_target`: `sync`, the standby
steward's pull, budgets.toml's fetch) goes to its explicit-user URL `https://x-access-token@github.com/<owner>/<repo>.git` with a
credential helper that reads the token from the environment.  A cloud VM's global `url.<cursor[bot] token>.insteadOf
https://github.com/` matches only the plain URL, so it no longer turns the request into the bot's, which the notes repo refuses;
the token itself is never in a URL, an argument or a message.  Unset, nothing changes.  With the token set, a notes command (and
machine resolution, which reads `<notes>/machines.d`) on a machine without the notes clone clones `NOTES_URL` there first
(`ensure_clone`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .secret_hint import FRESH_SHELL, lost

SYNC_ENV = "RESEARCH_NOTES_SYNC"
TOKEN_ENV = "RESEARCH_NOTES_TOKEN"
NOTES_URL = "https://github.com/danielreuter/research-notes.git"
GITHUB_URL_RE = re.compile(r"^(?:https://(?:[^@/]+@)?github\.com/|(?:(?:git\+)?ssh://)?git@github\.com(?::22)?[:/])"
                           r"(?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?/?$")
#: the shell expands the variable when git asks for the password: the token's value is never in git's arguments or config
CREDENTIAL_HELPER = f'!f(){{ echo username=x-access-token; echo "password=${TOKEN_ENV}"; }}; f'
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
EVIDENCE_PATH = re.compile(r"^(?:renders/|campaigns/[^/]+/assets/|lanes/[^/]+/evidence/)")


def is_evidence(path: str) -> bool:
    """A path under the notes' evidence and render locations, which never enter its history."""
    return bool(EVIDENCE_PATH.match(path))
FRONT_LINE_RE = re.compile(r"^(?:[A-Za-z_][\w.-]*:(?:[ \t].*)?|[ \t]+\S.*|-(?:[ \t].*)?|#.*|[ \t]*)$")
FRONT_LANE_RE = re.compile(r"^lane:[ \t]*[\"'`]?([^\s\"'`]+)", re.M)
MAIL_RE = re.compile(r"handoff|asks|reply")
CREATED_STAMP_SLACK = dt.timedelta(days=1)
CITE_ENV = "RESEARCH_NOTES_CITE_CHECK"
CITE_ART_RE = re.compile(r"\bart:([0-9a-f]{8,64})\b")
CITE_RUN_RE = re.compile(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b")
CITE_MAX = 64
CITE_WAIT_S = 20.0
CITE_HINTS = {"run": "a run is in the store once its attempt is on the remote: `research data custody <run> --publish` where its "
                     "run dir is, else `research data label <run> custody waived --by <you>`",
              "art": "`research data push <art>` where it was put"}


def _run(root: Path, *args: str, timeout: float = 120, input: str | None = None) -> subprocess.CompletedProcess:
    """git in `root`; output that is not UTF-8 (a diff of a binary or Latin-1 file) is decoded with replacement characters."""
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, errors="replace", timeout=timeout, input=input)


def _out(root: Path, *args: str) -> str | None:
    r = _run(root, *args)
    return r.stdout if r.returncode == 0 else None


def token_url(url: str) -> tuple[list[str], str] | None:
    """(git options, explicit-user URL) that reach the github.com repository `url` names with RESEARCH_NOTES_TOKEN (module
    docstring); None when the token is unset or `url` is not github.com's."""
    m = GITHUB_URL_RE.match(url.strip())
    if not (os.environ.get(TOKEN_ENV) and m):
        return None
    return ["-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}"], f"https://x-access-token@github.com/{m['repo']}.git"


def remote_target(root: Path, remote: str = "origin") -> tuple[list[str], str]:
    """(git options, what to name as the repository) for a fetch, pull or push of the notes clone's `remote`: `token_url` of its
    URL as configured (`remote get-url` prints it rewritten by insteadOf), else ([], remote) as before."""
    url = _out(root, "config", "--get", f"remote.{remote}.url")
    return (token_url(url) if url else None) or ([], remote)


RACE = ("non-fast-forward", "fetch first", "[rejected]", "failed to update ref", "stale info")
REFUSED = ("Authentication failed", "could not read Username", "Permission denied", "denied to", "returned error: 401",
           "returned error: 403", "Repository not found", "does not appear to be a git repository")


def refused(stderr: str) -> bool:
    """Whether a failed fetch or push was refused (credentials, permissions, no such repository), which no retry changes. Anything
    else, a lost push race or a lock or transfer error two machines' syncs under load hit (6 Oct, load 156), is retried."""
    return any(k in stderr for k in REFUSED)


def token_hint(root: Path, remote: str = "origin") -> str:
    """'' unless RESEARCH_NOTES_TOKEN is empty here and `remote` is github.com's; then why git failed and what to do, for the end of
    a fetch or push failure (a laptop's own credentials need no token, so an empty one is no refusal by itself)."""
    url = _out(root, "config", "--get", f"remote.{remote}.url") or ""
    if os.environ.get(TOKEN_ENV) or not GITHUB_URL_RE.match(url.strip()):
        return ""
    return (f" -> {TOKEN_ENV} is empty in this shell, so git used the VM's own credentials, which the notes repo refuses: "
            + (lost(TOKEN_ENV) or f"set it (a Cursor secret, Cloud Agents > Secrets) or {FRESH_SHELL}"))


def pull(root: Path, remote: str = "origin") -> subprocess.CompletedProcess:
    """`git pull --ff-only` of the current branch from `remote` (`remote_target`), its remote-tracking branch updated as a plain
    pull does."""
    opts, where = remote_target(root, remote)
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()
    if not opts or not branch:
        return _run(root, "pull", "-q", "--ff-only", timeout=300)
    return _run(root, *opts, "pull", "-q", "--ff-only", where, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", timeout=300)


def ensure_clone(root: Path) -> str | None:
    """Clone NOTES_URL to `root` when `root` is missing and RESEARCH_NOTES_TOKEN is set (a fresh cloud VM), its `origin` the plain
    URL: a line saying it did, or why it failed; None when there was nothing to do."""
    root = Path(root).expanduser()
    if root.exists() or not (t := token_url(NOTES_URL)):
        return None
    opts, url = t
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.with_name(f".{root.name}.clone-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    r = subprocess.run(["git", *opts, "clone", "-q", url, str(tmp)], capture_output=True, text=True, errors="replace", timeout=900)
    if r.returncode == 0:
        _run(tmp, "remote", "set-url", "origin", NOTES_URL)
        try:
            tmp.rename(root)
        except OSError:
            pass                                            # another command cloned it first
    shutil.rmtree(tmp, ignore_errors=True)
    if not root.is_dir():
        return f"notes: clone of {NOTES_URL} into {root} failed: {r.stderr.strip()[-300:]}"
    return f"notes: cloned {NOTES_URL} into {root}"


def _secret_values(environ: dict | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return [v for k, v in env.items() if SECRET_ENV_RE.search(k) and v and len(v) >= 16]


def find_keys(text: str, secrets: list[str]) -> list[str]:
    """The names of the key patterns (and `env-secret` for a secret env var's value) that `text` contains."""
    hits = [name for name, rx in KEY_PATTERNS.items() if rx.search(text)]
    return hits + (["env-secret"] if any(s in text for s in secrets) else [])


def _changed(root: Path, paths: list[str] | None, max_bytes: int) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """(paths to commit, paths skipped as over max_bytes, evidence files left out, {staged rename's new path: its old path}) among
    the working tree's changes, limited to `paths` when given."""
    st = _out(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *(paths or []))
    entries, keep, big, evidence, renames, i = (st or "").split("\0"), [], [], [], {}, 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        code, path = e[:2], e[3:]
        if code[0] in "RC":
            keep.append(entries[i])
            if code[0] == "R":
                renames[path] = entries[i]
            i += 1
        if path in NEVER_COMMIT:
            continue
        p = root / path
        if p.is_file() and is_evidence(path):
            evidence.append(path)
            continue
        (big if p.is_file() and p.stat().st_size > max_bytes else keep).append(path)
    return keep, big, evidence, renames


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


def header_issues(path: str, text: str, now: dt.datetime) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for one note's front matter (module docstring); ([], []) for a note without front matter."""
    lines = text.split("\n")
    if lines[0].rstrip() != "---":
        return [], []
    end = next((i for i in range(1, len(lines)) if lines[i].rstrip() == "---"), None)
    if end is None:
        return ["front matter opened with --- is never closed: add a --- line under the header"], []
    refuse = [f"front matter line {n} is not `key: value`: {line.strip()[:60]!r}"
              for n, line in enumerate(lines[1:end], 2) if not FRONT_LINE_RE.match(line.rstrip("\r"))]
    block, warn, c = "\n".join(lines[1:end]), [], created_at(text)
    if (m := re.search(r"^created:[ \t]*(.*?)[ \t\r]*$", block, re.M)) and c is None:
        refuse.append(f"created: {m[1]!r} is not a UTC time; use {now:%Y-%m-%dT%H:%MZ} (date -u +%Y-%m-%dT%H:%MZ)")
    parts = path.split("/")
    if parts[0] == "lanes" and len(parts) > 2 and (m := FRONT_LANE_RE.search(block)) and m[1] != parts[1]:
        warn.append(f"lane: {m[1]} but the note is in lanes/{parts[1]}/" + (" (a handoff's lane is its recipient)" if MAIL_RE.search(parts[-1]) else ""))
    if c and (s := STAMP_RE.match(parts[-1])):
        try:
            off = c[1] - dt.datetime.strptime(s[1], "%Y%m%dT%H%MZ").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            off = dt.timedelta(0)
        if abs(off) > CREATED_STAMP_SLACK:
            warn.append(f"created: {c[0]} is {abs(off).total_seconds() / 3600:.0f} h from the file name's stamp {s[1]}")
    return refuse, warn


def _known_blobs(root: Path, paths: list[str]) -> set[str]:
    """The paths whose current content the repository already holds as a blob (a move, a revert, an unchanged file)."""
    hashes = _run(root, "hash-object", "--stdin-paths", input="\n".join(paths) + "\n").stdout.split()
    if len(hashes) != len(paths):
        return set()
    check = _run(root, "cat-file", "--batch-check", input="\n".join(hashes) + "\n").stdout.splitlines()
    missing = {line.split()[0] for line in check if line.endswith(" missing")}
    return {p for p, h in zip(paths, hashes) if h not in missing}


def header_problems(root: Path, paths: list[str], now: dt.datetime) -> tuple[dict[str, str], list[str]]:
    """({path: why} to leave out, [`path: warning`]) for the notes among `paths` with new content: the problems of
    `header_issues` that HEAD's copy of the path did not already have."""
    notes = [p for p in paths if p.startswith(("lanes/", "kb/")) and p.endswith(".md") and (root / p).is_file()]
    known = _known_blobs(root, notes) if notes else set()
    refused, warned = {}, []
    for path in notes:
        if path in known:
            continue
        try:
            refuse, warn = header_issues(path, (root / path).read_text(errors="replace"), now)
        except OSError:
            continue
        if (refuse or warn) and (old := _out(root, "show", f"HEAD:{path}")) is not None:
            old_refuse, old_warn = header_issues(path, old, now)
            refuse, warn = [x for x in refuse if x not in old_refuse], [x for x in warn if x not in old_warn]
        if refuse:
            refused[path] = "; ".join(refuse)
        warned += [f"{path}: {x}" for x in warn]
    return refused, warned


def commit(root: Path, paths: list[str] | None = None, message: str | None = None, max_bytes: int = MAX_BYTES,
           now: dt.datetime | None = None) -> dict:
    """Commit the caller's changes (all, or under `paths`) but evidence and renders (`evidence`), lane notes stamped ahead of `now`
    (`future`: `path: why`) and notes with a broken header (`header`: `path: why`); `warn` lists `path: warning`.  Retries a held
    index lock (another writer on this machine) briefly."""
    keep, big, evidence, renames = _changed(root, paths, max_bytes)
    now = now or _utcnow()
    future = future_stamps(root, keep, now)
    header, warn = header_problems(root, [p for p in keep if p not in future], now)
    if future or header or evidence:
        out = {*future, *header, *evidence}
        out |= {renames[p] for p in out if p in renames}                  # a file left out keeps its old path: never half a move
        keep = [p for p in keep if p not in out]
        _run(root, "--literal-pathspecs", "reset", "-q", "--", *sorted(out))  # a staged copy would be committed anyway
    left = {"skipped": big, "evidence": evidence, "future": [f"{p}: {why}" for p, why in future.items()],
            "header": [f"{p}: {why}" for p, why in header.items()], "warn": warn}
    if not keep:
        return {"commit": None, "paths": 0, **left}
    msg = message or f"notes sync {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}: {len(keep)} paths"
    moved_from = set(renames.values())          # a staged move's old path: its removal is staged already, and `add` finds no such path
    to_add = [p for p in keep if p not in moved_from]
    for attempt in range(10):
        add = _run(root, "--literal-pathspecs", "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(to_add)) \
            if to_add else None
        if add is None or add.returncode == 0:
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


def outgoing_citations(root: Path, base: str | None) -> dict[str, list[str]]:
    """{note path: [`art:<hex>` | run id, ...]} cited in the lines the commits after `base` (the last commit when None: a first
    push) add to notes under lanes/ and kb/; a moved note adds no lines."""
    rng = [f"{base}..HEAD"] if base else ["-1", "HEAD"]
    log = _out(root, "log", "-p", "-M", "--no-color", "--format=", "--unified=0", *rng, "--", "*.md") or ""
    out: dict[str, list[str]] = {}
    path = None
    for line in log.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") and line[6:].startswith(("lanes/", "kb/")) else None
        elif path and line.startswith("+"):
            for cited in ["art:" + m[1] for m in CITE_ART_RE.finditer(line)] + CITE_RUN_RE.findall(line):
                if cited not in out.setdefault(path, []):
                    out[path].append(cited)
    return out


def cite_check_on() -> bool:
    return os.environ.get(CITE_ENV, "").strip().lower() not in ("0", "off", "no", "false")


def missing_citations(cited: list[str], remote=None, workers: int = 8, wait_s: float = CITE_WAIT_S) -> tuple[set[str], str | None]:
    """(the ids among `cited` the store's remote does not have, why the answer is partial or None), `workers` lookups at a time
    for at most `wait_s` (daemon threads: a remote that hangs never holds up the exit).  (set(), None) with no usable remote: none
    configured, or its credentials absent here."""
    from .store.labels_sync import remote_labels_for  # noqa: PLC0415
    from .store.local import attempt_key  # noqa: PLC0415
    from .store.remote import ConfigError, load_remote  # noqa: PLC0415

    if remote is None:
        try:
            remote = load_remote()
        except ConfigError:
            return set(), None
    if remote is None or not cited:
        return set(), None

    def have(c: str) -> bool:
        if c.startswith("art:"):
            return bool(remote.list(f"manifests/{c[4:]}"))
        return remote.head(attempt_key(c)) is not None or bool(remote_labels_for(remote, c))

    todo, answers = list(cited), {}

    def work() -> None:
        while todo:
            try:
                c = todo.pop()
            except IndexError:
                return
            try:
                answers[c] = have(c)
            except Exception as e:  # noqa: BLE001
                answers[c] = e

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, min(workers, len(cited))))]
    for t in threads:
        t.start()
    deadline = time.monotonic() + wait_s
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    got = dict(answers)
    failed = [c for c in cited if isinstance(got.get(c), Exception)]
    late = [c for c in cited if c not in got]
    why = ([f"the store's remote did not answer for {len(failed)} cited ids ({type(got[failed[0]]).__name__})"] if failed else []) \
        + ([f"{len(late)} cited ids not answered within {wait_s:g} s"] if late else [])
    return {c for c in cited if got.get(c) is False}, "; ".join(why) or None


def citation_warnings(cites: dict[str, list[str]], remote=None) -> list[str]:
    """`path: cites ...` for each note among `cites` (`outgoing_citations`) citing an `art:` id or run id the store's remote does
    not have, with how to publish it; a `citation check: ...` line when the check was cut short (module docstring)."""
    ids = list(dict.fromkeys(c for v in cites.values() for c in v))
    if not ids or not cite_check_on():
        return []
    missing, why = missing_citations(ids[:CITE_MAX], remote)
    out = []
    for path, v in cites.items():
        if gone := [c for c in v if c in missing]:
            hints = [CITE_HINTS[k] for k in ("run", "art") if any(c.startswith("art:") == (k == "art") for c in gone)]
            out.append(f"{path}: cites {', '.join(gone)}, which the evidence store does not have ({'; '.join(hints)})")
    if why:
        out.append(f"citation check: {why}")
    if len(ids) > CITE_MAX:
        out.append(f"citation check: {len(ids) - CITE_MAX} more cited ids not looked up (at most {CITE_MAX} a sync)")
    return out


def sync(root: Path, paths: list[str] | None = None, message: str | None = None, retries: int = 5, remote: str = "origin",
         max_bytes: int = MAX_BYTES, secrets: list[str] | None = None, now: dt.datetime | None = None) -> dict:
    """Commit, then fetch / rebase / push with bounded retries (module docstring).  The report: `commit`, `paths`, `skipped`,
    `evidence` (evidence and renders left out), `future` (notes left out, stamped ahead of `now`), `header` (notes left out, broken
    front matter), `warn` (SYNC-WARN lines: front matter, and once pushed, citations the store lacks), `pushed` (bool), `attempts`,
    and on failure `error` plus `conflict` (files)
    or `keys` (hits); `rc` is the exit code."""
    root = Path(root)
    if _out(root, "rev-parse", "--git-dir") is None:
        return {"rc": 2, "error": f"{root} is not a git repository"}
    branch = (_out(root, "symbolic-ref", "--short", "HEAD") or "").strip()
    if not branch:
        return {"rc": 2, "pushed": False, "attempts": 0, "commit": None,
                "error": "HEAD is detached; nothing committed: check out the notes branch"}
    rep = {"rc": 0, "pushed": False, "attempts": 0, **commit(root, paths, message, max_bytes, now)}
    if rep["future"] or rep["header"]:
        rep["rc"] = FUTURE_RC
    if _out(root, "remote", "get-url", remote) is None:
        return {**rep, "note": f"no remote {remote!r}: committed locally only"}
    secrets = _secret_values() if secrets is None else secrets
    tracking = f"refs/remotes/{remote}/{branch}"
    opts, where = remote_target(root, remote)
    for n in range(retries + 1):
        rep["attempts"] = n + 1
        f = _run(root, *opts, "fetch", "-q", where, f"+refs/heads/{branch}:{tracking}", timeout=300)
        has_remote = f.returncode == 0
        if not has_remote and "couldn't find remote ref" not in f.stderr:
            if n < retries and not refused(f.stderr):
                time.sleep(min(2.0, 0.1 * 2 ** n))
                continue
            return {**rep, "rc": 1, "error": f"git fetch {remote} failed: {f.stderr.strip()[-300:]}{token_hint(root, remote)}"}
        base = tracking if has_remote else None
        if has_remote and _run(root, "merge-base", "--is-ancestor", tracking, "HEAD").returncode != 0:
            rb = _run(root, "rebase", "--autostash", tracking, timeout=300)
            if rb.returncode:
                conflict = [x for x in (_out(root, "diff", "--name-only", "--diff-filter=U") or "").splitlines() if x]
                _run(root, "rebase", "--abort")
                if not _out(root, "symbolic-ref", "-q", "HEAD"):
                    # an abort that lost its rebase-merge directory (NFS, 3 and 6 Oct) leaves HEAD detached at the remote branch
                    _run(root, "checkout", "-q", branch)
                return {**rep, "rc": 3, "conflict": conflict,
                        "error": "rebase conflict; nothing was dropped (the local commit is kept, the rebase aborted): resolve "
                                 + (", ".join(conflict) or rb.stderr.strip()[-200:]) + ", then sync again"}
        if base and (_out(root, "rev-parse", "HEAD") or "").strip() == (_out(root, "rev-parse", tracking) or "").strip():
            return {**rep, "pushed": False, "note": "already on the remote"}
        keys = outgoing_keys(root, base, secrets)
        if keys:
            return {**rep, "rc": 4, "keys": keys, "error": "key material in outgoing commits; nothing pushed: " + "; ".join(keys[:5])}
        cites = outgoing_citations(root, base) if cite_check_on() else {}
        p = _run(root, *opts, "push", "-q", where, f"HEAD:refs/heads/{branch}", timeout=300)
        if p.returncode == 0:
            _run(root, *opts, "fetch", "-q", where, f"+refs/heads/{branch}:{tracking}", timeout=300)
            return {**rep, "pushed": True, "warn": rep["warn"] + citation_warnings(cites)}
        if refused(p.stderr) or n == retries and not any(k in p.stderr for k in RACE):
            return {**rep, "rc": 1, "error": f"git push failed (local commit kept): {p.stderr.strip()[-300:]}{token_hint(root, remote)}"}
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
    if rep.get("evidence"):
        parts.append(f"left out {len(rep['evidence'])} evidence/render files (the evidence store holds them: research data put ... "
                     f"--preserve, cite the art: id): {', '.join(rep['evidence'][:3])}")
    if rep.get("future"):
        parts.append(f"left out {len(rep['future'])} stamped ahead of the clock (SYNC-REFUSED)")
    if rep.get("header"):
        parts.append(f"left out {len(rep['header'])} with broken front matter (SYNC-REFUSED)")
    if rep.get("warn"):
        parts.append(f"{len(rep['warn'])} warning{'s' if len(rep['warn']) > 1 else ''} (SYNC-WARN)")
    if rep.get("note"):
        parts.append(rep["note"])
    if rep.get("error"):
        parts.append(f"FAILED: {rep['error']}")
    return "; ".join(parts)


def refused_lines(rep: dict) -> list[str]:
    """`SYNC-REFUSED` for each note left out, then `SYNC-WARN` for each note committed with a warning."""
    return [f"SYNC-REFUSED {x}" for x in rep.get("future", []) + rep.get("header", [])] + [f"SYNC-WARN {x}" for x in rep.get("warn", [])]


def auto_sync(root: Path, lanes: list[str]) -> tuple[str | None, int]:
    """After a notes write command: `sync` the lane directories it wrote, when `RESEARCH_NOTES_SYNC=1`.  (report lines, FUTURE_RC
    when a note was left out for its stamp or its front matter, else 0; other failures stay advisory, the next sync retries them);
    (None, 0) when off."""
    if os.environ.get(SYNC_ENV, "").strip() not in ("1", "true", "yes", "on"):
        return None, 0
    rep = sync(root, paths=[f"lanes/{x}" for x in lanes if x], message=f"notes: {', '.join(x for x in lanes if x)}")
    return "\n".join([render_sync(rep), *refused_lines(rep)]), FUTURE_RC if rep.get("future") or rep.get("header") else 0


def main(argv: list[str], default_root: Path) -> int:
    ap = argparse.ArgumentParser(prog="research notes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sy = sub.add_parser("sync", help="commit your changes, rebase onto the remote, push (bounded retries; a conflict drops nothing; "
                                     "exit 5: lane notes stamped over 5 min ahead of the clock, or with broken front matter, were left out)")
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

"""`cluster models`: one model-checkpoint cache per node, at `/workspace/models` (`models.toml`'s `root`).

**Layout.** An entry is `<root>/<org>--<name>@<revision>/`, a Hugging Face repo's files at one commit (40 hex digits) as
plain files, and its `.entry.json` (`verity/model-entry/v1`): repo, revision, each file's sha256 and bytes, the total,
owner, phase, declared_gb, expires_h, created, last_used, expires (ISO UTC) and fetched_by (the run). `.entry.json` is
written last, atomically, so a directory without one is incomplete and no reader sees it. The entries are the registry.
A fetch streams into `<root>/.tmp/<entry>-<pid>/` on the same filesystem, holding `<root>/.locks/<entry>.lock`
(`flock`) throughout, so two fetches of one model wait for each other and the second finds the first's entry.

**Fetch** (`fetch`, the `model.fetch` kind) takes a pinned revision or nothing, and lists the files from Hugging Face's
metadata (`/api/models/<repo>/revision/<rev>?blobs=true`). It refuses a file the metadata gives no checksum (an LFS file's
`lfs.sha256`, else its git blob id), a listed total over the job's `declared_gb`, an owner's entries on the node past its
quota with this one, and a filesystem fuller than `max_use_pct` after it (counting other fetches still running). It
downloads each file from `/<repo>/resolve/<rev>/<path>`, resumes a partial one with HTTP Range, and checks it as it writes:
an LFS file's sha256 against `lfs.sha256`, any other's git blob sha1 (`sha1(b"blob <n>\\0" + data)`) against its blob id,
recording every file's sha256. A mismatch deletes the download. An existing entry of the same owner is only bumped.
The token at `HF_TOKEN_PATH` (default /etc/verity/hf-token), when readable, goes to huggingface.co alone, never to the CDN
it redirects to; it is never printed.

**Lifecycle.** A job calls `use` at its start, which sets `last_used` and `expires = last_used + expires_h`. The owner
`release`s an entry when its phase ends; otherwise `evict` deletes it once past `expires`. Neither deletes an entry a
process has open or mapped (`/proc/<pid>/fd`, `/proc/<pid>/maps`, by device and inode, so a container's view counts too);
`evict` sees every process only as root, and refuses to delete anything when it can't. Both remove `.entry.json` first.
Times are passed in (`now`, epoch seconds); only `main` reads a clock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from .model import instant

SCHEMA = "verity/model-entry/v1"
CONFIG = Path(__file__).parents[2] / "models.toml"
ENTRY = ".entry.json"
FETCH = ".fetch.json"
HF = "https://huggingface.co"
HF_HOST = "huggingface.co"
TOKEN_PATH = "/etc/verity/hf-token"
GB = 10**9
STALE_S = 86400
CHUNK = 8 << 20
TRIES = 8
DEFAULTS = {"root": "/workspace/models", "expires_h": 72, "max_use_pct": 85, "quota_gb": {"default": 200}}

REV = SHA1 = re.compile(r"[0-9a-f]{40}")
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
OWNER = re.compile(r"[a-z0-9][a-z0-9_.-]*")


class Refused(Exception):
    """A fetch or command refused before it changed anything: exit 2 (the request is malformed) or 3 (a rule refuses it)."""
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class Failed(Exception):
    """It went wrong on the way: a download, a checksum. Exit 1."""


class Mismatch(Failed):
    pass


class HubError(Exception):
    def __init__(self, status: int | None, msg: str):
        super().__init__(msg)
        self.status = status


@dataclass(frozen=True)
class Ref:
    repo: str
    revision: str

    @property
    def name(self) -> str:
        return f"{self.repo.replace('/', '--')}@{self.revision}"

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}"


def ref(s: str) -> Ref:
    """`ORG/NAME@REV`, REV a full 40-hex commit: pinned or nothing."""
    repo, at, rev = s.rpartition("@")
    if not at:
        raise Refused(2, f"{s}: give REPO@REV with REV a full 40-hex commit (pinned or nothing)")
    org, slash, name = repo.partition("/")
    if not (slash and PART.fullmatch(org) and PART.fullmatch(name)) or "--" in repo or ".." in repo or len(repo) > 96:
        raise Refused(2, f"{s}: {repo!r} isn't a Hugging Face repo id ORG/NAME")
    if not REV.fullmatch(rev):
        raise Refused(2, f"{s}: {rev!r} isn't a commit: give its full 40 hex digits, not a branch or tag (pinned or nothing)")
    return Ref(repo, rev)


def iso(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- configuration -----------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_config(path: Path = CONFIG) -> dict[str, Any]:
    if not Path(path).is_file():
        raise Refused(2, f"no cache configuration at {path}: ship tools/cluster/models.toml with the source, or pass --config")
    return tomllib.loads(Path(path).read_text())


def node_config(cfg: dict[str, Any], *, node: str | None = None, hostname: str | None = None) -> dict[str, Any]:
    """The settings for one node, found by name or hostname: its `[nodes.<name>]` table over the top level over DEFAULTS."""
    nodes = cfg.get("nodes", {})
    name = node if node in nodes else next((n for n, d in nodes.items() if hostname and d.get("hostname") == hostname), None)
    over = nodes.get(name, {})
    out = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "nodes"}, **over, "node": name}
    out["quota_gb"] = {**DEFAULTS["quota_gb"], **cfg.get("quota_gb", {}), **over.get("quota_gb", {})}
    return out


def quota_gb(cfg: dict[str, Any], owner: str) -> float:
    q = cfg["quota_gb"]
    return q.get(owner, q["default"])


# --- reading the cache -------------------------------------------------------------------------------------------------


def read_entry(d: Path) -> dict[str, Any] | None:
    """`d`'s entry when it is complete (its `.entry.json` of this schema, naming `d`), else None."""
    try:
        e = json.loads((d / ENTRY).read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None
    if e.get("schema") != SCHEMA or Ref(e.get("repo", ""), e.get("revision", "")).name != d.name:
        return None
    return e


def entries(root: Path) -> dict[str, dict[str, Any]]:
    """Every complete entry under `root`, by directory name."""
    out = {}
    for d in sorted(Path(root).iterdir()) if Path(root).is_dir() else ():
        if not d.name.startswith(".") and d.is_dir() and (e := read_entry(d)) is not None:
            out[d.name] = e
    return out


def parse_entries(text: str) -> list[dict[str, Any]]:
    """The entries in `text`, `.entry.json` files one after another (as `cluster status` reads them)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text):
            return out
        try:
            e, i = dec.raw_decode(text, i)
        except ValueError:
            return out
        if isinstance(e, dict) and e.get("schema") == SCHEMA:
            out.append(e)


def _tree(d: Path) -> tuple[int, int]:
    """Bytes under `d` and its newest mtime."""
    size, newest = 0, int(d.stat().st_mtime)
    for top, _, files in os.walk(d):
        for f in files:
            with contextlib.suppress(FileNotFoundError):
                st = os.lstat(os.path.join(top, f))
                size, newest = size + st.st_size, max(newest, int(st.st_mtime))
    return size, newest


def _held(root: Path, name: str) -> bool:
    """Whether a fetch, use, release or evict holds `name`'s lock now."""
    try:
        fd = os.open(Path(root) / ".locks" / f"{name}.lock", os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def incomplete(root: Path) -> list[dict[str, Any]]:
    """Downloads in `.tmp/` and entry directories without `.entry.json`: path, entry, bytes so far, newest mtime, whether
    a fetch holds the entry's lock, and the fetch's own record (owner, total) when it wrote one."""
    root, out = Path(root), []
    tmp = sorted((root / ".tmp").iterdir()) if (root / ".tmp").is_dir() else []
    bare = [d for d in sorted(root.iterdir()) if not d.name.startswith(".") and d.is_dir() and read_entry(d) is None] \
        if root.is_dir() else []
    for d in [*tmp, *bare]:
        name = d.name.rpartition("-")[0] if d.parent.name == ".tmp" else d.name
        try:
            rec = json.loads((d / FETCH).read_text())
        except (FileNotFoundError, NotADirectoryError, ValueError):
            rec = {}
        try:
            size, newest = _tree(d)
        except FileNotFoundError:
            continue
        out.append({"path": str(d), "entry": name, "bytes": size, "mtime": newest, "running": _held(root, name),
                    "owner": rec.get("owner"), "total_bytes": rec.get("total_bytes")})
    return out


# --- Hugging Face ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Remote:
    path: str
    size: int
    sha256: str | None
    blob: str | None


def _safe(p: str) -> bool:
    parts = p.split("/")
    return bool(p) and "\\" not in p and "\0" not in p and all(x not in ("", ".", "..") for x in parts) \
        and parts[0] not in (ENTRY, FETCH, ".cache")


def listing(meta: dict[str, Any], r: Ref, include: tuple[str, ...] = ()) -> list[Remote]:
    """The files of `r` that `include` selects (all when empty), each with its size and checksum; Refused for a file
    without one, an unsafe path, or metadata of another commit."""
    if meta.get("sha") != r.revision:
        raise Refused(3, f"{r}: Hugging Face's metadata is of commit {meta.get('sha')!r}, not {r.revision}")
    out = []
    for s in meta.get("siblings") or ():
        p = s.get("rfilename", "")
        if include and not any(fnmatch.fnmatchcase(p, g) for g in include):
            continue
        if not _safe(p):
            raise Refused(3, f"{r}: refusing the file path {p!r}")
        size, lfs = s.get("size"), s.get("lfs") or {}
        if not isinstance(size, int) or size < 0:
            raise Refused(3, f"{r}: {p} has no size in Hugging Face's metadata")
        if lfs:
            if not re.fullmatch(r"[0-9a-f]{64}", str(lfs.get("sha256", ""))) or lfs.get("size", size) != size:
                raise Refused(3, f"{r}: {p} is an LFS file without a sha256 of its {size} bytes in Hugging Face's metadata")
            out.append(Remote(p, size, lfs["sha256"], None))
        elif SHA1.fullmatch(str(s.get("blobId", ""))):
            out.append(Remote(p, size, None, s["blobId"]))
        else:
            raise Refused(3, f"{r}: {p} has no checksum in Hugging Face's metadata (no lfs.sha256, no blobId)")
    if not out:
        raise Refused(3, f"{r}: no files" + (f" match {' '.join(include)}" if include else ""))
    return sorted(out, key=lambda f: f.path)


class _Redirect(urllib.request.HTTPRedirectHandler):
    """Follows Hugging Face's redirects, carrying the token only to huggingface.co (a relative redirect), never the CDN."""
    def __init__(self, auth: str | None):
        self.auth = auth

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and self.auth and urllib.parse.urlsplit(newurl).hostname == HF_HOST:
            new.add_unredirected_header("Authorization", self.auth)
        return new


class Hub:
    """Hugging Face over HTTPS, stdlib only. `authenticated` says whether it sends a token."""
    def __init__(self, token: str | None = None, endpoint: str = HF, timeout_s: float = 60):
        self._auth = f"Bearer {token}" if token else None
        self.authenticated = bool(token)
        self.endpoint, self.timeout_s = endpoint.rstrip("/"), timeout_s
        self._open = urllib.request.build_opener(_Redirect(self._auth)).open

    def _get(self, url: str, headers: dict[str, str]) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "verity-cluster-models/1", **headers})
        if self._auth:
            req.add_unredirected_header("Authorization", self._auth)
        try:
            return self._open(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            e.close()
            raise HubError(e.code, f"HTTP {e.code} from {urllib.parse.urlsplit(e.url or url).hostname}") from None

    def metadata(self, repo: str, rev: str) -> dict[str, Any]:
        with self._get(f"{self.endpoint}/api/models/{repo}/revision/{rev}?blobs=true", {"Accept": "application/json"}) as r:
            return json.loads(r.read())

    def get(self, repo: str, rev: str, path: str, start: int) -> tuple[int, BinaryIO]:
        """The file from byte `start`, as (the offset the body starts at, the body): 0 when the server ignored the range."""
        r = self._get(f"{self.endpoint}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}",
                      {"Range": f"bytes={start}-"} if start else {})
        if start and r.status == 206:
            m = re.match(r"bytes (\d+)-", r.headers.get("Content-Range", ""))
            return (int(m.group(1)) if m else 0), r
        return 0, r


def token(path: str) -> tuple[str | None, str]:
    """The token at `path` and why there is none: "not installed" or "not readable by this user"."""
    try:
        return Path(path).read_text().strip() or None, "empty"
    except FileNotFoundError:
        return None, "not installed"
    except (PermissionError, IsADirectoryError):
        return None, "not readable by this user"


TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, ssl.SSLError)


class _Stopped(Exception):
    pass


def _hashers(f: Remote) -> tuple[Any, Any]:
    return hashlib.sha256(), (hashlib.sha1(b"blob %d\0" % f.size) if f.blob else None)


def _download(hub: Any, r: Ref, f: Remote, dest: Path, stop: threading.Event, sleep: Callable[[float], None],
              tries: int = TRIES) -> dict[str, Any]:
    """`f` at `dest`, resumed from what is there, hashed as it is written and checked against Hugging Face's checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(dest, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+b") as out:
        have = os.fstat(fd).st_size
        if have > f.size:
            out.truncate(0)
            have = 0
        h256, h1 = _hashers(f)
        left = have
        while left:
            b = out.read(min(CHUNK, left))
            h256.update(b)
            if h1:
                h1.update(b)
            left -= len(b)
        failures = 0
        while have < f.size:
            if stop.is_set():
                raise _Stopped
            try:
                offset, body = hub.get(r.repo, r.revision, f.path, have)
                with contextlib.closing(body):
                    if offset != have:
                        out.seek(0)
                        out.truncate(0)
                        have, (h256, h1) = 0, _hashers(f)
                    out.seek(have)
                    while have < f.size:
                        b = body.read(min(CHUNK, f.size - have))
                        if not b:
                            break
                        out.write(b)
                        h256.update(b)
                        if h1:
                            h1.update(b)
                        have += len(b)
                        if stop.is_set():
                            raise _Stopped
                    if have == f.size and body.read(1):
                        raise Mismatch(f"{f.path}: longer than the {f.size} bytes Hugging Face lists")
                if have < f.size:
                    raise ConnectionError(f"the stream ended at byte {have} of {f.size}")
            except HubError as e:
                if e.status in (401, 403):
                    raise
                if e.status is not None and e.status < 500 and e.status != 429:
                    raise Failed(f"{f.path}: {e}") from None
                failures = _retry(f, e, failures, tries, sleep)
            except TRANSIENT as e:
                failures = _retry(f, e, failures, tries, sleep)
        out.flush()
        os.fsync(out.fileno())
    got = h256.hexdigest()
    if f.sha256 and got != f.sha256:
        raise Mismatch(f"{f.path}: sha256 {got}, not Hugging Face's {f.sha256}")
    if h1 and h1.hexdigest() != f.blob:
        raise Mismatch(f"{f.path}: git blob {h1.hexdigest()}, not Hugging Face's {f.blob}")
    return {"sha256": got, "bytes": f.size}


def _retry(f: Remote, e: Exception, failures: int, tries: int, sleep: Callable[[float], None]) -> int:
    failures += 1
    if failures >= tries:
        raise Failed(f"{f.path}: {e} ({failures} tries; the partial download stays for the next fetch to resume)")
    sleep(min(60, 2 ** failures))
    return failures


def _download_all(hub: Any, r: Ref, files: list[Remote], tmp: Path, sleep: Callable[[float], None], jobs: int,
                  say: Callable[[str], None]) -> dict[str, dict[str, Any]]:
    stop, done, errors = threading.Event(), {}, []
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = {ex.submit(_download, hub, r, f, tmp / f.path, stop, sleep): f for f in sorted(files, key=lambda f: -f.size)}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                done[f.path] = fut.result()
                say(f"  {f.path}: {f.size / GB:.3f} GB, verified")
            except _Stopped:
                pass
            except Exception as e:
                stop.set()
                errors.append(e)
    if errors:
        raise next((e for e in errors if isinstance(e, Mismatch)), errors[0])
    return done


# --- writing the cache -------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _lock(root: Path, name: str, say: Callable[[str], None] | None = None, wait: bool = True) -> Iterator[bool]:
    """`name`'s lock, held for the block: True, or False at once when another holds it and `wait` is False."""
    locks = Path(root) / ".locks"
    locks.mkdir(exist_ok=True)
    fd = os.open(locks / f"{name}.lock", os.O_RDWR | os.O_CREAT, 0o664)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got = True
        except BlockingIOError:
            got = False
        if not got and wait:
            if say:
                say(f"waiting for {name}'s lock (another fetch, use or release of it is running)")
            fcntl.flock(fd, fcntl.LOCK_EX)
            got = True
        yield got
    finally:
        os.close(fd)


def _probe(root: Path, name: str, dry_run: bool) -> contextlib.AbstractContextManager[bool]:
    """For evict: the lock without waiting, or in a dry run only whether it is free, taking nothing."""
    return contextlib.nullcontext(not _held(root, name)) if dry_run else _lock(root, name, wait=False)


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_entry(d: Path, e: dict[str, Any]) -> None:
    """`d/.entry.json`, atomically: a temporary file beside it, fsynced, renamed over it."""
    tmp = d / f"{ENTRY}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(e, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, d / ENTRY)
    _fsync_dir(d)


def _remove(d: Path) -> None:
    """An entry gone: its `.entry.json` first, so readers stop seeing it, then its files."""
    with contextlib.suppress(FileNotFoundError):
        (d / ENTRY).unlink()
        _fsync_dir(d)
    shutil.rmtree(d, ignore_errors=True)


def _adopt(root: Path, r: Ref) -> Path:
    """This fetch's download directory, `.tmp/<entry>-<pid>`: an abandoned one of the same entry taken over (the lock is
    held, so no other fetch of it runs), the largest if several, the rest removed."""
    tmp = Path(root) / ".tmp"
    tmp.mkdir(exist_ok=True)
    mine = tmp / f"{r.name}-{os.getpid()}"
    old = [d for d in tmp.iterdir() if d.name.rpartition("-")[0] == r.name and d != mine and d.is_dir()]
    old.sort(key=lambda d: -_tree(d)[0])
    if old and not mine.exists():
        os.rename(old.pop(0), mine)
    for d in old:
        shutil.rmtree(d, ignore_errors=True)
    mine.mkdir(exist_ok=True)
    return mine


def _disk(path: Path) -> tuple[int, int]:
    """Used and available bytes of `path`'s filesystem, as `df` counts them."""
    s = os.statvfs(path)
    return (s.f_blocks - s.f_bfree) * s.f_frsize, s.f_bavail * s.f_frsize


def _gated(why: str, hub: Any, r: Ref, token_path: str, status: int | None = None) -> Refused:
    if not hub.authenticated:
        return Refused(3, f"gated repo: needs the HF token at {token_path} ({why})")
    return Refused(3, f"gated repo: the HF token at {token_path} has no access to {r.repo}"
                      + (f" (HTTP {status})" if status else "") + ": accept its terms on huggingface.co for the token's account")


def fetch(root: Path, r: Ref, *, owner: str, phase: str, declared_gb: float, now: int, hub: Any, cfg: dict[str, Any],
          expires_h: float | None = None, include: tuple[str, ...] = (), fetched_by: str | None = None,
          disk: Callable[[Path], tuple[int, int]] = _disk, sleep: Callable[[float], None] = time.sleep, jobs: int = 4,
          token_path: str = TOKEN_PATH, token_why: str = "not installed",
          say: Callable[[str], None] = lambda s: None) -> tuple[dict[str, Any], str]:
    """`r` cached and verified under `root` for `owner`: (its entry, "fetched" | "cached"). Refused or Failed otherwise
    (the module docstring's rules)."""
    if not OWNER.fullmatch(owner):
        raise Refused(2, f"owner {owner!r}: a lane or workstream name (lowercase letters, digits, '-', '_', '.')")
    if not phase or "\n" in phase:
        raise Refused(2, "give the phase the model serves (--phase)")
    if not declared_gb > 0:
        raise Refused(2, "give the disk the fetch asks for in GB (--declared-gb), more than 0")
    root = Path(root)
    try:
        root.mkdir(exist_ok=True)
    except (PermissionError, FileNotFoundError):
        raise Refused(3, f"no cache at {root}, and this user can't create it: sudo install -d -o $USER -m 0755 {root}") from None
    h = float(expires_h if expires_h is not None else cfg["expires_h"])
    include = tuple(sorted(set(include)))

    def files_of() -> list[Remote]:
        try:
            meta = hub.metadata(r.repo, r.revision)
        except HubError as e:
            if e.status in (401, 403):
                raise _gated(token_why, hub, r, token_path, e.status) from None
            if e.status == 404:
                raise Refused(3, f"{r}: Hugging Face has no such repo or commit (HTTP 404)") from None
            raise Failed(f"{r}: reading Hugging Face's metadata: {e}") from None
        except TRANSIENT as e:
            raise Failed(f"{r}: reading Hugging Face's metadata: {e}") from None
        if meta.get("gated") and not hub.authenticated:
            raise _gated(token_why, hub, r, token_path)
        return listing(meta, r, include)

    with _lock(root, r.name, say):
        dest = root / r.name
        if (e := read_entry(dest)) is not None:
            if e["owner"] != owner:
                raise Refused(3, f"{r} is cached here for {e['owner']}, not {owner}: it is theirs to share or release")
            if tuple(e.get("include") or ()) != include:
                missing = [f.path for f in files_of() if f.path not in e["files"]]
                if missing:
                    raise Refused(3, f"{r} is cached with other files ({len(missing)} asked for aren't in it, such as "
                                     f"{missing[0]}): release it and fetch again")
            lost = [p for p, m in e["files"].items() if not (dest / p).is_file() or (dest / p).stat().st_size != m["bytes"]]
            if not lost:
                e.update(phase=phase, last_used=iso(now), expires_h=h, expires=iso(now + int(h * 3600)))
                write_entry(dest, e)
                return e, "cached"
            say(f"{r}: {len(lost)} of its files are missing or the wrong size ({lost[0]}): fetching it again")
            _remove(dest)
        files = files_of()
        total = sum(f.size for f in files)
        if total > declared_gb * GB:
            raise Refused(3, f"{r}: Hugging Face lists {total / GB:.2f} GB in {len(files)} files, over the declared {declared_gb:g} GB")
        tmp = _adopt(root, r)
        have = sum(min((tmp / f.path).stat().st_size, f.size) for f in files if (tmp / f.path).is_file())
        others = [x for x in incomplete(root) if x["running"] and x["entry"] != r.name and x["total_bytes"] is not None]
        mine = [x for n, x in entries(root).items() if x["owner"] == owner and n != r.name]
        held = sum(x["total_bytes"] for x in mine) + sum(x["total_bytes"] for x in others if x["owner"] == owner)
        quota = quota_gb(cfg, owner)
        used, avail = disk(root)
        need = total - have + sum(max(0, x["total_bytes"] - x["bytes"]) for x in others)
        pct = 100 * (used + need) / (used + avail) if used + avail else 100.0
        why = (f"{owner}'s entries on this node would hold {(held + total) / GB:.1f} GB with this one ({total / GB:.1f} GB), "
               f"over its quota of {quota:g} GB: release one first (cluster models ls)" if held + total > quota * GB
               else f"{root}'s filesystem would be {pct:.1f}% used after this fetch ({need / GB:.1f} GB more, other fetches' "
                    f"included), over its max of {cfg['max_use_pct']}%" if pct > cfg["max_use_pct"] else None)
        if why:
            if not have:
                shutil.rmtree(tmp, ignore_errors=True)
            raise Refused(3, why)
        (tmp / FETCH).write_text(json.dumps({"repo": r.repo, "revision": r.revision, "owner": owner, "phase": phase,
                                             "total_bytes": total, "pid": os.getpid()}, sort_keys=True))
        say(f"fetching {r}: {len(files)} files, {total / GB:.3f} GB ({have / GB:.3f} GB already here) into {tmp}")
        try:
            done = _download_all(hub, r, files, tmp, sleep, jobs, say)
        except Mismatch as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise Mismatch(f"{r}: {e}; deleted the download") from None
        except HubError as e:
            raise _gated(token_why, hub, r, token_path, e.status) from None
        wanted = {f.path for f in files}
        for top, dirs, names in os.walk(tmp, topdown=False):
            for n in names:
                p = os.path.join(top, n)
                if os.path.relpath(p, tmp) not in wanted:
                    os.unlink(p)
                else:
                    os.chmod(p, 0o644)
            for n in dirs:
                p = os.path.join(top, n)
                if not os.listdir(p):
                    os.rmdir(p)
                else:
                    os.chmod(p, 0o755)
        os.chmod(tmp, 0o755)
        if dest.exists():
            shutil.rmtree(dest)
        os.rename(tmp, dest)
        e = {"schema": SCHEMA, "repo": r.repo, "revision": r.revision, "files": {p: done[p] for p in sorted(done)},
             "total_bytes": total, "owner": owner, "phase": phase, "declared_gb": declared_gb, "expires_h": h,
             "include": list(include) or None, "created": iso(now), "last_used": iso(now),
             "expires": iso(now + int(h * 3600)), "fetched_by": fetched_by}
        write_entry(dest, e)
        return e, "fetched"


def use(root: Path, r: Ref, now: int) -> dict[str, Any]:
    """A job starts on `r`: `last_used` is now and `expires` is `expires_h` later."""
    root = Path(root)
    if read_entry(root / r.name) is None:
        raise Refused(3, f"{r} isn't cached on this node: fetch it first (the model.fetch kind)")
    with _lock(root, r.name):
        if (e := read_entry(root / r.name)) is None:
            raise Refused(3, f"{r} was released while this waited for its lock")
        e.update(last_used=iso(now), expires=iso(now + int(float(e["expires_h"]) * 3600)))
        write_entry(root / r.name, e)
    return e


@dataclass
class Scan:
    """Which files processes hold open or mapped: by (device, inode), and by path as each process names it."""
    inodes: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    paths: dict[str, set[int]] = field(default_factory=dict)
    hidden: list[int] = field(default_factory=list)

    def holders(self, d: Path) -> set[int]:
        pids: set[int] = set()
        prefix = str(d).rstrip("/") + "/"
        for top, _, files in os.walk(d):
            for f in files:
                with contextlib.suppress(FileNotFoundError):
                    st = os.lstat(os.path.join(top, f))
                    pids |= self.inodes.get((st.st_dev, st.st_ino), set())
        for p, ps in self.paths.items():
            if p.startswith(prefix):
                pids |= ps
        return pids


def scan(proc: Path = Path("/proc")) -> Scan:
    """Every process's open files (`fd/`) and mapped files (`maps`); a process whose either this user can't read is hidden."""
    s = Scan()
    for d in Path(proc).iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        try:
            for fd in os.scandir(d / "fd"):
                with contextlib.suppress(FileNotFoundError):
                    target = os.readlink(fd.path)
                    if target.startswith("/"):
                        st = os.stat(fd.path)
                        s.inodes.setdefault((st.st_dev, st.st_ino), set()).add(pid)
                        s.paths.setdefault(target, set()).add(pid)
            for line in (d / "maps").read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[4] != "0":
                    major, _, minor = parts[3].partition(":")
                    s.inodes.setdefault((os.makedev(int(major, 16), int(minor, 16)), int(parts[4])), set()).add(pid)
                    s.paths.setdefault(parts[5], set()).add(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            s.hidden.append(pid)
    return s


def release(root: Path, r: Ref, owner: str, scan_: Callable[[], Scan] = scan) -> dict[str, Any]:
    """The owner deletes its entry now; Refused for another owner's, or while a process it can see holds a file of it."""
    root = Path(root)
    if read_entry(root / r.name) is None:
        raise Refused(3, f"{r} isn't cached on this node")
    with _lock(root, r.name):
        if (e := read_entry(root / r.name)) is None:
            raise Refused(3, f"{r} was released while this waited for its lock")
        if e["owner"] != owner:
            raise Refused(3, f"{r} is {e['owner']}'s: only its owner releases it (else it expires {e['expires']})")
        if pids := scan_().holders(root / r.name):
            raise Refused(3, f"{r} is in use by pid {', '.join(map(str, sorted(pids)))}: release it when they are done")
        _remove(root / r.name)
    return e


def evict(root: Path, now: int, scan_: Callable[[], Scan] = scan, dry_run: bool = False,
          stale_s: int = STALE_S) -> dict[str, Any]:
    """Entries past `expires`, and downloads and incomplete entries untouched for `stale_s` that no fetch holds, deleted,
    except what a process holds open or mapped. Refused, deleting nothing, when some processes' files can't be seen."""
    root = Path(root)
    due = [n for n, e in entries(root).items() if instant(e["expires"]) <= now]
    stale = [x for x in incomplete(root) if not x["running"] and now - x["mtime"] > stale_s]
    rep: dict[str, Any] = {"root": str(root), "now": iso(now), "dry_run": dry_run, "hidden": 0, "deleted": [], "kept": []}
    if not due and not stale:
        return rep
    s = scan_()
    rep["hidden"] = len(s.hidden)
    if s.hidden and not dry_run:
        raise Refused(3, f"can't see {len(s.hidden)} processes' open and mapped files, so deleting nothing: run evict as root "
                         "(sudo -n)")
    gone = rep["deleted"]
    for n in due:
        with _probe(root, n, dry_run) as got:
            e = read_entry(root / n) if got else None
            why = ("locked: a fetch, use or release of it is running" if not got else "released meanwhile" if e is None
                   else f"used meanwhile, expires {e['expires']}" if instant(e["expires"]) > now else None)
            if why is None and (pids := s.holders(root / n)):
                why = f"in use by pid {', '.join(map(str, sorted(pids)))}"
            if why:
                rep["kept"].append({"path": str(root / n), "why": why})
                continue
            gone.append({"path": str(root / n), "why": f"expired {e['expires']}", "bytes": e["total_bytes"],
                         "owner": e["owner"]})
            if not dry_run:
                _remove(root / n)
    for x in stale:
        with _probe(root, x["entry"], dry_run) as got:
            pids = s.holders(Path(x["path"])) if got else set()
            if not got or pids:
                rep["kept"].append({"path": x["path"], "why": "locked: a fetch of it is running" if not got
                                    else f"in use by pid {', '.join(map(str, sorted(pids)))}"})
                continue
            gone.append({"path": x["path"], "why": f"incomplete, untouched since {iso(x['mtime'])}", "bytes": x["bytes"],
                         "owner": x["owner"]})
            if not dry_run:
                shutil.rmtree(x["path"], ignore_errors=True)
    return rep


# --- reports -----------------------------------------------------------------------------------------------------------


def summary(es: list[dict[str, Any]], cfg: dict[str, Any], now: int, root: str | None = None) -> dict[str, Any]:
    """A node's cache in a line's numbers: entries, GB, each owner's GB against its quota, the next expiry."""
    owners: dict[str, dict[str, Any]] = {}
    for e in es:
        o = owners.setdefault(e["owner"], {"entries": 0, "bytes": 0})
        o["entries"] += 1
        o["bytes"] += e["total_bytes"]
    for o, d in owners.items():
        d.update(gb=round(d["bytes"] / GB, 1), quota_gb=quota_gb(cfg, o))
    ends = [instant(e["expires"]) for e in es]
    total = sum(e["total_bytes"] for e in es)
    return {"root": root or cfg["root"], "entries": len(es), "bytes": total, "gb": round(total / GB, 1),
            "owners": dict(sorted(owners.items())), "expired": sum(1 for t in ends if t <= now),
            "next_expiry": iso(min(t for t in ends if t > now)) if any(t > now for t in ends) else None,
            "items": [{"repo": e["repo"], "revision": e["revision"], "owner": e["owner"], "phase": e["phase"],
                       "gb": round(e["total_bytes"] / GB, 2), "last_used": e["last_used"], "expires": e["expires"]}
                      for e in sorted(es, key=lambda e: (e["owner"], e["repo"]))]}


def line(s: dict[str, Any]) -> str:
    """`models: 3 entries, 142.0 GB (circuits 120.3/500 GB, infra 21.7/200 GB), next expiry 2026-10-03T19:00Z`"""
    if not s["entries"]:
        return f"models: no entries in {s['root']}"
    own = ", ".join(f"{o} {d['gb']:.1f}/{d['quota_gb']:g} GB" for o, d in s["owners"].items())
    nxt = f", next expiry {s['next_expiry'][:16]}Z" if s["next_expiry"] else ""
    past = f"; {s['expired']} past expiry, awaiting evict" if s["expired"] else ""
    return f"models: {s['entries']} entr{'y' if s['entries'] == 1 else 'ies'}, {s['gb']:.1f} GB ({own}){nxt}{past}"


def _ls(root: Path, cfg: dict[str, Any], now: int, owner: str | None, used_pct: float) -> str:
    es = [e for e in entries(root).values() if owner is None or e["owner"] == owner]
    out = [f"{root} ({cfg['node'] or 'unknown node'}): " + line(summary(es, cfg, now, str(root))).removeprefix("models: ")
           + f"; disk {used_pct:.0f}% used, fetches refused past {cfg['max_use_pct']}%"]
    rows = [("ENTRY", "OWNER", "PHASE", "GB", "LAST_USED", "EXPIRES")] + [
        (f"{e['repo']}@{e['revision']}", e["owner"], e["phase"], f"{e['total_bytes'] / GB:.2f}", e["last_used"][:16] + "Z",
         e["expires"][:16] + "Z") for e in es]
    if es:
        w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        out += ["  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    inc = [x for x in incomplete(root) if owner is None or x["owner"] == owner]
    for x in inc:
        of = f" of {x['total_bytes'] / GB:.2f}" if x["total_bytes"] is not None else ""
        out.append(f"  incomplete: {os.path.relpath(x['path'], root)} ({x['owner'] or 'owner unknown'}, {x['bytes'] / GB:.2f}{of} GB, "
                   + ("a fetch is running)" if x["running"] else f"untouched since {iso(x['mtime'])})"))
    return "\n".join(out) + "\n"


clock = time.time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cluster models", description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(CONFIG), help="the cache's configuration (default tools/cluster/models.toml)")
    common.add_argument("--root", help="the cache (default: the configuration's root, /workspace/models)")
    common.add_argument("--now", help="the instant, ISO UTC (default: the clock)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help, parents=[common])

    f = add("fetch", "download REPO@REV, verified, into this node's cache (the model.fetch kind)")
    f.add_argument("ref", metavar="REPO@REV")
    f.add_argument("--owner", required=True)
    f.add_argument("--phase", required=True)
    f.add_argument("--declared-gb", type=float, required=True, help="the disk it asks for; refused when HF lists more")
    f.add_argument("--expires-h", type=float, help="hours after its last use it expires (default: the configuration's, 72)")
    f.add_argument("--include", action="append", default=[], metavar="GLOB", help="only the files matching (repeatable)")
    f.add_argument("--jobs", type=int, default=4, help="files downloaded at once")
    f.add_argument("--out", help="also write the entry to OUT/model-entry.json (a run's declared output)")
    ls = add("ls", "the entries and incomplete downloads")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--owner")
    u = add("use", "a job starts on REPO@REV: bump last_used and expires; prints the entry's directory")
    u.add_argument("ref", metavar="REPO@REV")
    rl = add("release", "the owner deletes its entry now")
    rl.add_argument("ref", metavar="REPO@REV")
    rl.add_argument("--owner", required=True)
    ev = add("evict", "delete expired entries and stale downloads not in use (run as root: sudo -n)")
    ev.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    def say(s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    try:
        cfg = node_config(load_config(Path(a.config)), hostname=socket.gethostname())
        root = Path(a.root or cfg["root"])
        try:
            now = instant(a.now) if a.now else int(clock())
        except ValueError:
            raise Refused(2, f"--now {a.now!r}: give an ISO instant with its zone, such as 2026-10-01T02:00:00Z") from None
        if a.cmd == "fetch":
            r = ref(a.ref)
            tpath = os.environ.get("HF_TOKEN_PATH", TOKEN_PATH)
            tok, why = token(tpath)
            e, how = fetch(root, r, owner=a.owner, phase=a.phase, declared_gb=a.declared_gb, now=now, hub=Hub(tok), cfg=cfg,
                           expires_h=a.expires_h, include=tuple(a.include), fetched_by=os.environ.get("RESEARCH_RUN_ID"),
                           jobs=a.jobs, token_path=tpath, token_why=why, say=say)
            say(f"{r}: {how}, {e['total_bytes'] / GB:.3f} GB in {len(e['files'])} files at {root / r.name}, owner {e['owner']}, "
                f"expires {e['expires']}")
            if a.out:
                Path(a.out).mkdir(parents=True, exist_ok=True)
                (Path(a.out) / "model-entry.json").write_text(json.dumps(e, indent=1, sort_keys=True) + "\n")
            print(json.dumps(e, sort_keys=True))
        elif a.cmd == "ls":
            if a.json:
                es = [e for e in entries(root).values() if a.owner is None or e["owner"] == a.owner]
                inc = [x for x in incomplete(root) if a.owner is None or x["owner"] == a.owner]
                print(json.dumps({"schema": "cluster/models/v1", "root": str(root), "node": cfg["node"], "t": iso(now),
                                  "summary": summary(es, cfg, now, str(root)), "entries": es, "incomplete": inc}, indent=1,
                                 sort_keys=True))
            else:
                used, avail = _disk(root) if root.is_dir() else (0, 1)
                print(_ls(root, cfg, now, a.owner, 100 * used / (used + avail)), end="")
        elif a.cmd == "use":
            r = ref(a.ref)
            e = use(root, r, now)
            say(f"{r}: last used {e['last_used']}, expires {e['expires']}")
            print(root / r.name)
        elif a.cmd == "release":
            r = ref(a.ref)
            e = release(root, r, a.owner)
            say(f"{r}: released, {e['total_bytes'] / GB:.3f} GB freed")
        else:
            rep = evict(root, now, dry_run=a.dry_run)
            if rep["hidden"]:
                say(f"can't see {rep['hidden']} processes' open and mapped files: a real evict would refuse (run it as root)")
            for x in rep["deleted"]:
                say(f"{'would delete' if a.dry_run else 'deleted'} {x['path']}: {x['why']}")
            for x in rep["kept"]:
                say(f"kept {x['path']}: {x['why']}")
            print(json.dumps(rep, sort_keys=True))
    except Refused as e:
        print(f"cluster models: {e}", file=sys.stderr)
        return e.code
    except Failed as e:
        print(f"cluster models: {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"cluster models: {e.strerror}: {e.filename} (the cache is the research user's; evict runs as root)", file=sys.stderr)
        return 3
    return 0


__all__ = ["SCHEMA", "CONFIG", "ENTRY", "GB", "Refused", "Failed", "Mismatch", "HubError", "Ref", "ref", "iso", "load_config",
           "node_config", "quota_gb", "read_entry", "entries", "parse_entries", "incomplete", "Remote", "listing", "Hub",
           "token", "write_entry", "fetch", "use", "Scan", "scan", "release", "evict", "summary", "line", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

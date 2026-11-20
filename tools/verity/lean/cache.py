#!/usr/bin/env python3
"""The Lean build cache: main's compiled modules, which integration publishes on merge and development machines restore,
every file checked against its sha256 first. Standard library only, with `git` and `lake`. `setup.sh` runs it with the
machine's own `python3`, which may be older than the workspace's 3.12, so it keeps to syntax Python 3.10 parses.

    python tools/verity/lean/cache.py get [PKG...] [--branch NAME]
                                                  restore main's cached build of each package (default: every package with
                                                  a `lean-audit.json`) into Lake's local cache; `lake build` then unpacks
                                                  those modules instead of compiling them
    python tools/verity/lean/cache.py get --bin DIR [PKG...] [--branch NAME] [--platform TRIPLE] [--exact]
                                                  write each package's built executables to DIR/<package>/<name> (the
                                                  package's path quoted, as `out_dir` names it), each checked first, and
                                                  the commit they were built at to DIR/<package>.json; --exact: only
                                                  HEAD's own build; no Lean toolchain is needed
    python tools/verity/lean/cache.py build PKG OUT      (integration, inside the build's sandbox) `lake build` PKG writing Lake's
                                                  cache under OUT, and stage what it built in OUT/stage and its executables
                                                  in OUT/bin
    python tools/verity/lean/cache.py publish --tree DIR --rev SHA --out DIR [--branch NAME] [--main REF] PKG...
                                                  (integration, as root outside the sandbox) hash each package's
                                                  `out_dir(OUT, PKG)`/stage and /bin and upload them with the key in
                                                  /etc/verity/lean-cache.env
    python tools/verity/lean/cache.py prune [--keep N] [--dry-run] [--branch NAME]
                                                  (integration) delete what no record of the last N published commits needs

The bucket (`LEAN_CACHE_BUCKET`) holds `main/`, which a 7-day lock keeps from being overwritten or deleted, and, for a
branch's commits, `branch/<name>/`, which no lock holds, so that it can go when the branch lands; each holds, under its
own prefix:
- `<toolchain>/<platform>/<package>/<manifest>/<rev>.sha256.json`: one record per published commit and package, the
  sha256 and size of every file `lake cache stage` wrote: Lake's `outputs.jsonl` (each module's input hash to its archive)
  and one archive per module, then for each dependency the build compiled (ArkLib and the packages it brings; not Mathlib
  and its own dependencies, which Mathlib's cache provides, nor a path dependency, a package of its own) its mappings
  `dep.<scope>.jsonl` and its modules' artifacts; and under `bin`, each executable the build left in `.lake/build/bin/`
  (an empty `bin` for a package that builds none);
- `blobs/<sha256>`: each of those files under its own hash, shared across that prefix's commits.

`main/` holds only commits on `origin/main`, and `branch/<name>/` only commits on `origin/<name>`, with its own blobs and its
own prune; `publish --main REF` names the ref instead where the line was fetched under another name, as the sandbox's
checkout fetches it into `refs/remotes/origin/main`. `get` reads a branch only when told to (`--branch`), and then takes the newest ancestor of `HEAD` across both
prefixes, main's on a tie: main holds records of the branch's old ancestors, and an executable of one is another verifier.
With `--bin` a record that lists no executables is passed over, and what was fetched says which commit it is of.

A fetched executable is for the prover finding out whether its evidence passes, and for development. Its integrity is the
publisher's key and a sha256 against a record in this bucket: enough for completeness, not for a verdict an auditor relies
on, which comes from executables built from the commit's source by whoever relies on them. A run of record that takes its
verifier from `get --bin` says so in its output.

The client downloads every file itself and checks it against the record before Lake sees it, then hands them to Lake
(`lake cache unstage`, then `lake cache unstage DIR <scope>` per dependency). Lake never reads the bucket, because its
hashes aren't cryptographic. A module whose input hash
has no checked file is built from source, as is every module after a mismatch, when the kill switch (the docs site's
`/api/lean-cache/state`) is disabled, quarantines the scope or can't be reached, or when `LEAN_CACHE=off`. Lake's input
hashes don't depend on where a checkout is, so a build published from `/build/verity` restores in any checkout, and one
published before a layout move renamed the package's directory restores at its new path: where a package has no record
under its path, the client reads the newest under each path it had before (`moved.py`, from the recorded moves).
A development machine restores the newest published build that is an ancestor of its `HEAD` (`cache.py get`, after
`setup.sh`), so a fresh machine compiles none of ArkLib; a missing read credential (the `LEAN_CACHE_*` variables, else
`~/.config/verity/lean-cache-read.env`) means building from source too.

The publisher signs its uploads directly with the bucket-scoped key, as root, outside the sandbox the build ran in; no
broader credential sits on the machine to mint a shorter-lived one. Every write is conditional (`If-None-Match: *`). A
412 or 409 means the object exists: a blob then must be stored with its MD5 as ETag (blobs go up in one part, so it is),
and a record must be byte-identical, or the package isn't published. The record goes last, so every blob it names exists.
Integration's own builds never read the cache: `check.py` runs Lake with `LEAN_CACHE=off`, and with `LAKE_ARTIFACT_CACHE=false`
or, in check's audit and a records run, a local Lake cache of what that node's builds compiled in the sandbox, restoring a module
only under its key (`lean_audit.LakeArtifacts`, `module_keys.py`).
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "lean-cache/v1"
TOP = "main/"
BLOBS = TOP + "blobs/"
#: a branch's prefix, `branch/<name>/`, beside `TOP` (`top`)
BRANCH = "branch/"
RECORD = ".sha256.json"
MAPPING = "outputs.jsonl"
#: the record's key for the package's executables, and the directory beside its stage where `build` copies them
BIN = "bin"
#: a staged dependency's mappings are `dep.<scope>.jsonl`, beside the root package's `outputs.jsonl`
DEP = "dep."
#: Lake's output of `build`, beside its stage
BUILD_LOG = "build.log"
#: where the sandbox binds a package's fetched dependencies read-only; `build` works on its own copy at `.lake/packages`
DEPS_RO = "deps"
STATE_URL = "https://website-docs-sage.vercel.app/api/lean-cache/state"
STATE_TTL_S = 300
KEY_FILE = "/etc/verity/lean-cache.env"
#: the directory under the user's cache (`XDG_CACHE_HOME`) where `get` keeps what it downloaded
DOWNLOADS = "verity-lean-cache"  # pragma: allowlist secret
#: the read-only credential every development machine holds (the Cursor secrets)
READ_ENV = ("LEAN_CACHE_R2_ACCESS_KEY_ID", "LEAN_CACHE_R2_SECRET_ACCESS_KEY", "LEAN_CACHE_R2_ENDPOINT", "LEAN_CACHE_BUCKET")
#: under HOME, the same credential as `KEY=value` lines where a node's research user holds it (0600)
READ_FILE = ".config/verity/lean-cache-read.env"
#: the names in `KEY_FILE`, the publisher's read-write key
WRITE_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")
MAX_BYTES = 32 << 30
THREADS = 16
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PACKAGE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)*\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
REV = re.compile(r"[0-9a-f]{40}\Z")
OFF = ("off", "0", "false", "no")


class CacheError(Exception):
    """A check failed: nothing of this package is restored or published."""


# ---------------------------------------------------------------- S3 (AWS Signature Version 4, which R2 speaks)

def _enc(s: str, safe: str = "") -> str:
    return urllib.parse.quote(s, safe="-_.~" + safe)


def _query(q: dict[str, str]) -> str:
    return "&".join(f"{_enc(k)}={_enc(v)}" for k, v in sorted(q.items()))


def sign(method: str, host: str, path: str, query: dict[str, str], headers: dict[str, str], key_id: str, secret: str,
         region: str) -> str:
    """The `Authorization` header of a SigV4 request. Every header in ``headers`` is signed, with `host`; ``headers`` holds
    `x-amz-date` and `x-amz-content-sha256`."""
    h = {**{k.lower(): " ".join(str(v).split()) for k, v in headers.items()}, "host": host}
    names = sorted(h)
    amz = h["x-amz-date"]
    canonical = "\n".join([method, _enc(path, "/"), _query(query), "".join(f"{n}:{h[n]}\n" for n in names), ";".join(names),
                           h["x-amz-content-sha256"]])
    scope = f"{amz[:8]}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + secret).encode()
    for part in (amz[:8], region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return (f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, SignedHeaders={';'.join(names)}, "
            f"Signature={hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()}")


#: R2's transient answers, which a call retries after each of `RETRY_S` seconds in turn: one HTTP 500 on one package
#: would otherwise end a whole publish
TRANSIENT = frozenset({500, 502, 503, 504})
RETRY_S = (2, 8, 30)
_sleep = time.sleep


class S3:
    """The few S3 calls the cache makes, path-style against one bucket."""

    def __init__(self, endpoint: str, bucket: str, key_id: str, secret: str, region: str = "auto",
                 now=lambda: datetime.datetime.now(datetime.UTC)):
        u = urllib.parse.urlsplit(endpoint)
        if u.scheme not in ("https", "http") or not u.netloc:
            raise ValueError(f"not an endpoint URL: {endpoint!r}")
        self.scheme, self.host, self.base = u.scheme, u.netloc, u.path.rstrip("/")
        self.bucket, self.key_id, self.secret, self.region, self.now = bucket, key_id, secret, region, now

    def open(self, method: str, key: str = "", query: dict[str, str] | None = None, body: bytes | None = None,
             headers: dict[str, str] | None = None, timeout: float = 120):
        """The response; a transient answer (`TRANSIENT`, or no connection) is retried after each of `RETRY_S`, signed
        anew, and any other error is raised at once."""
        query = query or {}
        path = f"{self.base}/{self.bucket}" + (f"/{key}" if key else "")
        url = f"{self.scheme}://{self.host}{_enc(path, '/')}" + (f"?{_query(query)}" if query else "")
        for wait in (*RETRY_S, None):
            h = {"x-amz-date": self.now().strftime("%Y%m%dT%H%M%SZ"),
                 "x-amz-content-sha256": hashlib.sha256(body).hexdigest() if body is not None else EMPTY_SHA,
                 **(headers or {})}
            auth = sign(method, self.host, path, query, h, self.key_id, self.secret, self.region)
            req = urllib.request.Request(url, data=body, method=method, headers={**h, "Authorization": auth})  # noqa: S310  (as below)
            if body is not None:
                req.add_header("Content-Type", "application/octet-stream")
            try:
                return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  (`__init__` takes only https or http)
            except urllib.error.HTTPError as e:
                if e.code not in TRANSIENT or wait is None:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if wait is None:
                    raise
            _sleep(wait)

    def get(self, key: str) -> bytes | None:
        try:
            with self.open("GET", key) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def download(self, key: str, dest: Path, sha: str, size: int) -> None:
        """``key`` written to ``dest`` only when its bytes have ``sha`` and ``size``; CacheError otherwise."""
        part = dest.with_name(dest.name + ".part")
        h, n = hashlib.sha256(), 0
        try:
            with self.open("GET", key) as r, open(part, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    h.update(chunk)
                    n += len(chunk)
                    if n > size:
                        break
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            if e.code == 404:
                raise CacheError(f"{key} is missing") from None
            raise
        if (h.hexdigest(), n) != (sha, size):
            part.unlink(missing_ok=True)
            raise CacheError(f"{key} is not the file its record names (sha256 or size differs)")
        os.replace(part, dest)

    def put(self, key: str, body: bytes) -> bool:
        """Write ``key`` if it doesn't exist, in one part; False when it does (412, or 409 for a concurrent write)."""
        md5 = hashlib.md5(body).digest()
        try:
            with self.open("PUT", key, body=body, headers={"if-none-match": "*",
                                                          "content-md5": base64.b64encode(md5).decode()}) as r:
                etag = (r.headers.get("ETag") or "").strip('"')
        except urllib.error.HTTPError as e:
            if e.code in (409, 412):
                return False
            raise
        if etag and etag != md5.hex():
            raise CacheError(f"{key} was stored with ETag {etag}, not its MD5")
        return True

    def head(self, key: str) -> dict | None:
        try:
            with self.open("HEAD", key) as r:
                return {"etag": (r.headers.get("ETag") or "").strip('"'), "bytes": int(r.headers.get("Content-Length") or -1)}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def delete(self, key: str) -> None:
        with self.open("DELETE", key):
            pass

    def list(self, prefix: str) -> list[dict]:
        """Every object under ``prefix``: {key, bytes, etag, modified (seconds since the epoch)}."""
        out, token = [], None
        while True:
            q = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
            with self.open("GET", query=q) as r:
                root = ET.fromstring(r.read())
            ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
            for c in root.iter(f"{ns}Contents"):
                when = datetime.datetime.strptime(c.findtext(f"{ns}LastModified")[:19], "%Y-%m-%dT%H:%M:%S")
                out.append({"key": c.findtext(f"{ns}Key"), "bytes": int(c.findtext(f"{ns}Size") or 0),
                            "etag": (c.findtext(f"{ns}ETag") or "").strip('"'),
                            "modified": when.replace(tzinfo=datetime.UTC).timestamp()})
            if root.findtext(f"{ns}IsTruncated") != "true":
                return out
            token = root.findtext(f"{ns}NextContinuationToken")


def reader(env=os.environ) -> S3 | None:
    """The bucket through the read-only credential, or None when this machine has none: READ_ENV from ``env``, else those four
    keys from READ_FILE under ``env``'s HOME (where the nodes' research user holds it)."""
    v = [env.get(k, "") for k in READ_ENV]
    if not all(v) and env.get("HOME") and (f := Path(env["HOME"]) / READ_FILE).is_file():
        held = _key_lines(f)
        v = [held.get(k, "") for k in READ_ENV]
    return S3(v[2], v[3], v[0], v[1]) if all(v) else None


def _key_lines(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def writer(path: str = KEY_FILE) -> S3:
    """The bucket through the publisher's key, read from ``path`` (`KEY=value` lines)."""
    env = _key_lines(Path(path))
    missing = [k for k in WRITE_ENV if not env.get(k)]
    if missing:
        raise CacheError(f"{path} lacks {', '.join(missing)}")
    return S3(env["R2_ENDPOINT"], env["R2_BUCKET"], env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"])


# ---------------------------------------------------------------- what a package's builds are keyed by

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _elan_env(env=os.environ) -> dict[str, str]:
    return {**env, "PATH": f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"}


_PLATFORMS: dict[str, str] = {}


def parse_platform(version: str) -> str | None:
    """The target triple in `lean --version`'s output."""
    m = re.search(r"\(version [^,]+, ([A-Za-z0-9_.-]+),", version)
    return m.group(1) if m else None


def platform(pkg: Path, toolchain: str) -> str:
    """The target triple of ``toolchain``'s `lean`, as Lake keys its builds by it (e.g. `x86_64-unknown-linux-gnu`)."""
    if toolchain not in _PLATFORMS:
        try:
            r = subprocess.run(["lean", "--version"], cwd=pkg, capture_output=True, text=True, timeout=300, env=_elan_env())
        except FileNotFoundError:
            raise CacheError(f"no `lean` here to name the platform (tools/verity/lean/setup.sh {pkg} installs it)") from None
        if r.returncode != 0 or not (p := parse_platform(r.stdout)):
            raise CacheError(f"`lean --version` in {pkg} printed no platform")
        _PLATFORMS[toolchain] = p
    return _PLATFORMS[toolchain]


def identity(root: Path, pkg: str, plat: str | None = None) -> dict:
    """What a record must match: the package, its toolchain, the platform and its `lake-manifest.json`."""
    if not PACKAGE.match(pkg) or ".." in pkg.split("/"):
        raise CacheError(f"not a package path: {pkg!r}")
    d = root / pkg
    toolchain = (d / "lean-toolchain").read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9._/:+-]+", toolchain):
        raise CacheError(f"{pkg}: unexpected lean-toolchain {toolchain!r}")
    plat = plat or platform(d, toolchain)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", plat):
        raise CacheError(f"not a platform: {plat!r}")
    m = d / "lake-manifest.json"
    return {"package": pkg, "toolchain": toolchain, "platform": plat, "manifest": sha256(m) if m.is_file() else EMPTY_SHA}


def prefix_of(branch: str | None = None) -> str:
    """Where ``branch``'s records and blobs live (`branch/<name>/`), or main's (`TOP`) for none."""
    if branch is None:
        return TOP
    if not NAME.match(branch):
        raise CacheError(f"not a branch name: {branch!r}")
    return f"{BRANCH}{branch}/"


def scope(ident: dict, prefix: str = TOP) -> str:
    """The key prefix of a package's records (elan's spelling of the toolchain: `leanprover--lean4---v4.34.0`)."""
    tc = ident["toolchain"].replace("/", "--").replace(":", "---")
    return f"{prefix}{tc}/{ident['platform']}/{ident['package']}/{ident['manifest'][:16]}"


def record_key(ident: dict, rev: str, prefix: str = TOP) -> str:
    return f"{scope(ident, prefix)}/{rev}{RECORD}"


def check_record(raw: bytes, ident: dict, rev: str) -> dict:
    """The record, or CacheError when it isn't one for ``ident`` at ``rev`` or names a file it shouldn't."""
    try:
        rec = json.loads(raw)
    except ValueError:
        raise CacheError("its record is not JSON") from None
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA or rec.get("rev") != rev:
        raise CacheError(f"its record is not a {SCHEMA} record of {rev[:12]}")
    for k, v in ident.items():
        if rec.get(k) != v:
            raise CacheError(f"its record's {k} is {rec.get(k)!r}, not {v!r}")
    files = rec.get("files")
    if not isinstance(files, dict) or MAPPING not in files:
        raise CacheError(f"its record lists no {MAPPING}")
    bins = rec.get(BIN, {})
    if not isinstance(bins, dict):
        raise CacheError(f"its record's {BIN} is not a table of executables")
    total = 0
    for name, f in [*files.items(), *bins.items()]:
        if not (isinstance(name, str) and NAME.match(name) and isinstance(f, dict) and isinstance(f.get("sha256"), str)
                and SHA.match(f["sha256"]) and type(f.get("bytes")) is int and f["bytes"] >= 0):
            raise CacheError(f"its record has a bad entry {name!r}")
        total += f["bytes"]
    if total > MAX_BYTES:
        raise CacheError(f"its record names {total >> 20} MiB, more than the {MAX_BYTES >> 30} GiB a package may")
    return rec


# ---------------------------------------------------------------- the client

def packages(root: Path) -> list[str]:
    """The packages `check.py --all` audits: directories with a tracked `lean-audit.json`, the auditor's fixtures aside
    (`check.FIXTURES`, where `check.py` is beside this file; none where it isn't, in another repository's copy)."""
    p, fixtures = Path(__file__).resolve().with_name("check.py"), ()
    if p.is_file():
        spec = importlib.util.spec_from_file_location("lean_check_for_cache", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fixtures = (f"{m.FIXTURES.relative_to(m.ROOT).as_posix()}/",)
    r = subprocess.run(["git", "-C", str(root), "ls-files", "*lean-audit.json"], capture_output=True, text=True)
    return sorted({f.rsplit("/", 1)[0] for f in r.stdout.split() if "/" in f and not f.startswith(fixtures)})


def ancestors(root: Path, n: int = 2000) -> list[str]:
    """HEAD and its ancestors, newest first."""
    r = subprocess.run(["git", "-C", str(root), "rev-list", f"--max-count={n}", "HEAD"], capture_output=True, text=True)
    return r.stdout.split() if r.returncode == 0 else []


def state(url: str, cache: Path, now: float, fetch=None) -> tuple[bool, list[str], str]:
    """(enabled, quarantined scopes, why not) from the kill switch, kept for `STATE_TTL_S`. Anything but a well-formed
    `{"enabled": true, ...}` reads as disabled."""
    kept = cache / "state.json"
    doc = None
    try:
        k = json.loads(kept.read_text())
        if k.get("url") == url and 0 <= now - k["fetched"] < STATE_TTL_S:
            doc = k["state"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    if doc is None:
        try:
            raw = fetch(url) if fetch else urllib.request.urlopen(url, timeout=10).read()  # noqa: S310  (STATE_URL or its override)
            doc = json.loads(raw)
        except (OSError, ValueError) as e:
            return False, [], f"the kill switch ({url}) is unreachable ({getattr(e, 'code', None) or type(e).__name__})"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = kept.with_name(f".state.{os.getpid()}.json")
            tmp.write_text(json.dumps({"url": url, "fetched": now, "state": doc}))
            os.replace(tmp, kept)
        except OSError:
            pass
    if not isinstance(doc, dict) or doc.get("enabled") is not True:
        return False, [], "the cache is disabled (kill switch)"
    q = doc.get("quarantined", [])
    if not isinstance(q, list) or not all(isinstance(s, str) for s in q):
        return False, [], "the kill switch's quarantine list is malformed"
    return True, q, ""


def quarantined(sc: str, q: list[str]) -> bool:
    return any(s and (sc == s.rstrip("/") or sc.startswith(s.rstrip("/") + "/")) for s in q)


def lake_cache(ident: dict, env=os.environ) -> Path | None:
    """Lake's local cache for ``ident``'s toolchain, where `lake cache unstage` puts files (LAKE_CACHE_DIR, else elan's)."""
    if "LAKE_CACHE_DIR" in env:
        return Path(env["LAKE_CACHE_DIR"]) if env["LAKE_CACHE_DIR"] else None
    elan = Path(env.get("ELAN_HOME") or Path.home() / ".elan")
    return elan / "toolchains" / ident["toolchain"].replace("/", "--").replace(":", "---") / "lake" / "cache"


def mapping(name: str) -> bool:
    """Whether a staged file is a mappings file (the root package's or a dependency's) rather than an artifact."""
    return name == MAPPING or (name.startswith(DEP) and name.endswith(".jsonl"))


def fetched_deps(d: Path) -> bool:
    """Whether a package's non-path dependencies are in place (`lake` would otherwise clone them first)."""
    return not unfetched(d)


def unfetched(d: Path) -> list[str]:
    """The non-path dependencies of the package at ``d`` that are not in place, each as Lake names its directory and in
    its manifest's order, the order Lake clones them in."""
    m = d / "lake-manifest.json"
    deps = [p for p in json.loads(m.read_text()).get("packages", []) if p.get("type") != "path"] if m.is_file() else []
    names = [unescape(p["name"]) for p in deps]
    return [n for n in names if not (d / ".lake" / "packages" / n).is_dir()]


def unescape(name: str) -> str:
    """A manifest's package name as Lake names its directory: `«doc-gen4»` (a name Lean must escape) is `doc-gen4`."""
    return name[1:-1] if name.startswith("«") and name.endswith("»") else name


def newest(s3: S3, sc: str, revs: list[str]) -> str | None:
    """The first of ``revs`` (newest first) with a record under the scope ``sc``."""
    have = {o["key"][len(sc) + 1: -len(RECORD)] for o in s3.list(sc + "/") if o["key"].endswith(RECORD)}
    return next((r for r in revs if r in have), None)


def former(root: Path, pkg: str) -> list[str]:
    """The paths ``pkg`` had before the layout moves ``root`` records, latest first (`moved.py`, beside this file); none
    where that file isn't (another repository's copy of the cache)."""
    p = Path(__file__).resolve().with_name("moved.py")
    if not p.is_file():
        return []
    spec = importlib.util.spec_from_file_location("lean_moved_for_cache", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return [f for f in m.former(pkg, m.recorded(root)) if PACKAGE.match(f) and ".." not in f.split("/")]


def find(root: Path, pkg: str, s3: S3, revs: list[str], q: list[str], ident: dict,
         branch: str | None = None) -> list[tuple[str, dict, str]]:
    """(prefix, identity, rev) of ``pkg``'s newest record in each place one can be, the newest commit among ``revs``
    (newest first) first: under main's prefix and, with ``branch``, under the branch's, each at the package's path and at
    each path it had before a layout move. On one commit main's comes first, then the package's own path; a quarantined
    scope is passed over. Main holds records of a branch's old ancestors (its fork point and before), so the branch's newer
    build has to win over them, not wait behind them."""
    index = {r: i for i, r in enumerate(revs)}
    # A layout move renames a package's directory and no byte Lake hashes, and records are keyed by that directory:
    # until integration publishes a commit after the move, the build is under the path the package had before it.
    places = [ident, *({**ident, "package": old} for old in former(root, pkg))]
    hits = []
    for p, prefix in enumerate((TOP, *((prefix_of(branch),) if branch else ()))):
        for k, at in enumerate(places):
            sc = scope(at, prefix)
            if not quarantined(sc, q) and (rev := newest(s3, sc, revs)) is not None:
                hits.append((index[rev], p, k, prefix, at, rev))
    return [(prefix, at, rev) for _, _, _, prefix, at, rev in sorted(hits, key=lambda h: h[:3])]


def built_by(prefix: str) -> str:
    """Whose build a record under ``prefix`` is, for the log: `main's` or `branch/<name>'s`."""
    return "main's" if prefix == TOP else f"{prefix.rstrip('/')}'s"


def restore(root: Path, pkg: str, s3: S3, revs: list[str], q: list[str], log=print, env=os.environ, plat: str | None = None,
            lake: list[str] | None = None, branch: str | None = None) -> bool:
    """The newest cached build of ``pkg`` among ``revs`` (`find`: main's, or with ``branch`` the newer of main's and the
    branch's) into Lake's local cache, every file checked first. False when there is nothing to restore; CacheError when a
    check fails."""
    if plat is None and shutil.which("lean", path=_elan_env(env)["PATH"]) is None:
        log(f"lean-cache: {pkg}: no Lean toolchain here (bash tools/verity/lean/setup.sh {pkg} installs elan, the pinned "
            f"toolchain and the dependencies); building from source")
        return False
    ident = identity(root, pkg, plat)
    sc = scope(ident)
    if quarantined(sc, q):
        log(f"lean-cache: {pkg}: {sc} is quarantined; building from source")
        return False
    if not fetched_deps(root / pkg):
        log(f"lean-cache: {pkg}: its dependencies aren't fetched (run tools/verity/lean/setup.sh {pkg}); building from source")
        return False
    found = find(root, pkg, s3, revs, q, ident, branch)
    if not found:
        log(f"lean-cache: {pkg}: no cached build of an ancestor of HEAD ({sc}"
            f"{f', nor under {prefix_of(branch)}' if branch else ''}); building from source")
        return False
    prefix, at, rev = found[0]
    raw = s3.get(record_key(at, rev, prefix))
    if raw is None:
        raise CacheError(f"{record_key(at, rev, prefix)} was listed but can't be read")
    rec = check_record(raw, at, rev)
    local = lake_cache(ident, env)
    deps = sorted(n[len(DEP):-len(".jsonl")] for n in rec["files"] if n != MAPPING and mapping(n))
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lean-cache-") as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()
        todo = []
        for name, f in rec["files"].items():
            mine = local / "artifacts" / name if local and not mapping(name) else None
            if mine and mine.is_file() and mine.stat().st_size == f["bytes"] and sha256(mine) == f["sha256"]:
                shutil.copyfile(mine, stage / name)
            else:
                todo.append((name, f))
        blobs = prefix + "blobs/"
        with ThreadPoolExecutor(THREADS) as ex:
            list(ex.map(lambda nf: s3.download(blobs + nf[1]["sha256"], stage / nf[0], nf[1]["sha256"], nf[1]["bytes"]), todo))
        for dep in [None, *deps]:
            if dep is not None:
                os.replace(stage / f"{DEP}{dep}.jsonl", stage / MAPPING)
            r = subprocess.run([*(lake or ["lake"]), "cache", "unstage", str(stage), *([dep] if dep else [])],
                               cwd=root / pkg, capture_output=True, text=True, env=_elan_env(env))
            if r.returncode != 0:
                log(f"lean-cache: {pkg}: `lake cache unstage`{f' of {dep}' if dep else ''} failed: "
                    f"{(r.stdout + r.stderr).strip()[-400:]}; building from source")
                return False
    got = sum(f["bytes"] for _, f in todo)
    arts = sum(not mapping(n) for n in rec["files"])
    with_deps = f", with {len(deps)} dependencies ({', '.join(deps)})" if deps else ""
    moved = "" if at is ident else f" (published as {at['package']}, before a layout move)"
    log(f"lean-cache: {pkg}: {built_by(prefix)} build at {rev[:12]}{moved} restored ({arts} artifacts"
        f"{with_deps}; {got >> 20} MiB in {len(todo)} files "
        f"fetched and checked in {time.monotonic() - t0:.0f} s, {len(rec['files']) - len(todo)} already here); "
        f"`lake build` unpacks them")
    return True


def default_platform() -> str:
    """The target triple Lake names this machine's builds by, where no `lean` is here to say it (`get --bin`)."""
    u = os.uname()
    if u.sysname != "Linux":
        raise CacheError(f"no published executables run on {u.sysname}: give --platform")
    return f"{u.machine}-unknown-linux-gnu"


def fetch_bin(root: Path, pkg: str, s3: S3, revs: list[str], q: list[str], dest: Path, log=print, plat: str | None = None,
              branch: str | None = None, exact: bool = False) -> bool:
    """``pkg``'s executables in its newest published build among ``revs`` whose record lists them (`find`; a record
    published before records held them is passed over), written to `out_dir(dest, pkg)`/<name> with mode 755 once every
    one of them is checked against the record, and described in `out_dir(dest, pkg)`.json: the commit they were built at,
    HEAD, the prefix, the record and each one's sha256 and size. The directory is made anew and swapped in, so it holds
    only what the record lists; the description is removed before the swap and written after it. False when nothing is
    published; CacheError when a check fails, when no record found lists executables, or with ``exact`` when the build is
    not HEAD's own (``revs[0]``)."""
    if plat is None and shutil.which("lean", path=_elan_env()["PATH"]) is None:
        plat = default_platform()
    ident = identity(root, pkg, plat)
    found = find(root, pkg, s3, revs, q, ident, branch)
    if not found:
        log(f"lean-cache: {pkg}: no published build of an ancestor of HEAD ({scope(ident)}"
            f"{f', nor under {prefix_of(branch)}' if branch else ''})")
        return False
    for prefix, at, rev in found:
        key = record_key(at, rev, prefix)
        raw = s3.get(key)
        if raw is None:
            raise CacheError(f"{key} was listed but can't be read")
        rec = check_record(raw, at, rev)
        if BIN in rec:
            break
    else:
        raise CacheError(f"{record_key(found[0][1], found[0][2], found[0][0])} lists no executables: it was published "
                         f"before records held them")
    if exact and rev != revs[0]:
        raise CacheError(f"the newest published build with executables is of {rev[:12]}, not HEAD's own ({revs[0][:12]}): "
                         f"--exact")
    out = out_dir(dest, pkg)
    about = out.with_name(out.name + ".json")
    dest.mkdir(parents=True, exist_ok=True)
    blobs, exes = prefix + "blobs/", sorted(rec[BIN].items())
    new = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=dest))
    try:
        with ThreadPoolExecutor(THREADS) as ex:
            list(ex.map(lambda nf: s3.download(blobs + nf[1]["sha256"], new / nf[0], nf[1]["sha256"], nf[1]["bytes"]), exes))
        for name, _ in exes:
            os.chmod(new / name, 0o755)
        os.chmod(new, 0o755)
        about.unlink(missing_ok=True)
        if os.path.lexists(out):
            gone = new.with_name(new.name + ".old")
            os.rename(out, gone)
            os.rename(new, out)
            if gone.is_dir() and not gone.is_symlink():
                shutil.rmtree(gone)
            else:
                gone.unlink()
        else:
            os.rename(new, out)
        tmp = new.with_name(new.name + ".json")
        tmp.write_text(json.dumps({"package": pkg, "rev": rev, "head": revs[0], "prefix": prefix, "record": key,
                                   BIN: rec[BIN]}, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, about)
    finally:
        shutil.rmtree(new, ignore_errors=True)
    of = "HEAD's own" if rev == revs[0] else f"an ancestor of HEAD ({revs[0][:12]})"
    log(f"lean-cache: {pkg}: {len(exes)} executables of {built_by(prefix)} build at {rev[:12]}, {of}, in {out}, each "
        f"checked against its record and described in {about.name} (for the prover and development: a verdict of record "
        f"is built from the commit's source)")
    return True


def get(root: Path, pkgs: list[str], env=os.environ, log=print, now=time.time, fetch_state=None, bin_dir: Path | None = None,
        branch: str | None = None, plat: str | None = None, exact: bool = False) -> int:
    """`get`: 0 when every package was restored or had nothing to restore, 1 when a check failed (that package builds from
    source). With ``bin_dir``, each package's executables instead (`fetch_bin`, with ``exact``), and 1 also for a package
    that has none published or a cache that can't be read: what was asked for isn't there."""
    want = 1 if bin_dir is not None else 0
    if env.get("LEAN_CACHE", "").strip().lower() in OFF:
        log("lean-cache: off on this machine (LEAN_CACHE); building from source")
        return want
    s3 = reader(env)
    if s3 is None:
        log(f"lean-cache: no read credential ({', '.join(READ_ENV)}); building from source")
        return want
    cache = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / DOWNLOADS
    ok, q, why = state(env.get("LEAN_CACHE_STATE_URL") or STATE_URL, cache, now(), fetch_state)
    if not ok:
        log(f"lean-cache: {why}; building from source")
        return want
    revs, status = ancestors(root), 0
    for pkg in pkgs:
        try:
            if bin_dir is not None:
                if not fetch_bin(root, pkg, s3, revs, q, bin_dir, log, plat, branch, exact):
                    status = 1
            else:
                restore(root, pkg, s3, revs, q, log, env, branch=branch)
        except CacheError as e:
            log(f"lean-cache: {pkg}: {e}; building from source")
            status = 1
        except (OSError, ValueError, ET.ParseError) as e:     # urllib's errors are OSErrors
            log(f"lean-cache: {pkg}: the cache is unreachable ({getattr(e, 'code', None) or type(e).__name__}); "
                f"building from source")
            status = max(status, want)
    return status


# ---------------------------------------------------------------- integration: build and publish

def out_dir(out: Path, pkg: str) -> Path:
    """Where `build` writes ``pkg``'s outputs below ``out``: one directory per package, never one inside another's (a
    package may sit inside another, as `security/Proofs` does in `security`)."""
    return out / urllib.parse.quote(pkg, safe="")


def _scope(name: str) -> str:
    """A manifest's package name as Lake scopes its cache (`«doc-gen4»` is `doc-gen4`)."""
    return name.removeprefix("«").removesuffix("»")


def unstaged(pkg: Path) -> set[str]:
    """The cache scopes `build` stages no dependency mappings for: the package's own, each path dependency's (a package
    with records of its own), and Mathlib's with every package in Mathlib's manifest (`lake exe cache get`, which
    `setup.sh` runs, provides them)."""
    m = json.loads((pkg / "lake-manifest.json").read_text()) if (pkg / "lake-manifest.json").is_file() else {}
    skip = {_scope(m.get("name") or pkg.name)}
    skip |= {_scope(p["name"]) for p in m.get("packages", []) if p.get("type") == "path"}
    if any(p["name"] == "mathlib" for p in m.get("packages", [])):
        skip.add("mathlib")
        mm = pkg / ".lake" / "packages" / "mathlib" / "lake-manifest.json"
        if mm.is_file():
            skip |= {_scope(p["name"]) for p in json.loads(mm.read_text()).get("packages", [])}
    return skip


def dep_mappings(cache: Path, skip: set[str], header: str) -> dict[str, str]:
    """{scope: mappings file} of each dependency Lake recorded outputs for in its cache ``cache``
    (`outputs/<scope>/<input hash>.json`), other than those in ``skip``: ``header`` (the first line of the root package's
    `outputs.jsonl`, the format's version) then one `[input hash, outputs]` line per module."""
    found = {}
    d = cache / "outputs"
    for s in sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []:
        if s in skip or not NAME.match(s):
            continue
        lines = [header]
        for f in sorted((d / s).glob("*.json")):
            o = json.loads(f.read_text())
            lines.append(json.dumps([f.stem, o["data"] if isinstance(o, dict) and "schemaVersion" in o else o],
                                    separators=(",", ":")))
        if len(lines) > 1:
            found[s] = "\n".join(lines) + "\n"
    return found


def stage_deps(pkg: Path, out: Path, env: dict, lake: list[str]) -> int:
    """Into ``out/stage``, beside the root package's stage: each dependency's mappings (`dep.<scope>.jsonl`) and the
    artifacts they name, staged by Lake (`lake cache stage` writes its mappings as the stage's `outputs.jsonl`, so each goes
    through a stage of its own first)."""
    stage, tmp = out / "stage", out / "dep-stage"
    header = (out / MAPPING).read_text().split("\n", 1)[0]
    for scope, text in dep_mappings(out / "lake-cache", unstaged(pkg), header).items():
        m = out / f"{DEP}{scope}.jsonl"
        m.write_text(text)
        shutil.rmtree(tmp, ignore_errors=True)
        rc = subprocess.run([*lake, "cache", "stage", str(m), str(tmp)], cwd=pkg, env=env).returncode
        if rc != 0:
            return rc
        for f in tmp.iterdir():
            if f.name != MAPPING and not (stage / f.name).exists():
                os.replace(f, stage / f.name)
        os.replace(m, stage / m.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


def build(pkg: Path, out: Path, lake: list[str] | None = None) -> int:
    """Inside the build's sandbox: `lake build` ``pkg`` with Lake's cache writable under ``out``, then `lake cache stage`
    the root package's outputs into ``out/stage``, and each dependency's that Mathlib's cache doesn't provide
    (`stage_deps`). Dependencies bound read-only at ``pkg/.lake/deps`` are built from a writable copy at
    ``pkg/.lake/packages``, deleted after. Lake's output goes to this process's and to ``out/build.log``. Lake's cache is
    deleted after: it holds copies of Mathlib's artifacts too (1.5 GB for core's package against Mathlib's 2 MB stage)."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LAKE_ARTIFACT_CACHE": "true", "LAKE_CACHE_DIR": str(out / "lake-cache"), "LEAN_CACHE": "off"}
    lake = lake or ["lake"]
    ro, packages = pkg / ".lake" / DEPS_RO, pkg / ".lake" / "packages"
    copied = ro.is_dir() and not packages.exists()
    try:
        if copied:
            # with the artifact cache on, Lake writes beside every dependency's artifacts (each `.hash`, and their modes)
            subprocess.run(["cp", "-a", "--reflink=auto", str(ro), str(packages)], check=True)
            subprocess.run(["chmod", "-R", "u+w", str(packages)], check=True)
        with open(out / BUILD_LOG, "wb") as f:
            p = subprocess.Popen([*lake, "build", "-o", str(out / MAPPING)], cwd=pkg, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            for line in p.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                f.write(line)
        if p.wait() != 0:
            return p.returncode
        shutil.rmtree(out / "stage", ignore_errors=True)
        rc = subprocess.run([*lake, "cache", "stage", str(out / MAPPING), str(out / "stage")], cwd=pkg, env=env).returncode
        return rc or stage_deps(pkg, out, env, lake) or stage_bin(pkg, out)
    finally:
        shutil.rmtree(out / "lake-cache", ignore_errors=True)
        if copied:
            shutil.rmtree(packages, ignore_errors=True)


def git(tree: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise CacheError(f"`git {' '.join(args)}` in {tree} failed: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def merged(tree: Path, rev: str, main: str = "origin/main") -> None:
    """CacheError unless ``tree`` is ``rev`` checked out cleanly (its build directories, which git ignores, aside) and
    ``rev`` is on ``main``."""
    if not REV.match(rev):
        raise CacheError(f"not a full commit id: {rev!r}")
    if git(tree, "rev-parse", "HEAD") != rev:
        raise CacheError(f"{tree} is not at {rev[:12]}")
    if subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", rev, main]).returncode != 0:
        raise CacheError(f"{rev[:12]} is not on {main}")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise CacheError(f"{tree} has changes outside its build directories: {dirty.splitlines()[0]} ...")


def stage_bin(pkg: Path, out: Path) -> int:
    """Into ``out``/bin, made anew: each executable `lake build` left in ``pkg``/.lake/build/bin (a regular file with an
    execute bit and a plain name; Lake's traces beside them are neither), copied by the build's user inside the sandbox.
    Empty for a package that builds none."""
    dest = out / BIN
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    src = pkg / ".lake" / "build" / "bin"
    for p in sorted(src.iterdir()) if src.is_dir() else []:
        st = p.lstat()
        if stat.S_ISREG(st.st_mode) and st.st_mode & 0o111 and NAME.match(p.name):
            shutil.copyfile(p, dest / p.name)
    return 0


def _opendir(name: str, at: int | None = None) -> int:
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=at)
    except OSError as e:
        raise CacheError(f"{name} is not a directory (a link, or missing: {e.strerror})") from None


def staged(out: Path, pkg: str, into: Path) -> dict[str, dict]:
    """{name: {sha256, bytes}} of every file `lake cache stage` wrote in `out_dir(out, pkg)`/stage, each copied into ``into`` and
    hashed as it is copied; CacheError on anything there but a regular file with one link and a plain name.

    The build's user wrote everything below ``out`` and this runs as root, so no link is followed below ``out``: each
    directory is opened relative to its parent and each file relative to the stage, all with `O_NOFOLLOW`, and a file's
    bytes are read once, from the descriptor whose `fstat` was checked, into ``into``, a directory only root writes. A link
    planted to the key (or a hard link to it) is refused, and a file changed afterwards changes nothing uploaded."""
    files = _staged(out, pkg, into, "stage")
    if MAPPING not in files:
        raise CacheError(f"{out}/{pkg}/stage has no {MAPPING}")
    return files


def staged_bin(out: Path, pkg: str, into: Path) -> dict[str, dict]:
    """{name: {sha256, bytes}} of every executable `build` copied into `out_dir(out, pkg)`/bin (`stage_bin`), read as
    `staged` reads the stage; {} where there is no such directory, and CacheError where it is a link."""
    return _staged(out, pkg, into, BIN, missing_ok=True)


def _absent(name: str, at: int) -> bool:
    return name not in os.listdir(at)


def _staged(out: Path, pkg: str, into: Path, part: str, missing_ok: bool = False) -> dict[str, dict]:
    fd = _opendir(str(out))
    try:
        for d in [out_dir(out, pkg).name, part]:
            if missing_ok and d == part and _absent(d, fd):
                return {}
            nxt = _opendir(d, fd)
            os.close(fd)
            fd = nxt
        files, total = {}, 0
        for name in sorted(os.listdir(fd)):
            where = f"{out}/{pkg}/{part}/{name}"
            if not NAME.match(name):
                raise CacheError(f"{where!r} is not a plain file: its name")
            try:
                f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            except OSError as e:
                raise CacheError(f"{where} is not a plain file ({e.strerror})") from None
            try:
                st = os.fstat(f)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                    raise CacheError(f"{where} is not a plain file (mode {st.st_mode:o}, {st.st_nlink} links)")
                total += st.st_size
                if total > MAX_BYTES:
                    raise CacheError(f"{out}/{pkg}/{part} holds more than the {MAX_BYTES >> 30} GiB a package may")
                h, n = hashlib.sha256(), 0
                with open(into / name, "xb") as w:
                    while chunk := os.read(f, 1 << 20):
                        n += len(chunk)
                        if n > st.st_size:
                            break
                        h.update(chunk)
                        w.write(chunk)
                if n != st.st_size:
                    raise CacheError(f"{where} changed size while it was read")
            finally:
                os.close(f)
            files[name] = {"sha256": h.hexdigest(), "bytes": n}
    finally:
        os.close(fd)
    return files


def put_blob(s3: S3, p: Path, f: dict, blobs: str = BLOBS) -> bool:
    """``p`` under its sha256 below ``blobs``; True when it was uploaded, False when the same bytes were already stored."""
    body = p.read_bytes()
    if hashlib.sha256(body).hexdigest() != f["sha256"]:
        raise CacheError(f"{p} is not the file that was hashed")
    key = blobs + f["sha256"]
    if s3.put(key, body):
        return True
    have = s3.head(key)
    if have is None or have["etag"] != hashlib.md5(body).hexdigest() or have["bytes"] != f["bytes"]:
        raise CacheError(f"{key} is already stored with other bytes (ETag {have and have['etag']})")
    return False


def publish_one(tree: Path, rev: str, pkg: str, out: Path, s3: S3, log=print, plat: str | None = None,
                prefix: str = TOP) -> str:
    """Upload one package's build of ``rev``, staged in `out_dir(out, pkg)`/stage and /bin, under ``prefix``: its blobs, then
    its record. Returns what happened."""
    ident = identity(tree, pkg, plat)
    with tempfile.TemporaryDirectory(prefix="lean-cache-publish-") as tmp:
        snap, snap_bin = Path(tmp) / "stage", Path(tmp) / BIN
        snap.mkdir()
        snap_bin.mkdir()
        files = staged(out, pkg, snap)
        bins = staged_bin(out, pkg, snap_bin)
        rec = json.dumps({"schema": SCHEMA, "rev": rev, **ident, "files": files, BIN: bins}, indent=1,
                         sort_keys=True).encode() + b"\n"
        key = record_key(ident, rev, prefix)
        old = s3.get(key)
        if old is not None:
            if old != rec:
                # a rerun of one commit: its first build stays, every blob of it checked on `get`
                log(f"lean-cache: {pkg}: {key} is already published, from a build whose artifacts differ from this one's; "
                    f"kept it")
                return "kept"
            return "already published"
        t0 = time.monotonic()
        uploads = [(snap / n, f) for n, f in sorted(files.items())] + [(snap_bin / n, f) for n, f in sorted(bins.items())]
        with ThreadPoolExecutor(THREADS) as ex:
            new = list(ex.map(lambda pf: put_blob(s3, pf[0], pf[1], prefix + "blobs/"), uploads))
    if not s3.put(key, rec) and s3.get(key) != rec:
        raise CacheError(f"{key} was published meanwhile with other contents")
    up = sum(f["bytes"] for (_, f), was in zip(uploads, new) if was)
    deps = sum(mapping(n) for n in files) - 1
    log(f"lean-cache: {pkg}: published {key} ({len(files) - 1 - deps} artifacts, {deps} dependencies, {len(bins)} "
        f"executables; {sum(new)} new blobs, {up >> 20} MiB, in {time.monotonic() - t0:.0f} s)")
    return "published"


def publish(tree: Path, rev: str, out: Path, pkgs: list[str], s3: S3, log=print, main: str | None = None,
            plat: str | None = None, branch: str | None = None) -> int:
    """`publish`: each package in ``pkgs`` from `out_dir(out, pkg)`/stage and /bin, under main's prefix, or under ``branch``'s,
    for a commit on ``main`` (by default `origin/main`, or `origin/<branch>` with ``branch``). 0 when each was published
    (or already was), 1 when any was refused. ``out`` is a directory root made; run it only once no process of the build's
    user is left."""
    prefix = prefix_of(branch)
    merged(tree, rev, main or (f"origin/{branch}" if branch else "origin/main"))
    status = 0
    for pkg in pkgs:
        try:
            publish_one(tree, rev, pkg, out, s3, log, plat, prefix)
        except CacheError as e:
            log(f"lean-cache: {pkg}: not published: {e}")
            status = 1
    return status


def prune(s3: S3, keep: int = 20, min_age_s: float = 7 * 86400, now: float | None = None, dry_run: bool = False,
          log=print, prefix: str = TOP) -> int:
    """Delete the records under ``prefix`` of all but the ``keep`` newest commits published there, and the blobs no remaining
    record there names, each only once it is older than ``min_age_s`` (the lock refuses earlier)."""
    now = time.time() if now is None else now
    blob_prefix = prefix + "blobs/"
    objs = s3.list(prefix)
    records = [o for o in objs if o["key"].endswith(RECORD)]

    def _rev(o: dict) -> str:
        return o["key"].rsplit("/", 1)[1][: -len(RECORD)]

    newest: dict[str, float] = {}
    for o in records:
        newest[_rev(o)] = max(newest.get(_rev(o), 0), o["modified"])
    kept_revs = set(sorted(newest, key=lambda r: -newest[r])[:keep])
    drop = [o for o in records if _rev(o) not in kept_revs and now - o["modified"] >= min_age_s]
    dropped = {o["key"] for o in drop}
    needed = set()
    for o in records:
        if o["key"] in dropped:
            continue
        raw = s3.get(o["key"])
        try:
            rec = json.loads(raw) if raw is not None else None
            needed |= {f["sha256"] for f in [*rec["files"].values(), *rec.get(BIN, {}).values()]}
        except (ValueError, KeyError, TypeError, AttributeError):
            log(f"lean-cache: prune: {o['key']} is unreadable; deleting nothing")
            return 1
    blobs = [o for o in objs if o["key"].startswith(blob_prefix) and o["key"][len(blob_prefix):] not in needed
             and now - o["modified"] >= min_age_s]
    for o in drop + blobs:
        if not dry_run:
            s3.delete(o["key"])
    total = sum(o["bytes"] for o in objs)
    log(f"lean-cache: prune: {len(objs)} objects, {total >> 20} MiB; {len(kept_revs)} commits kept; "
        f"{'would delete' if dry_run else 'deleted'} {len(drop)} records and {len(blobs)} blobs "
        f"({sum(o['bytes'] for o in blobs) >> 20} MiB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get", help="restore main's cached build of each package, or with --bin write its executables")
    g.add_argument("packages", nargs="*")
    g.add_argument("--root", type=Path, default=None)
    g.add_argument("--bin", type=Path, default=None, metavar="DIR",
                   help="write each package's executables to DIR/<package>/<name> (the package's path quoted: "
                        "`security%%2FProofs`), each checked first, and the commit they were built at to "
                        "DIR/<package>.json; for the prover and development, not for a verdict of record (built from the "
                        "commit's source by whoever relies on it)")
    g.add_argument("--exact", action="store_true", help="with --bin: only HEAD's own build, never an ancestor's")
    g.add_argument("--branch", help="read this branch's builds (`branch/<name>/`) too: the newest ancestor of HEAD across "
                                    "it and main's is taken, main's on a tie")
    g.add_argument("--platform", help="the target triple of the executables (default: `lean --version`'s, else this "
                                      "machine's `<arch>-unknown-linux-gnu`)")
    b = sub.add_parser("build", help="(integration, in the sandbox) build PKG writing Lake's cache, and stage it")
    b.add_argument("package", type=Path)
    b.add_argument("out", type=Path)
    p = sub.add_parser("publish", help="(integration, root, outside the sandbox) upload staged builds of a merged commit")
    p.add_argument("packages", nargs="+")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key-file", default=KEY_FILE)
    p.add_argument("--main", help="the ref the commit must be on (default `origin/main`, or `origin/<name>` with "
                                  "--branch); `vy-lean-sandbox checkout` fetches the line it builds into "
                                  "`refs/remotes/origin/main`, whichever branch that is")
    p.add_argument("--branch", help="publish a commit of `origin/<name>` under `branch/<name>/` (not main's prefix)")
    p.add_argument("--platform", help="the toolchain's target triple, so root never runs elan (default: `lean --version`)")
    r = sub.add_parser("prune", help="(integration) delete what the last N published commits don't need")
    r.add_argument("--keep", type=int, default=20)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--key-file", default=KEY_FILE)
    r.add_argument("--branch", help="prune `branch/<name>/` (default: main's prefix)")
    a = ap.parse_args(argv)
    if a.cmd == "get" and a.exact and a.bin is None:
        ap.error("--exact goes with --bin")
    if a.cmd == "get":
        here = Path.cwd()
        root = a.root or Path(subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"],
                                             capture_output=True, text=True).stdout.strip() or here)
        pkgs = [os.path.relpath((here / x).resolve(), root.resolve()) for x in a.packages] or packages(root)
        try:
            return get(root.resolve(), pkgs, bin_dir=a.bin and a.bin.resolve(), branch=a.branch, plat=a.platform,
                       exact=a.exact)
        except CacheError as e:
            print(f"lean-cache: {e}", file=sys.stderr)
            return 2
    if a.cmd == "build":
        return build(a.package.resolve(), a.out.resolve())
    try:
        s3 = writer(a.key_file)
    except (OSError, CacheError) as e:
        print(f"lean-cache: no publisher key: {e}", file=sys.stderr)
        return 2
    if a.cmd == "publish":
        try:
            return publish(a.tree.resolve(), a.rev, Path(os.path.abspath(a.out)), a.packages, s3, main=a.main,
                           plat=a.platform, branch=a.branch)
        except CacheError as e:
            print(f"lean-cache: nothing published: {e}", file=sys.stderr)
            return 1
    try:
        return prune(s3, a.keep, dry_run=a.dry_run, prefix=prefix_of(a.branch))
    except CacheError as e:
        print(f"lean-cache: nothing pruned: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

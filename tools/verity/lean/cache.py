#!/usr/bin/env python3
"""The Lean build cache: main's compiled modules, which integration publishes on merge and development machines restore,
every file checked against its sha256 first. Standard library only, with `git` and `lake`. `setup.sh` runs it with the
machine's own `python3`, which may be older than the workspace's 3.12, so it keeps to syntax Python 3.10 parses.

    python tools/verity/lean/cache.py get [PKG...]       restore main's cached build of each package (default: every package with
                                                  a `lean-audit.json`) into Lake's local cache; `lake build` then unpacks
                                                  those modules instead of compiling them
    python tools/verity/lean/cache.py build PKG OUT      (integration, inside the build's sandbox) `lake build` PKG writing Lake's
                                                  cache under OUT, and stage what it built in OUT/stage
    python tools/verity/lean/cache.py publish --tree DIR --rev SHA --out DIR PKG...
                                                  (integration, as root outside the sandbox) hash each package's
                                                  `out_dir(OUT, PKG)`/stage and upload it with the key in /etc/verity/lean-cache.env
    python tools/verity/lean/cache.py prune [--keep N] [--dry-run]
                                                  (integration) delete what no record of the last N published commits needs

The bucket (`LEAN_CACHE_BUCKET`, `verity-lean-cache`) holds everything under `main/`, which a 7-day lock keeps from being
overwritten or deleted:
- `main/<toolchain>/<platform>/<package>/<manifest>/<rev>.sha256.json`: one record per published commit and package, the
  sha256 and size of every file `lake cache stage` wrote: Lake's `outputs.jsonl` (each module's input hash to its archive)
  and one archive per module, then for each dependency the build compiled (ArkLib and the packages it brings; not Mathlib
  and its own dependencies, which Mathlib's cache provides, nor a path dependency, a package of its own) its mappings
  `dep.<scope>.jsonl` and its modules' artifacts;
- `main/blobs/<sha256>`: each of those files under its own hash, shared across commits.

The client downloads every file itself and checks it against the record before Lake sees it, then hands them to Lake
(`lake cache unstage`, then `lake cache unstage DIR <scope>` per dependency). Lake never reads the bucket, because its
hashes aren't cryptographic. A module whose input hash
has no checked file is built from source, as is every module after a mismatch, when the kill switch (the docs site's
`/api/lean-cache/state`) is disabled, quarantines the scope or can't be reached, or when `LEAN_CACHE=off`. Lake's input
hashes don't depend on where a checkout is, so a build published from `/build/verity` restores in any checkout.

The publisher signs its uploads directly with the bucket-scoped key, as root, outside the sandbox the build ran in; no
broader credential sits on the machine to mint a shorter-lived one. Every write is conditional (`If-None-Match: *`). A
412 or 409 means the object exists: a blob then must be stored with its MD5 as ETag (blobs go up in one part, so it is),
and a record must be byte-identical, or the package isn't published. The record goes last, so every blob it names exists.
Integration's own builds never read the cache: `audit.py` runs Lake with `LEAN_CACHE=off`, and with `LAKE_ARTIFACT_CACHE=false`
or, in check's audit, a local Lake cache that only a passing audit on that node writes (`lean_audit.LakeArtifacts`).
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
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
RECORD = ".sha256.json"
MAPPING = "outputs.jsonl"
#: a staged dependency's mappings are `dep.<scope>.jsonl`, beside the root package's `outputs.jsonl`
DEP = "dep."
#: Lake's output of `build`, beside its stage
BUILD_LOG = "build.log"
#: where the sandbox binds a package's fetched dependencies read-only; `build` works on its own copy at `.lake/packages`
DEPS_RO = "deps"
TOOL = "tools/verity/lean/"
STATE_URL = "https://website-docs-sage.vercel.app/api/lean-cache/state"
STATE_TTL_S = 300
KEY_FILE = "/etc/verity/lean-cache.env"
#: the read-only credential every development machine holds (the Cursor secrets)
READ_ENV = ("LEAN_CACHE_R2_ACCESS_KEY_ID", "LEAN_CACHE_R2_SECRET_ACCESS_KEY", "LEAN_CACHE_R2_ENDPOINT", "LEAN_CACHE_BUCKET")
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


class S3:
    """The few S3 calls the cache makes, path-style against one bucket."""

    def __init__(self, endpoint: str, bucket: str, key_id: str, secret: str, region: str = "auto",
                 now=lambda: datetime.datetime.now(datetime.timezone.utc)):
        u = urllib.parse.urlsplit(endpoint)
        if u.scheme not in ("https", "http") or not u.netloc:
            raise ValueError(f"not an endpoint URL: {endpoint!r}")
        self.scheme, self.host, self.base = u.scheme, u.netloc, u.path.rstrip("/")
        self.bucket, self.key_id, self.secret, self.region, self.now = bucket, key_id, secret, region, now

    def open(self, method: str, key: str = "", query: dict[str, str] | None = None, body: bytes | None = None,
             headers: dict[str, str] | None = None, timeout: float = 120):
        query = query or {}
        path = f"{self.base}/{self.bucket}" + (f"/{key}" if key else "")
        h = {"x-amz-date": self.now().strftime("%Y%m%dT%H%M%SZ"),
             "x-amz-content-sha256": hashlib.sha256(body).hexdigest() if body is not None else EMPTY_SHA, **(headers or {})}
        auth = sign(method, self.host, path, query, h, self.key_id, self.secret, self.region)
        url = f"{self.scheme}://{self.host}{_enc(path, '/')}" + (f"?{_query(query)}" if query else "")
        req = urllib.request.Request(url, data=body, method=method, headers={**h, "Authorization": auth})
        if body is not None:
            req.add_header("Content-Type", "application/octet-stream")
        return urllib.request.urlopen(req, timeout=timeout)

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
                            "modified": when.replace(tzinfo=datetime.timezone.utc).timestamp()})
            if root.findtext(f"{ns}IsTruncated") != "true":
                return out
            token = root.findtext(f"{ns}NextContinuationToken")


def reader(env=os.environ) -> S3 | None:
    """The bucket through the read-only credential, or None when this machine has none."""
    v = [env.get(k, "") for k in READ_ENV]
    return S3(v[2], v[3], v[0], v[1]) if all(v) else None


def writer(path: str = KEY_FILE) -> S3:
    """The bucket through the publisher's key, read from ``path`` (`KEY=value` lines)."""
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
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


def scope(ident: dict) -> str:
    """The key prefix of a package's records (elan's spelling of the toolchain: `leanprover--lean4---v4.34.0`)."""
    tc = ident["toolchain"].replace("/", "--").replace(":", "---")
    return f"{TOP}{tc}/{ident['platform']}/{ident['package']}/{ident['manifest'][:16]}"


def record_key(ident: dict, rev: str) -> str:
    return f"{scope(ident)}/{rev}{RECORD}"


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
    total = 0
    for name, f in files.items():
        if not (isinstance(name, str) and NAME.match(name) and isinstance(f, dict) and isinstance(f.get("sha256"), str)
                and SHA.match(f["sha256"]) and type(f.get("bytes")) is int and f["bytes"] >= 0):
            raise CacheError(f"its record has a bad entry {name!r}")
        total += f["bytes"]
    if total > MAX_BYTES:
        raise CacheError(f"its record names {total >> 20} MiB, more than the {MAX_BYTES >> 30} GiB a package may")
    return rec


# ---------------------------------------------------------------- the client

def packages(root: Path) -> list[str]:
    """The packages `audit.py --all` audits: directories with a tracked `lean-audit.json`, the auditor's fixtures aside."""
    r = subprocess.run(["git", "-C", str(root), "ls-files", "*lean-audit.json"], capture_output=True, text=True)
    return sorted({f.rsplit("/", 1)[0] for f in r.stdout.split() if "/" in f and not f.startswith(TOOL)})


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
            raw = fetch(url) if fetch else urllib.request.urlopen(url, timeout=10).read()
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
    m = d / "lake-manifest.json"
    deps = [p for p in json.loads(m.read_text()).get("packages", []) if p.get("type") != "path"] if m.is_file() else []
    return not deps or all((d / ".lake" / "packages" / p["name"]).is_dir() for p in deps)


def restore(root: Path, pkg: str, s3: S3, revs: list[str], q: list[str], log=print, env=os.environ, plat: str | None = None,
            lake: list[str] | None = None) -> bool:
    """Main's newest cached build of ``pkg`` among ``revs`` into Lake's local cache, every file checked first. False when
    there is nothing to restore; CacheError when a check fails."""
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
    have = {o["key"][len(sc) + 1: -len(RECORD)] for o in s3.list(sc + "/") if o["key"].endswith(RECORD)}
    rev = next((r for r in revs if r in have), None)
    if rev is None:
        log(f"lean-cache: {pkg}: no cached build of an ancestor of HEAD ({sc}); building from source")
        return False
    raw = s3.get(record_key(ident, rev))
    if raw is None:
        raise CacheError(f"{record_key(ident, rev)} was listed but can't be read")
    rec = check_record(raw, ident, rev)
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
        with ThreadPoolExecutor(THREADS) as ex:
            list(ex.map(lambda nf: s3.download(BLOBS + nf[1]["sha256"], stage / nf[0], nf[1]["sha256"], nf[1]["bytes"]), todo))
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
    log(f"lean-cache: {pkg}: main's build at {rev[:12]} restored ({arts} artifacts"
        f"{with_deps}; {got >> 20} MiB in {len(todo)} files "
        f"fetched and checked in {time.monotonic() - t0:.0f} s, {len(rec['files']) - len(todo)} already here); "
        f"`lake build` unpacks them")
    return True


def get(root: Path, pkgs: list[str], env=os.environ, log=print, now=time.time, fetch_state=None) -> int:
    """`get`: 0 when every package was restored or had nothing to restore, 1 when a check failed (that package builds from
    source)."""
    if env.get("LEAN_CACHE", "").strip().lower() in OFF:
        log("lean-cache: off on this machine (LEAN_CACHE); building from source")
        return 0
    s3 = reader(env)
    if s3 is None:
        log(f"lean-cache: no read credential ({', '.join(READ_ENV)}); building from source")
        return 0
    cache = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity-lean-cache"
    ok, q, why = state(env.get("LEAN_CACHE_STATE_URL") or STATE_URL, cache, now(), fetch_state)
    if not ok:
        log(f"lean-cache: {why}; building from source")
        return 0
    revs, status = ancestors(root), 0
    for pkg in pkgs:
        try:
            restore(root, pkg, s3, revs, q, log, env)
        except CacheError as e:
            log(f"lean-cache: {pkg}: {e}; building from source")
            status = 1
        except (OSError, ValueError, ET.ParseError) as e:     # urllib's errors are OSErrors
            log(f"lean-cache: {pkg}: the cache is unreachable ({getattr(e, 'code', None) or type(e).__name__}); "
                f"building from source")
    return status


# ---------------------------------------------------------------- integration: build and publish

def out_dir(out: Path, pkg: str) -> Path:
    """Where `build` writes ``pkg``'s outputs below ``out``: one directory per package, never one inside another's (a
    package may sit inside another, as `Security/Proofs` does in `Security`)."""
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
        return rc or stage_deps(pkg, out, env, lake)
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
    fd = _opendir(str(out))
    try:
        for part in [out_dir(out, pkg).name, "stage"]:
            nxt = _opendir(part, fd)
            os.close(fd)
            fd = nxt
        files, total = {}, 0
        for name in sorted(os.listdir(fd)):
            where = f"{out}/{pkg}/stage/{name}"
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
                    raise CacheError(f"{out}/{pkg}/stage holds more than the {MAX_BYTES >> 30} GiB a package may")
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
    if MAPPING not in files:
        raise CacheError(f"{out}/{pkg}/stage has no {MAPPING}")
    return files


def put_blob(s3: S3, p: Path, f: dict) -> bool:
    """``p`` under its sha256; True when it was uploaded, False when the same bytes were already stored."""
    body = p.read_bytes()
    if hashlib.sha256(body).hexdigest() != f["sha256"]:
        raise CacheError(f"{p} is not the file that was hashed")
    key = BLOBS + f["sha256"]
    if s3.put(key, body):
        return True
    have = s3.head(key)
    if have is None or have["etag"] != hashlib.md5(body).hexdigest() or have["bytes"] != f["bytes"]:
        raise CacheError(f"{key} is already stored with other bytes (ETag {have and have['etag']})")
    return False


def publish_one(tree: Path, rev: str, pkg: str, out: Path, s3: S3, log=print, plat: str | None = None) -> str:
    """Upload one package's build of ``rev``, staged in `out_dir(out, pkg)`/stage: its blobs, then its record. Returns what
    happened."""
    ident = identity(tree, pkg, plat)
    with tempfile.TemporaryDirectory(prefix="lean-cache-publish-") as tmp:
        snap = Path(tmp)
        files = staged(out, pkg, snap)
        rec = json.dumps({"schema": SCHEMA, "rev": rev, **ident, "files": files}, indent=1, sort_keys=True).encode() + b"\n"
        key = record_key(ident, rev)
        old = s3.get(key)
        if old is not None:
            if old != rec:
                raise CacheError(f"{key} is already published with other contents")
            return "already published"
        t0 = time.monotonic()
        with ThreadPoolExecutor(THREADS) as ex:
            new = list(ex.map(lambda nf: put_blob(s3, snap / nf[0], nf[1]), sorted(files.items())))
    if not s3.put(key, rec) and s3.get(key) != rec:
        raise CacheError(f"{key} was published meanwhile with other contents")
    up = sum(f["bytes"] for (n, f), was in zip(sorted(files.items()), new) if was)
    deps = sum(mapping(n) for n in files) - 1
    log(f"lean-cache: {pkg}: published {key} ({len(files) - 1 - deps} artifacts, {deps} dependencies; {sum(new)} new "
        f"blobs, {up >> 20} MiB, in "
        f"{time.monotonic() - t0:.0f} s)")
    return "published"


def publish(tree: Path, rev: str, out: Path, pkgs: list[str], s3: S3, log=print, main: str = "origin/main",
            plat: str | None = None) -> int:
    """`publish`: each package in ``pkgs`` from `out_dir(out, pkg)`/stage. 0 when each was published (or already was), 1
    when any was refused. ``out`` is a directory root made; run it only once no process of the build's user is left."""
    merged(tree, rev, main)
    status = 0
    for pkg in pkgs:
        try:
            publish_one(tree, rev, pkg, out, s3, log, plat)
        except CacheError as e:
            log(f"lean-cache: {pkg}: not published: {e}")
            status = 1
    return status


def prune(s3: S3, keep: int = 20, min_age_s: float = 7 * 86400, now: float | None = None, dry_run: bool = False,
          log=print) -> int:
    """Delete the records of all but the ``keep`` newest published commits, and the blobs no remaining record names, each
    only once it is older than ``min_age_s`` (the lock refuses earlier)."""
    now = time.time() if now is None else now
    objs = s3.list(TOP)
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
            files = json.loads(raw)["files"] if raw is not None else None
            needed |= {f["sha256"] for f in files.values()}
        except (ValueError, KeyError, TypeError, AttributeError):
            log(f"lean-cache: prune: {o['key']} is unreadable; deleting nothing")
            return 1
    blobs = [o for o in objs if o["key"].startswith(BLOBS) and o["key"][len(BLOBS):] not in needed
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
    g = sub.add_parser("get", help="restore main's cached build of each package")
    g.add_argument("packages", nargs="*")
    g.add_argument("--root", type=Path, default=None)
    b = sub.add_parser("build", help="(integration, in the sandbox) build PKG writing Lake's cache, and stage it")
    b.add_argument("package", type=Path)
    b.add_argument("out", type=Path)
    p = sub.add_parser("publish", help="(integration, root, outside the sandbox) upload staged builds of a merged commit")
    p.add_argument("packages", nargs="+")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--rev", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key-file", default=KEY_FILE)
    p.add_argument("--main", default="origin/main")
    p.add_argument("--platform", help="the toolchain's target triple, so root never runs elan (default: `lean --version`)")
    r = sub.add_parser("prune", help="(integration) delete what the last N published commits don't need")
    r.add_argument("--keep", type=int, default=20)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--key-file", default=KEY_FILE)
    a = ap.parse_args(argv)
    if a.cmd == "get":
        here = Path.cwd()
        root = a.root or Path(subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"],
                                             capture_output=True, text=True).stdout.strip() or here)
        pkgs = [os.path.relpath((here / x).resolve(), root.resolve()) for x in a.packages] or packages(root)
        return get(root.resolve(), pkgs)
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
                           plat=a.platform)
        except CacheError as e:
            print(f"lean-cache: nothing published: {e}", file=sys.stderr)
            return 1
    return prune(s3, a.keep, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())

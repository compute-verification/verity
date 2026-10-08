"""The local store (README §3): content-addressed blobs, canonical manifests, write-once attempts, append-only labels, and the
disposable catalog, under `~/.research/store` (or `$RESEARCH_STORE`).

    objects/sha256/<hex>          blobs, temp + hard-link (never overwritten, mode 0444)
    manifests/<hex>.json          canonical artifact manifests; hex == art id hex
    attempts/<run id>.json        canonical attempt manifests; write-once, a different second content is AttemptExists
    labels/<target>/<ts>-<by>.json one immutable assertion each (O_EXCL; a same-second collision gets a `~N` suffix)
    trees/<hex>/                  LOCAL ONLY: default materialisation of tree payloads for `fetch(art, to=None)`
    catalog.sqlite                LOCAL ONLY: index.Index; `reindex()` rebuilds it from the three directories above
    scratch/                      LOCAL ONLY: tool caches keyed by reuse_key (owned by the run integration, §6.5)

Replica resolution for a blob (`fetch`): local objects -> repo replica map `<repo>/fixtures/artifacts.json` (`{art id: repo-relative
path}`, every blob sha256-checked against the manifest before use, mismatch warns and falls through) -> remote (`remote.py`).  Bytes
that arrive from a repo or the remote are checked against the manifest before they enter `objects/`.  `push` uploads an artifact's
blobs concurrently (conditional create per object, verified by HEAD size + MD5 ETag or by read-back) and the manifest last; `verify`
is the standalone PRESERVED check (§7): remote head size + full read-back sha256 for the manifest and every blob.  Nothing is called
PRESERVED without a remote, and the replica record says which check was done (`verified`: `etag-md5` | `sha256-readback`).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

from ..secret_hint import FRESH_SHELL, lost
from . import canon, ids, kinds
from .index import Index
from .model import ArtifactManifest, Label, Payload, TreeFile, utc_now, validate_attempt_doc
from .remote import ConfigError, Remote, config_path, load_remote
from .remote_s3 import is_md5_etag

DEFAULT_ROOT = Path.home() / ".research" / "store"
REPLICA_MAP = Path("fixtures") / "artifacts.json"
SNAPSHOT_KIND = "dataset-snapshot/v1"
VERIFY_HEAD, VERIFY_READBACK = "etag-md5", "sha256-readback"     # what a replica record's `verified` says was checked
VERIFY_MULTIPART = "etag-multipart"                              # ... a > 5 GiB blob: the service's multipart ETag == the one the local parts yield
VERIFY_MODES = ("head", "readback")                             # `push(verify=...)`: HEAD size + MD5 ETag, or download + sha256
WHERE_JOBS = 16                                                 # concurrent HEADs in `where`
REFRESH_JOBS = 32                                               # concurrent GETs in `refresh`
REFRESH_PAYLOAD_MAX = 1 << 20                                   # `refresh(payload_kinds=...)` fetches file payloads up to this size
WRITE_THROUGH_ENV = "RESEARCH_WRITE_THROUGH"
_UNSET = object()


class StoreError(Exception):
    """A store operation that cannot proceed; the message is the reason."""


class NotFound(StoreError):
    """No replica (local, repo, remote) holds the requested object."""


class AttemptExists(StoreError):
    """`attempts/<id>.json` exists with different content; attempts are write-once."""


class IntegrityError(StoreError):
    """Bytes did not match the sha256 the manifest promised."""


def store_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_STORE", DEFAULT_ROOT)).expanduser()


def find_repo(start: Path | None = None) -> Path | None:
    """`$RESEARCH_REPO`, else the nearest ancestor of cwd holding `fixtures/artifacts.json`."""
    if os.environ.get("RESEARCH_REPO"):
        return Path(os.environ["RESEARCH_REPO"]).expanduser()
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / REPLICA_MAP).is_file():
            return d
    return None


def sha256_file(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def object_key(hex_: str) -> str:
    return f"objects/sha256/{hex_}"


def manifest_key(hex_: str) -> str:
    return f"manifests/{hex_}.json"


def attempt_key(run_id: str) -> str:
    return f"attempts/{run_id}.json"


def _warn(msg: str) -> None:
    print(f"research data: {msg}", file=sys.stderr)


def write_once(dst: Path, data: bytes | Path, sha256: str | None = None) -> bool:
    """Temp file beside `dst`, then hard-link into place: the first writer wins, nothing is ever overwritten.  Returns whether this
    call created it.  Falls back to rename where the filesystem refuses links.  `sha256`: the copy must hash to it, else IntegrityError
    and nothing is written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    tmp = dst.parent / f".tmp-{dst.name}-{os.getpid()}-{secrets.token_hex(4)}"
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        shutil.copyfile(data, tmp)
    try:
        if sha256 is not None and (got := sha256_file(tmp))[0] != sha256:
            # custody stages a run's files as hard links: a writer still appending to one (a check's orphaned suite) changed it here
            raise IntegrityError(f"{data} changed between hashing and storing ({sha256[:16]} hashed, {got[0][:16]} copied, "
                                 f"{got[1]} bytes): is something still writing it?")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.chmod(tmp, 0o444)
    try:
        os.link(tmp, dst)
        return True
    except FileExistsError:
        return False
    except OSError:
        if dst.exists():
            return False
        os.replace(tmp, dst)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def write_through_wanted(explicit: str | os.PathLike[str] | None = None) -> bool:
    """Whether `label` / `put` upload in the same command (README §7.2).  `$RESEARCH_WRITE_THROUGH` (1/0) decides when set; else the
    store.toml `[remote] write_through` flag; else, with no store.toml at the default path and `$RESEARCH_STORE_CONFIG` unset, whether
    the environment names the R2 remote (`R2_ENDPOINT` + `R2_BUCKET`: a cloud machine).  Off on the laptop until its store.toml says so."""
    v = os.environ.get(WRITE_THROUGH_ENV, "").strip().lower()
    if v:
        return v not in ("0", "false", "no", "off")
    p = config_path(explicit)
    if p.is_file():
        try:
            return (tomllib.loads(p.read_text(encoding="utf-8")).get("remote") or {}).get("write_through") is True
        except (OSError, tomllib.TOMLDecodeError):
            return False
    return explicit is None and "RESEARCH_STORE_CONFIG" not in os.environ and bool(os.environ.get("R2_ENDPOINT") and os.environ.get("R2_BUCKET"))


def _rel_files(tree: Path) -> list[Path]:
    """Every file under `tree` (symlinks to files included by content), deterministic order."""
    out = []
    for dirpath, dirnames, filenames in os.walk(tree):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_file():
                out.append(p)
    return out


class LocalStore:
    def __init__(self, root: str | os.PathLike[str] | None = None, *, config_path: str | os.PathLike[str] | None = None, repo: Path | None = None):
        self.root = store_root(root)
        for d in ("objects/sha256", "manifests", "attempts", "labels", "trees", "scratch"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._config_path = config_path
        self._repo: Any = repo if repo is not None else _UNSET
        self._remote: Any = _UNSET
        self._index: Index | None = None
        self._map: dict[str, str] | None = None

    # ---- properties and paths -----------------------------------------------------------------------------------------------------------

    @property
    def index(self) -> Index:
        """The catalog; reopened when `reindex` (here or in another process) has renamed a rebuilt file over it."""
        path = self.root / "catalog.sqlite"
        if self._index is not None:
            try:
                if path.stat().st_ino != self._index_ino:
                    self.close()
            except FileNotFoundError:
                pass
        if self._index is None:
            self._index = Index(path)
            self._index_ino = path.stat().st_ino
        return self._index

    @property
    def remote(self) -> Remote | None:
        """The configured remote (README §7), or None: then the local store is the only replica.

        A remote that is configured but unusable here (e.g. `store.toml` names an S3 bucket and the credential env vars are not set,
        as on a pod or in a test) degrades to None so the local store keeps working; the reason is kept in `remote_error` and is
        raised by `push`/`push_attempt`, the only operations that need the remote."""
        if self._remote is _UNSET:
            try:
                self._remote = load_remote(self._config_path)
                self.remote_error: str | None = None
            except ConfigError as e:
                self._remote = None
                self.remote_error = str(e)
        return self._remote

    @property
    def write_through(self) -> bool:
        return write_through_wanted(self._config_path)

    def _require_remote(self) -> Remote:
        r = self.remote
        if r is None:
            why = getattr(self, "remote_error", None)
            raise StoreError(why or lost("R2_ENDPOINT", "R2_BUCKET") or self.remote_lost()
                             or "no remote configured (~/.research/store.toml or $RESEARCH_STORE_CONFIG)")
        return r

    def remote_lost(self) -> str:
        """'' unless nothing here names a remote (no `$RESEARCH_WRITE_THROUGH`, store.toml, `$RESEARCH_STORE_CONFIG`, `R2_ENDPOINT` or
        `R2_BUCKET`) yet the catalog holds artifacts pulled from or verified on one; then the hint.  A cloud shell can lose
        $CLOUD_AGENT_INJECTED_SECRET_NAMES along with the secrets (so `lost` says nothing), and its writes would stay local unannounced."""
        if (self._config_path is not None or os.environ.get(WRITE_THROUGH_ENV, "").strip() or "RESEARCH_STORE_CONFIG" in os.environ
                or os.environ.get("R2_ENDPOINT") or os.environ.get("R2_BUCKET") or config_path().is_file()):
            return ""
        if self.index.db.execute("SELECT 1 FROM replicas WHERE location='remote' UNION ALL SELECT 1 FROM artifacts WHERE remote=1 LIMIT 1").fetchone() is None:
            return ""
        return ("R2_ENDPOINT and R2_BUCKET are not in this shell, but this store has used the remote (its catalog holds artifacts from it): "
                f"the shell lost the VM's secrets; {FRESH_SHELL}")

    @property
    def repo(self) -> Path | None:
        if self._repo is _UNSET:
            self._repo = find_repo()
        return self._repo

    def replica_map(self) -> dict[str, str]:
        """`<repo>/fixtures/artifacts.json`: art id -> repo-relative path of the payload (a file, or a tree's directory)."""
        if self._map is None:
            self._map = {}
            if self.repo and (self.repo / REPLICA_MAP).is_file():
                try:
                    doc = json.loads((self.repo / REPLICA_MAP).read_text(encoding="utf-8"))
                    self._map = {k: v for k, v in doc.items() if ids.is_art(k) and isinstance(v, str)} if isinstance(doc, dict) else {}
                except (OSError, json.JSONDecodeError) as e:
                    _warn(f"ignoring unreadable replica map {self.repo / REPLICA_MAP}: {e}")
        return self._map

    def object_path(self, hex_: str) -> Path:
        return self.root / object_key(hex_)

    def manifest_path(self, hex_: str) -> Path:
        return self.root / manifest_key(hex_)

    def attempt_path(self, run_id: str) -> Path:
        return self.root / attempt_key(run_id)

    def labels_dir(self, target: str) -> Path:
        return self.root / "labels" / ids.target_dirname(target)

    def artifact_ids(self) -> list[str]:
        return sorted("art:" + p.stem for p in (self.root / "manifests").glob("*.json") if ids.is_sha256(p.stem))

    def attempt_ids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "attempts").glob("*.json") if ids.is_run_id(p.stem))

    # ---- artifacts ----------------------------------------------------------------------------------------------------------------------

    def put_artifact(self, kind: str, meta: dict, refs: dict | None = None, *, file: Path | None = None, tree: Path | None = None) -> str:
        """Hash the payload, store its blobs, write the canonical manifest, index; return the art id.  Idempotent: the same content
        yields the same id and touches nothing that already exists."""
        if file is not None and tree is not None:
            raise ValueError("put_artifact takes file= or tree=, not both")
        blobs: list[tuple[Path, str]] = []
        payload = None
        if file is not None:
            file = Path(file)
            if not file.is_file():
                raise FileNotFoundError(str(file))
            h, n = sha256_file(file)
            payload, blobs = Payload.file(h, n, file.name), [(file, h)]
        elif tree is not None:
            tree = Path(tree)
            if not tree.is_dir():
                raise NotADirectoryError(str(tree))
            files = []
            for p in _rel_files(tree):
                h, n = sha256_file(p)
                files.append(TreeFile(p.relative_to(tree).as_posix(), h, n))
                blobs.append((p, h))
            payload = Payload.tree(files)
        m = ArtifactManifest(kind=kind, meta={} if meta is None else meta, refs=dict(refs or {}), payload=payload).validate()
        if not kinds.is_known(kind):
            _warn(f"unknown kind {kind!r} (known: {', '.join(sorted(kinds.KINDS))}); storing anyway")
        for p, h in blobs:
            self._store_blob(p, h)
        return self.put_manifest(m)

    def put_manifest(self, m: ArtifactManifest) -> str:
        """Write a validated manifest whose blobs are already in `objects/` (or elsewhere: refs are syntax-checked only)."""
        m.validate()
        art = m.id
        write_once(self.manifest_path(ids.hex_of(art)), m.canonical())
        self.index.index_artifact(m.to_doc(), art, local=self._blobs_local(m), repo_path=self.replica_map().get(art))
        return art

    def _store_blob(self, src: Path, hex_: str, verify: bool = True) -> Path:
        dst = self.object_path(hex_)
        if not dst.exists():
            write_once(dst, src, hex_ if verify else None)
        return dst

    def _blobs_local(self, m: ArtifactManifest) -> bool:
        return all(self.object_path(h).exists() for h, _ in m.blobs())

    def _read_manifest(self, path: Path, hex_: str) -> ArtifactManifest:
        data = path.read_bytes()
        if canon.sha256_bytes(data) != hex_:
            raise IntegrityError(f"{path}: content hash does not match art:{hex_}")
        return ArtifactManifest.from_doc(json.loads(data))

    def get_manifest(self, art_id: str) -> ArtifactManifest:
        """Local manifest, else the remote's copy (then cached locally); NotFound otherwise."""
        if not ids.is_art(art_id):
            raise ValueError(f"not an art: id: {art_id!r}")
        hex_ = ids.hex_of(art_id)
        p = self.manifest_path(hex_)
        if p.exists():
            return self._read_manifest(p, hex_)
        r = self.remote
        if r is not None:
            tmp = p.parent / f".tmp-{hex_}-{os.getpid()}-{threading.get_ident()}"
            try:
                r.get(manifest_key(hex_), tmp)
            except KeyError:
                raise NotFound(f"{art_id}: no manifest locally or on {r.describe()}") from None
            try:
                m = self._read_manifest(tmp, hex_)
                write_once(p, tmp)
            finally:
                tmp.unlink(missing_ok=True)
            self.index.index_artifact(m.to_doc(), art_id, local=self._blobs_local(m), remote=True, repo_path=self.replica_map().get(art_id))
            return m
        raise NotFound(f"{art_id}: no manifest locally (no remote configured)")

    def resolve_art(self, ref: str, limit: int = 2) -> list[str]:
        """Full ids an `art:` id or abbreviation (>= 8 hex digits) names, at most `limit`: the local catalog first, else the remote's
        `manifests/` listing under that prefix (a pod's upload this machine never pulled).  Several = ambiguous; none = unknown."""
        if ids.is_art(ref):
            return [ref]
        if not ids.is_short_art(ref):
            raise ValueError(f"not an art: id or abbreviation: {ref!r}")
        hits = self.index.resolve_art_prefix(ref, limit=limit)
        if hits or self.remote is None:
            return hits
        pre = ids.hex_of(ref)
        keys = self.remote.list(f"manifests/{pre}")
        return sorted("art:" + k[len("manifests/"):-len(".json")] for k in keys
                      if k.endswith(".json") and ids.is_sha256(k[len("manifests/"):-len(".json")]))[:limit]

    def has_local(self, art_id: str) -> bool:
        """Manifest and every payload blob present in this store."""
        if not ids.is_art(art_id):
            raise ValueError(f"not an art: id: {art_id!r}")
        hex_ = ids.hex_of(art_id)
        if not self.manifest_path(hex_).exists():
            return False
        return self._blobs_local(self._read_manifest(self.manifest_path(hex_), hex_))

    def _repo_candidate(self, art_id: str, rel: str | None) -> Path | None:
        mapped = self.replica_map().get(art_id)
        if not mapped or not self.repo:
            return None
        base = self.repo / mapped
        return base / rel if rel else base

    def _repo_verified(self, m: ArtifactManifest, art_id: str) -> bool:
        """Every payload blob of `m` is present at its mapped repo path with the manifest's sha256 (the 'verified on open' check)."""
        if m.payload is None:
            return False
        pairs = [(f.sha256, f.path) for f in m.payload.files or []] if m.payload.type == "tree" else [(m.payload.sha256, None)]
        for h, rel in pairs:
            c = self._repo_candidate(art_id, rel)
            if c is None or not c.is_file() or sha256_file(c)[0] != h:
                return False
        return True

    def _resolve_blob(self, hex_: str, art_id: str, rel: str | None) -> Path:
        """Make blob `hex_` local (objects -> repo replica -> remote) and return its object path."""
        dst = self.object_path(hex_)
        if dst.exists():
            return dst
        cand = self._repo_candidate(art_id, rel)
        if cand is not None:
            if cand.is_file() and sha256_file(cand)[0] == hex_:
                return self._store_blob(cand, hex_)
            _warn(f"repo replica {cand} {'is missing' if not cand.is_file() else 'does not match its manifest'} for {art_id}; trying the remote")
        r = self.remote
        if r is None:
            why = getattr(self, "remote_error", None)
            raise NotFound(f"{art_id}: blob {hex_} is not local" + (", not in the repo replica" if cand else "")
                           + (f", and the configured remote is unusable here: {why}" if why else ", and no remote is configured"))
        tmp = dst.parent / f".tmp-{hex_}-{os.getpid()}"
        try:
            r.get(object_key(hex_), tmp)
        except KeyError:
            raise NotFound(f"{art_id}: blob {hex_} is not local and not on {r.describe()}") from None
        try:
            got = sha256_file(tmp)[0]
            if got != hex_:
                raise IntegrityError(f"{art_id}: remote {object_key(hex_)} hashes to {got}")
            return self._store_blob(tmp, hex_, verify=False)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def select_files(files: list[TreeFile], paths: list[str] | None) -> list[TreeFile]:
        """The tree files matching any of `paths` (fnmatch globs on the relative path; a bare directory name selects its subtree:
        `rep1` == `rep1/*`).  None / empty selects everything."""
        if not paths:
            return list(files)
        pats = [p.rstrip("/") for p in paths if p]
        out = []
        for f in files:
            if any(fnmatch.fnmatchcase(f.path, p) or f.path.startswith(p + "/") for p in pats):
                out.append(f)
        return out

    def listing(self, art_id: str, paths: list[str] | None = None) -> list[dict]:
        """One row per payload file: `path`, `bytes`, `sha256`, `local` (blob in objects/), `remote` (the artifact's catalog flag:
        pushed or listed, no HEAD per file).  A file payload lists as one row."""
        m = self.get_manifest(art_id)
        if m.payload is None:
            raise StoreError(f"{art_id} has no payload (kind {m.kind})")
        row = self.index.artifact(art_id)
        remote = bool(row and row.get("remote"))
        files = m.payload.files if m.payload.type == "tree" else [TreeFile(m.payload.name, m.payload.sha256, m.payload.bytes)]
        return [{"path": f.path, "bytes": f.bytes, "sha256": f.sha256, "local": self.object_path(f.sha256).exists(), "remote": remote}
                for f in self.select_files(files or [], paths)]

    def fetch(self, art_id: str, to: Path | None = None, *, paths: list[str] | None = None) -> Path:
        """Materialise the payload.  File: returns the immutable object path (or a copy at/under `to`).  Tree: `to` or
        `<root>/trees/<hex>/`.  `to` is used as a directory when it is one (file lands as `to/<name>`).  `paths` (tree only)
        materialises just the matching subtree (`select_files`): the layout under `to` is the same, only the selected files
        land; the trees/ cache is then complete only when every file has been fetched."""
        m = self.get_manifest(art_id)
        if m.payload is None:
            raise StoreError(f"{art_id} has no payload (kind {m.kind})")
        hex_ = ids.hex_of(art_id)
        if m.payload.type == "file":
            blob = self._resolve_blob(m.payload.sha256, art_id, None)
            self.index.set_presence(art_id, local=True)
            if to is None:
                return blob
            to = Path(to)
            dst = to / m.payload.name if to.is_dir() else to
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blob, dst)
            return dst
        wanted = self.select_files(m.payload.files or [], paths)
        if paths and not wanted:
            raise NotFound(f"{art_id}: no file matches {paths}")
        srcs = [(f, self._resolve_blob(f.sha256, art_id, f.path)) for f in wanted]
        self.index.set_presence(art_id, local=self._blobs_local(m))
        if to is None:
            final = self.root / "trees" / hex_
            if not (final.is_dir() and all((final / f.path).is_file() for f in wanted)):
                self._land_tree(final, srcs, partial=bool(paths))
            os.utime(final)   # eviction leaves a tree fetched within `evict.TREE_KEEP_S`: whoever fetched it may still read it
            return final
        self._materialise(srcs, Path(to))
        return Path(to)

    def _land_tree(self, final: Path, srcs: list[tuple[TreeFile, Path]], *, partial: bool) -> None:
        # A tree already in trees/ (a partial fetch, or another process's fetch that landed first) is completed in place with the
        # files it lacks, and nothing in it is replaced or removed: another process may be reading it.  Each file lands whole, so
        # the cache is complete when every file is there.
        if partial or final.is_dir():
            self._materialise([(f, b) for f, b in srcs if not (final / f.path).is_file()], final)
            return
        work = final.parent / f".tmp-{final.name}-{os.getpid()}"
        shutil.rmtree(work, ignore_errors=True)
        self._materialise(srcs, work)
        try:
            os.rename(work, final)
        except OSError:
            self._materialise([(f, b) for f, b in srcs if not (final / f.path).is_file()], final)
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _materialise(srcs: list[tuple[TreeFile, Path]], dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for f, blob in srcs:
            p = dst / f.path
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.parent / f".tmp-{p.name}-{os.getpid()}"
            shutil.copyfile(blob, tmp)
            os.replace(tmp, p)

    def where(self, art_id: str, *, jobs: int = WHERE_JOBS) -> list[dict]:
        """Replicas: local (manifest + blobs present?), repo (mapped path, verified?), remote (present? verified_utc).  The remote
        check HEADs the manifest and every blob through a pool of `jobs` threads (a run-files tree can hold 10^4 blobs)."""
        m = self.get_manifest(art_id)
        hex_ = ids.hex_of(art_id)
        blobs = m.blobs()
        n_local = sum(1 for h, _ in blobs if self.object_path(h).exists())
        reps = self.index.replicas(art_id)
        out = [{"location": "local", "path": str(self.root), "present": n_local == len(blobs), "blobs": f"{n_local}/{len(blobs)}",
                "verified_utc": reps.get("local", {}).get("verified_utc")}]
        mapped = self.replica_map().get(art_id)
        if mapped:
            ok = self._repo_verified(m, art_id)
            out.append({"location": "repo", "path": mapped, "repo": str(self.repo), "present": ok, "verified_utc": utc_now() if ok else None})
        r = self.remote
        if r is None and getattr(self, "remote_error", None):   # a broken vault is reported as broken, never as absent; the local facts still stand
            out.append({"location": "remote", "configured": True, "present": None, "verified_utc": None, "error": f"ConfigError: {self.remote_error}"})
        elif r is None:
            out.append({"location": "remote", "configured": False, "present": False, "verified_utc": None})
        else:
            heads = [(manifest_key(hex_), len(m.canonical()))] + [(object_key(h), n) for h, n in blobs]
            missing = self._heads_missing(r, heads, jobs=jobs)
            out.append({"location": "remote", "remote": r.describe(), "present": not missing, "verified_utc": reps.get("remote", {}).get("verified_utc"),
                        "missing": missing})
        return out

    @staticmethod
    def _heads_missing(r: Remote, heads: list[tuple[str, int]], *, jobs: int = WHERE_JOBS) -> int:
        """How many of `heads` (key, expected bytes) the remote lacks or holds at another size; HEADs run `jobs` at a time."""
        def check(item: tuple[str, int]) -> bool:
            hd = r.head(item[0])
            return hd is not None and hd["bytes"] == item[1]
        if len(heads) <= 1:
            return sum(1 for it in heads if not check(it))
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(heads)))) as ex:
            return sum(1 for ok in ex.map(check, heads) if not ok)

    # ---- attempts -----------------------------------------------------------------------------------------------------------------------

    def put_attempt(self, doc: dict) -> None:
        """Write `attempts/<id>.json` once.  Same content again is a no-op; different content is AttemptExists."""
        validate_attempt_doc(doc)
        data = canon.dumps(doc)
        p = self.attempt_path(doc["id"])
        if not write_once(p, data) and p.read_bytes() != data:
            raise AttemptExists(f"{p} exists with different content; attempts are immutable (a rerun gets a fresh run id)")
        self.index.index_attempt(doc)

    def get_attempt(self, run_id: str) -> dict | None:
        """Local attempt manifest, else the remote's copy (cached locally), else None."""
        if not ids.is_run_id(run_id):
            raise ValueError(f"not a run id: {run_id!r}")
        p = self.attempt_path(run_id)
        if not p.exists():
            r = self.remote
            if r is None:
                return None
            tmp = p.parent / f".tmp-{run_id}-{os.getpid()}-{threading.get_ident()}"
            try:
                r.get(attempt_key(run_id), tmp)
            except KeyError:
                return None
            try:
                write_once(p, tmp)
            finally:
                tmp.unlink(missing_ok=True)
            self.index.index_attempt(json.loads(p.read_bytes()))
        return json.loads(p.read_bytes())

    # ---- labels -------------------------------------------------------------------------------------------------------------------------

    def add_label(self, target: str, key: str, value: Any, by: str, ref: str | None = None, ts: str | None = None) -> Path:
        """Append one assertion as its own file; never read-modify-write."""
        label = Label(target=target, key=key, value=value, by=by, ts=ts or utc_now(), ref=ref).validate()
        d = self.labels_dir(target)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{re.sub(r'[^0-9TZ]', '', label.ts)}-{re.sub(r'[^A-Za-z0-9._-]', '_', by)}"
        data = canon.dumps(label.to_doc())
        for n in range(1, 1000):
            p = d / (f"{stem}.json" if n == 1 else f"{stem}~{n}.json")   # '~' sorts after '.', so history order survives a same-second collision
            try:
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            except FileExistsError:
                continue
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            self.index.index_label(label, p)
            return p
        raise StoreError(f"could not allocate a label file under {d}")

    @staticmethod
    def _label_order(name: str) -> tuple[str, int]:
        """`<stem>.json` -> (stem, 1); `<stem>~N.json` -> (stem, N): the same-second suffix sorts numerically (`~10` after `~9`)."""
        stem = name[:-5] if name.endswith(".json") else name
        base, sep, n = stem.rpartition("~")
        if sep and n.isdigit():
            return base, int(n)
        return stem, 1

    def labels(self, target: str, key: str | None = None, by_glob: str | None = None) -> list[Label]:
        """Every assertion about `target`, sorted by ts (then file name, `~N` numerically); `key` exact, `by_glob` fnmatch."""
        d = self.labels_dir(target)
        out = []
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                lab = Label.from_doc(json.loads(p.read_bytes()))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                _warn(f"skipping unreadable label {p}: {e}")
                continue
            if (key is None or lab.key == key) and (by_glob is None or fnmatch.fnmatchcase(lab.by, by_glob)):
                out.append((lab.ts, self._label_order(p.name), lab))
        return [lab for _, _, lab in sorted(out, key=lambda t: (t[0], t[1]))]

    # ---- snapshots ----------------------------------------------------------------------------------------------------------------------

    def snapshot(self, name: str, members: list[str], note: str | None = None, query: dict | None = None) -> str:
        """Freeze a member set as a `dataset-snapshot/v1` artifact.  Identity is content: the same (name, members, note, query)
        yields the same id regardless of member order or who freezes it."""
        members = sorted(set(members))
        bad = [x for x in members if not ids.is_art(x)]
        if bad:
            raise ValueError(f"snapshot members must be art: ids: {bad[:3]}")
        meta: dict[str, Any] = {"name": name}
        if note is not None:
            meta["note"] = note
        if query is not None:
            meta["query"] = query
        return self.put_manifest(ArtifactManifest(kind=SNAPSHOT_KIND, meta=meta, refs={"members": members}))

    # ---- remote: verify / push ----------------------------------------------------------------------------------------------------------

    def _readback(self, r: Remote, key: str, sha: str) -> str | None:
        """Download `key` and sha256 it; None when it matches `sha`, else the reason.  Thread-safe (one temp file per thread)."""
        tmp = self.root / f".verify-{os.getpid()}-{threading.get_ident()}"
        try:
            r.get(key, tmp)
            got = sha256_file(tmp)[0]
        except KeyError:
            return f"{key}: absent"
        finally:
            tmp.unlink(missing_ok=True)
        return None if got == sha else f"{key}: read-back sha256 {got} != {sha}"

    def verify(self, art_id: str) -> dict:
        """PRESERVED check (§7) by read-back: for the manifest and every blob, remote head size matches and a full read-back sha256
        matches.  Records `verified_utc` (+ `verified: "sha256-readback"`) for the remote replica.  Without a remote: preserved=False,
        reason given, nothing raised."""
        m = self.get_manifest(art_id)
        hex_ = ids.hex_of(art_id)
        r = self.remote
        rep: dict[str, Any] = {"id": art_id, "preserved": False, "remote": r.describe() if r else None, "checked": [], "errors": [], "verified_utc": None}
        if r is None:
            rep["reason"] = f"remote unusable: {self.remote_error}" if getattr(self, "remote_error", None) else "no remote configured"
            return rep
        keys = [(manifest_key(hex_), hex_, len(m.canonical()))] + [(object_key(h), h, n) for h, n in m.blobs()]
        for key, sha, size in keys:
            head = r.head(key)
            if head is None:
                rep["errors"].append(f"{key}: absent")
                continue
            if head["bytes"] != size:
                rep["errors"].append(f"{key}: remote size {head['bytes']} != {size}")
                continue
            err = self._readback(r, key, sha)                            # streamed to a temp file: a > 5 GiB (multipart) object reads back the same way
            if err:
                rep["errors"].append(err)
            else:
                rep["checked"].append(key)
        if not rep["errors"]:
            rep["preserved"], rep["verified_utc"], rep["verified"] = True, utc_now(), VERIFY_READBACK
            self.index.set_replica(art_id, "remote", rep["verified_utc"], r.describe(), verified=VERIFY_READBACK)
            self.index.set_presence(art_id, remote=True)
        return rep

    def _push_object(self, r: Remote, key: str, path: Path, size: int, sha: str, verify: str | None) -> dict:
        """Upload one object (the remote's conditional `put`: HEAD, skip when present, else create) and verify it.  `verify` is
        `head` (size + single-part ETag == MD5, falling back to read-back when the remote gives no MD5 ETag), `readback` (download
        + sha256), or None (upload only).  Never raises: the outcome is in the dict (`error` set = not preserved)."""
        out: dict[str, Any] = {"key": key, "bytes": size, "action": None, "verified": None, "fallback": None, "error": None}
        try:
            res = r.put(key, path)
            out["action"], out["fallback"] = res["action"], res.get("fallback")
            if verify is None:
                return out
            if verify == "head":
                if res["action"] == "skipped":                          # the HEAD `put` did is the check
                    head = {"bytes": res["bytes"], "etag": res["etag"]}
                elif res["action"] == "created" and res.get("bytes") == size and is_md5_etag(res.get("etag")) and not res.get("multipart"):
                    head = {"bytes": size, "etag": res["etag"]}        # the PUT response's ETag is the service's word on what it stored: a HEAD would repeat it
                else:
                    head = r.head(key)
                if head is None:
                    out["error"] = f"{key}: absent after upload"
                elif head["bytes"] != size:
                    out["error"] = f"{key}: remote size {head['bytes']} != {size}"
                elif is_md5_etag(head.get("etag")):
                    if head["etag"] == res["md5"]:
                        out["verified"] = VERIFY_HEAD
                    else:
                        out["error"] = f"{key}: remote ETag {head['etag']} != local MD5 {res['md5']}"
                elif res.get("multipart") and head.get("etag") == res["multipart"]["etag"]:   # the multipart ETag the local parts yield
                    out["verified"] = VERIFY_MULTIPART
                else:                                                   # a multipart ETag we cannot reproduce, or a remote without one: content must be read
                    out["error"], out["verified"] = self._readback(r, key, sha), VERIFY_READBACK
            else:
                out["error"], out["verified"] = self._readback(r, key, sha), VERIFY_READBACK
            if out["error"]:
                out["verified"] = None
        except Exception as e:  # noqa: BLE001  one object's failure is reported, not raised: the artifact ends up NOT preserved
            out["error"] = f"{key}: {type(e).__name__}: {e}"
        return out

    def push(self, art_id: str, preserve: bool = True, *, jobs: int = 32, verify: str = "head", progress: Callable[[dict], None] | None = None) -> dict:
        """Upload every blob (`jobs` threads) and then the manifest under immutable keys; a manifest is never visible before its
        blobs, so it goes up only when every blob is present and (with `preserve`) verified.  `verify`: `head` (default; size +
        single-part ETag == MD5, recorded as `etag-md5`) or `readback` (download + sha256, `sha256-readback`); the manifest is
        always read back.  The report is the `verify` shape (`preserved`, `checked`, `errors`, `verified_utc`) plus `verified`,
        `objects`, `uploaded`, `skipped`, `bytes`, `seconds`, `notes` (e.g. the remote switching conditional PUT off).  Results are
        reported in manifest order whatever the completion order.  `progress`, when given, is called with running counts."""
        if verify not in VERIFY_MODES:
            raise ValueError(f"verify must be one of {VERIFY_MODES}, got {verify!r}")
        t0 = time.monotonic()
        m = self.get_manifest(art_id)
        hex_ = ids.hex_of(art_id)
        r = self._require_remote()
        files = m.payload.files if m.payload and m.payload.type == "tree" else []
        rel = {f.sha256: f.path for f in files or []}
        blobs = [(object_key(h), self._resolve_blob(h, art_id, rel.get(h)), n, h) for h, n in m.blobs()]
        mode = verify if preserve else None
        results: list[dict | None] = [None] * len(blobs)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(blobs) or 1))) as ex:
            futs = {ex.submit(self._push_object, r, k, p, n, h, mode): i for i, (k, p, n, h) in enumerate(blobs)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if progress and (done % 100 == 0 or done == len(blobs)):
                    progress({"id": art_id, "done": done, "objects": len(blobs) + 1, "seconds": time.monotonic() - t0})
        objs = [x for x in results if x is not None]
        errors = [x["error"] for x in objs if x["error"]]
        notes = sorted({x["fallback"] for x in objs if x["fallback"]})
        if not errors:                                                  # blobs first, manifest last, and only over a complete blob set
            man = self._push_object(r, manifest_key(hex_), self.manifest_path(hex_), len(m.canonical()), hex_, VERIFY_READBACK if preserve else None)
            objs.append(man)
            if man["error"]:
                errors.append(man["error"])
            if man["fallback"]:
                notes.append(man["fallback"])
        else:
            notes.append(f"manifest not uploaded: {len(errors)} blob(s) failed")
        preserved = (not errors) if preserve else None
        methods = sorted({x["verified"] for x in objs[:len(blobs)] if x["verified"]})   # the blobs' check(s); the manifest is always read back
        rep: dict[str, Any] = {"id": art_id, "remote": r.describe(), "preserved": preserved, "checked": [x["key"] for x in objs if x["verified"]],
                               "errors": errors, "verified_utc": None, "verified": (",".join(methods) or VERIFY_READBACK) if preserved else None,
                               "objects": len(blobs) + 1, "uploaded": sum(1 for x in objs if x["action"] == "created"),
                               "skipped": sum(1 for x in objs if x["action"] in ("skipped", "exists")),
                               "bytes": sum(x["bytes"] for x in objs if x["action"]), "seconds": round(time.monotonic() - t0, 3), "notes": notes}
        if not errors:
            self.index.set_presence(art_id, remote=True)
            if preserve:
                rep["verified_utc"] = utc_now()
                self.index.set_replica(art_id, "remote", rep["verified_utc"], r.describe(), verified=rep["verified"])
            elif "remote" not in self.index.replicas(art_id):
                self.index.set_replica(art_id, "remote", None, r.describe())
        return rep

    def push_attempt(self, run_id: str) -> None:
        r = self._require_remote()
        p = self.attempt_path(run_id)
        if not p.exists():
            raise NotFound(f"attempt {run_id} is not local")
        r.put(attempt_key(run_id), p)

    def push_labels(self, targets: list[str], *, jobs: int = 16) -> dict:
        """Push the assertions about `targets` (art / drv / run ids) the remote lacks, each under `labels/<target>/<content sha256>.json`
        (`labels_sync.push_targets`).  Per-assertion failures are in the report's `failed`, never raised; a missing remote is."""
        from . import labels_sync as LS  # noqa: PLC0415

        return LS.push_targets(self, targets, jobs=jobs)

    def labels_durable(self, targets: list[str]) -> dict:
        """The labels leg of the durability gate (`labels_sync.verify_targets`): every local assertion of every target listed on the remote."""
        from . import labels_sync as LS  # noqa: PLC0415

        return LS.verify_targets(self, targets)

    def pending(self) -> dict:
        """What this store holds that the remote does not (the queue `push --pending` drains), from one listing per prefix:
        `artifacts` without a verified remote replica whose manifest the remote lacks, or whose blobs are all here to verify
        cheaply (an artifact pulled from the remote without its blobs is the remote's already: `preserved` verifies it on demand);
        `attempts` the remote lacks; `labels` (content hashes) the remote lacks."""
        from . import labels_sync as LS  # noqa: PLC0415

        r = self._require_remote()
        on_remote = {k[len("manifests/"):-len(".json")] for k in r.list("manifests/") if k.endswith(".json")}
        attempts_remote = {k[len("attempts/"):-len(".json")] for k in r.list("attempts/") if k.endswith(".json")}
        verified = {row[0] for row in self.index.db.execute("SELECT id FROM replicas WHERE location='remote' AND verified_utc IS NOT NULL")}
        arts = []
        for a in self.artifact_ids():
            h = ids.hex_of(a)
            if a in verified:
                continue
            if h in on_remote:
                try:
                    if not self._blobs_local(self._read_manifest(self.manifest_path(h), h)):
                        continue
                except (IntegrityError, ValueError, json.JSONDecodeError):
                    continue
            arts.append(a)
        mine, theirs = LS.local_labels(self), LS.remote_labels(r)
        labels = sorted(set(mine) - set(theirs))
        return {"remote": r.describe(), "artifacts": arts, "attempts": [x for x in self.attempt_ids() if x not in attempts_remote],
                "labels": [{"hash": h, "target": mine[h][1].target, "key": mine[h][1].key, "by": mine[h][1].by} for h in labels]}

    def refresh(self, *, jobs: int = REFRESH_JOBS, log: Callable[[str], None] | None = None,
                progress: Callable[[str, int, int], None] | None = None, payload_kinds: Sequence[str] = (),
                payload_max_bytes: int = REFRESH_PAYLOAD_MAX, attempts_only: bool = False) -> dict:
        """Bring this store up to date from the remote without wiping anything: list `manifests/`, `attempts/` and `labels/`, fetch
        only what is missing (`jobs` concurrent GETs; each manifest re-hashed to its id, each attempt parsed and named by its id),
        and index just those.  The way a fresh machine builds its catalog from the remote alone.  `payload_kinds`: also make
        local the file payloads (at most `payload_max_bytes` each) of every catalogued artifact of those kinds whose blob is not
        here yet -- what a reader of payloads (a render) needs; the count is `payloads`.  `attempts_only`: the attempts alone (what
        `research merge` reads), one listing instead of three."""
        from . import labels_sync as LS  # noqa: PLC0415

        t0 = time.monotonic()
        r = self._require_remote()
        want_m = [] if attempts_only else \
            [k for k in r.list("manifests/") if k.endswith(".json") and ids.is_sha256(k[len("manifests/"):-len(".json")])]
        want_m = [k for k in want_m if not (self.root / k).exists()]
        want_a = [k for k in r.list("attempts/") if k.endswith(".json") and ids.is_run_id(k[len("attempts/"):-len(".json")])]
        want_a = [k for k in want_a if not (self.root / k).exists()]
        rep: dict[str, Any] = {"remote": r.describe(), "manifests": 0, "attempts": 0, "labels": 0, "errors": []}
        tmpdir = self.root / "scratch"
        tmpdir.mkdir(parents=True, exist_ok=True)

        def get(key: str) -> tuple[str, bytes | None, str | None]:     # threads download; the calling thread writes and indexes
            tmp = tmpdir / f".refresh-{os.getpid()}-{threading.get_ident()}"
            try:
                r.get(key, tmp)
                return key, tmp.read_bytes(), None
            except Exception as e:  # noqa: BLE001
                return key, None, f"{key}: {type(e).__name__}: {e}"
            finally:
                tmp.unlink(missing_ok=True)

        rmap = self.replica_map()
        tick = progress or (lambda what, done, total: None)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            tick("manifests", 0, len(want_m))
            for key, data, err in ex.map(get, want_m):
                tick("manifests", 1, len(want_m))
                h = key[len("manifests/"):-len(".json")]
                if err is None and canon.sha256_bytes(data) != h:
                    err = f"{key}: content hash does not match its name"
                if err:
                    rep["errors"].append(err)
                    continue
                m = ArtifactManifest.from_doc(json.loads(data))
                write_once(self.manifest_path(h), data)
                self.index.index_artifact(m.to_doc(), "art:" + h, local=self._blobs_local(m), remote=True, repo_path=rmap.get("art:" + h))
                rep["manifests"] += 1
            tick("attempts", 0, len(want_a))
            for key, data, err in ex.map(get, want_a):
                tick("attempts", 1, len(want_a))
                rid = key[len("attempts/"):-len(".json")]
                try:
                    doc = json.loads(data) if err is None else None
                    if doc is not None and doc.get("id") != rid:
                        err = f"{key}: names attempt {doc.get('id')!r}"
                except json.JSONDecodeError as e:
                    err = f"{key}: {e}"
                if err:
                    rep["errors"].append(err)
                    continue
                write_once(self.attempt_path(rid), data)
                self.index.index_attempt(doc)
                rep["attempts"] += 1
        if not attempts_only:
            tick("labels", 0, 0)
            lrep = LS.sync(self, push=False, pull=True, jobs=jobs, log=log)
            rep["labels"], rep["errors"] = len(lrep["pulled"]), rep["errors"] + lrep["bad"]
        if payload_kinds:
            got, errs = self.fetch_payloads(payload_kinds, max_bytes=payload_max_bytes, jobs=jobs)
            rep["payloads"], rep["errors"] = got, rep["errors"] + errs
        rep["seconds"] = round(time.monotonic() - t0, 1)
        return rep

    def fetch_payloads(self, kinds: Sequence[str], *, max_bytes: int = REFRESH_PAYLOAD_MAX, jobs: int = REFRESH_JOBS) -> tuple[int, list[str]]:
        """Make local the file payloads (<= `max_bytes`) of every catalogued artifact of `kinds` whose blob is not here.
        Returns (fetched, errors)."""
        want = []
        for kind in kinds:
            for row in self.index.select_artifacts(kind=kind):
                pay = (row.get("manifest") or {}).get("payload") or {}
                if (pay.get("type") == "file" and isinstance(pay.get("sha256"), str) and (pay.get("bytes") or 0) <= max_bytes
                        and not self.object_path(pay["sha256"]).exists()):
                    want.append((row["id"], pay["sha256"]))

        def one(item: tuple[str, str]) -> tuple[str, str | None]:     # threads download; the calling thread records presence
            art_id, hex_ = item
            try:
                self._resolve_blob(hex_, art_id, None)
                return art_id, None
            except Exception as e:  # noqa: BLE001  one payload's failure is reported, the rest still land
                return art_id, f"payload {art_id}: {type(e).__name__}: {e}"

        errs = []
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            for art_id, err in ex.map(one, want):
                if err:
                    errs.append(err)
                else:
                    self.index.set_presence(art_id, local=True)
        return len(want) - len(errs), errs

    # ---- index --------------------------------------------------------------------------------------------------------------------------

    def reindex(self, remote: bool = False) -> dict:
        """Rebuild the catalog from manifests/, attempts/, labels/ (and, with `remote`, the remote listing: manifests and attempts not
        held locally are fetched and cached first).  Replica verifications already recorded are kept."""
        remote_hexes: set[str] = set()
        if remote and self.remote is not None:
            r = self.remote
            remote_hexes = {k[len("manifests/"):-len(".json")] for k in r.list("manifests/") if k.endswith(".json")}
            for h in sorted(remote_hexes):
                if ids.is_sha256(h) and not self.manifest_path(h).exists():
                    self._remote_copy(manifest_key(h), self.manifest_path(h))
            for k in r.list("attempts/"):
                rid = k[len("attempts/"):-len(".json")]
                if ids.is_run_id(rid) and not self.attempt_path(rid).exists():
                    self._remote_copy(k, self.attempt_path(rid))
        rmap = self.replica_map()
        pushed = {r[0] for r in self.index.db.execute("SELECT id FROM replicas WHERE location='remote'")}   # survives the rebuild
        arts = []
        for art in self.artifact_ids():
            h = ids.hex_of(art)
            try:
                m = self._read_manifest(self.manifest_path(h), h)
            except (IntegrityError, ValueError, json.JSONDecodeError) as e:
                _warn(f"skipping {self.manifest_path(h)}: {e}")
                continue
            arts.append((m.to_doc(), art, self._blobs_local(m), h in remote_hexes or art in pushed, rmap.get(art)))
        attempts = []
        for rid in self.attempt_ids():
            try:
                attempts.append(json.loads(self.attempt_path(rid).read_bytes()))
            except json.JSONDecodeError as e:
                _warn(f"skipping {self.attempt_path(rid)}: {e}")
        labels = []
        for p in sorted((self.root / "labels").glob("*/*.json")):
            try:
                labels.append((Label.from_doc(json.loads(p.read_bytes())), p))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                _warn(f"skipping label {p}: {e}")
        return self._swap_in_rebuilt(arts, attempts, labels)

    def _swap_in_rebuilt(self, arts: list, attempts: list, labels: list) -> dict:
        """Build the new catalog in a side file (the replica records carried over) and rename it over `catalog.sqlite` only once it
        is complete: a rebuild killed partway leaves the previous catalog answering.  A replica verification recorded by another
        process while this runs is lost with the old file (the next `preserved` re-verifies)."""
        path = self.root / "catalog.sqlite"
        tmp = self.root / f".catalog.sqlite.rebuild-{os.getpid()}"
        for q in (tmp, tmp.with_name(tmp.name + "-journal")):
            q.unlink(missing_ok=True)
        new = Index(tmp)
        try:
            old = self.index.db
            new.db.execute("BEGIN")
            new.db.executemany("INSERT OR REPLACE INTO replicas(id, location, verified_utc, detail) VALUES (?, ?, ?, ?)",
                               old.execute("SELECT id, location, verified_utc, detail FROM replicas").fetchall())
            new.db.executemany("INSERT OR REPLACE INTO replica_checks(id, location, verified) VALUES (?, ?, ?)",
                               old.execute("SELECT id, location, verified FROM replica_checks").fetchall())
            new.db.execute("COMMIT")
            n = new.rebuild(arts, attempts, labels)
        except BaseException:
            new.close()
            tmp.unlink(missing_ok=True)
            raise
        new.close()
        self.close()
        os.replace(tmp, path)
        return n

    def _remote_copy(self, key: str, dst: Path) -> None:
        tmp = dst.parent / f".tmp-{dst.name}-{os.getpid()}"
        try:
            self.remote.get(key, tmp)
            write_once(dst, tmp)
        except KeyError:
            pass
        finally:
            tmp.unlink(missing_ok=True)

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

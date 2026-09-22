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
that arrive from a repo or the remote are checked against the manifest before they enter `objects/`.  `verify` is the PRESERVED
check (§7): remote head size + full read-back sha256 for the manifest and every blob; nothing is called PRESERVED without a remote.
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
from pathlib import Path
from typing import Any

from . import canon, ids, kinds
from .index import Index
from .model import ArtifactManifest, Label, Payload, TreeFile, utc_now, validate_attempt_doc
from .remote import ConfigError, Remote, load_remote

DEFAULT_ROOT = Path.home() / ".research" / "store"
REPLICA_MAP = Path("fixtures") / "artifacts.json"
SNAPSHOT_KIND = "dataset-snapshot/v1"
BIG = 4 << 30
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


def write_once(dst: Path, data: bytes | Path) -> bool:
    """Temp file beside `dst`, then hard-link into place: the first writer wins, nothing is ever overwritten.  Returns whether this
    call created it.  Falls back to rename where the filesystem refuses links."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    tmp = dst.parent / f".tmp-{dst.name}-{os.getpid()}-{secrets.token_hex(4)}"
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        shutil.copyfile(data, tmp)
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
        if self._index is None:
            self._index = Index(self.root / "catalog.sqlite")
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

    def _require_remote(self) -> Remote:
        r = self.remote
        if r is None:
            why = getattr(self, "remote_error", None)
            raise StoreError(why or "no remote configured (~/.research/store.toml or $RESEARCH_STORE_CONFIG)")
        return r

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

    def _store_blob(self, src: Path, hex_: str) -> Path:
        dst = self.object_path(hex_)
        if not dst.exists():
            write_once(dst, src)
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
            tmp = p.parent / f".tmp-{hex_}-{os.getpid()}"
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
            raise NotFound(f"{art_id}: blob {hex_} is not local" + (", not in the repo replica" if cand else "") + ", and no remote is configured")
        tmp = dst.parent / f".tmp-{hex_}-{os.getpid()}"
        try:
            r.get(object_key(hex_), tmp)
        except KeyError:
            raise NotFound(f"{art_id}: blob {hex_} is not local and not on {r.describe()}") from None
        try:
            got = sha256_file(tmp)[0]
            if got != hex_:
                raise IntegrityError(f"{art_id}: remote {object_key(hex_)} hashes to {got}")
            return self._store_blob(tmp, hex_)
        finally:
            tmp.unlink(missing_ok=True)

    def fetch(self, art_id: str, to: Path | None = None) -> Path:
        """Materialise the payload.  File: returns the immutable object path (or a copy at/under `to`).  Tree: `to` or
        `<root>/trees/<hex>/`.  `to` is used as a directory when it is one (file lands as `to/<name>`)."""
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
        srcs = [(f, self._resolve_blob(f.sha256, art_id, f.path)) for f in m.payload.files or []]
        self.index.set_presence(art_id, local=True)
        if to is None:
            final = self.root / "trees" / hex_
            if final.is_dir():
                return final
            work = final.parent / f".tmp-{hex_}-{os.getpid()}"
            shutil.rmtree(work, ignore_errors=True)
            self._materialise(srcs, work)
            try:
                os.rename(work, final)
            except OSError:
                shutil.rmtree(work, ignore_errors=True)
            return final
        self._materialise(srcs, Path(to))
        return Path(to)

    @staticmethod
    def _materialise(srcs: list[tuple[TreeFile, Path]], dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for f, blob in srcs:
            p = dst / f.path
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.parent / f".tmp-{p.name}-{os.getpid()}"
            shutil.copyfile(blob, tmp)
            os.replace(tmp, p)

    def where(self, art_id: str) -> list[dict]:
        """Replicas: local (manifest + blobs present?), repo (mapped path, verified?), remote (present? verified_utc)."""
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
            present = all((hd := r.head(k)) is not None and hd["bytes"] == n for k, n in heads)
            out.append({"location": "remote", "remote": r.describe(), "present": present, "verified_utc": reps.get("remote", {}).get("verified_utc")})
        return out

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
            tmp = p.parent / f".tmp-{run_id}-{os.getpid()}"
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

    def verify(self, art_id: str) -> dict:
        """PRESERVED check (§7): for the manifest and every blob, remote head size matches and a full read-back sha256 matches.
        Records `verified_utc` for the remote replica.  Without a remote: preserved=False, reason given, nothing raised."""
        m = self.get_manifest(art_id)
        hex_ = ids.hex_of(art_id)
        r = self.remote
        rep: dict[str, Any] = {"id": art_id, "preserved": False, "remote": r.describe() if r else None, "checked": [], "errors": [], "verified_utc": None}
        if r is None:
            rep["reason"] = f"remote unusable: {self.remote_error}" if getattr(self, "remote_error", None) else "no remote configured"
            return rep
        keys = [(manifest_key(hex_), hex_, len(m.canonical()))] + [(object_key(h), h, n) for h, n in m.blobs()]
        tmp = self.root / f".verify-{os.getpid()}"
        for key, sha, size in keys:
            head = r.head(key)
            if head is None:
                rep["errors"].append(f"{key}: absent")
                continue
            if head["bytes"] != size:
                rep["errors"].append(f"{key}: remote size {head['bytes']} != {size}")
                continue
            if size >= BIG:
                rep["errors"].append(f"{key}: >= 4 GiB; per-part check via {key}.parts.json is not implemented")
                continue
            try:
                r.get(key, tmp)
                got = sha256_file(tmp)[0]
            finally:
                tmp.unlink(missing_ok=True)
            if got != sha:
                rep["errors"].append(f"{key}: read-back sha256 {got} != {sha}")
            else:
                rep["checked"].append(key)
        if not rep["errors"]:
            rep["preserved"], rep["verified_utc"] = True, utc_now()
            self.index.set_replica(art_id, "remote", rep["verified_utc"], r.describe())
            self.index.set_presence(art_id, remote=True)
        return rep

    def push(self, art_id: str, preserve: bool = True) -> dict:
        """Upload every blob and the manifest under immutable keys; with `preserve`, run `verify` and return its report."""
        m = self.get_manifest(art_id)
        r = self._require_remote()
        files = m.payload.files if m.payload and m.payload.type == "tree" else []
        rel = {f.sha256: f.path for f in files or []}
        for h, _ in m.blobs():
            r.put(object_key(h), self._resolve_blob(h, art_id, rel.get(h)))
        r.put(manifest_key(ids.hex_of(art_id)), self.manifest_path(ids.hex_of(art_id)))
        self.index.set_presence(art_id, remote=True)
        if "remote" not in self.index.replicas(art_id):
            self.index.set_replica(art_id, "remote", None, r.describe())
        if preserve:
            return self.verify(art_id)
        return {"id": art_id, "remote": r.describe(), "preserved": None, "pushed": True}

    def push_attempt(self, run_id: str) -> None:
        r = self._require_remote()
        p = self.attempt_path(run_id)
        if not p.exists():
            raise NotFound(f"attempt {run_id} is not local")
        r.put(attempt_key(run_id), p)

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
        return self.index.rebuild(arts, attempts, labels)

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

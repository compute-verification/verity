"""Repo replicas (README §7) and `research data refresh-fixtures`: keeping `fixtures/artifacts.json` honest when a fixture's bytes change.

A repo-registered fixture has *content-only identity*: kind `fixture/v1`, `meta {}`, `refs {}`, payload = the file (bare name) or
the directory as a tree, so its id is a function of the bytes alone and `tools/research/tests/test_repo_replicas.py` recomputes it
with no store state.  Edit the bytes and the id changes; until today that meant a `research data put --kind fixture/v1 --file …` by
hand and an edit of the map.  `refresh` does exactly that for every entry whose path no longer hashes to its id: `put` under the new
id (`push --preserve` on request), rewrite the entry, and record the superseded id under the map's `previous` block:

    {"art:<new>": "fixtures/x.json", ..., "previous": {"art:<new>": ["art:<oldest>", ..., "art:<old>"]}}

`LocalStore.replica_map()` reads `art:` keys only, so the block is invisible to `fetch` in every worktree.  Idempotent: nothing is
put or written when every entry already hashes to its id.  `dry_run` computes and reports, writes nothing.

An entry is *tracked* (git holds its bytes: the small ones) or *fetched* (its path is gitignored and `research data fetch-fixtures`
materialises it: everything over `tests/test_repository.py`'s 256 KiB cap).  `fetch_registry` makes the fetched entries under the
given paths present and exactly their ids (a tree's directory holds its files and nothing else), fetching only what is missing
through `fetchset`, and never writes over a file git tracks.  `export` copies every entry, `previous` included, to the public
export, and `hydrate_stubs` puts the bytes back where the history rewrite left a stub (`STUB`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import fetchset, ids
from .local import REPLICA_MAP, LocalStore, _rel_files, manifest_key, object_key, sha256_file
from .model import ArtifactManifest, Payload, TreeFile
from .remote import HttpRemote, Remote, RemoteConflict

FIXTURE_KIND = "fixture/v1"
PREVIOUS = "previous"
#: What the history rewrite leaves in old commits in place of a fetched fixture's bytes (fixture process plan §4.1).
STUB = re.compile(rb"\Averity-fixture/v1 sha256:([0-9a-f]{64}) bytes:(\d+)\n")
STUB_MAX = 512


def fixture_manifest(path: Path) -> ArtifactManifest:
    """The manifest a repo replica is registered under (content-only identity); FileNotFoundError when `path` is neither file nor dir."""
    path = Path(path)
    if path.is_file():
        h, n = sha256_file(path)
        payload = Payload.file(h, n, path.name)
    elif path.is_dir():
        files = []
        for p in _rel_files(path):
            h, n = sha256_file(p)
            files.append(TreeFile(p.relative_to(path).as_posix(), h, n))
        payload = Payload.tree(files)
    else:
        raise FileNotFoundError(str(path))
    return ArtifactManifest(kind=FIXTURE_KIND, meta={}, refs={}, payload=payload).validate()


def fixture_id(path: Path) -> str:
    return fixture_manifest(path).id


def load_map(repo: Path) -> dict[str, Any]:
    p = Path(repo) / REPLICA_MAP
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: the replica map must be a JSON object")
    return doc


def entries(doc: dict[str, Any]) -> dict[str, str]:
    """The `art id -> repo-relative path` entries of a map document; the `previous` block (or anything else) is not an entry."""
    return {k: v for k, v in doc.items() if ids.is_art(k) and isinstance(v, str)}


def previous(doc: dict[str, Any]) -> dict[str, list[str]]:
    blk = doc.get(PREVIOUS)
    return {k: list(v) for k, v in blk.items() if isinstance(v, list)} if isinstance(blk, dict) else {}


def _history(prev: list[str], old: str, new: str) -> list[str]:
    """The `previous` list for `new`: every id the path has hashed to before, oldest first, each once, never `new` itself (bytes can revert)."""
    out: list[str] = []
    for x in [*prev, old]:
        if x != new and x not in out:
            out.append(x)
    return out


def dump_map(doc: dict[str, Any]) -> str:
    """The map as the repo writes it: two-space indent, one entry per line, trailing newline; `previous` last."""
    body = {k: v for k, v in doc.items() if k != PREVIOUS}
    if doc.get(PREVIOUS):
        body[PREVIOUS] = doc[PREVIOUS]
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"


def refresh(store: LocalStore, repo: Path, *, preserve: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Recompute every entry's id from its bytes; put + rewrite the changed ones.  Returns the report (`entries`: one per map entry with
    `status` unchanged | changed | missing | conflict, `new_id`, `previous`, `preserved`, `errors`; `changed`, `missing`, `written`)."""
    repo = Path(repo)
    doc = load_map(repo)
    prev = previous(doc)
    rep: dict[str, Any] = {"repo": str(repo), "map": str(repo / REPLICA_MAP), "dry_run": dry_run, "preserve": preserve, "entries": [],
                           "changed": 0, "missing": 0, "unchanged": 0, "written": False}
    new_doc: dict[str, Any] = {}
    new_prev: dict[str, list[str]] = dict(prev)
    for k, v in doc.items():
        if k == PREVIOUS:
            continue
        if not (ids.is_art(k) and isinstance(v, str)):
            new_doc[k] = v                                          # not an entry: carried over untouched
            continue
        path = repo / v
        e: dict[str, Any] = {"id": k, "path": v, "status": None}
        try:
            m = fixture_manifest(path)
        except FileNotFoundError:
            e["status"] = "missing"
            rep["missing"] += 1
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        if m.id == k:
            e["status"] = "unchanged"
            rep["unchanged"] += 1
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        if m.id in new_doc or m.id in entries(doc):
            e.update(status="conflict", new_id=m.id, errors=[f"{v} now hashes to {m.id}, which is already registered as {doc.get(m.id, new_doc.get(m.id))}"])
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        e.update(status="changed", new_id=m.id, previous=_history(prev.get(k, []), k, m.id), preserved=None, errors=[])
        if not dry_run:
            got = store.put_artifact(FIXTURE_KIND, {}, None, file=path if path.is_file() else None, tree=path if path.is_dir() else None)
            if got != m.id:                                         # the same computation twice; anything else is a bug worth stopping on
                raise RuntimeError(f"{v}: put_artifact returned {got}, expected {m.id}")
            if preserve:
                try:
                    push = store.push(m.id, preserve=True)
                    e["preserved"], e["errors"] = bool(push["preserved"]), list(push["errors"])
                except Exception as ex:  # noqa: BLE001  one entry's vault trouble is reported; the map is still made honest
                    e["preserved"], e["errors"] = False, [f"{type(ex).__name__}: {ex}"]
        new_doc[m.id] = v
        new_prev.pop(k, None)
        new_prev[m.id] = e["previous"]
        rep["changed"] += 1
        rep["entries"].append(e)
    if new_prev:
        new_doc[PREVIOUS] = new_prev
    if rep["changed"] and not dry_run:
        (repo / REPLICA_MAP).write_text(dump_map(new_doc), encoding="utf-8")
        rep["written"] = True
    return rep


# ---- fetched entries ----------------------------------------------------------------------------------------------------------------

def tracked(repo: Path) -> set[str] | None:
    """The repo-relative paths git tracks, or None in a tree without git (a shipped `git archive`, whose files are the commit's)."""
    p = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True)
    if p.returncode:
        return None
    return {x.decode() for x in p.stdout.split(b"\0") if x}


def _under(path: str, prefixes: Sequence[str]) -> bool:
    return not prefixes or any(p in (".", "") or path == p or path.startswith(p.rstrip("/") + "/") or p.startswith(path.rstrip("/") + "/")
                               for p in prefixes)


def _is_tracked(path: str, git: set[str] | None) -> bool:
    if git is None:
        return False
    return path in git or any(t.startswith(path.rstrip("/") + "/") for t in git)


def _expected(m: ArtifactManifest) -> dict[str, tuple[str, int]]:
    """{path relative to the entry (a file payload: ""): (sha256, bytes)}."""
    if m.payload is None:
        raise ValueError(f"{m.id} has no payload")
    if m.payload.type == "file":
        return {"": (m.payload.sha256, m.payload.bytes)}
    return {f.path: (f.sha256, f.bytes) for f in m.payload.files or []}


def _stale(dst: Path, m: ArtifactManifest) -> list[str]:
    """What differs between `dst` and the manifest's payload: missing or changed files, and (a tree) files it does not name."""
    want = _expected(m)
    if m.payload.type == "file":
        return [] if dst.is_file() and sha256_file(dst) == (m.payload.sha256, m.payload.bytes) else [dst.name]
    if not dst.is_dir():
        return sorted(want)
    have = {p.relative_to(dst).as_posix() for p in _rel_files(dst)}
    return sorted([r for r, (h, n) in want.items() if r not in have or sha256_file(dst / r) != (h, n)] + sorted(have - set(want)))


def _place(src: Path, dst: Path, tmp_dir: Path | None = None) -> None:
    """A private, writable copy of an immutable store object at `dst` (never a hard link: a test running as root could write
    through one into the store), copied in full to a temp file in `tmp_dir` (default `dst`'s directory, on its filesystem) and
    renamed into place, so no reader sees part of it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = (tmp_dir or dst.parent) / f".tmp-{dst.name}-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _entry_lock(store: LocalStore, dst: Path):
    """The lock every fetch holds while it reads or writes one entry's path in a tree (`fetchset.lock`)."""
    return fetchset.lock(store, "entry-" + hashlib.sha256(str(dst.resolve()).encode()).hexdigest())


def _materialise(store: LocalStore, m: ArtifactManifest, dst: Path) -> None:
    if m.payload.type == "file":
        _place(store.object_path(m.payload.sha256), dst)
        return
    want = _expected(m)
    if dst.is_file():
        dst.unlink()
    for rel, (h, _) in want.items():
        target = dst / rel
        if not (target.is_file() and sha256_file(target)[0] == h):
            _place(store.object_path(h), target, dst.parent)          # the temp beside the tree: a reader hashing it sees its files only
    for p in _rel_files(dst) if dst.is_dir() else []:
        if p.relative_to(dst).as_posix() not in want:
            p.unlink()


def fetch_registry(store: LocalStore, repo: Path, paths: Sequence[str] = (), *, srcs: Sequence[Remote], jobs: int = fetchset.JOBS,
                   check_only: bool = False) -> dict[str, Any]:
    """Make every fetched entry under `paths` (repo-relative; none = all) present and exactly its id.  `check_only` fetches
    manifests only and writes nothing to the tree.  The report's `entries` say, per entry, `tracked` (git holds it), `present`
    (already exact, which the bytes alone show: no source is needed), `fetched`, `stale` (check_only: missing or different) or
    `missing` (no source held it).  Concurrent fetches into one tree (a check's pytest and lean-suites steps) take turns on each
    entry (`_entry_lock`) and each blob (`fetchset.lock`), and every file lands whole by rename, so all of them succeed."""
    repo = Path(repo)
    doc = load_map(repo)
    git = tracked(repo)
    chosen = {a: p for a, p in entries(doc).items() if _under(p, paths)}
    rep: dict[str, Any] = {"repo": str(repo), "entries": [], "sources": [s.describe() for s in srcs]}
    want = {a: p for a, p in chosen.items() if not _is_tracked(p, git)}
    for a, p in chosen.items():
        if a not in want:
            rep["entries"].append({"id": a, "path": p, "status": "tracked"})
    for a, p in list(want.items()):                          # an entry's id is its bytes: an exact copy needs no manifest and no source
        with _entry_lock(store, repo / p):
            exact = (repo / p).exists() and fixture_id(repo / p) == a
        if exact:
            rep["entries"].append({"id": a, "path": p, "status": "present"})
            del want[a]
    fr = fetchset._report()
    ms = fetchset.fetch_manifests(store, want, srcs, jobs=jobs, report=fr)
    todo: dict[str, list[str]] = {}
    for a, p in want.items():
        if a not in ms:
            rep["entries"].append({"id": a, "path": p, "status": "missing"})
            continue
        with _entry_lock(store, repo / p):
            differs = _stale(repo / p, ms[a])
        if differs:
            todo[a] = differs
        else:
            rep["entries"].append({"id": a, "path": p, "status": "present"})
    if check_only:
        rep["entries"] += [{"id": a, "path": want[a], "status": "stale", "differs": d[:5]} for a, d in todo.items()]
    else:
        fetchset.fetch_blobs(store, (b for a in todo for b in ms[a].blobs()), srcs, jobs=jobs, report=fr)
        for a in todo:
            p = want[a]
            if not all(store.object_path(h).exists() for h, _ in ms[a].blobs()):
                rep["entries"].append({"id": a, "path": p, "status": "missing"})
                continue
            with _entry_lock(store, repo / p):                # another fetch may have placed it meanwhile: then it is present
                status = "fetched" if _stale(repo / p, ms[a]) else "present"
                if status == "fetched":
                    _materialise(store, ms[a], repo / p)
            rep["entries"].append({"id": a, "path": p, "status": status})
    rep["fetch"] = fr
    rep["ok"] = not any(e["status"] in ("missing", "stale") for e in rep["entries"])
    return rep


def hydrate_stubs(store: LocalStore, repo: Path, paths: Sequence[str] = (), *, srcs: Sequence[Remote], jobs: int = fetchset.JOBS) -> dict[str, Any]:
    """Put the bytes back into every tracked file under `paths` that is a history stub (`STUB`): an old commit's checkout."""
    repo = Path(repo)
    git = tracked(repo) or set()
    stubs: dict[str, tuple[str, int]] = {}
    for rel in sorted(git):
        if not _under(rel, paths):
            continue
        f = repo / rel
        try:
            if f.is_symlink() or not f.is_file() or f.stat().st_size > STUB_MAX:
                continue
            mt = STUB.match(f.read_bytes())
        except OSError:
            continue
        if mt:
            stubs[rel] = (mt.group(1).decode(), int(mt.group(2)))
    fr = fetchset.fetch_blobs(store, stubs.values(), srcs, jobs=jobs)
    done = []
    for rel, (h, _) in stubs.items():
        if store.object_path(h).exists():
            _place(store.object_path(h), repo / rel)
            done.append(rel)
    return {"stubs": len(stubs), "hydrated": done, "fetch": fr, "ok": len(done) == len(stubs)}


# ---- the public export --------------------------------------------------------------------------------------------------------------

def export_ids(doc: dict[str, Any], *, include_previous: bool = True) -> list[str]:
    """Every id the export must hold: the entries, and the ids they superseded (old commits still name them)."""
    out = list(entries(doc))
    if include_previous:
        out += [x for olds in previous(doc).values() for x in olds]
    return list(dict.fromkeys(out))


def _from_history(store: LocalStore, repo: Path, rel: str, want: set[str]) -> None:
    """Put every version of the file `rel` that git's history holds and whose id is in `want` (a superseded id that was never
    preserved: its bytes are still in the history of its path)."""
    p = subprocess.run(["git", "-C", str(repo), "log", "--all", "--format=", "--raw", "--no-abbrev", "--no-renames", "--", rel],
                       capture_output=True, text=True)
    blobs = {line.split()[3] for line in p.stdout.splitlines() if line.startswith(":") and len(line.split()) > 4}
    with tempfile.TemporaryDirectory() as tmp:
        for b in sorted(blobs - {"0" * 40}):
            data = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", b], capture_output=True)
            if data.returncode:
                continue
            f = Path(tmp) / Path(rel).name
            f.write_bytes(data.stdout)
            if fixture_id(f) in want:
                store.put_artifact(FIXTURE_KIND, {}, None, file=f)


def _ensure_local(store: LocalStore, repo: Path, doc: dict[str, Any], want: list[str], srcs: Sequence[Remote], jobs: int,
                  report: dict[str, Any]) -> dict[str, ArtifactManifest]:
    """Every id's manifest and blobs in the local store: put from the repo when its path holds the bytes, else fetched, else (a
    superseded id) recovered from the history of its entry's path."""
    paths = entries(doc)
    for a in want:
        p = repo / paths[a] if a in paths else None
        if not store.manifest_path(ids.hex_of(a)).exists() and p is not None and p.exists() and fixture_id(p) == a:
            store.put_artifact(FIXTURE_KIND, {}, None, file=p if p.is_file() else None, tree=p if p.is_dir() else None)
    ms = fetchset.fetch_manifests(store, want, srcs, jobs=jobs, report=fetchset._report())
    lost = set(want) - set(ms)
    for new, olds in previous(doc).items():
        if lost & set(olds) and new in paths:
            _from_history(store, repo, paths[new], lost & set(olds))
    ms = fetchset.fetch_manifests(store, want, srcs, jobs=jobs, report=report)
    fetchset.fetch_blobs(store, (b for m in ms.values() for b in m.blobs()), srcs, jobs=jobs, report=report)
    return ms


def _split_missing(fr: dict[str, Any], doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """What `fr` could not find, split into superseded ids only the `previous` block names (a warning) and the rest (an error)."""
    current = set(entries(doc))
    old = {x for olds in previous(doc).values() for x in olds} - current
    names = [x.get("id") or x.get("sha256") for x in fr["missing"]]
    return [n for n in names if n in old], [n for n in names if n not in old]


def export(store: LocalStore, repo: Path, dest: Remote, *, srcs: Sequence[Remote], include_previous: bool = True,
           blobs: Sequence[tuple[str, int]] = (), arts: Sequence[str] = (), jobs: int = fetchset.JOBS,
           dry_run: bool = False) -> dict[str, Any]:
    """Copy every registry entry's manifest and blobs (and `arts`, artifacts published that the registry doesn't hold, and `blobs`,
    the history stubs' bytes) to `dest`, blobs first and manifests last, as `push` does: a manifest the export holds is never
    missing a blob.  Objects already there are skipped (conditional create; a key holding different bytes is a RemoteConflict,
    reported).  `dry_run` HEADs and reports what would be uploaded."""
    repo = Path(repo)
    doc = load_map(repo)
    want = list(dict.fromkeys([*export_ids(doc, include_previous=include_previous), *arts]))
    fr = fetchset._report()
    ms = _ensure_local(store, repo, doc, want, srcs, jobs, fr)
    fetchset.fetch_blobs(store, blobs, srcs, jobs=jobs, report=fr)
    objs = list(dict.fromkeys([b for m in ms.values() for b in m.blobs()] + [(h, int(n)) for h, n in blobs]))
    rep: dict[str, Any] = {"dest": dest.describe(), "ids": len(want), "manifests": len(ms), "blobs": len(objs), "dry_run": dry_run,
                           "uploaded": 0, "skipped": 0, "bytes_uploaded": 0, "errors": [], "warnings": [], "fetch": fr}
    lost, current = _split_missing(fr, doc)
    rep["errors"] += [f"not local and not on any source: {x}" for x in current]
    rep["warnings"] += [f"superseded id in neither the store nor git history (no commit can name it): {x}" for x in lost]
    items = [(object_key(h), store.object_path(h), n) for h, n in objs if store.object_path(h).exists()]
    mans = [(manifest_key(ids.hex_of(a)), store.manifest_path(ids.hex_of(a)), store.manifest_path(ids.hex_of(a)).stat().st_size) for a in ms]

    def one(item: tuple[str, Path, int]) -> tuple[str, str, int]:
        key, path, n = item
        if dry_run:
            hd = dest.head(key)
            return key, "skipped" if hd is not None and hd.get("bytes") in (n, None) else "would upload", n
        try:
            return key, dest.put(key, path)["action"], n
        except RemoteConflict as e:
            return key, f"conflict: {e}", n
        except Exception as e:  # noqa: BLE001  (one object's failure is reported; the rest still go up)
            return key, f"error: {type(e).__name__}: {e}", n

    for batch in (items, mans):
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(batch) or 1))) as ex:
            for key, action, n in ex.map(one, batch):
                if action in ("created", "would upload"):
                    rep["uploaded"] += 1
                    rep["bytes_uploaded"] += n
                elif action in ("skipped", "exists"):
                    rep["skipped"] += 1
                else:
                    rep["errors"].append(f"{key}: {action}")
    rep["ok"] = not rep["errors"]
    return rep


def verify_url(repo: Path, url: str, *, include_previous: bool = True, blobs: Sequence[tuple[str, int]] = (),
               jobs: int = fetchset.JOBS) -> dict[str, Any]:
    """Fetch every id the export must hold (and `blobs`) through `url` alone, into a throwaway store, checking every byte against
    its id: the end-to-end test of a public front end."""
    doc = load_map(Path(repo))
    want = export_ids(doc, include_previous=include_previous)
    with tempfile.TemporaryDirectory(prefix="research-verify-url-") as tmp:
        scratch = LocalStore(tmp)
        src = [HttpRemote(url)]
        ms, fr = fetchset.fetch_set(scratch, want, src, jobs=jobs)
        fetchset.fetch_blobs(scratch, blobs, src, jobs=jobs, report=fr)
        scratch.close()
    lost, current = _split_missing(fr, doc)
    return {"url": url, "ids": len(want), "manifests": len(ms), "fetch": fr, "missing": current, "warnings": lost, "ok": not current}

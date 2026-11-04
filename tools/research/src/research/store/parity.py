"""Content parity between a store and a catalog rebuilt from the remote alone: the gate before any machine trusts the remote catalog.

    research data parity --fresh DIR [--store ROOT] [--jobs N] [--json-out FILE]

1. Build a store at DIR from the remote alone (`LocalStore.refresh`); DIR is a new directory (or a previous parity store, caught up
   incrementally), never the store under test.
2. Compare, read-only against the store under test (its record files, and its catalog opened `?mode=ro`):
     artifacts  the same ids, with byte-identical manifests;
     attempts   the same attempt documents (parsed JSON equal);
     labels     the same assertions, by content hash (`labels_sync.label_hash`);
     replicas   every verified remote replica the store under test records is confirmed by HEAD alone (nothing is downloaded):
                the manifest present with its size and an ETag equal to its MD5; each blob present with the manifest's size, and
                where the blob is held locally its ETag equal to the local MD5 (`etag-verified`), else `remote-only, size-verified`
                (an evicted blob is never re-fetched);
3. Every difference gets its cause: `pending here` (the remote lacks it: the store under test never pushed it), `not pulled` (only
   the remote has it: a pod or another machine wrote it and the store under test never pulled it), `uploaded during the run`
   (absent from the listing, present at the re-check); records written during the run are counted, not compared.  A mismatched
   attempt names `research data reconcile-attempts` (the remote's write-once copy is canonical).  Unexplained: a content mismatch, a remote key that failed to fetch or hash,
   an unconfirmed replica.  `ok` (exit 0) iff nothing is unexplained.

A progress line goes to stderr every `PROGRESS_EVERY_S` (phase, done / total, elapsed); every remote request has a hard deadline
with retries (`remote_s3.REQUEST_DEADLINE_S`), so no phase can sit silent on a stuck read.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import ids
from . import labels_sync as LS
from .local import LocalStore, attempt_key, manifest_key, object_key
from .model import ArtifactManifest
from .remote_s3 import is_md5_etag

PROGRESS_EVERY_S = 30.0


class Progress:
    """`phase(name, total)` / `tick(n)` from any thread; a daemon thread logs `parity: <phase> <done>/<total> (<elapsed>s)` every
    `every` seconds while a phase runs, so a long phase is never silent."""

    def __init__(self, log: Callable[[str], None] | None, every: float = PROGRESS_EVERY_S):
        self.log, self.every, self.t0 = log, every, time.monotonic()
        self.name, self.total, self.done = "start", 0, 0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True) if log else None
        if self._thread:
            self._thread.start()

    def phase(self, name: str, total: int = 0) -> None:
        with self._lock:
            self.name, self.total, self.done = name, total, 0
        if self.log:
            self.log(f"parity: {name}" + (f" ({total})" if total else "") + f" at {time.monotonic() - self.t0:.0f}s")

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self.done += n

    def line(self) -> str:
        with self._lock:
            count = f"{self.done}/{self.total}" if self.total else (str(self.done) if self.done else "running")
            return f"parity: {self.name} {count} ({time.monotonic() - self.t0:.0f}s)"

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            self.log(self.line())

    def stop(self) -> None:
        self._stop.set()


def _files(root: Path, sub: str, ok: Callable[[str], bool]) -> dict[str, Path]:
    return {p.stem: p for p in sorted((root / sub).glob("*.json")) if ok(p.stem)}


def _verified_replicas(root: Path) -> list[str]:
    db = root / "catalog.sqlite"
    if not db.is_file():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT id FROM replicas WHERE location='remote' AND verified_utc IS NOT NULL ORDER BY id")]
    finally:
        con.close()


def _md5(p: Path) -> str:
    h = hashlib.md5()  # noqa: S324  (an S3 ETag is an MD5; this is a comparison, not a security use)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(store_root: Path, fresh: LocalStore, *, jobs: int = 32, progress: Progress | None = None) -> dict[str, Any]:
    """The comparison of step 2 and the causes of step 3 (module docstring); `fresh` is already built."""
    prog = progress or Progress(None)
    here = LocalStore(store_root)                                    # record files only: its catalog is never opened read-write here
    r = fresh._require_remote()  # noqa: SLF001
    rep: dict[str, Any] = {"store": str(store_root), "fresh": str(fresh.root), "remote": r.describe(), "unexplained": []}

    def recheck(keys: dict[str, str]) -> dict[str, str]:             # id -> cause, for records only the store under test has
        prog.tick(0)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            heads = dict(zip(keys, ex.map(r.head, keys.values())))
        return {k: "uploaded during the run" if heads[k] is not None else "pending here" for k in keys}

    def side(name: str, mine: dict[str, Any], theirs: dict[str, Any], same: Callable[[str], bool], key: Callable[[str], str]) -> None:
        both = sorted(set(mine) & set(theirs))
        bad = [k for k in both if not same(k)]
        only_here = recheck({k: key(k) for k in sorted(set(mine) - set(theirs))})
        sec = {"both": len(both), "mismatched": bad, "only_here": only_here, "not_pulled": sorted(set(theirs) - set(mine))}
        rep[name] = sec
        rep["unexplained"] += [f"{name} {k}: " + ("the local copy differs from the remote's write-once copy (the remote's is canonical: "
                                                 f"`research data reconcile-attempts {k}`)" if name == "attempts" else "content differs from the remote's")
                               for k in bad]

    prog.phase("compare: artifacts, attempts, labels")
    ma = _files(store_root, "manifests", ids.is_sha256)
    mb = _files(fresh.root, "manifests", ids.is_sha256)
    side("artifacts", ma, mb, lambda h: ma[h].read_bytes() == mb[h].read_bytes(), manifest_key)
    aa = _files(store_root, "attempts", ids.is_run_id)
    ab = _files(fresh.root, "attempts", ids.is_run_id)
    side("attempts", aa, ab, lambda k: json.loads(aa[k].read_bytes()) == json.loads(ab[k].read_bytes()), attempt_key)
    la, lb = LS.local_labels(here), LS.local_labels(fresh)
    side("labels", la, lb, lambda h: True, lambda h: LS.label_key(la[h][1], h))

    # Replica rows: one pass over the remote's listing (size + ETag, 1,000 keys a request) instead of a HEAD per blob -- two fixture
    # trees hold ~259k blobs each, and a per-artifact HEAD loop left one worker on one of them for hours.  Work and progress are per blob.
    rows = _verified_replicas(store_root)
    failed: dict[str, str] = {}
    need: dict[str, list[tuple[str, int, str | None]]] = {}          # art -> [(key, size, md5 of the manifest bytes | None for a blob)]
    for art in rows:
        h = ids.hex_of(art)
        p = ma.get(h) or mb.get(h)
        if p is None:
            failed[art] = "no manifest file in either store"
            continue
        try:
            data = p.read_bytes()
            need[art] = [(manifest_key(h), len(data), hashlib.md5(data).hexdigest())] + \
                        [(object_key(b), n, None) for b, n in ArtifactManifest.from_doc(json.loads(data)).blobs()]  # noqa: S324
        except Exception as e:  # noqa: BLE001
            failed[art] = f"{type(e).__name__}: {e}"
    wanted = {k for items in need.values() for k, _, _ in items}
    seen: dict[str, tuple] = {}
    if hasattr(r, "list_meta"):
        prog.phase("replicas: listing the remote's manifests/ and objects/ (sizes + ETags)")
        for prefix in ("manifests/", "objects/"):
            seen.update(r.list_meta(prefix, progress=prog.tick, keep=wanted.__contains__))
    late = sorted(wanted - set(seen))                                 # a remote without listings, or keys that appeared after the listing
    if late:
        prog.phase("replicas: HEAD for keys the listing did not show", len(late))

        head_errors: dict[str, str] = {}

        def head(k: str) -> tuple[str, tuple | None]:
            try:
                hd = r.head(k)
            except Exception as e:  # noqa: BLE001  (a check that exhausted its retries: named in the report, never a hang)
                head_errors[k] = f"{type(e).__name__}: {e}"
                return k, None
            finally:
                prog.tick()
            return k, (hd["bytes"], hd.get("etag")) if hd else None
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            seen.update({k: v for k, v in ex.map(head, late) if v is not None})
        for art, items in need.items():
            bad = next((k for k, _, _ in items if k in head_errors), None)
            if bad:
                failed[art] = f"{bad}: HEAD failed after its retries ({head_errors[bad]})"
    local_blobs = sorted({k for items in need.values() for k, _, m in items
                          if m is None and k in seen and is_md5_etag(seen[k][1]) and (store_root / k).is_file()})
    prog.phase("replicas: local blobs' MD5 against the remote's ETag", len(local_blobs))

    def md5_of(k: str) -> tuple[str, str | None]:
        try:
            return k, _md5(store_root / k)
        except OSError:
            return k, None
        finally:
            prog.tick()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        local_md5 = dict(ex.map(md5_of, local_blobs))
    counts = {"etag_verified": 0, "remote_only": 0}
    for art, items in need.items():
        if art in failed:
            continue
        for k, n, mmd5 in items:
            got = seen.get(k)
            if got is None:
                failed[art] = f"{k}: absent"
                break
            size, etag = got
            if size != n:
                failed[art] = f"{k}: remote size {size} != {n}"
                break
            if mmd5 is not None:                                      # the manifest: ETag == MD5 of its bytes
                if is_md5_etag(etag) and etag != mmd5:
                    failed[art] = f"{k}: remote size/ETag does not match the manifest"
                    break
            elif local_md5.get(k):
                if local_md5[k] != etag:
                    failed[art] = f"{k}: remote ETag {etag} != the local blob's MD5"
                    break
                counts["etag_verified"] += 1
            else:
                counts["remote_only"] += 1
    rep["replicas"] = {"rows": len(rows), "confirmed": len(rows) - len(failed), "not_confirmed": failed,
                       "blobs_etag_verified": counts["etag_verified"], "blobs_remote_only_size_verified": counts["remote_only"]}
    rep["unexplained"] += [f"replica {a}: {why}" for a, why in failed.items()]

    rep["ok"] = not rep["unexplained"]
    return rep


def run(store_root: Path, fresh_dir: Path, *, jobs: int = 32, log: Callable[[str], None] | None = None) -> dict:
    """Steps 1-3: build `fresh_dir` from the remote alone, then `compare`."""
    store_root, fresh_dir = store_root.expanduser().resolve(), fresh_dir.expanduser().resolve()
    if fresh_dir == store_root or store_root in fresh_dir.parents or fresh_dir in store_root.parents:
        raise ValueError(f"--fresh {fresh_dir} must be a separate directory from the store under test {store_root}")
    t0 = time.monotonic()
    prog = Progress(log)
    try:
        fresh = LocalStore(fresh_dir)
        prog.phase("build: the fresh store from the remote alone")
        built = fresh.refresh(jobs=jobs, log=log, progress=lambda what, done, total: (prog.phase(f"build: {what}", total) if done == 0 else prog.tick()))
        rep = compare(store_root, fresh, jobs=jobs, progress=prog)
    finally:
        prog.stop()
    rep["build"] = built
    rep["unexplained"] = [f"remote key not fetched: {e}" for e in built["errors"]] + rep["unexplained"]
    remote = getattr(fresh, "_remote", None)
    rep["remote_requests"] = {"deadline_hits": list(getattr(remote, "deadline_hits", []) or []),
                              "exhausted": list(getattr(remote, "exhausted", []) or [])}
    rep["ok"] = not rep["unexplained"]
    rep["seconds"] = round(time.monotonic() - t0, 1)
    return rep


def format_report(rep: dict) -> list[str]:
    lines = [f"parity: {rep['store']} vs {rep['fresh']} (built from {rep['remote']}: +{rep['build']['manifests']} manifests, "
             f"+{rep['build']['attempts']} attempts, +{rep['build']['labels']} labels; {rep['seconds']}s)"]
    for name in ("artifacts", "attempts", "labels"):
        s = rep[name]
        causes: dict[str, int] = {}
        for c in s["only_here"].values():
            causes[c] = causes.get(c, 0) + 1
        lines.append(f"{name:<10} {s['both']} on both, {len(s['mismatched'])} mismatched, "
                     + ", ".join(f"{n} {c}" for c, n in sorted(causes.items()))
                     + (", " if causes else "") + f"{len(s['not_pulled'])} not pulled (remote only)")
    rp = rep["replicas"]
    lines.append(f"{'replicas':<10} {rp['confirmed']}/{rp['rows']} verified replica rows confirmed by HEAD on the remote (blobs: "
                 f"{rp.get('blobs_etag_verified', 0)} ETag == local MD5, {rp.get('blobs_remote_only_size_verified', 0)} remote-only, size-verified; "
                 "nothing downloaded)")
    rr = rep.get("remote_requests") or {}
    if rr.get("deadline_hits") or rr.get("exhausted"):
        lines.append(f"{'requests':<10} {len(rr.get('deadline_hits') or [])} cut off by their deadline, {len(rr.get('exhausted') or [])} failed on "
                     f"every try: " + ", ".join(f"{x['method']} {x['key']}" for x in (rr.get("exhausted") or rr.get("deadline_hits"))[:5]))
    for u in rep["unexplained"][:30]:
        lines.append(f"  UNEXPLAINED {u}")
    lines.append("PARITY OK: every difference is explained" if rep["ok"] else f"PARITY FAILED: {len(rep['unexplained'])} unexplained difference(s)")
    return lines


__all__ = ["Progress", "compare", "format_report", "run"]

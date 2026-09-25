"""Content parity between a store and a catalog rebuilt from the remote alone: the gate before any machine trusts the remote catalog.

    research data parity --fresh DIR [--store ROOT] [--render] [--jobs N] [--json-out FILE]

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
     render     with `--render`: the daily render's own command (`notes.render_lines`: `verity_numerical.bench.tables` and
                `.drilldown --format md`, the steward.toml `[[render]] source` tree on PYTHONPATH and as cwd; `--render-source`
                overrides) over the SAME records on both sides: the store under test's snapshot (the record files compared above,
                frozen, never the live directories) plus the records only the remote holds, against the fresh store plus the records
                only the store under test holds.  Views are per-file symlink stores with their own catalog and the fresh store's
                blob cache.  The store under test is also rendered itself, exactly as the daily render reads it, and must match its
                snapshot view: records written to it while parity runs are reported as concurrency, not as a difference.
3. Every difference gets its cause: `pending here` (the remote lacks it: the store under test never pushed it), `not pulled` (only
   the remote has it: a pod or another machine wrote it and the store under test never pulled it), `uploaded during the run`
   (absent from the listing, present at the re-check); records written during the run are counted, not compared.  A mismatched
   attempt names `research data reconcile-attempts` (the remote's write-once copy is canonical).  Unexplained: a content mismatch, a remote key that failed to fetch or hash,
   an unconfirmed replica, a render difference.  `ok` (exit 0) iff nothing is unexplained.

A progress line goes to stderr every `PROGRESS_EVERY_S` (phase, done / total, elapsed); every remote request has a hard deadline
with retries (`remote_s3.REQUEST_DEADLINE_S`), so no phase can sit silent on a stuck read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
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

RENDER_TIMEOUT_S = 1800
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


def render_view(fresh_root: Path, name: str, records: list[Path]) -> Path:
    """A store of per-file symlinks to exactly `records` (manifest, attempt and label files: a frozen snapshot, never the live
    directories), with its own catalog built from them and the fresh store's blob cache.  Rendering it downloads nothing the fresh
    render did not, and writes nothing where the records live."""
    import shutil  # noqa: PLC0415

    view = fresh_root / f".view-{name}"
    if view.exists():
        shutil.rmtree(view)
    view.mkdir()
    for i, p in enumerate(records):
        sub = "labels/" + p.parent.name if p.parent.parent.name == "labels" else p.parent.name
        dst = view / sub / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst = dst.with_name(f"{dst.stem}~v{i}{dst.suffix}")
        dst.symlink_to(p)
    for sub in ("manifests", "attempts", "labels"):
        (view / sub).mkdir(exist_ok=True)
    (view / "objects").symlink_to(fresh_root / "objects", target_is_directory=True)
    st = LocalStore(view)
    st.reindex()
    st.close()
    return view


RENDER_MODULES = (("tables", "tables"), ("drilldown", "drilldowns"))


def render_source(explicit: Path | None = None, notes_root: Path | None = None) -> tuple[Path, str]:
    """The tree the daily render runs (`notes.render_lines`): `explicit`, else the first `[[render]] source` of the notes'
    steward.toml, else the checkout this code lives in.  -> (tree, how it was chosen)."""
    import tomllib  # noqa: PLC0415

    if explicit:
        return Path(explicit).expanduser().resolve(), "--render-source"
    root = notes_root or Path(os.environ.get("RESEARCH_NOTES", Path.home() / ".research" / "notes"))
    try:
        entries = tomllib.loads((root / "steward.toml").read_text()).get("render") or []
        if entries and entries[0].get("source"):
            return Path(str(entries[0]["source"])).expanduser().resolve(), f"{root / 'steward.toml'} [[render]] source"
    except (OSError, ValueError):
        pass
    return Path(__file__).resolve().parents[4], "this checkout (no steward.toml [[render]] entry)"


def _render(root: Path, timeout: float, source: Path | None = None) -> tuple[str | None, str | None]:
    """The daily render's own invocation (`notes.render_lines`): each of `verity_numerical.bench.tables` / `.drilldown --root ROOT
    --format md` with the source tree on PYTHONPATH and as the cwd; the outputs joined.  -> (text, None) or (None, why)."""
    src = source or render_source()[0]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(src / s) for s in ("backends/numerical/python", "tools/research/src", "."))}
    parts = []
    for mod, suffix in RENDER_MODULES:
        try:
            out = subprocess.run([sys.executable, "-m", f"verity_numerical.bench.{mod}", "--root", str(root), "--format", "md"],
                                 capture_output=True, text=True, timeout=timeout, env=env, cwd=src)
        except (OSError, subprocess.SubprocessError) as e:
            return None, f"{mod}: {type(e).__name__}: {e}"
        if out.returncode:
            return None, f"{mod}: rc {out.returncode}: {out.stderr.strip()[-300:]}"
        parts.append(f"<!-- {suffix} -->\n{out.stdout}")
    return "\n".join(parts), None


def _diff(a: str, b: str) -> list[str]:
    la, lb = a.splitlines(), b.splitlines()
    sa, sb = set(la), set(lb)
    return [f"- {x}" for x in la if x not in sb] + [f"+ {x}" for x in lb if x not in sa]


def _label_files(root: Path) -> set[Path]:
    return set((root / "labels").glob("*/*.json"))


def compare(store_root: Path, fresh: LocalStore, *, jobs: int = 32, render: bool = False, progress: Progress | None = None,
            source: Path | None = None) -> dict[str, Any]:
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
    snap_labels = _label_files(store_root)
    la, lb = LS.local_labels(here), LS.local_labels(fresh)
    side("labels", la, lb, lambda h: True, lambda h: LS.label_key(la[h][1], h))

    counts = {"etag_verified": 0, "remote_only": 0}
    clock = threading.Lock()

    def confirm(art: str) -> str | None:
        h = ids.hex_of(art)
        p = ma.get(h) or mb.get(h)
        try:
            if p is None:
                return "no manifest file in either store"
            data = p.read_bytes()
            hd = r.head(manifest_key(h))
            if hd is None:
                return f"{manifest_key(h)}: absent"
            if hd["bytes"] != len(data) or (is_md5_etag(hd.get("etag")) and hd["etag"] != hashlib.md5(data).hexdigest()):  # noqa: S324
                return f"{manifest_key(h)}: remote size/ETag does not match the manifest"
            for b, n in ArtifactManifest.from_doc(json.loads(data)).blobs():
                k = object_key(b)
                hd = r.head(k)
                if hd is None:
                    return f"{k}: absent"
                if hd["bytes"] != n:
                    return f"{k}: remote size {hd['bytes']} != {n}"
                local = store_root / k
                if local.is_file() and is_md5_etag(hd.get("etag")):
                    if hd["etag"] != _md5(local):
                        return f"{k}: remote ETag {hd['etag']} != the local blob's MD5"
                    with clock:
                        counts["etag_verified"] += 1
                else:
                    with clock:
                        counts["remote_only"] += 1
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        finally:
            prog.tick()
        return None

    rows = _verified_replicas(store_root)
    prog.phase("replicas: HEAD + ETag, nothing downloaded", len(rows))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        failed = {a: why for a, why in zip(rows, ex.map(confirm, rows)) if why}
    rep["replicas"] = {"rows": len(rows), "confirmed": len(rows) - len(failed), "not_confirmed": failed,
                       "blobs_etag_verified": counts["etag_verified"], "blobs_remote_only_size_verified": counts["remote_only"]}
    rep["unexplained"] += [f"replica {a}: {why}" for a, why in failed.items()]

    if render:
        src, how = render_source(source)
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        rd: dict[str, Any] = {"source": str(src), "source_chosen_by": how, "source_commit": commit}
        rep["render"] = rd
        # both sides over the same records: the frozen snapshot of the store under test plus what only the remote holds, and the fresh
        # store plus what only the store under test holds; a mismatched record keeps each side's own copy (its effect shows)
        mine = list(ma.values()) + list(aa.values()) + [la[h][0] for h in la]
        theirs = list(mb.values()) + list(ab.values()) + [lb[h][0] for h in lb]
        from_remote = [mb[h] for h in rep["artifacts"]["not_pulled"]] + [ab[k] for k in rep["attempts"]["not_pulled"]] + \
                      [lb[h][0] for h in rep["labels"]["not_pulled"]]
        from_here = [ma[h] for h in rep["artifacts"]["only_here"]] + [aa[k] for k in rep["attempts"]["only_here"]] + \
                    [la[h][0] for h in rep["labels"]["only_here"]]
        rd["records"] = {"common": len(mine) - len(from_here), "only_remote": len(from_remote), "only_here": len(from_here)}
        prog.phase("render: the fresh store + the store under test's pending records (the real render command)")
        c, ec = _render(render_view(fresh.root, "fresh", theirs + from_here), RENDER_TIMEOUT_S, src)
        prog.phase("render: the store under test's snapshot + the remote-only records")
        b_, eb = (None, ec) if ec else _render(render_view(fresh.root, "store-under-test", mine + from_remote), RENDER_TIMEOUT_S, src)
        prog.phase("render: the store under test itself, as the daily render reads it")
        real, er = (None, eb) if eb else _render(store_root, RENDER_TIMEOUT_S, src)
        if not er and from_remote:
            prog.phase("render: the store under test's snapshot alone")
            b0, er = _render(render_view(fresh.root, "snapshot", mine), RENDER_TIMEOUT_S, src)
        else:
            b0 = b_
        late = {"manifests": len(set(_files(store_root, "manifests", ids.is_sha256)) - set(ma)),
                "attempts": len(set(_files(store_root, "attempts", ids.is_run_id)) - set(aa)),
                "labels": len(_label_files(store_root) - snap_labels)}
        rep["written_during_run"] = late
        if ec or eb or er:
            rd.update(identical=None, error=ec or eb or er)
            rep["unexplained"].append(f"render failed: {ec or eb or er}")
        else:
            d = _diff(b_, c)
            rd.update(identical=b_ == c, diff=d[:200], bytes=[len(b_), len(c)])
            if b_ != c:
                n_mis = sum(len(rep[k]["mismatched"]) for k in ("artifacts", "attempts", "labels"))
                rep["unexplained"].append(f"render differs ({len(d)} lines) over the same records"
                                          + (f"; the {n_mis} mismatched record(s) above can account for it" if n_mis else ""))
            dv = _diff(real, b0)
            rd["view_matches_real"] = real == b0
            rd["view_vs_real_diff"] = dv[:200]
            if real != b0:
                if any(late.values()):
                    rd["view_vs_real_cause"] = (f"records written to the store under test during the run (concurrency, not compared): "
                                                f"{late['manifests']} manifests, {late['attempts']} attempts, {late['labels']} labels")
                else:
                    rep["unexplained"].append(f"the snapshot view renders differently from the store under test itself ({len(dv)} lines)")
    rep["ok"] = not rep["unexplained"]
    return rep


def run(store_root: Path, fresh_dir: Path, *, jobs: int = 32, render: bool = False, log: Callable[[str], None] | None = None,
        source: Path | None = None) -> dict:
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
        rep = compare(store_root, fresh, jobs=jobs, render=render, progress=prog, source=source)
    finally:
        prog.stop()
    rep["build"] = built
    rep["unexplained"] = [f"remote key not fetched: {e}" for e in built["errors"]] + rep["unexplained"]
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
    if "render" in rep:
        rd = rep["render"]
        rc = rd.get("records") or {}
        lines.append(f"{'render':<10} " + ("identical" if rd.get("identical") else f"FAILED: {rd['error']}" if rd.get("error")
                                           else f"DIFFERENT ({len(rd.get('diff') or [])} lines)")
                     + f" over the same records ({rc.get('common', 0)} common + {rc.get('only_remote', 0)} remote-only + {rc.get('only_here', 0)} "
                     f"pending here); tables + drilldowns from {rd.get('source')} @ {(rd.get('source_commit') or '?')[:8]} ({rd.get('source_chosen_by')})")
        if "view_matches_real" in rd:
            lines.append(f"{'':<10} the snapshot view " + ("renders exactly as the store under test itself" if rd["view_matches_real"] else
                         f"differs from the store under test's own render ({len(rd['view_vs_real_diff'])} lines): "
                         + rd.get("view_vs_real_cause", "UNEXPLAINED")))
    late = rep.get("written_during_run") or {}
    if any(late.values()):
        lines.append(f"{'during run':<10} written to the store under test while parity ran (concurrency; not compared): {late['manifests']} "
                     f"manifests, {late['attempts']} attempts, {late['labels']} labels")
    for u in rep["unexplained"][:30]:
        lines.append(f"  UNEXPLAINED {u}")
    lines.append("PARITY OK: every difference is explained" if rep["ok"] else f"PARITY FAILED: {len(rep['unexplained'])} unexplained difference(s)")
    return lines


__all__ = ["Progress", "compare", "format_report", "render_source", "render_view", "run"]

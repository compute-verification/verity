"""Content parity between a store and a catalog rebuilt from the remote alone: the gate before any machine trusts the remote catalog.

    research data parity --fresh DIR [--store ROOT] [--render] [--jobs N] [--json-out FILE]

1. Build a store at DIR from the remote alone (`LocalStore.refresh`); DIR is a new directory (or a previous parity store, caught up
   incrementally), never the store under test.
2. Compare, read-only against the store under test (its record files, and its catalog opened `?mode=ro`):
     artifacts  the same ids, with byte-identical manifests;
     attempts   the same attempt documents (parsed JSON equal);
     labels     the same assertions, by content hash (`labels_sync.label_hash`);
     replicas   every verified remote replica the store under test records is confirmed by a HEAD on the remote: the manifest and
                every blob present with the manifest's size;
     render     with `--render`: `python -m verity_numerical.bench.tables --root <store> --format md` prints the same on both.
3. Every difference gets its cause: `pending here` (the remote lacks it: the store under test never pushed it), `not pulled` (only
   the remote has it: a pod or another machine wrote it and the store under test never pulled it), `uploaded during the run`
   (absent from the listing, present at the re-check).  Unexplained: a content mismatch, a remote key that failed to fetch or hash,
   an unconfirmed replica, a render difference.  `ok` (exit 0) iff nothing is unexplained.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import ids
from . import labels_sync as LS
from .local import LocalStore, attempt_key, manifest_key, object_key
from .model import ArtifactManifest

RENDER_TIMEOUT_S = 1800


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


def _render(root: Path, timeout: float) -> tuple[str | None, str | None]:
    try:
        out = subprocess.run([sys.executable, "-m", "verity_numerical.bench.tables", "--root", str(root), "--format", "md"],
                             capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {e}"
    if out.returncode:
        return None, f"rc {out.returncode}: {out.stderr.strip()[-300:]}"
    return out.stdout, None


def compare(store_root: Path, fresh: LocalStore, *, jobs: int = 32, render: bool = False) -> dict[str, Any]:
    """The comparison of step 2 and the causes of step 3 (module docstring); `fresh` is already built."""
    here = LocalStore(store_root)                                    # record files only: its catalog is never opened read-write here
    r = fresh._require_remote()  # noqa: SLF001
    rep: dict[str, Any] = {"store": str(store_root), "fresh": str(fresh.root), "remote": r.describe(), "unexplained": []}

    def recheck(keys: dict[str, str]) -> dict[str, str]:             # id -> cause, for records only the store under test has
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            heads = dict(zip(keys, ex.map(r.head, keys.values())))
        return {k: "uploaded during the run" if heads[k] is not None else "pending here" for k in keys}

    def side(name: str, mine: dict[str, Any], theirs: dict[str, Any], same: Callable[[str], bool], key: Callable[[str], str]) -> None:
        both = sorted(set(mine) & set(theirs))
        bad = [k for k in both if not same(k)]
        only_here = recheck({k: key(k) for k in sorted(set(mine) - set(theirs))})
        sec = {"both": len(both), "mismatched": bad, "only_here": only_here, "not_pulled": sorted(set(theirs) - set(mine))}
        rep[name] = sec
        rep["unexplained"] += [f"{name} {k}: content differs from the remote's" for k in bad]

    ma = _files(store_root, "manifests", ids.is_sha256)
    mb = _files(fresh.root, "manifests", ids.is_sha256)
    side("artifacts", ma, mb, lambda h: ma[h].read_bytes() == mb[h].read_bytes(), manifest_key)
    aa = _files(store_root, "attempts", ids.is_run_id)
    ab = _files(fresh.root, "attempts", ids.is_run_id)
    side("attempts", aa, ab, lambda k: json.loads(aa[k].read_bytes()) == json.loads(ab[k].read_bytes()), attempt_key)
    la, lb = LS.local_labels(here), LS.local_labels(fresh)
    side("labels", la, lb, lambda h: True, lambda h: LS.label_key(la[h][1], h))

    def confirm(art: str) -> str | None:
        h = ids.hex_of(art)
        p = ma.get(h) or mb.get(h)
        if p is None:
            return "no manifest file in either store"
        try:
            data = p.read_bytes()
            keys = [(manifest_key(h), len(data))] + [(object_key(b), n) for b, n in ArtifactManifest.from_doc(json.loads(data)).blobs()]
            for k, n in keys:
                hd = r.head(k)
                if hd is None:
                    return f"{k}: absent"
                if hd["bytes"] != n:
                    return f"{k}: remote size {hd['bytes']} != {n}"
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
        return None

    rows = _verified_replicas(store_root)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        failed = {a: why for a, why in zip(rows, ex.map(confirm, rows)) if why}
    rep["replicas"] = {"rows": len(rows), "confirmed": len(rows) - len(failed), "not_confirmed": failed}
    rep["unexplained"] += [f"replica {a}: {why}" for a, why in failed.items()]

    if render:
        a, ea = _render(store_root, RENDER_TIMEOUT_S)
        b, eb = _render(fresh.root, RENDER_TIMEOUT_S)
        if ea or eb:
            rep["render"] = {"identical": None, "error": ea or eb}
            rep["unexplained"].append(f"render failed: {ea or eb}")
        else:
            la_, lb_ = a.splitlines(), b.splitlines()
            sa, sb = set(la_), set(lb_)
            diff = [f"- {x}" for x in la_ if x not in sb] + [f"+ {x}" for x in lb_ if x not in sa]
            rep["render"] = {"identical": a == b, "diff": diff[:200], "bytes": [len(a), len(b)]}
            if a != b:
                rep["unexplained"].append(f"render differs ({len(diff)} lines)")
    rep["ok"] = not rep["unexplained"]
    return rep


def run(store_root: Path, fresh_dir: Path, *, jobs: int = 32, render: bool = False, log: Callable[[str], None] | None = None) -> dict:
    """Steps 1-3: build `fresh_dir` from the remote alone, then `compare`."""
    store_root, fresh_dir = store_root.expanduser().resolve(), fresh_dir.expanduser().resolve()
    if fresh_dir == store_root or store_root in fresh_dir.parents or fresh_dir in store_root.parents:
        raise ValueError(f"--fresh {fresh_dir} must be a separate directory from the store under test {store_root}")
    t0 = time.monotonic()
    fresh = LocalStore(fresh_dir)
    built = fresh.refresh(jobs=jobs, log=log)
    rep = compare(store_root, fresh, jobs=jobs, render=render)
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
    lines.append(f"{'replicas':<10} {rp['confirmed']}/{rp['rows']} verified replica rows confirmed by HEAD on the remote")
    if "render" in rep:
        rd = rep["render"]
        lines.append(f"{'render':<10} " + ("identical" if rd.get("identical") else f"FAILED: {rd['error']}" if rd.get("error")
                                           else f"DIFFERENT ({len(rd['diff'])} lines)"))
    for u in rep["unexplained"][:30]:
        lines.append(f"  UNEXPLAINED {u}")
    lines.append("PARITY OK: every difference is explained" if rep["ok"] else f"PARITY FAILED: {len(rep['unexplained'])} unexplained difference(s)")
    return lines


__all__ = ["compare", "format_report", "run"]

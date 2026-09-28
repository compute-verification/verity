"""Evict local copies whose bytes are PRESERVED on the remote (README §7): the remote is the durable home, `objects/` is a hot cache.

Two kinds of local copy can go, and only these:

* **store blobs** (`objects/sha256/<hex>`): a blob is evictable when EVERY artifact that holds it has a remote replica with a
  `verified_utc` in the catalog (PRESERVED), the local object has the manifest's size, and the blob is at least `min_bytes`.
  Largest first, until the target free space is reached.  `LocalStore._resolve_blob` re-fetches an evicted blob on demand;
  `has_local` / `where` report the eviction honestly, and `index.set_presence(local=...)` is refreshed for every touched artifact.
  `trees/` (materialised tree copies) is a pure cache and is cleared whole unless `keep_trees`.
* **run-directory files** (`~/.research/runs/<run>/...`, `evict_runs`): the big files of a finished run that appear, with the same
  size (and the same sha256 when the file is <= `hash_max` bytes), in a PRESERVED tree artifact of its attempt: `run_record` /
  `run_files`, or a typed output its `outputs.json` declares, laid out under that output's `tree` path (a Build's `outputs/build`).
  Oldest run first.  A run touched in the last `quiet_s` seconds, or whose status is not terminal, is left alone.
  `run-files.evicted.json` in the run dir lists what went and the `research data fetch`es that bring it back.

Never touched: pending (unpushed or unverified) objects, `manifests/`, `attempts/`, `labels/`, `catalog.sqlite`, `scratch/`.
Artifacts that are flagged `remote` in the catalog but never verified are counted in `held_back_bytes` so the operator can
`research data verify` / `push` them first.  `free_fn` is the free-space probe (an `os.statvfs` over the store root by default);
tests inject their own.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .local import LocalStore, sha256_file
from .model import utc_now

MIN_BYTES = 1 << 20          # blobs / files below this are not worth an eviction (and a re-fetch)
HASH_MAX = 64 << 20          # run files up to this size are sha256-checked against the artifact before eviction
QUIET_S = 30 * 60            # a run directory touched more recently than this is still being worked on
TERMINAL_STATES = ("done", "failed", "dead", "timed_out", "exited", "cancelled")
EVICTED_SCHEMA = "research/run-files-evicted/v0.1"
EVICTED_FILE = "run-files.evicted.json"


def free_bytes(path: Path | str) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


def verified_remote(store: LocalStore) -> set[str]:
    """Artifacts whose remote replica carries a `verified_utc` (PRESERVED per README §7)."""
    return {r[0] for r in store.index.db.execute("SELECT id FROM replicas WHERE location='remote' AND verified_utc IS NOT NULL")}


def flagged_remote(store: LocalStore) -> set[str]:
    """Artifacts the catalog flags as present on the remote (pushed or listed), verified or not."""
    return {r[0] for r in store.index.db.execute("SELECT id FROM artifacts WHERE remote=1")}


def plan_store(store: LocalStore, *, min_bytes: int = MIN_BYTES, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """What could go: `candidates` = [(bytes, hex, holders)] largest first, `trees_bytes`, `held_back_bytes` (local blobs of
    artifacts flagged remote but not verified), `skipped` (size mismatch / unreadable manifest)."""
    say = log or (lambda s: None)
    verified, flagged = verified_remote(store), flagged_remote(store)
    holders: dict[str, set[str]] = {}
    sizes: dict[str, int] = {}
    skipped: list[str] = []
    for art in store.artifact_ids():
        try:
            m = store.get_manifest(art)
        except Exception as e:  # noqa: BLE001  one unreadable manifest must not stop the sweep
            skipped.append(f"{art}: {type(e).__name__}: {e}")
            say(f"skip {art[:20]}: {e}")
            continue
        for h, n in m.blobs():
            holders.setdefault(h, set()).add(art)
            sizes[h] = n
    cands: list[tuple[int, str, set[str]]] = []
    held_back = 0
    for h, arts in holders.items():
        p = store.object_path(h)
        if not p.exists() or sizes[h] < min_bytes:
            continue
        if not arts <= verified:
            if arts <= (verified | flagged):
                held_back += sizes[h]
            continue
        if p.stat().st_size != sizes[h]:
            skipped.append(f"{h}: local size {p.stat().st_size} != manifest {sizes[h]}; keeping")
            say(f"size mismatch, keeping {h[:16]}")
            continue
        cands.append((sizes[h], h, arts))
    cands.sort(key=lambda t: (-t[0], t[1]))
    trees = store.root / "trees"
    trees_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(trees) for f in fs) if trees.is_dir() else 0
    return {"candidates": cands, "evictable_bytes": sum(c[0] for c in cands), "trees_bytes": trees_bytes, "held_back_bytes": held_back,
            "skipped": skipped, "verified_artifacts": len(verified)}


def evict_store(store: LocalStore, *, target_free_bytes: int, dry_run: bool = False, min_bytes: int = MIN_BYTES, keep_trees: bool = False,
                free_fn: Callable[[], int] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Evict verified-remote blobs (largest first) and the `trees/` cache until `free_fn()` >= `target_free_bytes`.  The report
    says what was (or, with `dry_run`, would be) removed; nothing else is touched."""
    say = log or (lambda s: None)
    free = free_fn or (lambda: free_bytes(store.root))
    plan = plan_store(store, min_bytes=min_bytes, log=log)
    rep: dict[str, Any] = {"store": str(store.root), "dry_run": dry_run, "target_free_bytes": target_free_bytes, "free_before": free(),
                           "evictable_bytes": plan["evictable_bytes"], "evictable_blobs": len(plan["candidates"]), "held_back_bytes": plan["held_back_bytes"],
                           "trees_bytes": plan["trees_bytes"], "trees_removed": False, "evicted_bytes": 0, "evicted_blobs": [], "artifacts_touched": [],
                           "skipped": plan["skipped"], "reached_target": False}
    say(f"free {rep['free_before'] / 1e9:.2f} GB, target {target_free_bytes / 1e9:.2f} GB; {len(plan['candidates'])} evictable blobs "
        f"({plan['evictable_bytes'] / 1e9:.2f} GB), trees/ {plan['trees_bytes'] / 1e9:.2f} GB, held back (remote but unverified) {plan['held_back_bytes'] / 1e9:.2f} GB")
    trees = store.root / "trees"
    if plan["trees_bytes"] and not keep_trees and free() < target_free_bytes:
        if not dry_run:
            for entry in os.listdir(trees):
                shutil.rmtree(trees / entry, ignore_errors=True)
        rep["trees_removed"] = True
        say(f"trees/ {plan['trees_bytes'] / 1e9:.2f} GB (materialised cache) {'would be' if dry_run else ''} removed")
    simulated = free() + (plan["trees_bytes"] if dry_run and rep["trees_removed"] else 0)
    touched: set[str] = set()
    for n, h, arts in plan["candidates"]:
        if (simulated if dry_run else free()) >= target_free_bytes:
            break
        if not dry_run:
            p = store.object_path(h)
            os.chmod(p, 0o644)
            p.unlink()
        simulated += n
        rep["evicted_bytes"] += n
        rep["evicted_blobs"].append({"sha256": h, "bytes": n, "artifacts": sorted(arts)})
        touched |= arts
    if not dry_run:
        for art in sorted(touched):
            try:
                store.index.set_presence(art, local=store.has_local(art))
            except Exception as e:  # noqa: BLE001
                say(f"presence update failed for {art[:20]}: {e}")
    rep["artifacts_touched"] = sorted(touched)
    rep["free_after"] = simulated if dry_run else free()
    rep["reached_target"] = rep["free_after"] >= target_free_bytes
    say(f"{'would evict' if dry_run else 'evicted'} {rep['evicted_bytes'] / 1e9:.2f} GB in {len(rep['evicted_blobs'])} blobs across {len(touched)} artifacts; "
        f"free {'would be' if dry_run else 'now'} {rep['free_after'] / 1e9:.2f} GB" + ("" if rep["reached_target"] else " (target NOT reached)"))
    return rep


# ---- run directories ---------------------------------------------------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    return sha256_file(p)[0]


def plan_runs(store: LocalStore, runs_root: Path, *, min_bytes: int = MIN_BYTES, quiet_s: float = QUIET_S, hash_max: int = HASH_MAX,
              now: float | None = None, log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Per run dir with evictable files: `{"run", "dir", "artifact", "files": [{"path", "bytes", "sha256", "artifact"}], "restore"}`,
    oldest run first; `artifact` is the first of the run's PRESERVED trees (`_sources`) that holds one of them."""
    say = log or (lambda s: None)
    now = time.time() if now is None else now
    verified = verified_remote(store)
    plan: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return plan
    for run in sorted(os.listdir(runs_root)):
        rd = runs_root / run
        if not rd.is_dir() or (rd / EVICTED_FILE).exists():
            continue
        big: list[tuple[str, int]] = []
        newest = 0.0
        for dp, _, fs in os.walk(rd):
            for f in fs:
                p = Path(dp) / f
                try:
                    st = p.stat()
                except OSError:
                    continue
                newest = max(newest, st.st_mtime)
                if st.st_size >= min_bytes:
                    big.append((p.relative_to(rd).as_posix(), st.st_size))
        if not big:
            continue
        big.sort(key=lambda t: (-t[1], t[0]))                      # largest first, not the filesystem's listing order
        if now - newest < quiet_s:
            say(f"{run}: touched {int((now - newest) / 60)} min ago, skipping")
            continue
        att = store.attempt_path(run)
        if not att.exists():
            say(f"{run}: no attempt record ({sum(b for _, b in big) / 1e6:.0f} MB big files), skipping")
            continue
        try:
            a = json.loads(att.read_bytes())
        except ValueError:
            say(f"{run}: unreadable attempt record, skipping")
            continue
        state = (a.get("status") or {}).get("state")
        if state not in TERMINAL_STATES:
            say(f"{run}: state {state!r}, skipping")
            continue
        sources = _sources(rd, a.get("outputs") or {})
        if not sources:
            say(f"{run}: no run_record, run_files or tree output artifact ({sum(b for _, b in big) / 1e6:.0f} MB), skipping")
            continue
        preserved = [(art, under) for art, under in sources if art in verified]
        if not preserved:
            say(f"{run}: {', '.join(art[:20] for art, _ in sources)} not verified on the remote, skipping")
            continue
        tree: dict[str, tuple[str, int, str]] = {}
        for art, under in preserved:
            try:
                m = store.get_manifest(art)
            except Exception as e:  # noqa: BLE001
                say(f"{run}: manifest {art[:20]} unreadable ({e}), keeping its files")
                continue
            for tf in m.payload.files if m.payload and m.payload.type == "tree" else []:
                tree.setdefault(f"{under}/{tf.path}" if under else tf.path, (tf.sha256, tf.bytes, art))
        files = []
        for rel, size in big:
            if rel not in tree or tree[rel][1] != size:
                continue
            if size <= hash_max and _sha256(rd / rel) != tree[rel][0]:
                say(f"{run}: {rel} sha256 differs from the artifact, keeping")
                continue
            files.append({"path": rel, "bytes": size, "sha256": tree[rel][0], "artifact": tree[rel][2]})
        if files:
            used = [(art, under) for art, under in preserved if any(f["artifact"] == art for f in files)]
            plan.append({"run": run, "dir": str(rd), "artifact": used[0][0], "files": files,
                         "restore": " && ".join(f"research data fetch {art} --to {rd / under if under else rd}" for art, under in used)})
    return plan


def _sources(rd: Path, outs: dict[str, Any]) -> list[tuple[str, str]]:
    """The attempt's tree artifacts that hold copies of the run dir's files, as (art id, the run-dir path the tree is laid out
    under, "" for the run dir itself): `run_record` / `run_files`, a `files` output (staged relative to the run dir) and a `tree`
    output at its `tree` path, as the run's `outputs.json` declares them (`attempt.typed_outputs`)."""
    src = [(outs[k], "") for k in ("run_record", "run_files") if isinstance(outs.get(k), str)]
    try:
        doc = json.loads((rd / "outputs.json").read_bytes())
    except (OSError, ValueError):
        doc = None
    entries = doc.get("outputs") if isinstance(doc, dict) else None
    for ent in entries if isinstance(entries, list) else []:
        art = outs.get(ent.get("name")) if isinstance(ent, dict) else None
        if not isinstance(art, str) or not art.startswith("art:"):
            continue
        if "files" in ent:
            src.append((art, ""))
        elif isinstance(ent.get("tree"), str) and ent["tree"]:
            p = Path(ent["tree"])
            if not p.is_absolute() and ".." not in p.parts and p.as_posix() != ".":
                src.append((art, p.as_posix()))
    return src


def evict_runs(store: LocalStore, runs_root: Path, *, target_free_bytes: int, dry_run: bool = False, min_bytes: int = MIN_BYTES,
               quiet_s: float = QUIET_S, hash_max: int = HASH_MAX, free_fn: Callable[[], int] | None = None, now: float | None = None,
               log: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = log or (lambda s: None)
    free = free_fn or (lambda: free_bytes(runs_root if runs_root.exists() else store.root))
    plan = plan_runs(store, runs_root, min_bytes=min_bytes, quiet_s=quiet_s, hash_max=hash_max, now=now, log=log)
    total = sum(f["bytes"] for r in plan for f in r["files"])
    rep: dict[str, Any] = {"runs_root": str(runs_root), "dry_run": dry_run, "target_free_bytes": target_free_bytes, "free_before": free(),
                           "evictable_bytes": total, "evictable_runs": len(plan), "runs": [], "evicted_bytes": 0, "reached_target": False}
    say(f"free {rep['free_before'] / 1e9:.2f} GB; {len(plan)} run dirs, {total / 1e9:.2f} GB evictable")
    simulated = free()
    for r in plan:
        if (simulated if dry_run else free()) >= target_free_bytes:
            break
        rd = Path(r["dir"])
        n = sum(f["bytes"] for f in r["files"])
        if not dry_run:
            for f in r["files"]:
                (rd / f["path"]).unlink(missing_ok=True)
            doc = {"schema": EVICTED_SCHEMA, "run_id": r["run"], "artifact": r["artifact"], "evicted_utc": utc_now(), "files": r["files"],
                   "restore": r["restore"]}
            (rd / EVICTED_FILE).write_text(json.dumps(doc, indent=1) + "\n")
        simulated += n
        rep["evicted_bytes"] += n
        rep["runs"].append({"run": r["run"], "artifact": r["artifact"], "files": len(r["files"]), "bytes": n})
        say(f"{'would evict' if dry_run else 'evicted'} {r['run']}: {len(r['files'])} files, {n / 1e6:.0f} MB -> {r['artifact'][:20]}")
    rep["free_after"] = simulated if dry_run else free()
    rep["reached_target"] = rep["free_after"] >= target_free_bytes
    say(f"{'would free' if dry_run else 'freed'} {rep['evicted_bytes'] / 1e9:.2f} GB; free {'would be' if dry_run else 'now'} {rep['free_after'] / 1e9:.2f} GB")
    return rep


def evict_custody(run_dir: Path, custody: dict[str, Any], *, min_bytes: int = MIN_BYTES) -> dict[str, Any]:
    """The run dir's files that `custody.verify` has just found in the run record on the remote (`custody`, its report with
    `preserved`), at least `min_bytes` each: `research data custody --publish` removes them, so a published run stops holding the
    disk (the control pod's filled six times on 2026-09-27/28).  A file is removed only while it is still the size it had and
    unchanged since `verified_from`; the small records (job.json, status.json, result.json, the logs under `min_bytes`) stay, so
    `status` and `inspect` still read the run.  `run-files.evicted.json` lists what went and the fetch that brings it back."""
    rd = Path(run_dir)
    rep: dict[str, Any] = {"run": rd.name, "artifact": custody.get("run_record"), "files": [], "evicted_bytes": 0}
    if not custody.get("preserved") or not custody.get("run_record"):
        return rep
    t0 = float(custody["verified_from"])
    for rel, (size, sha) in sorted((custody.get("files") or {}).items(), key=lambda kv: (-kv[1][0], kv[0])):
        if size < min_bytes:
            continue
        p = rd / rel
        try:
            st = p.lstat()
        except OSError:
            continue
        if st.st_size != size or st.st_mtime >= t0 or not p.is_file():
            continue
        p.unlink()
        rep["files"].append({"path": rel, "bytes": size, "sha256": sha})
        rep["evicted_bytes"] += size
    if rep["files"]:
        prev = _read_evicted(rd)
        files = {f["path"]: f for f in (prev.get("files") or [])} | {f["path"]: f for f in rep["files"]}
        doc = {"schema": EVICTED_SCHEMA, "run_id": custody.get("run_id") or rd.name, "artifact": rep["artifact"], "evicted_utc": utc_now(),
               "files": sorted(files.values(), key=lambda f: f["path"]), "restore": f"research data fetch {rep['artifact']} --to {rd}"}
        (rd / EVICTED_FILE).write_text(json.dumps(doc, indent=1) + "\n")
    return rep


def _read_evicted(rd: Path) -> dict[str, Any]:
    try:
        doc = json.loads((rd / EVICTED_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def parse_gb(text: str) -> int:
    return int(float(text) * 1e9)

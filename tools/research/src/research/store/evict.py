"""Evict local copies whose bytes are PRESERVED on the remote (README §7): the remote is the durable home, `objects/` is a hot cache.

Two kinds of local copy can go, and only these:

* **store blobs** (`objects/sha256/<hex>`): a blob is evictable when EVERY artifact that holds it has a remote replica with a
  `verified_utc` in the catalog (PRESERVED), the local object has the manifest's size, and the blob is at least `min_bytes`.
  Largest first, until the target free space is reached.  `LocalStore._resolve_blob` re-fetches an evicted blob on demand;
  `has_local` / `where` report the eviction honestly, and `index.set_presence(local=...)` is refreshed for every touched artifact.
  A blob any artifact in `keep` holds stays (`--keep-registry`: the fixture registry, for a node that can't refetch from the remote).
  `trees/` (materialised tree copies) is a pure cache: its entries nobody fetched within `TREE_KEEP_S` go unless `keep_trees`
  (a fresher one may be open in a run's view; the node's hourly eviction runs below its mark all day).
* **run-directory files** (`~/.research/runs/<run>/...`, `evict_runs`): the big files of a finished run that appear, with the same
  size (and the same sha256 when the file is <= `hash_max` bytes), in a PRESERVED tree artifact of its attempt: `run_record` /
  `run_files`, or a typed output its `outputs.json` declares, laid out under that output's `tree` path (a Build's `outputs/build`).
  Oldest run first.  A run touched in the last `quiet_s` seconds, or whose status is not terminal, is left alone.
  `run-files.evicted.json` in the run dir lists what went and the `research data fetch`es that bring it back.
  A sealed run dir (store.sealed) is the root helper's to change: `sudo -n vy-seal-runs evict` removes the files its
  `.custody` lists at the size it recorded and unchanged since (`evict_custody`'s rule), as root.

* **source trees** (`<root>/src/<sha>`, `evict_src`): the trees `research run --on` ships per commit (remote.py `ship_source`;
  up to 24 GB and 480k inodes with Lean builds), whose bytes are a commit's.  Oldest first, every one that meets root's three
  conditions (Slack, 4 Oct 03:48Z; condition 1 read as recoverability, root 4 Oct 17:27Z), each established afresh: (1) it is
  recoverable on the node: its commit is reachable from a ref of the node's `<root>/git/verity.git`, which holds
  `refs/research/src/<sha>` for every tree shipped and from which `research run` re-ships it (`git ls-remote`, or of
  `--src-origin`, `--no-replace-objects`), and the tree is exactly that commit: its `.git` is a ship's (no config key, exclude,
  object, ref or stash of its own), its HEAD is the commit, no index entry is marked skip-worktree or assume-unchanged, and
  `git status`, run without system or global config, shows nothing but READY.json, build caches and fetched fixtures, ignored
  files included (`_local_changes`); (2) nothing uses it: no live process names it (cwd, root, exe, open file, map, command line,
  environment, any `src/<sha>` they spell, any sha a process inside `<src>` names, a thread's cwd when its leader exited), no git worktree made from it, no run that has not ended and no request without a
  run dir names it, no live check or Lean slot holder or waiter (`<root>/locks`) names it, no job script queued for the fill
  (`retention.JobScripts`) names it, and nothing in it changed and no run naming it was recorded within `SRC_KEEP_S`; (3) no run
  record holds it as its source's only copy: its READY.json names its own commit, and every run record naming it, but a refused
  one (refused.json, no status.json: its workload never ran), names that commit as its source, clean, or dirty with the dirty bytes in the run's preserved record (the runner's capture, env.DIRTY_CAPTURE, at the
  size and sha256 job.json names, listed by the run's `.custody`; `_dirty_preserved`).  A tree with a `research keep` record is left
  to `research retention`.  This pass is the only thing that deletes a whole source tree (`node_sweep.sh` no longer does).  Not on a free-space target: an
  unused tree costs inodes as much as bytes (node 1 at 74% inodes, 4 Oct), and `research run` re-ships one from
  `refs/research/src/<sha>`.  Before a deletion the tree is checked again, renamed to `.evicting-<sha>-<nonce>` (never executable
  by name), checked once more (renamed back when something uses it, or on any error) and removed READY.json first, one pass at a
  time (`<src>/.evict.lock`).  It fails closed: a condition
  it cannot establish (the repository unlisted, a process it cannot read, a record it cannot read) deletes nothing and says why.
  The one process it passes over unread is an nginx worker whose configuration cannot name a tree (`_nginx_worker`): RunPod's
  proxy, which runs as `nobody` on every pod, where the pass has no CAP_SYS_PTRACE.

Never touched: pending (unpushed or unverified) objects, `manifests/`, `attempts/`, `labels/`, `catalog.sqlite`, `scratch/`.
Artifacts that are flagged `remote` in the catalog but never verified are counted in `held_back_bytes` so the operator can
`research data verify` / `push` them first.  `free_fn` is the free-space probe (an `os.statvfs` over the store root by default);
tests inject their own.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from . import sealed
from .local import LocalStore, sha256_file
from .model import utc_now

MIN_BYTES = 1 << 20          # blobs / files below this are not worth an eviction (and a re-fetch)
#: a `trees/` entry fetched within this long may still be read (a view a run links into: the vLLM regression's rebuilds read
#: theirs for most of an hour), so eviction leaves it; `LocalStore.fetch` refreshes an entry's mtime on every fetch
TREE_KEEP_S = 24 * 3600
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


def plan_store(store: LocalStore, *, min_bytes: int = MIN_BYTES, keep: frozenset[str] = frozenset(),
               log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """What could go: `candidates` = [(bytes, hex, holders)] largest first, `trees_bytes`, `held_back_bytes` (local blobs of
    artifacts flagged remote but not verified), `kept_bytes` (local blobs an artifact in `keep` holds), `skipped` (size mismatch /
    unreadable manifest)."""
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
    held_back = kept = 0
    for h, arts in holders.items():
        p = store.object_path(h)
        if not p.exists() or sizes[h] < min_bytes:
            continue
        if arts & keep:
            kept += sizes[h]
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
            "kept_bytes": kept, "skipped": skipped, "verified_artifacts": len(verified)}


def stale_trees(store: LocalStore, now: float) -> tuple[list[str], int, int]:
    """The `trees/` entries nobody fetched within TREE_KEEP_S of ``now``, their bytes, and how many entries are fresher."""
    trees = store.root / "trees"
    stale, n, kept = [], 0, 0
    for entry in os.listdir(trees) if trees.is_dir() else []:
        try:
            fresh = os.stat(trees / entry).st_mtime >= now - TREE_KEEP_S
        except FileNotFoundError:
            continue
        if fresh:
            kept += 1
            continue
        stale.append(entry)
        n += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(trees / entry) for f in fs)
    return stale, n, kept


def evict_store(store: LocalStore, *, target_free_bytes: int, dry_run: bool = False, min_bytes: int = MIN_BYTES, keep_trees: bool = False,
                keep: frozenset[str] = frozenset(), free_fn: Callable[[], int] | None = None, log: Callable[[str], None] | None = None,
                now: float | None = None) -> dict[str, Any]:
    """Evict verified-remote blobs (largest first) and the `trees/` entries not fetched within TREE_KEEP_S until `free_fn()` >=
    `target_free_bytes`.  The report says what was (or, with `dry_run`, would be) removed; nothing else is touched."""
    say = log or (lambda s: None)
    free = free_fn or (lambda: free_bytes(store.root))
    plan = plan_store(store, min_bytes=min_bytes, keep=keep, log=log)
    rep: dict[str, Any] = {"store": str(store.root), "dry_run": dry_run, "target_free_bytes": target_free_bytes, "free_before": free(),
                           "evictable_bytes": plan["evictable_bytes"], "evictable_blobs": len(plan["candidates"]), "held_back_bytes": plan["held_back_bytes"],
                           "kept_bytes": plan["kept_bytes"],
                           "trees_bytes": plan["trees_bytes"], "trees_removed": False, "evicted_bytes": 0, "evicted_blobs": [], "artifacts_touched": [],
                           "skipped": plan["skipped"], "reached_target": False}
    say(f"free {rep['free_before'] / 1e9:.2f} GB, target {target_free_bytes / 1e9:.2f} GB; {len(plan['candidates'])} evictable blobs "
        f"({plan['evictable_bytes'] / 1e9:.2f} GB), trees/ {plan['trees_bytes'] / 1e9:.2f} GB, held back (remote but unverified) {plan['held_back_bytes'] / 1e9:.2f} GB")
    trees = store.root / "trees"
    stale, stale_bytes, rep["trees_kept"] = stale_trees(store, time.time() if now is None else now)
    if stale and not keep_trees and free() < target_free_bytes:
        if not dry_run:
            for entry in stale:
                shutil.rmtree(trees / entry, ignore_errors=True)
        rep["trees_removed"] = True
        say(f"trees/: {len(stale)} entries not fetched within {TREE_KEEP_S // 3600} h, {stale_bytes / 1e9:.2f} GB (materialised "
            f"cache), {'would be ' if dry_run else ''}removed; {rep['trees_kept']} kept")
    simulated = free() + (stale_bytes if dry_run and rep["trees_removed"] else 0)
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
        if not dry_run and sealed.sealed(rd):
            try:
                got = sealed.evict(rd, min_bytes)
            except (OSError, ValueError) as e:
                say(f"{r['run']}: sealed, and vy-seal-runs refused to evict it: {e}")
                continue
            n = got["evicted_bytes"]
            r = {**r, "files": got["files"]}
        elif not dry_run:
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


# ---- source trees --------------------------------------------------------------------------------------------------------------------

SRC_KEEP_S = 3600            # a tree changed, or named by a run recorded, this recently may be being shipped or about to run;
#   a live run's tree is kept by its process anyway, and at 2 h cpu-5 kept 100 trees and reached 98% inodes (26 Oct, INF-248)
SRC_BUDGET_S = 30 * 60       # no deletion starts after this long, so the unit's `timeout 3000` never cuts a pass off mid-tree
GIT_TIMEOUT_S = 300
ASIDE = ".evicting-"         # <src>/.evicting-<sha>-<nonce>: a tree on its way out, never executable by name
READY_FILE = "READY.json"    # remote.READY_FILE: the marker that makes <src>/<sha> executable
LOCK = ".evict.lock"         # <src>/.evict.lock: one pass at a time
ENDED = TERMINAL_STATES + ("refused",)
PF_KTHREAD = 0x00200000      # /proc/<pid>/stat flags: a kernel thread, which opens no file of ours
NGINX_CONF = Path("/etc/nginx/nginx.conf")  # a bare `/usr/sbin/nginx` master's configuration (Debian/Ubuntu build)
_SHA = re.compile(r"[0-9a-f]{40}")


class Unestablished(Exception):
    """A condition of `evict_src` that could not be established; the message says which and why.  Nothing more is deleted."""


def _redact(url: str) -> str:
    return re.sub(r"//[^/@]*@", "//***@", url)


def _git(args: list[str], stdin: str | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--no-replace-objects", *args], input=stdin, capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
                          env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", **(env or {})})


def on_origin(shas: set[str], origin: str, git_dir: Path) -> set[str]:
    """The commits of `shas` reachable from a ref of `origin` (`git ls-remote`: any ref): a tip itself, or in the history of a tip
    that `git_dir` holds (a commit's id covers its parents, so an ancestor there is reachable from the tip)."""
    def run(args: list[str], what: str, stdin: str | None = None) -> str:
        try:
            r = _git(args, stdin)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Unestablished(_redact(f"1: {what}: {type(e).__name__}: {e}")) from None
        if r.returncode:
            raise Unestablished(f"1: {what} failed (rc {r.returncode}): {_redact((r.stderr or r.stdout).strip()[-300:])}")
        return r.stdout

    tips = {ln.split()[0] for ln in run(["ls-remote", origin], f"git ls-remote {_redact(origin)}").splitlines() if ln.strip()}
    if not tips:
        raise Unestablished(f"1: {_redact(origin)} lists no refs")
    found = shas & tips
    if shas - found:
        if bad := [str(f) for f in (git_dir / "shallow", git_dir / "info" / "grafts") if f.exists()]:
            raise Unestablished(f"1: {', '.join(bad)} would rewrite the history the ancestor walk reads")
        g = ["--git-dir", str(git_dir)]
        have = run([*g, "cat-file", "--batch-check"], f"git cat-file in {git_dir}", "\n".join(sorted(tips)) + "\n")
        local = [f[0] for f in (ln.split() for ln in have.splitlines()) if len(f) > 1 and f[1] in ("commit", "tag")]
        if local:
            found |= (shas - found) & set(run([*g, "rev-list", "--stdin"], f"git rev-list in {git_dir}", "\n".join(local) + "\n").split())
    return found


_LOOSE = re.compile(rf"(?:^|[/\0\s:=])src/+(?:\./)*(?:{re.escape(ASIDE)})?([0-9a-f]{{40}})(?![0-9a-f])")
"""A tree named however a process spells it (`src/<sha>` relative to its cwd, `//`, `/./`): any `src/<sha>` keeps that sha."""


_BARE = re.compile(rf"(?:^|[/\0\s:=])(?:{re.escape(ASIDE)})?([0-9a-f]{{40}})(?![0-9a-f])")
"""Any sha a process inside <src> names (`../<sha>` from a tree, `<sha>/x.py` from <src>): it may be a sibling tree."""


def _roots(src_root: Path) -> str:
    return "|".join(map(re.escape, sorted({os.path.abspath(src_root), os.path.realpath(src_root)}, key=len, reverse=True)))


def _tree_pattern(src_root: Path) -> re.Pattern[str]:
    return re.compile(f"(?:{_roots(src_root)})/(?:{re.escape(ASIDE)})?([0-9a-f]{{40}})(?![0-9a-f])")


def _inside(src_root: Path) -> re.Pattern[str]:
    return re.compile(f"(?:{_roots(src_root)})(?:/|$)")


def _quiet(fn: Callable[[], Any]) -> Any:
    """`fn()`, or None when what it reads is gone (a process that exited, a file removed meanwhile)."""
    try:
        return fn()
    except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
        return None


def _proc_texts(base: Path) -> list[str]:
    """Every path and string process `base` names: cwd, root, exe, open files, maps, command line, environment.  Empty for a kernel
    thread or a process gone meanwhile; PermissionError when it can't be read."""
    task = base.parent.name == "task"
    stat = _quiet((base / "stat").read_text)
    if not task and (stat is None or int(stat.rsplit(")", 1)[1].split()[6]) & PF_KTHREAD):
        return []
    out = [_quiet(lambda n=n: os.readlink(base / n)) for n in ("cwd", "root", "exe")]
    out += [_quiet(lambda f=f: os.readlink(base / "fd" / f)) for f in _quiet(lambda: os.listdir(base / "fd")) or []]
    out += [_quiet(lambda n=n: (base / n).read_bytes().decode(errors="replace")) for n in ("maps", "cmdline", "environ")]
    if out[0] is None and not task:     # a leader that exited while its threads run: they show under task/
        for tid in _quiet(lambda: os.listdir(base / "task")) or []:
            if tid != base.name:
                out += _proc_texts(base / "task" / tid)
    return [t for t in out if t]


def _field(status: str, key: str) -> list[str]:
    return next((ln.split()[1:] for ln in status.splitlines() if ln.startswith(f"{key}:")), [])


def _nginx_worker(proc: Path, pid: str, src_root: Path, conf: Path) -> bool:
    """Whether unreadable process `pid` is an nginx worker that cannot name a tree: RunPod's proxy (a worker as `nobody` on every
    pod, which a container without CAP_SYS_PTRACE can't read).  A worker is forked from its master without exec and opens only what
    the master's configuration names, so it qualifies when it runs as a uid other than root, its parent is readable and is a bare
    `/usr/sbin/nginx` master (no `-c`, `-p` or `-g`: the configuration is `conf`), and `conf` includes no other file, loads no
    module and never names the top directory of `src_root`."""
    status = _quiet((proc / pid / "status").read_text)
    if not status or _field(status, "Name") != ["nginx"] or "0" in _field(status, "Uid") or not _field(status, "PPid"):
        return False
    parent = proc / _field(status, "PPid")[0]
    try:
        exe = os.readlink(parent / "exe")
        title = (parent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        text = conf.read_text(errors="replace")
    except OSError:
        return False
    tops = {Path(p).parts[1] for p in (os.path.abspath(src_root), os.path.realpath(src_root)) if len(Path(p).parts) > 1}
    return (exe == "/usr/sbin/nginx" and title == "nginx: master process /usr/sbin/nginx"
            and not re.search(r"\b(include|load_module)\b", text) and not any(t in text for t in tops))


def proc_uses(pattern: re.Pattern[str], proc: Path = Path("/proc"), inside: re.Pattern[str] | None = None,
              src_root: Path | None = None, nginx_conf: Path = NGINX_CONF) -> dict[str, set[str]]:
    """{sha: reasons} for the trees a live process names; a process whose cwd `inside` matches (<src> or a tree in it) keeps every sha
    it names.  Unestablished when `proc` can't be listed or a live process outside the kernel can't be read: as anyone but root,
    without CAP_SYS_PTRACE, most can't.  Given `src_root`, an unreadable nginx worker that cannot name a tree under it
    (`_nginx_worker`) is passed over; its master is read like any process."""
    try:
        pids = sorted((p for p in os.listdir(proc) if p.isdigit()), key=int)
    except OSError as e:
        raise Unestablished(f"2: cannot list {proc}: {e}") from None
    if not pids:
        raise Unestablished(f"2: {proc} lists no process")
    uses: dict[str, set[str]] = {}
    for pid in pids:
        try:
            texts = _proc_texts(proc / pid)
            cwd = (_quiet(lambda pid=pid: os.readlink(proc / pid / "cwd")) or "") if texts and inside else ""
        except PermissionError as e:
            if src_root is not None and _nginx_worker(proc, pid, src_root, nginx_conf):
                continue
            raise Unestablished(f"2: {proc / pid} is unreadable ({e.strerror}) as uid {os.geteuid()}: the pass needs root or "
                                "CAP_SYS_PTRACE") from None
        joined = "\n".join(texts)
        bare = _BARE.finditer(joined) if cwd and inside and inside.match(cwd) else ()
        for m in [*pattern.finditer(joined), *_LOOSE.finditer(joined), *bare]:
            comm = (_quiet((proc / pid / "comm").read_text) or "?").strip()
            uses.setdefault(m.group(1), set()).add(f"2: process {pid} ({comm}) uses it")
    return uses


def _text(p: Path) -> str | None:
    try:
        return p.read_text(errors="replace")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        raise Unestablished(f"2: cannot read {p}: {e}") from None


def _entries(d: Path) -> list[str]:
    try:
        return sorted(os.listdir(d))
    except FileNotFoundError:
        return []
    except OSError as e:
        raise Unestablished(f"2: cannot list {d}: {e}") from None


def _doc(text: str | None) -> dict[str, Any] | None:
    """The JSON object `text` holds; {} when it holds something else, None when there is no text."""
    if text is None:
        return None
    try:
        d = json.loads(text)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def _run_state(rd: Path) -> str:
    status = _doc(_text(rd / "status.json"))
    if status is None:
        return "refused" if (rd / "refused.json").exists() else "without a status (not known to have ended)"
    tr = status.get("transitions") or []
    last = tr[-1] if tr and isinstance(tr[-1], dict) else {}
    return str(status.get("state") or last.get("state") or "unknown")


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _dirty_preserved(rd: Path, job: dict[str, Any], src: dict[str, Any]) -> bool:
    """Whether the run's record on the remote holds its dirty source: job.json names a capture (`store.dirty_capture`, cli.py) of the
    digest its conditions record, and the run's valid `.custody` (store.custody) lists that file at the capture's size and sha256."""
    cap = _get(job, "store", "dirty_capture")
    digest = src.get("dirty_digest")
    if not isinstance(cap, dict) or not digest or digest == "too-large" or cap.get("dirty_digest") != digest:
        return False
    marker = _doc(_text(rd / ".custody"))
    files = marker.get("files") if isinstance(marker, dict) and marker.get("preserved") is True else None
    return isinstance(files, dict) and files.get(cap.get("path")) == [cap.get("bytes"), cap.get("sha256")]


def record_uses(pattern: re.Pattern[str], runs_dirs: list[Path], requests_dir: Path, *, now: float,
                keep_s: float = SRC_KEEP_S) -> dict[str, set[str]]:
    """{sha: reasons} from the run records that name a tree (job.json, launch.json, remote.json) and the requests: a run that has
    not ended, or was recorded within `keep_s`, or whose source is not the tree's own clean commit (condition 3; a refused run ran
    nothing, so it is no run's source and keeps a tree only while recent), and a request
    with no run dir yet or whose run has not ended."""
    uses: dict[str, set[str]] = {}

    def add(shas: set[str], why: str) -> None:
        for s in shas:
            uses.setdefault(s, set()).add(why)

    for root in runs_dirs:
        for name in _entries(root):
            rd = root / name
            texts = {f: _text(rd / f) for f in ("job.json", "launch.json", "remote.json")}
            named = {m.group(1) for t in texts.values() if t for m in pattern.finditer(t)}
            if not named:
                continue
            if (state := _run_state(rd)) not in ENDED:
                add(named, f"2: run {name} is {state}")
            mtime = _quiet(lambda: os.stat(rd / "job.json").st_mtime)
            if mtime is not None and now - mtime < keep_s:
                add(named, f"recent: run {name} was recorded within {keep_s / 3600:g} h")
            if state == "refused":
                continue
            job = _doc(texts["job.json"])
            if job == {}:
                add(named, f"3: run {name}'s job.json does not parse")
                continue
            sha = _get(job, "run", "remote", "source_sha") or (_doc(texts["remote.json"]) or {}).get("source_sha")
            src = _get(job, "store", "conditions", "source") or {}
            if not isinstance(src, dict):
                add(named, f"3: run {name}'s store.conditions.source is not an object")
                continue
            for t in named:
                other = next((x for x in (sha, src.get("commit")) if x not in (None, t)), None)
                if other:
                    add({t}, f"3: run {name} names it, and its source is {other}")
                elif src.get("dirty") and not _dirty_preserved(rd, job, src):
                    add({t}, f"3: run {name}'s source is dirty, and no preserved record of the run holds its capture")
                elif sha is None and src.get("commit") is None:
                    add({t}, f"3: run {name} names it and records no source commit")
    for name in _entries(requests_dir):
        named = {m.group(1) for m in pattern.finditer(_text(requests_dir / name / "request.json") or "")}
        if not named:
            continue
        rd = next((r / name for r in runs_dirs if (r / name).is_dir()), None)
        if rd is None:
            add(named, f"2: request {name} has no run dir yet (queued)")
        elif (state := _run_state(rd)) not in ENDED:
            add(named, f"2: request {name}'s run is {state}")
    return uses


def slot_uses(pattern: re.Pattern[str], locks: Path, proc: Path = Path("/proc")) -> dict[str, set[str]]:
    """{sha: reasons} from the check and Lean slots (`tools/verity/check/slot.py`, `lean_slot.py`): a holder's `<locks>/*.who` and a waiter's
    ticket in `<locks>/*line*/`, each `{"pid", "run", "tree", "by"}`, while its pid lives (a pid reused since only keeps more)."""
    files = [locks / n for n in _entries(locks) if n.endswith(".who")]
    files += [locks / d / n for d in _entries(locks) if "line" in d and (locks / d).is_dir() for n in _entries(locks / d)]
    uses: dict[str, set[str]] = {}
    for f in files:
        for ln in (_text(f) or "").splitlines():
            named = {m.group(1) for m in pattern.finditer(ln)}
            if not named:
                continue
            pid = (_doc(ln) or {}).get("pid")
            if isinstance(pid, int) and not (proc / str(pid)).exists():
                continue
            what = "slot holder" if f.parent == locks else "slot waiter"
            for s in named:
                uses.setdefault(s, set()).add(f"2: {what} {f.relative_to(locks)}" + (f" (pid {pid})" if pid else ""))
    return uses


def _newest(tree: Path) -> float:
    """The newest mtime of the tree and its top-level entries (a ship writes them; a `research keep` record doesn't count)."""
    t = os.lstat(tree).st_mtime
    for e in os.listdir(tree):
        if e != ".retention.json":
            t = max(t, _quiet(lambda e=e: os.lstat(tree / e).st_mtime) or 0.0)
    return t


def _own_reasons(src_root: Path, sha: str, now: float, keep_s: float) -> tuple[set[str], float]:
    """What the tree itself says, and its `_newest` mtime: its READY.json must name its commit (condition 3), it must not have
    changed within `keep_s`, and a `research keep` record leaves it to `research retention`."""
    from .. import retention as RT  # noqa: PLC0415

    tree, why = src_root / sha, set()
    try:
        newest = _newest(tree)
    except OSError as e:
        raise Unestablished(f"2: cannot read {tree}: {e}") from None
    ready = _doc(_text(tree / READY_FILE))
    if ready is None:
        why.add(f"3: no {READY_FILE} (an unverified ship): its files may not be its commit's")
    elif ready.get("identity") != sha:
        why.add(f"3: {READY_FILE} names {ready.get('identity')!r}")
    try:
        if RT.load(tree) is not None:
            why.add("record: a `research keep` record covers it; `research retention` decides")
    except RT.RecordError as e:
        why.add(f"record: {e}")
    if (age := now - newest) < keep_s:
        why.add(f"recent: changed {age / 60:.0f} min ago (within {keep_s / 3600:g} h)")
    return why, newest


SHIP_CONFIG = re.compile(r"core\.(?:repositoryformatversion|filemode|bare|logallrefupdates)|remote\.origin\.(?:url|fetch)"
                         r"|branch\.[^\n]+\.(?:remote|merge)")
"""The keys a ship's clone has in .git/config (every tree on vy-nebius-1, 4 Oct).  Any other key may change what `git status`
reports (excludes, fsmonitor, filters, sparse checkout, an include) or run a command, so the tree is kept rather than trusted."""
SHIP_EXCLUDE = frozenset({f"/{READY_FILE}"})
REGENERABLE = re.compile(r"(?:^|/)(?:\.lake|\.venv|__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.hypothesis|node_modules"
                         r"|target|[^/]*\.egg-info)(?:/|$)|\.py[co]$")
"""Ignored paths a build or test run regenerates.  Every other ignored path but READY.json and a fetched fixture (re-fetchable by
its id) may be a workload's only output (`out/`, `results/`, `*.png`...) and keeps the tree."""
_CLEAN_GIT = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
_CLEAN_C = ["-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-c", "core.excludesFile=/dev/null",
            "-c", "core.hooksPath=/dev/null"]


def _fixture_paths(tree: Path) -> list[str]:
    """The paths `fixtures/artifacts.json` registers (fetched fixtures are among them); [] when it is absent or unreadable."""
    doc = _doc(_quiet((tree / "fixtures" / "artifacts.json").read_text))
    return [p.strip("/") for k, p in (doc or {}).items() if k.startswith("art:") and isinstance(p, str) and p.strip("/")]


def _ignored_ok(tree: Path, path: str, fixtures: list[str]) -> bool:
    """Whether ignored `path` (a directory ends in /) holds only READY.json, regenerable files and fetched fixtures."""
    def ok(p: str) -> bool:
        return p == READY_FILE or bool(REGENERABLE.search(p)) or any(p == f or p.startswith(f + "/") for f in fixtures)
    p = path.rstrip("/")
    if ok(p):
        return True
    if not path.endswith("/") or not any(f.startswith(p + "/") for f in fixtures):
        return False
    for dp, dns, fns in os.walk(tree / p, onerror=_raise_oserror):
        rel = os.path.relpath(dp, tree)
        dns[:] = [d for d in dns if not ok(f"{rel}/{d}")]
        if not all(ok(f"{rel}/{n}") for n in fns) or any(os.path.islink(os.path.join(dp, d)) for d in dns):
            return False
    return True


def _local_changes(tree: Path, sha: str) -> set[str]:
    """Why the tree is not exactly its commit, so deleting it would lose work no ref of the node's repo holds: a `.git` that is not
    a ship's (any config key but SHIP_CONFIG's, an exclude, a sparse checkout, an object or ref of its own, a stash, a worktree made
    from it), a HEAD other than `sha`, an index entry `git status` is told to skip, or any change `git status` shows, tracked,
    untracked, or ignored but READY.json, regenerable caches and fetched fixtures.  Git runs without system or global config and
    with every command-running key off.  Run only for a tree about to go: it reads every file's metadata."""
    gd = tree / ".git"
    if not os.path.lexists(gd):
        return {"1: not recoverable on the node: no .git to compare it with its commit"}
    if gd.is_symlink() or not gd.is_dir():
        return {"1: not recoverable on the node: .git is not a directory, so another repository may hold its state"}
    why: set[str] = set()
    if _quiet(lambda: os.listdir(gd / "worktrees")):
        why.add("2: a git worktree made from it (.git/worktrees) may be in use")

    def git(*args: str) -> str | None:
        try:
            r = _git([*_CLEAN_C, "--git-dir", str(gd), "--work-tree", str(tree), *args], env=_CLEAN_GIT)
        except (OSError, subprocess.TimeoutExpired) as e:
            why.add(f"1: git {args[0]} failed: {type(e).__name__}: {e}")
            return None
        if r.returncode:
            why.add(f"1: git {args[0]} failed (rc {r.returncode}): {(r.stderr or r.stdout).strip()[-200:]}")
            return None
        return r.stdout

    keys = git("config", "--file", str(gd / "config"), "--list", "--name-only", "-z")
    if keys is not None and (odd := sorted({k for k in keys.split("\0") if k and not SHIP_CONFIG.fullmatch(k)})):
        why.add(f"1: its .git/config sets {', '.join(odd[:3])}, which a ship's doesn't: git status may not report its changes")
    exclude = _quiet((gd / "info" / "exclude").read_text) or ""
    if odd := [ln for ln in exclude.splitlines() if ln.strip() and not ln.startswith("#") and ln.strip() not in SHIP_EXCLUDE]:
        why.add(f"1: its .git/info/exclude hides {odd[0][:80]!r}")
    if os.path.lexists(gd / "info" / "sparse-checkout"):
        why.add("1: a sparse checkout (.git/info/sparse-checkout)")
    if os.path.lexists(gd / "logs" / "refs" / "stash"):
        why.add("1: not recoverable on the node: a stash in its .git")
    counts = dict(ln.split(": ", 1) for ln in (git("count-objects", "-v") or "").splitlines() if ": " in ln)
    if any(counts.get(k, "0") != "0" for k in ("count", "in-pack", "garbage")):
        why.add(f"1: not recoverable on the node: objects in its own .git (count-objects: {counts}), a commit or stash made in it")
    refs = (git("for-each-ref", "--format=%(refname)") or "").split()
    if own := [r for r in refs if not r.startswith("refs/remotes/origin/")]:
        why.add(f"1: not recoverable on the node: ref(s) of its own: {', '.join(own[:3])}")
    if why:
        return why
    flags = git("ls-files", "-v", "-z")
    if hidden := [e[2:] for e in (flags or "").split("\0") if e and (e[0].islower() or e[0] == "S")]:
        why.add(f"1: {len(hidden)} file(s) git status is told to skip (assume-unchanged or skip-worktree), e.g. {hidden[0][-120:]!r}")
    out = git("status", "--porcelain=v2", "--branch", "-z", "--untracked-files=normal", "--ignored=matching",
              "--ignore-submodules=none")
    if out is None or flags is None:
        return why
    entries = [e for e in out.split("\0") if e]
    head = next((e.split()[2] for e in entries if e.startswith("# branch.oid ")), None)
    if head != sha:
        why.add(f"1: not recoverable on the node: its HEAD is {head}")
    changes = [e for e in entries if not e.startswith(("#", "! ")) and e != f"? {READY_FILE}"]
    if changes:
        why.add(f"1: not recoverable on the node: {len(changes)} local change(s) no commit holds, e.g. {changes[0][-120:]!r}")
    if why:
        return why
    fixtures = _fixture_paths(tree)
    try:
        kept = [e[2:] for e in entries if e.startswith("! ") and not _ignored_ok(tree, e[2:], fixtures)]
    except OSError as e:
        return {f"1: cannot read its ignored files: {type(e).__name__}: {e}"}
    if kept:
        why.add(f"1: not recoverable on the node: {len(kept)} ignored path(s) no build regenerates, e.g. {kept[0][-120:]!r}")
    return why


def _remove_tree(path: Path) -> tuple[int, int]:
    """`rm -r path`, READY.json first; (inodes, bytes) it frees, so a hard link into the uv cache counts once, where it goes."""
    inodes = nbytes = 0

    def unlink(p: str) -> None:
        nonlocal inodes, nbytes
        st = os.lstat(p)
        try:
            os.unlink(p)
        except PermissionError:
            d = os.path.dirname(p)
            os.chmod(d, os.lstat(d).st_mode | 0o700)
            os.unlink(p)
        if st.st_nlink == 1:
            inodes, nbytes = inodes + 1, nbytes + st.st_size

    if os.path.lexists(path / READY_FILE):
        unlink(str(path / READY_FILE))
    for dp, dns, fns in os.walk(path, topdown=False, onerror=_raise_oserror):
        for n in fns:
            unlink(os.path.join(dp, n))
        for n in dns:
            p = os.path.join(dp, n)
            if os.path.islink(p):
                unlink(p)
            else:
                os.rmdir(p)
                inodes += 1
    os.rmdir(path)
    return inodes + 1, nbytes


def _raise_oserror(e: OSError) -> None:
    raise e


def evict_src(src_root: Path, *, origin: str | None = None, dry_run: bool = False, git_dir: Path | None = None,
              runs_dirs: list[Path] | None = None, requests_dir: Path | None = None, locks_dir: Path | None = None,
              jobs_dirs: list[Path] | None = None, proc: Path = Path("/proc"), keep_s: float = SRC_KEEP_S,
              budget_s: float = SRC_BUDGET_S, now: float | None = None, clock: Callable[[], float] = time.monotonic,
              log: Callable[[str], None] | None = None, nginx_conf: Path = NGINX_CONF) -> dict[str, Any]:
    """Delete, oldest first, the source trees under `src_root` (`<root>/src`) that meet every condition (module docstring), and the
    `.evicting-*` leftovers of an interrupted pass that nothing uses.  The report lists every tree deleted (inodes, bytes) and every
    tree kept with its reasons, each prefixed by what it is: `1:`/`2:`/`3:` (root's conditions), `recent:`, `record:`, `budget:`;
    `error` says what could not be established, and then nothing more was deleted.  `<root>` is `src_root`'s parent: its
    `git/verity.git` (also `origin`, condition 1's repository), `runs/` and `runs-aside/`, `requests/` and `locks/` unless given."""
    from .. import retention as RT  # noqa: PLC0415

    say = log or (lambda s: None)
    src_root = Path(src_root)
    root = src_root.parent
    git_dir = git_dir or root / "git" / "verity.git"
    origin = origin or str(git_dir)
    runs_dirs = [root / "runs", root / "runs-aside"] if runs_dirs is None else [Path(d) for d in runs_dirs]
    requests_dir = requests_dir or root / "requests"
    locks_dir = locks_dir or root / "locks"
    jobs = RT.JobScripts([Path(d) for d in (RT.DEFAULT_JOBS_DIRS if jobs_dirs is None else jobs_dirs) if Path(d).is_dir()], root=root)
    t_wall, t0 = time.time() if now is None else now, clock()
    pattern, inside = _tree_pattern(src_root), _inside(src_root)
    rep: dict[str, Any] = {"src_root": str(src_root), "dry_run": dry_run, "origin": _redact(origin), "trees": 0, "deleted": [],
                           "kept": [], "deleted_inodes": 0, "deleted_bytes": 0, "kept_by": {}, "error": None}
    if not dry_run and src_root.is_dir():
        try:
            lock = open(src_root / LOCK, "a")  # held for the pass: another pass would take this one's aside tree for a leftover
        except OSError as e:
            rep["error"] = f"cannot open {src_root / LOCK}: {type(e).__name__}: {e} (made by another user? remove it)"
            say(f"src: nothing done: {rep['error']}")
            return rep
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            rep["error"] = f"another pass holds {src_root / LOCK}"
            say(f"src: nothing done: {rep['error']}")
            return rep

    def live(shas: set[str]) -> dict[str, set[str]]:
        """Condition 2 and the run records' part of 3, read afresh."""
        why: dict[str, set[str]] = {}
        for found in (proc_uses(pattern, proc, inside, src_root, nginx_conf), record_uses(pattern, runs_dirs, requests_dir, now=t_wall + clock() - t0, keep_s=keep_s),
                      slot_uses(pattern, locks_dir, proc)):
            for s, rs in found.items():
                if s in shas:
                    why.setdefault(s, set()).update(rs)
        for s in shas:
            if r := jobs(src_root / s):
                why.setdefault(s, set()).add(f"2: {r}")
        return why

    try:
        names = _entries(src_root)
    except Unestablished as e:
        rep["error"] = str(e)
        return rep
    trees = [n for n in names if _SHA.fullmatch(n) and (src_root / n).is_dir() and not (src_root / n).is_symlink()]
    leftovers = [n for n in names if n.startswith(ASIDE) and _SHA.fullmatch(n[len(ASIDE):len(ASIDE) + 40])]
    rep["trees"] = len(trees)
    reasons: dict[str, set[str]] = {s: set() for s in trees}
    mtimes: dict[str, float] = {}
    deleted: set[str] = set()
    try:
        if trees or leftovers:
            uses = live(set(trees) | {n[len(ASIDE):len(ASIDE) + 40] for n in leftovers})
            on = on_origin(set(trees), origin, git_dir) if trees else set()
            for s in trees:
                own, mtimes[s] = _own_reasons(src_root, s, t_wall, keep_s)
                reasons[s] |= uses.get(s, set()) | own
                if s not in on:
                    reasons[s].add(f"1: not recoverable on the node: on no ref of {_redact(origin)}")
            for n in leftovers:
                if why := uses.get(n[len(ASIDE):len(ASIDE) + 40]):
                    rep["kept"].append({"tree": n, "why": sorted(why)})
                    continue
                try:
                    inodes, nbytes = (None, None) if dry_run else _remove_tree(src_root / n)
                except OSError as e:
                    rep["kept"].append({"tree": n, "why": [f"error: {type(e).__name__}: {e}"]})
                    continue
                rep["deleted"].append({"tree": n, "leftover": True, "inodes": inodes, "bytes": nbytes})
        for mtime, s in sorted((mtimes[s], s) for s in trees if not reasons[s]):
            if clock() - t0 > budget_s:
                reasons[s].add(f"budget: no deletion starts after {budget_s / 60:g} min; the next pass goes on")
                continue
            if changes := _local_changes(src_root / s, s):
                reasons[s] |= changes
                continue
            if dry_run:
                rep["deleted"].append({"tree": s, "age_h": round((t_wall - mtime) / 3600, 1), "inodes": None, "bytes": None})
                deleted.add(s)
                continue
            own, mtime = _own_reasons(src_root, s, t_wall + clock() - t0, keep_s)
            if again := live({s}).get(s, set()) | own:
                reasons[s] |= again
                continue
            aside = src_root / f"{ASIDE}{s}-{secrets.token_hex(4)}"
            try:
                os.rename(src_root / s, aside)
            except OSError as e:
                reasons[s].add(f"error: {type(e).__name__}: {e}")
                continue
            try:
                after = live({s}).get(s, set()) | _local_changes(aside, s)
            except BaseException:
                _put_back(aside, src_root / s)
                raise
            if after:
                reasons[s] |= after | ({"2: moved back"} if _put_back(aside, src_root / s) else
                                       {f"2: left as {aside.name}: {s} was shipped again meanwhile"})
                continue
            try:
                inodes, nbytes = _remove_tree(aside)
            except OSError as e:
                reasons[s].add(f"error: {type(e).__name__}: {e} (left as {aside.name}, without {READY_FILE}; the next pass goes on)")
                continue
            deleted.add(s)
            rep["deleted"].append({"tree": s, "age_h": round((t_wall - mtime) / 3600, 1), "inodes": inodes, "bytes": nbytes})
            say(f"src: deleted {s[:12]} ({nbytes / 1e9:.2f} GB, {inodes} inodes, {(t_wall - mtime) / 3600:.1f} h old)")
    except Unestablished as e:
        rep["error"] = str(e)
        for s in trees:
            if s not in deleted:
                reasons[s].add(f"unestablished: {e}")
    rep["kept"] += [{"tree": s, "why": sorted(reasons[s])} for s in trees if s not in deleted]
    rep["deleted_inodes"] = sum(d["inodes"] or 0 for d in rep["deleted"])
    rep["deleted_bytes"] = sum(d["bytes"] or 0 for d in rep["deleted"])
    for k in rep["kept"]:
        for c in sorted({w.split(":", 1)[0] for w in k["why"]}):
            rep["kept_by"][c] = rep["kept_by"].get(c, 0) + 1
    say(f"src: {'would delete' if dry_run else 'deleted'} {len(rep['deleted'])} of {len(trees)} trees ({rep['deleted_bytes'] / 1e9:.1f} GB, "
        f"{rep['deleted_inodes']} inodes); kept {len(rep['kept'])} " + " ".join(f"{c}:{n}" for c, n in sorted(rep["kept_by"].items()))
        + (f"; nothing more deleted: {rep['error']}" if rep["error"] else ""))
    return rep


def _put_back(aside: Path, home: Path) -> bool:
    """Rename a tree back into place; False when it was shipped again meanwhile (or the rename failed) and it stays aside."""
    if os.path.lexists(home):
        return False
    try:
        os.rename(aside, home)
    except OSError:
        return False
    return True

#!/usr/bin/python3 -I
"""vy-seal-runs: seal each ended run dir under a node's runs roots, so agents can read it but not change it (root, stdlib only).

    vy-seal-runs seal [--dry-run] [--max N] [--quiet-min M] [--root DIR]...   (vy-seal-runs.timer, every 10 min)
    vy-seal-runs delete RUN_DIR                                                (research retention gc / rm, through sudo)
    vy-seal-runs evict RUN_DIR --min-bytes N                                   (research data evict --runs, through sudo)

A run dir is ENDED when its status.json state is terminal (store.attempt.TERMINAL) and custody vouches for it
(store.custody.has_custody): a `preserved.json` (a verified copy on the launcher), or a `.custody` marker that says preserved and
lists every file now in the dir, none changed at or after its `verified_from` (store.custody.marker_problem, the rule the pod
guard applies too). Time alone never ends a run; on top of that rule a dir any of whose files changed in the last --quiet-min
minutes waits for a later pass. tests/test_nebius_lockout.py holds this module to custody's own functions.

Sealing chowns every entry to root and removes its write bits, keeping it readable (files a+r, directories a+rx), deepest first
and the run dir itself last, so a dir whose top is root's and unwritable is sealed whole and a killed pass is redone. Every
open is relative to an O_NOFOLLOW directory fd (os.fwalk): no symlink is followed, a symlink is chowned itself, and a file
swapped for a symlink between the listing and the open is skipped. Each sealed dir is a line on stdout (the journal) and in
LEDGER with its former owner and mode, so vy-admin can undo one by hand. At most --max dirs per pass.

`delete` and `evict` are the only writes into a sealed run, for the two tools that free disk: each acts only on a sealed run
dir directly under a runs root whose custody still vouches for it, so all they remove is bytes the remote holds. `delete`
removes the dir (research.retention calls it after its own record checks); `evict` removes the files of at least --min-bytes
that `.custody` lists at the size it recorded and unchanged since, and writes run-files.evicted.json (store.evict.evict_custody's
rule). Exit 0 done, 1 an error, 3 refused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOTS = ("/workspace/research/runs", "/workspace/jobs/runs")
TERMINAL = ("done", "failed", "cancelled")
MARKER = ".custody"
SCHEMA = "research/run-custody/v0.1"
EVICTED_FILE = "run-files.evicted.json"
EVICTED_SCHEMA = "research/run-files-evicted/v0.1"
TRIMMED_LISTING = ".custody-trimmed.json"
EXCLUDED = frozenset({".fetched", ".fetched.tmp", MARKER, MARKER + ".tmp", "preserved.json", EVICTED_FILE, TRIMMED_LISTING})
LEDGER = "/var/lib/vy-seal-runs/sealed.jsonl"
MAX = 50
QUIET_MIN = 10
REFUSED = 3


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def run_files(run_dir: Path) -> list[str]:
    """Every regular file under the run dir, markers excluded, no symlink followed (store.custody.run_files)."""
    out = []
    for base, dirs, files in os.walk(run_dir):
        dirs.sort()
        for f in files:
            p = Path(base) / f
            rel = p.relative_to(run_dir).as_posix()
            if rel in EXCLUDED or rel.startswith(MARKER + ".tmp.") or p.is_symlink() or not p.is_file():
                continue
            out.append(rel)
    return sorted(out)


def marker_problem(run_dir: Path) -> str | None:
    """None when `.custody` vouches for the run dir as it is now; else why not (store.custody.marker_problem)."""
    try:
        m = json.loads((run_dir / MARKER).read_text())
    except FileNotFoundError:
        return "no .custody"
    except (OSError, ValueError):
        return "unreadable .custody"
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        return f".custody is not {SCHEMA}"
    if not m.get("preserved"):
        return f".custody: not preserved ({m.get('error') or '?'})"
    try:
        t0, listed = float(m["verified_from"]), set(m["files"])
    except (KeyError, TypeError, ValueError):
        return "unreadable .custody"
    for rel in run_files(run_dir):
        if rel not in listed:
            return f"{rel} is not in the run record (added after custody)"
        try:
            if os.lstat(run_dir / rel).st_mtime >= t0:
                return f"{rel} changed after custody"
        except OSError:
            continue
    return None


def not_ended(run_dir: Path) -> str | None:
    """None when the run has ended (module docstring); else why it hasn't, as far as this pass can tell."""
    state = _read(run_dir / "status.json").get("state")
    if state not in TERMINAL:
        return f"status {state or 'absent'}"
    if (run_dir / "preserved.json").exists():
        return None
    return marker_problem(run_dir)


def newest_mtime(run_dir: Path) -> float:
    newest = os.lstat(run_dir).st_mtime
    for base, dirs, files in os.walk(run_dir):
        for n in dirs + files:
            try:
                newest = max(newest, os.lstat(os.path.join(base, n)).st_mtime)
            except OSError:
                continue
    return newest


def is_sealed(st: os.stat_result, uid: int = 0) -> bool:
    return stat.S_ISDIR(st.st_mode) and st.st_uid == uid and not st.st_mode & 0o222


def _sealed_mode(mode: int) -> int:
    keep = stat.S_IMODE(mode) & ~0o222 & ~(stat.S_ISUID | stat.S_ISGID)
    return keep | (0o555 if stat.S_ISDIR(mode) else 0o444)


def seal_dir(parent_fd: int, name: str, uid: int = 0, gid: int = 0) -> int:
    """Seal the dir `name` of the directory `parent_fd`; the number of entries sealed. No symlink is followed (module docstring)."""
    os.close(os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd))   # ELOOP for a symlink
    n = 0
    # fwalk opens `name` itself and descends it only when it is the directory its lstat saw
    for _root, dirs, files, rootfd in os.fwalk(name, dir_fd=parent_fd, topdown=False, follow_symlinks=False):
        for entry in dirs + files:
            st = os.stat(entry, dir_fd=rootfd, follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                continue                                  # sealed through its own fd, when fwalk yields it
            if not stat.S_ISREG(st.st_mode):
                os.chown(entry, uid, gid, dir_fd=rootfd, follow_symlinks=False)
                n += 1
                continue
            try:
                fd = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY, dir_fd=rootfd)
            except OSError:
                continue                                  # swapped for a symlink, or gone
            try:
                fst = os.fstat(fd)
                if (fst.st_dev, fst.st_ino) == (st.st_dev, st.st_ino) and stat.S_ISREG(fst.st_mode):
                    os.fchown(fd, uid, gid)
                    os.fchmod(fd, _sealed_mode(fst.st_mode))
                    n += 1
            finally:
                os.close(fd)
        dst = os.fstat(rootfd)
        os.fchown(rootfd, uid, gid)
        os.fchmod(rootfd, _sealed_mode(dst.st_mode))
        n += 1
    return n


def candidates(root: Path, uid: int = 0) -> list[str]:
    """The names of the run dirs under `root` not sealed yet (no symlink, oldest name first: run ids sort by time)."""
    out = []
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return out
    for e in entries:
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISDIR(st.st_mode) and not is_sealed(st, uid):
            out.append(e.name)
    return out


def seal_pass(roots: list[str], *, dry_run: bool, max_dirs: int = MAX, quiet_s: float = QUIET_MIN * 60, now: float | None = None,
              uid: int = 0, gid: int = 0, ledger: str | None = LEDGER, say=print) -> dict:
    """One pass: seal up to `max_dirs` ended run dirs under `roots`; {"sealed": [...], "skipped": N, "errors": [...]}."""
    t = time.time() if now is None else now
    rep: dict = {"sealed": [], "skipped": 0, "errors": []}
    for root in roots:
        try:
            rst = os.lstat(root)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(rst.st_mode) or not stat.S_ISDIR(rst.st_mode):
            rep["errors"].append(f"{root}: not a directory (a symlink is never followed)")
            continue
        if rst.st_uid != uid or (rst.st_mode & 0o022 and not rst.st_mode & stat.S_ISVTX):
            say(f"vy-seal-runs: {root} is not root's and sticky, so a sealed run there can still be renamed (vy-lockout roots)")
        for name in candidates(Path(root), uid):
            if len(rep["sealed"]) >= max_dirs:
                say(f"vy-seal-runs: cap of {max_dirs} reached; the rest wait for the next pass")
                return rep
            d = Path(root) / name
            try:
                why = not_ended(d)
                if why is None and t - newest_mtime(d) < quiet_s:
                    why = f"changed in the last {quiet_s / 60:g} min"
            except OSError as e:
                why = f"unreadable: {e}"
            if why:
                rep["skipped"] += 1
                continue
            before = os.lstat(d)
            if dry_run:
                say(f"would seal {d} (owner {before.st_uid}:{before.st_gid}, mode {stat.S_IMODE(before.st_mode):o})")
                rep["sealed"].append(str(d))
                continue
            pfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                n = seal_dir(pfd, name, uid, gid)
            except OSError as e:
                rep["errors"].append(f"{d}: {e}")
                say(f"vy-seal-runs: {d}: {e}")
                continue
            finally:
                os.close(pfd)
            after = not_ended(d)
            say(f"sealed {d} ({n} entries; was {before.st_uid}:{before.st_gid} mode {stat.S_IMODE(before.st_mode):o})"
                + (f"; custody no longer vouches for it after sealing: {after}" if after else ""))
            rep["sealed"].append(str(d))
            if ledger:
                os.makedirs(os.path.dirname(ledger), mode=0o700, exist_ok=True)
                with open(ledger, "a") as fh:
                    fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), "path": str(d), "uid": before.st_uid,
                                         "gid": before.st_gid, "mode": oct(stat.S_IMODE(before.st_mode)), "entries": n,
                                         "custody_after": after}) + "\n")
    return rep


def sealed_run(path: str, roots: list[str], uid: int = 0) -> tuple[Path | None, str | None]:
    """(the run dir, None) when `path` is a sealed, ended run dir directly under one of `roots`; else (None, why not)."""
    if not os.path.isabs(path) or os.path.normpath(path) != path.rstrip("/") or "/../" in f"{path}/":
        return None, f"{path}: give the run dir's absolute, normalized path"
    p = Path(path.rstrip("/"))
    if str(p.parent) not in roots:
        return None, f"{p}: not a run dir directly under {' or '.join(roots)}"
    try:
        if os.path.realpath(p.parent) != str(p.parent):
            return None, f"{p.parent} resolves elsewhere"
        st = os.lstat(p)
    except OSError as e:
        return None, f"{p}: {e}"
    if not is_sealed(st, uid):
        return None, f"{p}: not sealed (its owner can change it, so it needs no root)"
    if why := not_ended(p):
        return None, f"{p}: custody doesn't vouch for it ({why}), so the remote may lack what it holds"
    return p, None


def delete(path: str, roots: list[str], uid: int = 0, say=print, remove=shutil.rmtree) -> int:
    p, why = sealed_run(path, roots, uid)
    if p is None:
        say(f"vy-seal-runs delete: refused: {why}")
        return REFUSED
    n = sum(len(fs) for _b, _d, fs in os.walk(p))
    remove(p)
    say(f"vy-seal-runs delete: deleted {p} ({n} files; sealed, custody vouched for it), for uid {os.environ.get('SUDO_UID', '?')}")
    return 0


def evict(path: str, roots: list[str], min_bytes: int, uid: int = 0, gid: int = 0, say=print) -> int:
    """store.evict.evict_custody's rule on a sealed run: stdout is its report as JSON, as `research data evict` reads it."""
    p, why = sealed_run(path, roots, uid)
    if p is None:
        say(f"vy-seal-runs evict: refused: {why}")
        return REFUSED
    m = _read(p / MARKER)
    rep: dict = {"run": p.name, "artifact": m.get("run_record"), "files": [], "evicted_bytes": 0}
    if (p / "preserved.json").exists() and not m.get("preserved"):
        print(json.dumps(rep))
        return 0
    t0 = float(m["verified_from"])
    for rel, (size, sha) in sorted((m.get("files") or {}).items(), key=lambda kv: (-kv[1][0], kv[0])):
        if size < min_bytes:
            continue
        f = p / rel
        try:
            st = f.lstat()
        except OSError:
            continue
        if st.st_size != size or st.st_mtime >= t0 or not stat.S_ISREG(st.st_mode):
            continue
        f.unlink()
        rep["files"].append({"path": rel, "bytes": size, "sha256": sha})
        rep["evicted_bytes"] += size
    if rep["files"]:
        prev = _read(p / EVICTED_FILE)
        files = {f["path"]: f for f in (prev.get("files") or [])} | {f["path"]: f for f in rep["files"]}
        doc = {"schema": EVICTED_SCHEMA, "run_id": m.get("run_id") or p.name, "artifact": rep["artifact"],
               "evicted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": sorted(files.values(), key=lambda f: f["path"]),
               "restore": f"research data fetch {rep['artifact']} --to {p}"}
        tmp = p / f".{EVICTED_FILE}.tmp"
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.chown(tmp, uid, gid, follow_symlinks=False)
        os.chmod(tmp, 0o444)
        os.replace(tmp, p / EVICTED_FILE)
        sys.stderr.write(f"vy-seal-runs evict: {p}: {len(rep['files'])} files, {rep['evicted_bytes']} bytes, for uid "
                         f"{os.environ.get('SUDO_UID', '?')}\n")
    print(json.dumps(rep))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vy-seal-runs", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal", help="seal ended run dirs (the timer's pass)")
    s.add_argument("--dry-run", action="store_true", help="say what would be sealed; change nothing")
    s.add_argument("--max", type=int, default=MAX, help=f"at most this many dirs per pass (default {MAX})")
    s.add_argument("--quiet-min", type=float, default=QUIET_MIN, help=f"skip a dir changed in the last M minutes (default {QUIET_MIN})")
    s.add_argument("--root", action="append", help=f"a runs root (default {', '.join(ROOTS)})")
    d = sub.add_parser("delete", help="delete a sealed, ended run dir (retention gc / rm through sudo)")
    d.add_argument("run_dir")
    e = sub.add_parser("evict", help="evict a sealed run's big preserved files (research data evict through sudo)")
    e.add_argument("run_dir")
    e.add_argument("--min-bytes", type=int, required=True)
    ns = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("vy-seal-runs: run as root", file=sys.stderr)
        return REFUSED
    if ns.cmd == "seal":
        rep = seal_pass(ns.root or list(ROOTS), dry_run=ns.dry_run, max_dirs=ns.max, quiet_s=ns.quiet_min * 60)
        print(f"vy-seal-runs: {'would seal' if ns.dry_run else 'sealed'} {len(rep['sealed'])}, skipped {rep['skipped']} not ended, "
              f"{len(rep['errors'])} errors")
        return 1 if rep["errors"] else 0
    if ns.cmd == "delete":
        return delete(ns.run_dir, list(ROOTS))
    return evict(ns.run_dir, list(ROOTS), ns.min_bytes, say=lambda s: print(s, file=sys.stderr))


if __name__ == "__main__":
    sys.exit(main())

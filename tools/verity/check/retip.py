#!/usr/bin/env python3
"""`tools/verity/check/retip.py BRANCH RUN... [--remote origin] [--store DIR] [--dry-run]`: carry the Lean records a failed check of
a train's tip regenerated onto that tip, as one commit.

Two pull requests each right on its side of a train can leave a `lean-audit.json` whose records no longer match the merged build
(one changes a definition that a guarantee the other adds reads).  The Lean check then fails with only record differences, and
writes the record `check.py --update` would write beside its report (`tools/verity/lean/check.py`: the report's `regenerated`
names `audit-TAG/lean-audit.json` under the run's `lean-audit/`).  This takes those files from the runs' records in the store and
commits each over its package's `lean-audit.json` on the train's prep branch, so the pull requests keep their heads and ready
marks and the next check of the tip passes on the regenerated records (the lander's friction pass, 10 Oct).  The landing still
DMs Daniel what changed in a guarantee (`research merge`'s `spec_alert`).

RUN is the failed check of BRANCH's tip on the remote, or every shard of one.  Each must be a `check` attempt of exactly that
clean commit, ended; every step that failed must be `lean-audit` (one it stopped proved nothing, and runs again), and every package that failed there must have a regenerated
record.  A both-sides change to one record entry never gets here: the train's merge driver (`tools/verity/lean/merge.py`) refuses
it when the train is built, and it goes back to its owner.  Anything else refuses, naming why, and commits nothing (exit 1).

The commit (subject naming the runs, a `Regenerated:` trailer naming the files, which `research merge`'s `reverted` reads as
rewrites the train means) is pushed to BRANCH as a fast-forward, never forced; `--dry-run` prints what it would write and pushes
nothing.  Then check the new tip again (`tools/verity/check/train.sh`).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = "lean-audit.json"
#: what `check.py`'s regeneration rewrites in a record (`regenerated`); every other key must be the tip's own
REWRITTEN = ("guarantees", "reads", "assumptions")


class Refused(Exception):
    pass


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def records(store, run: str, tip: str, scratch: Path) -> dict[str, bytes]:
    """{package: the regenerated record} of one failed check ``run`` of ``tip``; Refused when it is anything else."""
    doc = store.get_attempt(run)
    if not isinstance(doc, dict):
        raise Refused(f"{run}: no attempt in the store")
    st, src = doc.get("status") or {}, doc.get("source") or {}
    if (doc.get("tool") or {}).get("name") != "check":
        raise Refused(f"{run} is not a check")
    if src.get("commit") != tip or src.get("dirty") is not False:
        raise Refused(f"{run} checked {str(src.get('commit'))[:12]}{' (dirty)' if src.get('dirty') else ''}, not the tip {tip[:12]}")
    if st.get("state") not in ("done", "failed") or st.get("rc") in (None, 0):
        raise Refused(f"{run} did not fail (state {st.get('state')}, rc {st.get('rc')}): there is nothing to regenerate")
    out = doc.get("outputs") or {}
    steps = store.get_manifest(out["result"]).meta.get("steps", []) if out.get("result") else []
    if not steps:
        raise Refused(f"{run}: its result names no steps")
    if not any(s.get("step") == "lean-audit" and s.get("status") == "failed" for s in steps):
        raise Refused(f"{run}: its lean-audit step did not fail")
    # a step the failure stopped (`stopped`) proved nothing either way, and runs again on the next check
    if other := sorted({s["step"] for s in steps if s.get("status") == "failed" and s.get("step") != "lean-audit"}):
        raise Refused(f"{run} failed more than its Lean records: {', '.join(other)}")
    art = out.get("run_files") or out.get("run_record")
    if not art:
        raise Refused(f"{run}: no run files in the store")
    have = {f["path"] for f in store.listing(art)}
    if "lean-audit/audit.json" not in have:
        raise Refused(f"{run}: no lean-audit/audit.json among its files")
    d = scratch / run
    audit = json.loads((store.fetch(art, d, paths=["lean-audit/audit.json"]) / "lean-audit" / "audit.json").read_text())
    got, why = {}, []
    reports = {r["package"]: r for r in audit.get("packages", [])}
    for name in audit.get("failed", []):
        rep = reports.get(name)
        if rep is None or not rep.get("regenerated"):
            first = (rep or {}).get("failures", ["no report"])[0] if rep else "no report"
            why.append(f"{name} failed for more than its records ({str(first)[:160]})")
            continue
        path = f"lean-audit/{Path(rep['regenerated']).parent.name}/{RECORD}"
        if path not in have:
            why.append(f"{name}: {path} is not among {run}'s files")
            continue
        got[name] = (store.fetch(art, d, paths=[path]) / path).read_bytes()
    if why:
        raise Refused(f"{run}: " + "; ".join(why))
    if not got:
        raise Refused(f"{run}: no package's records were regenerated")
    return got


def plan(repo: Path, tip: str, regen: dict[str, bytes]) -> dict[str, bytes]:
    """{path: bytes} to write at ``tip``: each package's regenerated record where it differs from the tip's.  Refused when one
    changes a key regeneration doesn't rewrite (then it isn't this tip's record)."""
    out = {}
    for pkg, new in sorted(regen.items()):
        path = f"{pkg}/{RECORD}"
        old = _git(repo, "show", f"{tip}:{path}").encode()
        a, b = json.loads(old), json.loads(new)
        if moved := sorted(k for k in {*a, *b} - set(REWRITTEN) if a.get(k) != b.get(k)):
            raise Refused(f"{path}: the regenerated record also changes {', '.join(moved)}, so it is not this tip's")
        if new != old:
            out[path] = new
    return out


def commit(repo: Path, tip: str, files: dict[str, bytes], runs: list[str]) -> str:
    """A commit on ``tip`` writing ``files``, built in a scratch worktree; its id."""
    with tempfile.TemporaryDirectory(prefix="retip-") as tmp:
        wt = Path(tmp) / "tree"
        _git(repo, "worktree", "add", "-q", "--detach", str(wt), tip)
        try:
            for path, data in files.items():
                (wt / path).write_bytes(data)
            _git(wt, "add", "--", *files)
            msg = (f"Regenerate the Lean records at the train's tip from {', '.join(runs)}\n\n"
                   f"Each is the record `check.py --update` writes on this tree; the check found only record differences.\n\n"
                   f"Regenerated: {', '.join(files)}\n")
            _git(wt, "commit", "-q", "-m", msg)
            return _git(wt, "rev-parse", "HEAD").strip()
        finally:
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], capture_output=True)


def main(argv: list[str] | None = None, store=None, log=print) -> int:
    ap = argparse.ArgumentParser(prog="retip", description=__doc__.split("\n\n")[0])
    ap.add_argument("branch", help="the train's prep branch on the remote")
    ap.add_argument("runs", nargs="+", help="the failed check of its tip (or each of its shards)")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--store", default=None, help="evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        _git(a.repo, "fetch", "-q", a.remote, f"refs/heads/{a.branch}")
        tip = _git(a.repo, "rev-parse", "FETCH_HEAD").strip()
        if store is None:
            from research.store.local import LocalStore
            store = LocalStore(a.store)
        regen: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory(prefix="retip-runs-") as tmp:
            for run in a.runs:
                try:
                    got = records(store, run, tip, Path(tmp))
                except Refused:
                    raise
                except Exception as e:  # noqa: BLE001  (what can't be read refuses)
                    raise Refused(f"{run}: its records can't be read from the store ({type(e).__name__}: {e})") from e
                for pkg, data in got.items():
                    if regen.get(pkg, data) != data:
                        raise Refused(f"{pkg}: the runs regenerated different records")
                    regen[pkg] = data
        files = plan(a.repo, tip, regen)
        if not files:
            raise Refused(f"the regenerated records are the tip's own ({', '.join(regen)}): rerun the check")
        if a.dry_run:
            log(f"retip: would commit {', '.join(files)} on {a.branch} at {tip[:12]}")
            return 0
        new = commit(a.repo, tip, files, a.runs)
        _git(a.repo, "push", "-q", a.remote, f"{new}:refs/heads/{a.branch}")
    except Refused as e:
        log(f"retip: refused: {e}")
        return 1
    log(f"retip: {a.branch} = {new[:12]} ({', '.join(files)} regenerated from {', '.join(a.runs)}); "
        f"check it with `tools/verity/check/train.sh {new[:12]} MACHINE`")
    return 0


if __name__ == "__main__":
    sys.exit(main())

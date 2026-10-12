#!/usr/bin/env python3
"""`tools/verity/check/post_train.py [COMMIT] [--on MACHINE] [--cores LIST] [--workers N] [--force]`: the vLLM reference regression's
manifest rebuilds on a landing of main.

`check` blocks on the regression's cheap T0 tier; the checks that recompute from a row's Programs (marked `rebuild`: the manifest
rebuilds, `manifest_digest` on every row it applies to and the #57 refusal, and `step_segmentation`'s decode of every component;
about an hour of CPU in one process) run here instead, once per landing (Daniel, 3 Oct).  `research merge --push` starts
this, detached, after each push it lands (`after_land` in `tool.py`); run it by hand for a landing pushed another way, or to redo one.

COMMIT (default origin/main) must be a landing: its `Check: <attempt>` trailer names the check that let it in, or, for a train
`research queue drive` fast-forwarded main to, the store's passing check of that commit does.  Its tree is checked
out in a worktree of its own and the rebuilds run there through `research run --on MACHINE`, pinned to --cores, under
VERITY_REGRESSION_REBUILD=1 and VERITY_STORE_REQUIRED=1, at nice 19.  `research run --on` returns at launch, so this fetches the
run's records until they show its end (each POLL_S); then that check attempt is labelled `note=regression pass` or
`note=regression fail:<rows>` (`--by regression --ref <run>`), and circuits sweeps those labels daily.  A run cut off before its end
(past TIMEOUT_S on a busy machine, or cancelled) tested nothing: it is labelled `note=regression inconclusive:timeout` (or
`:cancelled`), which is no regression and no verdict to carry, so the next landing's job runs the rebuilds again.  A landing
whose attempt has no such label is one whose job didn't finish here (the caller's VM suspended, the machine refused the run): run
this again.  The
run is preserved from the machine either way; only the label needs this process to live.  An attempt that has its label already
is left alone unless --force.

Jobs started here run one at a time (a lock in the temp directory: the cores are one slot), so a landing waits for the one before
it.  When that one is labelled and the change between them touches none of the `verity-vllm` suite's inputs (`suites.py
--changed`), its verdict is carried: the same value, with the same run as `--ref`, and no rebuild.  About three trains in four
touch them (130 of 171 between 27 Sep and 3 Oct: 19 a day, 32 on the busiest), and a job takes one to two hours, so jobs queue:
one that waited, and finds main at a later landing, is labelled `note=regression superseded:<that landing's attempt>` and runs
nothing, since the later landing's job tests its change too.  A sweep reads `pass` and `fail:` and skips `superseded:` and
`inconclusive:`.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MACHINE = "vy-nebius-1"
#: node 1's guest cores; its check slots hold 8-95 and 128-159, and 160-191 is proofs' (infra, 4 Oct: 96-127 at nice 19)
CORES = "96-127"
BY = "regression"
PREFIX = "regression "
PASS = f"{PREFIX}pass"
FAIL = f"{PREFIX}fail:"
SUPERSEDED = f"{PREFIX}superseded:"
INCONCLUSIVE = f"{PREFIX}inconclusive:"
MARKS = "regression and rebuild"
SUITE = "verity-vllm"
LOCK = Path(tempfile.gettempdir()) / "verity-post-train.lock"
#: the run's bound on the machine: the rebuilds took 89 and 104 min on cores 96-127 beside other lanes' jobs (3 Oct; #39's is the
#: longest); a run that outlives it is cut off before its junit, and is inconclusive, not a failure
TIMEOUT_S = 4 * 3600
#: `research run --on` returns once the run is launched, so its records are fetched until they show its end (`research merge` too)
POLL_S = 60
#: a runner that died before writing its end is waited for no longer than the run's bound and the runner's publish margin, twice
POLLS = (TIMEOUT_S + 1800) // POLL_S
SLEEP = time.sleep
ENDED = ("done", "failed", "cancelled")
CHECK_RE = re.compile(r"^Check: (r20\d{6}-\d{6}-[0-9a-f]{4})$", re.M)
ROW_RE = re.compile(r"-(r\d+)\]$")

JOB = """set -u
out=$RESEARCH_RUN_DIR/out; mkdir -p "$out"
s=$(mktemp -d /workspace/research/scratch/post-train.XXXXXX 2>/dev/null || mktemp -d); trap 'rm -rf "$s"' EXIT
export TMPDIR=$s VERITY_REGRESSION_SCRATCH=$s BUILD_RAM_BUDGET_GB=32 VERITY_MANIFEST_JOBS=4
uv run -q --locked --extra torch-cpu python -m research data fetch-fixtures --repo "$PWD" --json \\
  integrations/vllm/tests/regression/expected > "$out/fetch-fixtures.json" || exit 1
cd integrations/vllm
VERITY_REGRESSION=1 VERITY_REGRESSION_REBUILD=1 VERITY_STORE_REQUIRED=1 VERITY_REGRESSION_RECORD_ACTUAL=$out/actual \\
  {pin}../../.venv/bin/python -m pytest tests/regression -m {marks} -n {workers} -rA --durations=0 \\
  --junitxml "$out/junit.xml" > "$out/pytest.log" 2>&1
rc=$?
tail -n 40 "$out/pytest.log"
exit $rc
"""


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def check_attempt(commit: str, cwd: Path = ROOT) -> str | None:
    """The attempt a landing of main was admitted on: its `Check:` line (`research merge`; a paragraph of its own, which git's
    `%(trailers)` doesn't read), else, for a commit on origin/main, the newest check that passed on its clean tree (`passed`): a
    train `research queue drive` fast-forwards main to was checked as it is, so it has no trailer.  None for any other commit."""
    m = CHECK_RE.search(git("log", "-1", "--format=%B", commit, cwd=cwd))
    if m:
        return m.group(1)
    sha = git("rev-parse", "--verify", f"{commit}^{{commit}}", cwd=cwd, check=False)
    if not sha or subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=cwd,
                                 capture_output=True).returncode:
        return None
    return passed(sha)


def passed(sha: str) -> str | None:
    """The newest `check` attempt that passed on ``sha``'s clean tree, as the merge gate reads the store; None when the store
    can't be read."""
    try:
        from research.store.local import LocalStore

        store = LocalStore()
        if store.remote is not None:
            store.refresh(attempts_only=True)
        rows = store.index.sql("SELECT id FROM attempts WHERE tool = 'check' AND source_commit = ? AND source_dirty = 0 "
                               "AND state = 'done' AND rc = 0 ORDER BY start_utc DESC LIMIT 1", (sha,))
    except Exception as e:  # noqa: BLE001  (no attempt found: the caller says the commit is no landing)
        print(f"post-train: the store can't be read for {sha[:12]}'s check ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    return rows[0][0] if rows else None


def _order(row: str) -> tuple[int, int, str]:
    return (0, int(row[1:]), "") if re.fullmatch(r"r\d+", row) else (1, 0, row)


def verdict(junit: Path, rc: int | None = None, cut: str | None = None) -> str:
    """`regression pass`, or `regression fail:<rows>`: the rows whose test failed (`r39`; `negative_57` for the refusal), or why
    there is nothing to read.  A skip (a check `fixtures.toml` retires on a row) is neither.  A run cut off before its end (``cut``:
    `timeout`, `cancelled`) is `regression inconclusive:<cut>`, whatever it wrote."""
    if cut:
        return INCONCLUSIVE + cut
    if not junit.is_file():
        return f"{FAIL}no junit (run rc={rc})"
    failed, passed = [], 0
    for case in ET.parse(junit).iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            passed += case.find("skipped") is None
            continue
        name = case.get("name", "")
        m = ROW_RE.search(name)
        failed.append(m.group(1) if m else name.removeprefix("test_").removesuffix("_refusal_reproduces"))
    if failed:
        return FAIL + ",".join(sorted(set(failed), key=_order))
    return PASS if passed else f"{FAIL}nothing ran (run rc={rc})"


def research(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "research", *args], cwd=cwd, capture_output=True, text=True)


def labelled(attempt: str) -> dict | None:
    """The attempt's latest `note` by `regression` (its `value` and `ref`), when it is a verdict of this job."""
    p = research("data", "labels", attempt, "--key", "note", "--by", BY, "--json")
    try:
        labs = json.loads(p.stdout or "[]")
    except json.JSONDecodeError:
        return None
    labs = [x for x in labs if str(x.get("value", "")).startswith(PREFIX)]
    return labs[-1] if labs else None


def untouched(tree: Path, rev: str) -> bool:
    """Whether the change from ``rev`` to ``tree`` touches none of the suite's inputs, as ``tree``'s own `suites.py` reads them."""
    p = subprocess.run([sys.executable, "tools/verity/check/suites.py", SUITE, "--changed", rev, "--list"], cwd=tree,
                       capture_output=True, text=True)
    return p.returncode == 0 and re.search(r"^suites: the change from \S+ touches 0: none$", p.stdout, re.M) is not None


def carried(sha: str, tree: Path) -> dict | None:
    """The verdict of the landing before ``sha`` (its first parent), with that landing's attempt as `from`, when the change since
    touches none of the suite's inputs: the rebuilds would read what they read there.  None, and the job runs, when the parent is
    no landing, has no verdict yet (none, `superseded` or `inconclusive`), or its verdict came from no junit (a run that failed
    before its tests)."""
    prev = check_attempt(f"{sha}^1", cwd=tree)
    lab = labelled(prev) if prev else None
    value = str(lab["value"]) if lab else ""
    if (not lab or not lab.get("ref") or not (value == PASS or value.startswith(FAIL)) or "(run rc=" in value
            or not untouched(tree, f"{sha}^1")):
        return None
    return {**lab, "from": prev}


def later_landing(sha: str, cwd: Path = ROOT) -> str | None:
    """The attempt of origin/main's tip when that tip is a landing after ``sha``, else None."""
    git("fetch", "-q", "origin", "main", cwd=cwd)
    tip = git("rev-parse", "origin/main", cwd=cwd)
    if tip == sha or subprocess.run(["git", "merge-base", "--is-ancestor", sha, tip], cwd=cwd, capture_output=True).returncode:
        return None
    return check_attempt(tip, cwd=cwd)


def _status(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def job(cores: str, workers: int) -> str:
    return JOB.format(pin="nice -n 19 " + (f"taskset -c {cores} " if cores else ""), marks=shlex.quote(MARKS), workers=workers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="post_train.py", description=__doc__.split("\n\n")[1])
    ap.add_argument("commit", nargs="?", default="origin/main", help="a landing of main (default origin/main)")
    ap.add_argument("--on", default=MACHINE, metavar="MACHINE", help=f"where the rebuilds run (default {MACHINE})")
    ap.add_argument("--cores", default=CORES, metavar="LIST", help=f"`taskset -c LIST` for them (default {CORES}; '' for none)")
    ap.add_argument("--workers", type=int, default=12, help="pytest-xdist workers: a row's rebuild each (default 12)")
    ap.add_argument("--project", default="verity")
    ap.add_argument("--force", action="store_true", help="run even when the attempt has its label, or could carry the one before's")
    ap.add_argument("--dry-run", action="store_true", help="run, and print the label instead of writing it (any commit: a branch's too)")
    ap.add_argument("--run", metavar="RID", help="a run this job launched for the commit before (its caller ended first): wait for "
                                                  "it and label from it instead of launching another")
    a = ap.parse_args(argv)
    git("fetch", "-q", "origin", "main")
    sha = git("rev-parse", "--verify", f"{a.commit}^{{commit}}")
    attempt = check_attempt(sha)
    if not attempt and not a.dry_run:
        print(f"post-train: {sha[:12]} has no `Check:` trailer and is no checked commit of origin/main, so it is no landing of main",
              file=sys.stderr)
        return 2
    with open(LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"post-train: {sha[:12]} waits for the job before it ({LOCK})", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if attempt and not (a.force or a.dry_run or a.run) and (later := later_landing(sha)):
                lab = research("data", "label", attempt, "note", f"{SUPERSEDED}{later}", "--by", BY, "--ref", later)
                print((lab.stdout + lab.stderr).strip()[-1000:])
                print(f"post-train: {attempt} ({sha[:12]}): note={SUPERSEDED}{later}: main landed {later} while this waited, and "
                      "its job tests this change too")
                return 0 if lab.returncode == 0 else 1
        return _land(a, sha, attempt)


def _land(a: argparse.Namespace, sha: str, attempt: str | None) -> int:
    fresh = not (a.force or a.dry_run)
    if attempt and fresh and (have := labelled(attempt)):
        print(f"post-train: {attempt} ({sha[:12]}) has note={have['value']} already; --force runs it again")
        return 0
    tree = Path(tempfile.mkdtemp(prefix=f"post-train-{sha[:12]}-"))
    git("worktree", "add", "-q", "--detach", str(tree), sha)
    try:
        if attempt and fresh and (c := carried(sha, tree)):
            lab = research("data", "label", attempt, "note", c["value"], "--by", BY, "--ref", c["ref"])
            print((lab.stdout + lab.stderr).strip()[-1000:])
            print(f"post-train: {attempt} ({sha[:12]}): note={c['value']} (ref {c['ref']}), carried from {c['from']}: the change from "
                  f"{sha[:12]}^1 touches none of {SUITE}'s inputs")
            return 0 if lab.returncode == 0 and c["value"] == PASS else 1
        from research.runs import new_run_id, runs_root
        rid = a.run or new_run_id()
        if not a.run:
            print(f"post-train: {sha[:12]} (check {attempt}) on {a.on}, cores {a.cores or 'any'}: run {rid}", flush=True)
            p = research("run", "--id", rid, "--on", a.on, "--project", a.project, "--stage", "post-train-regression", "--timeout",
                         str(TIMEOUT_S), "--source", str(tree), "--cwd", "source", "--declared-output", "out/*", "--",
                         "bash", "-c", job(a.cores, a.workers), cwd=tree)
            print((p.stdout + p.stderr)[-4000:], flush=True)
            if p.returncode:
                print(f"post-train: run {rid} was not launched; nothing labelled", file=sys.stderr)
                return 1
        status = runs_root() / rid / "status.json"
        st = _status(status)
        for _ in range(POLLS):
            if st.get("state") in ENDED:
                break
            SLEEP(POLL_S)
            research("fetch", rid, "--on", a.on, cwd=tree)
            st = _status(status)
        if st.get("state") not in ENDED:
            print(f"post-train: run {rid} showed no end in {POLLS} fetches; nothing labelled (`--run {rid}` waits for it again)",
                  file=sys.stderr)
            return 1
        f = research("fetch", rid, "--all", "--on", a.on, cwd=tree)
        print((f.stdout + f.stderr).strip()[-1000:], flush=True)
        cut = ("timeout" if st.get("timed_out") else "cancelled") if st.get("state") == "cancelled" else None
        v = verdict(runs_root() / rid / "out" / "junit.xml", st.get("rc"), cut)
        if a.dry_run:
            print(f"post-train: {sha[:12]}: --dry-run, so {attempt or 'no check attempt'} gets no note={v} (ref {rid})")
            return 0 if v == PASS else 1
        lab = research("data", "label", attempt, "note", v, "--by", BY, "--ref", rid)
        print((lab.stdout + lab.stderr).strip()[-1000:])
        print(f"post-train: {attempt} ({sha[:12]}): note={v} (ref {rid})")
        return 0 if lab.returncode == 0 and v == PASS else 1
    finally:
        git("worktree", "remove", "--force", str(tree), check=False)


if __name__ == "__main__":
    sys.exit(main())

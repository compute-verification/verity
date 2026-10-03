#!/usr/bin/env python3
"""`tools/check/post_train.py [COMMIT] [--on MACHINE] [--cores LIST] [--workers N] [--force]`: the vLLM reference regression's
manifest rebuilds on a landing of main.

`check` blocks on the regression's cheap T0 tier; the checks that recompute from a row's Programs (marked `rebuild`: the manifest
rebuilds, `manifest_digest` on every row it applies to and the #57 refusal, and `step_segmentation`'s decode of every component;
about an hour of CPU in one process) run here instead, once per landing (Daniel, 3 Oct).  `research merge --push` starts
this, detached, after each push it lands (`after_land` in `tool.py`); run it by hand for a landing pushed another way, or to redo one.

COMMIT (default origin/main) must be a landing: its `Check: <attempt>` trailer names the check that let it in.  Its tree is checked
out in a worktree of its own and the rebuilds run there through `research run --on MACHINE`, pinned to --cores, under
VERITY_REGRESSION_REBUILD=1 and VERITY_STORE_REQUIRED=1.  When the run ends, that check attempt is labelled `note=regression pass` or
`note=regression fail:<rows>` (`--by regression --ref <run>`), and circuits sweeps those labels daily.  A landing whose attempt has
no such label is one whose job didn't finish here (the caller's VM suspended, the machine refused the run): run this again.  The
run is preserved from the machine either way; only the label needs this process to live.  An attempt that has its label already
is left alone unless --force.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MACHINE = "vy-nebius-1"
#: node 1's guest cores; its check slots hold 8-95 and 128-159
CORES = "96-127"
BY = "regression"
PREFIX = "regression "
MARKS = "regression and rebuild"
RUN_RE = re.compile(r"\brun (r20\d{6}-\d{6}-[0-9a-f]{4}) launched\b")
CHECK_RE = re.compile(r"^Check: (r20\d{6}-\d{6}-[0-9a-f]{4})$", re.M)
ROW_RE = re.compile(r"-(r\d+)\]$")

JOB = """set -u
out=$RESEARCH_RUN_DIR/out; mkdir -p "$out"
s=$(mktemp -d /workspace/research/scratch/post-train.XXXXXX 2>/dev/null || mktemp -d); trap 'rm -rf "$s"' EXIT
export TMPDIR=$s VERITY_REGRESSION_SCRATCH=$s BUILD_RAM_BUDGET_GB=32 VERITY_MANIFEST_JOBS=4
uv run -q --locked --extra torch-cpu python -m research data fetch-fixtures --json integrations/vllm/tests/regression/expected \\
  > "$out/fetch-fixtures.json" || exit 1
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
    `%(trailers)` doesn't read), None for any other commit."""
    m = CHECK_RE.search(git("log", "-1", "--format=%B", commit, cwd=cwd))
    return m.group(1) if m else None


def _order(row: str) -> tuple[int, int, str]:
    return (0, int(row[1:]), "") if re.fullmatch(r"r\d+", row) else (1, 0, row)


def verdict(junit: Path, rc: int | None = None) -> str:
    """`regression pass`, or `regression fail:<rows>`: the rows whose test failed (`r39`; `negative_57` for the refusal), or why
    there is nothing to read.  A skip (a check `fixtures.toml` retires on a row) is neither."""
    if not junit.is_file():
        return f"{PREFIX}fail:no junit (run rc={rc})"
    failed, passed = [], 0
    for case in ET.parse(junit).iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            passed += case.find("skipped") is None
            continue
        name = case.get("name", "")
        m = ROW_RE.search(name)
        failed.append(m.group(1) if m else name.removeprefix("test_").removesuffix("_refusal_reproduces"))
    if failed:
        return f"{PREFIX}fail:" + ",".join(sorted(set(failed), key=_order))
    return f"{PREFIX}pass" if passed else f"{PREFIX}fail:nothing ran (run rc={rc})"


def research(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "research", *args], cwd=cwd, capture_output=True, text=True)


def labelled(attempt: str) -> str | None:
    """The attempt's latest `note` by `regression`, when it is a verdict of this job."""
    p = research("data", "labels", attempt, "--key", "note", "--by", BY, "--json")
    try:
        labs = json.loads(p.stdout or "[]")
    except json.JSONDecodeError:
        return None
    vals = [str(x.get("value")) for x in labs if str(x.get("value", "")).startswith(PREFIX)]
    return vals[-1] if vals else None


def job(cores: str, workers: int) -> str:
    return JOB.format(pin=f"taskset -c {cores} " if cores else "", marks=shlex.quote(MARKS), workers=workers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="post_train.py", description=__doc__.split("\n\n")[1])
    ap.add_argument("commit", nargs="?", default="origin/main", help="a landing of main (default origin/main)")
    ap.add_argument("--on", default=MACHINE, metavar="MACHINE", help=f"where the rebuilds run (default {MACHINE})")
    ap.add_argument("--cores", default=CORES, metavar="LIST", help=f"`taskset -c LIST` for them (default {CORES}; '' for none)")
    ap.add_argument("--workers", type=int, default=12, help="pytest-xdist workers: a row's rebuild each (default 12)")
    ap.add_argument("--project", default="verity")
    ap.add_argument("--force", action="store_true", help="run even when the attempt has its label")
    a = ap.parse_args(argv)
    git("fetch", "-q", "origin", "main")
    sha = git("rev-parse", "--verify", f"{a.commit}^{{commit}}")
    attempt = check_attempt(sha)
    if not attempt:
        print(f"post-train: {sha[:12]} has no `Check:` trailer, so it is no landing of main", file=sys.stderr)
        return 2
    if not a.force and (have := labelled(attempt)):
        print(f"post-train: {attempt} ({sha[:12]}) has note={have} already; --force runs it again")
        return 0
    tree = Path(tempfile.mkdtemp(prefix=f"post-train-{sha[:12]}-"))
    git("worktree", "add", "-q", "--detach", str(tree), sha)
    try:
        print(f"post-train: {sha[:12]} (check {attempt}) on {a.on}, cores {a.cores or 'any'}", flush=True)
        p = research("run", "--on", a.on, "--project", a.project, "--stage", "post-train-regression", "--source", str(tree),
                     "--cwd", "source", "--declared-output", "out/*", "--", "bash", "-c", job(a.cores, a.workers), cwd=tree)
        print((p.stdout + p.stderr)[-4000:], flush=True)
        m = RUN_RE.search(p.stdout + p.stderr)
        if not m:
            print("post-train: no run launched; nothing labelled", file=sys.stderr)
            return 1
        rid = m.group(1)
        f = research("fetch", rid, "--all", "--on", a.on, cwd=tree)
        print((f.stdout + f.stderr).strip()[-1000:], flush=True)
        from research.runs import runs_root
        v = verdict(runs_root() / rid / "out" / "junit.xml", p.returncode)
        lab = research("data", "label", attempt, "note", v, "--by", BY, "--ref", rid)
        print((lab.stdout + lab.stderr).strip()[-1000:])
        print(f"post-train: {attempt} ({sha[:12]}): note={v} (ref {rid})")
        return 0 if lab.returncode == 0 and v == f"{PREFIX}pass" else 1
    finally:
        git("worktree", "remove", "--force", str(tree), check=False)


if __name__ == "__main__":
    sys.exit(main())
